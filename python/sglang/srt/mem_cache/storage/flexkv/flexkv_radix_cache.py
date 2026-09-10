"""FlexKV-backed RadixCache for sglang.

This module exposes :class:`FlexKVRadixCache`, a subclass of
:class:`sglang.srt.mem_cache.radix_cache.RadixCache` that delegates
host-side prefix storage to a FlexKV ``KVManager``. The design mirrors
``LMCRadixCache`` (the LMCache integration) so the scheduler-side
contract is identical:

* MP (synchronous) mode — the default.
  ``match_prefix`` fires only a FlexKV LOOKUP and returns ``host_hit_length``;
  the scheduler then calls :meth:`init_load_back` at dispatch time which
  allocates slots and fires the FlexKV RETRIEVE. With ``--enable-flexkv``,
  the scheduler also runs enqueue-time :meth:`prefetch_from_storage` and
  waits via :meth:`check_prefetch_progress` so Remote/Mooncake blocks are
  on CPU before lookup/retrieve (compute GET no longer issues REMOTE2H).

* IP (layerwise) mode — enabled with ``FLEXKV_ENABLE_LAYERWISE_TRANSFER=1``.
  ``match_prefix`` allocates uncached slots and kicks off a layerwise
  load; the per-layer hook registered via
  ``register_layer_transfer_counter`` then waits on each layer's
  eventfd inside the model's forward pass.

Selection: ``--enable-flexkv`` on the sglang CLI routes the default
RadixCache factory here. See ``__init__.py`` in this package for the
``register_radix_cache_backend("flexkv", ...)`` entry-point that backs
the explicit ``--radix-cache-backend=flexkv`` form.
"""

from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    DecodeRestoreDriver,
    EvictParams,
    EvictResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
    RestoreCompletion,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from flexkv.integration.sglang.connector import (
    FlexKVConnector,
    FlexKVHostReleaseShim,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class FlexKVMode(enum.Enum):
    MP = enum.auto()  # synchronous lookup → retrieve in two phases
    IP = enum.auto()  # in-process layerwise transfer


def _assert_mode_supports_disaggregation(
    mode: FlexKVMode, server_args: ServerArgs
) -> None:
    """IP mode drains its per-layer eventfds from inside the model forward.

    On a PD decode server the restore has to finish *before* the request joins a
    batch — decode already promised prefill that prefix via ``decode_prefix_len``
    — so there is no forward left to drain them, and the waits would land on the
    wrong request's layers.
    """
    if mode is FlexKVMode.IP and server_args.disaggregation_mode == "decode":
        raise ValueError(
            "FLEXKV_ENABLE_LAYERWISE_TRANSFER=1 is not supported on a PD decode "
            "server: the layerwise restore has no forward pass to wait on. Unset "
            "it to use FlexKV MP mode."
        )


@dataclass
class _AsyncRestore:
    """One in-flight async RETRIEVE, holding everything needed to insert the
    node once the load lands.

    The node is deliberately *not* in the tree while the load is in flight:
    FlexKV writes these slots from another process over CUDA IPC, so there is
    no stream ordering that would make a concurrent reader safe — a node
    reachable by ``match_prefix`` could hand another request KV that has not
    been written yet. Insertion is deferred to completion, which also keeps
    slot ownership identical to the synchronous path (the tree owns the slots,
    and the request's ``cache_protected_len`` covers them).
    """

    req: Req
    key: RadixKey
    value_numel: int
    slots: torch.Tensor
    last_node: TreeNode

    @property
    def node_key(self) -> RadixKey:
        """The key the loaded node occupies under ``last_node``."""
        return self.key[self.value_numel : self.value_numel + int(self.slots.numel())]


@dataclass
class _LoadBackMarker:
    """State carried from a hit-producing ``match_prefix`` to its
    matching ``init_load_back``. The detached ``RadixKey`` is a snapshot
    of the matched key at lookup time (the live request key aliases
    ``req.fill_ids`` which keeps growing)."""

    key: RadixKey
    value_numel: int  # device tokens already present at lookup time


class FlexKVRadixCache(RadixCache):
    """RadixCache extended with FlexKV host-tier IO."""

    def __init__(
        self,
        params: CacheInitParams,
        model_config: Optional[ModelConfig],
        server_args: ServerArgs,
        tp_rank: int,
        tp_size: int,
        dp_rank: Optional[int],
        pp_rank: int,
        attn_cp_rank: int,
        tp_group=None,
        pp_group=None,
        attn_tp_group=None,
        attn_cp_group=None,
    ) -> None:
        super().__init__(params)

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        # ``tp_group`` and ``attn_tp_group`` are sometimes passed
        # interchangeably by sglang's factory; prefer the explicit
        # ``attn_tp_group`` when given.
        attn_tp_group_eff = attn_tp_group if attn_tp_group is not None else tp_group

        self.flexkv_connector = FlexKVConnector(
            sgl_model_config=model_config,
            server_args=server_args,
            page_size=params.page_size,
            kvcache=kvcache,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            attn_cp_rank=attn_cp_rank,
            pp_group=pp_group,
            attn_tp_group=attn_tp_group_eff,
            attn_cp_group=attn_cp_group,
        )

        self._mode = (
            FlexKVMode.IP if self.flexkv_connector.enable_layerwise else FlexKVMode.MP
        )
        _assert_mode_supports_disaggregation(self._mode, server_args)
        if self._mode is FlexKVMode.IP:
            # Register the eventfd counter onto sglang's KV pool so each
            # forward layer blocks on its own eventfd.
            self.flexkv_connector.register_layer_transfer_counter(kvcache)

        # Same hook HiCache uses: scheduler.release_host_resources → destroy().
        self.token_to_kv_pool_host = FlexKVHostReleaseShim(self.flexkv_connector)

        # CUDA streams (mirroring LMCRadixCache).
        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        # Two-phase MP load: stash marker between ``match_prefix`` and
        # ``init_load_back``.
        self._load_markers: dict[str, _LoadBackMarker] = {}
        # ``store_kv`` is async — we keep a lock on the source node
        # until FlexKV signals completion, draining in ``evict`` /
        # ``check_hicache_events``.
        self._inflight_store_nodes: dict[str, TreeNode] = {}
        self._node_lock = threading.Lock()

        # Async decode restore: opt-in, MP only, and only meaningful on a PD
        # decode server (that is the only place ``init_load_back`` runs on the
        # scheduler thread ahead of admission).
        self._async_restore_enabled = (
            self._mode is FlexKVMode.MP
            and server_args.disaggregation_mode == "decode"
            and envs.SGLANG_FLEXKV_ENABLE_ASYNC_DECODE_RESTORE.get()
        )
        self._async_restores: dict[str, _AsyncRestore] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:  # type: ignore[override]
        super().reset()
        if hasattr(self, "_load_markers"):
            self._load_markers.clear()
        if hasattr(self, "_async_restores"):
            # ``flexkv_connector.reset`` below drains the launches; the slots
            # themselves belong to the pool that ``super().reset()`` just wiped.
            self._async_restores.clear()
        if hasattr(self, "_inflight_store_nodes"):
            with self._node_lock:
                self._inflight_store_nodes.clear()
        if hasattr(self, "flexkv_connector"):
            self.flexkv_connector.reset()

    def shutdown(self) -> None:
        if hasattr(self, "token_to_kv_pool_host"):
            self.token_to_kv_pool_host.destroy()
        elif hasattr(self, "flexkv_connector"):
            self.flexkv_connector.shutdown()

    # ------------------------------------------------------------------
    # match_prefix
    # ------------------------------------------------------------------

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:  # type: ignore[override]
        """Look up the longest cached prefix on host KV (FlexKV).

        Dispatches to :meth:`_mp_match_prefix` or :meth:`_ip_match_prefix`
        depending on whether layerwise transfer is enabled.
        """
        key = params.key
        if self.disable or not key:
            return super().match_prefix(params)

        # FlexKV operates at page granularity — round the lookup query
        # down to a multiple of ``page_size`` so the hit count we report
        # back to sglang matches what FlexKV can actually serve.
        if self.page_size != 1:
            aligned_len = (len(key) // self.page_size) * self.page_size
            key = key[:aligned_len]

        base_res = super().match_prefix(params)
        if len(key) == 0:
            return base_res

        device_value: torch.Tensor = base_res.device_indices
        last_node: TreeNode = base_res.last_device_node

        if self._mode is FlexKVMode.MP:
            if params.req is None:
                return base_res
            return self._mp_match_prefix(
                key, base_res, device_value, last_node, params.req
            )
        return self._ip_match_prefix(key, base_res, device_value, last_node)

    def _mp_match_prefix(
        self,
        key: RadixKey,
        base_res: MatchResult,
        device_value: torch.Tensor,
        last_node: TreeNode,
        req: Req,
    ) -> MatchResult:
        """LOOKUP-only path. Sets ``host_hit_length`` on the result so
        the scheduler later invokes :meth:`init_load_back`."""
        if req.rid in self._load_markers:
            # A previous lookup for this rid never reached ``init_load_back``,
            # so FlexKV is still holding its task. The PD decode path matches
            # twice on purpose (once to size admission, once to restore); left
            # alone, the first task would be overwritten here and leak.
            self._load_markers.pop(req.rid, None)
            self.flexkv_connector.release_pending(req.rid)

        token_ids = key.raw_token_ids()
        device_len = int(device_value.numel())
        if device_len >= len(token_ids):
            return base_res

        # token_mask=True for tokens NOT on device — FlexKV decides
        # which of those it can serve.
        token_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        token_mask[device_len:] = True

        fkv_task_id, hit = self.flexkv_connector.lookup_kv(
            token_ids=token_ids,
            token_mask=token_mask,
            rid=req.rid,
            sglang_req_id=req.rid,
        )
        if hit <= 0:
            return base_res

        # Snapshot the matched key (the live key aliases ``req.fill_ids``).
        if token_ids is key.token_ids:
            token_ids_snap = token_ids[:]
        else:
            token_ids_snap = token_ids
        self._load_markers[req.rid] = _LoadBackMarker(
            key=RadixKey(token_ids_snap, key.extra_key, key.is_bigram),
            value_numel=device_len,
        )
        return MatchResult(
            device_indices=device_value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
            host_hit_length=hit,
        )

    def _ip_match_prefix(
        self,
        key: RadixKey,
        base_res: MatchResult,
        device_value: torch.Tensor,
        last_node: TreeNode,
    ) -> MatchResult:
        """Layerwise path: allocate slots and fire ``start_load_kv_layerwise``
        immediately. Per-layer hook waits during forward."""
        token_ids = key.raw_token_ids()
        device_len = int(device_value.numel())
        if device_len >= len(token_ids):
            return base_res

        # Quick LOOKUP first to discover how many slots we'd need.
        token_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        token_mask[device_len:] = True
        # No rid here — IP mode self-pops; pass a synthetic stable key.
        synthetic_rid = f"_ip_{id(key)}"
        _, hit = self.flexkv_connector.lookup_kv(
            token_ids=token_ids,
            token_mask=token_mask,
            rid=synthetic_rid,
            sglang_req_id=None,
        )
        if hit <= 0:
            return base_res

        result = self._allocate_and_load(
            key=key,
            value_numel=device_len,
            uncached_len=hit,
            last_node=last_node,
            load_fn=lambda slot_mapping: self.flexkv_connector.start_load_kv_layerwise(
                synthetic_rid, slot_mapping
            )[0],
        )
        if result is None:
            return base_res
        new_slots, new_node = result
        return MatchResult(
            device_indices=torch.cat([device_value, new_slots]),
            last_device_node=new_node,
            last_host_node=new_node,
            best_match_node=new_node,
        )

    # ------------------------------------------------------------------
    # init_load_back (MP RETRIEVE)
    # ------------------------------------------------------------------

    def init_load_back(  # type: ignore[override]
        self,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Optional[TreeNode]]:
        """MP RETRIEVE. Allocates uncached slots and fires the FlexKV
        load; inserts the resulting TreeNode."""
        req = params.req
        last_node: TreeNode = params.best_match_node
        marker = self._load_markers.pop(req.rid, None)
        if marker is None:
            # ``match_prefix`` decided there was no work to do, but the
            # scheduler still called us. Release any held task and
            # return an empty load.
            self.flexkv_connector.release_pending(req.rid)
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )

        if self._async_restore_enabled:
            return self._start_async_load_back(
                req=req,
                marker=marker,
                uncached_len=params.host_hit_length,
                last_node=last_node,
            )

        result = self._allocate_and_load(
            key=marker.key,
            value_numel=marker.value_numel,
            uncached_len=params.host_hit_length,
            last_node=last_node,
            load_fn=lambda slot_mapping: self.flexkv_connector.retrieve_kv(
                req.rid, slot_mapping
            ),
        )
        if result is None:
            # Allocation failed or load returned zero. ``retrieve_kv``
            # already cancels/cleans up on failure paths; release_pending
            # is idempotent for the case where allocation failed before
            # we even popped the held task.
            self.flexkv_connector.release_pending(req.rid)
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        return result

    # ------------------------------------------------------------------
    # Async MP RETRIEVE (PD decode)
    # ------------------------------------------------------------------

    def _start_async_load_back(
        self,
        *,
        req: Req,
        marker: _LoadBackMarker,
        uncached_len: int,
        last_node: TreeNode,
    ) -> Tuple[torch.Tensor, Optional[TreeNode]]:
        """Allocate the destination slots and launch the RETRIEVE without waiting.

        Unlike the synchronous path this does *not* insert a ``TreeNode`` yet:
        the slots hold no valid KV until FlexKV signals completion, and a node
        in the tree is reachable by ``match_prefix`` from any other request.
        The caller gets the slots and the unchanged ``last_node``;
        :meth:`poll_completed_restores` inserts the node once the KV is
        readable, which is what makes the tree — not the request — the owner of
        these slots, matching what ``cache_protected_len`` already assumes.
        """
        empty = torch.empty((0,), dtype=torch.int64, device=self.device)
        if uncached_len <= 0:
            self.flexkv_connector.release_pending(req.rid)
            return empty, last_node

        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.evict(EvictParams(num_tokens=uncached_len))
        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            self.flexkv_connector.release_pending(req.rid)
            return empty, last_node

        slots = token_slots.to(torch.int64)
        try:
            launched = self.flexkv_connector.start_retrieve_kv(req.rid, slots)
        except Exception:
            self.token_to_kv_pool_allocator.free(token_slots)
            self.flexkv_connector.release_pending(req.rid)
            raise
        if launched <= 0:
            self.token_to_kv_pool_allocator.free(token_slots)
            self.flexkv_connector.release_pending(req.rid)
            return empty, last_node

        # A short launch would leave a hole in the middle of the prefix, which
        # the restore contract cannot express — the decode side already told
        # prefill the whole prefix was covered. Fail the restore instead.
        if launched < uncached_len:
            logger.warning(
                "FlexKV async retrieve for rid=%s launched %d of %d slots; "
                "failing the restore rather than admitting a partial prefix",
                req.rid,
                launched,
                uncached_len,
            )
            self.flexkv_connector.wait_load(req.rid)
            self.token_to_kv_pool_allocator.free(token_slots)
            return empty, last_node

        self._async_restores[req.rid] = _AsyncRestore(
            req=req,
            key=marker.key,
            value_numel=marker.value_numel,
            slots=token_slots,
            last_node=last_node,
        )
        return token_slots, last_node

    def poll_completed_restores(  # type: ignore[override]
        self,
    ) -> dict[str, RestoreCompletion]:
        """Report async restores that finished, settling their slots.

        A success inserts the node the launch deferred; a failure frees the
        slots, which nothing else can reach.
        """
        if not self._async_restores:
            return {}
        results: dict[str, RestoreCompletion] = {}
        for rid, ok in self.flexkv_connector.check_completed_loads().items():
            restore = self._async_restores.pop(rid, None)
            if restore is None:
                continue
            if not ok:
                self._free_restored_slots(restore)
                results[rid] = RestoreCompletion(succeeded=False)
                continue
            results[rid] = RestoreCompletion(
                succeeded=True, node=self._insert_restored_node(restore)
            )
        return results

    def abort_restore(self, rid: str) -> None:  # type: ignore[override]
        """Settle an in-flight restore for a request that is going away.

        Draining first is mandatory: FlexKV is mid-flight writing these slots
        from another process, and there is no way to prove a cancelled task
        never started its copy. Only then are they safe to free.
        """
        restore = self._async_restores.pop(rid, None)
        if restore is None:
            return
        self.flexkv_connector.wait_load(rid)
        self._free_restored_slots(restore)

    def _insert_restored_node(self, restore: _AsyncRestore) -> Optional[TreeNode]:
        """Hand a completed restore's slots to the tree, returning the new node.

        Ownership has to land where the synchronous path leaves it: the
        request's ``cache_protected_len`` already covers these slots, so its
        release path will *not* free them and the tree must own them. The lock
        the launch took sits on ``last_node``, so it moves down to the new node
        — otherwise the node holding the request's own prefix would be
        evictable while the request is still waiting to run.

        Returns ``None`` when the tree won't take the slots. The launch left
        ``last_node`` free to grow, so another request may have inserted the
        very branch this node would occupy; overwriting it would orphan the
        slots already there. The KV in hand is still valid, so the restore
        stays a success and the request keeps the slots privately instead —
        pulling ``cache_protected_len`` back below them is what makes its own
        release path free them.
        """
        last_node = restore.last_node
        child_key = restore.node_key.child_key(self.page_size)
        if last_node.children.get(child_key) is not None:
            req = restore.req
            req.cache_protected_len = min(req.cache_protected_len, restore.value_numel)
            return None
        new_node = self._insert_loaded_node(
            key=restore.key,
            value_numel=restore.value_numel,
            slots=restore.slots,
            last_node=last_node,
        )
        # Raise before lowering so the path to the root never momentarily drops
        # to an evictable lock_ref.
        self.inc_lock_ref(new_node)
        self.dec_lock_ref(last_node)
        return new_node

    def _free_restored_slots(self, restore: _AsyncRestore) -> None:
        """Free a restore's slots directly.

        They are unreachable by anyone else — never inserted into the tree, and
        never written into ``req_to_token`` (that happens at commit, which a
        settled-as-failed restore never reaches), so the request's own release
        path would not find them.
        """
        self.token_to_kv_pool_allocator.free(restore.slots)

    def _allocate_and_load(
        self,
        *,
        key: RadixKey,
        value_numel: int,
        uncached_len: int,
        last_node: TreeNode,
        load_fn,
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        """Shared allocator + post-load bookkeeping for MP/IP.

        Returns ``(token_slots[:fetched], new_node)`` on success.
        ``None`` on either allocation failure or zero retrieved (in
        which case all slots are freed).
        """
        if uncached_len <= 0:
            return None

        # Evict to make room when needed.
        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.evict(EvictParams(num_tokens=uncached_len))
        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return None

        # The FlexKV ``launch`` interface takes the slot indices for the
        # tokens it will write — no leading ``-1`` padding (FlexKV has
        # no concept of "skip these device slots, they're already
        # cached"; we pass it exactly the destinations for the
        # uncached tail).
        num_retrieved = load_fn(token_slots.to(torch.int64))

        if num_retrieved <= 0:
            self.token_to_kv_pool_allocator.free(token_slots)
            return None

        # Free the tail of the over-allocation when FlexKV returned
        # fewer than expected.
        if num_retrieved < uncached_len:
            self.token_to_kv_pool_allocator.free(token_slots[num_retrieved:])
            fetched_slots = token_slots[:num_retrieved]
        else:
            fetched_slots = token_slots

        new_node = self._insert_loaded_node(
            key=key,
            value_numel=value_numel,
            slots=fetched_slots,
            last_node=last_node,
        )
        return fetched_slots, new_node

    def _insert_loaded_node(
        self,
        *,
        key: RadixKey,
        value_numel: int,
        slots: torch.Tensor,
        last_node: TreeNode,
    ) -> TreeNode:
        """Attach freshly-loaded device slots to the tree as a child of
        ``last_node``. Only call once the KV in ``slots`` is actually readable."""
        num_loaded = int(slots.numel())
        new_node = TreeNode(priority=last_node.priority)
        new_node.key = key[value_numel : value_numel + num_loaded]
        new_node.value = slots
        new_node.parent = last_node
        last_node.children[new_node.key.child_key(self.page_size)] = new_node
        self.evictable_size_ += num_loaded
        self._update_leaf_status(last_node)
        self._update_leaf_status(new_node)

        self._record_store_event(new_node.parent)
        self._record_store_event(new_node)
        return new_node

    # ------------------------------------------------------------------
    # cache_finished_req (STORE)
    # ------------------------------------------------------------------

    def cache_finished_req(  # type: ignore[override]
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ) -> None:
        """Base cache_finished_req then fire an async FlexKV store."""
        super().cache_finished_req(
            req, is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )
        if not is_insert:
            self._load_markers.pop(req.rid, None)
            return

        # Compute the committed prefix mirroring LMCRadixCache's logic.
        from sglang.srt.runtime_context import get_server_args

        global_server_args = get_server_args()
        topk = global_server_args.speculative_eagle_topk
        enable_kv_committed_len = topk is None or topk == 1
        if enable_kv_committed_len:
            kv_committed_len = req.kv_committed_len
        else:
            kv_committed_len = len(req.origin_input_ids) + max(
                len(req.output_ids) - 1, 0
            )

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        if not token_ids:
            return
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        # Anchor on the new last_device_node so FlexKV's lock matches
        # the node we'll later unlock when the store completes.
        match_result = super().match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids, req.extra_key))
        )
        new_last_node = match_result.last_device_node
        if new_last_node is None:
            return

        self.inc_lock_ref(new_last_node)
        try:
            with torch.cuda.stream(self.store_stream):
                fkv_task_id = self.flexkv_connector.store_kv(
                    rid=req.rid,
                    token_ids=list(token_ids),
                    kv_indices=kv_indices,
                    sglang_req_id=req.rid,
                )
        except Exception:  # noqa: BLE001
            self.dec_lock_ref(new_last_node)
            raise

        if fkv_task_id < 0:
            # Nothing to write back (either everything already in
            # FlexKV, or put_match failed / returned None).
            self.dec_lock_ref(new_last_node)
            return

        with self._node_lock:
            self._inflight_store_nodes[req.rid] = new_last_node

    # ------------------------------------------------------------------
    # evict + completion draining
    # ------------------------------------------------------------------

    def evict(self, params: EvictParams) -> EvictResult:  # type: ignore[override]
        """Drain completed stores before letting the base evict touch
        the source nodes."""
        if self.disable:
            return EvictResult()
        self._drain_completed_stores()
        # Make sure the store stream's GPU work is observed before any
        # eviction frees the source slots.
        self.store_stream.synchronize()
        return super().evict(params)

    def check_hicache_events(self) -> None:  # type: ignore[override]
        """Periodic non-blocking sweep called by the scheduler tick.

        Drains both store completions (so source nodes get unlocked
        quickly) and the launched-load tail (so the FlexKV pipe
        doesn't accumulate)."""
        self._drain_completed_stores()
        self.flexkv_connector.drain_launched_loads()

    def _drain_completed_stores(self) -> None:
        completed_rids = self.flexkv_connector.check_completed_stores()
        if not completed_rids:
            return
        with self._node_lock:
            for rid in completed_rids:
                node = self._inflight_store_nodes.pop(rid, None)
                if node is not None:
                    self.dec_lock_ref(node)

    # ------------------------------------------------------------------
    # Optional pass-throughs used by the scheduler
    # ------------------------------------------------------------------

    @property
    def decode_restore_driver(self) -> DecodeRestoreDriver:  # type: ignore[override]
        # Synchronous MP ``init_load_back`` is launch + wait, so the KV is on
        # device by the time it returns. IP mode is asynchronous, but it is
        # rejected on a decode server (see
        # ``_assert_mode_supports_disaggregation``).
        if self._async_restore_enabled:
            return DecodeRestoreDriver.PER_REQUEST_POLL
        return DecodeRestoreDriver.BLOCKING

    def has_inflight_io(self) -> bool:
        """True while a store or async restore is still in flight, so the
        scheduler can hold off destructive idle-time work (flush_cache /
        memory release)."""
        if self._async_restores:
            return True
        with self._node_lock:
            return bool(self._inflight_store_nodes)

    def release_aborted_request(self, rid: str) -> None:
        """Clean up tracking for an aborted request without invoking FlexKV."""
        self._load_markers.pop(rid, None)
        self.abort_restore(rid)
        with self._node_lock:
            node = self._inflight_store_nodes.pop(rid, None)
        if node is not None:
            self.dec_lock_ref(node)
        self.flexkv_connector.release_pending(rid)
        self.flexkv_connector.cancel_prefetch(rid)

    def prefetch_request(self, req: "Req") -> None:
        """Wait-complete FlexKV prefetch for a queued request.

        Owns fill-id refresh / page alignment so the scheduler only needs
        ``tree_cache.prefetch_request(req)``. Does not call FlexKV lookup
        (that happens at admission after prefetch completes).
        """
        req.init_next_round_input(tree_cache=None, cow_mamba=False)
        fill_ids = req.full_untruncated_fill_ids
        if not fill_ids:
            return
        match_end = req._compute_max_prefix_len(len(fill_ids))
        tokens = fill_ids[:match_end]
        self.prefetch_from_storage(req.rid, None, tokens)

    def prefetch_from_storage(
        self,
        rid: str,
        last_host_node=None,
        token_ids=None,
        last_hash=None,
        prefix_keys=None,
    ) -> None:
        """Kick off FlexKV prefetch (SSD/Remote/Mooncake → CPU).

        Extra HiCache-style args (``last_host_node`` / hashes) are ignored;
        FlexKV addresses blocks by token ids.
        """
        del last_host_node, last_hash, prefix_keys
        if not token_ids:
            return
        ids = list(token_ids)
        if self.page_size > 1:
            aligned = (len(ids) // self.page_size) * self.page_size
            ids = ids[:aligned]
        if not ids:
            return
        try:
            self.flexkv_connector.prefetch_async(
                rid, ids, sglang_req_id=rid
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[FlexKV] prefetch_from_storage: %s", exc)

    def check_prefetch_progress(self, rid: str) -> bool:
        return self.flexkv_connector.check_prefetch_progress(rid)

    def terminate_prefetch(self, rid: str) -> None:
        self.flexkv_connector.cancel_prefetch(rid)

    def pop_prefetch_loaded_tokens(self, rid: str) -> int:
        pop = getattr(self.flexkv_connector, "pop_prefetch_loaded_tokens", None)
        if callable(pop):
            return int(pop(rid))
        # Fallback until connector exposes actual prefetch hit length (M1).
        del rid
        return 0

    @property
    def hicache_storage_pass_prefix_keys(self) -> bool:
        # We pass token ids, not opaque key strings, so no prefix-key
        # accounting in the scheduler.
        return False

"""HiCache integration mixins for the decode side of PD disaggregation"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.mem_cache.base_prefix_cache import (
    DecodeRestoreDriver,
    InitLoadBackParams,
)

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass
class DecodePrefixMatch:
    prefix_indices: torch.Tensor
    l2_host_hit_length: int
    l3_storage_hit_length: int
    last_device_node: Any
    last_host_node: Any = None
    prefetch_registered: bool = False

    @property
    def l1_prefix_len(self) -> int:
        return len(self.prefix_indices)

    @property
    def decode_prefix_len(self) -> int:
        return self.l1_prefix_len + self.l2_host_hit_length + self.l3_storage_hit_length

    @property
    def needs_local_restore(self) -> bool:
        return self.decode_prefix_len > self.l1_prefix_len

    @property
    def restore_token_count(self) -> int:
        """Number of tokens that need L2/L3 load_back to device."""
        return self.decode_prefix_len - self.l1_prefix_len


class HiCacheRestoreResult(Enum):
    """Outcome of one tick of the HiCache local-restore state machine."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class DecodeHiCachePreallocMixin:
    """HiCache hooks for ``DecodePreallocQueue``: issue prefetch + reserve tokens."""

    def _build_decode_prefix_match(self, req: Req, result: Any) -> DecodePrefixMatch:
        """Convert a ``match_prefix_for_req`` result into ``DecodePrefixMatch``.

        Performs the optional L3 storage hit length query when decode-side
        HiCache is enabled and the last host node is backed up.
        """
        prefix_indices = result.device_indices
        l1_prefix_len = len(prefix_indices)
        l2_host_hit_length = result.host_hit_length

        l3_storage_hit_length = 0
        last_host_node = None
        # Only HiCache splits its host tier into L2 (host pool) + L3 (storage
        # backend) and needs the extra query. FlexKV reports everything it can
        # serve through ``host_hit_length`` and has no storage-hit query.
        if self.scheduler.enable_decode_hicache and (
            self.scheduler.enable_hierarchical_cache
        ):
            last_host_node = result.last_host_node
            if last_host_node.backuped or last_host_node is self.tree_cache.root_node:
                matched_len = l1_prefix_len + l2_host_hit_length
                suffix_tokens = req.origin_input_ids[matched_len:]
                last_hash = last_host_node.get_last_hash_value()
                prefix_keys = (
                    last_host_node.get_prefix_hash_values(last_host_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                l3_storage_hit_length = self.tree_cache.query_storage_hit_length(
                    last_host_node,
                    suffix_tokens,
                    last_hash,
                    prefix_keys,
                )

        return DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=l2_host_hit_length,
            l3_storage_hit_length=l3_storage_hit_length,
            last_device_node=result.last_device_node,
            last_host_node=last_host_node if l3_storage_hit_length > 0 else None,
        )

    def _start_hicache_prefetch(
        self, req: Req, prefix_match: Optional[DecodePrefixMatch]
    ) -> None:
        """Issue L3 storage prefetch after admission succeeds.

        On failure, degrades to L2-only restore by clearing l3 fields.
        """
        if (
            prefix_match is None
            or prefix_match.l3_storage_hit_length <= 0
            or prefix_match.last_host_node is None
        ):
            return
        try:
            node = prefix_match.last_host_node
            matched_len = prefix_match.l1_prefix_len + prefix_match.l2_host_hit_length
            suffix = req.origin_input_ids[
                matched_len : matched_len + prefix_match.l3_storage_hit_length
            ]
            last_hash = node.get_last_hash_value()
            prefix_keys = (
                node.get_prefix_hash_values(node.parent)
                if self.tree_cache.hicache_storage_pass_prefix_keys
                else None
            )
            self.tree_cache.prefetch_from_storage(
                req.rid, node, suffix, last_hash, prefix_keys
            )
            prefix_match.prefetch_registered = (
                req.rid in self.tree_cache.ongoing_prefetch
            )
        except Exception as e:
            logger.warning(
                "HiCache L3 prefetch failed for rid=%s: %s; falling back to L2-only LoadingBack",
                req.rid,
                e,
            )
            prefix_match.l3_storage_hit_length = 0
            prefix_match.prefetch_registered = False

    def _hicache_pending_restore_tokens(self) -> int:
        """Total device tokens reserved for pending HiCache L2/L3 load_back."""
        if not self.scheduler.enable_decode_hicache:
            return 0
        return sum(
            dr.prefix_match.restore_token_count
            for dr in self.transfer_queue.queue
            if dr.prefix_match is not None
            and dr.hicache_restore_status == HiCacheRestoreResult.PENDING
            and dr.hicache_restored_node is None
        )


class HiCacheRestoreGatedKVReceiver:
    """Wraps a kv_receiver so KVPoll.Success is gated on HiCache restore READY."""

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        poll = self.decode_req.kv_receiver.poll()
        if (
            poll == KVPoll.Success
            and self.decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
        ):
            return KVPoll.Transferring
        return poll


class DecodeHiCacheTransferMixin:
    """HiCache hooks for ``DecodeTransferQueue``: drive restore state machine."""

    def _clean_hicache_prefetch_resources(self, decode_req: DecodeRequest) -> None:
        # A FlexKV backend holds no prefetch, but it may hold a lookup task from
        # a match that never reached init_load_back, or an in-flight async load;
        # release_aborted_request is how it gives both back. Only the layerwise
        # path (MERGED_EVENT-like, self-popping) has nothing to hand over.
        holds_backend_state = (
            decode_req.prefix_match is not None
            and decode_req.prefix_match.prefetch_registered
        ) or (
            self.tree_cache.decode_restore_driver
            is not DecodeRestoreDriver.MERGED_EVENT
        )
        if holds_backend_state:
            self.tree_cache.release_aborted_request(decode_req.req.rid)
        if decode_req.hicache_restored_node is not None:
            self.tree_cache.dec_lock_ref(decode_req.hicache_restored_node)
            decode_req.hicache_restored_node = None

    def _try_hicache_queue_load_back(self, dr: DecodeRequest) -> bool:
        """Queue one L2->L1 load_back op for ``dr``; True iff a DMA was queued.

        On success, ``dr.hicache_restored_node`` and ``hicache_restored_kv_indices``
        are populated, and an inc_lock_ref is held until commit/abort.
        Trivial cases (all-on-device / no needed coverage) auto-flip to READY.
        Failback paths flip to FAILED.
        """
        pm = dr.prefix_match

        # Wait for L3 -> L2 prefetch to drain (skip when no L3 hit).
        if pm.l3_storage_hit_length > 0:
            if not self.tree_cache.check_prefetch_progress(dr.req.rid):
                return False
            self.tree_cache.pop_prefetch_loaded_tokens(dr.req.rid)

        # Re-match: req.last_node / prefix_indices updated to current device state.
        rematch = match_prefix_for_req(
            self.tree_cache,
            dr.req,
            dr.req.origin_input_ids,
            cow_mamba=False,
            include_req=True,
        )
        new_indices, restored_node = self.tree_cache.init_load_back(
            InitLoadBackParams(
                best_match_node=rematch.best_match_node,
                host_hit_length=rematch.host_hit_length,
                req=dr.req,
            )
        )
        # Failback: total coverage < required prefix means device alloc likely failed.
        if len(rematch.device_indices) + len(new_indices) < pm.decode_prefix_len:
            logger.warning(
                "HiCache load_back failed for rid=%s: device_indices=%d, "
                "new_indices=%d, expected decode_prefix_len=%d (l1=%d, l2=%d, l3=%d)",
                dr.req.rid,
                len(rematch.device_indices),
                len(new_indices),
                pm.decode_prefix_len,
                pm.l1_prefix_len,
                pm.l2_host_hit_length,
                pm.l3_storage_hit_length,
            )
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        dr.hicache_restored_kv_indices = torch.cat(
            [rematch.device_indices[pm.l1_prefix_len :], new_indices]
        )
        dr.hicache_restored_node = restored_node
        self.tree_cache.inc_lock_ref(restored_node)

        if len(new_indices) == 0:
            # Whole prefix already on device; no DMA needed.
            dr.hicache_restore_status = HiCacheRestoreResult.READY
            return False
        return True

    def _process_hicache_local_restores(self, decode_reqs: List[DecodeRequest]) -> None:
        driver = self.tree_cache.decode_restore_driver
        # Poll before collecting: a request that completes on this tick should
        # reach READY now rather than waiting a full tick for the next pass.
        if driver is DecodeRestoreDriver.PER_REQUEST_POLL:
            self._reap_polled_restores(decode_reqs)

        active = self._collect_active_restores(decode_reqs)
        if not active:
            return
        if driver is DecodeRestoreDriver.MERGED_EVENT:
            self._advance_merged_event_restores(active)
        elif driver is DecodeRestoreDriver.PER_REQUEST_POLL:
            self._advance_polled_restores(active)
        else:
            self._advance_blocking_restores(active)

    def _collect_active_restores(
        self, decode_reqs: List[DecodeRequest]
    ) -> List[DecodeRequest]:
        """Keep only PENDING reqs that still need restore work; trivially-done
        reqs (no prefix_match / nothing to restore) flip to READY."""
        active: List[DecodeRequest] = []
        for dr in decode_reqs:
            if dr.hicache_restore_status != HiCacheRestoreResult.PENDING:
                continue
            pm = dr.prefix_match
            if pm is None or not pm.needs_local_restore:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
                continue
            active.append(dr)
        return active

    def _advance_blocking_restores(self, active: List[DecodeRequest]) -> None:
        """Driver for backends whose ``init_load_back`` blocks until the KV is on
        device (FlexKV MP mode). No event polling — the load is done when the
        call returns.

        The blocking happens on the scheduler thread, so restore at most one
        request per tick: a queue of admitted requests would otherwise chain
        their host->device copies into a single stall that also holds up the
        running decode batch.
        """
        dr = active[0]
        if self._try_hicache_queue_load_back(dr):
            dr.hicache_restore_status = HiCacheRestoreResult.READY

    def _reap_polled_restores(self, decode_reqs: List[DecodeRequest]) -> None:
        """Apply out-of-band restore completions to their requests.

        The backend reports each rid exactly once, so a rid whose request has
        already left the queue must not be looked up again — it was settled by
        ``_clean_hicache_prefetch_resources`` on the way out.
        """
        completed = self.tree_cache.poll_completed_restores()
        if not completed:
            return
        by_rid = {dr.req.rid: dr for dr in decode_reqs}
        for rid, result in completed.items():
            dr = by_rid.get(rid)
            if dr is None or dr.hicache_restore_status != HiCacheRestoreResult.PENDING:
                continue
            if result.node is not None:
                # The backend deferred insertion until the KV landed and moved
                # the lock ref onto the new node; follow it, or commit/abort
                # would unlock a node that no longer covers this prefix.
                dr.hicache_restored_node = result.node
            dr.hicache_restore_status = (
                HiCacheRestoreResult.READY
                if result.succeeded
                else HiCacheRestoreResult.FAILED
            )

    def _advance_polled_restores(self, active: List[DecodeRequest]) -> None:
        """Driver for backends that start a per-request transfer and report
        completion out of band (FlexKV async MP).

        There is no shared event slot and no merged kick, so every request that
        has not launched yet launches now; completions arrive via
        ``_reap_polled_restores`` on a later tick.
        """
        for dr in active:
            if dr.hicache_restored_node is not None:
                # Already launched; waiting on the backend's completion report.
                continue
            self._try_hicache_queue_load_back(dr)

    def _advance_merged_event_restores(self, active: List[DecodeRequest]) -> None:
        if not hasattr(self.tree_cache, "is_load_back_event_done"):
            return

        # Phase A: advance in-flight DMAs to READY.
        for dr in active:
            if (
                dr.hicache_restored_node is not None
                and self.tree_cache.is_load_back_event_done(
                    dr.hicache_load_consumer_index
                )
            ):
                dr.hicache_restore_status = HiCacheRestoreResult.READY

        # Phase B: queue new load_back ops if the next slot is free.
        # The (producer_index + 1) check ensures we never overwrite a still-in-flight slot:
        # if a previous req holds that slot and isn't done, its event won't be signaled.
        counter = self.tree_cache.cache_controller.layer_done_counter
        if not self.tree_cache.is_load_back_event_done(
            (counter.producer_index + 1) % counter.num_counters
        ):
            return
        queued = [
            dr
            for dr in active
            if dr.hicache_restored_node is None
            and self._try_hicache_queue_load_back(dr)
        ]
        if not queued:
            return

        # Phase C: kick off merged DMA, bind consumer_index for Phase A polling next tick.
        consumer_index = self.tree_cache.ready_to_load_host_cache()
        if consumer_index < 0:
            for dr in queued:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
            return
        for dr in queued:
            dr.hicache_load_consumer_index = consumer_index

    def _commit_hicache_local_restore_to_req(self, decode_req: DecodeRequest) -> None:
        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.needs_local_restore:
            return

        self.tree_cache.dec_lock_ref(prefix_match.last_device_node)

        self.tree_cache.req_to_token_pool.write(
            (
                decode_req.req.req_pool_idx,
                slice(prefix_match.l1_prefix_len, prefix_match.decode_prefix_len),
            ),
            decode_req.hicache_restored_kv_indices,
        )
        decode_req.req.prefix_indices = torch.cat(
            [prefix_match.prefix_indices, decode_req.hicache_restored_kv_indices]
        )
        decode_req.req.last_node = decode_req.hicache_restored_node

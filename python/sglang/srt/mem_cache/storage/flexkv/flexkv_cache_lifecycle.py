"""Shared request identity, restore ownership, and prefetch lifecycle for FlexKV.

Tree insertion, SWA allocation, and store node locking stay in the adapters.
This mixin owns only the protocol shared by ordinary and hybrid caches.
"""

from __future__ import annotations

import json
import logging
import time
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.utils import (
    get_hash_str,
    hash_str_to_int64,
    storage_namespace_seed,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


def _request_key(handle: CacheRequestHandle) -> str:
    """Keep attempt identity in FlexKV's string tracking keys."""
    return json.dumps([handle.rid, handle.attempt_id], separators=(",", ":"))


def _namespace_kwargs(connector, extra_key, cache_salt) -> Optional[dict]:
    """Return scoped connector arguments, or None when reuse is unsupported.

    A single JSON component distinguishes null, empty strings and embedded
    separators without depending on FlexKV's namespace component delimiter.
    Unscoped requests keep their historical hash and connector call signature.
    """
    if extra_key is None and cache_salt is None:
        return {}
    if getattr(connector, "supports_cache_namespace", False) is not True:
        return None
    return {
        "namespace": [
            json.dumps(
                ["sglang-cache-v1", extra_key, cache_salt], separators=(",", ":")
            )
        ]
    }


def _namespace_parts(namespace: Optional[list[str]]) -> tuple[Any, Any]:
    """``(extra_key, cache_salt)`` back from a namespace :func:`_namespace_kwargs` built."""
    if not namespace:
        return None, None
    _, extra_key, cache_salt = json.loads(namespace[0])
    return extra_key, cache_salt


def _block_hash_kwargs(
    connector,
    token_ids: Sequence[int],
    page_size: int,
    extra_key=None,
    cache_salt=None,
) -> dict:
    """``block_hashes`` for a FlexKV that keys blocks by SGLang's page hashes: the
    namespace-seeded SHA-256 page chain of HiCache storage keys, as signed int64,
    which a router can recompute from request tokens. Empty otherwise."""
    if getattr(connector, "wants_caller_block_hashes", False) is not True:
        return {}
    n = len(token_ids) // page_size * page_size
    if n == 0:
        return {"block_hashes": np.empty(0, dtype=np.int64)}
    pages = get_hash_str(
        list(token_ids[:n]), storage_namespace_seed(extra_key, cache_salt), page_size
    )
    return {
        "block_hashes": np.fromiter(
            (hash_str_to_int64(h) for h in pages), dtype=np.int64, count=len(pages)
        )
    }


@dataclass
class _RestoreLease:
    """Fresh destination slots retained until cache commit or a fenced reset."""

    generation: int
    rid: str
    req: Req
    device_indices: torch.Tensor


class FlexKVCacheLifecycleMixin:
    # Mid-decode stores leave the SWA window out (see store_decode_progress).
    _decode_store_full_only = False

    def _init_decode_offload(self) -> None:
        """``SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE``: also store a decoding
        request's KV every N generated tokens (page-aligned, at least a page),
        so the store at finish only writes the tail. 0 = store at finish only."""
        stride = int(envs.SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE.get() or 0)
        if stride > 0:
            stride = max(self.page_size, stride // self.page_size * self.page_size)
        self.decode_offload_stride = stride
        # In-flight mid-decode stores: store rid -> the KV row it reads, and
        # per row how many are in flight. A row is fenced before any of its
        # slots can be freed (free_kv_row, checkpoint dedup).
        self._decode_stores: dict[str, int] = {}
        self._decode_store_rows: dict[int, int] = {}
        self._decode_store_seq = 0

    def store_decode_progress(self, req: Req) -> None:
        """Store the page-aligned KV a decoding request has computed so far,
        once ``decode_offload_stride`` new tokens accumulated since its last
        store.

        Reads the request's own KV row; nothing is inserted into the radix
        tree and no slot changes owner, so the row stays the request's until
        release_kv_cache. The store only reads it: the row is fenced (the store
        must complete) before release can free any of its slots. Hybrid (SWA)
        caches store Full KV only and hold the last page back, so the store at
        finish always has a new block to carry the SWA window."""
        stride = self.decode_offload_stride
        # The last output token's KV is computed by the next forward.
        committed = len(req.origin_input_ids) + len(req.output_ids) - 1
        aligned = committed // self.page_size * self.page_size
        if self._decode_store_full_only:
            aligned -= self.page_size
        stored = getattr(req, "_flexkv_decode_stored_len", None)
        if stored is None:
            if self.__dict__.get("_pd_decode", False):
                # PD decode: prefill stores the prompt it computed. Count the
                # stride from the prompt, so the first store here carries it
                # with new output once prefill's copy is visible cluster-wide.
                req._flexkv_decode_stored_len = (
                    len(req.origin_input_ids) // self.page_size * self.page_size
                )
                return
            stored = 0
        if aligned - stored < stride or aligned <= 0:
            return
        namespace_kwargs = _namespace_kwargs(
            self.flexkv_connector, req.extra_key, req.cache_salt
        )
        if namespace_kwargs is None or req.kv.req_pool_idx is None:
            return
        row = int(req.kv.req_pool_idx)
        token_ids = (req.origin_input_ids + req.output_ids)[:aligned]
        kv_indices = self.req_to_token_pool.req_to_token[row, :aligned].to(torch.int64)
        handle = req.cache_request_handle
        self._decode_store_seq += 1
        store_rid = json.dumps(
            [handle.rid, f"{handle.attempt_id}:decode-store:{self._decode_store_seq}"],
            separators=(",", ":"),
        )
        task_id = self._launch_decode_store(
            store_rid, req, list(token_ids), kv_indices, namespace_kwargs.get("namespace")
        )
        req._flexkv_decode_stored_len = aligned
        if task_id < 0:
            return  # nothing new to write; the row was never handed to FlexKV
        self._decode_stores[store_rid] = row
        self._decode_store_rows[row] = self._decode_store_rows.get(row, 0) + 1
        logger.info(
            "[FlexKV] decode store rid=%s tokens=%d (stride=%d, full_only=%s)",
            req.rid, aligned, stride, self._decode_store_full_only,
        )

    def _forget_decode_stores(self, completed_rids) -> None:
        stores = self.__dict__.get("_decode_stores")
        if not stores:
            return
        for rid in completed_rids:
            row = stores.pop(rid, None)
            if row is None:
                continue
            left = self._decode_store_rows.get(row, 0) - 1
            if left > 0:
                self._decode_store_rows[row] = left
            else:
                self._decode_store_rows.pop(row, None)

    def _fence_decode_stores(self, row: Optional[int]) -> None:
        """Wait until no mid-decode store reads KV row ``row``.

        Called on every TP/CP rank for the same request in the same order
        (release_kv_cache), and the store-completion poll is a cross-rank
        protocol whose result the leader decides, so the ranks leave the loop
        together."""
        rows = self.__dict__.get("_decode_store_rows")
        if not rows or row is None or not rows.get(int(row)):
            return
        row = int(row)
        start = time.perf_counter()
        while self._decode_store_rows.get(row):
            self._drain_completed_stores()
            if self._decode_store_rows.get(row):
                time.sleep(0.0002)
        waited_ms = (time.perf_counter() - start) * 1e3
        if waited_ms > 1.0:
            logger.info("[FlexKV] waited %.1f ms for decode stores of row %d", waited_ms, row)

    def _fence_request_decode_stores(self, req: Req) -> None:
        if self.__dict__.get("_decode_store_rows"):
            self._fence_decode_stores(req.kv.req_pool_idx)

    def free_kv_row(self, kv, ranges) -> None:
        # A mid-decode store may still be reading these slots.
        if self.__dict__.get("_decode_store_rows"):
            self._fence_decode_stores(kv.req_pool_idx)
        super().free_kv_row(kv, ranges)

    def _reset_decode_offload(self) -> None:
        # Only after connector.reset fenced every launched transfer.
        if "_decode_stores" in self.__dict__:
            self._decode_stores.clear()
            self._decode_store_rows.clear()

    def _init_restore_state(self) -> None:
        self._restore_leases: dict[str, _RestoreLease] = {}
        self._aborted_restore_leases: dict[int, _RestoreLease] = {}
        self._restore_generation = 0

    def _register_restore_lease(self, req: Req, slots: torch.Tensor) -> _RestoreLease:
        rid = _request_key(req.cache_request_handle)
        if rid in self._restore_leases:
            raise RuntimeError(f"FlexKV duplicate load-back: rid={rid}")
        lease = _RestoreLease(
            generation=self._restore_generation, rid=rid, req=req, device_indices=slots
        )
        self._restore_generation += 1
        self._restore_leases[rid] = lease
        req.pending_restore_generation = lease.generation
        req.pending_restore_slots = slots
        return lease

    def has_uncommitted_restore(self, req: Req) -> bool:
        return _request_key(req.cache_request_handle) in self._restore_leases

    def _release_load_marker(self, handle: CacheRequestHandle) -> None:
        # A held lookup pins its matched host prefix until init_load_back or an
        # explicit release. The connector only cancels it when a later lookup
        # for the same rid hits, so rematch and completion release it here.
        rid = _request_key(handle)
        if self._load_markers.pop(rid, None) is not None:
            self.flexkv_connector.release_pending(rid)

    @staticmethod
    def _restore_lease_matches_req(req: Req, lease: _RestoreLease) -> bool:
        return (
            lease.req is req
            and getattr(req, "pending_restore_generation", None) == lease.generation
            and getattr(req, "pending_restore_slots", None) is lease.device_indices
        )

    def _validate_restore_lease(self, req: Req) -> Optional[_RestoreLease]:
        lease = self._restore_leases.get(_request_key(req.cache_request_handle))
        if lease is None or lease.req is not req:
            # An older aborted Req may finish after a new Req reused its rid.
            # Find by object identity, never commit the successor's lease.
            lease = next(
                (
                    item
                    for item in self._aborted_restore_leases.values()
                    if item.req is req
                ),
                lease,
            )
        if lease is not None and not self._restore_lease_matches_req(req, lease):
            # Ordinary completion mutates/frees KV. Continuing on a mismatch
            # could free a different owner's slots and free them again at reset.
            raise RuntimeError(
                f"FlexKV restore lease mismatch: rid={_request_key(req.cache_request_handle)}"
            )
        return lease

    def _forget_restore_lease(self, lease: _RestoreLease) -> None:
        if self._restore_leases.get(lease.rid) is lease:
            self._restore_leases.pop(lease.rid)
        if self._aborted_restore_leases.get(lease.generation) is lease:
            self._aborted_restore_leases.pop(lease.generation)
        # Reset trusts the allocation ledger, not mutable request fields. Do
        # not overwrite fields belonging to another generation of the Req.
        if self._restore_lease_matches_req(lease.req, lease):
            lease.req.pending_restore_generation = None
            lease.req.pending_restore_slots = None
            lease.req._flexkv_uncached_restore = False

    def _commit_restore(self, req: Req) -> None:
        lease = self._validate_restore_lease(req)
        if lease is not None:
            self._forget_restore_lease(lease)
        else:
            req._flexkv_uncached_restore = False

    def _free_uncommitted_restores(self) -> None:
        # Only call after connector.reset has fenced all DMA. Request metadata
        # may be stale; each ledger entry still identifies the allocation to free.
        failed = []
        leases = list(self._restore_leases.values()) + list(
            self._aborted_restore_leases.values()
        )
        for lease in leases:
            try:
                self.token_to_kv_pool_allocator.free(lease.device_indices)
            except Exception:
                logger.exception(
                    "FlexKV failed to free restore slots rid=%s", lease.rid
                )
                failed.append(lease.rid)
                continue
            self._forget_restore_lease(lease)
        if failed:
            # Attempt every allocation, retain failures for diagnosis/retry, and
            # do not report a successful reset or discard the remaining ledger.
            raise RuntimeError(f"FlexKV failed to free restore allocations: {failed}")

    def prefetch_request(self, req: Req) -> None:
        """Start queued prefetch when device (L1) does not already cover the prefix.

        Only SGLang's GPU radix is probed here: FlexKV prefetch cannot see L1
        (``ignore_gpu``). A full L1 hit skips FlexKV entirely. Otherwise one
        ``prefetch_async`` jointly matches L2+Mooncake; L2-only hits produce no
        REMOTE transfer. L2→L1 H2D stays on the admission ``lookup_kv`` path.
        """
        # Do not call match_prefix with req= here: that path may plan host work.
        req.init_next_round_input(tree_cache=None, cow_mamba=False)
        fill_ids = req.full_untruncated_fill_ids
        if not fill_ids:
            return
        match_end = req._compute_max_prefix_len(len(fill_ids))
        raw = list(fill_ids[:match_end])
        aligned_len = len(raw) // self.page_size * self.page_size
        if aligned_len <= 0:
            return
        raw = raw[:aligned_len]

        key = RadixKey(
            array("q", raw),
            req.extra_key,
            is_bigram=bool(getattr(self, "is_eagle", False)),
            cache_salt=req.cache_salt,
        )
        device_len = int(
            self.match_prefix(MatchPrefixParams(key=key, req=None)).device_prefix_len
        )
        if device_len >= aligned_len:
            logger.info(
                "[FlexKV] skip prefetch: device hit rid=%s tokens=%d",
                req.rid,
                device_len,
            )
            return

        self.prefetch_from_storage(
            req.cache_request_handle,
            None,
            raw[device_len:],
            matched_prefix_tokens=raw[:device_len],
            extra_key=req.extra_key,
            cache_salt=req.cache_salt,
        )

    def prefetch_from_storage(
        self,
        handle: CacheRequestHandle,
        last_host_node=None,
        token_ids=None,
        last_hash=None,
        prefix_keys=None,
        *,
        matched_prefix_tokens=None,
        extra_key=None,
        cache_salt=None,
    ) -> None:
        """Pass the complete token hash chain and the candidate's absolute offset."""
        rid = _request_key(handle)
        del last_host_node, last_hash, prefix_keys
        namespace_kwargs = _namespace_kwargs(
            self.flexkv_connector, extra_key, cache_salt
        )
        if namespace_kwargs is None or not token_ids:
            return
        prefix = [] if matched_prefix_tokens is None else list(matched_prefix_tokens)
        ids = prefix + list(token_ids)
        ids = ids[: len(ids) // self.page_size * self.page_size]
        if len(ids) <= len(prefix):
            return
        hash_kwargs = _block_hash_kwargs(
            self.flexkv_connector, ids, self.page_size, extra_key, cache_salt
        )
        if getattr(self.flexkv_connector, "_chunked_prefetch", False):
            self.flexkv_connector.prefetch_async(
                rid,
                ids,
                sglang_req_id=handle.rid,
                candidate_start_token=len(prefix),
                **namespace_kwargs,
                **hash_kwargs,
            )
        else:
            self.flexkv_connector.prefetch_async(
                rid, ids, sglang_req_id=handle.rid, **namespace_kwargs, **hash_kwargs
            )

    def check_prefetch_progress(self, handle: CacheRequestHandle) -> bool:
        rid = _request_key(handle)
        return self.flexkv_connector.check_prefetch_progress(rid)

    def terminate_prefetch(self, handle: CacheRequestHandle) -> None:
        rid = _request_key(handle)
        self.flexkv_connector.cancel_prefetch(rid)

    def pop_prefetch_loaded_span(
        self, handle: CacheRequestHandle
    ) -> tuple[int, Optional[int]]:
        rid = _request_key(handle)
        if getattr(self.flexkv_connector, "_chunked_prefetch", False):
            return self.flexkv_connector.pop_prefetch_loaded_span(rid)
        return self.pop_prefetch_loaded_tokens(handle), None

    def pop_prefetch_loaded_tokens(self, handle: CacheRequestHandle) -> int:
        rid = _request_key(handle)
        pop = getattr(self.flexkv_connector, "pop_prefetch_loaded_tokens", None)
        if callable(pop):
            return int(pop(rid))
        # Older FlexKV builds do not track the materialized REMOTE2H prefix.
        # Reporting 0 attributes the whole hit to the host tier, which only
        # skews the #cached-host / #cached-storage split in the logs.
        del rid
        return 0

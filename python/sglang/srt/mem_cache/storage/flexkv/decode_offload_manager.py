"""Decode-side incremental KV offload backed by FlexKV."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache import (
        FlexKVRadixCache,
    )
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class FlexKVDecodeKVCacheOffloadManager:
    """Manage decode-side incremental KV offload through FlexKV.

    The lifecycle mirrors ``DecodeKVCacheOffloadManager``: incremental chunks
    are written out while decoding continues, and GPU slots are released only
    after request finish and all in-flight stores for that request complete.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: FlexKVRadixCache,
        server_args: ServerArgs,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.tree_cache = tree_cache
        self.page_size = server_args.page_size
        self.request_counter = 0
        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.page_size
        else:
            self.offload_stride = max(
                self.page_size, (env_stride // self.page_size) * self.page_size
            )

        self.ongoing_offload: dict[str, Req] = {}
        self.offloaded_state: dict[str, OffloadedState] = {}
        self.offload_inflight: dict[str, int] = {}
        logger.info("Enable FlexKV offload kv cache for decode side")

    def _mark_offload_started(self, rid: str) -> None:
        self.offload_inflight[rid] = self.offload_inflight.get(rid, 0) + 1

    def _mark_offload_finished(self, rid: str) -> None:
        count = self.offload_inflight.get(rid, 0)
        if count <= 1:
            self.offload_inflight.pop(rid, None)
        else:
            self.offload_inflight[rid] = count - 1

    def _has_inflight_offload(self, rid: str) -> bool:
        return self.offload_inflight.get(rid, 0) > 0

    def _get_state(self, req: Req) -> OffloadedState:
        state = self.offloaded_state.get(req.rid)
        if state is None:
            state = OffloadedState(
                prefill_len=len(req.origin_input_ids)
                // self.page_size
                * self.page_size,
                inc_len=0,
                last_hash=None,
            )
            self.offloaded_state[req.rid] = state
        return state

    def offload_kv_cache(self, req: Req) -> bool:
        if req.req_pool_idx == -1 or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        state = self._get_state(req)
        incremental_total = len(all_tokens) - state.prefill_len
        incremental_new = incremental_total - state.inc_len
        incremental_aligned_len = (
            incremental_new // self.offload_stride * self.offload_stride
        )
        if incremental_aligned_len == 0:
            return False

        end = state.prefill_len + state.inc_len + incremental_aligned_len
        token_ids = all_tokens[:end]
        kv_indices = token_indices[:end]

        self.request_counter += 1
        store_id = f"{req.rid}:decode_offload:{self.request_counter}"
        store_started = self.tree_cache.store_decode_offload(
            store_id, token_ids, kv_indices
        )
        state.inc_len += incremental_aligned_len
        if not store_started:
            if req.finished() and not self._has_inflight_offload(req.rid):
                self._release_finished_req(req, state.prefill_len)
            return True

        self._mark_offload_started(req.rid)
        self.ongoing_offload[store_id] = req
        return True

    def check_offload_progress(self) -> None:
        for store_id in self.tree_cache.pop_completed_decode_offloads():
            req = self.ongoing_offload.pop(store_id, None)
            if req is None:
                continue
            self._mark_offload_finished(req.rid)
            if req.finished() and not self._has_inflight_offload(req.rid):
                state = self.offloaded_state.get(req.rid)
                start_offset = state.prefill_len if state is not None else 0
                self._release_finished_req(req, start_offset)

    def _release_finished_req(self, req: Req, start_offset: int) -> None:
        if req.req_pool_idx is None or req.req_pool_idx == -1:
            return

        kv_committed_len = req.pop_committed_kv_cache()
        state = self.offloaded_state.get(req.rid)
        if state is not None and state.prefill_len > 0:
            prefill_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : state.prefill_len
            ]
            self.token_to_kv_pool_allocator.free(prefill_indices)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, start_offset:kv_committed_len
        ]
        self.token_to_kv_pool_allocator.free(kv_indices)

        start_p, end_p = req.pop_overallocated_kv_cache()
        if self.page_size > 1:
            start_p = ceil_align(start_p, self.page_size)
        if start_p < end_p:
            overalloc_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_p:end_p
            ]
            self.token_to_kv_pool_allocator.free(overalloc_indices)

        self.req_to_token_pool.free(req)
        self.tree_cache.protected_size_ -= len(req.prefix_indices)
        self.offloaded_state.pop(req.rid, None)

    def finalize_release_on_finish(self, req: Req) -> None:
        if req.req_pool_idx is None or req.req_pool_idx == -1:
            return
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_len = len(req.origin_input_ids) // self.page_size * self.page_size
            self.offloaded_state[req.rid] = OffloadedState(
                prefill_len=prefill_len, inc_len=0, last_hash=None
            )
        else:
            prefill_len = state.prefill_len
        if self._has_inflight_offload(req.rid):
            return
        self._release_finished_req(req, prefill_len)

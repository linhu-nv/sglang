"""PD decode helpers for the FlexKV radix-cache backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs


class DecodeFlexKVMixin:
    """FlexKV-specific PD decode restore path.

    PD decode's scheduler only understands device-resident prefix hits unless
    the backend is HiCache. This mixin turns FlexKV MP host hits into normal
    device hits inside ``match_prefix``.
    """

    def _init_decode_flexkv(self, server_args: ServerArgs) -> None:
        self._pd_decode_mode = (
            getattr(server_args, "disaggregation_mode", None) == "decode"
        )

    def _decode_flexkv_match_prefix(
        self,
        *,
        key: RadixKey,
        base_res: MatchResult,
        device_value: torch.Tensor,
        last_node: TreeNode,
        req: Req,
        hit: int,
    ) -> MatchResult:
        # Keep the on-device prefix alive while _allocate_and_load may evict.
        if last_node is not self.root_node:
            self.inc_lock_ref(last_node)
        try:
            result = self._allocate_and_load(
                key=key,
                value_numel=int(device_value.numel()),
                uncached_len=hit,
                last_node=last_node,
                load_fn=lambda slot_mapping: self.flexkv_connector.retrieve_kv(
                    req.rid, slot_mapping
                ),
            )
        finally:
            if last_node is not self.root_node:
                self.dec_lock_ref(last_node)

        if result is None:
            self.flexkv_connector.release_pending(req.rid)
            return base_res

        new_slots, new_node = result
        return MatchResult(
            device_indices=torch.cat([device_value, new_slots]),
            last_device_node=new_node,
            last_host_node=new_node,
            best_match_node=new_node,
        )

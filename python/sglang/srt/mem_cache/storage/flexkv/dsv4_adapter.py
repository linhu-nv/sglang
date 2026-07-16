"""DeepSeek V4 GPU-pool adapter for FlexKV's multi-group API.

SGLang stores DSV4 KV in four physically independent pools: c4, c128,
c4-indexer, and SWA.  FlexKV's ``dpskv4_refactor`` branch represents the first
three as heterogeneous layer groups and registers SWA as a separate channel.
This module translates between those two descriptions without making FlexKV
import SGLang internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from flexkv.common.config import (
    CacheConfig,
    LayerGroupSpec,
    ModelConfig,
    recompute_cache_block_counts,
)
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType


@dataclass
class DSV4Registration:
    """Arguments for ``KVTPClient.register_to_server`` on dpskv4_refactor."""

    kv_caches: list[torch.Tensor]
    kv_layout: KVCacheLayout
    layer_groups: list[LayerGroupSpec]
    gpu_layouts: list[KVCacheLayout]
    handles_per_group: list[list[torch.Tensor]]
    swa_caches: list[torch.Tensor]
    swa_layout: KVCacheLayout

    def register_kwargs(self) -> dict[str, Any]:
        return {
            "kv_caches": self.kv_caches,
            "kv_layout": self.kv_layout,
            "layer_groups": self.layer_groups,
            "gpu_layouts": self.gpu_layouts,
            "handles_per_group": self.handles_per_group,
            "swa_caches": self.swa_caches,
            "swa_layout": self.swa_layout,
        }


def is_dsv4_kv_pool(kvcache: Any) -> bool:
    """Return whether ``kvcache`` exposes SGLang's split DSV4 pool contract."""

    return all(
        hasattr(kvcache, attr)
        for attr in (
            "compression_ratios",
            "swa_kv_pool",
            "c4_kv_pool",
            "c128_kv_pool",
            "c4_indexer_kv_pool",
            "translate_loc_from_full_to_swa",
        )
    )


def _buffers(pool: Any, attribute: str) -> list[torch.Tensor]:
    values = getattr(pool, attribute, None)
    return [] if values is None else list(values)


def _validate_group_buffers(
    name: str,
    buffers: list[torch.Tensor],
    expected_layers: int,
) -> None:
    if len(buffers) != expected_layers:
        raise ValueError(
            f"DSV4 {name} buffer count {len(buffers)} does not match "
            f"its layer count {expected_layers}"
        )
    if not buffers:
        raise ValueError(f"DSV4 {name} group is empty")
    shape = buffers[0].shape
    dtype = buffers[0].dtype
    if len(shape) != 2:
        raise ValueError(
            f"DSV4 {name} buffers must be two-dimensional, got {tuple(shape)}"
        )
    if any(buffer.shape != shape or buffer.dtype != dtype for buffer in buffers):
        raise ValueError(f"DSV4 {name} buffers must have homogeneous shape/dtype")


def _make_group(
    *,
    name: str,
    pool: Any,
    buffers: list[torch.Tensor],
    layer_indices: list[int],
    compress_ratio: int,
    full_page_size: int,
) -> tuple[LayerGroupSpec, KVCacheLayout]:
    _validate_group_buffers(name, buffers, len(layer_indices))
    if full_page_size % compress_ratio != 0:
        raise ValueError(
            f"DSV4 {name} compression ratio {compress_ratio} does not divide "
            f"page size {full_page_size}"
        )
    group_tokens = full_page_size // compress_ratio
    pool_page_size = int(getattr(pool, "page_size", group_tokens))
    if pool_page_size != group_tokens:
        raise ValueError(
            f"DSV4 {name} pool page size {pool_page_size} does not match "
            f"the expected compressed page size {group_tokens}"
        )
    page_width = int(buffers[0].shape[1])
    if page_width % group_tokens != 0:
        raise ValueError(
            f"DSV4 {name} physical page width {page_width} is not divisible "
            f"by {group_tokens} tokens/page"
        )

    group = LayerGroupSpec(
        num_layers=len(layer_indices),
        num_kv_heads=1,
        head_size=page_width // group_tokens,
        layer_indices=layer_indices,
        dtype=buffers[0].dtype,
        compress_ratio=compress_ratio,
    )
    layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=group.num_layers,
        num_block=int(buffers[0].shape[0]),
        tokens_per_block=group_tokens,
        num_head=1,
        head_size=group.head_size,
        is_mla=True,
    )
    return group, layout


def build_dsv4_registration(
    kvcache: Any,
    model_config: ModelConfig,
    cache_config: CacheConfig,
) -> DSV4Registration:
    """Describe one non-unified SGLang ``DeepSeekV4TokenToKVPool``."""

    if not is_dsv4_kv_pool(kvcache):
        raise TypeError(f"unsupported DSV4 KV pool type: {type(kvcache).__name__}")
    if bool(getattr(kvcache, "_unified_kv", False)):
        raise NotImplementedError(
            "FlexKV requires SGLang's non-unified DSV4 pools; disable the "
            "experimental unified-KV Triton layout"
        )
    if model_config.pp_size != 1 or model_config.nnodes != 1:
        raise NotImplementedError(
            "DSV4 FlexKV integration currently requires one node and PP=1"
        )
    if model_config.cp_size != 1:
        raise NotImplementedError(
            "DSV4 FlexKV integration currently requires attention CP=1"
        )

    page_size = int(getattr(kvcache, "page_size", cache_config.tokens_per_block))
    if page_size != cache_config.tokens_per_block:
        raise ValueError(
            f"SGLang/FlexKV page size mismatch: {page_size} != "
            f"{cache_config.tokens_per_block}"
        )

    all_ratios = list(getattr(kvcache, "compression_ratios"))
    stage_start = int(getattr(kvcache, "_stage_start", 0))
    stage_end = int(getattr(kvcache, "_stage_end", len(all_ratios)))
    if stage_start != 0 or stage_end != model_config.num_layers:
        raise NotImplementedError("DSV4 heterogeneous registration requires PP=1")
    # Match DeepSeekV4TokenToKVPool's own allocation contract. Checkpoints may
    # carry an extra trailing compression ratio for an MTP/placeholder layer,
    # while the target model and its device pools cover only this stage slice.
    ratios = all_ratios[stage_start:stage_end]
    if len(ratios) != model_config.num_layers:
        raise ValueError(
            f"DSV4 stage has {len(ratios)} compression ratios from "
            f"{len(all_ratios)} total entries for "
            f"{model_config.num_layers} model layers"
        )

    unknown = sorted({ratio for ratio in ratios if ratio not in (0, 4, 128)})
    if unknown:
        raise ValueError(f"unsupported DSV4 compression ratios: {unknown}")
    c4_layers = [index for index, ratio in enumerate(ratios) if ratio == 4]
    c128_layers = [index for index, ratio in enumerate(ratios) if ratio == 128]

    groups: list[LayerGroupSpec] = []
    layouts: list[KVCacheLayout] = []
    group_buffers: list[list[torch.Tensor]] = []
    for name, pool, attribute, layers, ratio in (
        ("c4", kvcache.c4_kv_pool, "kv_buffer", c4_layers, 4),
        ("c128", kvcache.c128_kv_pool, "kv_buffer", c128_layers, 128),
        (
            "c4_indexer",
            kvcache.c4_indexer_kv_pool,
            "index_k_with_scale_buffer",
            c4_layers,
            4,
        ),
    ):
        if not layers:
            continue
        buffers = _buffers(pool, attribute)
        group, layout = _make_group(
            name=name,
            pool=pool,
            buffers=buffers,
            layer_indices=layers,
            compress_ratio=ratio,
            full_page_size=page_size,
        )
        groups.append(group)
        layouts.append(layout)
        group_buffers.append(buffers)

    if not groups:
        raise ValueError("DSV4 registration found no c4/c128 GPU groups")
    main_num_blocks = layouts[0].num_block
    if any(layout.num_block != main_num_blocks for layout in layouts):
        raise ValueError(
            "DSV4 c4/c128/indexer pools must expose equal logical page counts"
        )

    swa_pool = kvcache.swa_kv_pool
    swa_buffers = _buffers(swa_pool, "kv_buffer")
    _validate_group_buffers("swa", swa_buffers, model_config.num_layers)
    swa_page_size = int(getattr(kvcache, "swa_page_size", page_size))
    if swa_page_size != page_size:
        raise ValueError(
            f"DSV4 SWA page size {swa_page_size} does not match {page_size}"
        )
    swa_page_width = int(swa_buffers[0].shape[1])
    if swa_page_width % swa_page_size != 0:
        raise ValueError("DSV4 SWA physical page width is not token-divisible")
    swa_bytes_per_token = swa_page_width // swa_page_size

    swa_config = getattr(cache_config, "swa", None)
    if swa_config is None or not swa_config.enabled:
        raise ValueError("DeepSeek V4 requires an enabled FlexKV SWA host pool")
    if not cache_config.enable_swa_transfer:
        raise ValueError("DeepSeek V4 requires FlexKV SWA transfer to be enabled")
    if swa_config.num_swa_layers != len(swa_buffers):
        raise ValueError(
            f"FlexKV expects {swa_config.num_swa_layers} SWA layers but "
            f"SGLang exposes {len(swa_buffers)}"
        )
    if swa_config.bytes_per_token_per_layer != swa_bytes_per_token:
        raise ValueError(
            "FlexKV/SGLang SWA physical width mismatch: "
            f"{swa_config.bytes_per_token_per_layer} != {swa_bytes_per_token}"
        )

    # ``kv_layout`` and ``kv_caches`` retain the legacy registration fields.
    # Multi-group workers consume ``handles_per_group``/``gpu_layouts``; the
    # legacy layout supplies model-wide metadata such as the original layer
    # count and MLA KV dimension.
    compatibility_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=model_config.num_layers,
        num_block=main_num_blocks,
        tokens_per_block=page_size,
        num_head=1,
        head_size=1,
        is_mla=True,
    )
    swa_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=len(swa_buffers),
        num_block=int(swa_buffers[0].shape[0]),
        tokens_per_block=swa_page_size,
        num_head=1,
        head_size=swa_bytes_per_token,
        is_mla=True,
    )
    return DSV4Registration(
        kv_caches=list(group_buffers[0]),
        kv_layout=compatibility_layout,
        layer_groups=groups,
        gpu_layouts=layouts,
        handles_per_group=group_buffers,
        swa_caches=swa_buffers,
        swa_layout=swa_layout,
    )


def prepare_dsv4_registration(
    kvcache: Any,
    model_config: ModelConfig,
    cache_config: CacheConfig,
) -> DSV4Registration:
    """Build registration metadata before ``KVManager`` allocates host pools."""

    registration = build_dsv4_registration(kvcache, model_config, cache_config)
    model_config.layer_groups = registration.layer_groups
    # ``layer_groups`` is a cached property input and can be discovered after
    # FlexKV freezes ModelConfig.  Clear a possible earlier None result.
    model_config.__dict__.pop("layer_member_map", None)
    validate = getattr(model_config, "_validate_layer_groups", None)
    if callable(validate):
        validate()
    recompute_cache_block_counts(model_config, cache_config)
    return registration


def translate_dsv4_swa_slot_mapping(
    kvcache: Any,
    full_slot_mapping: torch.Tensor,
) -> np.ndarray:
    """Translate the trailing full-KV page to DSV4's independent SWA slots."""

    if full_slot_mapping.ndim != 1:
        raise ValueError("full_slot_mapping must be one-dimensional")
    page_size = int(getattr(kvcache, "swa_page_size", 0))
    if page_size <= 0 or full_slot_mapping.numel() < page_size:
        raise ValueError("slot mapping does not contain a complete SWA page")

    mapping_table = getattr(kvcache, "full_to_swa_index_mapping", None)
    if mapping_table is None:
        raise RuntimeError("SGLang has not registered its full-to-SWA mapping")
    final_page = full_slot_mapping[-page_size:].to(
        device=mapping_table.device,
        dtype=torch.int64,
    )
    if (final_page < 0).any():
        raise ValueError("full slot mapping contains an unallocated slot")
    swa_mapping = kvcache.translate_loc_from_full_to_swa(final_page)
    if (swa_mapping < 0).any():
        raise ValueError("SGLang returned an unallocated SWA slot")
    return swa_mapping.detach().cpu().numpy().astype(np.int64, copy=False)

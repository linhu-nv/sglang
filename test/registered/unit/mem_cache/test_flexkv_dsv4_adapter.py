from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

try:
    from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
except (ImportError, OSError) as first_import_error:
    if isinstance(first_import_error, ModuleNotFoundError) and (
        first_import_error.name == "flexkv"
    ):
        raise unittest.SkipTest("FlexKV is not installed") from first_import_error

    # These tests exercise only FlexKV's Python config/layout contracts. A
    # source checkout on a CPU-only host may lack ZeroMQ or libcudart, both of
    # which TensorSharedHandle imports even though this test never uses it.
    memory_handle_stub = ModuleType("flexkv.common.memory_handle")

    class _TensorSharedHandle:
        pass

    memory_handle_stub.TensorSharedHandle = _TensorSharedHandle
    sys.modules[memory_handle_stub.__name__] = memory_handle_stub
    sys.modules.pop("flexkv.common.storage", None)
    sys.modules.pop("flexkv.common.config", None)
    try:
        from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
    except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
        raise unittest.SkipTest("FlexKV config cannot be imported") from exc


_REPO_ROOT = Path(__file__).resolve().parents[4]
_ADAPTER_PATH = (
    _REPO_ROOT
    / "python/sglang/srt/mem_cache/storage/flexkv/dsv4_adapter.py"
)
_SPEC = importlib.util.spec_from_file_location("_flexkv_dsv4_adapter_test", _ADAPTER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ADAPTER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ADAPTER
_SPEC.loader.exec_module(_ADAPTER)


class _FakeDSV4Pool:
    page_size = 256
    swa_page_size = 256
    compression_ratios = [0, 4, 128, 4]
    _stage_start = 0
    _stage_end = 4
    _unified_kv = False

    def __init__(self) -> None:
        # DSV4 physical pages are byte-packed. C4 and SWA use a padded
        # 585-byte/token stride; C128 has its own 576-byte page padding.
        self.c4_kv_pool = SimpleNamespace(
            page_size=64,
            kv_buffer=[torch.empty(8, 64 * 585, dtype=torch.uint8) for _ in range(2)],
        )
        self.c128_kv_pool = SimpleNamespace(
            page_size=2,
            kv_buffer=[torch.empty(8, 2 * 864, dtype=torch.uint8)],
        )
        self.c4_indexer_kv_pool = SimpleNamespace(
            page_size=64,
            index_k_with_scale_buffer=[
                torch.empty(8, 64 * 164, dtype=torch.uint8) for _ in range(2)
            ],
        )
        # SWA capacity is intentionally different from the compressed pools;
        # it has an independent slot space.
        self.swa_kv_pool = SimpleNamespace(
            kv_buffer=[
                torch.empty(5, 256 * 585, dtype=torch.uint8) for _ in range(4)
            ]
        )
        self.full_to_swa_index_mapping = torch.arange(4096, dtype=torch.int64) + 9000

    def translate_loc_from_full_to_swa(self, indices: torch.Tensor) -> torch.Tensor:
        return self.full_to_swa_index_mapping[indices]


def _configs() -> tuple[ModelConfig, CacheConfig]:
    model = ModelConfig(
        num_layers=4,
        num_kv_heads=1,
        head_size=1,
        use_mla=True,
        dtype=torch.uint8,
    )
    cache = CacheConfig(
        tokens_per_block=256,
        num_cpu_blocks=7,
        swa=SWAPoolConfig(
            enabled=True,
            num_slots=32,
            num_swa_layers=4,
            bytes_per_token_per_layer=585,
        ),
        enable_swa_transfer=True,
    )
    return model, cache


class TestFlexKVDSV4Adapter(unittest.TestCase):
    def test_builds_multigroup_and_independent_swa_registration(self) -> None:
        model, cache = _configs()
        registration = _ADAPTER.build_dsv4_registration(
            _FakeDSV4Pool(), model, cache
        )

        self.assertEqual(
            [group.compress_ratio for group in registration.layer_groups],
            [4, 128, 4],
        )
        self.assertEqual(registration.layer_groups[0].layer_indices, [1, 3])
        self.assertEqual(registration.layer_groups[1].layer_indices, [2])
        self.assertEqual(registration.layer_groups[2].layer_indices, [1, 3])
        self.assertEqual(
            [layout.head_size for layout in registration.gpu_layouts],
            [585, 864, 164],
        )
        self.assertEqual(registration.kv_layout.num_layer, 4)
        self.assertEqual(registration.swa_layout.num_block, 5)
        self.assertEqual(registration.swa_layout.head_size, 585)
        self.assertEqual(
            set(registration.register_kwargs()),
            {
                "kv_caches",
                "kv_layout",
                "layer_groups",
                "gpu_layouts",
                "handles_per_group",
                "swa_caches",
                "swa_layout",
            },
        )

    def test_ignores_compression_ratios_after_the_model_stage(self) -> None:
        model, cache = _configs()
        pool = _FakeDSV4Pool()
        pool.compression_ratios = [0, 4, 128, 4, 0]

        registration = _ADAPTER.build_dsv4_registration(pool, model, cache)

        self.assertEqual(registration.layer_groups[0].layer_indices, [1, 3])
        self.assertEqual(registration.layer_groups[1].layer_indices, [2])
        self.assertEqual(registration.kv_layout.num_layer, 4)
        self.assertEqual(registration.swa_layout.num_layer, 4)

    def test_prepare_attaches_groups_before_host_pool_allocation(self) -> None:
        model, cache = _configs()
        cache._user_cpu_cache_gb = 0.01
        registration = _ADAPTER.prepare_dsv4_registration(
            _FakeDSV4Pool(), model, cache
        )

        self.assertIs(model.layer_groups, registration.layer_groups)
        self.assertEqual(model.layer_member_map.members_of(0), ())
        self.assertEqual(model.layer_member_map.members_of(1), ((0, 0), (2, 0)))
        self.assertNotEqual(cache.num_cpu_blocks, 7)

    def test_translates_only_the_trailing_full_page_to_swa(self) -> None:
        pool = _FakeDSV4Pool()
        full_slots = torch.arange(512, dtype=torch.int64)
        translated = _ADAPTER.translate_dsv4_swa_slot_mapping(pool, full_slots)

        np.testing.assert_array_equal(
            translated,
            np.arange(256, 512, dtype=np.int64) + 9000,
        )

    def test_rejects_unified_dsv4_layout(self) -> None:
        model, cache = _configs()
        pool = _FakeDSV4Pool()
        pool._unified_kv = True
        with self.assertRaisesRegex(NotImplementedError, "non-unified"):
            _ADAPTER.build_dsv4_registration(pool, model, cache)


if __name__ == "__main__":
    unittest.main()

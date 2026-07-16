"""Contract tests for FlexKV on SGLang's hybrid FULL/SWA allocator."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache import (
        FlexKVRadixCache,
    )
except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
    raise unittest.SkipTest("FlexKV runtime dependencies are unavailable") from exc


class TestFlexKVRadixCacheHybridContract(unittest.TestCase):
    @staticmethod
    def _cache(*, sliding_window_size=128) -> FlexKVRadixCache:
        cache = object.__new__(FlexKVRadixCache)
        cache.disable = False
        cache.sliding_window_size = sliding_window_size
        cache.evictable_size_ = 512
        cache.protected_size_ = 256
        cache.flexkv_connector = MagicMock()
        cache.flexkv_connector.check_completed_stores.return_value = []
        cache.store_stream = MagicMock()
        cache._node_lock = MagicMock()
        cache._inflight_store_nodes = {}
        cache._inflight_decode_offloads = set()
        cache._completed_decode_offloads = set()
        return cache

    def test_constructor_copies_scheduler_sliding_window_metadata(self) -> None:
        params = SimpleNamespace(sliding_window_size=128, page_size=256)
        allocator = MagicMock()
        connector = MagicMock(enable_layerwise=False)

        def init_radix(cache, _params) -> None:
            cache.token_to_kv_pool_allocator = allocator

        with (
            patch.object(RadixCache, "__init__", autospec=True) as radix_init,
            patch(
                "sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache."
                "FlexKVConnector",
                return_value=connector,
            ),
            patch.object(FlexKVRadixCache, "_init_decode_flexkv"),
            patch("torch.cuda.Stream", return_value=MagicMock()),
        ):
            radix_init.side_effect = init_radix
            cache = FlexKVRadixCache(
                params=params,
                model_config=MagicMock(),
                server_args=MagicMock(),
                tp_rank=0,
                tp_size=4,
                dp_rank=0,
                pp_rank=0,
                attn_cp_rank=0,
            )

        self.assertEqual(cache.sliding_window_size, 128)

    def test_exposes_coupled_full_and_swa_budget_views(self) -> None:
        cache = self._cache()

        self.assertEqual(cache.sliding_window_size, 128)
        self.assertEqual(cache.full_evictable_size(), 512)
        self.assertEqual(cache.swa_evictable_size(), 512)
        self.assertEqual(cache.full_protected_size(), 256)
        self.assertEqual(cache.swa_protected_size(), 256)
        # Independent SWA maintenance is deliberately disabled: this cache
        # releases FULL and SWA together when a radix node is evicted.
        self.assertFalse(cache.supports_swa())

    def test_swa_only_request_evicts_coupled_radix_nodes(self) -> None:
        cache = self._cache()
        with patch.object(
            RadixCache,
            "evict",
            return_value=EvictResult(num_tokens_evicted=384),
        ) as base_evict:
            result = cache.evict(EvictParams(num_tokens=0, swa_num_tokens=300))

        base_evict.assert_called_once()
        self.assertEqual(base_evict.call_args.args[0].num_tokens, 300)
        self.assertEqual(result.num_tokens_evicted, 384)
        self.assertEqual(result.swa_num_tokens_evicted, 384)


if __name__ == "__main__":
    unittest.main()

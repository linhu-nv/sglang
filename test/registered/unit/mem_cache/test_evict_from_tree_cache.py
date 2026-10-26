from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams as _EvictParams
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeSWAAllocator:
    # Exercise the allocator implementation that owns the eviction policy.
    evict_to_free_tokens = SWATokenToKVPoolAllocator.evict_to_free_tokens

    def __init__(self, availability):
        self.availability = availability

    def full_available_size(self):
        return self.availability["full"]

    def swa_available_size(self):
        return self.availability["swa"]


def _make_evict_fixture(*, full_available: int, swa_available: int):
    availability = {"full": full_available, "swa": swa_available}
    allocator = _FakeSWAAllocator(availability)

    def evict(params: _EvictParams):
        availability["full"] += params.num_tokens
        availability["swa"] += params.swa_num_tokens

    tree_cache = SimpleNamespace(
        is_chunk_cache=lambda: False,
        supports_prefix_sharing=lambda: True,
        token_to_kv_pool_allocator=allocator,
        evict_for_alloc=MagicMock(side_effect=evict),
    )
    return tree_cache, availability


def test_tail_only_demand_does_not_evict_available_swa():
    tree_cache, availability = _make_evict_fixture(full_available=80, swa_available=64)

    evict_from_tree_cache(tree_cache, 128, swa_num_tokens=64)

    tree_cache.evict_for_alloc.assert_called_once_with(
        _EvictParams(num_tokens=48, swa_num_tokens=0)
    )
    assert availability == {"full": 128, "swa": 64}


def test_tail_only_demand_evicts_only_swa_deficit():
    tree_cache, availability = _make_evict_fixture(full_available=128, swa_available=32)

    evict_from_tree_cache(tree_cache, 128, swa_num_tokens=64)

    tree_cache.evict_for_alloc.assert_called_once_with(
        _EvictParams(num_tokens=0, swa_num_tokens=32)
    )
    assert availability == {"full": 128, "swa": 64}


def test_tail_only_demand_with_enough_capacity_does_not_evict():
    tree_cache, _ = _make_evict_fixture(full_available=128, swa_available=64)

    evict_from_tree_cache(tree_cache, 128, swa_num_tokens=64)

    tree_cache.evict_for_alloc.assert_not_called()


def test_default_swa_demand_matches_full_demand():
    tree_cache, availability = _make_evict_fixture(full_available=128, swa_available=64)

    evict_from_tree_cache(tree_cache, 128)

    tree_cache.evict_for_alloc.assert_called_once_with(
        _EvictParams(num_tokens=0, swa_num_tokens=64)
    )
    assert availability == {"full": 128, "swa": 128}


def test_evict_stops_when_cache_cannot_make_progress():
    tree_cache, _ = _make_evict_fixture(full_available=0, swa_available=0)
    tree_cache.evict_for_alloc.side_effect = None

    evict_from_tree_cache(tree_cache, 128, swa_num_tokens=64)

    tree_cache.evict_for_alloc.assert_called_once_with(
        _EvictParams(num_tokens=128, swa_num_tokens=64)
    )

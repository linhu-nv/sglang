"""FlexKV mid-decode stores (SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE).

A decoding request's KV is stored every ``stride`` page-aligned tokens from its
own KV row: nothing is inserted into the radix tree and no slot is freed or
changes owner on that path; the row is fenced (in-flight stores complete)
before release can free it. Hybrid (SWA) caches store Full KV only and hold
the last page back so the store at finish carries the window.
"""

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.storage.flexkv.flexkv_cache_lifecycle import (
    FlexKVCacheLifecycleMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

PAGE = 4
STRIDE = 8


def _load(module_file: str, cls: str):
    connector_name = "flexkv.integration.sglang.connector"
    stub = ModuleType(connector_name)
    stub.FlexKVConnector = object
    stub.FlexKVHostReleaseShim = object
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/storage/flexkv"
        / module_file
    )
    name = f"_under_test_{module_file[:-3]}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        with patch.dict(sys.modules, {connector_name: stub}):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return getattr(module, cls)


FlexKVRadixCache = _load("flexkv_radix_cache.py", "FlexKVRadixCache")
FlexKVHybridRadixCache = _load("flexkv_hybrid_radix_cache.py", "FlexKVHybridRadixCache")


def _req(origin_len=10, row=3):
    return SimpleNamespace(
        rid="r1",
        origin_input_ids=list(range(1000, 1000 + origin_len)),
        output_ids=[],
        extra_key=None,
        cache_salt=None,
        kv=SimpleNamespace(req_pool_idx=row),
        cache_request_handle=SimpleNamespace(rid="r1", attempt_id=0),
    )


def _arm(cache, full_only=False):
    """Give a bare cache what store_decode_progress reads."""
    cache.page_size = PAGE
    cache.flexkv_connector = MagicMock()
    cache.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(8 * 256, dtype=torch.int32).reshape(8, 256)
    )
    with patch("sglang.srt.mem_cache.storage.flexkv.flexkv_cache_lifecycle.envs") as envs:
        envs.SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE.get.return_value = STRIDE
        cache._init_decode_offload()
    if full_only:
        cache._decode_store_full_only = True
    calls = []

    def _launch(store_rid, req, token_ids, kv_indices, namespace):
        calls.append((store_rid, list(token_ids), kv_indices.clone()))
        return 7 + len(calls)

    cache._launch_decode_store = _launch
    return calls


def _decode(cache, req, steps):
    for t in range(steps):
        req.output_ids.append(2000 + t)
        cache.store_decode_progress(req)


def test_stride_rounds_to_pages_and_zero_is_off():
    cache = FlexKVRadixCache.__new__(FlexKVRadixCache)
    cache.page_size = PAGE
    for raw, want in ((0, 0), (1, PAGE), (STRIDE + 3, STRIDE)):
        with patch(
            "sglang.srt.mem_cache.storage.flexkv.flexkv_cache_lifecycle.envs"
        ) as envs:
            envs.SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE.get.return_value = raw
            cache._init_decode_offload()
        assert cache.decode_offload_stride == want


def test_scheduler_hook_is_off_without_stride_or_with_spec():
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        SchedulerBatchResultProcessor,
        flexkv_decode_offload_stride,
    )

    plain = SimpleNamespace(is_none=lambda: True)
    spec = SimpleNamespace(is_none=lambda: False)
    for enable, stride, algo, want in (
        (False, 8, plain, 0),
        (True, 0, plain, 0),
        (True, 8, spec, 0),
        (True, 8, plain, 8),
    ):
        got = flexkv_decode_offload_stride(
            tree_cache=SimpleNamespace(decode_offload_stride=stride),
            enable_flexkv=enable,
            spec_algorithm=algo,
        )
        assert got == want
    # The processor is a frozen dataclass: the stride must be a field set at
    # construction, defaulting to off for callers that do not pass it.
    field = SchedulerBatchResultProcessor.__dataclass_fields__["flexkv_decode_offload_stride"]
    assert field.default == 0


def test_stores_every_stride_page_aligned_from_the_request_row():
    cache = FlexKVRadixCache.__new__(FlexKVRadixCache)
    calls = _arm(cache)
    req = _req()
    _decode(cache, req, 20)
    # committed = 10 + outputs - 1; stores at aligned 8, 16, 24
    assert [len(t) for _, t, _ in calls] == [8, 16, 24]
    full = req.origin_input_ids + req.output_ids
    row = cache.req_to_token_pool.req_to_token[3]
    for _, tokens, indices in calls:
        assert tokens == full[: len(tokens)]
        assert indices.tolist() == row[: len(tokens)].tolist()
        assert len(tokens) % PAGE == 0
    assert cache._decode_store_rows == {3: 3}
    assert len({rid for rid, _, _ in calls}) == 3
    assert all(json.loads(rid)[0] == "r1" for rid, _, _ in calls)


def test_hybrid_holds_the_last_page_back():
    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    calls = _arm(cache, full_only=True)
    req = _req()
    _decode(cache, req, 20)
    # aligned - PAGE: 8 at committed>=12, 16 at >=20, 24 at >=28
    assert [len(t) for _, t, _ in calls] == [8, 16, 24]
    assert FlexKVHybridRadixCache._decode_store_full_only


def test_intermediate_store_never_touches_the_tree_or_frees_slots():
    cache = FlexKVRadixCache.__new__(FlexKVRadixCache)
    _arm(cache)
    cache.token_to_kv_pool_allocator = MagicMock()
    req = _req()
    guarded = {
        name: MagicMock(name=name)
        for name in ("insert", "checkpoint", "inc_lock_ref", "dec_lock_ref", "evict")
    }
    with patch.multiple(RadixCache, **guarded), patch.object(
        BasePrefixCache, "free_kv_row"
    ) as free_row:
        _decode(cache, req, 20)
    for mock in guarded.values():
        mock.assert_not_called()
    free_row.assert_not_called()
    cache.token_to_kv_pool_allocator.free.assert_not_called()


class _Base:
    def __init__(self):
        self.events = []

    def free_kv_row(self, kv, ranges):
        self.events.append(("free", kv.req_pool_idx))


class _Cache(FlexKVCacheLifecycleMixin, _Base):
    page_size = PAGE


def test_free_waits_for_the_rows_in_flight_stores():
    cache = _Cache()
    cache.decode_offload_stride = STRIDE
    cache._decode_stores = {"a": 3, "b": 5}
    cache._decode_store_rows = {3: 1, 5: 1}
    polls = []

    def _drain():
        polls.append(len(polls))
        cache.events.append(("poll",))
        if len(polls) == 2:  # the store on row 3 completes on the 2nd poll
            cache._forget_decode_stores(["a"])

    cache._drain_completed_stores = _drain
    cache.free_kv_row(SimpleNamespace(req_pool_idx=3), [(0, 8)])
    assert cache.events == [("poll",), ("poll",), ("free", 3)]
    assert cache._decode_store_rows == {5: 1}
    cache.events.clear()
    cache.free_kv_row(SimpleNamespace(req_pool_idx=4), [(0, 8)])  # no store on row 4
    assert cache.events == [("free", 4)]


def test_checkpoint_fences_before_the_tree_insert():
    cache = FlexKVRadixCache.__new__(FlexKVRadixCache)
    order = []
    cache._fence_request_decode_stores = lambda req: order.append(("fence", req.kv.req_pool_idx))
    cache._validate_restore_lease = lambda req: None
    req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=3), finished=lambda: False)
    with patch.object(
        RadixCache, "checkpoint", side_effect=lambda *a, **k: order.append(("insert",))
    ):
        cache._commit_restore = lambda req: None
        cache._release_load_marker = lambda handle: None
        cache.has_uncommitted_restore = lambda req: True
        req.cache_request_handle = None
        cache.checkpoint(req, up_to=8)
    assert order[:2] == [("fence", 3), ("insert",)]


def test_hybrid_mid_decode_store_asks_flexkv_for_full_kv_only():
    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache.page_size = 1
    cache.flexkv_connector = MagicMock()
    cache.flexkv_connector.store_kv.return_value = 11
    with patch.object(torch.cuda, "current_stream", lambda: MagicMock()), patch.object(
        torch.cuda, "stream", lambda s: nullcontext()
    ):
        task = cache._launch_decode_store(
            '["r1","0:decode-store:1"]',
            SimpleNamespace(rid="r1"),
            [1, 2, 3],
            torch.tensor([4, 5, 6]),
            None,
        )
    assert task == 11
    assert cache.flexkv_connector.store_kv.call_args.kwargs["swa"] is False


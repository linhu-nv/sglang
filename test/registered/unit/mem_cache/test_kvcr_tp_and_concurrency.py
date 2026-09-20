"""KVCRStore under TP/DP rank colocation and under concurrent operations.

Two assumptions the TP=1 single-request validation never exercised:

  1. **Rank colocation.** Every ``(dp, attn_cp, attn_tp)`` rank builds its own
     KVCRStore on the same host holding a different head slice. The port it
     *binds* must follow the full coordinate; the port it *dials* must follow
     only the within-DP part, since the router already picked the source DP
     rank. Both failures are silent: block keys are token hashes carrying no
     rank identity, so a wrong dial succeeds and returns another rank's shard.

  2. **Concurrent operations.** ``poll_completed()`` both drains a queue and
     advances state machines, so the scheduler-thread ``tick`` and
     ``_drain_until`` race for every completion; ``_completed_ops`` keeps the
     loser's result.

CPU-only: no mem_pool, so ``_build_kvcr`` never runs and the core is a fake.

    python -m pytest test/registered/unit/mem_cache/test_kvcr_tp_and_concurrency.py -v
"""

from __future__ import annotations

import importlib.util
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.kvcr.router_hint import (
    ROUTER_HINT_KEY,
    build_envelope,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# ``kvcr`` absent is a legitimate tier configuration: the CPU suite runs without
# the wheel, so that case skips. ``kvcr`` present but this adapter failing to
# import is not -- it means the public API moved out from under us, so let the
# ImportError fail the file rather than report every case below as green.
_HAS_KVCR = importlib.util.find_spec("kvcr") is not None

if _HAS_KVCR:
    from kvcr.types import OpEntryStatus, QueryStatus

    from sglang.srt.mem_cache.storage.kvcr.kvcr_store import KVCRStore

# A module-level raise would be shorter, but SkipTest outside a test is an
# uncaught exception: the CI runner invokes this file as a subprocess and reads
# its exit code, so it would fail the whole CPU suite rather than skipping.
_needs_kvcr = unittest.skipUnless(_HAS_KVCR, "nvidia-kvcr wheel not installed")

_BASE_CONTROL_PORT = 25000

# Sentinel for "the helper's default", so a test can ask for no buffers at all.
_UNSET = object()
# Fabricated host address for the fake pools; nothing dereferences it.
_FAKE_BASE_PTR = 1 << 20


def _storage_config(
    tp_rank: int,
    tp_size: int,
    dp_rank: int = 0,
    dp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    **extra,
) -> HiCacheStorageConfig:
    """One scheduler's config.

    ``tp_rank``/``tp_size`` are attention-scoped whenever DP attention is on, so a
    DP=2/attn_tp=2 engine is four configs with ``tp_size=2``.
    """
    extra_config = {
        "local_dram_bytes": 1 << 20,
        "control_host": "127.0.0.1",
        "control_port": _BASE_CONTROL_PORT,
        "control_advertise_host": "127.0.0.1",
        "enable_remote_hint": True,
    }
    extra_config.update(extra)
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test-model",
        dp_rank=dp_rank,
        dp_size=dp_size,
        extra_config=extra_config,
    )


def _store(
    tp_rank: int,
    tp_size: int,
    dp_rank: int = 0,
    dp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    **extra,
) -> KVCRStore:
    """A KVCRStore with no mem_pool, so the core is never constructed."""
    return KVCRStore(
        _storage_config(
            tp_rank,
            tp_size,
            dp_rank,
            dp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            **extra,
        ),
        mem_pool=None,
    )


def _fake_pool(
    sizes: List[int],
    page_size: int = 64,
    buffers: object = _UNSET,
) -> SimpleNamespace:
    """A host pool shaped like the two accessors this backend reads.

    ``sizes`` is one page's component byte sizes, as ``get_page_buffer_meta``
    reports them; the pointers it returns are fabricated and only ever compared
    for count. ``buffers`` defaults to one registrable tensor.
    """

    def page_buffer_meta(indices):
        pages = max(1, len(indices) // page_size)
        return (
            [_FAKE_BASE_PTR + i for i in range(pages * len(sizes))],
            list(sizes) * pages,
        )

    return SimpleNamespace(
        page_size=page_size,
        get_page_buffer_meta=page_buffer_meta,
        get_hybrid_pool_buffer=lambda: (
            [torch.empty(64, dtype=torch.uint8)] if buffers is _UNSET else buffers
        ),
    )


def _hint_extra_info(endpoint: str) -> HiCacheStorageExtraInfo:
    """One kv.fetch hint, in the envelope the request path carries."""
    return HiCacheStorageExtraInfo(
        extra_info={
            ROUTER_HINT_KEY: build_envelope(
                source_control_endpoint=endpoint,
                block_hashes=[0x0123456789ABCDEF],
                message_id="test-envelope",
            )
        }
    )


class FakeEntry:
    """One completed block key, shaped like the core's ``OpEntryResult``.

    ``success`` is derived from ``status`` as the core derives it, so DROPPED and
    FAILED are both falsy and only the status tells them apart.
    """

    def __init__(self, status=OpEntryStatus.SUCCESS) -> None:
        self.status = status

    @property
    def success(self) -> bool:
        return self.status is OpEntryStatus.SUCCESS


def _raise_fault(*_args, **_kwargs):
    raise RuntimeError("kvcr core fault")


class _ExplodingKVCR:
    """A core whose every entry point raises.

    ``FakeKVCR`` reports failure through the normal result path, which is exactly
    the path an exception skips.
    """

    deposit = _raise_fault
    deliver = _raise_fault
    discard_hint = _raise_fault
    poll_completed = _raise_fault
    query = _raise_fault
    submit_hint = _raise_fault


class FakeKVCR:
    """Records deliver() calls and reports completions only when told to.

    Nothing completes on its own, so the interleaving between waiters is chosen by
    the test rather than by timing.
    """

    def __init__(self) -> None:
        self._pending: List[Tuple[int, Dict]] = []
        self._lock = threading.Lock()
        self.next_handle = 100

    def deliver(self, destinations, request_id=None) -> int:
        with self._lock:
            self.next_handle += 1
            return self.next_handle

    def deposit(self, descriptors) -> int:
        with self._lock:
            self.next_handle += 1
            return self.next_handle

    def finish(self, op_handle: int, keys: List[str]) -> None:
        with self._lock:
            self._pending.append((op_handle, {key: FakeEntry() for key in keys}))

    def poll_completed(self):
        with self._lock:
            drained = self._pending
            self._pending = []
            return drained


@_needs_kvcr
class RankAddressingTest(unittest.TestCase):
    """Which port each rank binds, and which port it dials for a hint.

    Regression for a wrong-shard bug at TP=2: both ranks dialed the single endpoint
    dynamo advertises, so rank 1 accepted rank 0's shard and reported a full hit.
    The only symptom was generated text differing from the same prefix computed
    locally. dynamo cannot fix it there -- it indexes by ``(worker_id, dp_rank)``
    and has no TP concept, so realigning is the consumer's job.
    """

    def test_the_agent_name_separates_every_rank_and_every_restart(self):
        """Peers key their remote-agent tables by name, so any collision makes
        one registration overwrite another's; the name carries the full rank
        coordinate plus a fresh id per start.
        """
        tp0, tp1 = _store(0, 2), _store(1, 2)
        dp1 = _store(0, 2, dp_rank=1, dp_size=2)
        restarted = _store(0, 2)

        self.assertIn("tp0", tp0._agent_name)
        self.assertIn("tp1", tp1._agent_name)
        names = [
            tp0._agent_name,
            tp1._agent_name,
            dp1._agent_name,
            restarted._agent_name,
        ]
        self.assertEqual(len(set(names)), len(names))

    def test_each_rank_binds_its_own_port_block(self):
        """DP rank r owns ``[base + r*attn_tp_size, +attn_tp_size)``.

        A contract with dynamo, which must stride by ``attn_tp_size``, not by 1. A
        collision fails in the background: the engine comes up and only peer
        fetches break.
        """
        ports = {
            (dp_rank, tp_rank): _store(tp_rank, 2, dp_rank, 2)._control_port()
            for dp_rank in range(2)
            for tp_rank in range(2)
        }

        self.assertEqual(
            ports,
            {
                (0, 0): _BASE_CONTROL_PORT,
                (0, 1): _BASE_CONTROL_PORT + 1,
                (1, 0): _BASE_CONTROL_PORT + 2,
                (1, 1): _BASE_CONTROL_PORT + 3,
            },
        )

    def test_the_dialed_source_port_follows_tp_rank_alone(self):
        """The router already picked the source DP rank; we add only our own
        within-DP offset.

        Our DP rank says nothing about which of the source's DP ranks holds the prefix.
        Adding it walks into another DP group's port block, and that fetch succeeds.
        """
        dialed = {
            (dp_rank, tp_rank): _store(tp_rank, 2, dp_rank, 2)
            ._parse_hint(_hint_extra_info("tcp://10.0.0.7:25000"))
            .source_control_endpoint
            for dp_rank in range(2)
            for tp_rank in range(2)
        }

        self.assertEqual(
            dialed,
            {
                (0, 0): "tcp://10.0.0.7:25000",
                (0, 1): "tcp://10.0.0.7:25001",
                (1, 0): "tcp://10.0.0.7:25000",
                (1, 1): "tcp://10.0.0.7:25001",
            },
        )

    def test_an_unparseable_endpoint_drops_the_hint(self):
        """A rank that cannot align must fetch nothing, not fetch wrongly:
        returning the endpoint verbatim puts rank 1 back on rank 0's shard.
        """
        store = _store(1, 2)

        self.assertIsNone(store._parse_hint(_hint_extra_info("tcp://10.0.0.7")))


@_needs_kvcr
class UnusableHostPoolTest(unittest.TestCase):
    """Host pool layouts the backend cannot run against must fail at startup."""

    def test_an_unusable_pool_refuses_to_start_the_backend(self):
        """Both directions address host pages by NIXL descriptor, into a local
        DRAM tier sized from the page layout. Warning and starting anyway lets a
        DeepSeek-V4-style per-layer pool name unregistered memory in every
        transfer, or reads downstream as a cold server while every peer serve
        silently has nothing to offer.

        Registration is where this has to fail, not the build: the core is built
        lazily on first use, long after the launch would have succeeded.
        """
        pools = {
            # Probe-able, so the local tier sizes fine, but the pool exposes no
            # tensor: nothing to register with NIXL.
            "register with NIXL": _fake_pool(
                sizes=[16, 16], buffers=[object(), object()]
            ),
            # Registrable buffer, so this gets past the NIXL gate and fails on
            # the page-layout probe alone.
            "positive number of bytes": _fake_pool(sizes=[], buffers=None),
        }

        for expected, pool in pools.items():
            with self.subTest(expected=expected):
                with self.assertRaises(RuntimeError) as raised:
                    _store(0, 1).register_mem_pool_host(pool)

                self.assertIn(expected, str(raised.exception))


@_needs_kvcr
class SidecarPoolTest(unittest.TestCase):
    """A hybrid stack's sidecar pools must be addressed as their own storage.

    A hybrid KV stack (DSA/MiniMax indexer, Mamba, SWA) sends one ``PoolTransfer``
    per pool, the sidecar one derived from the KV pool's indices and *hashes*.
    Two things follow, and both are silent-wrong-output if missed: the addresses
    must come from the sidecar's own host pool (resolving through the anchor
    would move KV bytes into KV pages, report success, and leave the sidecar
    untouched), and the block keys must carry the pool (sharing the KV page hash
    would make the second pool's deposit overwrite the first's blocks).

    A pool that never registered at all is still scored a miss, since nothing
    was written for it.
    """

    def _store_with_pools(self) -> KVCRStore:
        """A store with a KV pool and an indexer sidecar, and a fake core."""
        store = _store(0, 1)
        store.register_mem_pool_host(_fake_pool(sizes=[16, 16]))
        store.register_mem_host_pool_v2(_fake_pool(sizes=[8]), PoolName.INDEXER)
        store._kvcr = FakeKVCR()
        return store

    def _transfers(self, names=(PoolName.KV, PoolName.INDEXER)):
        """One transfer per pool, all carrying the KV page hashes."""
        return [
            PoolTransfer(
                name=name,
                keys=["p0", "p1"],
                host_indices=torch.arange(128, dtype=torch.int64),
                indices_from_pool=None if name == PoolName.KV else PoolName.KV,
            )
            for name in names
        ]

    def test_each_pool_is_addressed_through_its_own_host_pool(self):
        """The anchor forwards ``get_page_buffer_meta``, so a sidecar resolved
        through ``mem_pool_host`` is handed KV addresses -- and reports success
        after moving KV bytes into KV pages.
        """
        store = self._store_with_pools()
        store._drain_until = lambda handle, timeout_s=None: {}
        asked = []
        for name, layout in store._pool_layouts.items():
            inner = layout.host_pool.get_page_buffer_meta
            layout.host_pool.get_page_buffer_meta = (
                lambda indices, name=name, inner=inner: (
                    asked.append(name) or inner(indices)
                )
            )

        store.batch_set_v2(self._transfers())

        self.assertEqual(sorted(asked), [str(PoolName.INDEXER), str(PoolName.KV)])

    def test_two_pools_sharing_a_page_hash_get_distinct_block_keys(self):
        """A sidecar transfer carries the KV pool's hashes, so keys that did not
        carry the pool would collide and the second deposit would overwrite the
        first's blocks -- leaving one pool reading the other's bytes.
        """
        store = self._store_with_pools()
        deposited = []
        store._kvcr = SimpleNamespace(
            deposit=lambda descriptors: deposited.append(set(descriptors)) or 1
        )
        store._drain_until = lambda handle, timeout_s=None: {}

        store.batch_set_v2(self._transfers())

        kv_keys, sidecar_keys = deposited
        self.assertEqual(kv_keys & sidecar_keys, set())

    def test_an_unregistered_pool_is_a_miss_and_never_reaches_the_core(self):
        """Asserting the deliver never ran separates this guard from
        ``_fail_closed``, which would also produce all-False and is not the same fix.
        """
        store = self._store_with_pools()
        delivered = []

        def record_and_succeed(transfer, layout, request_id):
            delivered.append(transfer.name)
            return [True] * len(transfer.keys)

        store._deliver_transfer = record_and_succeed

        results = store.batch_get_v2(self._transfers((PoolName.KV, PoolName.SWA)))

        self.assertEqual(results[str(PoolName.SWA)], [False, False])
        self.assertEqual(delivered, [PoolName.KV])

    def test_an_unregistered_pool_makes_the_whole_prefix_unavailable(self):
        """``batch_exists_v2`` reports one prefix for the request and the
        controller issues gets for it across all pools, so reporting the KV pages
        held would promise pages this backend never wrote.
        """
        store = self._store_with_pools()
        store._kvcr = SimpleNamespace(
            query=lambda keys: [(QueryStatus.HIT, None)] * len(keys)
        )

        result = store.batch_exists_v2(
            ["p0", "p1"], self._transfers((PoolName.KV, PoolName.SWA))
        )

        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(result.restorable_prefix_pages, [])

    def test_a_hinted_prefix_is_available_for_sidecars_too(self):
        """A hint names page hashes, and a sidecar transfer carries the KV
        page's hashes -- so the source deposited that page's sidecar blocks
        under the same hash and they are pullable.

        Scoring sidecars on local residency alone made this return 0 for every
        sidecar: the prefix is one number across all pools, so nothing remote
        was ever fetched on a hybrid model and the whole P2P path was dead
        while KV-only stacks stayed green.
        """
        store = self._store_with_pools()
        store._kvcr = SimpleNamespace(
            query=lambda keys: [(QueryStatus.MISS, None)] * len(keys)
        )
        page_keys = ["0123456789abcdef" + "00" * 8, "fedcba9876543210" + "00" * 8]
        transfers = [
            PoolTransfer(
                name=name,
                keys=page_keys,
                host_indices=torch.arange(128, dtype=torch.int64),
                indices_from_pool=None if name == PoolName.KV else PoolName.KV,
            )
            for name in (PoolName.KV, PoolName.INDEXER)
        ]
        extra_info = HiCacheStorageExtraInfo(
            extra_info={
                ROUTER_HINT_KEY: build_envelope(
                    source_control_endpoint="tcp://10.0.0.7:25000",
                    block_hashes=list(page_keys),
                    message_id="hybrid-hint",
                )
            }
        )

        result = store.batch_exists_v2(page_keys, transfers, extra_info)

        self.assertEqual(result.kv_hit_pages, 2)
        self.assertEqual(result.extra_pool_hit_pages[str(PoolName.INDEXER)], 2)
        self.assertEqual(result.restorable_prefix_pages, [1, 2])

    def test_a_sidecar_short_of_the_kv_prefix_clamps_it(self):
        """``ALL_PAGES``: the sidecar is required for every page of the prefix, so
        a sidecar that holds only the first page caps the request at one.
        """
        store = self._store_with_pools()
        kv = store._pool_layouts[str(PoolName.KV)]
        indexer = store._pool_layouts[str(PoolName.INDEXER)]
        held = set(
            store._page_segment_keys(kv, "p0")
            + store._page_segment_keys(kv, "p1")
            + store._page_segment_keys(indexer, "p0")
        )
        store._kvcr = SimpleNamespace(
            query=lambda keys: [
                (QueryStatus.HIT if key in held else QueryStatus.MISS, None)
                for key in keys
            ]
        )

        result = store.batch_exists_v2(["p0", "p1"], self._transfers())

        self.assertEqual(result.kv_hit_pages, 1)
        self.assertEqual(result.extra_pool_hit_pages[str(PoolName.INDEXER)], 1)


@_needs_kvcr
class UnaddressableParallelismTest(unittest.TestCase):
    """Rank coordinates a KVCR block key cannot encode must fail at startup.

    A block key is ``sha256(token ids)#<segment>`` and a hint carries an endpoint
    plus page hashes, so nothing on the wire says which model slice produced the
    bytes; ``_rank_port_offset`` separates ``(dp, attn_cp, attn_tp)`` by port
    instead. Pipeline rank and head splitting have no such separation -- same port
    *and* same key, so the fetch lands another rank's bytes in pages the model
    attends over.
    """

    def _config(self, **overrides) -> HiCacheStorageConfig:
        config = _storage_config(0, 1)
        for field, value in overrides.items():
            setattr(config, field, value)
        return config

    def test_an_unencodable_rank_refuses_to_start_the_backend(self):
        for overrides, expected in (
            ({"pp_size": 2}, "pipeline parallelism"),
            ({"should_split_heads": True, "tp_lcm_size": 8}, "heterogeneous TP"),
        ):
            with self.subTest(expected=expected):
                with self.assertRaises(RuntimeError) as raised:
                    KVCRStore(self._config(**overrides), mem_pool=None)

                self.assertIn(expected, str(raised.exception))


@_needs_kvcr
class ConcurrentDrainTest(unittest.TestCase):
    """_drain_until and the scheduler tick racing for the same completion queue."""

    def setUp(self) -> None:
        self.store = _store(0, 1)
        self.core = FakeKVCR()
        self.store._kvcr = self.core

    def test_completion_drained_by_the_tick_still_reaches_its_waiter(self):
        """The case ``_completed_ops`` exists for: the scheduler tick routinely
        observes a get completing, and dropping it makes the waiter spin to its
        deadline and report a miss.
        """
        self.store._register_waiter(101)
        self.core.finish(101, ["seg-a"])

        self.store.tick()  # the scheduler thread drains it first
        result = self.store._drain_until(101, timeout_s=1.0)

        self.assertEqual(result, {"seg-a": True})


@_needs_kvcr
class CloseTest(unittest.TestCase):
    """Shutdown ordering between the two pollers and the KVCR core."""

    def test_a_poll_already_under_way_finishes_before_the_core_closes(self):
        """``kvcr.close()`` tears down the progress thread and the local tier,
        so closing under a live ``poll_completed()`` is a use-after-free
        reachable through NIXL. ``_poll_lock`` is what fences them; a close that
        did not take it would return while the poll was still inside the core.
        """
        entered = threading.Event()
        release = threading.Event()
        polled_to_completion = threading.Event()
        self.addCleanup(release.set)

        core = SimpleNamespace(closed=False)
        core.close = lambda: setattr(core, "closed", True)

        def slow_poll():
            entered.set()
            release.wait(timeout=5)
            polled_to_completion.set()
            return []

        core.poll_completed = slow_poll
        store = _store(0, 1)
        store._kvcr = core

        poller = threading.Thread(target=store.tick, daemon=True)
        poller.start()
        self.assertTrue(entered.wait(timeout=5))

        closer = threading.Thread(target=store.close, daemon=True)
        closer.start()
        # The close cannot have got in while the poll holds the lock.
        self.assertFalse(core.closed)

        release.set()
        closer.join(timeout=5)
        poller.join(timeout=5)

        self.assertTrue(polled_to_completion.is_set())
        self.assertTrue(core.closed)
        self.assertIsNone(store._kvcr)

    def test_a_poller_that_arrives_after_close_does_not_touch_the_core(self):
        """The mirror case: ``tick`` runs on the scheduler thread and ``close``
        on nobody's schedule, so a tick can read the core a moment before it is
        closed. Re-checking under the lock is what stops it polling a core whose
        progress thread is gone.
        """
        core = SimpleNamespace(closed=False, polls=0)
        core.close = lambda: setattr(core, "closed", True)

        def count_poll():
            core.polls += 1
            return []

        core.poll_completed = count_poll
        store = _store(0, 1)
        store._kvcr = core

        store.close()
        store._poll_once(core)  # a caller holding the stale reference

        self.assertEqual(core.polls, 0)


@_needs_kvcr
class RemoteFailureTest(unittest.TestCase):
    """A remote source that is slow, gone, or lying must degrade to recompute.

    All three are normal for a P2P cache. Raising out of ``batch_get_v2`` kills the
    prefetch thread HiCache never restarts, taking down *local* L3 for the life of
    the process; hanging stalls it just as permanently.
    """

    def setUp(self) -> None:
        self.store = _store(0, 1)
        self.store.register_mem_pool_host(_fake_pool(sizes=[16]))
        self.layout = self.store._pool_layouts[str(PoolName.KV)]

    def _transfer(self, keys: List[str]) -> SimpleNamespace:
        """A PoolTransfer-alike whose host descriptors always resolve."""
        return SimpleNamespace(name=PoolName.KV, keys=keys)

    def test_a_source_that_never_answers_reports_a_miss_rather_than_hanging(self):
        """``kvcr.abort()`` cancels nothing, so ``_drain_until``'s deadline is all
        that stands between a dead peer and a permanently wedged prefetch thread.
        """
        self.store._kvcr = FakeKVCR()  # finish() is never called
        self.store._host_descriptors = lambda transfer, layout: (
            {"seg-a": object()},
            [["seg-a"]],
        )

        started = time.monotonic()
        results = self.store._deliver_transfer(
            self._transfer(["page-a"]), self.layout, request_id="req-1"
        )
        elapsed = time.monotonic() - started

        self.assertEqual(results, [False])
        # Bounded by the configured get timeout, not by the caller giving up.
        self.assertLess(elapsed, self.store._config.get_timeout_s + 5.0)

    def test_a_hint_covering_nothing_leaves_the_prefix_at_zero(self):
        """``batch_exists_v2`` is the gate: the controller allocates host memory
        for the prefix reported here, released only after a full deliver round trip.
        """
        self.store._kvcr = FakeKVCR()
        self.store._locally_resident = lambda segment_keys: False
        extra_info = _hint_extra_info("tcp://10.0.0.7:25000")


        result = self.store.batch_exists_v2(
            ["hash-the-hint-does-not-cover"], extra_info=extra_info
        )

        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(self.store.stats()["exists_hint_covered_nothing"], 1)


@_needs_kvcr
class RaisingCoreTest(unittest.TestCase):
    """A core that raises must degrade to a miss, never out of the store.

    ``RemoteFailureTest`` covers a *peer* failing, which the core reports through
    its normal result path. This is the core itself raising, which skips that path
    entirely, and nothing above this backend catches it: each HiCache storage loop
    catches only ``Empty``, so it ends on the first exception, unsupervised, and
    takes the only ``append_host_mem_release`` and ``ack_backup_queue`` producers
    with it.

    What turns these red: removing a ``@_fail_closed`` decorator.
    """

    def setUp(self) -> None:
        self.store = _store(0, 1)
        self.store.register_mem_pool_host(_fake_pool(sizes=[16]))
        self.store._kvcr = _ExplodingKVCR()
        self.store._host_descriptors = lambda transfer, layout: (
            {"seg-a": object()},
            [["seg-a"]],
        )

    def _transfer(self, keys: List[str]) -> SimpleNamespace:
        return SimpleNamespace(name=PoolName.KV, keys=keys)

    def test_the_transfer_paths_report_every_page_unserved(self):
        set_results = self.store.batch_set_v2([self._transfer(["page-a", "page-b"])])
        get_results = self.store.batch_get_v2([self._transfer(["page-a"])])

        self.assertEqual(set_results, {str(PoolName.KV): [False, False]})
        self.assertEqual(get_results, {str(PoolName.KV): [False]})
        self.assertEqual(self.store.stats()["faults_batch_set_v2"], 1)
        self.assertEqual(self.store.stats()["faults_batch_get_v2"], 1)

    def test_a_raising_query_reports_no_available_prefix(self):
        """``batch_exists_v2`` gates the whole path, and any answer but zero makes
        the controller allocate host memory it releases only after a deliver round trip.
        """
        result = self.store.batch_exists_v2(["page-a", "page-b"])

        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(self.store.stats()["faults_batch_exists_v2"], 1)


if __name__ == "__main__":
    unittest.main()

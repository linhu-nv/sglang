"""Restore drivers for PD decode: the three ways a backend's load_back lands.

``BLOCKING`` (FlexKV MP) finishes inside ``init_load_back``, ``MERGED_EVENT``
(HiCache) batches behind one DMA event, and ``PER_REQUEST_POLL`` (FlexKV async
MP) completes out of band. The state machine must not poll an event for the
first, and must not serialize the last.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    DecodeRestoreDriver,
    RestoreCompletion,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_decode_req(rid: str, needs_restore: bool) -> SimpleNamespace:
    prefix_match = DecodePrefixMatch(
        prefix_indices=MagicMock(__len__=lambda self: 4),
        l2_host_hit_length=8 if needs_restore else 0,
        l3_storage_hit_length=0,
        last_device_node=object(),
    )
    return SimpleNamespace(
        req=SimpleNamespace(rid=rid),
        prefix_match=prefix_match,
        hicache_restore_status=HiCacheRestoreResult.PENDING,
        hicache_restored_node=None,
        hicache_load_consumer_index=-1,
    )


class _Driver(DecodeHiCacheTransferMixin):
    """Bare host for the mixin: only ``tree_cache`` is consulted."""

    def __init__(self, tree_cache):
        self.tree_cache = tree_cache


class TestDecodeBlockingRestore(CustomTestCase):
    def _blocking_driver(self):
        tree_cache = MagicMock()
        tree_cache.decode_restore_driver = DecodeRestoreDriver.BLOCKING
        driver = _Driver(tree_cache)
        driver._try_hicache_queue_load_back = MagicMock(return_value=True)
        return driver

    def test_blocking_backend_flips_to_ready_without_polling_events(self):
        driver = self._blocking_driver()
        dr = _make_decode_req("sync-1", needs_restore=True)

        driver._process_hicache_local_restores([dr])

        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.READY)
        driver._try_hicache_queue_load_back.assert_called_once_with(dr)
        # A blocking backend has no load-back event and no merged-DMA kick.
        driver.tree_cache.is_load_back_event_done.assert_not_called()
        driver.tree_cache.ready_to_load_host_cache.assert_not_called()

    def test_blocking_backend_restores_one_request_per_tick(self):
        # init_load_back blocks the scheduler thread, so chaining several
        # restores in one tick would stall the running decode batch.
        driver = self._blocking_driver()
        reqs = [_make_decode_req(f"sync-{i}", needs_restore=True) for i in range(3)]

        driver._process_hicache_local_restores(reqs)

        self.assertEqual(driver._try_hicache_queue_load_back.call_count, 1)
        self.assertEqual(reqs[0].hicache_restore_status, HiCacheRestoreResult.READY)
        for dr in reqs[1:]:
            self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.PENDING)

        driver._process_hicache_local_restores(reqs)
        self.assertEqual(reqs[1].hicache_restore_status, HiCacheRestoreResult.READY)

    def test_requests_without_restore_work_flip_to_ready(self):
        driver = self._blocking_driver()
        no_match = _make_decode_req("none", needs_restore=True)
        no_match.prefix_match = None
        fully_on_device = _make_decode_req("device", needs_restore=False)

        driver._process_hicache_local_restores([no_match, fully_on_device])

        self.assertEqual(no_match.hicache_restore_status, HiCacheRestoreResult.READY)
        self.assertEqual(
            fully_on_device.hicache_restore_status, HiCacheRestoreResult.READY
        )
        driver._try_hicache_queue_load_back.assert_not_called()

    def test_merged_event_backend_uses_event_polling(self):
        tree_cache = MagicMock()
        tree_cache.decode_restore_driver = DecodeRestoreDriver.MERGED_EVENT
        tree_cache.is_load_back_event_done.return_value = True
        tree_cache.ready_to_load_host_cache.return_value = 2
        counter = SimpleNamespace(producer_index=0, num_counters=4)
        tree_cache.cache_controller.layer_done_counter = counter

        driver = _Driver(tree_cache)
        driver._try_hicache_queue_load_back = MagicMock(return_value=True)
        reqs = [_make_decode_req(f"async-{i}", needs_restore=True) for i in range(2)]

        driver._process_hicache_local_restores(reqs)

        # Merged-event path batches every queued request behind one DMA.
        self.assertEqual(driver._try_hicache_queue_load_back.call_count, 2)
        tree_cache.ready_to_load_host_cache.assert_called_once()
        for dr in reqs:
            self.assertEqual(dr.hicache_load_consumer_index, 2)
            self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.PENDING)

    def test_blocking_backend_releases_lookup_task_on_abort(self):
        # FlexKV holds a lookup task from match_prefix even when no prefetch was
        # registered; abort has to hand it back or the task leaks.
        driver = self._blocking_driver()
        dr = _make_decode_req("abort", needs_restore=True)
        self.assertFalse(dr.prefix_match.prefetch_registered)

        driver._clean_hicache_prefetch_resources(dr)

        driver.tree_cache.release_aborted_request.assert_called_once_with("abort")


class TestDecodePolledRestore(CustomTestCase):
    def _polled_driver(self, completions=None):
        tree_cache = MagicMock()
        tree_cache.decode_restore_driver = DecodeRestoreDriver.PER_REQUEST_POLL
        tree_cache.poll_completed_restores.return_value = completions or {}
        driver = _Driver(tree_cache)
        driver._try_hicache_queue_load_back = MagicMock(return_value=True)
        return driver

    def _launch(self, driver, reqs):
        """Run one tick and mark the launched reqs the way the real
        ``_try_hicache_queue_load_back`` would."""
        driver._process_hicache_local_restores(reqs)
        for dr in reqs:
            if dr.hicache_restore_status == HiCacheRestoreResult.PENDING:
                dr.hicache_restored_node = object()

    def test_launches_every_pending_request_in_one_tick(self):
        # The whole point of the async path: no one-per-tick throttle, because
        # the launch does not block the scheduler thread.
        driver = self._polled_driver()
        reqs = [_make_decode_req(f"poll-{i}", needs_restore=True) for i in range(3)]

        driver._process_hicache_local_restores(reqs)

        self.assertEqual(driver._try_hicache_queue_load_back.call_count, 3)
        for dr in reqs:
            self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.PENDING)
        # No shared event slot and no merged kick on this path.
        driver.tree_cache.ready_to_load_host_cache.assert_not_called()

    def test_does_not_relaunch_an_inflight_request(self):
        driver = self._polled_driver()
        reqs = [_make_decode_req("poll-1", needs_restore=True)]
        self._launch(driver, reqs)
        driver._try_hicache_queue_load_back.reset_mock()

        driver._process_hicache_local_restores(reqs)

        driver._try_hicache_queue_load_back.assert_not_called()

    def test_completion_flips_status_and_retargets_the_node(self):
        driver = self._polled_driver()
        reqs = [_make_decode_req("poll-1", needs_restore=True)]
        self._launch(driver, reqs)
        launched_node = reqs[0].hicache_restored_node

        # The backend deferred insertion to completion, so it hands back the
        # node it created and moved the lock ref onto.
        inserted_node = object()
        driver.tree_cache.poll_completed_restores.return_value = {
            "poll-1": RestoreCompletion(succeeded=True, node=inserted_node)
        }
        driver._process_hicache_local_restores(reqs)

        self.assertEqual(reqs[0].hicache_restore_status, HiCacheRestoreResult.READY)
        self.assertIs(reqs[0].hicache_restored_node, inserted_node)
        self.assertIsNot(reqs[0].hicache_restored_node, launched_node)

    def test_completion_without_a_node_keeps_the_launch_handle(self):
        driver = self._polled_driver()
        reqs = [_make_decode_req("poll-1", needs_restore=True)]
        self._launch(driver, reqs)
        launched_node = reqs[0].hicache_restored_node

        driver.tree_cache.poll_completed_restores.return_value = {
            "poll-1": RestoreCompletion(succeeded=True)
        }
        driver._process_hicache_local_restores(reqs)

        self.assertEqual(reqs[0].hicache_restore_status, HiCacheRestoreResult.READY)
        self.assertIs(reqs[0].hicache_restored_node, launched_node)

    def test_failed_completion_flips_to_failed(self):
        driver = self._polled_driver()
        reqs = [_make_decode_req("poll-1", needs_restore=True)]
        self._launch(driver, reqs)

        driver.tree_cache.poll_completed_restores.return_value = {
            "poll-1": RestoreCompletion(succeeded=False)
        }
        driver._process_hicache_local_restores(reqs)

        self.assertEqual(reqs[0].hicache_restore_status, HiCacheRestoreResult.FAILED)

    def test_completion_for_a_departed_request_is_ignored(self):
        # The backend reports each rid exactly once; a rid whose request already
        # left the queue was settled by _clean_hicache_prefetch_resources.
        driver = self._polled_driver(
            completions={"gone": RestoreCompletion(succeeded=True)}
        )
        reqs = [_make_decode_req("poll-1", needs_restore=True)]

        driver._process_hicache_local_restores(reqs)

        self.assertEqual(reqs[0].hicache_restore_status, HiCacheRestoreResult.PENDING)

    def test_completion_lands_on_the_same_tick_it_is_polled(self):
        # Polling happens before collection, so a request that finished does not
        # wait a whole extra tick to reach READY.
        driver = self._polled_driver()
        reqs = [_make_decode_req("poll-1", needs_restore=True)]
        self._launch(driver, reqs)

        driver.tree_cache.poll_completed_restores.return_value = {
            "poll-1": RestoreCompletion(succeeded=True)
        }
        driver._process_hicache_local_restores(reqs)

        self.assertEqual(reqs[0].hicache_restore_status, HiCacheRestoreResult.READY)
        # Reaped, so it must not be relaunched on this tick.
        self.assertEqual(driver._try_hicache_queue_load_back.call_count, 1)

    def test_polled_backend_releases_inflight_load_on_abort(self):
        # An aborted request's slots are about to be freed while FlexKV may
        # still be writing them from another process; the backend has to be told.
        driver = self._polled_driver()
        dr = _make_decode_req("abort", needs_restore=True)

        driver._clean_hicache_prefetch_resources(dr)

        driver.tree_cache.release_aborted_request.assert_called_once_with("abort")


if __name__ == "__main__":
    unittest.main()

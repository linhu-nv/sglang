"""Late binding for attention DP: held requests go to the rank that frees up first, never past its budget."""

import unittest

from sglang.srt.managers.dp_pull_queue import DPPullQueue
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _snap(rank, ts, tokens, hit=0.0):
    return LoadSnapshot(dp_rank=rank, timestamp=ts, num_waiting_uncached_tokens=tokens, cache_hit_rate=hit)


class TestDPPullQueue(CustomTestCase):
    def setUp(self):
        self.clock = _Clock()
        self.q = DPPullQueue(num_ranks=2, budget_tokens=8192, clock=self.clock)

    def test_holds_until_a_rank_has_room(self):
        self.q.update([_snap(0, 99.0, 20000), _snap(1, 99.0, 9000)])
        self.q.push("a", prompt_tokens=1000)
        self.assertEqual(self.q.pop_ready([0, 1]), [])
        self.assertEqual(len(self.q), 1)
        self.q.update([_snap(1, 100.5, 0)])
        self.assertEqual(self.q.pop_ready([0, 1]), [("a", 1)])

    def test_sent_work_counts_until_a_later_snapshot(self):
        self.q.update([_snap(0, 99.0, 0), _snap(1, 99.0, 0)])
        for name in "abc":
            self.q.push(name, prompt_tokens=6000)
        # Each rank takes requests while its queued work is below the budget.
        out = self.q.pop_ready([0, 1])
        self.assertEqual([r for _, r in out], [0, 1, 0])
        self.assertEqual(self.q.load(0), 12000)
        self.q.push("d", prompt_tokens=100)
        self.assertEqual(self.q.pop_ready([0, 1]), [("d", 1)])
        # A snapshot taken after the sends replaces them.
        self.clock.t = 101.0
        self.q.update([_snap(0, 101.0, 500)])
        self.assertEqual(self.q.load(0), 500)

    def test_sends_just_before_a_snapshot_stay_counted(self):
        self.q.update([_snap(0, 99.0, 0), _snap(1, 99.0, 9000)])
        self.q.push("a", prompt_tokens=5000)
        self.q.pop_ready([0, 1])
        self.q.update([_snap(0, 100.01, 0)])   # may predate the scheduler receiving "a"
        self.assertEqual(self.q.load(0), 5000)

    def test_estimate_uses_hit_rate_and_inactive_ranks_are_skipped(self):
        self.q.update([_snap(0, 99.0, 0, hit=0.9), _snap(1, 99.0, 0)])
        self.q.push("a", prompt_tokens=10000)
        self.assertEqual(self.q.pop_ready([0]), [("a", 0)])
        self.assertEqual(self.q.load(0), 1000)
        self.assertEqual(self.q.load(1), 0)

    def test_stale_snapshot_ignored_and_abort_removes(self):
        self.q.update([_snap(0, 99.0, 9000), _snap(1, 99.0, 9000)])
        self.q.update([_snap(0, 98.0, 0)])
        self.assertEqual(self.q.load(0), 9000)
        self.q.push("a", 10)
        self.q.push("b", 10)
        self.assertEqual(self.q.remove(lambda r: r == "a"), ["a"])
        self.assertEqual(len(self.q), 1)


if __name__ == "__main__":
    unittest.main()

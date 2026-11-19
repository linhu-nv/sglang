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


def _snap(rank, ts, tokens, hit=0.0, waiting=0):
    return LoadSnapshot(dp_rank=rank, timestamp=ts, num_waiting_uncached_tokens=tokens, cache_hit_rate=hit,
                        num_waiting_reqs=waiting)


class TestDPPullQueue(CustomTestCase):
    def setUp(self):
        self.clock = _Clock()
        self.q = DPPullQueue(num_ranks=2, budget_tokens=8192, max_sent=4, clock=self.clock)

    def test_holds_until_a_rank_reports_room(self):
        self.q.update([_snap(0, 99.0, 20000), _snap(1, 99.0, 9000)])
        self.q.push("a", prompt_tokens=1000)
        self.assertEqual(self.q.pop_ready([0, 1]), [])
        self.assertEqual(len(self.q), 1)
        self.q.update([_snap(1, 100.5, 0)])
        self.assertEqual(self.q.pop_ready([0, 1]), [("a", 1)])

    def test_at_most_max_sent_per_rank_between_snapshots(self):
        self.q.update([_snap(0, 99.0, 0), _snap(1, 99.0, 0)])
        for i in range(10):
            self.q.push(i, prompt_tokens=500)
        out = self.q.pop_ready([0, 1])
        self.assertEqual(sorted(r for _, r in out), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(len(self.q), 2)
        self.clock.t = 101.0
        self.q.update([_snap(0, 101.0, 2000)])   # rank 0 took them in; its next step has room
        self.assertEqual([r for _, r in self.q.pop_ready([0, 1])], [0, 0])

    def test_big_request_steers_the_next_ones_elsewhere(self):
        self.q.update([_snap(0, 99.0, 0, hit=0.5), _snap(1, 99.0, 0, hit=0.5)])
        self.q.push("big", prompt_tokens=100000)
        self.q.push("s1", prompt_tokens=1000)
        self.q.push("s2", prompt_tokens=1000)
        self.assertEqual(self.q.pop_ready([0, 1]), [("big", 0), ("s1", 1), ("s2", 1)])
        self.assertEqual(self.q.load(0), 50000)

    def test_sends_just_before_a_snapshot_stay_counted(self):
        self.q.update([_snap(0, 99.0, 0), _snap(1, 99.0, 9000)])
        self.q.push("a", prompt_tokens=5000)
        self.q.pop_ready([0, 1])
        self.q.update([_snap(0, 100.01, 0)])   # may predate the scheduler receiving "a"
        self.assertEqual(self.q.load(0), 5000)

    def test_inactive_ranks_skipped_stale_snapshot_ignored_abort_removes(self):
        self.q.update([_snap(0, 99.0, 0), _snap(1, 99.0, 0)])
        self.q.update([_snap(0, 98.0, 9000)])
        self.assertEqual(self.q.load(0), 0)
        self.q.push("a", 10)
        self.q.push("b", 10)
        self.assertEqual(self.q.remove(lambda r: r == "a"), ["a"])
        self.assertEqual(self.q.pop_ready([1]), [("b", 1)])

    def test_waiting_cap_counts_reported_and_sent(self):
        q = DPPullQueue(num_ranks=2, budget_tokens=10**9, max_sent=4, max_waiting=2, clock=self.clock)
        q.update([_snap(0, 99.0, 500000, waiting=2), _snap(1, 99.0, 0, waiting=1)])
        for i in range(3):
            q.push(i, prompt_tokens=1000)
        self.assertEqual(q.pop_ready([0, 1]), [(0, 1)])
        self.assertEqual(len(q), 2)


if __name__ == "__main__":
    unittest.main()

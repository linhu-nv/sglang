"""Adaptive DP/CP step plans are a pure function of the gathered rows."""

import unittest

from sglang.srt.managers.scheduler_components import adaptive_cp_policy as pol
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CFG = pol.PolicyConfig()
MODEL = pol.StepModel()


def _row(dp=(), lane=(), free=10**7, rows=64, window=-1, running=0):
    dp = list(dp)
    return pol.encode_row(
        dp=dp[: pol.K_DP],
        tail_tokens=sum(x[0] for x in dp[pol.K_DP :]),
        tail_count=len(dp[pol.K_DP :]),
        free_tokens=free,
        free_rows=rows,
        lane=lane,
        window_free=window,
        running=running,
    )


def _lane(cp_id, length, done=0, wait_ms=0, started=False, alive=True):
    flags = (pol.FLAG_STARTED if started else 0) | (pol.FLAG_ALIVE if alive else 0)
    return (cp_id, length, done, wait_ms, flags)


def _plan(rows, cfg=CFG):
    return pol.decide(pol.decode_rows(rows), cfg, MODEL)


class TestRowCodec(CustomTestCase):
    def test_roundtrip(self):
        row = pol.encode_row(
            dp=[(100, 200), (300, 400)],
            tail_tokens=5,
            tail_count=1,
            free_tokens=77,
            free_rows=3,
            lane=[_lane(9, 50000, 1000, 1500, started=True)],
            arrival_rate=1.25,
            arrival_tokens=2000,
            arrival_len=60000,
        )
        self.assertEqual(len(row), pol.ROW_WIDTH)
        (view,) = pol.decode_rows([row])
        self.assertEqual(view.dp, [(100, 200), (300, 400)])
        self.assertEqual((view.free_tokens, view.free_rows), (77, 3))
        self.assertEqual(view.arrival_rate, 1.25)
        (entry,) = view.lane
        self.assertEqual(
            (entry.owner, entry.cp_id, entry.length, entry.done), (0, 9, 50000, 1000)
        )
        self.assertTrue(entry.started and entry.alive)

    def test_caps(self):
        row = _row(dp=[(1, 1)] * 20, lane=[_lane(i, 20000) for i in range(9)])
        (view,) = pol.decode_rows([row])
        self.assertEqual(len(view.dp), pol.K_DP)
        self.assertEqual(view.tail_count, 20 - pol.K_DP)
        self.assertEqual(len(view.lane), pol.K_LANE)


class TestDecide(CustomTestCase):
    def test_idle_ranks_open_a_cp_step(self):
        rows = [_row(lane=[_lane(1, 100000)])] + [_row() for _ in range(7)]
        plan = _plan(rows)
        self.assertTrue(plan.is_cp)
        (item,) = plan.items
        self.assertEqual(
            (item.owner, item.cp_id, item.start, item.take), (0, 1, 0, CFG.cp_chunk)
        )
        self.assertTrue(item.first and not item.last)

    def test_busy_ranks_keep_dp_then_release(self):
        busy = [(8192, 80000)] * 6
        rows = [_row(dp=busy, lane=[_lane(1, 20000, wait_ms=100)])] + [
            _row(dp=busy) for _ in range(7)
        ]
        plan = _plan(rows, pol.PolicyConfig(alpha=2.0))
        self.assertFalse(plan.is_cp)
        self.assertEqual(plan.release, [])
        rows[0] = _row(dp=busy, lane=[_lane(1, 20000, wait_ms=5000)])
        plan = _plan(rows, pol.PolicyConfig(alpha=2.0))
        self.assertEqual(plan.release, [(0, 1)])

    def test_started_request_continues(self):
        busy = [(8192, 80000)] * 6
        rows = [_row(dp=busy, lane=[_lane(1, 100000, done=65536, started=True)])]
        rows += [_row(dp=busy) for _ in range(7)]
        plan = _plan(rows, pol.PolicyConfig(alpha=100.0))
        (item,) = plan.items
        self.assertEqual((item.start, item.take), (65536, 100000 - 65536))
        self.assertTrue(item.last)

    def test_cp_burst_yields_one_dp_step(self):
        """A long started request must not starve DP work: after max_cp_burst CP
        steps one DP step runs, then the request continues."""
        rows = [_row(lane=[_lane(1, 400000, done=65536, started=True)])]
        rows += [_row(dp=[(2000, 30000)]) for _ in range(7)]
        views = pol.decode_rows(rows)
        self.assertTrue(
            pol.decide(views, CFG, MODEL, cp_streak=CFG.max_cp_burst - 1).is_cp
        )
        self.assertEqual(
            pol.decide(views, CFG, MODEL, cp_streak=CFG.max_cp_burst).reason, "yield"
        )
        idle = pol.decode_rows([rows[0]] + [_row() for _ in range(7)])
        self.assertTrue(pol.decide(idle, CFG, MODEL, cp_streak=CFG.max_cp_burst).is_cp)
        decoding = pol.decode_rows(
            [rows[0]] + [_row(running=3)] + [_row() for _ in range(6)]
        )
        self.assertEqual(pol.decide(decoding, CFG, MODEL, cp_streak=1).reason, "yield")

    def test_aborted_started_request_is_dropped_everywhere(self):
        rows = [_row(lane=[_lane(3, 100000, done=65536, started=True, alive=False)])]
        rows += [_row() for _ in range(7)]
        plan = _plan(rows)
        self.assertEqual(plan.drop, [(0, 3)])
        self.assertFalse(plan.is_cp)

    def test_memory_gates_cp(self):
        rows = [_row(lane=[_lane(1, 100000)])] + [_row() for _ in range(6)]
        rows.append(_row(free=1000))
        plan = _plan(rows)
        self.assertFalse(plan.is_cp)

    def test_window_pool_bounds_the_chunk(self):
        """A CP chunk's transient sliding-window KV must fit every rank's window pool."""
        rows = [_row(lane=[_lane(1, 100000)], window=40000)] + [
            _row(window=60000) for _ in range(7)
        ]
        (item,) = _plan(rows).items
        self.assertEqual(item.take, (40000 - CFG.window_margin) // 256 * 256)
        rows[3] = _row(window=CFG.min_cp_chunk)
        self.assertFalse(_plan(rows).is_cp)

    def test_batches_owners_oldest_first(self):
        rows = [_row(lane=[_lane(1, 30000, wait_ms=10)])]
        rows += [_row(lane=[_lane(1, 30000, wait_ms=500)])]
        rows += [_row() for _ in range(6)]
        plan = _plan(rows, pol.PolicyConfig(force="cp"))
        self.assertEqual([it.owner for it in plan.items], [1, 0])
        self.assertEqual(sum(it.take for it in plan.items), 60000)

    def test_only_the_last_item_is_cut(self):
        """The scheduler marks the one cut item as the batch's chunked request, so
        its result never commits even when the next chunk is built first."""
        rows = [_row(lane=[_lane(1, 30000, wait_ms=900)])]
        rows += [_row(lane=[_lane(1, 50000, wait_ms=800)])]
        rows += [_row(lane=[_lane(1, 20000, wait_ms=700)])]
        rows += [_row() for _ in range(5)]
        items = _plan(rows, pol.PolicyConfig(force="cp")).items
        self.assertEqual([it.last for it in items], [True, False])
        self.assertEqual(sum(it.take for it in items), CFG.cp_chunk)

    def test_eta_release_of_a_request_deep_in_the_lane(self):
        """A lane request behind a lot of CP work goes to DP when its idle owner
        would finish it first; the head of the lane keeps its CP step."""
        rows = [_row(lane=[_lane(1, 200000, wait_ms=9000)])]
        rows += [_row(lane=[_lane(1, 200000, wait_ms=8000)])]
        rows += [_row(lane=[_lane(1, 20000, wait_ms=7000)])]
        rows += [_row() for _ in range(5)]
        plan = _plan(rows)
        self.assertTrue(plan.is_cp)
        self.assertEqual(plan.items[0].owner, 0)
        self.assertIn((2, 1), plan.release)
        no_eta = _plan(rows, pol.PolicyConfig(release_eta=False))
        self.assertNotIn((2, 1), no_eta.release)

    def test_force_dp_releases(self):
        rows = [_row(lane=[_lane(1, 30000)])] + [_row() for _ in range(7)]
        plan = _plan(rows, pol.PolicyConfig(force="dp"))
        self.assertEqual(plan.release, [(0, 1)])

    def test_plan_is_deterministic(self):
        rows = [
            _row(
                dp=[(3000, 50000)] * (r % 3), lane=[_lane(r + 1, 40000 + r, wait_ms=r)]
            )
            for r in range(8)
        ]
        plans = {repr(_plan([list(x) for x in rows])) for _ in range(5)}
        self.assertEqual(len(plans), 1)


if __name__ == "__main__":
    unittest.main()


class TestDeadlineRule(CustomTestCase):
    CFG = pol.PolicyConfig(rule="deadline", deadline_s=20.0, light_tokens=8192)

    def test_hold_while_busy(self):
        busy = [(30000, 40000)]
        rows = [_row(dp=busy, lane=[_lane(1, 60000, wait_ms=1000)])] + [_row(dp=busy)] * 7
        plan = _plan(rows, self.CFG)
        self.assertFalse(plan.is_cp)
        self.assertEqual((plan.release, plan.reason), ([], "hold"))

    def test_cp_when_deadline_near(self):
        busy = [(30000, 40000)]
        rows = [_row(dp=busy, lane=[_lane(1, 60000, wait_ms=19000)])] + [_row(dp=busy)] * 7
        plan = _plan(rows, self.CFG)
        self.assertTrue(plan.is_cp)
        self.assertEqual(plan.reason, "deadline")

    def test_cp_when_light(self):
        rows = [_row(dp=[(500, 900)], lane=[_lane(1, 60000, wait_ms=0)])] + [_row()] * 7
        plan = _plan(rows, self.CFG)
        self.assertTrue(plan.is_cp)
        self.assertEqual(plan.reason, "light")


class TestWarmItems(CustomTestCase):
    def test_warm_first_chunk_is_fresh(self):
        # A warm lane request starts at its cached prefix; its first chunk is fresh.
        rows = [_row(lane=[_lane(1, 60000, done=20480, wait_ms=0)])] + [_row()] * 7
        plan = _plan(rows, pol.PolicyConfig(force="cp"))
        (item,) = plan.items
        self.assertEqual((item.start, item.fresh, item.first), (20480, True, True))

    def test_started_chunk_not_fresh(self):
        lane = [_lane(1, 200000, done=65536, wait_ms=0, started=True)]
        rows = [_row(lane=lane)] + [_row()] * 7
        (item,) = _plan(rows).items
        self.assertEqual((item.start, item.fresh, item.first), (65536, False, False))

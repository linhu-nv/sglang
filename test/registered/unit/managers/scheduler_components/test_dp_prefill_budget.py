"""Attention-DP prefill chunks shrink as the prefix grows so each chunk keeps the time of a base chunk."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from sglang.srt.managers.scheduler_components.dp_prefill_budget import (
    DPPrefillBudget,
    PrefillCostModel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _budget(a=1e-9, b=5e-5, base=8192, page=256, cap=32768, scale=1.0):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"a": a, "b": b, "c": 0.01}, f)
    env = {"SGLANG_DP_PREFILL_BUDGET": f.name, "SGLANG_DP_PREFILL_BUDGET_CAP": str(cap),
           "SGLANG_DP_PREFILL_BUDGET_SCALE": str(scale)}
    try:
        with patch.dict(os.environ, env):
            return DPPrefillBudget.from_env(base_chunk=base, page_size=page)
    finally:
        os.unlink(f.name)


class TestDPPrefillBudget(CustomTestCase):
    def test_disabled_without_model(self):
        with patch.dict(os.environ, {"SGLANG_DP_PREFILL_BUDGET": ""}):
            self.assertIsNone(DPPrefillBudget.from_env(base_chunk=8192, page_size=256))

    def test_empty_prefix_keeps_base_chunk(self):
        self.assertEqual(_budget().chunk_for(0), 8192)

    def test_chunk_shrinks_with_prefix_and_stays_within_budget(self):
        b = _budget()
        sizes = [b.chunk_for(h) for h in (0, 32768, 131072, 262144)]
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        self.assertLess(sizes[-1], sizes[0])
        for h, x in zip((0, 32768, 131072, 262144), sizes):
            self.assertLessEqual(b.model.chunk_cost(h, x), b.target * (1 + 1e-9))
            self.assertEqual(x % 256, 0)

    def test_cap_and_page_floor(self):
        self.assertEqual(_budget(scale=10.0, cap=16384).chunk_for(0), 16384)
        self.assertEqual(_budget(a=1e-3).chunk_for(10**9), 256)

    def test_rejects_degenerate_model(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"a": 0, "b": 0}, f)
        with self.assertRaises(ValueError):
            PrefillCostModel.from_file(f.name)
        os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()

"""Time-budgeted prefill chunks for attention-DP prefill.

Attention-DP ranks run in lockstep (MoE all-to-all), so a step lasts as long as
its slowest rank. A rank deep into a long prompt spends more time per token, so
a fixed chunk stretches every step. This sizes the next chunk of a chunked
request so its predicted time stays within the time of a base chunk on an empty
prefix; every rank uses the same target, so the steps line up.

Cost model: f(l) = a*l^2 + b*l + c, the time to prefill l tokens from scratch;
a chunk of x tokens after a prefix of L costs f(L + x) - f(L). Coefficients come
from the JSON file in SGLANG_DP_PREFILL_BUDGET (any time unit).
"""

from __future__ import annotations

import json
import logging
import math
from typing import Optional

import msgspec

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class PrefillCostModel(msgspec.Struct, frozen=True):
    a: float
    b: float
    c: float = 0.0

    @classmethod
    def from_file(cls, path: str) -> "PrefillCostModel":
        with open(path) as f:
            d = json.load(f)
        model = cls(a=float(d["a"]), b=float(d["b"]), c=float(d.get("c", 0.0)))
        if model.a < 0 or model.b < 0 or (model.a == 0 and model.b == 0):
            raise ValueError(f"prefill cost model {path}: need a >= 0, b >= 0, not both 0")
        return model

    def chunk_cost(self, history: float, tokens: float) -> float:
        return self.a * tokens * (2 * history + tokens) + self.b * tokens

    def tokens_within(self, history: float, budget: float) -> float:
        """Largest x with chunk_cost(history, x) <= budget."""
        if budget <= 0:
            return 0.0
        lin = 2 * self.a * history + self.b
        if self.a == 0:
            return budget / lin
        return (-lin + math.sqrt(lin * lin + 4 * self.a * budget)) / (2 * self.a)


class DPPrefillBudget(msgspec.Struct, frozen=True):
    model: PrefillCostModel
    target: float
    page_size: int
    cap: int

    @classmethod
    def from_env(cls, *, base_chunk: Optional[int], page_size: int) -> Optional["DPPrefillBudget"]:
        path = envs.SGLANG_DP_PREFILL_BUDGET.get()
        if not path or not base_chunk:
            return None
        model = PrefillCostModel.from_file(path)
        budget = cls(
            model=model,
            target=envs.SGLANG_DP_PREFILL_BUDGET_SCALE.get() * model.chunk_cost(0, base_chunk),
            page_size=max(1, page_size),
            cap=max(page_size, envs.SGLANG_DP_PREFILL_BUDGET_CAP.get()),
        )
        logger.info(
            f"[DP prefill budget] a={model.a:.3e} b={model.b:.3e} base_chunk={base_chunk} "
            f"target={budget.target:.4g} cap={budget.cap}"
        )
        return budget

    def chunk_for(self, history: int) -> int:
        """Page-aligned chunk (at least one page) for a request whose prefix is `history` tokens."""
        x = int(self.model.tokens_within(max(0, history), self.target) + 1e-6)
        return max(self.page_size, min(self.cap, x) // self.page_size * self.page_size)

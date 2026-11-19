"""Late binding of requests to attention-DP ranks (SGLANG_DP_PULL_TOKENS).

Dispatching a request to a DP rank on arrival commits it to that rank's queue
even if another rank frees up first; with a few requests per rank and very
uneven prefill times, some ranks then sit idle in lockstep steps while others
still have queued work. Here requests wait at the DP controller instead, and a
rank receives the next one only once its queued prefill work (uncached tokens
waiting or left in its chunked request, as its scheduler reports after every
prefill step) falls below a budget of about one step, and at most `max_sent`
requests go to a rank between two of its snapshots (the controller does not know
a request's prefix hit, so it cannot size it; the scheduler's next snapshot does).
A busy rank thus keeps about one step of work queued and the rest of the requests
go to whichever rank frees up first. `max_waiting` > 0 also caps the requests a rank
has waiting (reported plus sent since): the reported uncached tokens of requests not
yet prefix-matched come from a recent hit rate and can be several times too high.
"""

from __future__ import annotations

import collections
import time
from typing import Any, Callable, Iterable, List, Optional, Tuple

# A request sent shortly before a snapshot may not have reached the scheduler
# when the snapshot was taken, so it stays counted as sent for this long.
_SNAPSHOT_SLACK_S = 0.05


class DPPullQueue:
    def __init__(self, num_ranks: int, budget_tokens: int, max_sent: int = 4, max_waiting: int = 0,
                 clock: Callable[[], float] = time.time):
        self.num_ranks = num_ranks
        self.budget_tokens = budget_tokens
        self.max_sent = max_sent
        self.max_waiting = max_waiting
        self.snap_waiting = [0] * num_ranks
        self.clock = clock          # same clock as LoadSnapshot.timestamp
        self.held: collections.deque = collections.deque()   # (req, prompt_tokens, enqueue time)
        self.snap_tokens = [0] * num_ranks
        self.snap_ts = [0.0] * num_ranks
        self.sent: List[List[Tuple[float, int]]] = [[] for _ in range(num_ranks)]
        self.hit_rate = [0.0] * num_ranks

    def __len__(self) -> int:
        return len(self.held)

    def update(self, loads: Iterable[Any]) -> None:
        """Take the ranks' latest load snapshots (num_waiting_uncached_tokens, cache_hit_rate)."""
        for load in loads:
            r = load.dp_rank
            if r is None or not 0 <= r < self.num_ranks or load.timestamp <= self.snap_ts[r]:
                continue
            self.snap_ts[r] = load.timestamp
            self.snap_tokens[r] = load.num_waiting_uncached_tokens
            self.snap_waiting[r] = load.num_waiting_reqs
            self.hit_rate[r] = min(1.0, max(0.0, load.cache_hit_rate))
            cutoff = load.timestamp - _SNAPSHOT_SLACK_S
            self.sent[r] = [s for s in self.sent[r] if s[0] > cutoff]

    def load(self, rank: int) -> int:
        return self.snap_tokens[rank] + sum(n for _, n in self.sent[rank])

    def estimate(self, prompt_tokens: int, rank: int) -> int:
        # The controller does not know the prefix hit; use the rank's recent hit rate.
        return max(1, round(prompt_tokens * (1.0 - self.hit_rate[rank])))

    def push(self, req: Any, prompt_tokens: int) -> None:
        self.held.append((req, prompt_tokens, self.clock()))

    def pop_ready(self, active_ranks: List[int]) -> List[Tuple[Any, int]]:
        """Requests to send now, in arrival order, each to the least-loaded active rank whose
        reported queued work is below the budget and that has not been sent `max_sent` since."""
        out = []
        now = self.clock()
        while self.held:
            open_ranks = [r for r in active_ranks if self._open(r)]
            if not open_ranks:
                break
            rank = min(open_ranks, key=lambda r: (self.load(r), len(self.sent[r])))
            req, prompt_tokens, _ = self.held.popleft()
            self.sent[rank].append((now, self.estimate(prompt_tokens, rank)))
            out.append((req, rank))
        return out

    def _open(self, r: int) -> bool:
        if self.snap_tokens[r] >= self.budget_tokens or len(self.sent[r]) >= self.max_sent:
            return False
        return not self.max_waiting or self.snap_waiting[r] + len(self.sent[r]) < self.max_waiting

    def remove(self, match: Callable[[Any], bool]) -> List[Any]:
        """Drop held requests matching `match` (e.g. aborted); returns them."""
        kept, dropped = collections.deque(), []
        for item in self.held:
            (dropped if match(item[0]) else kept).append(item)
        self.held = kept
        return [item[0] for item in dropped]

    def oldest_wait(self) -> Optional[float]:
        return self.clock() - self.held[0][2] if self.held else None

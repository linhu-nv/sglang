"""Step plan of adaptive DP/CP: whether the next step of an attention-DP worker is
a DP step or a CP step over all its ranks, and for which long cold requests.

Pure Python on purpose: every rank evaluates ``decide`` on the same gathered
rows, so the plans agree, and the simulator imports this module as is.

Default rule ("rate"): open a CP step when its token rate is at least alpha
times the token rate of the DP step that would run instead, with the lane
requests on their owners; a lane request not taken within max_wait goes to DP.
Rule "lat" simulates both schedules and compares the sum of latencies.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence, Tuple

import msgspec

# Rank row layout (int64): header, then K_DP (rem, length) pairs, then K_LANE
# lane entries of LANE_FIELDS ints.
K_DP = 8
K_LANE = 4
LANE_FIELDS = 5
HEADER = (
    "n_dp",
    "tail_tokens",
    "tail_count",
    "free_tokens",
    "free_rows",
    "n_lane",
    "arrival_mreq_s",
    "arrival_tokens",
    "arrival_len",
    "window_free",
    "running",
)
ROW_WIDTH = len(HEADER) + 2 * K_DP + LANE_FIELDS * K_LANE
# Lane entry flags.
FLAG_STARTED = 1
FLAG_ALIVE = 2


class StepModel(msgspec.Struct, frozen=True):
    """Step times in seconds. DP: lockstep over ranks; q tokens, L request length."""

    dp_c: float = 0.339
    dp_b: float = 1.38e-4
    dp_a: float = 0.0964
    dp_d: float = 4.72e-5
    # CP8 fit on .12 g_cp1 (1169 steps): 0.958 s + 5.34e-5 s/token + 0.075 s/request.
    cp_c: float = 0.958
    cp_b: float = 5.34e-5
    cp_a: float = 0.0
    cp_e: float = 0.075

    @classmethod
    def from_json(cls, path: str) -> "StepModel":
        """Read ``{"dp": {c, b, a, d}, "cp": {c, b[, a]}}`` (sim/step_models.json)."""
        with open(path) as f:
            d = json.load(f)
        dp, cp = d["dp"], d["cp"]
        return cls(
            dp_c=dp["c"],
            dp_b=dp["b"],
            dp_a=dp.get("a", 0.0),
            dp_d=dp.get("d", 0.0),
            cp_c=cp["c"],
            cp_b=cp["b"],
            cp_a=cp.get("a", 0.0),
            cp_e=cp.get("e", 0.0),
        )

    def dp_time(self, qs: Sequence[int], lens: Sequence[int]) -> float:
        return (
            self.dp_c
            + self.dp_b * max(qs)
            + self.dp_a * max(q * l for q, l in zip(qs, lens)) / 1e9
            + self.dp_d * sum(qs)
        )

    def cp_time(self, chunks: Sequence[Tuple[int, int]]) -> float:
        """One CP step over ``chunks`` of (start, take)."""
        q = sum(take for _, take in chunks)
        attn = sum(take * (start + take) for start, take in chunks)
        return (
            self.cp_c + self.cp_b * q + self.cp_a * attn / 1e9 + self.cp_e * len(chunks)
        )


class PolicyConfig(msgspec.Struct, frozen=True):
    min_tokens: int = 16384
    cp_chunk: int = 65536
    dp_chunk: int = 8192
    max_wait_s: float = 4.0
    margin: float = 0.02
    switch_s: float = 0.02
    # "" = cost rule, "cp" = every lane request as soon as memory allows, "dp" = never.
    force: str = ""
    mem_margin_tokens: int = 4096
    max_batch_reqs: int = 4
    page_size: int = 256
    # Window tokens kept free on every rank beside a CP chunk.
    window_margin: int = 4096
    # No CP step smaller than this (the window pool would make it a DP-sized chunk).
    min_cp_chunk: int = 8192
    # Consecutive CP steps after which one DP step runs if any rank has DP work.
    max_cp_burst: int = 4
    # Release lane requests past max_wait also while CP steps run other requests
    # (otherwise only a DP decision releases them).
    release_always: bool = False
    # Release a waiting lane request (after min_wait_s) whose predicted DP finish on
    # its owner is earlier than its predicted CP finish behind the lane ahead of it.
    release_eta: bool = True
    min_wait_s: float = 1.0
    # Simulated DP steps per schedule; beyond it the remaining work is charged at
    # the horizon (both schedules alike).
    horizon_steps: int = 256
    # "lat": sum of latencies of two simulated schedules; "rate": the CP step's
    # token rate against alpha x the next DP step's, with the lane request in it.
    rule: str = "rate"
    alpha: float = 1.0


class LaneEntry(msgspec.Struct, frozen=True):
    owner: int
    cp_id: int
    length: int
    done: int
    wait_ms: int
    started: bool
    alive: bool

    @property
    def key(self) -> Tuple[int, int]:
        return (self.owner, self.cp_id)


class RankView(msgspec.Struct, frozen=True):
    rank: int
    dp: List[Tuple[int, int]]  # FIFO (remaining uncached tokens, request length)
    tail_tokens: int
    tail_count: int
    free_tokens: int
    free_rows: int
    lane: List[LaneEntry]
    # Recent DP arrivals: requests per second, mean uncached tokens and length.
    arrival_rate: float = 0.0
    arrival_tokens: int = 0
    arrival_len: int = 0
    # Free sliding-window KV tokens (a prefill chunk's transient footprint);
    # negative when the pool has no window partition.
    window_free: int = -1
    # Requests decoding on the rank (DP work a CP step also pauses).
    running: int = 0


class CPItem(msgspec.Struct, frozen=True):
    owner: int
    cp_id: int
    start: int
    take: int
    length: int

    @property
    def key(self) -> Tuple[int, int]:
        return (self.owner, self.cp_id)

    @property
    def first(self) -> bool:
        return self.start == 0

    @property
    def last(self) -> bool:
        return self.start + self.take >= self.length


class Plan(msgspec.Struct, frozen=True):
    items: List[CPItem] = []
    # Lane requests their owners hand to DP.
    release: List[Tuple[int, int]] = []
    # Started lane requests whose owner dropped them (aborted): every rank frees.
    drop: List[Tuple[int, int]] = []
    reason: str = ""
    cost_dp: float = 0.0
    cost_cp: float = 0.0

    @property
    def is_cp(self) -> bool:
        return bool(self.items)


# ----------------------------------------------------------------------------
# Row codec


def encode_row(
    *,
    dp: Sequence[Tuple[int, int]],
    tail_tokens: int,
    tail_count: int,
    free_tokens: int,
    free_rows: int,
    lane: Sequence[Tuple[int, int, int, int, int]],
    arrival_rate: float = 0.0,
    arrival_tokens: int = 0,
    arrival_len: int = 0,
    window_free: int = -1,
    running: int = 0,
) -> List[int]:
    """``lane`` items: (cp_id, length, done, wait_ms, flags), oldest first."""
    dp = list(dp)[:K_DP]
    lane = list(lane)[:K_LANE]
    row = [
        len(dp),
        tail_tokens,
        tail_count,
        free_tokens,
        free_rows,
        len(lane),
        round(arrival_rate * 1000),
        arrival_tokens,
        arrival_len,
        window_free,
        running,
    ]
    for rem, length in dp:
        row += [rem, length]
    row += [0, 0] * (K_DP - len(dp))
    for entry in lane:
        assert len(entry) == LANE_FIELDS
        row += list(entry)
    row += [0] * (LANE_FIELDS * (K_LANE - len(lane)))
    assert len(row) == ROW_WIDTH
    return [int(x) for x in row]


def decode_rows(rows: Sequence[Sequence[int]]) -> List[RankView]:
    views = []
    for rank, row in enumerate(rows):
        (
            n_dp,
            tail_tokens,
            tail_count,
            free_tokens,
            free_rows,
            n_lane,
            arrival_mreq_s,
            arrival_tokens,
            arrival_len,
            window_free,
            running,
        ) = row[: len(HEADER)]
        off = len(HEADER)
        dp = [(int(row[off + 2 * i]), int(row[off + 2 * i + 1])) for i in range(n_dp)]
        off += 2 * K_DP
        lane = []
        for i in range(n_lane):
            cp_id, length, done, wait_ms, flags = row[
                off + LANE_FIELDS * i : off + LANE_FIELDS * (i + 1)
            ]
            lane.append(
                LaneEntry(
                    owner=rank,
                    cp_id=int(cp_id),
                    length=int(length),
                    done=int(done),
                    wait_ms=int(wait_ms),
                    started=bool(flags & FLAG_STARTED),
                    alive=bool(flags & FLAG_ALIVE),
                )
            )
        views.append(
            RankView(
                rank=rank,
                dp=dp,
                tail_tokens=int(tail_tokens),
                tail_count=int(tail_count),
                free_tokens=int(free_tokens),
                free_rows=int(free_rows),
                lane=lane,
                arrival_rate=arrival_mreq_s / 1000.0,
                arrival_tokens=int(arrival_tokens),
                arrival_len=int(arrival_len),
                window_free=int(window_free),
                running=int(running),
            )
        )
    return views


# ----------------------------------------------------------------------------
# Schedule simulation


def _queues(views: Sequence[RankView]) -> List[List[List[float]]]:
    """Per rank FIFO of [remaining tokens, length, weight, arrival time]; the
    tail is one item weighted by its request count."""
    queues = []
    for v in views:
        q = [[rem, length, 1, 0.0] for rem, length in v.dp if rem > 0]
        if v.tail_count > 0 and v.tail_tokens > 0:
            q.append([v.tail_tokens, v.tail_tokens / v.tail_count, v.tail_count, 0.0])
        queues.append(q)
    return queues


class _Arrivals:
    """Deterministic per-rank arrival streams over a window."""

    def __init__(self, views: Sequence[RankView]):
        self.streams = [
            (
                1.0 / v.arrival_rate,
                v.arrival_tokens,
                max(v.arrival_len, v.arrival_tokens),
            )
            if v.arrival_rate > 0 and v.arrival_tokens > 0
            else None
            for v in views
        ]
        self.next = [s[0] if s is not None else None for s in self.streams]

    def admit(self, queues, until: float) -> None:
        for r, s in enumerate(self.streams):
            if s is None:
                continue
            period, tokens, length = s
            while self.next[r] <= until:
                queues[r].append([tokens, length, 1, self.next[r]])
                self.next[r] += period


def simulate_dp(
    queues: List[List[List[float]]],
    model: StepModel,
    *,
    dp_chunk: int,
    t0: float,
    t_end: float,
    arrivals: Optional[_Arrivals],
    watch: Optional[list] = None,
) -> Tuple[float, float]:
    """Lockstep DP steps from ``t0`` until every queue drains (no arrivals) or
    ``t_end``. Returns (sum of weighted latencies, time the ``watch`` item
    completed or inf); work left at ``t_end`` is charged with its drain time."""
    t = t0
    total = 0.0
    watched_done = float("inf")
    if arrivals is not None:
        arrivals.admit(queues, t)
    while t < t_end and any(queues):
        qs, lens = [], []
        for q in queues:
            left, taken, longest = dp_chunk, 0, 0
            for item in q:
                if left <= 0:
                    break
                take = min(item[0], left)
                item[0] -= take
                left -= take
                taken += take
                longest = max(longest, item[1])
            qs.append(taken)
            lens.append(longest)
        t += model.dp_time(qs, lens)
        for q in queues:
            while q and q[0][0] <= 0:
                item = q.pop(0)
                total += (t - item[3]) * item[2]
                if item is watch:
                    watched_done = t
        if arrivals is not None:
            arrivals.admit(queues, t)
    for q in queues:
        rem = 0.0
        for item in q:
            rem += item[0]
            drain = rem / dp_chunk * model.dp_time([dp_chunk], [item[1]])
            total += (t + drain - item[3]) * item[2]
    return total, watched_done


def _cp_steps(
    batch: Sequence[LaneEntry], cfg: PolicyConfig, cap: Optional[int] = None
) -> List[List[Tuple[int, int, int]]]:
    """The CP steps that run ``batch`` back to back: per step (index, start, take)."""
    cap = cfg.cp_chunk if cap is None else cap
    steps, cur, left = [], [], cap
    for i, e in enumerate(batch):
        start = e.done
        while start < e.length:
            take = min(e.length - start, left)
            cur.append((i, start, take))
            start += take
            left -= take
            if left == 0:
                steps.append(cur)
                cur, left = [], cap
    if cur:
        steps.append(cur)
    return steps


def schedule_costs(
    views: Sequence[RankView],
    batch: Sequence[LaneEntry],
    cfg: PolicyConfig,
    model: StepModel,
) -> Tuple[float, float]:
    """(cost if ``batch`` is released to DP, cost if it runs first as CP steps):
    the sum of latencies, from now, of the requests present and arriving until
    the last of ``batch`` would finish on DP."""
    base = _queues(views)
    lane_wait = [e.wait_ms / 1000.0 for e in batch]
    # Without arrivals: when would the batch finish on DP (the window end).
    dp_q = [[list(item) for item in q] for q in base]
    watched = []
    for e, w in zip(batch, lane_wait):
        item = [e.length - e.done, e.length, 1, -w]
        dp_q[e.owner].insert(0, item)
        watched.append(item)
    probe = [[list(item) for item in q] for q in dp_q]
    probe_watch = [probe[e.owner][0] for e in batch[-1:]]
    _, t_end = simulate_dp(
        probe,
        model,
        dp_chunk=cfg.dp_chunk,
        t0=0.0,
        t_end=float("inf"),
        arrivals=None,
        watch=probe_watch[0] if probe_watch else None,
    )
    if t_end == float("inf"):
        t_end = 0.0
    cost_dp, _ = simulate_dp(
        dp_q,
        model,
        dp_chunk=cfg.dp_chunk,
        t0=0.0,
        t_end=t_end,
        arrivals=_Arrivals(views),
    )
    t = 0.0
    cost_cp = 0.0
    for step in _cp_steps(batch, cfg):
        t += model.cp_time([(start, take) for _, start, take in step])
        for i, start, take in step:
            if start + take >= batch[i].length:
                cost_cp += t + lane_wait[i]
    busy = sum(1 for q in base if q)
    t += cfg.switch_s * (1 if busy else 0)
    rest, _ = simulate_dp(
        base,
        model,
        dp_chunk=cfg.dp_chunk,
        t0=t,
        t_end=max(t_end, t),
        arrivals=_Arrivals(views),
    )
    return cost_dp, cost_cp + rest


# ----------------------------------------------------------------------------
# Decision


def _lane_order(views: Sequence[RankView]) -> List[LaneEntry]:
    entries = [e for v in views for e in v.lane]
    return sorted(entries, key=lambda e: (not e.started, -e.wait_ms, e.owner, e.cp_id))


def _fits(
    items: Sequence[CPItem], views: Sequence[RankView], cfg: PolicyConfig
) -> bool:
    need = sum(it.take for it in items) + cfg.mem_margin_tokens
    rows = sum(1 for it in items if it.first)
    return all(v.free_tokens >= need and v.free_rows >= rows for v in views)


def step_cap(views: Sequence[RankView], cfg: PolicyConfig) -> int:
    """Tokens one CP step may take: the chunk size, bounded by the window pool
    every rank must hold the chunk's transient sliding-window KV in."""
    cap = cfg.cp_chunk
    windows = [v.window_free for v in views if v.window_free >= 0]
    if windows:
        room = min(windows) - cfg.window_margin
        cap = min(cap, room // cfg.page_size * cfg.page_size)
    return cap if cap >= cfg.min_cp_chunk else 0


def _first_step(
    batch: Sequence[LaneEntry], cfg: PolicyConfig, cap: Optional[int] = None
) -> List[CPItem]:
    if cap == 0:
        return []
    steps = _cp_steps(batch, cfg, cap)
    if not steps:
        return []
    return [
        CPItem(
            owner=batch[i].owner,
            cp_id=batch[i].cp_id,
            start=start,
            take=take,
            length=batch[i].length,
        )
        for i, start, take in steps[0]
    ]


def _shrink_to_fit(items, views, cfg) -> List[CPItem]:
    items = list(items)
    while items and not _fits(items, views, cfg):
        items.pop()
    return items


def decide(
    views: Sequence[RankView],
    cfg: PolicyConfig,
    model: StepModel,
    cp_streak: int = 0,
) -> Plan:
    """``cp_streak``: CP steps run back to back right before this one."""
    plan = _decide(views, cfg, model, cp_streak)
    if cfg.force:
        return plan
    taken = {it.key for it in plan.items} | set(plan.release) | set(plan.drop)
    if cfg.release_eta:
        late = _eta_releases(views, cfg, model, taken)
        if late:
            plan = msgspec.structs.replace(plan, release=sorted(plan.release + late))
            taken |= set(late)
    if not cfg.release_always:
        return plan
    stale = [
        e.key
        for v in views
        for e in v.lane
        if not e.started
        and e.alive
        and e.wait_ms >= cfg.max_wait_s * 1000
        and e.key not in taken
    ]
    if not stale:
        return plan
    return msgspec.structs.replace(plan, release=sorted(plan.release + stale))


def _eta_releases(views, cfg, model, taken) -> List[Tuple[int, int]]:
    """Waiting lane requests that would finish earlier on their owner's DP queue
    than behind the lane requests ahead of them in CP steps."""
    cap = step_cap(views, cfg) or cfg.min_cp_chunk
    qs, lens = _next_dp_step(views, [], cfg)
    ahead = 0
    out = []
    for e in _lane_order(views):
        if not e.alive:
            continue
        rem = e.length - e.done
        ahead += rem
        if e.started or e.key in taken or e.wait_ms < cfg.min_wait_s * 1000:
            continue
        steps = -(-ahead // cap)
        cp_eta = steps * (model.cp_c + model.cp_e) + model.cp_b * ahead
        owner = views[e.owner]
        queued = sum(r for r, _ in owner.dp) + owner.tail_tokens + rem
        dp_qs = list(qs)
        dp_qs[e.owner] = cfg.dp_chunk
        dp_lens = list(lens)
        dp_lens[e.owner] = max(dp_lens[e.owner], e.length)
        dp_eta = -(-queued // cfg.dp_chunk) * model.dp_time(dp_qs, dp_lens)
        if dp_eta < cp_eta:
            out.append(e.key)
            ahead -= rem
    return out


def _decide(
    views: Sequence[RankView],
    cfg: PolicyConfig,
    model: StepModel,
    cp_streak: int = 0,
) -> Plan:
    lane = _lane_order(views)
    cap = step_cap(views, cfg)
    drop = [e.key for e in lane if e.started and not e.alive]
    lane = [e for e in lane if e.alive]
    started = [e for e in lane if e.started]
    waiting = [e for e in lane if not e.started]

    dp_work = any(v.dp or v.tail_count or v.running for v in views)
    # Decoding requests get a DP step after every CP step.
    burst = 1 if any(v.running for v in views) else cfg.max_cp_burst
    if cp_streak >= burst and dp_work and cfg.force != "cp":
        return Plan(drop=drop, reason="yield")

    if started:
        # A started request runs its chunks back to back; fresh requests may
        # share the step's remaining room.
        batch = started + (waiting if cfg.force == "cp" else [])
        items = _shrink_to_fit(_first_step(batch, cfg, cap), views, cfg)
        if not items:
            return Plan(drop=drop, reason="continue:no-memory")
        return Plan(items=items, drop=drop, reason="continue")

    if not waiting:
        return Plan(drop=drop, reason="no-lane")

    if cfg.force == "dp":
        return Plan(release=[e.key for e in waiting], drop=drop, reason="force-dp")

    if cfg.force == "cp":
        items = _shrink_to_fit(_first_step(waiting, cfg, cap), views, cfg)
        reason = "force-cp" if items else "force-cp:no-memory"
        return Plan(items=items, drop=drop, reason=reason)

    if cfg.rule == "rate":
        return _decide_rate(views, waiting, drop, cfg, model, cap)

    best = None
    for k in range(1, min(len(waiting), cfg.max_batch_reqs) + 1):
        batch = waiting[:k]
        items = _first_step(batch, cfg, cap)
        if not items or not _fits(items, views, cfg):
            break
        cost_dp, cost_cp = schedule_costs(views, batch, cfg, model)
        gain = cost_dp * (1.0 - cfg.margin) - cost_cp
        if best is None or gain > best[0]:
            best = (gain, items, cost_dp, cost_cp)
    if best is not None and best[0] > 0:
        _, items, cost_dp, cost_cp = best
        return Plan(
            items=items, drop=drop, reason="cost", cost_dp=cost_dp, cost_cp=cost_cp
        )
    release = [e.key for e in waiting if e.wait_ms >= cfg.max_wait_s * 1000]
    reason = "dp" if best is not None else "dp:no-memory"
    plan = Plan(release=release, drop=drop, reason=reason)
    if best is not None:
        plan = msgspec.structs.replace(plan, cost_dp=best[2], cost_cp=best[3])
    return plan


def _next_dp_step(views, batch, cfg) -> Tuple[List[int], List[int]]:
    """Per-rank tokens and longest request of the next DP step, ``batch`` at the
    head of its owners' queues."""
    queues = _queues(views)
    for e in reversed(batch):
        queues[e.owner].insert(0, [e.length - e.done, e.length, 1, 0.0])
    qs, lens = [], []
    for q in queues:
        left, taken, longest = cfg.dp_chunk, 0, 0
        for item in q:
            if left <= 0:
                break
            take = min(item[0], left)
            left -= take
            taken += take
            longest = max(longest, item[1])
        qs.append(taken)
        lens.append(longest)
    return qs, lens


def _decide_rate(views, waiting, drop, cfg, model, cap) -> Plan:
    batch = waiting[: cfg.max_batch_reqs]
    items = _shrink_to_fit(_first_step(batch, cfg, cap), views, cfg)
    if not items:
        release = [e.key for e in waiting if e.wait_ms >= cfg.max_wait_s * 1000]
        return Plan(release=release, drop=drop, reason="dp:no-memory")
    q_cp = sum(it.take for it in items)
    t_cp = model.cp_time([(it.start, it.take) for it in items]) + cfg.switch_s
    used = {it.key for it in items}
    qs, lens = _next_dp_step(views, [e for e in batch if e.key in used], cfg)
    r_dp = sum(qs) / model.dp_time(qs, lens)
    r_cp = q_cp / t_cp
    if r_cp >= cfg.alpha * r_dp:
        return Plan(items=items, drop=drop, reason="rate", cost_dp=r_dp, cost_cp=r_cp)
    release = [e.key for e in waiting if e.wait_ms >= cfg.max_wait_s * 1000]
    return Plan(release=release, drop=drop, reason="dp", cost_dp=r_dp, cost_cp=r_cp)


def config_from_env() -> Tuple[PolicyConfig, StepModel]:
    from sglang.srt.environ import envs

    cfg = PolicyConfig(
        min_tokens=envs.SGLANG_ADAPTIVE_CP_MIN_TOKENS.get(),
        cp_chunk=envs.SGLANG_ADAPTIVE_CP_CHUNK.get(),
        max_wait_s=envs.SGLANG_ADAPTIVE_CP_MAX_WAIT_S.get(),
        margin=envs.SGLANG_ADAPTIVE_CP_MARGIN.get(),
        switch_s=envs.SGLANG_ADAPTIVE_CP_SWITCH_S.get(),
        force=envs.SGLANG_ADAPTIVE_CP_FORCE.get(),
        rule=envs.SGLANG_ADAPTIVE_CP_RULE.get(),
        alpha=envs.SGLANG_ADAPTIVE_CP_ALPHA.get(),
        release_always=envs.SGLANG_ADAPTIVE_CP_RELEASE_ALWAYS.get(),
        max_cp_burst=envs.SGLANG_ADAPTIVE_CP_MAX_BURST.get(),
        release_eta=envs.SGLANG_ADAPTIVE_CP_RELEASE_ETA.get(),
    )
    path: Optional[str] = envs.SGLANG_ADAPTIVE_CP_MODEL.get() or None
    model = StepModel.from_json(path) if path else StepModel()
    return cfg, model

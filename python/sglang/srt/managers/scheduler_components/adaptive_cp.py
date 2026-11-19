"""Adaptive DP/CP in the scheduler of an attention-DP worker.

Long cold requests ("lane" requests) wait in a lane instead of the waiting
queue. Every scheduler iteration, all ranks gather a summary row, evaluate the
same plan (``adaptive_cp_policy.decide``), and either run their own DP step or
together run one CP step over lane requests. In a CP step the rank that owns a
request (its DP rank) runs it with its real ``Req``; every other rank runs a
replica built from the owner's broadcast token ids and frees it after the last
chunk. The owner keeps the KV and finishes the request as a DP request.
"""

from __future__ import annotations

import logging
import time
from array import array
from collections import deque
from contextlib import nullcontext
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.cp.adaptive import (
    AdaptiveCP,
    adaptive_cp_enabled,
    get_adaptive_cp,
    init_adaptive_cp,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components import adaptive_cp_policy as policy
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    move,
)
from sglang.srt.mem_cache.capacity import shares_memory
from sglang.srt.mem_cache.allocation import retire_window
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

ARRIVAL_WINDOW_S = 30.0


class _Owned(msgspec.Struct):
    """A lane request this rank owns."""

    req: Req
    cp_id: int
    enqueued: float
    started: bool = False
    done: int = 0
    alive: bool = True


class AdaptiveStep(msgspec.Struct):
    """What a CP batch carries for the merge and the result processing."""

    items: List[policy.CPItem]
    # Requests that leave the batch at the merge: replicas and middle chunks.
    unmergeable: List[Req]
    # Replicas whose last chunk ran in this batch, freed after its result.
    finished_replicas: List[Req]


class AdaptiveCPController:
    """The lane, the per-step plan collective, and CP batch construction."""

    def __init__(self, scheduler: "Scheduler", scope: AdaptiveCP) -> None:
        self.s = scheduler
        self.scope_obj = scope
        parallel = get_parallel()
        self.rank = parallel.tp_rank
        self.size = parallel.tp_size
        self.group = parallel.tp_group
        self.cfg, self.model = policy.config_from_env()
        self.cfg = msgspec.structs.replace(
            self.cfg, dp_chunk=scheduler.chunked_prefill_size or self.cfg.dp_chunk
        )
        self.evict_budget = int(
            envs.SGLANG_ADAPTIVE_CP_EVICT_FRAC.get() * scheduler.max_total_num_tokens
        )
        self.has_window = bool(
            scheduler.token_to_kv_pool_allocator.window_pool_names()
        )
        self.lane = IndexedReqList(ListKind.ADAPTIVE_CP, abort=self._abort)
        self.owned: Dict[int, _Owned] = {}
        self.replicas: Dict[Tuple[int, int], Req] = {}
        self.next_cp_id = 1
        # CP steps run back to back up to now; identical on every rank.
        self.cp_streak = 0
        self.arrivals: Deque[Tuple[float, int, int]] = deque()
        self.stats = {
            "cp_steps": 0,
            "cp_tokens": 0,
            "lane_in": 0,
            "released": 0,
            "dropped": 0,
            "dp_steps": 0,
            "yields": 0,
        }
        self.last_plan_reason = ""
        self.last_log = time.monotonic()
        logger.info(
            "adaptive DP/CP on: rank %d/%d, %s, %s",
            self.rank,
            self.size,
            self.cfg,
            self.model,
        )

    # ------------------------------------------------------------------
    # Lists and abort

    def request_lists(self) -> Dict[ListKind, IndexedReqList]:
        return {ListKind.ADAPTIVE_CP: self.lane}

    def _abort(self, recv_req) -> None:
        for st in list(self.owned.values()):
            if not recv_req.matches(st.req.rid):
                continue
            if st.started:
                # Replicas exist on every rank: the next plan drops it everywhere.
                st.alive = False
            else:
                del self.owned[st.cp_id]
                self.s.request_retirer.abort(
                    st.req, finished_reason=recv_req.finished_reason
                )

    # ------------------------------------------------------------------
    # Lane admission

    def _eligible(self, req: Req) -> bool:
        # Output-token logprobs are fine; input logprobs would span replicas.
        input_logprobs = (
            req.return_logprob and req.logprob_start_len < len(req.origin_input_ids)
        )
        return (
            not input_logprobs
            and req.grammar is None
            and req.multimodal_inputs is None
            and req.session is None
            and req.lora_id is None
            and req.input_embeds is None
            and not req.return_hidden_states
            and req.beam_group is None
            and req.positional_embed_overrides is None
            and req.token_indices_to_pool is None
            and not req.is_retracted
        )

    def _uncached_estimate(self, req: Req) -> int:
        if req.adaptive_cp_checked:
            return max(0, req.seqlen - req.num_matched_prefix_tokens)
        return int(req.seqlen * (1.0 - self._hit_rate()))

    def _hit_rate(self) -> float:
        try:
            return float(self.s.load_inquirer.get_recent_cache_hit_rate())
        except Exception:
            return 0.0

    def admit(self) -> None:
        """Examine new waiting requests: long cold ones join the lane."""
        s = self.s
        now = time.monotonic()
        claimed = []
        storage = (
            s.enable_hicache_storage
            or s.enable_flexkv
            or s.enable_lmcache
            or s.enable_unified_cache_external_linker
        )
        for req in list(s.waiting_queue):
            if not req.adaptive_cp_seen:
                req.adaptive_cp_seen = True
                self.arrivals.append((now, self._uncached_estimate(req), req.seqlen))
            if req.adaptive_cp_checked:
                continue
            if req.seqlen < self.cfg.min_tokens or not self._eligible(req):
                req.adaptive_cp_checked = True
                continue
            if s.tree_cache.has_uncommitted_restore(req):
                continue
            if storage:
                if not s.tree_cache.check_prefetch_progress(req.cache_request_handle):
                    continue
                loaded, start = s.tree_cache.pop_prefetch_loaded_span(
                    req.cache_request_handle
                )
                if loaded > 0:
                    req.storage_hit_length = loaded
                    req.storage_hit_start = start
                    req.host_hit_is_storage = False
            defer_cow_slot = shares_memory(s.token_to_kv_pool_allocator)
            req.init_next_round_input(s.tree_cache, defer_cow_slot=defer_cow_slot)
            if defer_cow_slot:
                s._drop_unadmitted_match(req)
            req.adaptive_cp_checked = True
            cold = (
                req.prefix_len == 0
                and req.host_hit_length == 0
                and req.storage_hit_length == 0
            )
            if cold and self.cfg.force != "dp":
                claimed.append(req)
        for req in claimed:
            st = _Owned(req=req, cp_id=self.next_cp_id, enqueued=now)
            self.next_cp_id += 1
            self.owned[st.cp_id] = st
            req.adaptive_cp_owner = self.rank
            req.adaptive_cp_id = st.cp_id
            self.stats["lane_in"] += 1
        move(claimed, self.lane)
        while self.arrivals and self.arrivals[0][0] < now - ARRIVAL_WINDOW_S:
            self.arrivals.popleft()

    # ------------------------------------------------------------------
    # Plan collective

    def _local_row(self) -> List[int]:
        s = self.s
        now = time.monotonic()
        dp: List[Tuple[int, int]] = []
        chunked = s.chunked_req
        if chunked is not None:
            dp.append((max(0, chunked.seqlen - chunked.prefix_len), chunked.seqlen))
        for req in s.waiting_queue:
            dp.append((self._uncached_estimate(req), req.seqlen))
        head, tail = dp[: policy.K_DP], dp[policy.K_DP :]
        lane = []
        for st in sorted(self.owned.values(), key=lambda x: x.enqueued)[: policy.K_LANE]:
            flags = (policy.FLAG_STARTED if st.started else 0) | (
                policy.FLAG_ALIVE if st.alive and not st.req.finished() else 0
            )
            lane.append(
                (
                    st.cp_id,
                    st.req.seqlen,
                    st.done,
                    int((now - st.enqueued) * 1000),
                    flags,
                )
            )
        span = min(ARRIVAL_WINDOW_S, max(1.0, now - self.arrivals[0][0])) if self.arrivals else 1.0
        n = len(self.arrivals)
        # What admission would see; a CP step may evict at most evict_budget
        # cached tokens, so replicas never push out much of a rank's own prefixes.
        allocator = s.token_to_kv_pool_allocator
        budget = allocator.create_prefill_budget(s.tree_cache)
        free_tokens = min(
            int(budget.remaining_total),
            allocator.owner_available_size() + self.evict_budget,
        )
        window_free = int(budget.remaining_window_tokens) if self.has_window else -1
        return policy.encode_row(
            dp=head,
            tail_tokens=sum(x[0] for x in tail),
            tail_count=len(tail),
            free_tokens=free_tokens,
            free_rows=s.req_to_token_pool.available_size(),
            lane=lane,
            arrival_rate=n / span if n else 0.0,
            arrival_tokens=int(sum(x[1] for x in self.arrivals) / n) if n else 0,
            arrival_len=int(sum(x[2] for x in self.arrivals) / n) if n else 0,
            window_free=window_free,
            running=len(s.running_batch.reqs),
        )

    def _gather(self, row: List[int]) -> List[List[int]]:
        local = torch.tensor(row, dtype=torch.int64)
        out = [torch.empty_like(local) for _ in range(self.size)]
        torch.distributed.all_gather(out, local, group=self.group.cpu_group)
        return [t.tolist() for t in out]

    def plan_step(self) -> Optional[ScheduleBatch]:
        """Run on every rank once per iteration; return a CP batch or None (DP step)."""
        self.admit()
        self._settle_middle_chunks()
        rows = self._gather(self._local_row())
        views = policy.decode_rows(rows)
        plan = policy.decide(views, self.cfg, self.model, self.cp_streak)
        self.cp_streak = self.cp_streak + 1 if plan.is_cp else 0
        if plan.is_cp or plan.release or plan.drop:
            logger.info(
                "adaptive DP/CP plan %s items=%s release=%s drop=%s",
                plan.reason,
                [(it.owner, it.cp_id, it.start, it.take) for it in plan.items],
                plan.release,
                plan.drop,
            )
        self.last_plan_reason = plan.reason
        now = time.monotonic()
        if now - self.last_log > 30.0:
            self.last_log = now
            logger.info(
                "adaptive DP/CP: %s lane=%d replicas=%d last plan %s",
                self.stats,
                len(self.owned),
                len(self.replicas),
                plan.reason,
            )
            if any(v.lane for v in views) and self.rank == 0:
                logger.info(
                    "adaptive DP/CP: lane %s free %s rows %s",
                    [e for v in views for e in v.lane],
                    [v.free_tokens for v in views],
                    [v.free_rows for v in views],
                )
        for key in plan.drop:
            self._drop(key)
        for key in plan.release:
            self._release_to_dp(key)
        if plan.reason == "yield":
            self.stats["yields"] += 1
        if not plan.is_cp:
            self.stats["dp_steps"] += 1
            return None
        return self._build(plan)

    def _settle_middle_chunks(self) -> None:
        """Between two chunks of a started request: the owner checkpoints it like
        any chunked prefill; a replica (barred from the tree) retires its window
        rows. Both free the sliding-window KV the next chunk needs room for."""
        tree = self.s.tree_cache
        for st in self.owned.values():
            req = st.req
            if st.started and req.prefix_len < req.extend_end and not req.finished():
                checkpoint_kv_cache(req, tree)
        if tree.supports_swa():
            for req in self.replicas.values():
                if req.prefix_len < req.extend_end:
                    retire_window(req, req.extend_end, tree)
                    req.prefix_len = req.extend_end

    def _drop(self, key: Tuple[int, int]) -> None:
        owner, cp_id = key
        self.stats["dropped"] += 1
        if owner == self.rank:
            st = self.owned.pop(cp_id, None)
            if st is not None:
                self.s.request_retirer.abort(st.req)
        else:
            replica = self.replicas.pop(key, None)
            if replica is not None:
                self._free_replica(replica)

    def _release_to_dp(self, key: Tuple[int, int]) -> None:
        owner, cp_id = key
        if owner != self.rank:
            return
        st = self.owned.pop(cp_id)
        assert not st.started
        self.stats["released"] += 1
        move([st.req], self.s.waiting_queue, at=0)

    # ------------------------------------------------------------------
    # CP batch

    def _broadcast_inputs(self, item: policy.CPItem) -> Tuple[str, List[int]]:
        obj = [None]
        if item.owner == self.rank:
            req = self.owned[item.cp_id].req
            obj = [(req.rid, list(req.origin_input_ids))]
        torch.distributed.broadcast_object_list(
            obj, src=self.group.ranks[item.owner], group=self.group.cpu_group
        )
        return obj[0]

    def _new_replica(self, item: policy.CPItem, rid: str, ids: List[int]) -> Req:
        req = Req(
            rid=f"{rid}#cp{item.owner}",
            origin_input_text="",
            origin_input_ids=array("q", ids),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.skip_radix_cache_insert = True
        req.adaptive_cp_owner = item.owner
        req.adaptive_cp_id = item.cp_id
        assert len(ids) == item.length, (len(ids), item)
        move([req], self.lane)
        return req

    def _build(self, plan: policy.Plan) -> ScheduleBatch:
        s = self.s
        reqs: List[Req] = []
        unmergeable: List[Req] = []
        finished_replicas: List[Req] = []
        launched: List[Req] = []
        for item in plan.items:
            if item.first:
                rid, ids = self._broadcast_inputs(item)
            if item.owner == self.rank:
                st = self.owned[item.cp_id]
                req = st.req
                st.started = True
                st.done = item.start + item.take
                if item.last:
                    del self.owned[item.cp_id]
                    launched.append(req)
                else:
                    unmergeable.append(req)
            else:
                if item.first:
                    req = self._new_replica(item, rid, ids)
                    self.replicas[item.key] = req
                else:
                    req = self.replicas[item.key]
                unmergeable.append(req)
                if item.last:
                    finished_replicas.append(self.replicas.pop(item.key))
            req.prefix_len = item.start
            req.extend_end = item.start + item.take
            reqs.append(req)
        move(launched, s.prefill_launched_reqs)
        batch = ScheduleBatch.init_new(
            reqs,
            s.req_to_token_pool,
            s.token_to_kv_pool_allocator,
            s.tree_cache,
            s.model_config,
            s.enable_overlap,
            s.spec_algorithm,
        )
        batch.prepare_for_extend()
        if s.tp_worker.model_runner.prefill_aware_swa:
            for req in reqs:
                req.kv.shield_window_prefill(req.extend_end)
        self._check_agreement(plan, batch)
        batch.adaptive_cp_step = True
        batch.adaptive_cp = AdaptiveStep(
            items=list(plan.items),
            unmergeable=unmergeable,
            finished_replicas=finished_replicas,
        )
        self.stats["cp_steps"] += 1
        self.stats["cp_tokens"] += batch.extend_num_tokens
        return batch

    def _check_agreement(self, plan: policy.Plan, batch: ScheduleBatch) -> None:
        """Every rank must run the same CP batch: compare a signature."""
        sig = [
            len(batch.reqs),
            int(batch.extend_num_tokens),
            sum((it.owner + 1) * 1_000_003 + it.cp_id * 7919 + it.start for it in plan.items)
            % (1 << 61),
        ]
        local = torch.tensor(sig, dtype=torch.int64)
        lo, hi = local.clone(), local.clone()
        cpu = self.group.cpu_group
        torch.distributed.all_reduce(lo, op=torch.distributed.ReduceOp.MIN, group=cpu)
        torch.distributed.all_reduce(hi, op=torch.distributed.ReduceOp.MAX, group=cpu)
        if not torch.equal(lo, hi):
            raise RuntimeError(
                f"adaptive DP/CP: ranks disagree on the CP batch: local {sig}, "
                f"min {lo.tolist()}, max {hi.tolist()}, plan {plan}"
            )

    def scope(self, batch: Optional[ScheduleBatch]):
        if batch is None or batch.adaptive_cp is None:
            return nullcontext()
        return self.scope_obj.scope()

    @staticmethod
    def unmergeable(batch: ScheduleBatch) -> List[Req]:
        return batch.adaptive_cp.unmergeable

    def after_result(self, batch: ScheduleBatch) -> None:
        """Free the replicas whose last chunk this batch ran (its result is in)."""
        for req in batch.adaptive_cp.finished_replicas:
            self._free_replica(req)

    def _free_replica(self, req: Req) -> None:
        if req.owner_list is self.lane:
            self.lane._leave([req])
        if req.kv.holds_kv:
            release_kv_cache(req, self.s.tree_cache, checkpoint=False)


def maybe_init_adaptive_cp(scheduler: "Scheduler") -> Optional[AdaptiveCPController]:
    if not adaptive_cp_enabled():
        return None
    scope = get_adaptive_cp() or init_adaptive_cp(scheduler.tp_worker.model_runner.model)
    from sglang.srt.disaggregation.mode import DisaggregationMode

    if scheduler.disaggregation_mode != DisaggregationMode.NULL:
        logger.warning("adaptive DP/CP: not supported with PD disaggregation; lane off")
        return None
    return AdaptiveCPController(scheduler, scope)

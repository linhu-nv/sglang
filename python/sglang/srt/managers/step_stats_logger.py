"""Per-step, per-DP-rank prefill stats as jsonl (test instrumentation).

Enabled by SGLANG_STEP_STATS_DIR; writes <dir>/dp_rank_<r>.jsonl, one row per
forward pass of this rank (idle passes included), in the schema of an external
step_stats: cached_* count only on a request's first chunk.
"""

import collections
import json
import os
import time


class StepStatsLogger:
    def __init__(self, dp_rank: int):
        self.dir = os.environ.get("SGLANG_STEP_STATS_DIR", "")
        self.group = os.environ.get("SGLANG_STEP_STATS_GROUP", "A")
        self.dp_rank = int(dp_rank)
        self.pending = collections.deque()
        self.seen = collections.OrderedDict()
        self.f = None
        # Per-request detail: [rid, computed tokens, prefix tokens] per row.
        self.with_rids = os.environ.get("SGLANG_STEP_STATS_RIDS") == "1"
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)
            self.f = open(os.path.join(self.dir, f"dp_rank_{self.dp_rank}.jsonl"), "a", buffering=1)

    def on_launch(self, batch) -> None:
        if self.f is None:
            return
        reqs, prefix, extend = batch.reqs or [], batch.prefix_lens, batch.extend_lens
        extend_mode = batch.forward_mode.is_extend() and prefix is not None and extend is not None
        row = dict(timestamp=0.0, worker_group=self.group, worker_rank=self.dp_rank, tp_rank=0,
                   dp_rank=self.dp_rank, forward_iter=batch.forward_iter, mode=str(batch.forward_mode.name),
                   launch_ts=time.time())
        lens, sched, cached, comp, dev, host, sto = [], 0, 0, 0, 0, 0, 0
        detail = []
        if extend_mode and self.with_rids:
            detail = [[req.rid, int(extend[i]), int(prefix[i])] for i, req in enumerate(reqs)]
        if extend_mode:
            for i, req in enumerate(reqs):
                lens.append(len(req.origin_input_ids))
                first = req.rid not in self.seen
                if first:
                    self.seen[req.rid] = None
                    if len(self.seen) > 200000:
                        self.seen.popitem(last=False)
                c = int(prefix[i]) if first else 0
                cached += c
                comp += int(extend[i])
                sched += c + int(extend[i])
                if first:
                    dev += int(getattr(req, "cached_tokens_device", 0) or 0)
                    host += int(getattr(req, "cached_tokens_host", 0) or 0)
                    sto += int(getattr(req, "cached_tokens_storage", 0) or 0)
        row.update(is_idle=not lens, request_count=len(lens), request_token_sum=sum(lens),
                   request_token_mean=(sum(lens) / len(lens)) if lens else 0,
                   request_token_min=min(lens) if lens else 0, request_token_max=max(lens) if lens else 0,
                   scheduled_token_sum=sched, cached_token_sum=cached, compute_token_sum=comp,
                   device_cached_token_sum=dev, host_cached_token_sum=host, storage_cached_token_sum=sto,
                   num_reqs_batch=len(reqs))
        if self.with_rids:
            row["reqs"] = detail
            cp = getattr(batch, "adaptive_cp", None)
            if cp is not None:
                row["cp_items"] = [[it.owner, it.cp_id, it.start, it.take] for it in cp.items]
        self.pending.append(row)

    def on_result(self) -> None:
        if self.f is None or not self.pending:
            return
        row = self.pending.popleft()
        row["timestamp"] = time.time()
        self.f.write(json.dumps(row) + "\n")

"""Adaptive DP/CP: run one forward of an attention-DP worker as a prefill-CP step.

An attention-DP deployment with ``attn_tp_size == 1`` and a CP deployment with
``attn_cp_size == tp_size`` lay out the weights the same way (attention
replicated per rank, MoE sharded over the same EP group), so the ranks of one
attention-DP worker can run a forward together as an interleaved CP step: every
rank takes the same batch, computes a shard of its tokens, and ends with the
full KV of every request in its own pool.

``AdaptiveCP.scope()`` switches the process to the CP topology for one forward:
it overrides the parallel context the way a static CP launch publishes it,
installs the interleave strategy, and swaps in the layer-boundary plans that
were bound once under that topology at startup. Everything that reads the
topology per forward then takes its CP path.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import List, Optional, Tuple

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_flags, get_parallel

logger = logging.getLogger(__name__)


def adaptive_cp_enabled() -> bool:
    return envs.SGLANG_ADAPTIVE_CP.get()


def adaptive_cp_supported(reason: Optional[List[str]] = None) -> bool:
    """Whether this process's topology can run CP steps: attention DP over the
    whole TP group, one rank per attention-DP group, no pipeline."""
    parallel = get_parallel()
    problems = []
    if not parallel.attn_dp_enabled:
        problems.append("attention DP is off")
    if parallel.attn_tp_size != 1:
        problems.append(f"attn_tp_size={parallel.attn_tp_size}")
    if parallel.attn_cp_size != 1:
        problems.append(f"attn_cp_size={parallel.attn_cp_size}")
    if parallel.attn_dp_size != parallel.tp_size:
        problems.append(f"attn_dp_size={parallel.attn_dp_size} != tp_size")
    if parallel.pp_size != 1:
        problems.append(f"pp_size={parallel.pp_size}")
    if reason is not None:
        reason.extend(problems)
    return not problems


class AdaptiveCP:
    """The CP topology of an attention-DP worker, entered per forward."""

    def __init__(self, model) -> None:
        problems: List[str] = []
        if not adaptive_cp_supported(problems):
            raise ValueError("adaptive DP/CP needs " + "; ".join(problems))
        parallel = get_parallel()
        self.cp_size = parallel.tp_size
        tp_group = parallel.tp_group
        # The values a static `--tp N --enable-prefill-cp --cp-strategy interleave`
        # launch publishes, where they differ from this attention-DP launch.
        self.overrides = dict(
            attn_dp_size=1,
            attn_dp_rank=0,
            attn_dp_enabled=False,
            num_dp_ranks=1,
            attn_cp_size=self.cp_size,
            attn_cp_rank=parallel.tp_rank,
            attn_cp_group=tp_group,
            # CP partners share tokens before the MoE through the MoE-DP group.
            moe_dp_group=tp_group,
            enable_prefill_cp=True,
            cp_strategy="interleave",
            moe_dense_tp_size=1,
        )
        from sglang.srt.layers.cp.interleave import InterleaveCPStrategy

        self.strategy = InterleaveCPStrategy(cp_size=self.cp_size)
        self._model = model
        self._dp_plans: List[Tuple] = []
        self._cp_plans: List[Tuple] = []
        self._patches: List[Tuple[object, str, object]] = []
        self._active = False
        with self._topology():
            self._bind_plans()
            self._collect_patches(model)
        logger.info(
            "adaptive DP/CP: CP%d scope ready (%d stage plans, %d patches)",
            self.cp_size,
            len(self._cp_plans),
            len(self._patches),
        )

    @property
    def active(self) -> bool:
        return self._active

    @contextmanager
    def _topology(self):
        from sglang.srt.layers.cp import base as cp_base

        dp_flags = get_flags().dp
        saved_enabled = dp_flags.enabled
        saved_strategy = cp_base._STRATEGY
        with get_parallel().override(**self.overrides):
            dp_flags.enabled = False
            cp_base._STRATEGY = self.strategy
            try:
                yield
            finally:
                cp_base._STRATEGY = saved_strategy
                dp_flags.enabled = saved_enabled

    def _bind_plans(self) -> None:
        from sglang.srt.layers.layer_boundary.factories import bind_recorded_stacks

        self._cp_plans = bind_recorded_stacks()
        if not self._cp_plans:
            raise RuntimeError(
                "adaptive DP/CP: no layer stack was recorded at model construction "
                "(SGLANG_ADAPTIVE_CP must be set before the model is built)"
            )
        self._dp_plans = [(b, b.plan, b.declaration) for b, _, _ in self._cp_plans]

    def _collect_patches(self, model) -> None:
        """Attributes fixed at construction from the topology, with their CP values."""
        from sglang.srt.layers.logits_processor import LogitsProcessor

        for module in model.modules():
            if isinstance(module, LogitsProcessor):
                if module.do_tensor_parallel_all_gather_dp_attn:
                    self._patches.append(
                        (module, "do_tensor_parallel_all_gather_dp_attn", False)
                    )

    @staticmethod
    def _install(plans) -> None:
        for boundary, plan, declaration in plans:
            boundary.plan = plan
            boundary.declaration = declaration

    @contextmanager
    def scope(self):
        """Run the enclosed forward as a CP step of all ranks."""
        if self._active:
            raise RuntimeError("adaptive DP/CP scope is not reentrant")
        saved = [(obj, name, getattr(obj, name)) for obj, name, _ in self._patches]
        with self._topology():
            self._install(self._cp_plans)
            for obj, name, value in self._patches:
                setattr(obj, name, value)
            self._active = True
            try:
                yield
            finally:
                self._active = False
                for obj, name, value in saved:
                    setattr(obj, name, value)
                self._install(self._dp_plans)


_ADAPTIVE: Optional[AdaptiveCP] = None


def init_adaptive_cp(model) -> Optional[AdaptiveCP]:
    """Build the CP scope for this worker's model when adaptive DP/CP is on."""
    global _ADAPTIVE
    if not adaptive_cp_enabled():
        return None
    _ADAPTIVE = AdaptiveCP(model)
    return _ADAPTIVE


def get_adaptive_cp() -> Optional[AdaptiveCP]:
    return _ADAPTIVE

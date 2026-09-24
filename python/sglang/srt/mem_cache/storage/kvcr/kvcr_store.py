# SPDX-License-Identifier: Apache-2.0
"""KVCR as a SGLang HiCacheStorage backend (DRAFT / WIP).

KVCR is an asymmetric peer-to-peer G2 KV coordinator (the ``nvidia-kvcr`` wheel,
package ``kvcr``).
This module adapts it to SGLang's content-addressed ``HiCacheStorage`` contract.

Mapping (see the module docstring sections below for the honest status of each):

    HiCacheStorage                     KVCR core
    ------------------------------     --------------------------------------
    batch_set_v2  (host -> storage)    deposit()  -> KVCR-owned local DRAM slots
    batch_get_v2  local hit            deliver() -> local get-back (slot->host)
    batch_get_v2  remote hit           submit_hint() + deliver() (NIXL pull)
    batch_exists_v2                    local-DRAM residency + router-hint cover

Threading: KVCR owns a single daemon "progress thread" (``kvcr/progress.py``)
that solely holds the NIXL agent + ZMQ control socket and advances every
in-flight op. This backend never touches NIXL directly. Main-thread state
(residency, pins) is advanced by calling ``kvcr.poll_completed()``, which two
threads here do: the HiCache controller's prefetch thread (inside
``_drain_until``) and the scheduler thread (inside ``tick``, which HiCache
calls once per loop iteration). The tick exists because a *source*-side serve
makes no progress unless somebody polls, and an idle worker has no traffic of
its own to poll for -- see ``tick``. ``_poll_lock`` serializes the two.

Both directions of the KV path are now wired to real KVCR operations: set via
``deposit``, get via the unified ``deliver`` (which the core routes per key to
either its local DRAM tier or a source-peer NIXL pull, the latter gated on a
router hint registered for the request via ``submit_hint``). ``_drain_until``
blocks the calling thread until the op reports -- by design: that caller is the
controller's dedicated prefetch daemon, and ``_page_transfer`` reads
``completed_tokens`` as soon as it returns.

Both zero-copy call shapes are implemented: ``batch_*_v2`` (PoolTransfer, used
by HybridCacheController) and ``batch_*_v1`` (keys + host_indices, used by
HiRadixCache's ``_page_{get,set}_zero_copy``). v1 is a thin KV-pool wrapper
over v2. The remaining DRAFT edges are the byte-copy legacy methods
(``get``/``set``/``batch_get``/``batch_set``), which no zero-copy backend uses.

Segment sub-blocking: a host KV page is not one contiguous run. MHA stores K
and V in separate halves of the pool tensor (and per-layer sub-runs in
``layer_first`` layout), so ``get_page_buffer_meta`` returns several
non-contiguous segments per page. KVCR's local tier copies exactly one
``MemDescriptor`` of one slot's size into each slot, so each page deposits as
one KVCR block-key per segment (page key + ``#<seg>`` suffix). The segment
sizes and count are discovered by probing the pool once at registration.

Hybrid stacks (DSA/MiniMax indexer, Mamba, SWA, DeepSeek V4, EAGLE draft) add
sidecar host pools alongside the KV one, each with its own allocation and its
own segment sizes. Each (host pool, distinct segment size) pair becomes one
KVCR local-DRAM pool, since KVCR sizes a pool's slots uniformly and rejects a
descriptor whose size does not match its pool's; a sidecar page's segments are
keyed ``<page>#<pool>:<seg>`` so they never collide with the KV page's. See
``_kvcr_pools``.

DeepSeek V4 goes one step further: its KV pool is a pure page anchor holding no
KV bytes at all, and every byte lives in the sidecars. The anchor still has to
store and load, because the controller gates all sidecar IO on the KV pool
completing, so it is served a constant marker block per page -- see
``_logical_anchor_layout``.
"""

from __future__ import annotations

import functools
import logging
import socket
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

import msgspec
import torch
from kvcr import KVCR, KVCRBindings
from kvcr.config import (
    FrameworkDramInput,
    KVCRBackendConfigs,
    KVCRConfig,
    LocalDramOptions,
    RemoteFWDramOptions,
)
from kvcr.control_channels import ZmqPeerControlChannel
from kvcr.policy import (
    FIFOPolicy,
    G3FIFOPolicy,
    G3LRUPolicy,
    KVCachePolicy,
    LRUPolicy,
)
from kvcr.types import BlockKey, MemDescriptor, OpEntryStatus, QueryStatus

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pool_host import HostKVCache, HostPoolGroup
from sglang.srt.mem_cache.storage.kvcr.kvcr_config import (
    MAX_TCP_PORT,
    KVCRBackendConfig,
)
from sglang.srt.mem_cache.storage.kvcr.pin_adapter import NoFrameworkPinning
from sglang.srt.mem_cache.storage.kvcr.router_hint import (
    RouterHint,
    StrKeyAdapter,
)
from sglang.srt.mem_cache.storage.kvcr.router_hint import encode_key as _encode_key
from sglang.srt.utils import dynamic_import

logger = logging.getLogger(__name__)

# Multi-region framework registration. ``framework_dram`` is a single (addr,
# length) pair, which covers a stack whose host pools all live in one
# allocation -- i.e. a KV-only one. Every sidecar pool is a separate
# allocation, so a hybrid stack needs the typed list that superseded it. It is
# feature-detected rather than imported: the field is not on every nvidia-kvcr
# the KV-only path runs against, and a hard import would turn an unrelated
# version skew into an import error for a stack that never needed it.
try:  # pragma: no cover - depends on the installed nvidia-kvcr
    from kvcr.config import FrameworkMemoryRegion

    _HAS_FRAMEWORK_REGIONS = "framework_regions" in getattr(
        KVCRBackendConfigs, "__dataclass_fields__", {}
    )
except ImportError:  # pragma: no cover - older nvidia-kvcr
    FrameworkMemoryRegion = None
    _HAS_FRAMEWORK_REGIONS = False

# Backoff bounds for _drain_until's completion poll. Start tight so a local-tier
# hit (already resident, microseconds away) is not needlessly delayed, then back
# off so a remote NIXL fetch does not spin a core while the KVCR progress thread
# does the actual work.
_DRAIN_POLL_MIN_S = 50e-6
_DRAIN_POLL_MAX_S = 2e-3

# How often the remote-path counters are summarized to the log. The remote path
# is the whole point of this backend and it fails *silently* -- a hint that
# never arrives and a fetch that returns nothing both look like an ordinary
# cache miss from outside the process. One line per interval is the cheapest
# way for an operator to tell "the router stopped hinting" from "the transfers
# are failing", which need fixes in different repositories.
_STATS_LOG_INTERVAL_S = 30.0

# How many abandoned op handles to remember. Arbitrary; large enough to cover
# the ops in flight when a stall starts, small enough to stay negligible.
_ABANDONED_OP_HISTORY = 256

# Longest the scheduler may park under --sleep-on-idle while this backend is a
# live P2P source, in ms. It bounds how long a peer's pull waits for its first
# pump. Matches kvcr's own progress-thread idle wait (1ms) so the two sides of
# a serve run at the same cadence.
_IDLE_TICK_INTERVAL_MS = 1

# KVCR splits its local DRAM tier into named pools, each with its own block
# size, and a descriptor's ``info`` field names the pool it belongs to. A
# backend depositing one segment kind at one size uses the single unnamed pool
# -- KVCR's own convention for that case, and the only legal name once there is
# more than one pool is a real one (``config._validate_pool_layouts``). So this
# name is used exactly when the engine registers one host pool with one segment
# size, which keeps a KV-only stack byte-identical to the single-pool backend.
_ANONYMOUS_POOL_NAME = ""

# One logical-anchor page's marker block, in bytes. DeepSeek V4's KV pool owns
# page indices and no KV bytes, but its key still has to be deposited and
# delivered so the controller's sidecar gate opens -- see
# ``_logical_anchor_layout`` -- so every anchor page gets a block this size.
# Arbitrary; small enough that the whole marker buffer stays under a megabyte
# for any real pool. Both peers derive it from this constant, so a remote
# fetch's source and destination descriptor lists agree.
_ANCHOR_MARKER_BYTES = 64

# The only transport KVCR's ZMQ control channel can dial a peer over, and the
# bind wildcards that are legal to bind but cannot be dialed. Loopback is *not*
# here: colocated workers are the normal single-host topology. A hint arrives
# from outside this process, so both are checked before the endpoint reaches
# ``submit_hint`` -- see _split_control_endpoint.
_CONTROL_SCHEME = "tcp://"
_UNDIALABLE_HINT_HOSTS = frozenset({"0.0.0.0", "::", "[::]", "*"})


# KVCR core methods this backend calls. Checked once at startup because
# ``nvidia-kvcr`` is pre-1.0 and pinning its version would not help: the
# distribution has sat at 0.1.0 across renames that moved ``kvcc.kvcc`` to
# ``kvcc.api``, deleted ``nixl.py``, dropped ``has_pending_work``, changed
# ``query`` from returning statuses to ``(status, tier)`` pairs, and finally
# renamed the whole package from ``kvcc`` to ``kvcr`` -- still 0.1.0. A missing
# name surfaces here as one legible error naming what is absent, rather than as
# an AttributeError from inside a prefetch on the first cache miss.
_REQUIRED_KVCR_METHODS = (
    "deposit",
    "deliver",
    "discard_hint",
    "poll_completed",
    "query",
    "submit_hint",
)


def _require_kvcr_api() -> None:
    missing = [name for name in _REQUIRED_KVCR_METHODS if not hasattr(KVCR, name)]
    if missing:
        raise RuntimeError(
            f"KVCRStore: the installed nvidia-kvcr is missing {missing}. This "
            "backend tracks the kvcr core's current API; upgrade nvidia-kvcr "
            "or use a SGLang revision matching your kvcr."
        )


# Placement/eviction policies selectable by name from extra_config. Mirrors the
# same table on the vLLM side so a name means the same thing in both engines'
# configs and an A/B is comparable across them.
_BUILTIN_POLICIES: Dict[str, type] = {
    "fifo": FIFOPolicy,
    "lru": LRUPolicy,
    "g3_fifo": G3FIFOPolicy,
    "g3_lru": G3LRUPolicy,
}


class _PoolLayout(msgspec.Struct, frozen=True):
    """How one sglang host pool's page maps onto KVCR block keys and slots.

    ``segment_sizes`` is one page's components in ``get_page_buffer_meta``
    order; ``kvcr_pool_names`` names, per component, the KVCR local-DRAM pool
    whose slot receives it. The two lists are parallel and their length is the
    page's segment count.

    Components are grouped into KVCR pools by *byte size*, not by index: KVCR
    sizes a pool's slots uniformly and rejects a descriptor of any other size
    (``core._normalize_descriptors``), so one pool per distinct size is the
    coarsest grouping that satisfies it. Per-component pools would be correct
    too and are much worse -- an MHA ``layer_first`` page has ``2 * layer_num``
    components, so that spelling would declare a hundred-odd KVCR pools for a
    pool whose segments are all the same size.

    ``key_prefix`` disambiguates two pools' segments under one page hash. It is
    empty for KV, which keeps a KV-only stack's keys byte-identical to what the
    single-pool backend wrote (and what a deployed peer expects on the wire).

    ``marker_buffer`` is set only for a logical anchor (see
    ``_logical_anchor_layout``) and is then the pool's whole byte backing: the
    host pool itself holds none, so this is what descriptors address.
    """

    name: str
    host_pool: HostKVCache
    segment_sizes: Tuple[int, ...]
    kvcr_pool_names: Tuple[str, ...]
    key_prefix: str
    marker_buffer: Optional[torch.Tensor] = None

    @property
    def segments_per_page(self) -> int:
        return len(self.segment_sizes)

    @property
    def bytes_per_page(self) -> int:
        return sum(self.segment_sizes)

    @property
    def is_logical_anchor(self) -> bool:
        return self.marker_buffer is not None


def _kvcr_pool_name(pool_name: str, segment_size: int, uniform: bool) -> str:
    """Name of the KVCR pool holding ``pool_name``'s ``segment_size`` segments.

    Deterministic and derived only from values both peers compute identically:
    a remote fetch drops any key whose source and destination descriptor
    ``info`` lists differ (``remote_fw_dram.start_write``), silently, so a name
    carrying anything rank- or process-local would break P2P rather than fail.

    The size suffix appears only where it disambiguates, so the common uniform
    pool reads as its sglang name.
    """
    return pool_name if uniform else f"{pool_name}:{segment_size}"


def _resolve_policy(name: str) -> KVCachePolicy:
    """Build the policy named in extra_config.

    The core picks its own default when handed ``None``, and that default is not
    a stable interface -- it moved from FIFO to LRU in kvcc e3a816e. So this
    backend always names one, and the name is what gets logged and recorded
    alongside a benchmark number.
    """
    policy_type = _BUILTIN_POLICIES.get(name)
    if policy_type is None:
        if "." not in name:
            raise ValueError(
                f"KVCRStore: unknown policy {name!r}. Supported: "
                f"{sorted(_BUILTIN_POLICIES)}; an external policy must be given "
                "as a fully qualified module.Class path."
            )
        policy_type = dynamic_import(name)
        if not isinstance(policy_type, type) or not issubclass(
            policy_type, KVCachePolicy
        ):
            raise TypeError(f"KVCRStore: {name} is not a KVCachePolicy subclass")
    return policy_type()


# How often a fault escaping into the HiCacheStorage surface is logged with its
# traceback. The guard exists for repeatable faults (a peer that stays down, a
# core in a bad state), so one traceback per prefetch would be the loudest thing
# in the log while saying nothing new after the first.
_FAULT_LOG_INTERVAL_S = 30.0


def _fail_closed(on_error):
    """Never let an exception out of a ``HiCacheStorage`` entry point.

    HiCache's three storage threads (``prefetch_thread_func``,
    ``prefetch_io_aux_func``, ``backup_thread_func``) each catch only ``Empty``.
    Anything else ends the thread, and they are unsupervised daemons, so one
    exception disables L2 and L3 for the life of the process -- and does it
    silently, since every later request just reports a cache miss.

    It is not only a lost cache. Those loops are the only ones that give back
    what they reserved: ``prefetch_io_aux_func`` calls
    ``append_host_mem_release`` (without it ``prefetch_tokens_occupied`` climbs
    until the rate limiter blocks all prefetching, permanently), and
    ``backup_thread_func`` is the sole producer for ``ack_backup_queue`` (without
    it ``HiRadixCache`` never calls ``entry.release_host()`` and backed-up nodes
    pin host pages forever).

    So a fault degrades to "this batch missed": HiCache recomputes, which is
    always correct -- KV that was never delivered cannot be wrong KV. ``on_error``
    builds that miss from the same arguments the method received, because a
    caller reads the shape of the result, not just its truthiness.

    Deliberately not applied to ``close()`` or ``register_mem_pool_host()``:
    those run on the scheduler thread during setup and teardown, where an
    exception is visible and worth surfacing rather than swallowing.
    """

    def decorate(method):
        @functools.wraps(method)
        def guarded(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except Exception:
                self._note_fault(method.__name__)
                return on_error(self, *args, **kwargs)

        return guarded

    return decorate


def _miss_per_transfer(self, transfers, *_args, **_kwargs) -> Dict[str, List[bool]]:
    """Every page of every transfer failed, keyed as the v2 callers expect."""
    return {str(t.name): [False] * len(t.keys or []) for t in transfers}


def _miss_per_key(self, keys, *_args, **_kwargs) -> List[bool]:
    return [False] * len(keys)


def _no_prefix(self, *_args, **_kwargs) -> PoolTransferResult:
    return PoolTransferResult.empty()


def _split_control_endpoint(endpoint: str) -> Optional[Tuple[str, str, int]]:
    """``(scheme_and_host, host, port)`` for a well-formed control endpoint.

    Splits on the *last* colon so a bracketed IPv6 literal
    (``tcp://[fd00::1]:25000``), whose address contains colons of its own,
    survives intact. Returns None for anything the control channel should not
    be handed: only ``tcp://`` is a ZMQ transport a peer can be dialed over,
    and only a real port number names a peer rather than a guess.
    """
    prefix, sep, port = endpoint.rpartition(":")
    if not sep or not port.isdigit():
        return None
    port_num = int(port)
    if not 1 <= port_num <= MAX_TCP_PORT:
        return None
    if not prefix.startswith(_CONTROL_SCHEME):
        return None
    host = prefix[len(_CONTROL_SCHEME) :]
    if not host or host in _UNDIALABLE_HINT_HOSTS:
        return None
    return prefix, host.strip("[]"), port_num


def _offset_endpoint_port(endpoint: str, offset: int) -> Optional[str]:
    """A validated ``tcp://host:port`` with ``offset`` added, or None."""
    split = _split_control_endpoint(endpoint)
    if split is None:
        return None
    prefix, _host, port = split
    if port + offset > MAX_TCP_PORT:
        return None
    return f"{prefix}:{port + offset}"


def _reject_unaddressable_parallelism(storage_config: HiCacheStorageConfig) -> None:
    """Refuse the parallel layouts whose pages this backend cannot tell apart.

    A KVCR block key is ``sha256(token ids)#<segment>``: it names the tokens and
    nothing about which slice of the model produced the bytes. So every rank
    coordinate that changes a page's *contents* has to be separated some other
    way, or two ranks holding different bytes agree on a key and a fetch returns
    the wrong KV with no error anywhere. ``_rank_port_offset`` separates
    ``(dp, attn_cp, attn_tp)`` by giving each rank its own control port and
    realigning every incoming hint onto it; the two below have no such
    separation. Mooncake, which keys into a shared store rather than a per-rank
    one, instead folds both into its key suffixes.

    - Pipeline parallelism: ``pp_rank`` is absent from the port offset, so
      ``pp0/tp0`` and ``pp1/tp0`` derive the same port. The loser of that bind
      is invisible (see :meth:`KVCRStore._control_port`), and a hint aimed at
      one rank reaches the other, filling the second half of the layers with
      the first half's KV.
    - Heterogeneous TP: ``should_split_heads`` says the deployment expects
      per-head-slice keys so a tp4 and a tp8 peer can share a prefix. This
      backend emits one key per page either way, so those peers agree on a key
      while holding different head slices.
    """
    if storage_config.pp_size > 1:
        raise RuntimeError(
            "KVCRStore does not support pipeline parallelism (pp_size="
            f"{storage_config.pp_size}). KVCR block keys carry no pp_rank, so "
            "two pipeline stages of one engine would share both a control port "
            "and a key namespace, and a peer fetch would return another stage's "
            "layers. Run with pp_size=1, or use a backend that namespaces keys "
            "per stage (e.g. mooncake)."
        )
    if storage_config.should_split_heads:
        raise RuntimeError(
            "KVCRStore does not support heterogeneous TP (tp_lcm_size="
            f"{storage_config.tp_lcm_size} > tp_size={storage_config.tp_size}). "
            "Head splitting exists so peers at different TP degrees can share a "
            "prefix, but KVCR block keys are page-level and carry no head slice, "
            "so those peers would agree on a key while holding different heads. "
            "Drop tp_lcm_size from the backend extra config, or use a backend "
            "with split-head support (e.g. mooncake)."
        )


def _dp_stride(storage_config: HiCacheStorageConfig) -> int:
    """How many schedulers one attention-DP rank of this engine owns.

    Also the port stride between two DP ranks, which is what the dynamo side has
    to multiply the DP rank by when it advertises one source endpoint per rank.
    """
    return storage_config.attn_cp_size * storage_config.tp_size


def _within_dp_offset(storage_config: HiCacheStorageConfig) -> int:
    """This rank's port offset *inside* its attention-DP group.

    Within one DP group, attention shards along ``(attn_cp, attn_tp)``, and a
    peer's shard is interchangeable with ours only when both coordinates match.
    So this is also the offset to apply to a source endpoint that the router has
    already resolved to the right DP rank.
    """
    return storage_config.attn_cp_rank * storage_config.tp_size + storage_config.tp_rank


def _rank_port_offset(storage_config: HiCacheStorageConfig) -> int:
    """This scheduler's port offset from the configured base port.

    Every ``(dp, attn_cp, attn_tp)`` rank of one engine runs its own KVCRStore in
    its own process on the same host, all reading the same ``extra_config``, so
    the offset has to be the full rank coordinate: ``tp_rank`` alone repeats once
    per DP group, and two ranks that pick the same port is invisible from the
    outside (see ``_control_port``).

    Both branches compute the same thing -- this scheduler's engine-global TP
    rank -- from whichever coordinates the config carries. With attention DP on,
    ``tp_rank``/``tp_size`` are attention-scoped (``cache_controller``
    substitutes ``attn_tp_*``), and SGLang lays ranks out as
    ``tp_rank = (dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size +
    attn_tp_rank`` (``compute_dp_attention_world_info``), which is exactly what
    is reassembled here. With it off, ``tp_rank`` already spans every scheduler
    of the engine, and ``dp_rank`` is 0, so the offset stays byte-identical to
    the TP-only behaviour that was validated on hardware.
    """
    if storage_config.dp_size <= 1:
        return storage_config.tp_rank
    return storage_config.dp_rank * _dp_stride(storage_config) + _within_dp_offset(
        storage_config
    )


def _highest_rank_port_offset(storage_config: HiCacheStorageConfig) -> int:
    """The largest offset ``_rank_port_offset`` can return for this engine.

    Mirrors ``_rank_port_offset`` with every rank coordinate at its maximum, so
    the two must be edited together.
    """
    if storage_config.dp_size <= 1:
        return storage_config.tp_size - 1
    return storage_config.dp_size * _dp_stride(storage_config) - 1


def _ephemeral_port() -> int:
    """Reserve an OS-assigned free TCP port and return it.

    There is an inherent bind-then-rebind race, but KVCR's control channel and
    NIXL listener are the only consumers and both bind immediately afterwards.
    """
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class KVCRStore(HiCacheStorage):
    """HiCacheStorage backend backed by the KVCR P2P coordinator (draft)."""

    # Each rank runs its own core over its own local DRAM tier, so a page one
    # rank deposits is invisible to the others.
    rank_local_namespace = True

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        mem_pool: Optional[HostKVCache] = None,
    ) -> None:
        _reject_unaddressable_parallelism(storage_config)
        self._storage_config = storage_config
        self._config = KVCRBackendConfig.from_extra_config(storage_config.extra_config)
        self.mem_pool_host = mem_pool

        # A per-worker unique NIXL agent name and control endpoint. Colocated
        # workers must not collide, so include the rank coordinate + a uuid. The
        # uuid also keeps a restarted rank from reusing a name its peers still
        # have in their remote-agent tables.
        self._agent_name = (
            f"kvcr-sgl-dp{storage_config.dp_rank}"
            f"-tp{storage_config.tp_rank}-{uuid.uuid4().hex[:8]}"
        )
        self._pinning = NoFrameworkPinning()
        self._key_adapter = StrKeyAdapter()

        # The core is built lazily -- see _ensure_kvcr. Registration only
        # records and validates layouts.
        self._kvcr: Optional[KVCR] = None
        self._control: Optional[ZmqPeerControlChannel] = None
        # One _PoolLayout per registered sglang host pool, keyed by pool name,
        # in registration order (which puts KV first, since the anchor
        # registers before any sidecar). Insertion order is what pins the
        # KVCR pool_layouts / LocalDramOptions.pools ordering the core
        # requires to agree -- see _local_dram_region.
        self._pool_layouts: Dict[str, _PoolLayout] = {}
        # Serializes the lazy build against the several threads that can be
        # first to a data-path entry point (prefetch daemon, scheduler tick).
        self._build_lock = threading.RLock()
        # Why the core is absent, since absent no longer means "not built yet":
        # a build that already failed (retrying repeats one traceback per
        # transfer), and a close that already ran (rebuilding during teardown
        # would stand up a fresh NIXL agent and rebind the control port).
        self._build_failed = False
        self._closed = False
        # Completions drained from poll_completed() that belong to an op other
        # than the one currently being waited on. poll_completed() clears the
        # core's queue, so a result seen by the wrong waiter would be lost
        # without this stash. Guarded by _poll_lock -- the scheduler tick drains
        # the same queue, so it can be the one to observe the completion of a get.
        self._completed_ops: Dict[int, Dict] = {}
        # Handles a _drain_until is currently blocked on. A completion for
        # anything else is dropped rather than stashed.
        #
        # Tracking live waiters is what bounds this: the obvious alternative --
        # a set of *abandoned* handles, pruned when the late result shows up --
        # assumes every op eventually reports, and one class of them never does.
        # kvcr.abort() is a no-op stub (core.py returns False with a TODO), so a
        # timed-out op stays in flight; a remote deliver whose source went silent
        # parks in KVCR's WAITING_TERMINAL state, which is only left by a
        # write_done notification that a dead peer never sends. Measured against
        # the real core: 6/6 hinted delivers at a dead source never reported.
        # Keyed on abandoned handles, each of those leaves an entry behind for
        # the life of the scheduler; keyed on live waiters, the set is bounded by
        # concurrency and a never-reporting op costs nothing here.
        self._waiting_ops: Set[int] = set()
        # Handles a _drain_until gave up on. Kept only so a completion that
        # arrives afterwards can be reported as the hazard it is rather than
        # dropped as an ordinary late tick (see _poll_once). Bounded by the
        # deque, because the ops that never report would otherwise accumulate
        # for the life of the scheduler -- the same reason _waiting_ops keys on
        # live waiters. Old entries fall out and degrade to the previous
        # behaviour, which is the right way to lose this signal.
        self._abandoned_ops: Deque[int] = deque(maxlen=_ABANDONED_OP_HISTORY)
        # Serializes poll_completed() between the prefetch thread (_drain_until)
        # and the scheduler thread (tick). poll_completed() both drains a queue
        # and advances core state machines, so two callers must not interleave.
        # Also fences close() against both -- see close().
        self._poll_lock = threading.Lock()
        # Source of request ids for the core's hint table; see
        # _hint_request_id. Locked because HiCache runs one prefetch thread but
        # the v1 entry points are reachable from the scheduler thread too.
        self._hint_id_lock = threading.Lock()
        self._next_hint_id = 0
        # Remote-path counters, logged periodically by _note. Without them a
        # hinted get that returns nothing is indistinguishable from a request
        # the router never attached a hint to -- the two have opposite causes
        # (a broken fetch here vs. an index miss upstream) and the backend is
        # the only place that can tell them apart. Guarded by _stats_lock
        # because the v2 entry points run on the prefetch thread while the
        # scheduler tick touches the same counters.
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, int] = defaultdict(int)
        self._next_stats_log_at = 0.0
        # Separate clock from the stats line: a fault is rarer and carries a
        # traceback, so throttling the two together would let a chatty stats
        # interval swallow the first report of a fault.
        self._next_fault_log_at = 0.0

        if mem_pool is not None:
            self.register_mem_pool_host(mem_pool)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register_mem_pool_host(self, mem_pool_host: HostKVCache) -> None:
        """Register the anchor pool: the KV pool of a hybrid stack, or the only one.

        A hybrid stack passes the ``HostPoolGroup``'s anchor here and each pool
        (including this same KV one) again through ``register_mem_host_pool_v2``,
        so registering under ``PoolName.KV`` is what makes the two spellings land
        on one layout rather than two.
        """
        super().register_mem_pool_host(mem_pool_host)
        self._register_pool(PoolName.KV, self._anchor_pool(mem_pool_host))

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name) -> None:
        """Register one pool of a hybrid stack, KV or sidecar.

        Every pool is probed here, at startup, rather than on first use: a
        layout this backend cannot address has to fail the launch. Scoring it a
        miss later would be worse than it sounds -- the pool would simply never
        be written, and a model attending over a sidecar page that was never
        filled produces wrong output with no error anywhere.
        """
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        self._register_pool(host_pool_name, host_pool)

    @staticmethod
    def _anchor_pool(mem_pool_host: HostKVCache) -> HostKVCache:
        """The pool whose buffers and page meta the KV path addresses.

        A hybrid stack hands over a ``HostPoolGroup``, which forwards
        ``get_page_buffer_meta`` to its anchor; resolving to the anchor itself
        means the KV layout is read off the same object the sidecars are, so
        all of them are addressed the one way.
        """
        if isinstance(mem_pool_host, HostPoolGroup):
            return mem_pool_host.anchor_entry.host_pool
        return mem_pool_host

    def _register_pool(self, pool_name, host_pool: HostKVCache) -> None:
        """Probe ``host_pool``'s page layout and record it, or refuse to start.

        Idempotent per pool name, because the anchor arrives twice (see
        ``register_mem_pool_host``). Re-registering a *different* pool under a
        name already taken is a bug in the caller, not a layout we can serve, so
        it raises rather than silently keeping either one.

        Registering after the core exists also raises. The alternative --
        serving the pools we have and ignoring the late one -- is the silent
        wrong-output case above; rebuilding the core instead is not available
        either, since closing it tears down the progress thread that owns the
        bound ZMQ control port and the rebind races every peer already dialing
        it. In practice nothing reaches this: every pool registers during
        engine construction, and the build is deferred past all of it.
        """
        name = str(pool_name)
        existing = self._pool_layouts.get(name)
        if existing is not None:
            if existing.host_pool is not host_pool:
                raise RuntimeError(
                    f"KVCRStore: two different host pools registered as "
                    f"'{name}'; KVCR block keys carry the pool name, so the "
                    "second would overwrite the first's blocks."
                )
            return
        if self._kvcr is not None:
            raise RuntimeError(
                f"KVCRStore: host pool '{name}' registered after the KVCR core "
                "was built, so its pages have no slots and would never be "
                "stored. The core's control port cannot be rebound to pick it "
                "up; this is a registration-order bug."
            )
        self._pool_layouts[name] = self._probe_pool_layout(name, host_pool)

    def _control_port(self) -> int:
        """Bind port for this rank's KVCR control channel.

        Every scheduler of one engine runs its own KVCRStore in its own process
        on the same host, all reading the same ``extra_config``. A configured
        port is therefore a *base* port that must be offset by rank -- without
        the offset, rank 1 binds the port rank 0 already holds. That failure is
        invisible from the outside: ``ZmqPeerControlChannel`` binds from the
        progress thread, so the engine still starts, still registers, and still
        advertises an endpoint; only peer fetches to the losing rank break.
        ``_rank_port_offset`` covers the whole rank coordinate, so DP ranks get
        disjoint port blocks rather than colliding on the same ``base +
        tp_rank``.

        Port 0 means "ask the OS", which already guarantees distinctness -- and
        offsetting an OS-assigned port would land on an arbitrary port belonging
        to someone else, so the offset applies only to configured ports. It is
        reachable only local-only: ``KVCRBackendConfig`` refuses port 0 together
        with ``enable_remote_hint``, because an OS-assigned port is known only
        inside this process and so cannot be registered for peers to dial.

        A base port near the top of the range would push high ranks past 65535,
        where ``bind`` fails on a rank the operator never named. The whole block
        is checked here rather than just this rank's port, so every rank of the
        engine fails at startup with the same message instead of the low ranks
        coming up and the high ones dying.
        """
        configured = int(self._config.control_port)
        if configured <= 0:
            return _ephemeral_port()
        offset = _rank_port_offset(self._storage_config)
        highest = configured + _highest_rank_port_offset(self._storage_config)
        if highest > MAX_TCP_PORT:
            raise ValueError(
                f"KVCR control_port {configured} leaves no room for this "
                f"engine's {highest - configured + 1} schedulers: the highest "
                f"rank would bind {highest}, above {MAX_TCP_PORT}. Lower "
                "control_port in --hicache-storage-backend-extra-config."
            )
        return configured + offset

    def _ensure_kvcr(self) -> Optional[KVCR]:
        """Build the core on first use, once every host pool has registered.

        Deferred rather than built in ``register_mem_pool_host`` because that
        call is not the end of registration: ``attach_storage_backend`` hands
        over the anchor first and each pool of the group afterwards, and a
        draft sidecar arrives later still, from ``maybe_register_hicache_draft``
        during cache construction. Building at the anchor would therefore size
        the local tier for the KV pool alone and leave every sidecar unstorable
        -- and a sidecar that is never written is the silent wrong-output case
        ``_register_pool`` exists to prevent.

        Every seam that can be first is a build point: the data-path entry
        points, and ``tick``, which the scheduler loop starts calling only after
        the caches are built. Under the lock because those run on different
        threads (HiCache's prefetch daemon and the scheduler).

        A failed build is remembered, not retried. It fails on pool layout,
        which does not change, so retrying would repeat one traceback per
        transfer for the life of the process.
        """
        if self._kvcr is not None:
            return self._kvcr
        with self._build_lock:
            if self._kvcr is not None or self._build_failed or self._closed:
                return self._kvcr
            if not self._pool_layouts:
                return None
            try:
                self._build_kvcr()
            except Exception:
                self._build_failed = True
                raise
            return self._kvcr

    def _build_kvcr(self) -> None:
        _require_kvcr_api()
        self._adopt_anonymous_pool_name()
        framework_regions = self._framework_regions()
        local_dram = self._local_dram_region()

        advertise = self._config.control_advertise_host or socket.gethostname()
        self._control = ZmqPeerControlChannel(
            self._config.control_host,
            self._control_port(),
            advertise,
        )

        # Give the NIXL listen socket a distinct ephemeral port per worker.
        nixl_listen_port = _ephemeral_port()

        config = KVCRConfig(
            # One block size per pool: every descriptor this backend hands KVCR
            # is a single page component, and KVCR rejects any whose size does
            # not match the layout its pool was declared with. Same order as
            # local_dram.pools, which _LocalDram compares element-wise.
            pool_layouts=[
                (name, slot_size) for name, slot_size, _slots in self._kvcr_pools()
            ],
            nixl_agent_name=self._agent_name,
            enable_telemetry=self._config.enable_telemetry,
            operation_timeout_ms=self._config.operation_timeout_ms,
            abandon_timeout_ms=self._config.abandon_timeout_ms,
            nixl_listen_port=nixl_listen_port,
        )
        bindings = KVCRBindings(
            request_pin=self._pinning.request_pin,
            poll_pin_results=self._pinning.poll_pin_results,
            release_pin=self._pinning.release_pin,
            cancel_pin_request=self._pinning.cancel_pin_request,
            framework_control=self._control,
            key_adapter=self._key_adapter,
            policy=_resolve_policy(self._config.policy),
        )
        # eager_ctrl_connect / opportunistic_query / metadata_retry moved out of
        # KVCRConfig into the remote-forward-DRAM options in the wheel core.
        backend_configs = KVCRBackendConfigs(
            local_dram=local_dram,
            remote_fw_dram=RemoteFWDramOptions(
                eager_ctrl_connect=self._config.eager_ctrl_connect,
                opportunistic_query=self._config.opportunistic_query,
                metadata_retry_interval_ms=self._config.metadata_retry_interval_ms,
            ),
            **framework_regions,
        )
        self._kvcr = KVCR(config, bindings, backend_configs)
        logger.info(
            "KVCRStore initialized (agent=%s, pools=%s, remote_hint=%s, policy=%s)",
            self._agent_name,
            [
                (name, slot_size, slots)
                for name, slot_size, slots in self._kvcr_pools()
            ],
            self._config.enable_remote_hint,
            self._config.policy,
        )

    # ------------------------------------------------------------------
    # Scheduler-thread tick (source-side progress)
    # ------------------------------------------------------------------

    @_fail_closed(lambda self, *a, **kw: None)
    def tick(self) -> None:
        """Advance KVCR state even when this worker issues no traffic of its own.

        ``poll_completed()`` is what moves the core's state machines forward, and
        the only other caller is ``_drain_until`` -- which runs solely while
        *this* worker is doing a get or set. That is sufficient for the target
        side of a P2P fetch, but not for the source side: a peer's ``start_write``
        lands in the progress queue as a ``_SourcePinOp``, and until someone
        pumps, it is never pinned and never written. An otherwise idle worker
        would therefore serve nothing, and the requesting peer would sit until
        its deadline expired.

        So this tick is what makes a worker usable as a P2P *source*. It runs on
        the scheduler thread, from ``check_hicache_events``, once per loop
        iteration -- idle iterations included, which is exactly when a peer needs
        us. A pump thread would be the obvious alternative and was the previous
        implementation; it cost a wakeup per interval on a thread the GIL hands
        off to only when the scheduler releases it, and measured ~1.08s of pure
        wakeup delay inside a 1.15s source-side serve. Here the handoff count is
        zero.

        ``_fail_closed`` for the usual reason plus one specific to this seam:
        the scheduler thread has no handler above ``check_hicache_events``, so
        an exception here ends the engine rather than merely a storage thread.
        A tick that fails is a source-side serve that stalls; the next tick
        retries, and ``faults_tick`` is the counter that says how often.

        Cost when nothing is in flight is one lock acquire plus an empty queue
        check.
        """
        self._poll_once(self._ensure_kvcr())

    def idle_poll_timeout_ms(self) -> Optional[int]:
        """Cap the idle park so a peer's pull is not stalled behind it.

        ``--sleep-on-idle`` parks the loop in a 1s poll on the request sockets.
        A peer's pull wakes neither of them, so under it a source-side serve
        would advance once per second. Capping the park at the tick interval
        bounds that to one interval instead.

        Not "skip the park entirely", which is what this returned first. A
        scheduler loop that never parks holds the GIL and spins; the threads it
        starves are the ones driving the very transfer it is spinning for --
        kvcr's progress thread and HiCache's prefetch daemon are both in this
        process. Measured on the target side of a hinted fetch: the scheduler
        thread ticked ~10k times a second while its own GIL canary was
        scheduled 45 times to a sibling process's 5935, and ``remote_deliver``
        took 3.9s against a 2s budget. The park is what yields.
        """
        # Registration, not a built core: the core is built lazily by the first
        # tick, and that tick only happens if the park is capped first.
        if not self._config.enable_remote_hint or not self._pool_layouts:
            return None
        return _IDLE_TICK_INTERVAL_MS

    def _poll_once(self, kvcr: Optional[KVCR]) -> None:
        """Drain one round of completions, stashing them for their waiters.

        Both ``tick`` and ``_drain_until`` call this. Whoever gets there first
        drains the queue, so every result must be stashed rather than assumed to
        belong to the current caller -- except results for ops nobody is waiting
        on any more, which are dropped (see ``_waiting_ops``).

        The caller's core is re-checked against the live one under the lock:
        ``close()`` clears the field while holding it, so a caller that read the
        core just before would otherwise poll one that is being torn down.
        """
        with self._poll_lock:
            if kvcr is None or kvcr is not self._kvcr:
                return
            for done_handle, entries in kvcr.poll_completed():
                if done_handle not in self._waiting_ops:
                    if done_handle in self._abandoned_ops:
                        # The op we gave up on was still live afterwards, so its
                        # transfers were in flight while HiCache owned the pages
                        # again. Nothing here can undo that; naming it is the
                        # only way an operator learns the hazard fired at all.
                        self._note("abandoned_op_reported_late")
                        logger.warning(
                            "KVCRStore: abandoned op %s reported after its "
                            "deadline; its transfers outlived the host pages "
                            "HiCache reclaimed. Raise get_timeout_s.",
                            done_handle,
                        )
                    else:
                        self._note("late_completions_dropped")
                    continue
                self._completed_ops[done_handle] = entries

    def _probe_pool_buffers(self, name: str, host_pool: HostKVCache) -> None:
        """Refuse a pool whose pages live in no registrable host tensor.

        The build reads these again (``_host_buffers``), but the build is lazy
        and this has to fail the *launch*: a pool whose buffers NIXL cannot
        register is addressable by neither direction, and the alternative to a
        failed launch is a server that comes up reporting every page a miss
        while every peer serve silently has nothing to offer.
        """
        buffers = [
            buffer
            for buffer in host_pool.get_hybrid_pool_buffer() or []
            if isinstance(buffer, torch.Tensor) and buffer.numel() > 0
        ]
        if not buffers:
            raise RuntimeError(
                f"KVCRStore: host pool '{name}' exposes no host tensor to "
                "register with NIXL, so neither a deposit nor a deliver can "
                "name its pages. This backend cannot run against it."
            )

    def _host_buffers(self) -> List[torch.Tensor]:
        """Every engine-owned host allocation the KV path addresses, deduplicated.

        Registering these with NIXL is what lets either direction name a host
        page in a descriptor, and NIXL rejects descriptors outside a registered
        region. Both directions need it, including the ones that never leave
        this machine: ``deposit`` hands the core host pages as transfer
        *sources*, ``deliver`` hands it host pages as *destinations*, and
        KVCR's local tier moves both by submitting a transfer addressed to its
        own agent rather than by memcpy.

        ``get_hybrid_pool_buffer`` is the pools' own answer to "which tensors
        hold my pages", which is the same question Mooncake's
        ``_iter_host_pool_buffers`` asks -- including for the layouts that keep
        ``kv_buffer`` as a per-layer list, where it is a list of tensors rather
        than one. Deduplicated by storage identity because a hybrid stack
        registers its anchor under two names, and two views of one allocation
        would otherwise be registered twice.
        """
        buffers: List[torch.Tensor] = []
        seen: Set[Tuple[int, int]] = set()
        for layout in self._pool_layouts.values():
            # A logical anchor's pages live in the marker buffer this backend
            # allocated for it, not in the pool -- which owns no bytes at all.
            pool_buffers = (
                [layout.marker_buffer]
                if layout.is_logical_anchor
                else layout.host_pool.get_hybrid_pool_buffer() or []
            )
            for buffer in pool_buffers:
                if not isinstance(buffer, torch.Tensor) or buffer.numel() == 0:
                    continue
                identity = (buffer.data_ptr(), buffer.numel() * buffer.element_size())
                if identity in seen:
                    continue
                seen.add(identity)
                buffers.append(buffer)
        return buffers

    def _framework_regions(self) -> Dict[str, object]:
        """The ``KVCRBackendConfigs`` framework-endpoint kwargs for this stack.

        One buffer keeps the legacy ``framework_dram`` spelling, so a KV-only
        stack builds the same config it always did and runs against an
        nvidia-kvcr without the typed list. More than one needs
        ``framework_regions``, which is where multi-allocation support landed;
        without it there is nothing to degrade to, since an unregistered
        sidecar buffer makes every descriptor into it invalid.
        """
        buffers = self._host_buffers()
        if not buffers:
            raise RuntimeError(
                "KVCRStore: no host pool exposed a buffer to register with "
                f"NIXL (pools={list(self._pool_layouts)}). Both directions of "
                "the KV path address those buffers, so the backend cannot run "
                "against this pool layout."
            )
        if len(buffers) == 1:
            buffer = buffers[0]
            return {
                "framework_dram": FrameworkDramInput(
                    address=buffer.data_ptr(),
                    length=buffer.numel() * buffer.element_size(),
                )
            }
        if not _HAS_FRAMEWORK_REGIONS:
            raise RuntimeError(
                f"KVCRStore: this model's host pools ({list(self._pool_layouts)}) "
                f"are {len(buffers)} separate allocations, but the installed "
                "nvidia-kvcr registers only a single framework_dram region. "
                "Upgrade to an nvidia-kvcr carrying framework_regions, or run "
                "without --hicache-storage-backend kvcr."
            )
        return {
            "framework_regions": tuple(
                FrameworkMemoryRegion(
                    address=buffer.data_ptr(),
                    length=buffer.numel() * buffer.element_size(),
                    mem_type="DRAM",
                    device_id=0,
                    # Hand the tensor over as the owner so kvcr holds it for as
                    # long as the registration lives. The single-region path
                    # has no such field, which is why this backend keeps its
                    # own reference to the local-tier buffer below.
                    owner=buffer,
                )
                for buffer in buffers
            )
        }

    def _adopt_anonymous_pool_name(self) -> None:
        """Rename the sole KVCR pool to the empty name, where that is legal.

        kvcr allows the empty pool name only while it is the only name
        (``config._validate_pool_layouts``), and the single-pool backend that
        preceded this one always used it. Taking it back for a KV-only stack
        with uniform components keeps descriptors and block keys byte-identical
        to what that backend wrote, so a peer running it still decodes ours --
        ``remote_fw_dram.start_write`` drops any key whose source and
        destination ``info`` lists differ, without an error.

        Decided here rather than at registration because it depends on the
        *whole* set of pools, and a sidecar can register after KV does.
        """
        names = {
            pool_name
            for layout in self._pool_layouts.values()
            for pool_name in layout.kvcr_pool_names
        }
        if len(names) != 1:
            return
        self._pool_layouts = {
            name: msgspec.structs.replace(
                layout,
                kvcr_pool_names=(_ANONYMOUS_POOL_NAME,) * layout.segments_per_page,
            )
            for name, layout in self._pool_layouts.items()
        }

    def _kvcr_pools(self) -> List[Tuple[str, int, int]]:
        """``(kvcr_pool_name, slot_size, slots)`` for every KVCR local-DRAM pool.

        Capacity is split at equal *page depth* rather than equal bytes: a
        deposit writes one page's components across the pools in one op and a
        page is only stored when all of them land, so a pool holding fewer
        pages than its siblings caps the tier at its own depth and wastes the
        rest. Depth is therefore the quantity to make uniform, and slot counts
        follow from how many components each pool takes per page.

        ``local_dram_slots``, when set, still means the KV pool's slot count as
        it always has, so an existing single-pool config sizes to exactly the
        same tier it did before.
        """
        components: Dict[str, int] = defaultdict(int)
        slot_sizes: Dict[str, int] = {}
        for layout in self._pool_layouts.values():
            for pool_name, size in zip(layout.kvcr_pool_names, layout.segment_sizes):
                components[pool_name] += 1
                slot_sizes[pool_name] = size
        bytes_per_page = sum(
            slot_sizes[name] * count for name, count in components.items()
        )
        pages = self._local_dram_pages(bytes_per_page, components, slot_sizes)
        return [
            (name, slot_sizes[name], pages * count) for name, count in components.items()
        ]

    def _local_dram_pages(
        self,
        bytes_per_page: int,
        components: Dict[str, int],
        slot_sizes: Dict[str, int],
    ) -> int:
        """How many pages the local DRAM tier holds, from either sizing knob."""
        configured_slots = self._config.local_dram_slots
        if configured_slots > 0:
            kv_layout = self._pool_layouts.get(str(PoolName.KV))
            per_page = kv_layout.segments_per_page if kv_layout is not None else 1
            return max(1, configured_slots // per_page)
        pages = self._config.local_dram_bytes // bytes_per_page
        if pages < 1:
            raise RuntimeError(
                f"KVCRStore: local_dram_bytes={self._config.local_dram_bytes} "
                f"is below one page across this model's pools "
                f"({bytes_per_page} bytes: "
                f"{ {name: (slot_sizes[name], count) for name, count in components.items()} }"
                "). Raise local_dram_bytes."
            )
        return pages

    def _local_dram_region(self) -> LocalDramOptions:
        """Allocate KVCR's own local DRAM tier (the buffer-only L3 pool).

        One slot holds one page *component*, so a pool's slot size is its
        components' shared byte size -- see ``_probe_pool_layout``. deposit()
        copies each component into exactly one slot.

        One allocation carved into per-pool sub-ranges, rather than one tensor
        per pool: kvcr only requires the ranges not to overlap, and a single
        buffer keeps the whole tier inside one NIXL registration. Sub-ranges
        are packed, not padded: each is a whole number of its own slots, so a
        slot never straddles the boundary, and the tier is the one allocation
        whose alignment nothing downstream reads.
        """
        offsets: List[Tuple[str, int, int]] = []
        total = 0
        for name, slot_size, slots in self._kvcr_pools():
            length = slot_size * slots
            offsets.append((name, total, length))
            total += length

        # Anchor a contiguous host buffer for the slots and keep a reference so
        # it is not garbage-collected while NIXL has it registered.
        self._local_dram_buffer = torch.empty(total, dtype=torch.uint8)
        base = self._local_dram_buffer.data_ptr()
        # Same pool order as KVCRConfig.pool_layouts: _LocalDram compares the
        # two name lists element-wise and refuses a permutation.
        return LocalDramOptions(
            [(name, base + offset, length) for name, offset, length in offsets]
        )

    def _probe_pool_layout(self, name: str, host_pool: HostKVCache) -> _PoolLayout:
        """Learn one pool's page component sizes, or refuse to start.

        A host page is split into several non-contiguous components (MHA: K/V
        halves, times per-layer runs in ``layer_first``; Mamba: one temporal
        plus one per conv state, all of different sizes). Rather than re-derive
        that layout arithmetic per pool class, ask the pool's own zero-copy
        accessor (``get_page_buffer_meta``) for one page and read the component
        sizes straight off it -- the same source ``_host_descriptors`` reads at
        transfer time, so the two cannot disagree about a pool's shape.

        Raises rather than returning None. The local DRAM tier is this backend's
        only storage, so a pool it cannot size has no degraded mode: it would
        fail every deposit for that pool individually, which reads downstream as
        "the cache never hits" rather than "the backend is unusable".
        """
        if self._is_logical_anchor(host_pool):
            return self._logical_anchor_layout(name, host_pool)
        sizes = self._probe_page_component_sizes(name, host_pool)
        self._probe_pool_buffers(name, host_pool)
        uniform = len(set(sizes)) == 1
        pool_names = tuple(_kvcr_pool_name(name, size, uniform) for size in sizes)
        return _PoolLayout(
            name=name,
            host_pool=host_pool,
            segment_sizes=tuple(sizes),
            kvcr_pool_names=pool_names,
            # KV keeps the bare "<page>#<seg>" spelling for the same wire-
            # compatibility reason; only a sidecar needs the pool component,
            # and only to keep its segments from colliding with the KV page's
            # under the same page hash.
            key_prefix="" if name == str(PoolName.KV) else f"{name}:",
        )

    @staticmethod
    def _is_logical_anchor(host_pool: HostKVCache) -> bool:
        """Whether this pool anchors pages for sidecars and holds no bytes itself.

        DeepSeek V4's KV pool is a ``LogicalHostPool``: it allocates page-aligned
        token slots that the compressed side pools use as stable page anchors,
        but it owns no KV tensor, so ``kv_buffer`` is None and
        ``get_page_buffer_meta`` returns None by design. Every other pool this
        backend serves answers both.
        """
        return getattr(host_pool, "kv_buffer", None) is None

    def _logical_anchor_layout(self, name: str, host_pool: HostKVCache) -> _PoolLayout:
        """Layout for a bytes-free anchor, backed by a marker block per page.

        The anchor cannot simply be skipped. ``_page_transfer_sidecar`` runs a
        sidecar's IO only when the KV pool reported every page of the operation
        complete, so an anchor that always misses would leave DeepSeek V4's real
        KV -- which lives entirely in the sidecars -- never stored and never
        loaded, with no error anywhere.

        So the anchor gets one small block per page, the same shape every other
        pool has: one segment, deposited and delivered like any other, which
        makes its key a real residency record in the core rather than a special
        case the query path has to know about. The bytes are a constant marker;
        only the key's presence carries meaning.

        The marker buffer is one tensor for the whole pool with a distinct
        per-page slot, not one shared slot: deposit and deliver hand KVCR
        descriptors into it, and a remote deliver writes through them, so two
        pages sharing an address would have concurrent transfers writing the
        same bytes. The buffer is held on the layout so NIXL's registration
        stays valid for as long as the layout does.
        """
        page_size = getattr(host_pool, "page_size", None)
        if not page_size:
            raise RuntimeError(
                f"KVCRStore: logical anchor pool '{name}' reports no page size, "
                "so its pages cannot be given marker blocks."
            )
        pages = int(getattr(host_pool, "size", 0)) // int(page_size)
        if pages < 1:
            raise RuntimeError(
                f"KVCRStore: logical anchor pool '{name}' holds no whole page "
                f"(size={getattr(host_pool, 'size', 0)}, page_size={page_size})."
            )
        marker = torch.empty(pages, _ANCHOR_MARKER_BYTES, dtype=torch.uint8)
        # Deposited as-is, so it must be initialized: uninitialized bytes would
        # be read by NIXL and, on a peer fetch, shipped over the wire.
        marker.fill_(1)
        return _PoolLayout(
            name=name,
            host_pool=host_pool,
            segment_sizes=(_ANCHOR_MARKER_BYTES,),
            kvcr_pool_names=(_kvcr_pool_name(name, _ANCHOR_MARKER_BYTES, True),),
            key_prefix="" if name == str(PoolName.KV) else f"{name}:",
            marker_buffer=marker,
        )

    def _probe_page_component_sizes(
        self, name: str, host_pool: HostKVCache
    ) -> List[int]:
        """One page's component byte sizes, in ``get_page_buffer_meta`` order."""
        page_size = getattr(host_pool, "page_size", None)
        if not page_size:
            raise RuntimeError(
                f"KVCRStore: host pool '{name}' reports no page size, so its "
                "pages cannot be split into KVCR blocks."
            )
        probe_indices = torch.arange(int(page_size), dtype=torch.int64)
        try:
            meta = host_pool.get_page_buffer_meta(probe_indices)
        except Exception as err:
            raise RuntimeError(
                f"KVCRStore: page-layout probe failed for host pool '{name}'. "
                "KVCR addresses host pages through get_page_buffer_meta, so a "
                "pool that cannot describe one cannot be served."
            ) from err
        # Pools with no zero-copy support return None rather than a pair.
        if meta is None:
            raise RuntimeError(
                f"KVCRStore: host pool '{name}' has no zero-copy page meta. "
                "KVCR moves pages by NIXL descriptor, so there is no byte-copy "
                "path to fall back to."
            )
        _ptr_list, size_list = meta
        sizes = [int(size) for size in size_list or []]
        if not sizes or any(size <= 0 for size in sizes):
            raise RuntimeError(
                f"KVCRStore: host pool '{name}' described a page as {sizes}; "
                "every component must be a positive number of bytes for the "
                "local DRAM tier to be sized."
            )
        return sizes

    def _locally_resident(self, segment_keys: List[BlockKey]) -> bool:
        """True iff KVCR's local DRAM tier holds every segment of a page.

        ``query`` is KVCR's own residency table, which is the only copy of that
        state: it moves keys to FILLING on deposit, to HIT on fill completion,
        and drops them on eviction, all inside the core. Mirroring it into a
        dict here would be a second copy that eviction can silently desync --
        and a stale "resident" answer is not a miss, it is a page we promise
        HiCache and then fail to deliver.

        Passing no request_id keeps this to residency only: a hint-covered key
        would otherwise report FETCHABLE, and the remote branch is the caller's
        to decide (see ``batch_exists_v2``).
        """
        return all(
            status is QueryStatus.HIT
            for status, _tier in self._kvcr.query(segment_keys)
        )

    def close(self) -> None:
        """Retire the core from every poller, then close it -- not the reverse.

        ``poll_completed()`` walks core state the core's own ``close()`` tears
        down (it closes the progress thread and the local tier), so a poller
        still inside one when the core goes away is a use-after-free on the KVCR
        side, not a benign late tick. Two pollers reach it: ``tick`` on the
        scheduler thread and ``_drain_until`` on HiCache's prefetch daemon.

        Dropping the reference under ``_poll_lock`` is what fences them. Holding
        the lock means no poll is in progress; clearing the field means none
        starts, because both pollers read it and bail on None. Close the core
        only after that, using the reference we took.

        The core's own ``close()`` makes a related trade one level down: when
        its progress loop does not go quiescent it keeps the backend resources
        and raises, precisely so nothing unmaps memory a native transfer still
        references. We put our reference back in that case for the same reason,
        and report rather than propagate -- ``close()`` is a teardown path, and
        the rule for this backend is that it never raises at a HiCache seam.
        """
        # Set before the core is dropped, and never cleared: the build is lazy
        # now, so without it a tick arriving after close would read the None
        # field as "not built yet" and stand a fresh core -- with a fresh NIXL
        # agent and a rebind of the control port -- during teardown.
        self._closed = True
        with self._poll_lock:
            kvcr = self._kvcr
            self._kvcr = None
        if kvcr is None:
            return
        try:
            kvcr.close()
        except BaseException:
            # Core-side close is idempotent, so keeping the reference costs
            # nothing and leaves a later attempt possible.
            with self._poll_lock:
                self._kvcr = kvcr
            logger.exception(
                "KVCRStore: KVCR core did not close cleanly; keeping the "
                "core so its still-registered memory is not unmapped."
            )

    # ------------------------------------------------------------------
    # v2 interface (the real HiCache path)
    # ------------------------------------------------------------------

    @_fail_closed(_miss_per_transfer)
    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        """Offload host pages into KVCR's local DRAM tier via deposit()."""
        results: Dict[str, List[bool]] = {}
        if self._ensure_kvcr() is None:
            return {str(t.name): [False] * len(t.keys or []) for t in transfers}
        for transfer in transfers:
            layout = self._layout_for(transfer)
            if layout is None:
                results[str(transfer.name)] = [False] * len(transfer.keys or [])
                continue
            results[str(transfer.name)] = self._deposit_transfer(transfer, layout)
        return results

    def _segment_key(self, layout: _PoolLayout, page_key: str, seg: int) -> BlockKey:
        """KVCR block identity for one component of one pool's host page.

        A page fans out into one KVCR block per component; the suffix keeps
        them distinct in the local tier, and ``key_prefix`` keeps two pools'
        components distinct under the same page hash (a sidecar transfer
        carries the KV page's hashes -- see ``indices_from_pool``).

        The page hash stays the leading ``#``-delimited component because that
        is what the router-hint path parses out (``router_hint.page_hash_int``,
        ``RouterHint.covers``), so a remote fetch still matches on it.
        """
        return _encode_key(f"{page_key}#{layout.key_prefix}{seg}")

    def _page_segment_keys(self, layout: _PoolLayout, page_key: str) -> List[BlockKey]:
        return [
            self._segment_key(layout, page_key, seg)
            for seg in range(layout.segments_per_page)
        ]

    def _layout_for(self, transfer: PoolTransfer) -> Optional[_PoolLayout]:
        """The registered layout serving ``transfer``, or None with a counter.

        ``_register_pool`` refuses an unservable pool at startup, so reaching
        None means a transfer arrived for a pool that never registered at all.
        Scoring it a miss is what keeps that fail-closed: the caller records 0
        pages for the pool and clamps the usable prefix to 0, so HiCache
        recomputes rather than reading pages this backend never wrote.
        """
        layout = self._pool_layouts.get(str(transfer.name))
        if layout is None:
            self._note(f"unregistered_pool_{transfer.name}")
        return layout

    def _deposit_transfer(
        self, transfer: PoolTransfer, layout: _PoolLayout
    ) -> List[bool]:
        keys = transfer.keys or []
        if not keys:
            return []
        # Build one source descriptor per (page, segment).
        built = self._host_descriptors(transfer, layout)
        if built is None:
            logger.warning(
                "KVCRStore deposit skipped: no host descriptors for %d pages",
                len(keys),
            )
            return [False] * len(keys)
        descriptors, per_page_keys = built

        op_handle, result_map = self._submit_and_wait(
            lambda: self._kvcr.deposit(descriptors)
        )
        missing = len(descriptors) - len(result_map)
        failed = sum(1 for ok in result_map.values() if not ok)
        if failed or missing:
            # HiCache only reports "N pages failed", which cannot distinguish a
            # rejected deposit from a segment KVCR never reported on at all.
            logger.warning(
                "KVCRStore deposit op=%s: %d/%d segments failed, %d unreported "
                "(pages=%d)",
                op_handle,
                failed,
                len(descriptors),
                missing,
                len(keys),
            )

        # A page is stored iff every one of its segments landed. Nothing is
        # recorded on our side: the copy is now in KVCR's own slots, and its
        # residency table is what ``_locally_resident`` and the source path both
        # read. ``descriptors`` names the *host* pages we copied out of, which
        # HiCache is free to reuse the moment this call returns.
        results = [
            all(result_map.get(seg_key, False) for seg_key in page_keys)
            for page_keys in per_page_keys
        ]
        # Counted because the first question about any missed P2P fetch is
        # whether the source ever held the blocks, and until now every counter
        # here was on the get side -- so a source that quietly stored nothing
        # looked exactly like a target that quietly fetched nothing.
        self._note("deposit_pages_offered", len(keys))
        self._note("deposit_pages_stored", sum(results))
        # Per pool as well as in total: a hybrid model deposits one page across
        # seven pools, and the totals cannot show which of them stopped storing.
        self._note(f"deposit_{transfer.name}", sum(results))
        return results

    def _host_descriptors(
        self, transfer: PoolTransfer, layout: _PoolLayout
    ) -> Optional[Tuple[Dict[BlockKey, List[MemDescriptor]], List[List[BlockKey]]]]:
        """Map each page key's segments to per-segment source MemDescriptors.

        Returns ``(descriptors, per_page_keys)``, or None if the pool meta can't
        be lined up with the requested keys. ``descriptors`` is the flat
        ``{segment_key: [MemDescriptor]}`` mapping KVCR takes, with one entry
        per page component; each descriptor is exactly its KVCR pool's slot
        size, so it lands in one slot. ``per_page_keys`` is the same segment
        keys grouped by page, handed back so callers scoring the result map
        index into it instead of re-formatting every key -- a ``layer_first``
        layout puts ``2 * layer_num`` segments on a page, which makes that
        string building the dominant cost of the call.

        Addresses come from ``layout.host_pool``, not ``self.mem_pool_host``:
        the latter is the ``HostPoolGroup`` for a hybrid stack, and its
        ``get_page_buffer_meta`` forwards to the anchor, so a sidecar transfer
        resolved through it would be handed KV addresses -- and would report
        success after moving KV bytes into KV pages, leaving the sidecar
        untouched.
        """
        host_indices = transfer.host_indices
        keys = transfer.keys or []
        if host_indices is None or not keys:
            return None
        if layout.is_logical_anchor:
            return self._anchor_descriptors(host_indices, keys, layout)
        try:
            ptr_list, size_list = layout.host_pool.get_page_buffer_meta(host_indices)
        except Exception:
            logger.warning(
                "KVCRStore: get_page_buffer_meta failed for pool %s",
                layout.name,
                exc_info=True,
            )
            return None
        segments = layout.segments_per_page
        if len(ptr_list) != len(keys) * segments:
            logger.warning(
                "KVCRStore: pool %s page meta count %d != keys %d * segments "
                "%d; layout changed since registration?",
                layout.name,
                len(ptr_list),
                len(keys),
                segments,
            )
            return None
        descriptors: Dict[BlockKey, List[MemDescriptor]] = {}
        per_page_keys: List[List[BlockKey]] = []
        for page_idx, key in enumerate(keys):
            base = page_idx * segments
            page_keys: List[BlockKey] = []
            for seg in range(segments):
                ptr = int(ptr_list[base + seg])
                size = int(size_list[base + seg])
                if size != layout.segment_sizes[seg]:
                    logger.warning(
                        "KVCRStore: pool %s segment %d is %d bytes, was %d at "
                        "registration",
                        layout.name,
                        seg,
                        size,
                        layout.segment_sizes[seg],
                    )
                    return None
                segment_key = self._segment_key(layout, key, seg)
                page_keys.append(segment_key)
                # A one-descriptor list per key: KVCR's API is plural but its
                # core rejects any block that does not carry exactly one span.
                descriptors[segment_key] = [
                    MemDescriptor(
                        end_point_name=self._agent_name,
                        mem_type="DRAM",
                        addr=ptr,
                        size=size,
                        device_Id=0,
                        info=layout.kvcr_pool_names[seg],
                    )
                ]
            per_page_keys.append(page_keys)
        return descriptors, per_page_keys

    def _anchor_descriptors(
        self,
        host_indices: torch.Tensor,
        keys: List[str],
        layout: _PoolLayout,
    ) -> Optional[Tuple[Dict[BlockKey, List[MemDescriptor]], List[List[BlockKey]]]]:
        """``_host_descriptors`` for a logical anchor: one marker block per page.

        The anchor's transfer carries host indices into a pool that holds no
        bytes, so the marker buffer this backend allocated is what a descriptor
        can name. Each page addresses the marker slot matching its own host
        page, mirroring how a real pool's page meta resolves -- HiCache holds a
        host page for the life of a transfer, so two in-flight transfers cannot
        be handed the same slot and cannot alias in the buffer.
        """
        page_size = int(layout.host_pool.page_size)
        marker = layout.marker_buffer
        if len(host_indices) != len(keys) * page_size:
            logger.warning(
                "KVCRStore: anchor pool %s got %d host indices for %d pages of "
                "%d slots",
                layout.name,
                len(host_indices),
                len(keys),
                page_size,
            )
            return None
        descriptors: Dict[BlockKey, List[MemDescriptor]] = {}
        per_page_keys: List[List[BlockKey]] = []
        for page_idx, key in enumerate(keys):
            slot = int(host_indices[page_idx * page_size]) // page_size
            if not 0 <= slot < marker.shape[0]:
                logger.warning(
                    "KVCRStore: anchor pool %s page %d maps to marker slot %d, "
                    "outside the %d the pool declared",
                    layout.name,
                    page_idx,
                    slot,
                    marker.shape[0],
                )
                return None
            segment_key = self._segment_key(layout, key, 0)
            per_page_keys.append([segment_key])
            descriptors[segment_key] = [
                MemDescriptor(
                    end_point_name=self._agent_name,
                    mem_type="DRAM",
                    addr=marker[slot].data_ptr(),
                    size=_ANCHOR_MARKER_BYTES,
                    device_Id=0,
                    info=layout.kvcr_pool_names[0],
                )
            ]
        return descriptors, per_page_keys

    @_fail_closed(_miss_per_transfer)
    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        """Load KV into host memory via KVCR ``deliver()``.

        ``deliver`` is a single unified entry: the core auto-routes each block
        key by residency (see ``KVCR.deliver``). A key that ``deposit`` made
        locally resident is served from KVCR's own DRAM tier; a key that is only
        covered by this request's router hint is pulled from the source peer
        over NIXL. We hand ``deliver`` the *host page* segment descriptors as
        write destinations, so both paths land straight in the engine KV pool.

        The remote branch is gated on a well-formed hint having been registered
        with the core for this request_id (via ``submit_hint``); without one the
        core reports MISS for non-resident keys and we return them as failures,
        letting HiCache fall back to recompute.
        """
        results: Dict[str, List[bool]] = {}
        if self._ensure_kvcr() is None:
            return {str(t.name): [False] * len(t.keys or []) for t in transfers}

        request_id = self._register_hint(extra_info)
        try:
            for transfer in transfers:
                layout = self._layout_for(transfer)
                if layout is None:
                    results[str(transfer.name)] = [False] * len(transfer.keys or [])
                    continue
                results[str(transfer.name)] = self._deliver_transfer(
                    transfer, layout, request_id
                )
        finally:
            self._discard_hint(request_id)
        return results

    def _discard_hint(self, request_id: Optional[str]) -> None:
        """Unregister the request's hint, without letting that fail the batch.

        Two reasons this is not a bare call in the ``finally``. It runs on the
        exception path too, where raising would replace the original fault with
        a less informative one; and on the success path a raise would discard a
        result set whose pages are already in host memory, turning a completed
        fetch into a recompute.

        The leak it cannot prevent is the core's: a hint we failed to discard
        stays in KVCR's request-scoped table. That is bounded per request and
        harmless to correctness (a stale entry only ever names a source we would
        have consulted anyway), so it is counted rather than retried.
        """
        if request_id is None:
            return
        try:
            self._kvcr.discard_hint(request_id)
        except Exception:
            self._note_fault("discard_hint")

    def _register_hint(
        self, extra_info: Optional[HiCacheStorageExtraInfo]
    ) -> Optional[str]:
        """Parse a router hint and register it with the core for this request.

        Returns the request_id the core keys the hint on, or None when no
        well-formed hint is present / remote hints are disabled. ``submit_hint``
        only records the advisory routing entry (and, if eager, warms the
        control connection) -- the actual pull is driven by ``deliver`` in
        ``_deliver_transfer``, which lets us target engine host pages rather
        than KVCR-owned slots.

        The core re-parses the envelope itself and a hint it rejects raises
        rather than degrading to local-only. We have already validated the same
        fields in ``_parse_hint``, so a raise here means the two parsers
        disagree -- report it as a fault instead of failing the batch.
        """
        hint = self._parse_hint(extra_info)
        if hint is None:
            self._note("get_without_hint")
            return None
        self._note("get_with_hint")
        self._note("hinted_blocks", len(hint.block_hashes))
        # KVCR keys the request-scoped hint table on request_id; the controller
        # does not thread one through extra_info, so we mint our own.
        request_id = self._hint_request_id()
        try:
            self._kvcr.submit_hint(hint.to_kvcr_hint(), request_id=request_id)
        except Exception:
            self._note_fault("submit_hint")
            return None
        logger.debug(
            "KVCRStore: registered router hint (source=%s, %d blocks, req=%s)",
            hint.source_control_endpoint,
            len(hint.block_hashes),
            request_id,
        )
        return request_id

    def _parse_hint(
        self, extra_info: Optional[HiCacheStorageExtraInfo]
    ) -> Optional[RouterHint]:
        """The request's router hint, with its source endpoint aligned to us.

        The endpoint on the hint already names the right *DP rank* of the source:
        dynamo's router indexes workers as ``(worker_id, dp_rank)`` and resolves
        the source's advertised per-DP-rank map down to one endpoint before the
        hint ships. What it cannot resolve is the rank *within* that DP group --
        it has no TP concept, so the port it names is that DP rank's first
        scheduler. Each attention rank holds a different slice of every head, so
        rank ``i`` of our DP group must pull from rank ``i`` of the source's.
        Realigning here mirrors what :meth:`_control_port` does on the bind side,
        with only the within-DP part of the offset since the DP part is the
        router's to apply.

        Getting this wrong does not fail -- KVCR block keys are token hashes and
        carry no rank identity, so a rank that dials the wrong peer receives a
        shard it will happily accept and decode from. Correctness therefore rests
        entirely on this offset, which is why an endpoint we cannot realign drops
        the hint (costing a recompute) rather than passing it through.

        The endpoint is an address this process will connect out to, taken from
        request-scoped data, so ``_offset_endpoint_port`` also decides whether it
        is dialable at all: transport, port range, and bind wildcards. What that
        cannot decide is whether the *named peer* is one we should trust, since
        nothing in the hint is authenticated. ``kv_hints`` is documented as
        router-set and never client-set; enforcing that is the ingress's job,
        not reconstructible here.
        """
        if not self._config.enable_remote_hint:
            return None
        hint = RouterHint.maybe_from_extra_info(extra_info)
        if hint is None:
            return None
        offset = _within_dp_offset(self._storage_config)
        endpoint = _offset_endpoint_port(hint.source_control_endpoint, offset)
        if endpoint is None:
            logger.warning(
                "KVCRStore: dropping router hint, cannot align source endpoint "
                "%s to within-DP rank offset %d",
                hint.source_control_endpoint,
                offset,
            )
            return None
        return msgspec.structs.replace(hint, source_control_endpoint=endpoint)

    def _note_entry_statuses(self, entries: Dict) -> None:
        """Count how KVCR classified each block key, not just pass/fail.

        ``OpEntryResult.success`` is ``status is SUCCESS``, so a block the
        *policy* declined (``DROPPED``, returned when ``decide_ingest`` answers
        DROP) and a block that genuinely broke (``FAILED``) collapse into the
        same falsy value everywhere downstream. Both are correct to treat as
        "not stored" -- but they are not the same event to a reader: a rising
        DROPPED count means the local tier is under capacity pressure and the
        policy is doing its job, while a rising FAILED count means something is
        wrong. Keeping them apart is what makes the counters usable as evidence
        when a policy is being tuned, which the fault-injection run relies on.
        """
        dropped = failed = 0
        for entry in entries.values():
            if entry.status is OpEntryStatus.DROPPED:
                dropped += 1
            elif entry.status is not OpEntryStatus.SUCCESS:
                failed += 1
        if dropped:
            self._note("entries_dropped_by_policy", dropped)
        if failed:
            self._note("entries_failed", failed)

    def _note(self, event: str, count: int = 1) -> None:
        """Count a remote-path event, and summarize periodically at INFO.

        Per-event logging is not an option on this path -- it runs per prefetch,
        which is per request -- but silence is worse: when the remote path stops
        working there is nothing in any log to say so, and the first symptom is
        a throughput number nobody can attribute. So counters accumulate and one
        line goes out every ``_STATS_LOG_INTERVAL_S``, only while something is
        happening (an idle worker stays quiet because nothing increments).
        """
        with self._stats_lock:
            self._stats[event] += count
            now = time.monotonic()
            if now < self._next_stats_log_at:
                return
            self._next_stats_log_at = now + _STATS_LOG_INTERVAL_S
            snapshot = dict(self._stats)
        logger.info(
            "KVCRStore remote path (cumulative): %s",
            " ".join(f"{name}={value}" for name, value in sorted(snapshot.items())),
        )

    def _note_fault(self, method_name: str) -> None:
        """Record a fault the ``_fail_closed`` guard caught, and log it sparsely.

        Counted per entry point, not in aggregate: a fault in ``batch_set_v2``
        means offload is failing while reads may be fine, and the two are
        repaired in different places. The traceback goes out at most once per
        ``_FAULT_LOG_INTERVAL_S`` because the faults this guard is for repeat
        every prefetch -- but the *first* one is logged immediately, since a
        counter alone would not say what broke.
        """
        self._note(f"faults_{method_name}")
        with self._stats_lock:
            now = time.monotonic()
            if now < self._next_fault_log_at:
                return
            self._next_fault_log_at = now + _FAULT_LOG_INTERVAL_S
        logger.warning(
            "KVCRStore: %s failed; reporting a miss so HiCache recomputes. "
            "Repeated faults are counted in stats() as faults_%s.",
            method_name,
            method_name,
            exc_info=True,
        )

    def stats(self) -> Dict[str, int]:
        """Snapshot of the remote-path counters, for tests and for operators."""
        with self._stats_lock:
            return dict(self._stats)

    def _hint_request_id(self) -> str:
        """Fresh id scoping one ``batch_get_v2`` call's hint in the core.

        Must be unique per *call*, not per prefix. The id keys the core's
        request-scoped hint table, and this call ends by unregistering it
        (``discard_hint`` in the ``finally``) -- so two concurrent calls sharing
        an id means the first one to finish revokes the hint the second is
        still fetching against, which downgrades it to a silent local-only miss.

        Deriving the id from the hint content instead is what made that
        collision reachable: two requests sharing a prefix is the normal case
        here, not a rare one, and it is exactly when both want the same source.
        A counter has no such structure. It is process-local, which is all the
        core requires -- the hint table lives in this worker.
        """
        with self._hint_id_lock:
            self._next_hint_id += 1
            return f"kvcr-get-{self._next_hint_id}"

    def _deliver_transfer(
        self,
        transfer: PoolTransfer,
        layout: _PoolLayout,
        request_id: Optional[str],
    ) -> List[bool]:
        """Pull one transfer's pages into host memory via ``deliver``.

        Builds a ``{segment_key: host destination descriptor}`` map (the same
        page->segment fan-out as deposit) and issues a single ``deliver``. A
        page counts as loaded only when every one of its segments succeeded.
        """
        keys = transfer.keys or []
        if not keys:
            return []
        built = self._host_descriptors(transfer, layout)
        if built is None:
            return [False] * len(keys)
        destinations, per_page_keys = built

        _, result_map = self._submit_and_wait(
            lambda: self._kvcr.deliver(destinations, request_id=request_id)
        )

        results = [
            all(result_map.get(seg_key, False) for seg_key in page_keys)
            for page_keys in per_page_keys
        ]
        loaded = sum(results)
        self._note("pages_requested", len(results))
        self._note("pages_loaded", loaded)
        if request_id is not None:
            # Separate from pages_loaded: a hinted deliver that lands nothing is
            # the failure this backend exists to make visible, and it is not the
            # same event as a local-tier miss on an unhinted request.
            self._note("hinted_pages_requested", len(results))
            self._note("hinted_pages_loaded", loaded)
        return results

    @_fail_closed(_no_prefix)
    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        """Longest available prefix, across the KV pool and every sidecar.

        A KV page is available when either (a) all its segments are resident in
        KVCR's local DRAM tier, or (b) it is covered by this request's router
        hint (a peer holds it and ``batch_get_v2`` can pull it). The prefix is
        root-aligned and contiguous, so it stops at the first page that is
        neither. This gate is what makes the remote branch of ``batch_get_v2``
        reachable -- the controller only issues gets for the prefix reported
        here.

        Each sidecar pool then removes the stop points it cannot serve, since
        the controller issues gets for one prefix across every pool. The whole
        set of surviving stop points is returned, not just its maximum: a
        ``TRAILING_PAGES`` pool leaves holes, and the caller intersects these
        sets across ranks.
        """
        if self._ensure_kvcr() is None:
            return PoolTransferResult.empty()
        hint = self._parse_hint(extra_info)
        kv_pages, remote_prefix = self._kv_prefix(keys, hint)
        hit_count: Dict[str, int] = {str(PoolName.KV): kv_pages} if kv_pages else {}
        restorable = list(range(1, kv_pages + 1))

        for transfer in pool_transfers or []:
            if not restorable:
                break
            if transfer.name == PoolName.KV:
                # The KV prefix above already is this pool's answer; scoring it
                # again through the sidecar path would double-count it.
                continue
            boundary, pool_restorable = self._pool_prefix(transfer, keys, kv_pages, hint)
            if boundary:
                hit_count[str(transfer.name)] = boundary
            restorable = [page for page in restorable if page in pool_restorable]

        # This gate is where the remote path is won or lost: the controller only
        # issues gets for the prefix reported here, so a hint that covers
        # nothing produces no get at all and leaves no other trace.
        self._note("exists_calls")
        if keys and not restorable:
            # One pool reporting nothing zeroes the whole prefix, and the
            # aggregate counters cannot say which -- the KV anchor missing and
            # a single sidecar collapsing the intersection look identical.
            self._note("exists_miss_pages", len(keys))
            self._note("exists_miss_kv" if not kv_pages else "exists_miss_sidecar")
        if remote_prefix is not None:
            self._note("exists_with_hint")
            if remote_prefix:
                self._note("exists_hint_covered_pages", remote_prefix)
            else:
                self._note("exists_hint_covered_nothing")
        return PoolTransferResult(
            restorable[-1] if restorable else 0, hit_count, restorable
        )

    def _kv_prefix(
        self, keys: List[str], hint: Optional[RouterHint]
    ) -> Tuple[int, Optional[int]]:
        """``(available KV pages, pages the hint covered)``; the latter None if unhinted."""
        layout = self._pool_layouts.get(str(PoolName.KV))
        if layout is None:
            return 0, None
        prefix = 0
        remote_prefix = 0
        for key in keys:
            if not self._available(layout, key, hint):
                break
            if not self._locally_resident(self._page_segment_keys(layout, key)):
                remote_prefix += 1
            prefix += 1
        return prefix, (remote_prefix if hint is not None else None)

    def _available(
        self, layout: _PoolLayout, key: str, hint: Optional[RouterHint]
    ) -> bool:
        """Whether ``batch_get_v2`` can serve this pool's page, locally or remotely.

        The hint branch applies to every pool, not just KV. A hint names page
        hashes, and a sidecar transfer carries the KV page's hashes (see
        ``indices_from_pool``) -- so the source deposited that page's sidecar
        blocks under the same hash, and ``deliver`` pulls them by the same
        routing. Restricting the hint to KV was a real regression rather than a
        conservative choice: the controller issues one prefix across all pools,
        so a sidecar that reports local-only collapses the prefix to what is
        already here and no hybrid model can fetch remotely at all.

        The pool names both peers put in the descriptors are derived from the
        pool name and component sizes alone (``_kvcr_pool_name``), so two peers
        running the same model agree -- which is the precondition, since
        ``remote_fw_dram.start_write`` drops a key whose source and destination
        ``info`` lists differ and says nothing.
        """
        if self._locally_resident(self._page_segment_keys(layout, key)):
            return True
        return hint is not None and hint.covers(key)

    def _pool_prefix(
        self,
        transfer: PoolTransfer,
        keys: List[str],
        kv_pages: int,
        hint: Optional[RouterHint],
    ) -> Tuple[int, Set[int]]:
        """``(longest prefix, every usable stop point)`` for one sidecar pool.

        Scored over the KV prefix only: a page whose KV is unavailable is not a
        stop point no matter what its sidecar holds.
        """
        layout = self._layout_for(transfer)
        if layout is None:
            return 0, set()
        page_exists = [
            self._available(layout, key, hint) for key in keys[:kv_pages]
        ]
        if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
            boundary = (
                page_exists.index(False) if False in page_exists else len(page_exists)
            )
            return boundary, set(range(1, boundary + 1))
        if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
            # A stop point works when the window ending there is complete, so
            # scan every one instead of stopping at the longest.
            trailing = max(1, len(transfer.keys) if transfer.keys else 1)
            usable = {
                prefix_len
                for prefix_len in range(1, kv_pages + 1)
                if all(
                    page_exists[i]
                    for i in range(max(0, prefix_len - trailing), prefix_len)
                )
            }
            return (max(usable) if usable else 0), usable
        # An unknown policy is a pool whose restore rule this backend has not
        # been taught, so serve none of it rather than guess the wrong one.
        self._note(f"unsupported_hit_policy_{transfer.hit_policy}")
        return 0, set()

    # ------------------------------------------------------------------
    # Op submission and completion
    # ------------------------------------------------------------------

    def _submit_and_wait(self, submit: Callable[[], int]) -> Tuple[int, Dict]:
        """Issue one KVCR op and block for its result, as ``(handle, results)``.

        ``submit`` runs under ``_poll_lock`` so the op is registered as awaited
        before anyone can drain its completion. Registering afterwards would
        race: a local-tier deposit can finish in microseconds while the scheduler
        ticks once per loop iteration, so the tick would see a completion with no
        waiter, drop it as late, and the caller would sit out the full
        ``get_timeout_s`` before reporting a miss on an op that actually
        succeeded.

        The handle comes back because it is the only join between our logs and
        KVCR's -- a failure here is usually diagnosed from the core's side.
        """
        with self._poll_lock:
            op_handle = submit()
            self._waiting_ops.add(op_handle)
        return op_handle, self._drain_until(op_handle)

    def _register_waiter(self, op_handle: int) -> None:
        """Claim ``op_handle``'s completion, so ``_poll_once`` stashes it.

        Idempotent: ``_submit_and_wait`` already did this under the lock it held
        across the submit, and ``_drain_until`` repeats it to cover a direct
        call.
        """
        with self._poll_lock:
            self._waiting_ops.add(op_handle)

    def _drain_until(self, op_handle: int, timeout_s: Optional[float] = None) -> Dict:
        """Pump kvcr.poll_completed() until op_handle reports, or the deadline passes.

        Blocking here is the contract, not a compromise: this runs on the HiCache
        controller's dedicated ``prefetch_io_aux_func`` daemon thread, and
        ``_page_transfer`` inspects ``operation.completed_tokens`` immediately
        after ``page_get_func`` returns -- so results must be in hand by then.
        The scheduler thread is never involved; it only observes the resulting
        ``completed_tokens`` via the existing ``check_prefetch_progress`` tick.

        KVCR's transfer progress is owned by its own "kvcr-progress" daemon
        thread, which appends finished ops to a completion queue;
        ``poll_completed`` is a non-blocking drain of that queue with no
        condition variable to wait on. So we poll -- but yield between attempts
        (backing off to ``_DRAIN_POLL_MAX_S``) instead of spinning a bare loop,
        which otherwise burns a core and starves the progress thread that we are
        waiting on. Completions for other in-flight ops are stashed, never
        dropped.

        Leaving deregisters this handle, whether the result arrived or the
        deadline did. Those two exits are not equally safe and the difference is
        not visible to the caller, so they are counted separately here.

        A *reported* op is finished: the core has retired its transfers, and the
        host pages HiCache frees on our return are nobody's target. An
        *abandoned* op is not. ``kvcr.abort()`` is a no-op stub, so we cannot
        cancel it, only agree to ignore whatever it reports -- or never reports.
        ``get_timeout_s > abandon_timeout_ms`` (enforced in
        ``KVCRBackendConfig``) means both ends have passed their own deadline by
        the time we give up, so no *new* descriptor is submitted after this
        point; it does not fence a descriptor the NIC has already begun. Closing
        that needs a per-op quiescence signal from KVCR, which is filed upstream.

        Until it exists, an abandoned handle is remembered (bounded) so a result
        that shows up afterwards is reported as such rather than dropped as an
        ordinary late tick. That late report is the only observable the hazard
        has: it says a transfer was still live after HiCache took its pages
        back. Do not shorten this wait below the core's deadline.
        """
        timeout = self._config.get_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout
        sleep_s = _DRAIN_POLL_MIN_S
        abandoned = False
        try:
            self._register_waiter(op_handle)
            while True:
                # Always go through the stash: the scheduler tick drains the same
                # queue, so our own completion may well be observed by it rather
                # than by the poll below.
                self._poll_once(self._kvcr)
                with self._poll_lock:
                    stashed = self._completed_ops.pop(op_handle, None)
                if stashed is not None:
                    self._note_entry_statuses(stashed)
                    return {k: v.success for k, v in stashed.items()}
                if time.monotonic() >= deadline:
                    self._note("op_abandoned_on_timeout")
                    logger.warning(
                        "KVCRStore: op %s did not complete within %.1fs; "
                        "abandoning it. Its host pages return to HiCache while "
                        "the core still owns the op.",
                        op_handle,
                        timeout,
                    )
                    abandoned = True
                    return {}
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 2, _DRAIN_POLL_MAX_S)
        finally:
            with self._poll_lock:
                self._waiting_ops.discard(op_handle)
                # A completion can land between the last poll and here; drop it
                # now rather than leave it for a pop that will never come.
                self._completed_ops.pop(op_handle, None)
                if abandoned:
                    self._abandoned_ops.append(op_handle)

    # ------------------------------------------------------------------
    # v1 zero-copy interface (HiRadixCache path)
    # ------------------------------------------------------------------
    #
    # HiCache has two zero-copy call shapes and a backend needs both: the
    # HybridCacheController drives `batch_*_v2` with PoolTransfers, while
    # HiRadixCache's `_page_{get,set}_zero_copy` drives `batch_*_v1` with
    # (keys, host_indices). Only the pool name differs, so v1 wraps v2.

    def _kv_transfer(self, keys: List[str], host_indices) -> PoolTransfer:
        return PoolTransfer(
            name=PoolName.KV, host_indices=host_indices, keys=list(keys)
        )

    # v1 delegates to an already-guarded v2, so the guard here only covers what
    # v1 itself does -- building the PoolTransfer and reading the result out.
    @_fail_closed(_miss_per_key)
    def batch_set_v1(
        self,
        keys: List[str],
        host_indices,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        results = self.batch_set_v2([self._kv_transfer(keys, host_indices)], extra_info)
        return results.get(str(PoolName.KV), [False] * len(keys))

    @_fail_closed(_miss_per_key)
    def batch_get_v1(
        self,
        keys: List[str],
        host_indices,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        results = self.batch_get_v2([self._kv_transfer(keys, host_indices)], extra_info)
        return results.get(str(PoolName.KV), [False] * len(keys))

    @_fail_closed(lambda self, *a, **kw: 0)
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        return self.batch_exists_v2(keys, None, extra_info).kv_hit_pages

    def clear(self) -> None:
        """Not supported: KVCR exposes no bulk invalidation.

        ``HiCacheStorage.clear`` is a bare ``pass``, so inheriting it makes
        ``/flush_cache`` report success while every block this worker deposited
        stays resident and peer-visible. That is worse than an error: the
        operator's reason for flushing -- a poisoned tier, a model swap -- is
        exactly the case where a stale block being served to a peer is a
        correctness bug, and the caller (``clear_storage_backend``) has a False
        return that says "this backend cannot".

        The core has ``release()`` for handles this store holds and eviction
        driven by capacity pressure, but nothing that drops a block by key, and
        the tier's contents are not enumerable from here. Implementing this
        needs a KVCR-side invalidate; until then it refuses honestly.
        """
        self._note("clear_unsupported")
        raise NotImplementedError(
            "KVCRStore does not support clear(): the KVCR core has no bulk "
            "invalidation, so blocks already deposited would stay resident and "
            "peer-visible after a flush that reported success."
        )

    # ------------------------------------------------------------------
    # byte-copy legacy ABC methods -- draft stubs
    # ------------------------------------------------------------------

    def get(self, key, target_location=None, target_sizes=None):
        return None  # DRAFT-STUB: v2 path is the supported one.

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        return [None] * len(keys)  # DRAFT-STUB

    def set(self, key, value=None, target_location=None, target_sizes=None) -> bool:
        return False  # DRAFT-STUB

    def batch_set(
        self, keys, values=None, target_locations=None, target_sizes=None
    ) -> bool:
        return False  # DRAFT-STUB

    @_fail_closed(lambda self, *a, **kw: False)
    def exists(self, key: str) -> bool:
        layout = self._pool_layouts.get(str(PoolName.KV))
        if layout is None or self._ensure_kvcr() is None:
            return False
        return self._locally_resident(self._page_segment_keys(layout, key))

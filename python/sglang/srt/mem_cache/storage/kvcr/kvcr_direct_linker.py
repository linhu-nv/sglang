"""Layer-wise direct delivery through a single KVCR API owner."""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import socket
import threading
import time
import uuid
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty, Queue

import msgspec
import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.storage.kvcr.layout import KVCRLayout
from sglang.srt.mem_cache.storage.kvcr.runtime import KVCRRuntime
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.runtime_context import get_memory, get_model, get_parallel

logger = logging.getLogger(__name__)


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.reset()

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        future = self.futures[index][layer]
        if not future.done():
            future.set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
        except BaseException as error:
            raise RuntimeError("KVCR layer-wise load failed") from error
        finally:
            if threshold == self.num_layers - 1:
                self.futures.pop(index, None)

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}


@dataclass
class _Load:
    rid: str
    scope: str | None
    transfers: tuple
    done: bool = False
    error: BaseException | None = None


@dataclass
class _Batch:
    index: int
    loads: tuple[_Load, ...]
    ready: object
    work: object
    remaining: list[int]
    active: int = 0
    exhausted: bool = False
    error: BaseException | None = None
    started: float = field(default_factory=time.perf_counter)


@dataclass
class _Offload:
    transfers: tuple
    ready: object
    dependencies: tuple[_Load, ...]
    keys: set[bytes]
    preparation: Future | None = None
    work: object = None
    retained_keys: set[bytes] = field(default_factory=set)
    claims: list = field(default_factory=list)
    active: int = 0
    exhausted: bool = False
    success: bool = True


@dataclass
class _Confirm:
    active: int = 0
    claims: list = field(default_factory=list)
    success: bool = True
    exhausted: bool = False


@dataclass
class _Operation:
    owner: _Batch | _Offload
    keys: set[bytes]
    layer: int | None = None
    result: object = None


def _present(status, tier):
    return (status.value, tier.value if tier is not None else None) in (
        ("HIT", "DRAM"),
        ("FETCHABLE", "REMOTE_G2"),
    )


def _remote(status, tier):
    return status.value == "FETCHABLE" and tier is not None and tier.value == "REMOTE_G2"


def _snapshot_indices(transfers):
    """Clone index tensors on the stream; the tree may reuse the originals."""
    return tuple(
        dataclasses.replace(
            transfer,
            device_indices=(
                transfer.device_indices.clone()
                if isinstance(transfer.device_indices, torch.Tensor)
                else transfer.device_indices
            ),
            host_indices=(
                transfer.host_indices.clone()
                if isinstance(transfer.host_indices, torch.Tensor)
                else transfer.host_indices
            ),
        )
        for transfer in transfers
    )


def _offset_source_ports(hints, offset):
    """Point each kv.fetch source endpoint ``offset`` control ports up.

    Every rank listens on ``control_port + rank``, so this names the source
    rank with the same number as the caller.
    """
    for action in hints.get("actions", ()):
        payload = action.get("payload") if isinstance(action, dict) else None
        endpoint = (
            payload.get("source_control_endpoint") if isinstance(payload, dict) else None
        )
        if isinstance(endpoint, str):
            host, sep, port = endpoint.rpartition(":")
            if sep and port.isdigit():
                payload["source_control_endpoint"] = f"{host}:{int(port) + offset}"
    return hints


class KVCRDirectLinker(UnifiedCacheLinker):
    loaded_path_is_stored = False
    local_storage_medium = StorageMedium.CPU
    local_storage_ownership = "kvcr"

    def __init__(self, server_args, params, *, components, runtime_factory=None):
        if get_memory().enable_linker_mla_dedup:
            raise ValueError("KVCR does not support --enable-linker-mla-dedup")
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=params.token_to_kv_pool_allocator.get_kvcache(),
            page_size=params.page_size,
            params=params,
            components=components,
        )
        tp_rank, tp_size, rank = 0, get_parallel().tp_size, 0
        if torch.distributed.is_initialized():
            group = params.attn_tp_cache_group or params.tp_cache_group
            tp_rank = torch.distributed.get_rank(group)
            tp_size = torch.distributed.get_world_size(group)
            rank = torch.distributed.get_rank()
        self._remote_hints_supported = params.attn_cp_size == params.pp_size == 1 and (
            tp_size == 1 or self.pool_group.rank_replicated
        )
        # The router names one control endpoint per source worker, its rank 0.
        # Rank-replicated pools are identical on every source rank, so pulling
        # from the same-numbered rank spreads a restore over all source ranks
        # and their NICs instead of serializing it on rank 0.
        self._source_rank_offset = tp_rank if self.pool_group.rank_replicated else 0
        if not self._remote_hints_supported:
            logger.warning(
                "KVCR remote hints disabled: this layout requires shard-specific "
                "source endpoints. Local DRAM reuse remains available."
            )
        self.layout = KVCRLayout(
            self.pool_group,
            model_name=json.dumps(
                (
                    get_model().model_path,
                    get_model().revision,
                    get_model().quantization,
                    get_model().kv_cache_dtype,
                )
            ),
            endpoint_name=f"sglang-{socket.gethostname()}-{rank}-{uuid.uuid4().hex}",
            tp_rank=tp_rank,
            tp_size=tp_size,
            cp_rank=params.attn_cp_rank,
            cp_size=params.attn_cp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
        )
        options, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        self._make_runtime = lambda: (runtime_factory or KVCRRuntime)(
            self.layout, options, rank=rank
        )
        self.layer_done_counter = LayerWiseLoadCounter(self.pool_group.num_layers)
        self._commands = Queue()
        self._command_lock = threading.Lock()
        self._completed_loads = Queue()
        self._completed_offloads = Queue()
        self._loads = {}
        self._queued = []
        self._batches = deque()
        self._offloads = deque()
        # Only the scheduler acknowledges offloads and suppresses their evictions.
        self._unacknowledged_keys = Counter()
        self._preparer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kvcr-offload-prepare"
        )
        self._operations = {}
        self._hazards = set()
        self._scopes = {}
        self._request_scopes = {}
        self._confirmed = {}
        self._released_scopes = set()
        self._fatal = None
        self._closed = False
        self._runtime = None
        self._started = Future()
        self._thread = threading.Thread(
            target=self._run, name=f"kvcr-owner-{rank}", daemon=True
        )
        self._thread.start()
        self._started.result()

    def _raise_if_failed(self):
        if self._fatal is not None:
            raise self._fatal

    def _send(self, function, *args, wait=False):
        future = Future() if wait else None
        with self._command_lock:
            if self._closed:
                raise RuntimeError("KVCR linker is closed")
            if function != self._close:
                self._raise_if_failed()
            self._commands.put((function, args, future))
        return future

    def _call(self, function, *args):
        return self._send(function, *args, wait=True).result()

    def _run(self):
        try:
            buffer = self.pool_group.entries[0].kv_buffer[0]
            if buffer.is_cuda:
                torch.cuda.set_device(buffer.device)
            self._runtime = self._make_runtime()
            self._started.set_result(None)
        except BaseException as error:
            self._fatal = error
            self._started.set_exception(error)
            # KVCRStartupError may retain registrations and DMA; this owner
            # keeps the exception, runtime and framework tensors alive.
            if not hasattr(error, "kvcr_runtime"):
                self._closed = True
                return
            self._runtime = error.kvcr_runtime
        while not self._closed:
            active = self._fatal is None and (self._batches or self._offloads)
            try:
                function, args, future = self._commands.get(
                    timeout=0 if active else 0.001
                )
            except Empty:
                if active:
                    # Yield the GIL without pacing active transfers at 1 ms.
                    time.sleep(0)
            else:
                try:
                    if self._fatal is not None and function != self._close:
                        raise self._fatal
                    result = function(*args)
                    if future is not None:
                        future.set_result(result)
                except BaseException as error:
                    if future is None or function in (self._close, self._reset):
                        if hasattr(error, "kvcr_runtime"):
                            self._runtime = error.kvcr_runtime
                        self._fatal = error
                        for batch in self._batches:
                            self._fail_batch(batch, error)
                    if future is not None:
                        future.set_exception(error)
                    else:
                        logger.exception("KVCR asynchronous command failed")
            if self._fatal is None and not self._closed:
                try:
                    self._progress()
                    self._runtime.log_stats()
                except BaseException as error:
                    self._fatal = error
                    for batch in self._batches:
                        self._fail_batch(batch, error)
                    logger.exception(
                        "KVCR owner failed; retaining transfer ownership until close"
                    )
        while True:
            try:
                _, _, future = self._commands.get_nowait()
            except Empty:
                break
            if future is not None:
                future.set_exception(RuntimeError("KVCR linker is closed"))

    def set_request_context(self, handle, kv_hints):
        hints = (
            msgspec.to_builtins(kv_hints)
            if isinstance(kv_hints, msgspec.Struct)
            else copy.deepcopy(kv_hints)
        )
        self._call(self._set_context, handle, hints)

    def _set_context(self, handle, hints):
        scope = f"{len(handle.rid)}:{handle.rid}:{handle.attempt_id}"
        self._scopes[handle] = scope
        self._request_scopes[handle.rid] = scope
        if hints is not None and self._remote_hints_supported:
            if self._source_rank_offset:
                hints = _offset_source_ports(hints, self._source_rank_offset)
            try:
                self._runtime.client.submit_hint(hints, request_id=scope)
            except ValueError as error:
                # SGLang envelopes may carry actions that KVCR does not consume.
                logger.warning(
                    "Ignoring unsupported KVCR hint for %s: %s", handle.rid, error
                )

    def release_request_context(self, handle):
        if not self._closed:
            self._send(self._release_context, handle)

    def _release_context(self, handle):
        scope = self._scopes.pop(handle, None)
        if scope is not None:
            if self._request_scopes.get(handle.rid) == scope:
                self._request_scopes.pop(handle.rid)
            self._confirmed.pop(scope, None)
            self._released_scopes.add(scope)
            self._cleanup_scopes()

    def _cleanup_scopes(self):
        in_use = {load.scope for load in self._loads.values()}
        for scope in self._released_scopes - in_use:
            self._runtime.client.discard_hint(scope)
            self._released_scopes.remove(scope)

    def lookup(self, rid, transfers):
        prepared = self.layout.group.resolve_transfers(transfers)
        keys = (
            next((tuple(t.keys) for t in transfers if t.name == PoolName.KV), ())
            if prepared
            else ()
        )
        return self._call(self._lookup, rid, keys, prepared)

    def _lookup(self, rid, keys, prepared):
        scope = self._request_scopes.get(rid)
        statuses = {}
        for transfer in prepared:
            encoded = [self.layout.encode_key(key, transfer.name) for key in keys]
            result = self._runtime.client.query(encoded, request_id=scope)
            if len(result) != len(encoded):
                raise RuntimeError("KVCR query returned an incomplete result")
            statuses[transfer.name] = (encoded, result)
        if scope is not None:
            self._confirm_trailing(scope, prepared, statuses)
        valid = set(range(1, len(keys) + 1))
        for transfer in prepared:
            _, result = statuses[transfer.name]
            hits = [_present(*entry) for entry in result]
            missing = [0]
            for hit in hits:
                missing.append(missing[-1] + (not hit))
            window = max(1, len(transfer.keys))
            valid = {
                end
                for end in valid
                if missing[end]
                == missing[
                    0
                    if transfer.hit_policy == PoolHitPolicy.ALL_PAGES
                    else max(0, end - window)
                ]
            }
        return sorted(valid)

    def _confirm_trailing(self, scope, prepared, statuses):
        """Admit a remote trailing page only after it has been fetched.

        A hint names page hashes, so query() answers FETCHABLE for every pool's
        key under them, but a trailing pool (SWA, compress state) has state on
        the peer only where it stored a sequence end or chunk boundary. A
        delivery that misses one fails after the tree has committed to it. Only
        the longest boundary the all-pages pools allow is worth restoring, so
        fetch that boundary's trailing window and report every other remote
        trailing page missing.
        """
        from kvcr.types import QueryStatus

        trailing = [t for t in prepared if t.hit_policy != PoolHitPolicy.ALL_PAGES]
        if not trailing:
            return
        longest = min(
            (
                next(
                    (
                        index
                        for index, entry in enumerate(statuses[t.name][1])
                        if not _present(*entry)
                    ),
                    len(statuses[t.name][1]),
                )
                for t in prepared
                if t.hit_policy == PoolHitPolicy.ALL_PAGES
            ),
            default=len(statuses[trailing[0].name][1]),
        )
        confirmed = self._confirmed.setdefault(scope, set())
        owner = _Confirm()
        operations = []
        for transfer in trailing:
            encoded, result = statuses[transfer.name]
            window = range(max(0, longest - max(1, len(transfer.keys))), longest)
            keys = [
                encoded[index]
                for index in window
                if _remote(*result[index]) and encoded[index] not in confirmed
            ]
            if not keys:
                continue
            labels = [
                label
                for component in self.layout._labels[transfer.name]
                for label in component
            ]
            handle = self._runtime.client.fetch(
                keys, request_id=scope, expected_layout=labels
            )
            operation = _Operation(owner, set(keys))
            self._operations[handle] = operation
            operations.append(operation)
            owner.active += 1
        while owner.active:
            self._progress()
            time.sleep(0)
        # A fetch dropped locally never asks the peer, so only a page that
        # landed counts; KVCR would keep answering FETCHABLE for the rest.
        for operation in operations:
            confirmed.update(
                key for key, entry in (operation.result or {}).items() if entry.success
            )
        if owner.claims:
            self._runtime.client.release(owner.claims)
        for transfer in trailing:
            encoded, _ = statuses[transfer.name]
            result = self._runtime.client.query(encoded, request_id=scope)
            statuses[transfer.name] = (
                encoded,
                [
                    (QueryStatus.MISS, None)
                    if _remote(*entry) and key not in confirmed
                    else entry
                    for key, entry in zip(encoded, result)
                ],
            )

    def load(self, rid, transfers):
        prepared = self.layout.prepare(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        return self._call(self._load, rid, prepared)

    def _load(self, rid, prepared):
        if not prepared:
            return False
        if rid in self._loads:
            raise RuntimeError(f"KVCR load already accepted for {rid}")
        load = _Load(rid, self._request_scopes.get(rid), prepared)
        self._loads[rid] = load
        self._queued.append(load)
        return True

    def cancel_queued_load(self, rid):
        # The wrapper has already published these destinations in the tree.
        return False

    def start_layer_wise_loading(self):
        ready = torch.cuda.Event()
        ready.record()
        return self._call(self._start, ready)

    def _start(self, ready):
        if not self._queued:
            return -1
        loads, self._queued = tuple(self._queued), []
        index = self.layer_done_counter.update_producer()
        self._batches.append(
            _Batch(
                index,
                loads,
                ready,
                self._layer_work(loads),
                [1] * self.pool_group.num_layers,
            )
        )
        return index

    def _layer_work(self, loads):
        for layer in range(self.pool_group.num_layers):
            for load in loads:
                for blocks in self.layout.mappings(load.transfers, layer):
                    yield layer, load.scope, blocks
            # The sentinel accounts for work not yet submitted in this layer.
            yield layer, None, None

    def offload(self, transfers):
        # Resolving device indices to rows copies them to the host, which here
        # would wait for every forward already queued on the stream. Snapshot
        # them instead; the preparer resolves them once the ready event fires.
        keys = self.layout.encoded_keys(transfers, allow_partial=True)
        if not keys:
            return False
        transfers = _snapshot_indices(transfers)
        ready = torch.cuda.Event()
        ready.record()
        self._send(self._offload, transfers, ready, keys)
        self._unacknowledged_keys.update(keys)
        return True

    def _offload(self, prepared, ready, keys):
        self._offloads.append(
            _Offload(prepared, ready, tuple(self._loads.values()), keys)
        )

    def _offload_work(self, transfers):
        prepared = self.layout.prepare(list(transfers), allow_partial=True)
        for layer in range(self.pool_group.num_layers):
            yield from self.layout.mappings(prepared, layer)

    def _fail_batch(self, batch, error):
        batch.error = error
        batch.exhausted = True
        batch.work = None
        self.layer_done_counter.fail(batch.index, error)

    def _progress(self):
        results = list(self._runtime.client.poll_completed())
        while self._runtime.events:
            event = self._runtime.events.popleft()
            if getattr(event, "state", None) == "uncertain":
                self._hazards.add(event.op_handle)
            elif getattr(event, "state", None) == "quiesced":
                self._hazards.discard(event.op_handle)
            else:
                raise event
        for handle, result in results:
            operation = self._operations[handle]
            operation.result = result
            if set(result) != operation.keys or not all(
                entry.success for entry in result.values()
            ):
                if isinstance(operation.owner, _Batch):
                    self._fail_batch(
                        operation.owner, RuntimeError(f"KVCR delivery {handle} failed")
                    )
                else:
                    operation.owner.success = False
                    operation.owner.exhausted = True
        # Lifecycle handles may name internal copies rather than public ops.
        # Conservatively retain every accepted operation while DMA is uncertain.
        if self._hazards:
            return
        for handle, operation in list(self._operations.items()):
            if operation.result is None:
                continue
            owner = operation.owner
            owner.active -= 1
            if isinstance(owner, _Batch):
                owner.remaining[operation.layer] -= 1
                self._complete_layer(owner, operation.layer)
            else:
                owner.claims.extend(
                    entry.release_handle
                    for entry in operation.result.values()
                    if entry.release_handle is not None
                )
            del self._operations[handle]
        deliveries = sum(
            isinstance(op.owner, _Batch) for op in self._operations.values()
        )
        for batch in self._batches:
            if not batch.ready.query():
                continue
            while not batch.exhausted and deliveries < 4:
                try:
                    layer, scope, blocks = next(batch.work)
                except StopIteration:
                    batch.exhausted = True
                    break
                if blocks is None:
                    batch.remaining[layer] -= 1
                    self._complete_layer(batch, layer)
                    continue
                started = time.perf_counter()
                handle = self._runtime.client.deliver(blocks, request_id=scope)
                logger.debug(
                    "KVCR submission batch=%d layer=%d keys=%d cpu_ms=%.3f",
                    batch.index,
                    layer,
                    len(blocks),
                    (time.perf_counter() - started) * 1000,
                )
                self._operations[handle] = _Operation(batch, set(blocks), layer)
                batch.active += 1
                batch.remaining[layer] += 1
                deliveries += 1
        # Completion queues preserve admission order even if transport does not.
        while (
            self._batches and self._batches[0].exhausted and not self._batches[0].active
        ):
            batch = self._batches.popleft()
            logger.debug(
                "KVCR batch=%d drained_ms=%.3f success=%s",
                batch.index,
                (time.perf_counter() - batch.started) * 1000,
                batch.error is None,
            )
            for load in batch.loads:
                load.done, load.error = True, batch.error
                del self._loads[load.rid]
            self._completed_loads.put([load.rid for load in batch.loads])
            self._cleanup_scopes()
        if self._offloads:
            task = self._offloads[0]
            if not task.ready.query() or any(
                not load.done for load in task.dependencies
            ):
                return
            if any(load.error for load in task.dependencies):
                task.success, task.exhausted = False, True
            while not task.exhausted and task.active < 4:
                if task.work is None:
                    task.work = self._offload_work(task.transfers)
                if task.preparation is None:
                    task.preparation = self._preparer.submit(next, task.work, None)
                if not task.preparation.done():
                    return
                try:
                    blocks = task.preparation.result()
                except Exception:
                    logger.exception("KVCR offload preparation failed")
                    task.success, task.exhausted = False, True
                    break
                finally:
                    task.preparation = None
                if blocks is None:
                    task.exhausted = True
                else:
                    # Retain early pieces until every layer has drained.
                    retain = not task.retained_keys.issuperset(blocks)
                    handle = self._runtime.client.deposit(blocks, no_evict=retain)
                    if retain:
                        task.retained_keys.update(blocks)
                    self._operations[handle] = _Operation(task, set(blocks))
                    task.active += 1
            if task.exhausted and task.preparation is not None:
                task.preparation.cancel()
                if not task.preparation.done():
                    return
                task.preparation = None
            if task.exhausted and not task.active:
                if task.claims:
                    released = self._runtime.client.release(task.claims)
                    if len(released) != len(task.claims) or not all(
                        success for _, success in released
                    ):
                        raise RuntimeError("KVCR offload claim release failed")
                self._completed_offloads.put((task.success, task.keys))
                self._offloads.popleft()

    def _complete_layer(self, batch, layer):
        if not batch.error and batch.remaining[layer] == 0:
            logger.debug(
                "KVCR batch=%d layer=%d ready_ms=%.3f",
                batch.index,
                layer,
                (time.perf_counter() - batch.started) * 1000,
            )
            self.layer_done_counter.complete(batch.index, layer)

    def num_completed_loads(self):
        self._raise_if_failed()
        return self._completed_loads.qsize()

    def pop_completed_load(self):
        self._raise_if_failed()
        return self._completed_loads.get_nowait()

    def num_completed_offloads(self):
        self._raise_if_failed()
        return self._completed_offloads.qsize()

    def pop_completed_offload(self):
        self._raise_if_failed()
        success, keys = self._completed_offloads.get_nowait()
        for key in keys:
            self._unacknowledged_keys[key] -= 1
            if not self._unacknowledged_keys[key]:
                del self._unacknowledged_keys[key]
        return success

    def pop_storage_removals(self):
        self._raise_if_failed()
        # A delayed rank-agreed acknowledgement must not resurrect an eviction.
        with self._runtime.inventory_lock:
            return [
                self._runtime.removals.pop(key)
                for key in list(self._runtime.removals)
                if key not in self._unacknowledged_keys
            ]

    def reset(self):
        self._call(self._reset)

    def _reset(self):
        self._preparer.shutdown(wait=True, cancel_futures=True)
        self._runtime.close()
        removals = self._runtime.removals
        for batch in self._batches:
            self._fail_batch(batch, RuntimeError("KVCR linker reset"))
        self._loads.clear()
        self._queued.clear()
        self._batches.clear()
        self._offloads.clear()
        self._unacknowledged_keys.clear()
        self._operations.clear()
        self._hazards.clear()
        self._scopes.clear()
        self._request_scopes.clear()
        self._released_scopes.clear()
        self._completed_loads = Queue()
        self._completed_offloads = Queue()
        self.layer_done_counter.reset()
        self._runtime = self._make_runtime()
        self._preparer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kvcr-offload-prepare"
        )
        # Engine reset must not drop pending removals from retained inventory.
        self._runtime.removals.update(removals)

    def close(self):
        if not self._closed:
            self._call(self._close)
            self._thread.join()

    def _close(self):
        self._preparer.shutdown(wait=True, cancel_futures=True)
        self._runtime.close()
        for batch in self._batches:
            self._fail_batch(batch, RuntimeError("KVCR linker closed"))
        with self._command_lock:
            self._closed = True

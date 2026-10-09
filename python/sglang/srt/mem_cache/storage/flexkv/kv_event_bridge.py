"""Forward FlexKV's host-tier placement events onto sglang's KV event wire.

A KV-aware router (llm-d, dynamo) subscribes to one publisher per engine and
believes what it says about what it holds. Everything sglang offloads into
FlexKV is, from the router's side of that socket, simply gone — so a prefix
that is one host-memory read away looks exactly like one that has to be
prefilled from scratch, and the router sends the request somewhere else.

This bridge closes that gap without giving the router a second thing to talk
to: FlexKV's tier reports its own stores and evictions, sglang relays them on
the channel it already publishes, and the router keeps seeing a single engine
that simply has a larger cache than its device pool.

Two asymmetries shape the payload:

* A store carries ``token_ids``; a removal does not. The router hashes
  content its own way, so equating its key with FlexKV's requires the tokens
  — but only once. The store establishes the mapping and the removal, which
  happens where no tokens survive, is resolved through it.
* The medium is ``CPU``, not ``GPU``. These blocks cost a host-to-device
  copy, and a router that scores tiers separately should be able to tell.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sglang.srt.disaggregation.kv_events import StorageMedium

logger = logging.getLogger(__name__)


class FlexKVEventBridge:
    """Pumps :mod:`flexkv.common.kv_events` events into a recorder.

    Inert unless both sides are on: sglang publishing KV events at all, and
    a FlexKV tier that can report evictions as well as stores. Half of that
    — stores with no evictions — is worse than nothing, because the router's
    view would then only ever grow.
    """

    def __init__(self, connector: Any, recorder: Any, page_size: int) -> None:
        self._connector = connector
        self._recorder = recorder
        self._page_size = int(page_size)
        self._active = False
        self._warned_block_size = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        """Arm the bridge if both sides support it. Safe to call twice."""
        if self._active:
            return
        if self._recorder is None or not getattr(self._recorder, "enabled", False):
            return
        start = getattr(self._connector, "start_kv_events", None)
        if start is None:
            return
        try:
            self._active = bool(start())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FlexKV] KV event bridge failed to start: %s", exc)
            return
        if self._active:
            logger.info(
                "[FlexKV] KV event bridge active: host-tier blocks are "
                "published as medium=%s",
                StorageMedium.CPU.value,
            )

    def drain(self) -> None:
        """Move one batch of FlexKV events into the recorder. Non-blocking."""
        if not self._active:
            return
        try:
            events = self._connector.take_kv_events()
        except Exception as exc:  # noqa: BLE001 - never fail a scheduler tick
            logger.warning("[FlexKV] KV event drain failed: %s", exc)
            return
        for event in events:
            try:
                self._forward(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[FlexKV] dropping malformed KV event %s: %s",
                    type(event).__name__,
                    exc,
                )

    def _forward(self, event: Any) -> None:
        name = type(event).__name__
        if name == "BlocksStored":
            self._check_block_size(event.block_size)
            self._recorder.record_external_store(
                block_hashes=event.block_hashes,
                token_ids=event.token_ids,
                parent_block_hash=event.parent_block_hash,
                medium=_medium(event.medium),
            )
        elif name == "BlocksRemoved":
            self._recorder.record_external_remove(
                block_hashes=event.block_hashes,
                medium=_medium(event.medium),
            )
        elif name == "AllBlocksCleared":
            # FlexKV lost track of what it told us it had (tier reset, or an
            # eviction ring that lapped). This retracts the device tier too,
            # which is heavy-handed but is the only correct thing on this
            # wire: the protocol has no per-tier clear.
            logger.warning("[FlexKV] host tier requested a KV event resync")
            self._recorder.record_all_cleared()
        else:
            raise ValueError(f"unknown event type {name}")

    def _check_block_size(self, block_size: int) -> None:
        """A hash covers ``block_size`` tokens on FlexKV's side; the router
        assumes it covers ``page_size``. They agree by construction (the
        connector sets ``tokens_per_block = page_size``), so a mismatch is a
        configuration bug that would silently corrupt every hit estimate."""
        if int(block_size) == self._page_size or self._warned_block_size:
            return
        self._warned_block_size = True
        logger.error(
            "[FlexKV] KV event block_size=%d does not match sglang "
            "page_size=%d; a router will mis-score every host-tier hit",
            block_size,
            self._page_size,
        )


def _medium(value: Optional[str]) -> StorageMedium:
    """FlexKV names its tier in its own vocabulary ('CPU'); the wire wants
    sglang's enum, whose CPU member serializes as 'CPU_PINNED'."""
    if value is None:
        return StorageMedium.CPU
    name = str(value).strip().upper()
    if name in ("CPU", "CPU_PINNED", "HOST"):
        return StorageMedium.CPU
    if name == "DISK" or name == "SSD":
        return StorageMedium.DISK
    if name in ("EXTERNAL", "REMOTE"):
        return StorageMedium.EXTERNAL
    if name == "GPU":
        return StorageMedium.GPU
    return StorageMedium.CPU

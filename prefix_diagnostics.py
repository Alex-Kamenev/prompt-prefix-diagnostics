"""Compare leading system-message/tool-schema fingerprints, not model caches."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["PrefixReport", "PrefixTracker", "prefix_fingerprint"]

_MAX_ID_BYTES = 256
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 100_000
_DEFAULT_PREFIX_BYTES = 1_048_576


def _positive_integer(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_json(value: Any, *, string_limit: int | None = None) -> None:
    """Reject coercion, non-finite numbers, cycles and excessive nesting."""
    active: set[int] = set()
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise ValueError("JSON structure exceeds the diagnostic limit")
        kind = type(item)
        if item is None or kind in (bool, int):
            return
        if kind is float:
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            return
        if kind is str:
            if string_limit is not None and len(item) > string_limit:
                raise ValueError("Prefix exceeds max_prefix_bytes")
            return
        if kind not in (list, dict):
            raise TypeError("Only built-in JSON value types are supported")
        identity = id(item)
        if identity in active:
            raise ValueError("JSON structures must not contain cycles")
        active.add(identity)
        try:
            if kind is list:
                for child in item:
                    visit(child, depth + 1)
            else:
                for key, child in item.items():
                    if type(key) is not str:
                        raise TypeError("JSON object keys must be strings")
                    visit(key, depth + 1)
                    visit(child, depth + 1)
        finally:
            active.remove(identity)

    visit(value, 0)


def prefix_fingerprint(
    messages: list[dict[str, Any]],
    tools: list[Any] | None = None,
    *,
    max_prefix_bytes: int = _DEFAULT_PREFIX_BYTES,
) -> str:
    """Return a SHA-256 digest of selected JSON fields, without retaining them.

    Only the leading consecutive messages with role exactly ``system`` and the
    supplied tool list are fingerprinted. ``None`` tools means an empty list.
    Object key order is ignored; array order and exact string values matter.
    Invalid or oversized inputs raise generic errors, without string coercion.
    """
    _positive_integer(max_prefix_bytes, "max_prefix_bytes")
    if type(messages) is not list:
        raise TypeError("messages must be a JSON array")
    if tools is not None and type(tools) is not list:
        raise TypeError("tools must be a JSON array or None")
    _validate_json(messages)
    for message in messages:
        if type(message) is not dict or type(message.get("role")) is not str:
            raise TypeError("Each message must be an object with a string role")

    leading = []
    for message in messages:
        if message["role"] != "system":
            break
        leading.append(message)
    payload = {
        "format": "leading-system-and-tools/v1",
        "system_messages": leading,
        "tools": [] if tools is None else tools,
    }
    _validate_json(payload, string_limit=max_prefix_bytes)
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    digest = hashlib.sha256()
    length = 0
    try:
        for text in encoder.iterencode(payload):
            chunk = text.encode("ascii")
            length += len(chunk)
            if length > max_prefix_bytes:
                raise ValueError("Prefix exceeds max_prefix_bytes")
            digest.update(chunk)
    except (OverflowError, RecursionError) as error:
        raise ValueError("JSON value cannot be canonicalized") from error
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PrefixReport:
    """An observation; the digest is not a privacy or cache guarantee."""

    status: Literal["new", "stable", "changed"]
    fingerprint: str


class PrefixTracker:
    """Thread-safe, bounded LRU history keyed by explicit conversation and model."""

    def __init__(
        self, *, max_tracked: int = 128, max_prefix_bytes: int = _DEFAULT_PREFIX_BYTES
    ) -> None:
        self._max = _positive_integer(max_tracked, "max_tracked")
        self._prefix_limit = _positive_integer(max_prefix_bytes, "max_prefix_bytes")
        self._seen: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _identity(value: str) -> str:
        if type(value) is not str or not value:
            raise ValueError("Identifiers must be nonempty strings")
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("Identifiers must be valid UTF-8 strings") from error
        if len(encoded) > _MAX_ID_BYTES:
            raise ValueError("Identifiers must not exceed 256 UTF-8 bytes")
        return value

    def observe(
        self,
        *,
        conversation_id: str,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> PrefixReport:
        """Compare with the last valid observation for this exact identity pair."""
        key = (self._identity(conversation_id), self._identity(model))
        fingerprint = prefix_fingerprint(
            messages, tools, max_prefix_bytes=self._prefix_limit
        )
        with self._lock:
            old = self._seen.get(key)
            self._seen[key] = fingerprint
            self._seen.move_to_end(key)
            if len(self._seen) > self._max:
                self._seen.popitem(last=False)
        status = "new" if old is None else "stable" if old == fingerprint else "changed"
        return PrefixReport(status, fingerprint)

    def clear(self) -> None:
        """Discard all remembered identities and digests."""
        with self._lock:
            self._seen.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)

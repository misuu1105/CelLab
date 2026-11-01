"""Canonical byte keys for WorldState snapshots, used for exact cycle detection."""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np

from cel_core.types import WorldState

# Grids with at most this many cells use the raw payload as the key; larger
# grids use a 16-byte BLAKE2b digest of the payload.
RAW_KEY_MAX_CELLS = 65536
DIGEST_SIZE = 16

_GRID_RAW = 0
_GRID_PACKED = 1


def _canonical(obj: Any) -> Any:
    """Convert a nested rng_state structure into a deterministic, repr-stable form."""
    if isinstance(obj, dict):
        return tuple(sorted((str(k), _canonical(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_canonical(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return (str(obj.dtype), obj.shape, obj.tobytes())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def state_payload(state: WorldState, stochastic: bool) -> bytes:
    """Serialize the dynamics-relevant parts of `state` into bytes.

    Layout: header (H, W, grid encoding, halted, agent count), grid bytes
    (bit-packed when every cell is 0 or 1), agent (y, x, direction) tuples, and,
    when `stochastic`, the canonical repr of `rng_state`. The step counter is
    excluded so that states at different times compare equal.
    """
    grid = np.ascontiguousarray(state.grid, dtype=np.uint8)
    h, w = grid.shape
    binary = grid.size == 0 or int(grid.max()) <= 1
    grid_bytes = np.packbits(grid, axis=None).tobytes() if binary else grid.tobytes()

    parts = [
        struct.pack(
            "<IIBBI",
            h,
            w,
            _GRID_PACKED if binary else _GRID_RAW,
            int(state.halted),
            len(state.agents),
        ),
        grid_bytes,
    ]
    for a in state.agents:
        parts.append(struct.pack("<qqB", a.y, a.x, int(a.direction)))
    if stochastic:
        parts.append(repr(_canonical(state.rng_state)).encode())
    return b"".join(parts)


def state_key(state: WorldState, stochastic: bool) -> bytes:
    """Return a hashable key identifying `state` for cycle detection.

    The raw payload is used when H*W <= RAW_KEY_MAX_CELLS (collision-free);
    otherwise the key is a 16-byte BLAKE2b digest of the payload.
    """
    payload = state_payload(state, stochastic)
    h, w = state.grid.shape
    if h * w <= RAW_KEY_MAX_CELLS:
        return payload
    return hashlib.blake2b(payload, digest_size=DIGEST_SIZE).digest()


__all__ = ["DIGEST_SIZE", "RAW_KEY_MAX_CELLS", "state_key", "state_payload"]


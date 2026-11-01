"""Initial world generation: named presets and seeded random grids."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from cel_core.types import Agent, Direction, WorldState

SeedLike = int | np.random.Generator | None

# Tolerance used when validating that state densities sum to 1.
_DENSITY_SUM_TOL = 1e-9


def _parse_pattern(rows: Sequence[str]) -> np.ndarray:
    """Convert ASCII rows to a uint8 array.

    '.' -> 0, '#'/'O' -> 1, digits -> their value. All rows must have equal length.
    """
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("All pattern rows must have the same length")
    lut = {".": 0, "#": 1, "O": 1}
    out = np.zeros((len(rows), width), dtype=np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            out[y, x] = int(ch) if ch.isdigit() else lut[ch]
    return out


@dataclass(frozen=True)
class Preset:
    """A named initial pattern.

    `cells` is the pattern's bounding box; `agents` hold positions relative to
    the pattern's top-left corner.
    """

    name: str
    cells: np.ndarray
    agents: tuple[Agent, ...] = field(default_factory=tuple)
    description: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.cells.shape[0]), int(self.cells.shape[1]))


PRESETS: dict[str, Preset] = {
    p.name: p
    for p in (
        Preset(
            "block",
            _parse_pattern(["##", "##"]),
            description="Game of Life still life (2x2).",
        ),
        Preset(
            "blinker",
            _parse_pattern(["###"]),
            description="Game of Life period-2 oscillator, horizontal phase.",
        ),
        Preset(
            "glider",
            _parse_pattern([".#.", "..#", "###"]),
            description="Game of Life spaceship moving south-east by (1, 1) every 4 steps.",
        ),
        Preset(
            "r_pentomino",
            _parse_pattern([".##", "##.", ".#."]),
            description="Game of Life methuselah; stabilizes after 1103 steps on an unbounded plane.",
        ),
        Preset(
            "pulsar",
            _parse_pattern(
                [
                    "..###...###..",
                    ".............",
                    "#....#.#....#",
                    "#....#.#....#",
                    "#....#.#....#",
                    "..###...###..",
                    ".............",
                    "..###...###..",
                    "#....#.#....#",
                    "#....#.#....#",
                    "#....#.#....#",
                    ".............",
                    "..###...###..",
                ]
            ),
            description="Game of Life period-3 oscillator (13x13).",
        ),
        Preset(
            "pentadecathlon",
            _parse_pattern(
                [
                    "..#....#..",
                    "##.####.##",
                    "..#....#..",
                ]
            ),
            description="Game of Life period-15 oscillator (3x10 phase).",
        ),
        Preset(
            "single_ant",
            _parse_pattern(["."]),
            agents=(Agent(y=0, x=0, direction=Direction.N),),
            description="Langton's Ant: one north-facing ant on a WHITE cell.",
        ),
        Preset(
            "brian_seed",
            _parse_pattern(["##", "##"]),
            description="Brian's Brain: 2x2 FIRING block that emits four gliders.",
        ),
    )
}


def list_presets() -> list[str]:
    """Return the names of all registered presets."""
    return sorted(PRESETS)


def get_preset(name: str) -> Preset:
    """Look up a preset by case-insensitive name; '-' and ' ' are treated as '_'."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return PRESETS[key]
    except KeyError:
        raise ValueError(f"Unknown preset {name!r}; available: {list_presets()}") from None


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"shape must be (H, W), got {shape!r}")
    h, w = shape
    for v in (h, w):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0:
            raise ValueError(f"shape dimensions must be positive integers, got {shape!r}")
    return int(h), int(w)


def stamp(
    grid: np.ndarray,
    pattern: np.ndarray,
    offset: tuple[int, int] = (0, 0),
    *,
    wrap: bool = False,
) -> np.ndarray:
    """Return a copy of `grid` with `pattern` written at `offset` = (y, x) of its top-left.

    With `wrap=False` the pattern must fit inside the grid; with `wrap=True`
    coordinates are taken modulo the grid shape (toroidal placement).
    """
    out = np.array(grid, dtype=np.uint8, copy=True)
    pat = np.asarray(pattern, dtype=np.uint8)
    if out.ndim != 2 or pat.ndim != 2:
        raise ValueError("grid and pattern must be 2D")
    h, w = out.shape
    ph, pw = pat.shape
    oy, ox = int(offset[0]), int(offset[1])

    if wrap:
        if ph > h or pw > w:
            raise ValueError(f"Pattern of shape {pat.shape} does not fit in grid {out.shape}")
        rows = (np.arange(ph) + oy) % h
        cols = (np.arange(pw) + ox) % w
        out[np.ix_(rows, cols)] = pat
        return out

    if oy < 0 or ox < 0 or oy + ph > h or ox + pw > w:
        raise ValueError(
            f"Pattern of shape {pat.shape} at offset {(oy, ox)} exceeds grid {out.shape}; "
            "use wrap=True for toroidal placement"
        )
    out[oy : oy + ph, ox : ox + pw] = pat
    return out


def place_preset(
    name: str,
    shape: tuple[int, int],
    offset: tuple[int, int] | None = None,
    *,
    wrap: bool = False,
    background: int = 0,
    direction: Direction | int | None = None,
) -> WorldState:
    """Build a WorldState of `shape` = (H, W) with preset `name` placed at `offset`.

    `offset` is the (y, x) of the pattern's top-left corner; None centers it.
    `direction` overrides the initial direction of preset agents.
    """
    h, w = _validate_shape(shape)
    preset = get_preset(name)
    ph, pw = preset.shape
    if offset is None:
        if ph > h or pw > w:
            raise ValueError(f"Preset {preset.name!r} of shape {preset.shape} does not fit in {(h, w)}")
        offset = ((h - ph) // 2, (w - pw) // 2)

    grid = np.full((h, w), background, dtype=np.uint8)
    grid = stamp(grid, preset.cells, offset, wrap=wrap)

    oy, ox = int(offset[0]), int(offset[1])
    agents = []
    for a in preset.agents:
        y, x = a.y + oy, a.x + ox
        if wrap:
            y, x = y % h, x % w
        agents.append(Agent(y=y, x=x, direction=a.direction if direction is None else direction))
    return WorldState(grid=grid, agents=agents)


def _normalize_densities(
    state_densities: Mapping[int, float] | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (states, probabilities) with any unassigned mass given to state 0."""
    if isinstance(state_densities, Mapping):
        items = [(int(k), float(v)) for k, v in state_densities.items()]
    else:
        items = [(i, float(v)) for i, v in enumerate(state_densities)]
    if not items:
        raise ValueError("state_densities must not be empty")

    dens: dict[int, float] = {}
    for s, p in items:
        if not 0 <= s <= 255:
            raise ValueError(f"State values must be in [0, 255], got {s}")
        if not np.isfinite(p) or p < 0:
            raise ValueError(f"Density for state {s} must be a finite non-negative number, got {p}")
        dens[s] = p

    total = sum(dens.values())
    if total > 1 + _DENSITY_SUM_TOL:
        raise ValueError(f"State densities sum to {total}, which exceeds 1")
    remainder = 1.0 - total
    if remainder > _DENSITY_SUM_TOL:
        if 0 in dens:
            raise ValueError(
                f"State densities sum to {total} < 1 while state 0 is specified explicitly"
            )
        dens[0] = remainder

    states = np.array(sorted(dens), dtype=np.uint8)
    probs = np.array([dens[s] for s in sorted(dens)], dtype=np.float64)
    return states, probs / probs.sum()


def random_grid(
    shape: tuple[int, int],
    state_densities: Mapping[int, float] | Sequence[float],
    seed: SeedLike = None,
) -> np.ndarray:
    """Sample an (H, W) uint8 grid where each cell is drawn i.i.d. from `state_densities`.

    `state_densities` is either {state: probability} or a sequence indexed by state.
    If the probabilities sum to less than 1 and state 0 is not listed, state 0
    receives the remaining mass. The same integer `seed` always yields the same grid.
    """
    h, w = _validate_shape(shape)
    states, probs = _normalize_densities(state_densities)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    return rng.choice(states, size=(h, w), p=probs).astype(np.uint8, copy=False)


__all__ = [
    "PRESETS",
    "Preset",
    "get_preset",
    "list_presets",
    "place_preset",
    "random_grid",
    "stamp",
]


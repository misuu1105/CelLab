from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Literal

import numpy as np


class BoundaryCondition(StrEnum):
    """2D grid boundary handling modes."""

    TOROIDAL = "toroidal"
    FIXED = "fixed"
    REFLECTIVE = "reflective"


class Neighborhood(StrEnum):
    """2D cell neighborhood stencils."""

    MOORE = "moore"
    VON_NEUMANN = "von_neumann"


RegimeType = Literal[
    "fixed_point",
    "periodic",
    "no_cycle_within_limit",
    "stochastic_no_exact_cycle",
]


class Direction(IntEnum):
    N = 0
    E = 1
    S = 2
    W = 3

@dataclass
class Agent:
    """State of an agent on a 2D grid"""

    y: int
    x: int
    direction: Direction

    def __post_init__(self) -> None:
        self.y = int(self.y)
        self.x = int(self.x)
        self.direction = Direction(self.direction % 4)

    def copy(self) -> Agent:
        return Agent(y=self.y, x=self.x, direction=self.direction)

    def to_dict(self) -> dict[str, int]:
        return {"y": self.y, "x": self.x, "direction": int(self.direction)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        return cls(
            y=int(data["y"]),
            x=int(data["x"]),
            direction=data["direction"],
        )


@dataclass
class WorldState:
    """Complete snapshot of a 2D cellular automaton at a discrete time step."""

    grid: np.ndarray
    agents: list[Agent] = field(default_factory=list)
    step: int = 0
    rng_state: dict[str, Any] | None = None
    halted: bool = False

    def __post_init__(self) -> None:
        arr = np.asarray(self.grid, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"WorldState.grid must be 2D, got shape {arr.shape}")
        self.grid = arr
        self.step = int(self.step)
        self.halted = bool(self.halted)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.grid.shape[0]), int(self.grid.shape[1]))

    def copy(self) -> WorldState:
        return WorldState(
            grid=self.grid.copy(),
            agents=[a.copy() for a in self.agents],
            step=self.step,
            rng_state=copy.deepcopy(self.rng_state),
            halted=self.halted,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to a JSON-compatible dictionary."""
        return {
            "grid": self.grid.tolist(),
            "agents": [a.to_dict() for a in self.agents],
            "step": self.step,
            "rng_state": copy.deepcopy(self.rng_state),
            "halted": self.halted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldState:
        """Restore state from a dictionary produced by to_dict."""
        grid = np.asarray(data["grid"], dtype=np.uint8)
        agents = [Agent.from_dict(a) for a in data.get("agents", [])]
        return cls(
            grid=grid,
            agents=agents,
            step=int(data.get("step", 0)),
            rng_state=copy.deepcopy(data.get("rng_state")),
            halted=bool(data.get("halted", False)),
        )


@dataclass
class QueryResultQ1:
    """Q1 result: automaton state at step N."""

    step: int
    state: WorldState


@dataclass
class QueryResultQ2:
    """Q2 result: static equilibrium detection."""

    reached: bool
    step: int | None


@dataclass
class QueryResultQ3:
    """Q3 result: target pattern or full-grid occurrence."""

    found: bool
    first_step: int | None
    positions: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class QueryResultQ4:
    """Q4 result: count of cells in target state at step N."""

    step: int
    state_value: int
    count: int


@dataclass
class QueryResultQ5:
    """Q5 result: cycle detection (transient T, period L) and dynamical regime."""

    has_cycle: bool
    transient_t: int | None
    period_l: int | None
    regime: RegimeType
"""Abstract rule interface and RNG state helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from cel_core.spec import BaseAutomatonSpec
from cel_core.types import WorldState


def make_rng_state(seed: int | None) -> dict[str, Any]:
    """Return the bit-generator state dict of a fresh `Generator(PCG64(seed))`."""
    return np.random.Generator(np.random.PCG64(seed)).bit_generator.state


def generator_from_state(rng_state: dict[str, Any]) -> np.random.Generator:
    """Rebuild a Generator from a stored bit-generator state dict without mutating the dict."""
    name = rng_state.get("bit_generator", "PCG64")
    bit_gen_cls = getattr(np.random, name, None)
    if bit_gen_cls is None:
        raise ValueError(f"Unknown bit generator in rng_state: {name!r}")
    bit_gen = bit_gen_cls()
    bit_gen.state = rng_state
    return np.random.Generator(bit_gen)


class BaseRule(ABC):
    """Pure transition function S_t -> S_{t+1} built from a RuleSpec.

    `step` never mutates its input; it always returns a new WorldState with
    `step` incremented by one. A halted state maps to an identical copy
    (fixed point) with only the step counter advanced.
    """

    spec_type: ClassVar[type[BaseAutomatonSpec]] = BaseAutomatonSpec

    def __init__(self, spec: BaseAutomatonSpec) -> None:
        if not isinstance(spec, self.spec_type):
            raise TypeError(
                f"{type(self).__name__} requires {self.spec_type.__name__}, "
                f"got {type(spec).__name__}"
            )
        self.spec = spec

    @property
    def is_stochastic(self) -> bool:
        return self.spec.is_stochastic

    def step(self, state: WorldState) -> WorldState:
        """Return the successor of `state` without modifying it."""
        if state.halted:
            nxt = state.copy()
            nxt.step = state.step + 1
            return nxt
        return self._transition(state)

    @abstractmethod
    def _transition(self, state: WorldState) -> WorldState:
        """Compute the successor of a non-halted state."""

    def _validate_grid_values(self, grid: np.ndarray) -> None:
        if grid.size and int(grid.max()) >= self.spec.num_states:
            raise ValueError(
                f"Grid contains state {int(grid.max())} outside [0, {self.spec.num_states - 1}]"
            )


"""Outer-totalistic rules: Life-like (B/S) and Generations (B/S/C)."""

from __future__ import annotations

import copy

import numpy as np

from cel_core.grid import neighbor_count
from cel_core.rules.base import BaseRule
from cel_core.spec import BaseAutomatonSpec, GenerationsSpec, LifeLikeSpec
from cel_core.types import WorldState


def _count_lut(counts: set[int], max_neighbors: int) -> np.ndarray:
    """Boolean lookup table indexed by neighbor count."""
    lut = np.zeros(max_neighbors + 1, dtype=bool)
    for c in counts:
        lut[c] = True
    return lut


class _TotalisticRule(BaseRule):
    """Shared neighbor counting of state-1 ("alive"/"firing") cells."""

    def __init__(self, spec: BaseAutomatonSpec) -> None:
        super().__init__(spec)
        self._birth_lut = _count_lut(spec.birth, spec.max_neighbors)
        self._survive_lut = _count_lut(spec.survival, spec.max_neighbors)

    def _alive_counts(self, grid: np.ndarray) -> np.ndarray:
        return neighbor_count(
            grid,
            state=1,
            neighborhood=self.spec.neighborhood,
            boundary=self.spec.boundary,
            fixed_value=self.spec.fixed_boundary_state,
        )

 
class LifeLikeRule(_TotalisticRule):
    """2-state rule: dead -> alive if count in B; alive stays alive if count in S."""

    spec_type = LifeLikeSpec

    def _transition(self, state: WorldState) -> WorldState:
        grid = state.grid
        self._validate_grid_values(grid)
        counts = self._alive_counts(grid)
        alive = grid == 1
        born = ~alive & self._birth_lut[counts]
        survives = alive & self._survive_lut[counts]
        new_grid = (born | survives).astype(np.uint8)
        return WorldState(
            grid=new_grid,
            agents=[a.copy() for a in state.agents],
            step=state.step + 1,
            rng_state=copy.deepcopy(state.rng_state),
            halted=False,
        )


class GenerationsRule(_TotalisticRule):
    """C-state rule: 0=off, 1=firing, 2..C-1=refractory.

    off -> firing if count in B; firing stays firing if count in S, else -> 2;
    refractory k -> k+1, and C-1 -> off. Only firing cells are counted.
    """

    spec_type = GenerationsSpec

    def _transition(self, state: WorldState) -> WorldState:
        grid = state.grid
        self._validate_grid_values(grid)
        c = self.spec.num_states
        counts = self._alive_counts(grid)

        new_grid = np.zeros_like(grid, dtype=np.uint8)
        off = grid == 0
        firing = grid == 1
        refractory = grid >= 2

        new_grid[off & self._birth_lut[counts]] = 1
        stays = firing & self._survive_lut[counts]
        new_grid[stays] = 1
        new_grid[firing & ~stays] = 2
        aged = grid[refractory].astype(np.int32) + 1
        aged[aged >= c] = 0
        new_grid[refractory] = aged.astype(np.uint8)

        return WorldState(
            grid=new_grid,
            agents=[a.copy() for a in state.agents],
            step=state.step + 1,
            rng_state=copy.deepcopy(state.rng_state),
            halted=False,
        )

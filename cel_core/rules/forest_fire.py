"""Drossel-Schwabl Forest Fire rule (states: 0=EMPTY, 1=TREE, 2=FIRE)."""

from __future__ import annotations

import copy

import numpy as np

from cel_core.grid import neighbor_count
from cel_core.rules.base import BaseRule, generator_from_state, make_rng_state
from cel_core.spec import ForestFireSpec
from cel_core.types import WorldState

EMPTY, TREE, FIRE = 0, 1, 2


class ForestFireRule(BaseRule):
    """FIRE -> EMPTY; TREE -> FIRE if >= min_fire_neighbors burning neighbors or u < q;
    EMPTY -> TREE if u < p.

    One uniform array of shape (H, W) is drawn per step from the Generator
    rebuilt from `state.rng_state`; the advanced generator state is written to
    the returned state. If `state.rng_state` is None, the generator is seeded
    from `spec.seed`. When p = q = 0 the rule is deterministic: no draws are made
    and `rng_state` is passed through unchanged.
    """

    spec_type = ForestFireSpec

    def _transition(self, state: WorldState) -> WorldState:
        grid = state.grid
        self._validate_grid_values(grid)
        spec = self.spec

        fire_counts = neighbor_count(
            grid,
            state=FIRE,
            neighborhood=spec.neighborhood,
            boundary=spec.boundary,
            fixed_value=spec.fixed_boundary_state,
        )
        trees = grid == TREE
        ignite = trees & (fire_counts >= spec.min_fire_neighbors)
        grow = np.zeros_like(trees)

        if spec.is_stochastic:
            rng_state = state.rng_state if state.rng_state is not None else make_rng_state(spec.seed)
            gen = generator_from_state(rng_state)
            u = gen.random(grid.shape)
            ignite |= trees & (u < spec.p_lightning)
            grow = (grid == EMPTY) & (u < spec.p_growth)
            new_rng_state = gen.bit_generator.state
        else:
            new_rng_state = copy.deepcopy(state.rng_state)

        new_grid = grid.copy()
        new_grid[grid == FIRE] = EMPTY
        new_grid[ignite] = FIRE
        new_grid[grow] = TREE

        return WorldState(
            grid=new_grid,
            agents=[a.copy() for a in state.agents],
            step=state.step + 1,
            rng_state=new_rng_state,
            halted=False,
        )


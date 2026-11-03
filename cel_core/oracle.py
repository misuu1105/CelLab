"""Analytical queries (Q1-Q5) over a SimulationEngine."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from cel_core.engine import CycleTracker, SimulationEngine
from cel_core.hashing import state_key
from cel_core.types import (
    BoundaryCondition,
    QueryResultQ2,
    QueryResultQ3,
    QueryResultQ4,
    QueryResultQ5,
    WorldState,
)

WILDCARD = -1


def _check_steps(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _same_dynamics(a: WorldState, b: WorldState) -> bool:
    """True when grid, agents and halt flag match (step counter and RNG are ignored)."""
    return a.halted == b.halted and a.agents == b.agents and np.array_equal(a.grid, b.grid)


class AutomataOracle:
    """Answers the five analytical questions for the automaton run by `engine`.

    Queries that scan trajectories (stability, occurrence, cycle) replay the
    engine's initial state with the engine's rule, using their own cycle
    tracker, so they never modify the engine. A scan stops as soon as a state
    repeats: the dynamics are then periodic and no new state can appear.
    Steps are absolute, as in `SimulationEngine`.
    """

    def __init__(self, engine: SimulationEngine) -> None:
        self.engine = engine

    # ------------------------------------------------------------------- scan

    def _scan(self, until_step: int) -> Iterator[tuple[WorldState, CycleTracker]]:
        """Yield (state, tracker) for each step from the origin up to `until_step`.

        The last yielded state is the one that closed a cycle when
        `tracker.has_cycle`; it repeats an earlier state. When the tracker
        reaches its state limit (`tracker.exhausted`) stepping continues
        untracked; consumers may stop on it.
        """
        engine = self.engine
        stochastic = engine.stochastic
        tracker = CycleTracker(engine.tracker.max_tracked_states)
        state = engine.initial_state
        tracker.observe(state_key(state, stochastic), state.step)
        yield state, tracker
        while state.step < until_step and not tracker.has_cycle:
            state = engine.rule.step(state)
            tracker.observe(state_key(state, stochastic), state.step)
            yield state, tracker

    # --------------------------------------------------------------------- Q1

    def state_at(self, n: int) -> WorldState:
        """Q1: the state at absolute step `n`."""
        return self.engine.state_at(n)

    # --------------------------------------------------------------------- Q2

    def stability(self, max_steps: int) -> QueryResultQ2:
        """Q2: first step t <= origin + max_steps where state[t] equals state[t-1].

        Only grid, agents and halt flag are compared. For stochastic rules a
        one-step repeat is reported as observed, not as a permanent equilibrium.
        """
        _check_steps(max_steps, "max_steps")
        origin = self.engine.initial_state.step
        prev: WorldState | None = None
        for state, tracker in self._scan(origin + max_steps):
            if prev is not None and _same_dynamics(prev, state):
                return QueryResultQ2(True, state.step)
            if tracker.has_cycle:
                break
            prev = state
        return QueryResultQ2(False, None)

    # --------------------------------------------------------------------- Q3

    def occurs(self, pattern: np.ndarray, n: int, full_grid: bool = False) -> QueryResultQ3:
        """Q3: first step in [origin, n] whose grid contains `pattern`.

        With `full_grid=True` the whole grid must equal `pattern`. Otherwise
        `pattern` is searched as a sub-grid at every top-left position; windows
        wrap around the edges when the boundary is toroidal. Entries equal to
        -1 match any cell. A pattern larger than the grid never matches.
        `positions` lists the (y, x) top-left corners at the first matching step.
        """
        _check_steps(n, "n")
        pat = np.asarray(pattern)
        if pat.ndim != 2 or pat.size == 0:
            raise ValueError(f"pattern must be a non-empty 2D array, got shape {pat.shape}")
        pat = pat.astype(np.int16, copy=False)
        wrap = self.engine.spec.boundary == BoundaryCondition.TOROIDAL
        for state, tracker in self._scan(n):
            positions = self._match(state.grid, pat, full_grid, wrap)
            if positions:
                return QueryResultQ3(True, state.step, positions)
            if tracker.has_cycle:
                break
        return QueryResultQ3(False, None, [])

    @staticmethod
    def _match(
        grid: np.ndarray, pat: np.ndarray, full_grid: bool, wrap: bool
    ) -> list[tuple[int, int]]:
        ph, pw = pat.shape
        h, w = grid.shape
        if ph > h or pw > w:
            return []
        wildcard = pat == WILDCARD
        if full_grid:
            if pat.shape != grid.shape:
                return []
            return [(0, 0)] if bool(np.all((grid == pat) | wildcard)) else []
        g = grid.astype(np.int16, copy=False)
        if wrap:
            g = np.pad(g, ((0, ph - 1), (0, pw - 1)), mode="wrap")
        windows = sliding_window_view(g, (ph, pw))
        hits = np.all((windows == pat) | wildcard, axis=(2, 3))
        ys, xs = np.nonzero(hits)
        return [(int(y), int(x)) for y, x in zip(ys, xs)]

    # --------------------------------------------------------------------- Q4

    def count_at(self, state: int, n: int) -> QueryResultQ4:
        """Q4: number of cells equal to `state` at step `n`."""
        if isinstance(state, bool) or not isinstance(state, (int, np.integer)):
            raise ValueError(f"state must be an integer, got {state!r}")
        grid = self.engine.state_at(n).grid
        return QueryResultQ4(n, int(state), int(np.count_nonzero(grid == state)))

    # --------------------------------------------------------------------- Q5

    def cycle(self, max_steps: int) -> QueryResultQ5:
        """Q5: transient T and period L of the trajectory within `max_steps` steps.

        The regime is `fixed_point` (L == 1) or `periodic` when a state repeats.
        Otherwise it is `stochastic_no_exact_cycle` for stochastic rules and
        `no_cycle_within_limit` for deterministic ones.
        """
        _check_steps(max_steps, "max_steps")
        origin = self.engine.initial_state.step
        for _, tracker in self._scan(origin + max_steps):
            if tracker.has_cycle:
                regime = "fixed_point" if tracker.period == 1 else "periodic"
                return QueryResultQ5(True, tracker.transient, tracker.period, regime)
            if tracker.exhausted:
                break
        regime = "stochastic_no_exact_cycle" if self.engine.stochastic else "no_cycle_within_limit"
        return QueryResultQ5(False, None, None, regime)


__all__ = ["AutomataOracle"]

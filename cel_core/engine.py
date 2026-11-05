"""Simulation engine with exact cycle detection and modular fast-forward."""

from __future__ import annotations

import threading
from typing import Any

from cel_core.hashing import state_key
from cel_core.rules import BaseRule, build_rule
from cel_core.spec import BaseAutomatonSpec, parse_spec
from cel_core.types import QueryResultQ5, WorldState

DEFAULT_MAX_TRACKED_STATES = 100_000


class CycleTracker:
    """Maps state keys to the first step at which they were observed.

    When a key repeats at step t, the transient is T = first_step(key) and the
    period is L = t - T. Tracking stops after a cycle is found or once
    `max_tracked_states` distinct keys are stored (`exhausted`).
    """

    def __init__(self, max_tracked_states: int | None = DEFAULT_MAX_TRACKED_STATES) -> None:
        if max_tracked_states is not None and max_tracked_states < 1:
            raise ValueError(f"max_tracked_states must be >= 1 or None, got {max_tracked_states}")
        self.max_tracked_states = max_tracked_states
        self._seen: dict[bytes, int] = {}
        self.transient: int | None = None
        self.period: int | None = None
        self.exhausted = False

    @property
    def has_cycle(self) -> bool:
        return self.period is not None and self.period > 0

    @property
    def active(self) -> bool:
        return not self.has_cycle and not self.exhausted

    @property
    def num_tracked(self) -> int:
        return len(self._seen)

    def observe(self, key: bytes, step: int) -> bool:
        """Record `key` at `step`; return True if this observation closes a cycle."""
        if not self.active:
            return False
        first = self._seen.get(key)
        if first is not None:
            if step <= first:
                return False
            self.transient = first
            self.period = step - first
            self._seen.clear()
            return True
        if self.max_tracked_states is not None and len(self._seen) >= self.max_tracked_states:
            self.exhausted = True
            self._seen.clear()
            return False
        self._seen[key] = step
        return False


class SimulationEngine:
    """Steps a rule from an initial state while tracking state recurrence.

    Steps are absolute: `initial_state.step` is the origin and the transient
    T is reported as an absolute step index. With `store_history=True`, every
    state observed while the tracker is active is retained, so states at
    steps in [origin, T + L] are available in O(1). With `store_history=False`
    only the current state is kept and fast-forward advances (N - c) mod L
    steps from the current step c (c >= T once a cycle is known).
    """

    def __init__(
        self,
        spec: BaseAutomatonSpec | dict[str, Any] | str,
        initial_state: WorldState,
        store_history: bool = True,
        max_tracked_states: int | None = DEFAULT_MAX_TRACKED_STATES,
    ) -> None:
        self.spec = parse_spec(spec)
        self.rule: BaseRule = build_rule(self.spec)
        self.stochastic = self.rule.is_stochastic
        self.store_history = store_history
        self.tracker = CycleTracker(max_tracked_states)
        self._lock = threading.RLock()

        self._initial = initial_state.copy()
        self._origin = self._initial.step
        self._current = self._initial.copy()
        self._history: list[WorldState] = []
        self._record(self._current)

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> WorldState:
        """Copy of the current state."""
        return self._current.copy()

    @property
    def initial_state(self) -> WorldState:
        """Copy of the state the engine was created with."""
        return self._initial.copy()

    @property
    def current_step(self) -> int:
        return self._current.step

    @property
    def history(self) -> list[WorldState]:
        """Stored states, indexed by `step - initial_state.step` (empty if disabled)."""
        return self._history

    @property
    def transient(self) -> int | None:
        return self.tracker.transient

    @property
    def period(self) -> int | None:
        return self.tracker.period

    def cycle_info(self) -> QueryResultQ5:
        """Summarize the cycle detected so far as a Q5 result."""
        t = self.tracker
        if t.has_cycle:
            regime = "fixed_point" if t.period == 1 else "periodic"
            return QueryResultQ5(True, t.transient, t.period, regime)
        regime = "stochastic_no_exact_cycle" if self.stochastic else "no_cycle_within_limit"
        return QueryResultQ5(False, None, None, regime)

    # -------------------------------------------------------------- stepping

    def _record(self, state: WorldState) -> None:
        if not self.tracker.active:
            return
        closed = self.tracker.observe(state_key(state, self.stochastic), state.step)
        if (
            self.store_history
            and (closed or self.tracker.active)
            and state.step - self._origin == len(self._history)
        ):
            self._history.append(state)

    def step(self) -> WorldState:
        """Advance one step and return a copy of the new current state."""
        with self._lock:
            self._current = self.rule.step(self._current)
            self._record(self._current)
            return self._current.copy()

    def run(self, n: int) -> WorldState:
        """Advance `n` steps and return a copy of the resulting state.

        Steps one at a time until a cycle is detected, then jumps directly to
        the target step using the cycle's period.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n!r}")
        with self._lock:
            target = self._current.step + n
            while self._current.step < target:
                if self.tracker.has_cycle:
                    self._current = self._cycle_state(target)
                    break
                self._current = self.rule.step(self._current)
                self._record(self._current)
            return self._current.copy()

    def state_at(self, n: int) -> WorldState:
        """Return the state at absolute step `n`.

        Uses stored history or the detected cycle when possible; advances the
        engine when `n` lies beyond the current step; otherwise replays from
        the initial state without modifying the engine.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < self._origin:
            raise ValueError(f"n must be an integer >= {self._origin}, got {n!r}")
        with self._lock:
            cur = self._current.step
            if n == cur:
                return self._current.copy()
            idx = n - self._origin
            if idx < len(self._history):
                return self._history[idx].copy()
            if self.tracker.has_cycle and n >= self.tracker.transient:
                return self._cycle_state(n)
            if n > cur:
                return self.run(n - cur)
            return self._replay(n)

    # ----------------------------------------------------------- fast-forward

    def _cycle_state(self, n: int) -> WorldState:
        """State at step `n` >= T using the detected (T, L)."""
        t, period = self.tracker.transient, self.tracker.period
        assert t is not None and period is not None and period > 0 and n >= t
        phase_step = t + (n - t) % period
        hist_idx = phase_step - self._origin
        if self.store_history and 0 <= hist_idx < len(self._history):
            out = self._history[hist_idx].copy()
        else:
            # The current step c satisfies c >= T, so S_n = S_{c + ((n - c) mod L)}.
            out = self._current
            for _ in range((n - self._current.step) % period):
                out = self.rule.step(out)
            out = out.copy()
        out.step = n
        return out

    def _replay(self, n: int) -> WorldState:
        out = self._initial
        while out.step < n:
            out = self.rule.step(out)
        return out.copy()


__all__ = ["DEFAULT_MAX_TRACKED_STATES", "CycleTracker", "SimulationEngine"]

"""Langton's Ant / multi-color turmite rule."""

from __future__ import annotations

import copy

from cel_core.rules.base import BaseRule
from cel_core.spec import LangtonSpec
from cel_core.types import Agent, Direction, WorldState

# Direction delta per turn character: R=clockwise, L=counter-clockwise, N=none, U=u-turn.
_TURN_DELTA: dict[str, int] = {"R": 1, "L": 3, "N": 0, "U": 2}

# (dy, dx) per Direction; y grows downward.
_MOVE: dict[Direction, tuple[int, int]] = {
    Direction.N: (-1, 0),
    Direction.E: (0, 1),
    Direction.S: (1, 0),
    Direction.W: (0, -1),
}


class LangtonRule(BaseRule):
    """Each ant, in list order: turn by the rule for its cell's color, advance that
    cell's color by one (mod num_states), then move one cell forward.

    Border policy when the move would leave the grid:
    - wrap: coordinates are taken modulo the grid shape.
    - halt: the ant stays on its cell (turn and flip already applied) and the
      state is marked `halted`, after which it is a fixed point.
    - bounce: the ant stays on its cell and its direction is reversed.
    """

    spec_type = LangtonSpec

    def __init__(self, spec: LangtonSpec) -> None:
        super().__init__(spec)
        self._turns = [_TURN_DELTA[ch] for ch in spec.turn_rules]

    def _transition(self, state: WorldState) -> WorldState:
        grid = state.grid.copy()
        self._validate_grid_values(grid)
        h, w = grid.shape
        num_states = self.spec.num_states
        policy = self.spec.border_policy
        halted = False
        new_agents: list[Agent] = []

        for agent in state.agents:
            y, x = agent.y, agent.x
            if not (0 <= y < h and 0 <= x < w):
                raise ValueError(f"Agent at ({y}, {x}) is outside grid of shape {(h, w)}")

            color = int(grid[y, x])
            direction = Direction((int(agent.direction) + self._turns[color]) % 4)
            grid[y, x] = (color + 1) % num_states

            dy, dx = _MOVE[direction]
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                if policy == "wrap":
                    ny, nx = ny % h, nx % w
                elif policy == "halt":
                    ny, nx = y, x
                    halted = True
                else:  # bounce
                    ny, nx = y, x
                    direction = Direction((int(direction) + 2) % 4)
            new_agents.append(Agent(y=ny, x=nx, direction=direction))

        return WorldState(
            grid=grid,
            agents=new_agents,
            step=state.step + 1,
            rng_state=copy.deepcopy(state.rng_state),
            halted=halted,
        )


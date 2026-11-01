import unittest

import numpy as np

from cel_core.rules import (
    ForestFireRule,
    GenerationsRule,
    LangtonRule,
    LifeLikeRule,
    build_rule,
    make_rng_state,
)
from cel_core.spec import ForestFireSpec, GenerationsSpec, LangtonSpec, LifeLikeSpec
from cel_core.types import Agent, BoundaryCondition, Direction, WorldState


def run(rule, state: WorldState, n: int) -> WorldState:
    for _ in range(n):
        state = rule.step(state)
    return state


class TestBuildRule(unittest.TestCase):
    def test_dispatch_from_model_dict_and_json(self) -> None:
        self.assertIsInstance(build_rule(LifeLikeSpec()), LifeLikeRule)
        self.assertIsInstance(build_rule({"kind": "generations"}), GenerationsRule)
        self.assertIsInstance(build_rule(LangtonSpec().model_dump_json()), LangtonRule)
        self.assertIsInstance(build_rule({"kind": "forest_fire", "p": 0.1, "q": 0.0}), ForestFireRule)

    def test_rule_rejects_wrong_spec_type(self) -> None:
        with self.assertRaises(TypeError):
            LifeLikeRule(LangtonSpec())


class TestLifeLike(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = build_rule(LifeLikeSpec())  # B3/S23, toroidal

    def test_blinker_oscillates_with_period_two(self) -> None:
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 1:4] = 1  # horizontal
        vertical = np.zeros_like(grid)
        vertical[1:4, 2] = 1

        s0 = WorldState(grid=grid)
        s1 = self.rule.step(s0)
        s2 = self.rule.step(s1)
        np.testing.assert_array_equal(s1.grid, vertical)
        np.testing.assert_array_equal(s2.grid, grid)
        self.assertEqual((s1.step, s2.step), (1, 2))

    def test_glider_shifts_one_diagonal_cell_every_four_steps(self) -> None:
        grid = np.zeros((10, 10), dtype=np.uint8)
        # South-east moving glider.
        for y, x in [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
            grid[y + 2, x + 2] = 1
        s4 = run(self.rule, WorldState(grid=grid), 4)
        np.testing.assert_array_equal(s4.grid, np.roll(grid, shift=(1, 1), axis=(0, 1)))

    def test_glider_wraps_on_torus(self) -> None:
        grid = np.zeros((8, 8), dtype=np.uint8)
        for y, x in [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
            grid[y, x] = 1
        s32 = run(self.rule, WorldState(grid=grid), 32)
        np.testing.assert_array_equal(s32.grid, grid)

    def test_step_is_pure(self) -> None:
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 1:4] = 1
        s0 = WorldState(grid=grid, rng_state={"k": [1]})
        before = s0.grid.copy()
        s1 = self.rule.step(s0)
        np.testing.assert_array_equal(s0.grid, before)
        self.assertEqual(s0.step, 0)
        self.assertIsNot(s1.grid, s0.grid)
        self.assertIsNot(s1.rng_state, s0.rng_state)

    def test_fixed_boundary_blinker_at_edge(self) -> None:
        rule = build_rule(LifeLikeSpec(boundary=BoundaryCondition.FIXED))
        grid = np.zeros((4, 4), dtype=np.uint8)
        grid[0, 0:3] = 1
        s1 = rule.step(WorldState(grid=grid))
        expected = np.zeros_like(grid)
        expected[0:2, 1] = 1  # upper cell is clipped by the dead border
        np.testing.assert_array_equal(s1.grid, expected)


class TestBriansBrain(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = build_rule(GenerationsSpec(birth={2}, survival=set(), num_states=3))

    def test_isolated_cell_cycles_firing_dying_off(self) -> None:
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 2] = 1
        s1 = self.rule.step(WorldState(grid=grid))
        s2 = self.rule.step(s1)
        self.assertEqual(s1.grid[2, 2], 2)
        self.assertEqual(s2.grid[2, 2], 0)
        self.assertEqual(int(s1.grid.sum()), 2)
        self.assertEqual(int(s2.grid.sum()), 0)

    def test_birth_requires_exactly_two_firing_neighbors(self) -> None:
        grid = np.zeros((6, 6), dtype=np.uint8)
        grid[2, 2] = 1
        grid[2, 3] = 1
        s1 = self.rule.step(WorldState(grid=grid))
        expected = np.zeros_like(grid)
        expected[1, 2:4] = 1
        expected[3, 2:4] = 1
        expected[2, 2:4] = 2
        np.testing.assert_array_equal(s1.grid, expected)

    def test_every_cell_follows_off_firing_dying_order(self) -> None:
        rng = np.random.default_rng(0)
        state = WorldState(grid=rng.integers(0, 3, size=(16, 16)).astype(np.uint8))
        for _ in range(20):
            nxt = self.rule.step(state)
            prev, cur = state.grid, nxt.grid
            self.assertTrue(np.all(cur[prev == 1] == 2))
            self.assertTrue(np.all(cur[prev == 2] == 0))
            self.assertTrue(np.all(np.isin(cur[prev == 0], [0, 1])))
            state = nxt

    def test_longer_refractory_chain(self) -> None:
        rule = build_rule(GenerationsSpec(birth={2}, survival=set(), num_states=5))
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 2] = 1
        state = WorldState(grid=grid)
        seen = []
        for _ in range(4):
            state = rule.step(state)
            seen.append(int(state.grid[2, 2]))
        self.assertEqual(seen, [2, 3, 4, 0])


class TestLangtonAnt(unittest.TestCase):
    def _start(self, size=11, y=5, x=5, direction=Direction.N) -> WorldState:
        return WorldState(
            grid=np.zeros((size, size), dtype=np.uint8),
            agents=[Agent(y=y, x=x, direction=direction)],
        )

    def test_first_ten_steps_match_hand_trace(self) -> None:
        rule = build_rule(LangtonSpec(turn_rules="RL"))
        # (y, x, direction) of the ant after each step, traced by hand from (5, 5) facing N.
        expected_trace = [
            (5, 6, Direction.E),
            (6, 6, Direction.S),
            (6, 5, Direction.W),
            (5, 5, Direction.N),
            (5, 4, Direction.W),
            (4, 4, Direction.N),
            (4, 5, Direction.E),
            (5, 5, Direction.S),
            (5, 4, Direction.W),
            (6, 4, Direction.S),
        ]
        state = self._start()
        for i, (y, x, d) in enumerate(expected_trace, start=1):
            state = rule.step(state)
            ant = state.agents[0]
            self.assertEqual((ant.y, ant.x, ant.direction), (y, x, d), f"step {i}")

        expected_black = {(5, 6), (6, 6), (6, 5), (4, 4), (4, 5), (5, 5)}
        black = {(int(y), int(x)) for y, x in zip(*np.nonzero(state.grid))}
        self.assertEqual(black, expected_black)
        self.assertEqual(state.step, 10)
        self.assertFalse(state.halted)

    def test_step_does_not_mutate_input_agents(self) -> None:
        rule = build_rule(LangtonSpec())
        s0 = self._start()
        rule.step(s0)
        self.assertEqual((s0.agents[0].y, s0.agents[0].x, s0.agents[0].direction), (5, 5, Direction.N))
        self.assertEqual(int(s0.grid.sum()), 0)

    def test_wrap_policy(self) -> None:
        rule = build_rule(LangtonSpec(border_policy="wrap"))
        # White cell, facing N -> turn R to E; at the east edge, wraps to x=0.
        s1 = rule.step(self._start(size=5, y=2, x=4))
        self.assertEqual((s1.agents[0].y, s1.agents[0].x, s1.agents[0].direction), (2, 0, Direction.E))

    def test_halt_policy_is_fixed_point(self) -> None:
        rule = build_rule(LangtonSpec(border_policy="halt"))
        s1 = rule.step(self._start(size=5, y=2, x=4))
        self.assertTrue(s1.halted)
        self.assertEqual((s1.agents[0].y, s1.agents[0].x), (2, 4))
        self.assertEqual(s1.grid[2, 4], 1)
        s2 = rule.step(s1)
        np.testing.assert_array_equal(s2.grid, s1.grid)
        self.assertEqual(s2.agents, s1.agents)
        self.assertTrue(s2.halted)
        self.assertEqual(s2.step, 2)

    def test_bounce_policy_reverses_direction(self) -> None:
        rule = build_rule(LangtonSpec(border_policy="bounce"))
        s1 = rule.step(self._start(size=5, y=2, x=4))
        self.assertEqual((s1.agents[0].y, s1.agents[0].x, s1.agents[0].direction), (2, 4, Direction.W))
        self.assertFalse(s1.halted)
        # Cell (2,4) is now black -> turn L from W to S, flip to white, move to (3,4).
        s2 = rule.step(s1)
        self.assertEqual((s2.agents[0].y, s2.agents[0].x, s2.agents[0].direction), (3, 4, Direction.S))
        self.assertEqual(s2.grid[2, 4], 0)

    def test_multicolor_turmite_cycles_colors(self) -> None:
        rule = build_rule(LangtonSpec(turn_rules="RLR"))
        s1 = rule.step(self._start())
        self.assertEqual(s1.grid[5, 5], 1)

    def test_agent_outside_grid_raises(self) -> None:
        rule = build_rule(LangtonSpec())
        with self.assertRaises(ValueError):
            rule.step(self._start(size=5, y=7, x=0))


class TestForestFire(unittest.TestCase):
    def _random_forest(self, seed: int = 1) -> np.ndarray:
        return np.random.default_rng(seed).choice(3, size=(32, 32), p=[0.4, 0.55, 0.05]).astype(np.uint8)

    def test_same_seed_produces_identical_trajectory(self) -> None:
        rule = build_rule(ForestFireSpec(p_growth=0.05, p_lightning=0.001))
        grid = self._random_forest()
        a = WorldState(grid=grid, rng_state=make_rng_state(42))
        b = WorldState(grid=grid.copy(), rng_state=make_rng_state(42))
        for _ in range(50):
            a, b = rule.step(a), rule.step(b)
            np.testing.assert_array_equal(a.grid, b.grid)
            self.assertEqual(a.rng_state, b.rng_state)

    def test_different_seed_diverges(self) -> None:
        rule = build_rule(ForestFireSpec(p_growth=0.05, p_lightning=0.001))
        grid = self._random_forest()
        a = run(rule, WorldState(grid=grid, rng_state=make_rng_state(1)), 20)
        b = run(rule, WorldState(grid=grid, rng_state=make_rng_state(2)), 20)
        self.assertFalse(np.array_equal(a.grid, b.grid))

    def test_rng_state_written_back_and_input_untouched(self) -> None:
        rule = build_rule(ForestFireSpec(p_growth=0.5, p_lightning=0.5))
        s0 = WorldState(grid=self._random_forest(), rng_state=make_rng_state(7))
        snapshot = make_rng_state(7)
        s1 = rule.step(s0)
        self.assertEqual(s0.rng_state, snapshot)
        self.assertNotEqual(s1.rng_state, s0.rng_state)

    def test_resume_from_serialized_state_continues_identically(self) -> None:
        rule = build_rule(ForestFireSpec(p_growth=0.05, p_lightning=0.01))
        s = run(rule, WorldState(grid=self._random_forest(), rng_state=make_rng_state(3)), 10)
        restored = WorldState.from_dict(s.to_dict())
        np.testing.assert_array_equal(run(rule, s, 15).grid, run(rule, restored, 15).grid)

    def test_seed_from_spec_when_state_has_no_rng(self) -> None:
        rule = build_rule(ForestFireSpec(p_growth=0.05, p_lightning=0.01, seed=11))
        grid = self._random_forest()
        a = rule.step(WorldState(grid=grid))
        b = rule.step(WorldState(grid=grid, rng_state=make_rng_state(11)))
        np.testing.assert_array_equal(a.grid, b.grid)
        self.assertEqual(a.rng_state, b.rng_state)

    def test_deterministic_transitions(self) -> None:
        rule = build_rule(ForestFireSpec(p_growth=0.0, p_lightning=0.0))
        grid = np.array(
            [
                [1, 1, 0, 1],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.uint8,
        )
        s1 = rule.step(WorldState(grid=grid))
        expected = np.array(
            [
                [2, 2, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.uint8,
        )
        # Trees touching the fire at (1,1) ignite; (0,3) and (3,3) have no burning neighbor on the torus.
        np.testing.assert_array_equal(s1.grid, expected)
        self.assertIsNone(s1.rng_state)

    def test_certain_growth_and_lightning(self) -> None:
        rule = build_rule(ForestFireSpec(p_growth=1.0, p_lightning=1.0))
        grid = np.array([[0, 1, 2]], dtype=np.uint8)
        s1 = rule.step(WorldState(grid=grid, rng_state=make_rng_state(0)))
        np.testing.assert_array_equal(s1.grid, [[1, 2, 0]])


if __name__ == "__main__":
    unittest.main()

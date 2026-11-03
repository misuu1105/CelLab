import unittest

import numpy as np

from cel_core.engine import SimulationEngine
from cel_core.init_world import place_preset, random_grid
from cel_core.oracle import AutomataOracle
from cel_core.rules import build_rule
from cel_core.spec import ForestFireSpec, GenerationsSpec, LangtonSpec, LifeLikeSpec
from cel_core.types import BoundaryCondition, WorldState

LIFE = LifeLikeSpec()
BRAIN = GenerationsSpec()


def make_oracle(spec, init, **kwargs) -> AutomataOracle:
    return AutomataOracle(SimulationEngine(spec, init, **kwargs))


def brute(spec, init, n):
    rule = build_rule(spec)
    state = init
    for _ in range(n):
        state = rule.step(state)
    return state


class TestLife(unittest.TestCase):
    def test_blinker_cycle_and_no_stability(self):
        o = make_oracle(LIFE, place_preset("blinker", (7, 7)))
        res = o.cycle(50)
        self.assertEqual((res.has_cycle, res.transient_t, res.period_l, res.regime), (True, 0, 2, "periodic"))
        self.assertFalse(o.stability(50).reached)

    def test_block_fixed_point(self):
        o = make_oracle(LIFE, place_preset("block", (6, 6)))
        res = o.cycle(10)
        self.assertEqual((res.transient_t, res.period_l, res.regime), (0, 1, "fixed_point"))
        q2 = o.stability(10)
        self.assertTrue(q2.reached)
        self.assertEqual(q2.step, 1)

    def test_stability_limit(self):
        init = WorldState(grid=random_grid((16, 16), {1: 0.4}, seed=3))
        o = make_oracle(LIFE, init)
        self.assertEqual(o.stability(0).reached, False)
        res = o.cycle(2)
        self.assertEqual(res.regime, "no_cycle_within_limit")
        self.assertFalse(res.has_cycle)

    def test_glider_cycle_on_torus(self):
        o = make_oracle(LIFE, place_preset("glider", (8, 8)))
        res = o.cycle(100)
        self.assertEqual((res.transient_t, res.period_l), (0, 32))

    def test_state_at_and_count(self):
        init = WorldState(grid=random_grid((12, 12), {1: 0.35}, seed=5))
        o = make_oracle(LIFE, init)
        expected = brute(LIFE, init, 25)
        np.testing.assert_array_equal(o.state_at(25).grid, expected.grid)
        q4 = o.count_at(1, 25)
        self.assertEqual(q4.count, int(expected.grid.sum()))
        self.assertEqual(o.count_at(0, 25).count, 144 - q4.count)

    def test_occurs_subgrid_wraps(self):
        grid = np.zeros((6, 6), dtype=np.uint8)
        grid[5, 5] = grid[5, 0] = grid[0, 5] = grid[0, 0] = 1
        o = make_oracle(LIFE, WorldState(grid=grid))
        res = o.occurs(np.ones((2, 2), dtype=np.uint8), 0)
        self.assertEqual((res.found, res.first_step, res.positions), (True, 0, [(5, 5)]))

    def test_occurs_no_wrap_for_fixed_boundary(self):
        spec = LifeLikeSpec(boundary=BoundaryCondition.FIXED)
        grid = np.zeros((6, 6), dtype=np.uint8)
        grid[5, 5] = grid[5, 0] = grid[0, 5] = grid[0, 0] = 1
        o = make_oracle(spec, WorldState(grid=grid))
        self.assertFalse(o.occurs(np.ones((2, 2), dtype=np.uint8), 0).found)

    def test_occurs_glider_phase_later(self):
        o = make_oracle(LIFE, place_preset("glider", (10, 10)))
        phase = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])
        res = o.occurs(phase, 8)
        self.assertTrue(res.found)
        self.assertEqual(res.first_step, 0)
        shifted = o.state_at(4).grid
        res4 = o.occurs(shifted[1:4, 1:4], 8)
        self.assertTrue(res4.found)
        self.assertLessEqual(res4.first_step, 4)

    def test_occurs_wildcard_and_full_grid(self):
        init = place_preset("blinker", (5, 5))
        o = make_oracle(LIFE, init)
        vertical = np.zeros((5, 5), dtype=np.uint8)
        vertical[1:4, 2] = 1
        res = o.occurs(vertical, 5, full_grid=True)
        self.assertEqual((res.found, res.first_step), (True, 1))
        self.assertFalse(o.occurs(np.ones((6, 6)), 5, full_grid=True).found)
        self.assertTrue(o.occurs(np.array([[-1, 1, -1]]), 0).found)

    def test_occurs_early_stop_on_cycle(self):
        o = make_oracle(LIFE, place_preset("blinker", (5, 5)))
        steps = []
        original = o.engine.rule.step

        def counting(state):
            steps.append(1)
            return original(state)

        o.engine.rule.step = counting
        self.assertFalse(o.occurs(np.ones((4, 4)), 10_000).found)
        self.assertLess(len(steps), 10)

    def test_oracle_does_not_modify_engine(self):
        eng = SimulationEngine(LIFE, place_preset("glider", (8, 8)))
        o = AutomataOracle(eng)
        o.cycle(100)
        o.stability(100)
        o.occurs(np.ones((2, 2)), 50)
        self.assertEqual(eng.current_step, 0)

    def test_invalid_arguments(self):
        o = make_oracle(LIFE, place_preset("block", (5, 5)))
        with self.assertRaises(ValueError):
            o.cycle(-1)
        with self.assertRaises(ValueError):
            o.stability(True)
        with self.assertRaises(ValueError):
            o.occurs(np.ones(3), 1)


class TestBriansBrain(unittest.TestCase):
    def test_brian_seed_cycle_matches_brute_force(self):
        init = place_preset("brian_seed", (8, 8), wrap=True)
        o = make_oracle(BRAIN, init)
        res = o.cycle(2000)
        if res.has_cycle:
            t, period = res.transient_t, res.period_l
            a = brute(BRAIN, init, t)
            b = brute(BRAIN, init, t + period)
            np.testing.assert_array_equal(a.grid, b.grid)
            self.assertEqual(res.regime, "fixed_point" if period == 1 else "periodic")
        else:
            self.assertEqual(res.regime, "no_cycle_within_limit")

    def test_small_random_torus_cycles(self):
        init = WorldState(grid=random_grid((4, 4), {1: 0.3, 2: 0.2}, seed=2))
        o = make_oracle(BRAIN, init)
        res = o.cycle(100_000)
        self.assertTrue(res.has_cycle)
        a = brute(BRAIN, init, res.transient_t)
        b = brute(BRAIN, init, res.transient_t + res.period_l)
        np.testing.assert_array_equal(a.grid, b.grid)

    def test_count_dying_cells(self):
        init = WorldState(grid=random_grid((10, 10), {1: 0.3, 2: 0.2}, seed=9))
        o = make_oracle(BRAIN, init)
        expected = brute(BRAIN, init, 7).grid
        self.assertEqual(o.count_at(2, 7).count, int((expected == 2).sum()))

    def test_empty_grid_is_stable(self):
        o = make_oracle(BRAIN, WorldState(grid=np.zeros((5, 5), dtype=np.uint8)))
        self.assertEqual(o.stability(5).step, 1)


class TestLangtonsAnt(unittest.TestCase):
    def test_halt_policy_is_stable(self):
        spec = LangtonSpec(border_policy="halt")
        o = make_oracle(spec, place_preset("single_ant", (5, 5)))
        q2 = o.stability(10_000)
        self.assertTrue(q2.reached)
        res = o.cycle(10_000)
        self.assertEqual(res.regime, "fixed_point")
        self.assertEqual(res.period_l, 1)

    def test_wrap_torus_is_periodic(self):
        spec = LangtonSpec(border_policy="wrap")
        init = place_preset("single_ant", (4, 4))
        o = make_oracle(spec, init)
        res = o.cycle(100_000)
        self.assertEqual(res.regime, "periodic")
        a = brute(spec, init, res.transient_t)
        b = brute(spec, init, res.transient_t + res.period_l)
        np.testing.assert_array_equal(a.grid, b.grid)
        self.assertEqual(a.agents, b.agents)
        self.assertFalse(o.stability(res.transient_t + res.period_l).reached)

    def test_highway_does_not_cycle_within_limit(self):
        o = make_oracle(LangtonSpec(), place_preset("single_ant", (60, 60)))
        res = o.cycle(500)
        self.assertEqual(res.regime, "no_cycle_within_limit")

    def test_count_black_cells(self):
        spec = LangtonSpec()
        init = place_preset("single_ant", (30, 30))
        o = make_oracle(spec, init)
        self.assertEqual(o.count_at(1, 11).count, int(brute(spec, init, 11).grid.sum()))


class TestForestFire(unittest.TestCase):
    def test_deterministic_burnout_is_fixed_point(self):
        grid = np.zeros((8, 8), dtype=np.uint8)
        grid[2:5, 2:5] = 1
        grid[3, 3] = 2
        o = make_oracle(ForestFireSpec(p_growth=0.0, p_lightning=0.0), WorldState(grid=grid))
        res = o.cycle(50)
        self.assertEqual((res.regime, res.period_l), ("fixed_point", 1))
        q2 = o.stability(50)
        self.assertTrue(q2.reached)
        self.assertEqual(o.count_at(2, q2.step).count, 0)
        self.assertEqual(o.count_at(1, 50).count, 0)

    def test_stochastic_regime(self):
        spec = ForestFireSpec(p_growth=0.05, p_lightning=0.01, seed=7)
        init = WorldState(grid=random_grid((10, 10), {1: 0.5}, seed=1))
        o = make_oracle(spec, init)
        res = o.cycle(300)
        self.assertFalse(res.has_cycle)
        self.assertEqual(res.regime, "stochastic_no_exact_cycle")

    def test_stochastic_state_matches_brute_force(self):
        spec = ForestFireSpec(p_growth=0.05, p_lightning=0.01, seed=7)
        init = WorldState(grid=random_grid((10, 10), {1: 0.5}, seed=1))
        o = make_oracle(spec, init)
        np.testing.assert_array_equal(o.state_at(40).grid, brute(spec, init, 40).grid)
        self.assertEqual(o.count_at(2, 40).count, int((brute(spec, init, 40).grid == 2).sum()))

    def test_stochastic_occurs_is_reproducible(self):
        spec = ForestFireSpec(p_growth=0.05, p_lightning=0.01, seed=7)
        init = WorldState(grid=random_grid((10, 10), {1: 0.5}, seed=1))
        o = make_oracle(spec, init)
        fire = np.array([[2]])
        first = o.occurs(fire, 100)
        again = o.occurs(fire, 100)
        self.assertEqual(first, again)


if __name__ == "__main__":
    unittest.main()


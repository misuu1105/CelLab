import unittest

import numpy as np

from cel_core.init_world import PRESETS, get_preset, list_presets, place_preset, random_grid, stamp
from cel_core.rules import build_rule
from cel_core.spec import GenerationsSpec, LangtonSpec, LifeLikeSpec
from cel_core.types import Direction, WorldState


def run(rule, state: WorldState, n: int) -> WorldState:
    for _ in range(n):
        state = rule.step(state)
    return state


class TestPresetPlacement(unittest.TestCase):
    def test_all_required_presets_registered(self) -> None:
        expected = {
            "block",
            "blinker",
            "glider",
            "r_pentomino",
            "pulsar",
            "pentadecathlon",
            "single_ant",
            "brian_seed",
        }
        self.assertTrue(expected <= set(list_presets()))

    def test_live_cell_counts(self) -> None:
        counts = {
            "block": 4,
            "blinker": 3,
            "glider": 5,
            "r_pentomino": 5,
            "pulsar": 48,
            "pentadecathlon": 12,
            "single_ant": 0,
            "brian_seed": 4,
        }
        for name, n in counts.items():
            with self.subTest(name=name):
                state = place_preset(name, (20, 20), (2, 3))
                self.assertEqual(int(np.count_nonzero(state.grid)), n)
                self.assertEqual(state.shape, (20, 20))
                self.assertEqual(state.grid.dtype, np.uint8)

    def test_offset_positions_pattern_top_left(self) -> None:
        state = place_preset("glider", (8, 10), (3, 5))
        expected = np.zeros((8, 10), dtype=np.uint8)
        for y, x in [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
            expected[3 + y, 5 + x] = 1
        np.testing.assert_array_equal(state.grid, expected)

    def test_default_offset_centers_pattern(self) -> None:
        state = place_preset("block", (6, 6))
        expected = np.zeros((6, 6), dtype=np.uint8)
        expected[2:4, 2:4] = 1
        np.testing.assert_array_equal(state.grid, expected)

    def test_out_of_bounds_raises_without_wrap(self) -> None:
        with self.assertRaises(ValueError):
            place_preset("blinker", (5, 5), (0, 3))
        with self.assertRaises(ValueError):
            place_preset("block", (5, 5), (-1, 0))
        with self.assertRaises(ValueError):
            place_preset("pulsar", (10, 10))

    def test_wrap_places_toroidally(self) -> None:
        state = place_preset("blinker", (5, 5), (1, 4), wrap=True)
        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[1, [4, 0, 1]] = 1
        np.testing.assert_array_equal(state.grid, expected)

    def test_single_ant_agent_offset_and_direction(self) -> None:
        state = place_preset("single_ant", (11, 11), (5, 7))
        self.assertEqual(len(state.agents), 1)
        ant = state.agents[0]
        self.assertEqual((ant.y, ant.x, ant.direction), (5, 7, Direction.N))
        self.assertFalse(state.grid.any())

        east = place_preset("single_ant", (11, 11), (0, 0), direction=Direction.E)
        self.assertEqual(east.agents[0].direction, Direction.E)

    def test_name_lookup_and_unknown(self) -> None:
        self.assertIs(get_preset("R-Pentomino"), PRESETS["r_pentomino"])
        with self.assertRaises(ValueError):
            get_preset("gosper_gun")

    def test_invalid_shape(self) -> None:
        for shape in [(0, 5), (5, -1), (5,), (2.5, 3)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                place_preset("block", shape, (0, 0))

    def test_preset_cells_not_mutated(self) -> None:
        before = PRESETS["block"].cells.copy()
        place_preset("block", (4, 4), (0, 0), background=0)
        stamp(np.zeros((4, 4), dtype=np.uint8), PRESETS["block"].cells)
        np.testing.assert_array_equal(PRESETS["block"].cells, before)


class TestPresetDynamics(unittest.TestCase):
    def setUp(self) -> None:
        self.life = build_rule(LifeLikeSpec())

    def test_block_is_still_life(self) -> None:
        s0 = place_preset("block", (6, 6))
        np.testing.assert_array_equal(self.life.step(s0).grid, s0.grid)

    def test_oscillator_periods(self) -> None:
        for name, period in [("blinker", 2), ("pulsar", 3), ("pentadecathlon", 15)]:
            with self.subTest(name=name):
                s0 = place_preset(name, (24, 24))
                for t in range(1, period):
                    self.assertFalse(np.array_equal(run(self.life, s0, t).grid, s0.grid))
                np.testing.assert_array_equal(run(self.life, s0, period).grid, s0.grid)

    def test_glider_translates(self) -> None:
        s0 = place_preset("glider", (12, 12), (1, 1))
        s4 = run(self.life, s0, 4)
        np.testing.assert_array_equal(s4.grid, np.roll(s0.grid, shift=(1, 1), axis=(0, 1)))

    def test_brian_seed_emits_four_gliders(self) -> None:
        rule = build_rule(GenerationsSpec())
        s1 = rule.step(place_preset("brian_seed", (10, 10), (4, 4)))
        self.assertEqual(int(np.count_nonzero(s1.grid == 1)), 8)
        self.assertEqual(int(np.count_nonzero(s1.grid == 2)), 4)

    def test_single_ant_runs(self) -> None:
        rule = build_rule(LangtonSpec())
        s = run(rule, place_preset("single_ant", (9, 9)), 4)
        self.assertEqual(int(np.count_nonzero(s.grid)), 4)


class TestRandomGrid(unittest.TestCase):
    def test_same_seed_is_deterministic(self) -> None:
        a = random_grid((64, 48), {1: 0.3, 2: 0.1}, seed=123)
        b = random_grid((64, 48), {1: 0.3, 2: 0.1}, seed=123)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (64, 48))
        self.assertEqual(a.dtype, np.uint8)

    def test_different_seeds_differ(self) -> None:
        a = random_grid((64, 64), {1: 0.5}, seed=1)
        b = random_grid((64, 64), {1: 0.5}, seed=2)
        self.assertFalse(np.array_equal(a, b))

    def test_generator_seed_matches_int_seed(self) -> None:
        a = random_grid((16, 16), [0.5, 0.5], seed=7)
        b = random_grid((16, 16), [0.5, 0.5], seed=np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_density_tolerance(self) -> None:
        shape = (200, 200)
        n = shape[0] * shape[1]
        densities = {1: 0.3, 2: 0.15}
        for seed in range(5):
            grid = random_grid(shape, densities, seed=seed)
            for state, p in {**densities, 0: 0.55}.items():
                with self.subTest(seed=seed, state=state):
                    observed = np.count_nonzero(grid == state) / n
                    self.assertAlmostEqual(observed, p, delta=0.02)

    def test_sequence_densities_indexed_by_state(self) -> None:
        grid = random_grid((150, 150), [0.2, 0.5, 0.3], seed=0)
        n = grid.size
        for state, p in enumerate([0.2, 0.5, 0.3]):
            self.assertAlmostEqual(np.count_nonzero(grid == state) / n, p, delta=0.02)

    def test_extreme_densities(self) -> None:
        self.assertTrue((random_grid((10, 10), {1: 1.0}, seed=0) == 1).all())
        self.assertFalse(random_grid((10, 10), {1: 0.0}, seed=0).any())

    def test_only_listed_states_appear(self) -> None:
        grid = random_grid((50, 50), {3: 0.4, 5: 0.6}, seed=9)
        self.assertTrue(set(np.unique(grid).tolist()) <= {3, 5})

    def test_invalid_densities(self) -> None:
        bad = [
            {1: 0.7, 2: 0.4},  # sum > 1
            {1: -0.1},  # negative
            {0: 0.2, 1: 0.3},  # sum < 1 with explicit state 0
            {},  # empty
            {256: 0.5},  # state out of uint8 range
            {1: float("nan")},
        ]
        for dens in bad:
            with self.subTest(dens=dens), self.assertRaises(ValueError):
                random_grid((5, 5), dens, seed=0)


if __name__ == "__main__":
    unittest.main()


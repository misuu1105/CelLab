import unittest

import numpy as np

from cel_core.engine import CycleTracker, SimulationEngine
from cel_core.hashing import RAW_KEY_MAX_CELLS, state_key
from cel_core.init_world import place_preset, random_grid
from cel_core.rules import build_rule
from cel_core.spec import ForestFireSpec, LangtonSpec, LifeLikeSpec
from cel_core.types import Agent, BoundaryCondition, Direction, WorldState

LIFE = LifeLikeSpec()

# --------------------------------------------------------------------------
# Independent 8x8 toroidal Life implementation on a 64-bit bitboard
# (bit index y * 8 + x), used to brute-force long runs quickly.
# --------------------------------------------------------------------------
_FULL = (1 << 64) - 1
_COL0 = sum(1 << (8 * y) for y in range(8))
_COL7 = sum(1 << (8 * y + 7) for y in range(8))


def _grid_to_bits(grid: np.ndarray) -> int:
    return int.from_bytes(np.packbits(grid.reshape(-1), bitorder="little").tobytes(), "little")


def _bits_to_grid(b: int) -> np.ndarray:
    raw = np.frombuffer(b.to_bytes(8, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little").reshape(8, 8)


def _from_east(b: int) -> int:
    return ((b >> 1) & ~_COL7) | ((b << 7) & _COL7)


def _from_west(b: int) -> int:
    return ((b << 1) & ~_COL0 & _FULL) | ((b >> 7) & _COL0)


def _from_south(b: int) -> int:
    return ((b >> 8) | (b << 56)) & _FULL


def _from_north(b: int) -> int:
    return ((b << 8) | (b >> 56)) & _FULL


def _bitboard_life_step(b: int) -> int:
    n, s = _from_north(b), _from_south(b)
    neighbors = (n, s, _from_east(b), _from_west(b), _from_east(n), _from_west(n), _from_east(s), _from_west(s))
    s0 = s1 = s2 = s3 = 0
    for x in neighbors:
        c0 = s0 & x
        s0 ^= x
        c1 = s1 & c0
        s1 ^= c0
        c2 = s2 & c1
        s2 ^= c1
        s3 |= c2
    low = s1 & ~s2 & ~s3
    return ((s0 & low) | (b & ~s0 & low)) & _FULL


def brute_force(spec, state: WorldState, n: int) -> WorldState:
    rule = build_rule(spec)
    for _ in range(n):
        state = rule.step(state)
    return state


def make_engines(spec, state: WorldState, **kwargs) -> tuple[SimulationEngine, SimulationEngine]:
    return (
        SimulationEngine(spec, state, store_history=True, **kwargs),
        SimulationEngine(spec, state, store_history=False, **kwargs),
    )


class TestStateKey(unittest.TestCase):
    def test_step_counter_ignored_and_content_sensitive(self) -> None:
        a = place_preset("glider", (8, 8))
        b = a.copy()
        b.step = 99
        self.assertEqual(state_key(a, False), state_key(b, False))
        b.grid[0, 0] ^= 1
        self.assertNotEqual(state_key(a, False), state_key(b, False))

    def test_binary_grid_is_bit_packed(self) -> None:
        s = WorldState(grid=np.ones((16, 16), dtype=np.uint8))
        multi = WorldState(grid=np.full((16, 16), 2, dtype=np.uint8))
        self.assertLess(len(state_key(s, False)), 16 * 16)
        self.assertGreater(len(state_key(multi, False)), 16 * 16)

    def test_shape_is_part_of_key(self) -> None:
        a = WorldState(grid=np.zeros((2, 8), dtype=np.uint8))
        b = WorldState(grid=np.zeros((4, 4), dtype=np.uint8))
        self.assertNotEqual(state_key(a, False), state_key(b, False))

    def test_agents_and_halted_included(self) -> None:
        g = np.zeros((4, 4), dtype=np.uint8)
        base = WorldState(grid=g, agents=[Agent(1, 1, Direction.N)])
        moved = WorldState(grid=g, agents=[Agent(1, 2, Direction.N)])
        turned = WorldState(grid=g, agents=[Agent(1, 1, Direction.E)])
        halted = WorldState(grid=g, agents=[Agent(1, 1, Direction.N)], halted=True)
        keys = {state_key(s, False) for s in (base, moved, turned, halted)}
        self.assertEqual(len(keys), 4)

    def test_rng_state_only_when_stochastic(self) -> None:
        g = np.zeros((4, 4), dtype=np.uint8)
        a = WorldState(grid=g, rng_state=np.random.default_rng(1).bit_generator.state)
        b = WorldState(grid=g, rng_state=np.random.default_rng(2).bit_generator.state)
        self.assertEqual(state_key(a, False), state_key(b, False))
        self.assertNotEqual(state_key(a, True), state_key(b, True))
        a2 = WorldState(grid=g, rng_state=np.random.default_rng(1).bit_generator.state)
        self.assertEqual(state_key(a, True), state_key(a2, True))

    def test_large_grid_uses_16_byte_digest(self) -> None:
        small = WorldState(grid=np.zeros((256, 256), dtype=np.uint8))
        large = WorldState(grid=np.zeros((256, 257), dtype=np.uint8))
        self.assertEqual(256 * 256, RAW_KEY_MAX_CELLS)
        self.assertGreater(len(state_key(small, False)), 16)
        self.assertEqual(len(state_key(large, False)), 16)
        flipped = large.copy()
        flipped.grid[5, 5] = 1
        self.assertNotEqual(state_key(large, False), state_key(flipped, False))


class TestCycleTracker(unittest.TestCase):
    def test_detects_transient_and_period(self) -> None:
        t = CycleTracker()
        seq = [b"a", b"b", b"c", b"d", b"b"]
        results = [t.observe(k, i) for i, k in enumerate(seq)]
        self.assertEqual(results, [False, False, False, False, True])
        self.assertEqual((t.transient, t.period), (1, 3))
        self.assertFalse(t.active)

    def test_exhausts_at_limit(self) -> None:
        t = CycleTracker(max_tracked_states=3)
        for i, k in enumerate([b"a", b"b", b"c", b"d", b"a"]):
            t.observe(k, i)
        self.assertTrue(t.exhausted)
        self.assertFalse(t.has_cycle)


class TestKnownPeriods(unittest.TestCase):
    CASES = [
        ("block", (6, 6), 1),
        ("blinker", (5, 5), 2),
        ("pulsar", (17, 17), 3),
        ("pentadecathlon", (18, 24), 15),
        ("glider", (8, 8), 32),
    ]

    def test_periods_both_history_modes(self) -> None:
        for name, shape, period in self.CASES:
            with self.subTest(pattern=name):
                init = place_preset(name, shape)
                for eng in make_engines(LIFE, init):
                    eng.run(3 * period + 5)
                    self.assertEqual((eng.transient, eng.period), (0, period))
                    info = eng.cycle_info()
                    self.assertTrue(info.has_cycle)
                    self.assertEqual(info.regime, "fixed_point" if period == 1 else "periodic")

    def test_fast_forward_matches_brute_force(self) -> None:
        for name, shape, period in self.CASES:
            with self.subTest(pattern=name):
                init = place_preset(name, shape)
                n = 5 * period + 7
                expected = brute_force(LIFE, init, n)
                for eng in make_engines(LIFE, init):
                    out = eng.run(n)
                    self.assertEqual(out.step, n)
                    np.testing.assert_array_equal(out.grid, expected.grid)


class TestMillionSteps(unittest.TestCase):
    def test_bitboard_matches_rule(self) -> None:
        rule = build_rule(LIFE)
        state = WorldState(grid=random_grid((8, 8), {1: 0.4}, seed=3))
        bits = _grid_to_bits(state.grid)
        for _ in range(200):
            state = rule.step(state)
            bits = _bitboard_life_step(bits)
            np.testing.assert_array_equal(_bits_to_grid(bits), state.grid)

    def test_glider_8x8_one_million_steps(self) -> None:
        init = place_preset("glider", (8, 8), offset=(1, 2))
        targets = (1_000_000, 1_000_013)
        bits = _grid_to_bits(init.grid)
        expected: dict[int, np.ndarray] = {}
        for t in range(1, max(targets) + 1):
            bits = _bitboard_life_step(bits)
            if t in targets:
                expected[t] = _bits_to_grid(bits)

        for n in targets:
            hist, nohist = make_engines(LIFE, init)
            a, b = hist.run(n), nohist.run(n)
            self.assertEqual((a.step, b.step), (n, n))
            np.testing.assert_array_equal(a.grid, expected[n])
            np.testing.assert_array_equal(b.grid, expected[n])
            self.assertEqual((hist.period, nohist.period), (32, 32))


class TestHistoryModesAgree(unittest.TestCase):
    def test_random_soup_with_transient(self) -> None:
        init = WorldState(grid=random_grid((6, 6), {1: 0.45}, seed=11))
        hist, nohist = make_engines(LIFE, init)
        hist.run(2000)
        nohist.run(2000)
        self.assertTrue(hist.tracker.has_cycle)
        self.assertGreater(hist.transient, 0)
        self.assertEqual((hist.transient, hist.period), (nohist.transient, nohist.period))

        for n in (0, 1, hist.transient, hist.transient + hist.period + 3, 1777, 2000, 12_345):
            expected = brute_force(LIFE, init, n)
            np.testing.assert_array_equal(hist.state_at(n).grid, expected.grid)
            np.testing.assert_array_equal(nohist.state_at(n).grid, expected.grid)

    def test_incremental_runs_and_step(self) -> None:
        init = place_preset("glider", (8, 8))
        hist, nohist = make_engines(LIFE, init)
        for chunk in (10, 100, 7, 1000, 3):
            a, b = hist.run(chunk), nohist.run(chunk)
            np.testing.assert_array_equal(a.grid, b.grid)
            a, b = hist.step(), nohist.step()
            np.testing.assert_array_equal(a.grid, b.grid)
            self.assertEqual(a.step, b.step)
        expected = brute_force(LIFE, init, hist.current_step)
        np.testing.assert_array_equal(hist.state.grid, expected.grid)

    def test_langton_halt_is_fixed_point(self) -> None:
        spec = LangtonSpec(border_policy="halt")
        init = place_preset("single_ant", (5, 5))
        for eng in make_engines(spec, init):
            out = eng.run(10_000)
            self.assertTrue(out.halted)
            self.assertEqual(eng.period, 1)
            np.testing.assert_array_equal(out.grid, brute_force(spec, init, 10_000).grid)

    def test_langton_wrap_torus(self) -> None:
        spec = LangtonSpec(boundary=BoundaryCondition.TOROIDAL)
        init = place_preset("single_ant", (4, 4))
        hist, nohist = make_engines(spec, init)
        n = 50_003
        a, b = hist.run(n), nohist.run(n)
        self.assertTrue(hist.tracker.has_cycle)
        expected = brute_force(spec, init, n)
        for out in (a, b):
            np.testing.assert_array_equal(out.grid, expected.grid)
            self.assertEqual(out.agents, expected.agents)

    def test_stochastic_tracker_limit(self) -> None:
        spec = ForestFireSpec(p_growth=0.05, p_lightning=0.01, seed=7)
        init = WorldState(grid=random_grid((10, 10), {1: 0.5}, seed=1))
        hist, nohist = make_engines(spec, init, max_tracked_states=50)
        a, b = hist.run(200), nohist.run(200)
        np.testing.assert_array_equal(a.grid, b.grid)
        np.testing.assert_array_equal(a.grid, brute_force(spec, init, 200).grid)
        self.assertTrue(hist.tracker.exhausted)
        self.assertEqual(hist.cycle_info().regime, "stochastic_no_exact_cycle")
        self.assertLessEqual(len(hist.history), 50)
        self.assertEqual(nohist.history, [])


if __name__ == "__main__":
    unittest.main()

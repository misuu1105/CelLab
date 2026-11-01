import unittest

from pydantic import ValidationError

from cel_core.spec import (
    SPEC_ADAPTER,
    ForestFireSpec,
    GenerationsSpec,
    LangtonSpec,
    LifeLikeSpec,
    parse_spec,
)
from cel_core.types import BoundaryCondition, Direction, Neighborhood


class TestAutomatonSpecs(unittest.TestCase):
    def test_discriminated_union_dispatch(self) -> None:
        lifelike = parse_spec({"kind": "lifelike", "birth": [3], "survival": [2, 3]})
        self.assertIsInstance(lifelike, LifeLikeSpec)
        self.assertEqual(lifelike.birth, {3})
        self.assertEqual(lifelike.survival, {2, 3})

        generations = parse_spec({"kind": "generations", "birth": [2], "survival": [], "num_states": 3})
        self.assertIsInstance(generations, GenerationsSpec)
        self.assertEqual(generations.num_states, 3)

        langton = parse_spec({"kind": "langton", "turn_rules": "RL", "default_direction": "E"})
        self.assertIsInstance(langton, LangtonSpec)
        self.assertIsInstance(langton.default_direction, Direction)
        self.assertEqual(langton.default_direction, Direction.E)

        forest = parse_spec({"kind": "forest_fire", "p_growth": 0.05, "p_lightning": 0.001})
        self.assertIsInstance(forest, ForestFireSpec)
        self.assertTrue(forest.is_stochastic)

    def test_discriminated_union_json_roundtrip(self) -> None:
        specs = [
            LifeLikeSpec(birth={3, 6}, survival={2, 3}),
            GenerationsSpec(birth={2}, survival={3, 4}, num_states=4),
            LangtonSpec(turn_rules="RLR", default_direction=Direction.S),
            ForestFireSpec(p_growth=0.1, p_lightning=0.01, min_fire_neighbors=2, seed=42),
        ]
        for spec in specs:
            json_str = spec.model_dump_json()
            restored = SPEC_ADAPTER.validate_json(json_str)
            self.assertEqual(restored, spec)

    def test_unknown_kind_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            parse_spec({"kind": "sandpile"})

    def test_neighbor_count_ranges_moore_vs_von_neumann(self) -> None:
        # Valid Moore range: 0..8
        spec_moore = LifeLikeSpec(
            neighborhood=Neighborhood.MOORE,
            birth={0, 8},
            survival={4},
        )
        self.assertEqual(spec_moore.max_neighbors, 8)

        # Out of range for Moore: -1 and 9
        with self.assertRaises(ValidationError):
            LifeLikeSpec(neighborhood=Neighborhood.MOORE, birth={-1})
        with self.assertRaises(ValidationError):
            LifeLikeSpec(neighborhood=Neighborhood.MOORE, survival={9})

        # Valid Von Neumann range: 0..4
        spec_vn = LifeLikeSpec(
            neighborhood=Neighborhood.VON_NEUMANN,
            birth={0, 4},
            survival={1, 2},
        )
        self.assertEqual(spec_vn.max_neighbors, 4)

        # Out of range for Von Neumann: 5..8
        with self.assertRaises(ValidationError):
            LifeLikeSpec(neighborhood=Neighborhood.VON_NEUMANN, birth={5})
        with self.assertRaises(ValidationError):
            GenerationsSpec(neighborhood=Neighborhood.VON_NEUMANN, survival={6})

        # ForestFire min_fire_neighbors range: 1..max_neighbors
        with self.assertRaises(ValidationError):
            ForestFireSpec(min_fire_neighbors=0)
        with self.assertRaises(ValidationError):
            ForestFireSpec(neighborhood=Neighborhood.MOORE, min_fire_neighbors=9)
        with self.assertRaises(ValidationError):
            ForestFireSpec(neighborhood=Neighborhood.VON_NEUMANN, min_fire_neighbors=5)

    def test_probabilities_validation(self) -> None:
        det = ForestFireSpec(p_growth=0.0, p_lightning=0.0)
        self.assertFalse(det.is_stochastic)

        stoch = ForestFireSpec(p=1.0, q=0.25)
        self.assertTrue(stoch.is_stochastic)
        self.assertEqual(stoch.p_growth, 1.0)
        self.assertEqual(stoch.p_lightning, 0.25)

        for bad_prob in (-0.01, 1.01, float("nan"), float("inf"), -float("inf"), True):
            with self.assertRaises(ValidationError):
                ForestFireSpec(p_growth=bad_prob)  # type: ignore[arg-type]
            with self.assertRaises(ValidationError):
                ForestFireSpec(p_lightning=bad_prob)  # type: ignore[arg-type]

    def test_state_names_length_validation(self) -> None:
        # LifeLikeSpec requires 2 state names
        with self.assertRaises(ValidationError):
            LifeLikeSpec(state_names=["ONLY_ONE"])
        with self.assertRaises(ValidationError):
            LifeLikeSpec(state_names=["A", "B", "C"])
        with self.assertRaises(ValidationError):
            LifeLikeSpec(state_names=["DEAD", "   "])

        # GenerationsSpec explicit mismatch rejected, omitted state_names auto-populated
        gen4 = GenerationsSpec(num_states=4)
        self.assertEqual(len(gen4.state_names), 4)
        with self.assertRaises(ValidationError):
            GenerationsSpec(num_states=4, state_names=["OFF", "ON", "DYING"])

        # LangtonSpec explicit mismatch rejected, multi-color turn_rules auto-adjusts
        ant3 = LangtonSpec(turn_rules="RLR")
        self.assertEqual(ant3.num_states, 3)
        self.assertEqual(len(ant3.state_names), 3)
        with self.assertRaises(ValidationError):
            LangtonSpec(turn_rules="RL", state_names=["WHITE", "BLACK", "RED"])
        with self.assertRaises(ValidationError):
            LangtonSpec(turn_rules="RLR", num_states=2)

        # ForestFireSpec requires 3 state names
        with self.assertRaises(ValidationError):
            ForestFireSpec(state_names=["EMPTY", "TREE"])

    def test_langton_direction_enum_validation(self) -> None:
        for raw, expected in [
            (Direction.N, Direction.N),
            (Direction.W, Direction.W),
            (0, Direction.N),
            (1, Direction.E),
            (2, Direction.S),
            (3, Direction.W),
            ("n", Direction.N),
            ("S", Direction.S),
        ]:
            spec = LangtonSpec(default_direction=raw)  # type: ignore[arg-type]
            self.assertIsInstance(spec.default_direction, Direction)
            self.assertEqual(spec.default_direction, expected)

        for invalid_dir in (-1, 4, "NE", True):
            with self.assertRaises(ValidationError):
                LangtonSpec(default_direction=invalid_dir)  # type: ignore[arg-type]

    def test_fixed_boundary_state_validation(self) -> None:
        spec = LifeLikeSpec(boundary=BoundaryCondition.FIXED, fixed_boundary_state=1)
        self.assertEqual(spec.fixed_boundary_state, 1)
        with self.assertRaises(ValidationError):
            LifeLikeSpec(fixed_boundary_state=2)
        with self.assertRaises(ValidationError):
            LifeLikeSpec(fixed_boundary_state=-1)


if __name__ == "__main__":
    unittest.main()


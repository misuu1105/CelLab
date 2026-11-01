import unittest
import numpy as np

from cel_core.types import (
    Agent,
    BoundaryCondition,
    Neighborhood,
    QueryResultQ1,
    QueryResultQ2,
    QueryResultQ3,
    QueryResultQ4,
    QueryResultQ5,
    WorldState,
)


class TestSmokeTypes(unittest.TestCase):
    def test_enums(self) -> None:
        self.assertEqual(BoundaryCondition.TOROIDAL, "toroidal")
        self.assertEqual(BoundaryCondition.FIXED, "fixed")
        self.assertEqual(BoundaryCondition.REFLECTIVE, "reflective")
        self.assertEqual(Neighborhood.MOORE, "moore")
        self.assertEqual(Neighborhood.VON_NEUMANN, "von_neumann")

    def test_agent_roundtrip_and_direction_wrap(self) -> None:
        agent = Agent(y=3, x=7, direction=5)
        self.assertEqual(agent.direction, 1)
        restored = Agent.from_dict(agent.to_dict())
        self.assertEqual(restored, agent)

    def test_world_state_rejects_non_2d(self) -> None:
        with self.assertRaises(ValueError):
            WorldState(grid=np.zeros((8,), dtype=np.uint8))
        with self.assertRaises(ValueError):
            WorldState(grid=np.zeros((2, 2, 2), dtype=np.uint8))

    def test_world_state_roundtrip_and_copy(self) -> None:
        rng = np.random.default_rng(123)
        grid = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
        state = WorldState(
            grid=grid,
            agents=[Agent(y=1, x=2, direction=3)],
            step=12,
            rng_state=rng.bit_generator.state,
            halted=False,
        )
        self.assertEqual(state.shape, (2, 3))

        dumped = state.to_dict()
        restored = WorldState.from_dict(dumped)

        np.testing.assert_array_equal(restored.grid, state.grid)
        self.assertEqual(restored.agents, state.agents)
        self.assertEqual(restored.step, 12)
        self.assertEqual(restored.rng_state, state.rng_state)
        self.assertFalse(restored.halted)

        cloned = state.copy()
        cloned.grid[0, 0] = 99
        cloned.agents[0].x = 0
        self.assertEqual(state.grid[0, 0], 0)
        self.assertEqual(state.agents[0].x, 2)

    def test_query_result_dataclasses(self) -> None:
        ws = WorldState(grid=np.zeros((4, 4), dtype=np.uint8), step=5)
        q1 = QueryResultQ1(step=5, state=ws)
        q2 = QueryResultQ2(reached=True, step=3)
        q3 = QueryResultQ3(found=True, first_step=2, positions=[(0, 1), (2, 3)])
        q4 = QueryResultQ4(step=5, state_value=1, count=7)
        q5 = QueryResultQ5(has_cycle=True, transient_t=4, period_l=2, regime="periodic")

        self.assertEqual(q1.state.shape, (4, 4))
        self.assertTrue(q2.reached)
        self.assertEqual(q3.positions, [(0, 1), (2, 3)])
        self.assertEqual(q4.count, 7)
        self.assertEqual(q5.regime, "periodic")


if __name__ == "__main__":
    unittest.main()


import unittest
import numpy as np

from cel_core.grid import MOORE_KERNEL, VON_NEUMANN_KERNEL, get_kernel, neighbor_count, pad
from cel_core.types import BoundaryCondition, Neighborhood


class TestGridPadAndNeighborCount(unittest.TestCase):
    def setUp(self) -> None:
        # 3x3 grid with distinct values at corners, edges, and center
        self.seq_grid = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            dtype=np.uint8,
        )
        # All-ones 3x3 grid for exhaustive edge and corner degree testing
        self.ones_3x3 = np.ones((3, 3), dtype=np.uint8)

        # Asymmetric binary 3x3 grid to verify directional wrapping/reflection at corners & edges:
        # [[1, 0, 1],
        #  [0, 1, 0],
        #  [0, 1, 1]]
        self.asym_grid = np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
            ],
            dtype=np.uint8,
        )

    def test_pad_toroidal_wrap(self) -> None:
        padded = pad(self.seq_grid, BoundaryCondition.TOROIDAL, width=1)
        expected = np.array(
            [
                [9, 7, 8, 9, 7],
                [3, 1, 2, 3, 1],
                [6, 4, 5, 6, 4],
                [9, 7, 8, 9, 7],
                [3, 1, 2, 3, 1],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(padded, expected)

    def test_pad_fixed_constant(self) -> None:
        padded0 = pad(self.seq_grid, BoundaryCondition.FIXED, width=1, fixed_value=0)
        expected0 = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 7, 8, 9, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(padded0, expected0)

        padded9 = pad(self.seq_grid, BoundaryCondition.FIXED, width=1, fixed_value=9)
        self.assertTrue(np.all(padded9[0, :] == 9))
        self.assertTrue(np.all(padded9[-1, :] == 9))
        self.assertTrue(np.all(padded9[:, 0] == 9))
        self.assertTrue(np.all(padded9[:, -1] == 9))

    def test_pad_reflective_symmetric(self) -> None:
        padded = pad(self.seq_grid, BoundaryCondition.REFLECTIVE, width=1)
        expected = np.array(
            [
                [1, 1, 2, 3, 3],
                [1, 1, 2, 3, 3],
                [4, 4, 5, 6, 6],
                [7, 7, 8, 9, 9],
                [7, 7, 8, 9, 9],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(padded, expected)

    def test_pad_width_zero_and_invalid_inputs(self) -> None:
        unpadded = pad(self.seq_grid, BoundaryCondition.TOROIDAL, width=0)
        np.testing.assert_array_equal(unpadded, self.seq_grid)
        self.assertIsNot(unpadded, self.seq_grid)

        with self.assertRaises(ValueError):
            pad(np.ones((4,), dtype=np.uint8), BoundaryCondition.TOROIDAL)
        with self.assertRaises(ValueError):
            pad(self.seq_grid, BoundaryCondition.TOROIDAL, width=-1)

    def test_all_ones_edge_and_corner_counts_every_boundary(self) -> None:
        # 1. FIXED (constant 0):
        # Moore: corners have 3, edges have 5, center has 8
        fixed_moore = neighbor_count(
            self.ones_3x3,
            state=1,
            neighborhood=Neighborhood.MOORE,
            boundary=BoundaryCondition.FIXED,
        )
        expected_fixed_moore = np.array(
            [
                [3, 5, 3],
                [5, 8, 5],
                [3, 5, 3],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(fixed_moore, expected_fixed_moore)

        # Von Neumann under FIXED: corners have 2, edges have 3, center has 4
        fixed_vn = neighbor_count(
            self.ones_3x3,
            state=1,
            neighborhood=Neighborhood.VON_NEUMANN,
            boundary=BoundaryCondition.FIXED,
        )
        expected_fixed_vn = np.array(
            [
                [2, 3, 2],
                [3, 4, 3],
                [2, 3, 2],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(fixed_vn, expected_fixed_vn)

        # 2. TOROIDAL (wrap): every cell (corners, edges, interior) sees 8 (Moore) or 4 (VN)
        toroidal_moore = neighbor_count(
            self.ones_3x3,
            state=1,
            neighborhood=Neighborhood.MOORE,
            boundary=BoundaryCondition.TOROIDAL,
        )
        np.testing.assert_array_equal(toroidal_moore, np.full((3, 3), 8, dtype=np.int32))

        toroidal_vn = neighbor_count(
            self.ones_3x3,
            state=1,
            neighborhood=Neighborhood.VON_NEUMANN,
            boundary=BoundaryCondition.TOROIDAL,
        )
        np.testing.assert_array_equal(toroidal_vn, np.full((3, 3), 4, dtype=np.int32))

        # 3. REFLECTIVE (symmetric): every cell (corners, edges, interior) sees 8 (Moore) or 4 (VN)
        reflective_moore = neighbor_count(
            self.ones_3x3,
            state=1,
            neighborhood=Neighborhood.MOORE,
            boundary=BoundaryCondition.REFLECTIVE,
        )
        np.testing.assert_array_equal(reflective_moore, np.full((3, 3), 8, dtype=np.int32))

        reflective_vn = neighbor_count(
            self.ones_3x3,
            state=1,
            neighborhood=Neighborhood.VON_NEUMANN,
            boundary=BoundaryCondition.REFLECTIVE,
        )
        np.testing.assert_array_equal(reflective_vn, np.full((3, 3), 4, dtype=np.int32))

    def test_asymmetric_edge_and_corner_counts_toroidal(self) -> None:
        # In a 3x3 toroidal grid with total of 5 ones:
        # Every cell's 3x3 window covers the entire 3x3 grid, so Moore count is (5 - cell_value).
        counts_moore = neighbor_count(
            self.asym_grid,
            state=1,
            neighborhood=Neighborhood.MOORE,
            boundary=BoundaryCondition.TOROIDAL,
        )
        expected_moore = 5 - self.asym_grid.astype(np.int32)
        np.testing.assert_array_equal(counts_moore, expected_moore)

        # Von Neumann on asym_grid under TOROIDAL:
        # [[1, 0, 1],
        #  [0, 1, 0],
        #  [0, 1, 1]]
        counts_vn = neighbor_count(
            self.asym_grid,
            state=1,
            neighborhood=Neighborhood.VON_NEUMANN,
            boundary=BoundaryCondition.TOROIDAL,
        )
        expected_vn = np.array(
            [
                [1, 4, 2],
                [2, 1, 3],
                [3, 2, 2],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(counts_vn, expected_vn)

    def test_asymmetric_edge_and_corner_counts_fixed(self) -> None:
        # asym_grid:
        # [[1, 0, 1],
        #  [0, 1, 0],
        #  [0, 1, 1]]
        counts_moore = neighbor_count(
            self.asym_grid,
            state=1,
            neighborhood=Neighborhood.MOORE,
            boundary=BoundaryCondition.FIXED,
        )
        expected_moore = np.array(
            [
                [1, 3, 1],  # corners (0,0)->1, (0,2)->1; top edge (0,1)->3
                [3, 4, 4],  # left edge (1,0)->3; center (1,1)->4; right edge (1,2)->4
                [2, 2, 2],  # bottom-left (2,0)->2; bottom edge (2,1)->2; bottom-right (2,2)->2
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(counts_moore, expected_moore)

        counts_vn = neighbor_count(
            self.asym_grid,
            state=1,
            neighborhood=Neighborhood.VON_NEUMANN,
            boundary=BoundaryCondition.FIXED,
        )
        expected_vn = np.array(
            [
                [0, 3, 0],
                [2, 1, 3],
                [1, 2, 1],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(counts_vn, expected_vn)

    def test_asymmetric_edge_and_corner_counts_reflective(self) -> None:
        # Under REFLECTIVE (symmetric), padded asym_grid is:
        # [[1, 1, 0, 1, 1],
        #  [1, 1, 0, 1, 1],
        #  [0, 0, 1, 0, 0],
        #  [0, 0, 1, 1, 1],
        #  [0, 0, 1, 1, 1]]
        counts_moore = neighbor_count(
            self.asym_grid,
            state=1,
            neighborhood=Neighborhood.MOORE,
            boundary=BoundaryCondition.REFLECTIVE,
        )
        expected_moore = np.array(
            [
                [4, 5, 4],
                [4, 4, 6],
                [3, 4, 6],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(counts_moore, expected_moore)

        counts_vn = neighbor_count(
            self.asym_grid,
            state=1,
            neighborhood=Neighborhood.VON_NEUMANN,
            boundary=BoundaryCondition.REFLECTIVE,
        )
        expected_vn = np.array(
            [
                [2, 3, 2],  # (0,0): N=1,W=1,E=0,S=0 -> 2; (0,1): N=0,W=1,E=1,S=1 -> 3; (0,2): N=1,E=1,W=0,S=0 -> 2
                [2, 1, 3],  # (1,0): N=1,S=0,W=0,E=1 -> 2; (1,1): N=0,S=1,W=0,E=0 -> 1; (1,2): N=1,S=1,W=1,E=0 -> 3
                [1, 3, 3],  # (2,0): N=0,S=0,W=0,E=1 -> 1; (2,1): N=1,S=1,W=0,E=1 -> 3; (2,2): N=0,S=1,W=1,E=1 -> 3
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(counts_vn, expected_vn)

    def test_fixed_value_affects_matching_state_count(self) -> None:
        zeros = np.zeros((3, 3), dtype=np.uint8)
        # If fixed border is state 2, and we count state 2 on a FIXED boundary:
        counts = neighbor_count(
            zeros,
            state=2,
            neighborhood=Neighborhood.MOORE,
            boundary=BoundaryCondition.FIXED,
            fixed_value=2,
        )
        expected = np.array(
            [
                [5, 3, 5],
                [3, 0, 3],
                [5, 3, 5],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(counts, expected)

    def test_get_kernel(self) -> None:
        np.testing.assert_array_equal(get_kernel(Neighborhood.MOORE), MOORE_KERNEL)
        np.testing.assert_array_equal(get_kernel("von_neumann"), VON_NEUMANN_KERNEL)
        with self.assertRaises(ValueError):
            get_kernel("hexagonal")


if __name__ == "__main__":
    unittest.main()


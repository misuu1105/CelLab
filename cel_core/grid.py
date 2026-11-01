"""2D boundary padding and convolution-based neighbor counting."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from cel_core.types import BoundaryCondition, Neighborhood

MOORE_KERNEL: np.ndarray = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=np.int32,
)

VON_NEUMANN_KERNEL: np.ndarray = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ],
    dtype=np.int32,
)


def get_kernel(neighborhood: Neighborhood | str) -> np.ndarray:
    """Return the 3x3 convolution kernel for the given 2D neighborhood stencil."""
    nb = Neighborhood(neighborhood)
    if nb == Neighborhood.MOORE:
        return MOORE_KERNEL
    if nb == Neighborhood.VON_NEUMANN:
        return VON_NEUMANN_KERNEL
    raise ValueError(f"Unsupported neighborhood: {neighborhood!r}")


def pad(
    grid: np.ndarray,
    boundary: BoundaryCondition | str = BoundaryCondition.TOROIDAL,
    width: int = 1,
    fixed_value: int = 0,
) -> np.ndarray:
    """Pad a 2D grid according to the specified boundary condition.

    Maps boundary conditions to numpy.pad modes:
    - TOROIDAL -> mode="wrap"
    - FIXED -> mode="constant", constant_values=fixed_value
    - REFLECTIVE -> mode="symmetric"
    """
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError(f"pad requires a 2D array, got shape {arr.shape}")
    if isinstance(width, bool) or not isinstance(width, int) or width < 0:
        raise ValueError(f"pad width must be a non-negative integer, got {width!r}")
    bc = BoundaryCondition(boundary)
    if width == 0:
        return arr.copy()

    if bc == BoundaryCondition.TOROIDAL:
        return np.pad(arr, pad_width=width, mode="wrap")
    if bc == BoundaryCondition.FIXED:
        return np.pad(arr, pad_width=width, mode="constant", constant_values=fixed_value)
    if bc == BoundaryCondition.REFLECTIVE:
        return np.pad(arr, pad_width=width, mode="symmetric")
    raise ValueError(f"Unsupported boundary condition: {boundary!r}")


def neighbor_count(
    grid: np.ndarray,
    state: int = 1,
    neighborhood: Neighborhood | str = Neighborhood.MOORE,
    boundary: BoundaryCondition | str = BoundaryCondition.TOROIDAL,
    fixed_value: int = 0,
) -> np.ndarray:
    """Count neighbors in `state` for every cell using 2D convolution (`mode="valid"`)."""
    kernel = get_kernel(neighborhood)
    padded = pad(grid, boundary=boundary, width=1, fixed_value=fixed_value)
    mask = (padded == state).astype(np.int32)
    counts = convolve2d(mask, kernel, mode="valid")
    return counts.astype(np.int32, copy=False)


"""Core 2D cellular automata engine and analytical query modules."""

from cel_core.grid import MOORE_KERNEL, VON_NEUMANN_KERNEL, get_kernel, neighbor_count, pad
from cel_core.rules import BaseRule, build_rule, make_rng_state
from cel_core.spec import (
    SPEC_ADAPTER,
    AutomatonSpec,
    BaseAutomatonSpec,
    ForestFireSpec,
    GenerationsSpec,
    LangtonSpec,
    LifeLikeSpec,
    RuleSpec,
    max_neighbors_for,
    parse_spec,
)
from cel_core.types import (
    Agent,
    BoundaryCondition,
    Direction,
    Neighborhood,
    QueryResultQ1,
    QueryResultQ2,
    QueryResultQ3,
    QueryResultQ4,
    QueryResultQ5,
    WorldState,
)

__all__ = [
    "MOORE_KERNEL",
    "SPEC_ADAPTER",
    "VON_NEUMANN_KERNEL",
    "Agent",
    "AutomatonSpec",
    "BaseAutomatonSpec",
    "BaseRule",
    "BoundaryCondition",
    "Direction",
    "ForestFireSpec",
    "GenerationsSpec",
    "LangtonSpec",
    "LifeLikeSpec",
    "Neighborhood",
    "QueryResultQ1",
    "QueryResultQ2",
    "QueryResultQ3",
    "QueryResultQ4",
    "QueryResultQ5",
    "RuleSpec",
    "WorldState",
    "build_rule",
    "get_kernel",
    "make_rng_state",
    "max_neighbors_for",
    "neighbor_count",
    "pad",
    "parse_spec",
]

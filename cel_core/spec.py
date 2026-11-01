"""Pydantic v2 rule specifications for 2D cellular automata."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from cel_core.types import BoundaryCondition, Direction, Neighborhood

VALID_TURN_CHARS = frozenset({"R", "L", "N", "U"})


def max_neighbors_for(neighborhood: Neighborhood) -> int:
    """Return maximum neighbor count for a 2D neighborhood stencil."""
    if neighborhood == Neighborhood.MOORE:
        return 8
    if neighborhood == Neighborhood.VON_NEUMANN:
        return 4
    raise ValueError(f"Unsupported neighborhood: {neighborhood}")


def _validate_neighbor_counts(
    counts: set[int],
    neighborhood: Neighborhood,
    field_name: str,
) -> set[int]:
    """Validate that all neighbor counts lie within [0, max_neighbors]."""
    max_n = max_neighbors_for(neighborhood)
    normalized: set[int] = set()
    for c in counts:
        if isinstance(c, bool) or not isinstance(c, int):
            raise ValueError(f"{field_name} elements must be integers, got {c!r}")
        if c < 0 or c > max_n:
            raise ValueError(
                f"{field_name} count {c} out of valid range [0, {max_n}] "
                f"for neighborhood '{neighborhood.value}'"
            )
        normalized.add(int(c))
    return normalized


class BaseAutomatonSpec(BaseModel):
    """Shared configuration and validation across all 2D automaton families."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid", populate_by_name=True)

    name: str = "Automaton"
    boundary: BoundaryCondition = BoundaryCondition.TOROIDAL
    neighborhood: Neighborhood = Neighborhood.MOORE
    num_states: int = 2
    state_names: list[str] = Field(default_factory=lambda: ["DEAD", "ALIVE"])
    fixed_boundary_state: int = 0

    @property
    def max_neighbors(self) -> int:
        return max_neighbors_for(self.neighborhood)

    @property
    def is_stochastic(self) -> bool:
        return False

    @field_validator("num_states", mode="before")
    @classmethod
    def _reject_bool_num_states(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("num_states must be an integer")
        if v < 2:
            raise ValueError(f"num_states must be >= 2, got {v}")
        return int(v)

    @field_validator("fixed_boundary_state", mode="before")
    @classmethod
    def _reject_bool_fixed_state(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("fixed_boundary_state must be an integer")
        return int(v)

    @field_validator("state_names")
    @classmethod
    def _validate_state_names_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("state_names must not be empty")
        cleaned: list[str] = []
        for name in v:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each entry in state_names must be a non-empty string")
            cleaned.append(name.strip())
        return cleaned

    def _validate_common_invariants(self) -> None:
        if len(self.state_names) != self.num_states:
            raise ValueError(
                f"state_names length ({len(self.state_names)}) must equal "
                f"num_states ({self.num_states})"
            )
        if self.fixed_boundary_state < 0 or self.fixed_boundary_state >= self.num_states:
            raise ValueError(
                f"fixed_boundary_state ({self.fixed_boundary_state}) must be in "
                f"[0, {self.num_states - 1}]"
            )


class LifeLikeSpec(BaseAutomatonSpec):
    """2-state outer-totalistic Life-like cellular automaton specification (B/S)."""

    kind: Literal["lifelike"] = "lifelike"
    name: str = "Conway's Game of Life"
    birth: set[int] = Field(default_factory=lambda: {3})
    survival: set[int] = Field(default_factory=lambda: {2, 3})
    num_states: int = 2
    state_names: list[str] = Field(default_factory=lambda: ["DEAD", "ALIVE"])

    @field_validator("birth", "survival", mode="before")
    @classmethod
    def _coerce_count_set(cls, v: Any) -> set[int]:
        if isinstance(v, (set, frozenset, list, tuple)):
            result: set[int] = set()
            for item in v:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise ValueError(f"Neighbor counts must be integers, got {item!r}")
                result.add(int(item))
            return result
        raise ValueError(f"Expected an iterable of integer neighbor counts, got {type(v).__name__}")

    @model_validator(mode="after")
    def _validate_lifelike(self) -> LifeLikeSpec:
        if self.num_states != 2:
            raise ValueError(f"LifeLikeSpec requires num_states == 2, got {self.num_states}")
        self._validate_common_invariants()
        _validate_neighbor_counts(self.birth, self.neighborhood, "birth")
        _validate_neighbor_counts(self.survival, self.neighborhood, "survival")
        return self


class GenerationsSpec(BaseAutomatonSpec):
    """Multi-state Generations automaton specification (e.g., Brian's Brain)."""

    kind: Literal["generations"] = "generations"
    name: str = "Brian's Brain"
    birth: set[int] = Field(default_factory=lambda: {2})
    survival: set[int] = Field(default_factory=set)
    num_states: int = 3
    state_names: list[str] = Field(default_factory=lambda: ["OFF", "FIRING", "DYING"])

    @field_validator("birth", "survival", mode="before")
    @classmethod
    def _coerce_count_set(cls, v: Any) -> set[int]:
        if isinstance(v, (set, frozenset, list, tuple)):
            result: set[int] = set()
            for item in v:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise ValueError(f"Neighbor counts must be integers, got {item!r}")
                result.add(int(item))
            return result
        raise ValueError(f"Expected an iterable of integer neighbor counts, got {type(v).__name__}")

    @model_validator(mode="after")
    def _validate_generations(self) -> GenerationsSpec:
        if self.num_states < 3:
            raise ValueError(f"GenerationsSpec requires num_states >= 3, got {self.num_states}")
        if "state_names" not in self.model_fields_set and len(self.state_names) != self.num_states:
            object.__setattr__(
                self,
                "state_names",
                ["OFF", "FIRING", *[f"DYING_{i}" for i in range(1, self.num_states - 1)]],
            )
        self._validate_common_invariants()
        _validate_neighbor_counts(self.birth, self.neighborhood, "birth")
        _validate_neighbor_counts(self.survival, self.neighborhood, "survival")
        return self


class LangtonSpec(BaseAutomatonSpec):
    """Langton's Ant / multi-color turmite automaton specification."""

    kind: Literal["langton"] = "langton"
    name: str = "Langton's Ant"
    turn_rules: str = Field(
        default="RL",
        validation_alias=AliasChoices("turn_rules", "rules"),
    )
    default_direction: Direction = Field(
        default=Direction.N,
        validation_alias=AliasChoices("default_direction", "initial_direction", "direction"),
    )
    border_policy: Literal["wrap", "halt", "bounce"] = "wrap"
    num_states: int = 2
    state_names: list[str] = Field(default_factory=lambda: ["WHITE", "BLACK"])
    neighborhood: Neighborhood = Neighborhood.VON_NEUMANN

    @property
    def rules(self) -> str:
        return self.turn_rules

    @field_validator("turn_rules", mode="before")
    @classmethod
    def _validate_turn_rules(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("turn_rules must be a string")
        normalized = v.strip().upper()
        if len(normalized) < 2:
            raise ValueError("turn_rules must contain at least 2 characters")
        invalid = set(normalized) - VALID_TURN_CHARS
        if invalid:
            raise ValueError(
                f"turn_rules contains invalid characters {sorted(invalid)}; "
                f"allowed characters are {sorted(VALID_TURN_CHARS)}"
            )
        return normalized

    @field_validator("default_direction", mode="before")
    @classmethod
    def _validate_direction(cls, v: Any) -> Direction:
        if isinstance(v, Direction):
            return v
        if isinstance(v, str):
            key = v.strip().upper()
            if key in Direction.__members__:
                return Direction[key]
            raise ValueError(f"Unknown direction name {v!r}; expected one of N, E, S, W")
        if not isinstance(v, bool) and isinstance(v, int):
            if 0 <= v < 4:
                return Direction(v)
            raise ValueError(f"Direction integer must be in [0, 3], got {v}")
        raise ValueError(f"Invalid direction value: {v!r}")

    @model_validator(mode="after")
    def _validate_langton(self) -> LangtonSpec:
        if "num_states" not in self.model_fields_set and len(self.turn_rules) != self.num_states:
            object.__setattr__(self, "num_states", len(self.turn_rules))
        if len(self.turn_rules) != self.num_states:
            raise ValueError(
                f"turn_rules length ({len(self.turn_rules)}) must equal "
                f"num_states ({self.num_states})"
            )
        if "state_names" not in self.model_fields_set and len(self.state_names) != self.num_states:
            object.__setattr__(
                self,
                "state_names",
                [f"STATE_{i}" for i in range(self.num_states)],
            )
        self._validate_common_invariants()
        return self


class ForestFireSpec(BaseAutomatonSpec):
    """Drossel-Schwabl 2D Forest Fire automaton specification."""

    kind: Literal["forest_fire"] = "forest_fire"
    name: str = "Forest Fire"
    p_growth: float = Field(
        default=0.0,
        validation_alias=AliasChoices("p_growth", "p", "growth_prob"),
    )
    p_lightning: float = Field(
        default=0.0,
        validation_alias=AliasChoices("p_lightning", "q", "f", "lightning_prob"),
    )
    min_fire_neighbors: int = 1
    seed: int | None = None
    num_states: int = 3
    state_names: list[str] = Field(default_factory=lambda: ["EMPTY", "TREE", "FIRE"])

    @property
    def p(self) -> float:
        return self.p_growth

    @property
    def q(self) -> float:
        return self.p_lightning

    @property
    def is_stochastic(self) -> bool:
        return bool(self.p_growth > 0.0 or self.p_lightning > 0.0)

    @field_validator("p_growth", "p_lightning", mode="before")
    @classmethod
    def _validate_probability(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"Probability must be a float in [0.0, 1.0], got {v!r}")
        val = float(v)
        if not math.isfinite(val) or val < 0.0 or val > 1.0:
            raise ValueError(f"Probability must be between 0.0 and 1.0, got {val}")
        return val

    @field_validator("min_fire_neighbors", mode="before")
    @classmethod
    def _validate_min_fire_type(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"min_fire_neighbors must be an integer, got {v!r}")
        return int(v)

    @model_validator(mode="after")
    def _validate_forest_fire(self) -> ForestFireSpec:
        if self.num_states != 3:
            raise ValueError(f"ForestFireSpec requires num_states == 3, got {self.num_states}")
        self._validate_common_invariants()
        max_n = self.max_neighbors
        if self.min_fire_neighbors < 1 or self.min_fire_neighbors > max_n:
            raise ValueError(
                f"min_fire_neighbors ({self.min_fire_neighbors}) must be in [1, {max_n}] "
                f"for neighborhood '{self.neighborhood.value}'"
            )
        return self


AutomatonSpec = Annotated[
    Union[LifeLikeSpec, GenerationsSpec, LangtonSpec, ForestFireSpec],
    Field(discriminator="kind"),
]

RuleSpec = AutomatonSpec

SPEC_ADAPTER: TypeAdapter[AutomatonSpec] = TypeAdapter(AutomatonSpec)


def parse_spec(data: dict[str, Any] | str | BaseAutomatonSpec) -> BaseAutomatonSpec:
    """Validate and return a concrete automaton spec from a dict, JSON string, or model."""
    if isinstance(data, BaseAutomatonSpec):
        return data
    if isinstance(data, str):
        return SPEC_ADAPTER.validate_json(data)
    return SPEC_ADAPTER.validate_python(data)


"""Rule factory: build a BaseRule from a RuleSpec."""

from __future__ import annotations

from typing import Any

from cel_core.rules.base import BaseRule, generator_from_state, make_rng_state
from cel_core.rules.forest_fire import ForestFireRule
from cel_core.rules.langton import LangtonRule
from cel_core.rules.totalistic import GenerationsRule, LifeLikeRule
from cel_core.spec import (
    BaseAutomatonSpec,
    ForestFireSpec,
    GenerationsSpec,
    LangtonSpec,
    LifeLikeSpec,
    parse_spec,
)

RULE_REGISTRY: dict[type[BaseAutomatonSpec], type[BaseRule]] = {
    LifeLikeSpec: LifeLikeRule,
    GenerationsSpec: GenerationsRule,
    LangtonSpec: LangtonRule,
    ForestFireSpec: ForestFireRule,
}


def build_rule(spec: BaseAutomatonSpec | dict[str, Any] | str) -> BaseRule:
    """Return the rule implementation for a spec model, dict, or JSON string."""
    parsed = parse_spec(spec)
    rule_cls = RULE_REGISTRY.get(type(parsed))
    if rule_cls is None:
        raise ValueError(f"No rule registered for spec type {type(parsed).__name__}")
    return rule_cls(parsed)


__all__ = [
    "RULE_REGISTRY",
    "BaseRule",
    "ForestFireRule",
    "GenerationsRule",
    "LangtonRule",
    "LifeLikeRule",
    "build_rule",
    "generator_from_state",
    "make_rng_state",
]


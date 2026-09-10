"""Finite abstractions used by the exact PC falsification oracle."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Generic, TypeAlias, TypeVar

StateT = TypeVar("StateT", bound=Hashable)
ControlT = TypeVar("ControlT", bound=Hashable)
Belief: TypeAlias = frozenset[StateT]


@dataclass(frozen=True)
class FinitePlant(Generic[StateT, ControlT]):
    """A nondeterministic finite plant with a total transition relation."""

    states: frozenset[StateT]
    safe_states: frozenset[StateT]
    recovery_states: frozenset[StateT]
    controls: tuple[ControlT, ...]
    transitions: Mapping[tuple[StateT, ControlT], frozenset[StateT]]

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("states must be nonempty")
        if not self.controls:
            raise ValueError("controls must be nonempty")
        if not self.recovery_states <= self.safe_states <= self.states:
            raise ValueError("require recovery_states <= safe_states <= states")
        for state in self.states:
            for control in self.controls:
                successors = self.transitions.get((state, control))
                if not successors:
                    raise ValueError(f"missing transition for {(state, control)!r}")
                if not successors <= self.states:
                    raise ValueError("transition contains a state outside the plant")

    def step(self, belief: Belief[StateT], control: ControlT) -> Belief[StateT]:
        """Return the robust one-step image of a belief under one control."""

        if not belief or not belief <= self.states:
            raise ValueError("belief must be a nonempty subset of plant states")
        if control not in self.controls:
            raise ValueError(f"unknown control: {control!r}")
        return frozenset(
            successor
            for state in belief
            for successor in self.transitions[(state, control)]
        )

    def is_safe(self, belief: Belief[StateT]) -> bool:
        return bool(belief) and belief <= self.safe_states

    def safe_beliefs(self) -> tuple[Belief[StateT], ...]:
        """Enumerate all nonempty safe beliefs; intended only for tiny PC instances."""

        ordered = tuple(sorted(self.safe_states, key=repr))
        return tuple(
            frozenset(items)
            for size in range(1, len(ordered) + 1)
            for items in combinations(ordered, size)
        )

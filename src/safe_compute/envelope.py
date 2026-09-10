"""Exact safe-compute envelopes on small finite abstractions."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from functools import cache
from typing import Generic, TypeVar

from .finite import Belief, FinitePlant

StateT = TypeVar("StateT", bound=Hashable)
ControlT = TypeVar("ControlT", bound=Hashable)


@dataclass(frozen=True)
class EnvelopeResult(Generic[StateT, ControlT]):
    """A capped envelope and witness actions for every belief/layer."""

    max_pending_ticks: int
    recovery_beliefs: frozenset[Belief[StateT]]
    layers: tuple[frozenset[Belief[StateT]], ...]
    pending_policy: dict[tuple[Belief[StateT], int], ControlT]

    def budget(self, belief: Belief[StateT]) -> int | None:
        """Largest certified wait, or None when immediate recovery is not certified."""

        answer: int | None = None
        for ticks, layer in enumerate(self.layers):
            if belief in layer:
                answer = ticks
        return answer

    def reaches_cap(self, belief: Belief[StateT]) -> bool:
        return belief in self.layers[-1]


@dataclass(frozen=True)
class PointwiseEnvelopeResult(Generic[StateT, ControlT]):
    """On-demand budgets and witness controls for requested beliefs."""

    max_pending_ticks: int
    budgets: dict[Belief[StateT], int | None]
    pending_policy: dict[tuple[Belief[StateT], int], ControlT]
    recovery_policy: dict[tuple[Belief[StateT], int], ControlT]

    def budget(self, belief: Belief[StateT]) -> int | None:
        if belief not in self.budgets:
            raise KeyError("belief was not requested from the pointwise solver")
        return self.budgets[belief]

    def reaches_cap(self, belief: Belief[StateT]) -> bool:
        return self.budget(belief) == self.max_pending_ticks


class SafeComputeEnvelope(Generic[StateT, ControlT]):
    """Backward finite recursion for abortable, perception-independent compute time."""

    def __init__(
        self,
        plant: FinitePlant[StateT, ControlT],
        *,
        pending_controls: tuple[ControlT, ...] | None = None,
        recovery_controls: tuple[ControlT, ...] | None = None,
        recovery_horizon: int,
        max_pending_ticks: int,
    ) -> None:
        if recovery_horizon < 0 or max_pending_ticks < 0:
            raise ValueError("horizons must be nonnegative")
        self.plant = plant
        self.pending_controls = plant.controls if pending_controls is None else pending_controls
        self.recovery_controls = plant.controls if recovery_controls is None else recovery_controls
        self.recovery_horizon = recovery_horizon
        self.max_pending_ticks = max_pending_ticks
        if not self.pending_controls or not self.recovery_controls:
            raise ValueError("pending and recovery controls must be nonempty")
        for control in (*self.pending_controls, *self.recovery_controls):
            if control not in plant.controls:
                raise ValueError(f"unknown control: {control!r}")

    def _recovery_basin(self) -> frozenset[Belief[StateT]]:
        beliefs = self.plant.safe_beliefs()
        basin = {belief for belief in beliefs if belief <= self.plant.recovery_states}
        for _ in range(self.recovery_horizon):
            previous = set(basin)
            for belief in beliefs:
                if belief in basin:
                    continue
                if any(
                    self.plant.is_safe(image := self.plant.step(belief, control))
                    and image in previous
                    for control in self.recovery_controls
                ):
                    basin.add(belief)
            if basin == previous:
                break
        return frozenset(basin)

    def solve(self) -> EnvelopeResult[StateT, ControlT]:
        """Compute beliefs that can wait exactly-at-least 0..N ticks and still abort."""

        recovery_beliefs = self._recovery_basin()
        layers: list[frozenset[Belief[StateT]]] = [recovery_beliefs]
        policy: dict[tuple[Belief[StateT], int], ControlT] = {}

        for ticks in range(1, self.max_pending_ticks + 1):
            previous = layers[-1]
            current: set[Belief[StateT]] = set()
            for belief in recovery_beliefs:
                for control in self.pending_controls:
                    image = self.plant.step(belief, control)
                    if self.plant.is_safe(image) and image in previous:
                        current.add(belief)
                        policy[(belief, ticks)] = control
                        break
            layers.append(frozenset(current))

        return EnvelopeResult(
            max_pending_ticks=self.max_pending_ticks,
            recovery_beliefs=recovery_beliefs,
            layers=tuple(layers),
            pending_policy=policy,
        )

    def solve_for(
        self, beliefs: Iterable[Belief[StateT]]
    ) -> PointwiseEnvelopeResult[StateT, ControlT]:
        """Solve only beliefs reachable from requested queries, avoiding power-set enumeration."""

        requested = tuple(dict.fromkeys(beliefs))
        if not requested:
            raise ValueError("at least one belief is required")
        for belief in requested:
            if not belief or not belief <= self.plant.states:
                raise ValueError("beliefs must be nonempty subsets of plant states")

        pending_policy: dict[tuple[Belief[StateT], int], ControlT] = {}
        recovery_policy: dict[tuple[Belief[StateT], int], ControlT] = {}

        @cache
        def can_recover(current: Belief[StateT], remaining: int) -> bool:
            if not self.plant.is_safe(current):
                return False
            if current <= self.plant.recovery_states:
                return True
            if remaining == 0:
                return False
            for control in self.recovery_controls:
                image = self.plant.step(current, control)
                if self.plant.is_safe(image) and can_recover(image, remaining - 1):
                    recovery_policy[(current, remaining)] = control
                    return True
            return False

        @cache
        def can_wait(current: Belief[StateT], remaining: int) -> bool:
            if not can_recover(current, self.recovery_horizon):
                return False
            if remaining == 0:
                return True
            for control in self.pending_controls:
                image = self.plant.step(current, control)
                if self.plant.is_safe(image) and can_wait(image, remaining - 1):
                    pending_policy[(current, remaining)] = control
                    return True
            return False

        budgets: dict[Belief[StateT], int | None] = {}
        for belief in requested:
            if not can_wait(belief, 0):
                budgets[belief] = None
                continue
            budget = 0
            for ticks in range(1, self.max_pending_ticks + 1):
                if not can_wait(belief, ticks):
                    break
                budget = ticks
            budgets[belief] = budget

        return PointwiseEnvelopeResult(
            max_pending_ticks=self.max_pending_ticks,
            budgets=budgets,
            pending_policy=pending_policy,
            recovery_policy=recovery_policy,
        )

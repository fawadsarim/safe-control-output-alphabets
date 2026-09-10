"""Witness-generating covers without enumerating captured-state subsets.

For fixed completion and recovery horizons this construction, followed by unit-cost
greedy Set Cover, is polynomial in an explicitly tabulated finite plant. It is not
polynomial uniformly in the two horizons and does not compute the exact optimum.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from itertools import product
from typing import Generic, TypeVar

from .finite import Belief, FinitePlant

StateT = TypeVar("StateT", bound=Hashable)
ControlT = TypeVar("ControlT", bound=Hashable)


@dataclass(frozen=True)
class RecoveryCoverCell(Generic[StateT, ControlT]):
    captured_states: Belief[StateT]
    recovery_sequences: tuple[tuple[ControlT, ...], ...]


@dataclass(frozen=True)
class RecoveryCoverResult(Generic[StateT, ControlT]):
    pending_sequence: tuple[ControlT, ...]
    cells: tuple[RecoveryCoverCell[StateT, ControlT], ...]

    @property
    def symbols(self) -> int:
        return len(self.cells)


def recovery_word_success_set(
    plant: FinitePlant[StateT, ControlT], word: tuple[ControlT, ...]
) -> Belief[StateT]:
    """States from which this entire word is safe and ends in the terminal set.

    Universal predecessors handle nondeterminism. Intersecting with the safe set
    at every step rejects unsafe intermediate states even if the final state is
    terminal. A shorter successful prefix is represented by a different word.
    """

    for control in word:
        if control not in plant.controls:
            raise ValueError(f"unknown control: {control!r}")
    success = plant.recovery_states
    for control in reversed(word):
        success = frozenset(
            state for state in plant.safe_states
            if plant.transitions[(state, control)] <= success
        )
    return success


def _success_sets(plant, recovery_controls, recovery_horizon):
    if recovery_horizon < 0:
        raise ValueError("recovery_horizon must be nonnegative")
    if not recovery_controls:
        raise ValueError("recovery_controls must be nonempty")
    for control in recovery_controls:
        if control not in plant.controls:
            raise ValueError(f"unknown control: {control!r}")
    # Equal domains may share a witness without losing any feasible cell.
    domains = {}
    for length in range(recovery_horizon + 1):
        for word in product(recovery_controls, repeat=length):
            domain = recovery_word_success_set(plant, word)
            if domain:
                domains.setdefault(domain, word)
    return tuple(domains.items())


def _cover_with_success_sets(plant, belief, pending_sequence, domains):
    if not belief or not belief <= plant.safe_states:
        raise ValueError("belief must be a nonempty subset of safe states")
    for control in pending_sequence:
        if control not in plant.controls:
            raise ValueError(f"unknown control: {control!r}")
    images = {state: frozenset({state}) for state in sorted(belief, key=repr)}
    intersections = {belief: ()}
    for age in range(len(pending_sequence) + 1):
        if any(not plant.is_safe(image) for image in images.values()):
            return ()
        age_sets = {}
        for domain, word in domains:
            captured = frozenset(state for state, image in images.items() if image <= domain)
            if captured:
                age_sets.setdefault(captured, word)
        next_intersections = {}
        for previous, witnesses in intersections.items():
            for captured, word in age_sets.items():
                common = previous & captured
                if common:
                    next_intersections.setdefault(common, (*witnesses, word))
        intersections = next_intersections
        if not intersections:
            return ()
        if age < len(pending_sequence):
            images = {
                state: plant.step(image, pending_sequence[age])
                for state, image in images.items()
            }
    return tuple(
        RecoveryCoverCell(captured, witnesses)
        for captured, witnesses in intersections.items()
    )


def sequence_recovery_cover(
    plant: FinitePlant[StateT, ControlT],
    belief: Belief[StateT],
    pending_sequence: tuple[ControlT, ...],
    *,
    recovery_controls: tuple[ControlT, ...],
    recovery_horizon: int,
) -> tuple[RecoveryCoverCell[StateT, ControlT], ...]:
    """Generate feasible cells containing every feasible captured subset.

    There are at most R**(d+1) candidates, R=sum(r**h, h=0..H). Every
    candidate carries one recovery word per completion age. The returned family
    may overlap and may fail to cover the belief; an unsafe pending sequence
    returns an empty family. Neither maximality nor subset enumeration is needed.
    """

    domains = _success_sets(plant, recovery_controls, recovery_horizon)
    return _cover_with_success_sets(plant, belief, pending_sequence, domains)


def greedy_symbols_via_recovery_cover(
    plant: FinitePlant[StateT, ControlT],
    belief: Belief[StateT],
    *,
    pending_controls: tuple[ControlT, ...],
    recovery_controls: tuple[ControlT, ...],
    recovery_horizon: int,
    delay: int,
) -> RecoveryCoverResult[StateT, ControlT] | None:
    """Return a feasible disjoint partition within H_n of the exact optimum.

    None certifies infeasibility in the stated finite policy class. The finite
    symbols value is an upper bound, not an exact minimum. Witnesses apply to the
    captured belief; states outside it may be put in any existing output cell.
    """

    if delay < 0:
        raise ValueError("delay must be nonnegative")
    if not belief or not belief <= plant.safe_states:
        raise ValueError("belief must be a nonempty subset of safe states")
    if not pending_controls:
        raise ValueError("pending_controls must be nonempty")
    for control in pending_controls:
        if control not in plant.controls:
            raise ValueError(f"unknown control: {control!r}")
    domains = _success_sets(plant, recovery_controls, recovery_horizon)
    best = None
    for sequence in product(pending_controls, repeat=delay):
        candidates = _cover_with_success_sets(plant, belief, sequence, domains)
        if not candidates:
            continue
        uncovered = belief
        cells = []
        while uncovered:
            chosen = max(candidates, key=lambda cell: len(cell.captured_states & uncovered))
            newly_covered = chosen.captured_states & uncovered
            if not newly_covered:
                break
            cells.append(RecoveryCoverCell(newly_covered, chosen.recovery_sequences))
            uncovered = uncovered - newly_covered
        if not uncovered and (best is None or len(cells) < best.symbols):
            best = RecoveryCoverResult(sequence, tuple(cells))
    return best

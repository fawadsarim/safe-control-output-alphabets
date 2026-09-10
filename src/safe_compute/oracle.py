"""Independent exhaustive-sequence oracle for tiny safe-compute instances."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import product
from typing import TypeVar

from .finite import Belief, FinitePlant
from .separation import OutputPartition

StateT = TypeVar("StateT", bound=Hashable)
ControlT = TypeVar("ControlT", bound=Hashable)


def brute_force_budget(
    plant: FinitePlant[StateT, ControlT],
    belief: Belief[StateT],
    *,
    pending_controls: tuple[ControlT, ...],
    recovery_controls: tuple[ControlT, ...],
    recovery_horizon: int,
    max_pending_ticks: int,
) -> int | None:
    """Enumerate complete control sequences without using the layer recursion or memoization."""

    if recovery_horizon < 0 or max_pending_ticks < 0:
        raise ValueError("horizons must be nonnegative")

    def can_recover_exhaustively(current: Belief[StateT]) -> bool:
        if not plant.is_safe(current):
            return False
        if current <= plant.recovery_states:
            return True
        for length in range(1, recovery_horizon + 1):
            for controls in product(recovery_controls, repeat=length):
                image = current
                for control in controls:
                    image = plant.step(image, control)
                    if not plant.is_safe(image):
                        break
                    if image <= plant.recovery_states:
                        return True
        return False

    def sequence_is_abortable(controls: tuple[ControlT, ...]) -> bool:
        image = belief
        if not can_recover_exhaustively(image):
            return False
        for control in controls:
            image = plant.step(image, control)
            if not plant.is_safe(image) or not can_recover_exhaustively(image):
                return False
        return True

    if not sequence_is_abortable(()):
        return None
    budget = 0
    for ticks in range(1, max_pending_ticks + 1):
        if not any(
            sequence_is_abortable(controls)
            for controls in product(pending_controls, repeat=ticks)
        ):
            break
        budget = ticks
    return budget


def brute_force_guaranteed_completion(
    plant: FinitePlant[StateT, ControlT],
    belief: Belief[StateT],
    *,
    output_partition: OutputPartition[StateT],
    pending_controls: tuple[ControlT, ...],
    recovery_controls: tuple[ControlT, ...],
    recovery_horizon: int,
    delay: int,
) -> bool:
    """Exhaust all open-loop pending and recovery sequences for one deadline.

    This implementation intentionally does not call the information--liveness
    recursion or its recovery-basin construction.
    """

    if recovery_horizon < 0 or delay < 0:
        raise ValueError("horizons must be nonnegative")
    if not belief or not belief <= plant.safe_states:
        raise ValueError("belief must be a nonempty subset of safe states")
    output_partition.validate(plant.states)

    def recoverable(current: Belief[StateT]) -> bool:
        if not plant.is_safe(current):
            return False
        if current <= plant.recovery_states:
            return True
        for length in range(1, recovery_horizon + 1):
            for sequence in product(recovery_controls, repeat=length):
                image = current
                for control in sequence:
                    image = plant.step(image, control)
                    if not plant.is_safe(image):
                        break
                    if image <= plant.recovery_states:
                        return True
        return False

    for pending_sequence in product(pending_controls, repeat=delay):
        posteriors = tuple(
            frozenset(posterior)
            for cell in output_partition.cells
            if (posterior := belief.intersection(cell))
        )
        if not all(recoverable(posterior) for posterior in posteriors):
            continue
        valid = True
        for control in pending_sequence:
            posteriors = tuple(plant.step(posterior, control) for posterior in posteriors)
            if not all(recoverable(posterior) for posterior in posteriors):
                valid = False
                break
        if valid:
            return True
    return False


def brute_force_minimum_symbols(
    plant: FinitePlant[StateT, ControlT],
    belief: Belief[StateT],
    *,
    pending_controls: tuple[ControlT, ...],
    recovery_controls: tuple[ControlT, ...],
    recovery_horizon: int,
    delay: int,
) -> int | None:
    """Independently enumerate canonical sensor maps and return their minimum size."""

    def canonical_labels(length: int, symbols: int):
        """Generate restricted-growth strings without using partition utilities."""

        labels = [0]

        def visit(index: int, maximum: int):
            if index == length:
                if maximum + 1 == symbols:
                    yield tuple(labels)
                return
            for label in range(min(maximum + 1, symbols - 1) + 1):
                labels.append(label)
                yield from visit(index + 1, max(maximum, label))
                labels.pop()

        yield from visit(1, 0)

    safe = tuple(sorted(plant.safe_states, key=repr))
    unsafe = plant.states.difference(plant.safe_states)
    for symbols in range(1, len(safe) + 1):
        for labels in canonical_labels(len(safe), symbols):
            cells = [set() for _ in range(symbols)]
            for state, label in zip(safe, labels, strict=True):
                cells[label].add(state)
            cells[0].update(unsafe)
            partition = OutputPartition(
                name=f"oracle_{symbols}",
                cells=tuple(frozenset(cell) for cell in cells),
            )
            if brute_force_guaranteed_completion(
                plant,
                belief,
                output_partition=partition,
                pending_controls=pending_controls,
                recovery_controls=recovery_controls,
                recovery_horizon=recovery_horizon,
                delay=delay,
            ):
                return symbols
    return None

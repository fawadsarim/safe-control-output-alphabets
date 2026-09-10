"""Exact information--completion co-design on small finite plants.

The routines in this module are deliberately finite and exponential.  They are
research oracles for the information pattern in which a sound output may arrive
at any decision epoch and is guaranteed to arrive after at most ``d`` pending
actions.  A fixed output partition is used for the entire job.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from itertools import combinations, product
from typing import Generic, TypeAlias, TypeVar

from .envelope import SafeComputeEnvelope
from .finite import Belief, FinitePlant
from .separation import OutputPartition

StateT = TypeVar("StateT", bound=Hashable)
ControlT = TypeVar("ControlT", bound=Hashable)
ConditionedBeliefs: TypeAlias = tuple[Belief[StateT], ...]


@dataclass(frozen=True)
class GuaranteedCompletionResult(Generic[StateT, ControlT]):
    """Winning layers and witnesses for one fixed output partition."""

    partition: OutputPartition[StateT]
    recovery_beliefs: frozenset[Belief[StateT]]
    completion_beliefs: frozenset[Belief[StateT]]
    layers: tuple[frozenset[Belief[StateT]], ...]
    pending_policy: dict[tuple[ConditionedBeliefs[StateT], int], ControlT]

    def supports_delay(self, belief: Belief[StateT], delay: int) -> bool:
        """Return whether completion by ``delay`` pending ticks is certified."""

        if delay < 0 or delay >= len(self.layers):
            raise ValueError("delay is outside the computed range")
        return belief in self.layers[delay]

    def witness(
        self,
        plant: FinitePlant[StateT, ControlT],
        belief: Belief[StateT],
        delay: int,
    ) -> tuple[ControlT, ...]:
        """Return the pending-input sequence stored for a certified query."""

        if not self.supports_delay(belief, delay):
            raise ValueError("belief is not certified at the requested delay")
        controls: list[ControlT] = []
        conditioned = tuple(
            frozenset(posterior)
            for cell in self.partition.cells
            if (posterior := belief.intersection(cell))
        )
        for remaining in range(delay, 0, -1):
            control = self.pending_policy[(conditioned, remaining)]
            controls.append(control)
            conditioned = tuple(plant.step(group, control) for group in conditioned)
        return tuple(controls)


@dataclass(frozen=True)
class MinimumSymbolResult(Generic[StateT, ControlT]):
    """An exact minimum-alphabet certificate for one completion deadline."""

    delay: int
    minimum_symbols: int | None
    partition: OutputPartition[StateT] | None
    pending_controls: tuple[ControlT, ...] | None


@dataclass(frozen=True)
class HypergraphCoDesignResult(Generic[StateT, ControlT]):
    """Exact minimum information and a pending-sequence witness."""

    delay: int
    minimum_symbols: int | None
    pending_controls: tuple[ControlT, ...] | None
    conflict_hyperedges: tuple[Belief[StateT], ...] | None


class GuaranteedCompletionEnvelope(Generic[StateT, ControlT]):
    """Solve safety when sound completion may occur now and is forced by a deadline.

        The partition labels the state at capture.  At every possible completion
        age, each captured cell is propagated through the common pending inputs;
        every resulting current-state posterior must admit bounded recovery.  If
        the result is still pending, the controller applies one common input.  At
        age ``max_delay`` completion is forced.
    """

    def __init__(
        self,
        plant: FinitePlant[StateT, ControlT],
        *,
        output_partition: OutputPartition[StateT],
        pending_controls: tuple[ControlT, ...] | None = None,
        recovery_controls: tuple[ControlT, ...] | None = None,
        recovery_horizon: int,
        max_delay: int,
    ) -> None:
        if recovery_horizon < 0 or max_delay < 0:
            raise ValueError("horizons must be nonnegative")
        output_partition.validate(plant.states)
        self.plant = plant
        self.partition = output_partition
        self.pending_controls = plant.controls if pending_controls is None else pending_controls
        self.recovery_controls = plant.controls if recovery_controls is None else recovery_controls
        self.recovery_horizon = recovery_horizon
        self.max_delay = max_delay
        if not self.pending_controls or not self.recovery_controls:
            raise ValueError("control sets must be nonempty")
        for control in (*self.pending_controls, *self.recovery_controls):
            if control not in plant.controls:
                raise ValueError(f"unknown control: {control!r}")

    def solve(self) -> GuaranteedCompletionResult[StateT, ControlT]:
        """Compute the exact fixed-partition winning layers."""

        recovery_beliefs = SafeComputeEnvelope(
            self.plant,
            recovery_controls=self.recovery_controls,
            recovery_horizon=self.recovery_horizon,
            max_pending_ticks=0,
        ).solve().recovery_beliefs
        safe_beliefs = self.plant.safe_beliefs()

        def condition(belief: Belief[StateT]) -> ConditionedBeliefs[StateT]:
            return tuple(
                frozenset(posterior)
                for cell in self.partition.cells
                if (posterior := belief.intersection(cell))
            )

        def completion_safe(groups: ConditionedBeliefs[StateT]) -> bool:
            return all(group in recovery_beliefs for group in groups)

        policy: dict[tuple[ConditionedBeliefs[StateT], int], ControlT] = {}

        @cache
        def can_wait(groups: ConditionedBeliefs[StateT], remaining: int) -> bool:
            if not completion_safe(groups):
                return False
            if remaining == 0:
                return True
            for control in self.pending_controls:
                image = tuple(self.plant.step(group, control) for group in groups)
                if can_wait(image, remaining - 1):
                    policy[(groups, remaining)] = control
                    return True
            return False

        completion_beliefs = frozenset(
            belief for belief in safe_beliefs if completion_safe(condition(belief))
        )
        layers = tuple(
            frozenset(
                belief for belief in safe_beliefs if can_wait(condition(belief), delay)
            )
            for delay in range(self.max_delay + 1)
        )

        return GuaranteedCompletionResult(
            partition=self.partition,
            recovery_beliefs=recovery_beliefs,
            completion_beliefs=completion_beliefs,
            layers=layers,
            pending_policy=policy,
        )


def _set_partitions(items: tuple[StateT, ...]) -> Iterator[tuple[frozenset[StateT], ...]]:
    """Yield each unlabeled set partition once in deterministic order."""

    if not items:
        return

    blocks: list[list[StateT]] = [[items[0]]]

    def visit(index: int) -> Iterator[tuple[frozenset[StateT], ...]]:
        if index == len(items):
            yield tuple(frozenset(block) for block in blocks)
            return
        item = items[index]
        for block in blocks:
            block.append(item)
            yield from visit(index + 1)
            block.pop()
        blocks.append([item])
        yield from visit(index + 1)
        blocks.pop()

    yield from visit(1)


def enumerate_output_partitions(
    plant: FinitePlant[StateT, ControlT],
) -> Iterator[OutputPartition[StateT]]:
    """Enumerate behaviorally distinct partitions of the safe state set.

    Unsafe states never occur in a certified completion posterior, so assigning
    all of them to the first cell avoids redundant Bell-number growth while
    still producing a partition of the complete plant state space.
    """

    safe = tuple(sorted(plant.safe_states, key=repr))
    unsafe = plant.states.difference(plant.safe_states)
    for index, safe_cells in enumerate(_set_partitions(safe)):
        cells = list(safe_cells)
        cells[0] = frozenset(cells[0].union(unsafe))
        yield OutputPartition(name=f"partition_{index}", cells=tuple(cells))


def minimum_symbols_for_delay(
    plant: FinitePlant[StateT, ControlT],
    belief: Belief[StateT],
    *,
    pending_controls: tuple[ControlT, ...],
    recovery_controls: tuple[ControlT, ...],
    recovery_horizon: int,
    delay: int,
) -> MinimumSymbolResult[StateT, ControlT]:
    """Enumerate static partitions and return an exact minimum-symbol witness."""

    if not belief or not belief <= plant.safe_states:
        raise ValueError("belief must be a nonempty subset of safe states")
    if delay < 0:
        raise ValueError("delay must be nonnegative")

    best: MinimumSymbolResult[StateT, ControlT] | None = None
    for partition in enumerate_output_partitions(plant):
        symbols = len(partition.cells)
        if best is not None and symbols >= best.minimum_symbols:
            continue
        solved = GuaranteedCompletionEnvelope(
            plant,
            output_partition=partition,
            pending_controls=pending_controls,
            recovery_controls=recovery_controls,
            recovery_horizon=recovery_horizon,
            max_delay=delay,
        ).solve()
        if solved.supports_delay(belief, delay):
            best = MinimumSymbolResult(
                delay=delay,
                minimum_symbols=symbols,
                partition=partition,
                pending_controls=solved.witness(plant, belief, delay),
            )
    if best is not None:
        return best
    return MinimumSymbolResult(
        delay=delay,
        minimum_symbols=None,
        partition=None,
        pending_controls=None,
    )


def _candidate_subbeliefs(belief: Belief[StateT]) -> tuple[Belief[StateT], ...]:
    candidates: set[Belief[StateT]] = set()
    ordered = tuple(sorted(belief, key=repr))
    for size in range(1, len(ordered) + 1):
        candidates.update(frozenset(items) for items in combinations(ordered, size))
    return tuple(sorted(candidates, key=lambda item: (len(item), tuple(sorted(map(repr, item))))))


def maximal_feasible_color_classes(
    vertices: frozenset[StateT], hyperedges: Iterable[Belief[StateT]]
) -> tuple[Belief[StateT], ...]:
    """Return all maximal vertex subsets containing no complete hyperedge."""

    if not vertices:
        raise ValueError("vertices must be nonempty")
    edges = tuple(hyperedges)
    if any(not edge or not edge <= vertices for edge in edges):
        raise ValueError("hyperedges must be nonempty subsets of vertices")
    feasible = tuple(
        candidate
        for candidate in _candidate_subbeliefs(vertices)
        if not any(edge <= candidate for edge in edges)
    )
    maximal = tuple(
        candidate
        for candidate in feasible
        if all(
            any(edge <= candidate.union({state}) for edge in edges)
            for state in vertices.difference(candidate)
        )
    )
    return tuple(
        sorted(maximal, key=lambda item: (-len(item), tuple(sorted(map(repr, item)))))
    )


def greedy_feasible_cover(
    vertices: frozenset[StateT], hyperedges: Iterable[Belief[StateT]]
) -> tuple[Belief[StateT], ...] | None:
    """Greedily partition vertices using maximal feasible color classes.

    The selected maximal sets form the standard unit-cost greedy Set Cover solution.
    Returning only newly covered vertices makes the witness cells disjoint; downward
    closure preserves their feasibility.  ``None`` means a singleton edge prevents
    any cover.
    """

    classes = maximal_feasible_color_classes(vertices, hyperedges)
    uncovered = set(vertices)
    cells: list[Belief[StateT]] = []
    while uncovered:
        best = max(
            classes,
            key=lambda candidate: (
                len(candidate.intersection(uncovered)),
                tuple(sorted(map(repr, candidate))),
            ),
            default=frozenset(),
        )
        cell = frozenset(best.intersection(uncovered))
        if not cell:
            return None
        cells.append(cell)
        uncovered.difference_update(cell)
    return tuple(cells)


def _sequence_conflicts(
    plant: FinitePlant[StateT, ControlT],
    belief: Belief[StateT],
    pending_sequence: tuple[ControlT, ...],
    *,
    recovery_controls: tuple[ControlT, ...],
    recovery_horizon: int,
) -> tuple[Belief[StateT], ...]:
    candidates = _candidate_subbeliefs(belief)
    trajectories: dict[Belief[StateT], tuple[Belief[StateT], ...]] = {}
    images: list[Belief[StateT]] = []
    for candidate in candidates:
        trajectory = [candidate]
        current = candidate
        for control in pending_sequence:
            current = plant.step(current, control)
            trajectory.append(current)
        trajectories[candidate] = tuple(trajectory)
        images.extend(trajectory)

    pointwise = SafeComputeEnvelope(
        plant,
        recovery_controls=recovery_controls,
        recovery_horizon=recovery_horizon,
        max_pending_ticks=0,
    ).solve_for(images)
    valid = {
        candidate: all(pointwise.budget(image) == 0 for image in trajectory)
        for candidate, trajectory in trajectories.items()
    }
    conflicts = (
        candidate
        for candidate in candidates
        if not valid[candidate]
        and (
            len(candidate) == 1
            or all(valid[frozenset(candidate.difference({state}))] for state in candidate)
        )
    )
    return tuple(conflicts)


def minimum_hypergraph_colors(
    vertices: frozenset[StateT], hyperedges: Iterable[Belief[StateT]]
) -> int | None:
    """Return the exact weak chromatic number by symmetry-broken backtracking."""

    edges = tuple(hyperedges)
    if any(not edge or not edge <= vertices for edge in edges):
        raise ValueError("hyperedges must be nonempty subsets of vertices")
    if any(len(edge) == 1 for edge in edges):
        return None
    if not vertices:
        raise ValueError("vertices must be nonempty")
    if not edges:
        return 1

    incidence = Counter(state for edge in edges for state in edge)
    ordered = tuple(sorted(vertices, key=lambda state: (-incidence[state], repr(state))))
    positions = {state: index for index, state in enumerate(ordered)}
    indexed_edges = tuple(tuple(positions[state] for state in edge) for edge in edges)
    incident_edges: list[list[tuple[int, ...]]] = [[] for _ in ordered]
    for edge in indexed_edges:
        for index in edge:
            incident_edges[index].append(edge)

    def colorable(symbols: int) -> bool:
        colors = [-1] * len(ordered)

        def visit(index: int, largest_used: int) -> bool:
            if index == len(ordered):
                return True
            largest_candidate = min(largest_used + 1, symbols - 1)
            for color in range(largest_candidate + 1):
                colors[index] = color
                violates_edge = False
                for edge in incident_edges[index]:
                    edge_colors = [colors[item] for item in edge]
                    if -1 not in edge_colors and len(set(edge_colors)) == 1:
                        violates_edge = True
                        break
                if not violates_edge and visit(index + 1, max(largest_used, color)):
                    return True
            colors[index] = -1
            return False

        return visit(0, -1)

    for symbols in range(1, len(ordered) + 1):
        if colorable(symbols):
            return symbols
    raise AssertionError("distinct colors must color every nonsingleton hyperedge")


def sequence_information_requirement(
    plant: FinitePlant[StateT, ControlT],
    belief: Belief[StateT],
    pending_sequence: tuple[ControlT, ...],
    *,
    recovery_controls: tuple[ControlT, ...],
    recovery_horizon: int,
) -> int | None:
    """Compute the hypergraph characterization for one pending sequence."""

    if not belief or not belief <= plant.safe_states:
        raise ValueError("belief must be a nonempty subset of safe states")
    current = belief
    for control in pending_sequence:
        current = plant.step(current, control)
        if not plant.is_safe(current):
            return None
    conflicts = _sequence_conflicts(
        plant,
        belief,
        pending_sequence,
        recovery_controls=recovery_controls,
        recovery_horizon=recovery_horizon,
    )
    return minimum_hypergraph_colors(belief, conflicts)


def minimum_symbols_via_hypergraph(
    plant: FinitePlant[StateT, ControlT],
    belief: Belief[StateT],
    *,
    pending_controls: tuple[ControlT, ...],
    recovery_controls: tuple[ControlT, ...],
    recovery_horizon: int,
    delay: int,
) -> HypergraphCoDesignResult[StateT, ControlT]:
    """Minimize the recovery-conflict chromatic number over pending sequences."""

    if delay < 0 or recovery_horizon < 0:
        raise ValueError("horizons must be nonnegative")
    if not belief or not belief <= plant.safe_states:
        raise ValueError("belief must be a nonempty subset of safe states")
    if not pending_controls or not recovery_controls:
        raise ValueError("control sets must be nonempty")
    for control in (*pending_controls, *recovery_controls):
        if control not in plant.controls:
            raise ValueError(f"unknown control: {control!r}")

    best_symbols: int | None = None
    best_sequence: tuple[ControlT, ...] | None = None
    best_conflicts: tuple[Belief[StateT], ...] | None = None
    for sequence in product(pending_controls, repeat=delay):
        prefixes = [belief]
        current = belief
        for control in sequence:
            current = plant.step(current, control)
            if not plant.is_safe(current):
                break
            prefixes.append(current)
        else:
            conflicts = _sequence_conflicts(
                plant,
                belief,
                sequence,
                recovery_controls=recovery_controls,
                recovery_horizon=recovery_horizon,
            )
            symbols = minimum_hypergraph_colors(belief, conflicts)
            if symbols is not None and (
                best_symbols is None or symbols < best_symbols
            ):
                best_symbols = symbols
                best_sequence = sequence
                best_conflicts = conflicts

    return HypergraphCoDesignResult(
        delay=delay,
        minimum_symbols=best_symbols,
        pending_controls=best_sequence,
        conflict_hyperedges=best_conflicts,
    )

from __future__ import annotations

import random
from fractions import Fraction
from itertools import combinations, product

from safe_compute.finite import FinitePlant
from safe_compute.information_liveness import (
    _sequence_conflicts,
    greedy_feasible_cover,
    minimum_symbols_via_hypergraph,
)
from safe_compute.oracle import brute_force_minimum_symbols
from safe_compute.separation import OutputPartition


def _recoverable_by_definition(
    plant: FinitePlant,
    belief: frozenset,
    recovery_controls: tuple,
    horizon: int,
) -> bool:
    if not plant.is_safe(belief):
        return False
    if belief <= plant.recovery_states:
        return True
    for length in range(1, horizon + 1):
        for sequence in product(recovery_controls, repeat=length):
            image = belief
            for control in sequence:
                image = plant.step(image, control)
                if not plant.is_safe(image):
                    break
            else:
                if image <= plant.recovery_states:
                    return True
    return False


def _padded_set_cover_plant(
    universe: tuple[str, ...], covers: tuple[frozenset[str], ...]
) -> tuple[FinitePlant[str, str], tuple[str, ...]]:
    recovery = tuple(f"recover_{index}" for index in range(len(covers)))
    controls = ("wait", *recovery)
    states = frozenset((*universe, "recovered", "unsafe"))
    transitions: dict[tuple[str, str], frozenset[str]] = {}
    for state in states:
        for control in controls:
            if state in {"recovered", "unsafe"}:
                successor = state
            elif control == "wait":
                successor = state
            else:
                index = int(control.removeprefix("recover_"))
                successor = "recovered" if state in covers[index] else "unsafe"
            transitions[(state, control)] = frozenset({successor})
    return (
        FinitePlant(
            states=states,
            safe_states=states.difference({"unsafe"}),
            recovery_states=frozenset({"recovered"}),
            controls=controls,
            transitions=transitions,
        ),
        recovery,
    )


def test_fixed_deadline_padding_preserves_random_set_cover_optima() -> None:
    rng = random.Random(20260905)
    for _ in range(12):
        universe = tuple(f"x{index}" for index in range(rng.randint(3, 4)))
        mutable_covers = [
            {state for state in universe if rng.random() < 0.5} for _ in range(4)
        ]
        for state in universe:
            if not any(state in cover for cover in mutable_covers):
                rng.choice(mutable_covers).add(state)
        covers = tuple(frozenset(cover) for cover in mutable_covers)
        optimum = min(
            size
            for size in range(1, len(covers) + 1)
            if any(
                set().union(*(covers[index] for index in indices)) == set(universe)
                for indices in combinations(range(len(covers)), size)
            )
        )
        plant, recovery = _padded_set_cover_plant(universe, covers)
        belief = frozenset(universe)
        for delay in range(4):
            independent = brute_force_minimum_symbols(
                plant,
                belief,
                pending_controls=("wait",),
                recovery_controls=recovery,
                recovery_horizon=1,
                delay=delay,
            )
            production = minimum_symbols_via_hypergraph(
                plant,
                belief,
                pending_controls=("wait",),
                recovery_controls=recovery,
                recovery_horizon=1,
                delay=delay,
            ).minimum_symbols
            assert independent == production == optimum


def test_recovery_horizon_monotonicity_on_nondeterministic_plants() -> None:
    rng = random.Random(20260906)
    states = frozenset({0, 1, 2, 3})
    safe = frozenset({0, 1, 2})
    controls = ("p0", "p1", "r0", "r1")
    for _ in range(24):
        transitions: dict[tuple[int, str], frozenset[int]] = {}
        for state in states:
            for control in controls:
                if state in {0, 3}:
                    successors = frozenset({state})
                else:
                    successors = frozenset(
                        rng.sample(tuple(states), rng.choice((1, 2)))
                    )
                transitions[(state, control)] = successors
        transitions[(1, "p0")] = frozenset({0, 1})
        plant = FinitePlant(
            states=states,
            safe_states=safe,
            recovery_states=frozenset({0}),
            controls=controls,
            transitions=transitions,
        )
        belief = frozenset({1, 2})
        for delay in range(3):
            previous = len(belief) + 1
            for horizon in range(4):
                independent = brute_force_minimum_symbols(
                    plant,
                    belief,
                    pending_controls=controls[:2],
                    recovery_controls=controls[2:],
                    recovery_horizon=horizon,
                    delay=delay,
                )
                production = minimum_symbols_via_hypergraph(
                    plant,
                    belief,
                    pending_controls=controls[:2],
                    recovery_controls=controls[2:],
                    recovery_horizon=horizon,
                    delay=delay,
                ).minimum_symbols
                assert production == independent
                value = len(belief) + 1 if independent is None else independent
                assert value <= previous
                previous = value


def _recoverable_singleton_random_plant(
    rng: random.Random,
) -> tuple[FinitePlant[str, str], frozenset[str], tuple[str, ...], tuple[str, ...]]:
    uncertain = ("a", "b", "c")
    pending = ("p0", "p1")
    recovery = ("r0", "r1", "r2")
    controls = (*pending, *recovery)
    states = frozenset((*uncertain, "recovered", "unsafe"))
    allowed = {
        state: frozenset(control for control in recovery if rng.random() < 0.5)
        for state in uncertain
    }
    for state in uncertain:
        if not allowed[state]:
            allowed[state] = frozenset({rng.choice(recovery)})
    transitions: dict[tuple[str, str], frozenset[str]] = {}
    for state in states:
        for control in controls:
            if state in {"recovered", "unsafe"}:
                successors = frozenset({state})
            elif control in recovery:
                successors = frozenset(
                    {"recovered" if control in allowed[state] else "unsafe"}
                )
            else:
                successors = frozenset(rng.sample(uncertain, rng.choice((1, 2))))
            transitions[(state, control)] = successors
    return (
        FinitePlant(
            states=states,
            safe_states=states.difference({"unsafe"}),
            recovery_states=frozenset({"recovered"}),
            controls=controls,
            transitions=transitions,
        ),
        frozenset(uncertain),
        pending,
        recovery,
    )


def test_greedy_frontier_witnesses_against_nondeterministic_sequence_oracle() -> None:
    rng = random.Random(20260908)
    for _ in range(24):
        plant, belief, pending, recovery = _recoverable_singleton_random_plant(rng)
        harmonic = sum(Fraction(1, index) for index in range(1, len(belief) + 1))
        for delay in range(3):
            optimum = brute_force_minimum_symbols(
                plant,
                belief,
                pending_controls=pending,
                recovery_controls=recovery,
                recovery_horizon=2,
                delay=delay,
            )
            candidate_sizes = []
            for sequence in product(pending, repeat=delay):
                image = belief
                for control in sequence:
                    image = plant.step(image, control)
                    if not plant.is_safe(image):
                        break
                else:
                    conflicts = _sequence_conflicts(
                        plant, belief, sequence,
                        recovery_controls=recovery, recovery_horizon=2,
                    )
                    cells = greedy_feasible_cover(belief, conflicts)
                    if cells is None:
                        continue
                    assert frozenset().union(*cells) == belief
                    assert sum(map(len, cells)) == len(belief)
                    for cell in cells:
                        image = cell
                        # Direct recovery-sequence enumeration at every age checks
                        # the returned witness without using the conflict predicate.
                        assert _recoverable_by_definition(plant, image, recovery, 2)
                        for control in sequence:
                            image = plant.step(image, control)
                            assert _recoverable_by_definition(plant, image, recovery, 2)
                    candidate_sizes.append(len(cells))
            if optimum is None:
                assert not candidate_sizes
            else:
                assert candidate_sizes
                assert optimum <= min(candidate_sizes) <= harmonic * optimum


def _immediate_pair_clique_number(
    plant: FinitePlant,
    belief: frozenset,
    recovery_controls: tuple,
    horizon: int,
) -> int:
    vertices = tuple(sorted(belief, key=repr))
    conflict = {
        frozenset(pair)
        for pair in combinations(vertices, 2)
        if not _recoverable_by_definition(
            plant, frozenset(pair), recovery_controls, horizon
        )
    }
    return max(
        len(candidate)
        for size in range(1, len(vertices) + 1)
        for candidate in combinations(vertices, size)
        if all(frozenset(pair) in conflict for pair in combinations(candidate, 2))
    )


def test_immediate_pair_clique_is_lower_bound_on_random_nondeterministic_frontiers() -> None:
    rng = random.Random(20260907)
    for _ in range(30):
        plant, belief, pending, recovery = _recoverable_singleton_random_plant(rng)
        lower_bound = _immediate_pair_clique_number(plant, belief, recovery, 1)
        for delay in range(3):
            optimum = brute_force_minimum_symbols(
                plant,
                belief,
                pending_controls=pending,
                recovery_controls=recovery,
                recovery_horizon=1,
                delay=delay,
            )
            if optimum is not None:
                assert lower_bound <= optimum


def test_static_partition_is_strictly_more_costly_than_age_dependent_outputs() -> None:
    states = frozenset({"a", "b", "c", "recovered", "unsafe"})
    recovery = ("recover_ac", "recover_bc")
    controls = ("cycle", *recovery)
    transitions: dict[tuple[str, str], frozenset[str]] = {}
    cycle = {"a": "c", "b": "a", "c": "b"}
    covered = {
        "recover_ac": frozenset({"a", "c"}),
        "recover_bc": frozenset({"b", "c"}),
    }
    for state in states:
        for control in controls:
            if state in {"recovered", "unsafe"}:
                successor = state
            elif control == "cycle":
                successor = cycle[state]
            else:
                successor = "recovered" if state in covered[control] else "unsafe"
            transitions[(state, control)] = frozenset({successor})
    plant = FinitePlant(
        states=states,
        safe_states=states.difference({"unsafe"}),
        recovery_states=frozenset({"recovered"}),
        controls=controls,
        transitions=transitions,
    )
    belief = frozenset({"a", "b", "c"})

    static = brute_force_minimum_symbols(
        plant,
        belief,
        pending_controls=("cycle",),
        recovery_controls=recovery,
        recovery_horizon=1,
        delay=2,
    )
    assert static == 3
    assert minimum_symbols_via_hypergraph(
        plant,
        belief,
        pending_controls=("cycle",),
        recovery_controls=recovery,
        recovery_horizon=1,
        delay=2,
    ).minimum_symbols == static

    age_partitions = (
        OutputPartition(
            "age_0",
            (frozenset({"a", "c", "recovered", "unsafe"}), frozenset({"b"})),
        ),
        OutputPartition(
            "age_1",
            (frozenset({"a", "b", "recovered", "unsafe"}), frozenset({"c"})),
        ),
        OutputPartition(
            "age_2",
            (frozenset({"a", "b", "recovered", "unsafe"}), frozenset({"c"})),
        ),
    )
    for age, partition in enumerate(age_partitions):
        full_image = belief
        for _ in range(age):
            full_image = plant.step(full_image, "cycle")
        assert not _recoverable_by_definition(plant, full_image, recovery, 1)

        for cell in partition.cells:
            posterior = belief.intersection(cell)
            if not posterior:
                continue
            for _ in range(age):
                posterior = plant.step(posterior, "cycle")
            assert _recoverable_by_definition(plant, posterior, recovery, 1)

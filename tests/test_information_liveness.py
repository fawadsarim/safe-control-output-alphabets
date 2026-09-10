from __future__ import annotations

import random

import pytest

from safe_compute.finite import FinitePlant
from safe_compute.information_liveness import (
    GuaranteedCompletionEnvelope,
    enumerate_output_partitions,
    minimum_symbols_for_delay,
    minimum_symbols_via_hypergraph,
    sequence_information_requirement,
)
from safe_compute.oracle import (
    brute_force_guaranteed_completion,
    brute_force_minimum_symbols,
)
from safe_compute.separation import OutputPartition


def unique_recovery_plant(size: int) -> FinitePlant[str, str]:
    uncertain = tuple(f"x{index}" for index in range(size))
    states = frozenset((*uncertain, "recovered", "unsafe"))
    controls = tuple(f"recover_{index}" for index in range(size))
    transitions: dict[tuple[str, str], frozenset[str]] = {}
    for state in states:
        for index, control in enumerate(controls):
            if state == f"x{index}":
                successor = "recovered"
            elif state in {"recovered", "unsafe"}:
                successor = state
            else:
                successor = "unsafe"
            transitions[(state, control)] = frozenset({successor})
    return FinitePlant(
        states=states,
        safe_states=frozenset((*uncertain, "recovered")),
        recovery_states=frozenset({"recovered"}),
        controls=controls,
        transitions=transitions,
    )


@pytest.mark.parametrize("size", range(2, 7))
def test_arbitrary_alphabet_gap_family(size: int) -> None:
    plant = unique_recovery_plant(size)
    assert len(plant.states) == size + 2
    assert len(plant.controls) == size
    belief = frozenset(f"x{index}" for index in range(size))
    result = minimum_symbols_for_delay(
        plant,
        belief,
        pending_controls=plant.controls,
        recovery_controls=plant.controls,
        recovery_horizon=1,
        delay=0,
    )

    assert result.minimum_symbols == size
    assert result.pending_controls == ()
    assert brute_force_minimum_symbols(
        plant,
        belief,
        pending_controls=plant.controls,
        recovery_controls=plant.controls,
        recovery_horizon=1,
        delay=0,
    ) == size


def test_fixed_partition_layers_match_exhaustive_oracle() -> None:
    rng = random.Random(20260904)
    for instance in range(20):
        safe = frozenset(range(4))
        states = safe.union({4})
        controls = (0, 1)
        transitions = {
            (state, control): frozenset(
                {rng.randrange(5), rng.randrange(5)}
            )
            for state in states
            for control in controls
        }
        # Make the absorbing unsafe state and at least one terminal state explicit.
        transitions[(4, 0)] = transitions[(4, 1)] = frozenset({4})
        plant = FinitePlant(
            states=states,
            safe_states=safe,
            recovery_states=frozenset({0}),
            controls=controls,
            transitions=transitions,
        )
        partitions = tuple(enumerate_output_partitions(plant))
        sample = rng.sample(partitions, min(4, len(partitions)))
        belief = frozenset(rng.sample(tuple(safe), rng.randint(1, len(safe))))
        for partition in sample:
            solved = GuaranteedCompletionEnvelope(
                plant,
                output_partition=partition,
                pending_controls=controls,
                recovery_controls=controls,
                recovery_horizon=2,
                max_delay=2,
            ).solve()
            for delay in range(3):
                assert solved.supports_delay(
                    belief, delay
                ) == brute_force_guaranteed_completion(
                    plant,
                    belief,
                    output_partition=partition,
                    pending_controls=controls,
                    recovery_controls=controls,
                    recovery_horizon=2,
                    delay=delay,
                ), (instance, partition, belief, delay)


def test_delayed_output_conditions_the_captured_state_then_propagates() -> None:
    states = frozenset({"a", "b", "c", "d", "recovered", "unsafe"})
    controls = ("advance", "recover_initial", "recover_c", "recover_d")
    transitions: dict[tuple[str, str], frozenset[str]] = {}
    for state in states:
        for control in controls:
            if state in {"recovered", "unsafe"}:
                successor = state
            elif control == "advance":
                successor = {"a": "c", "b": "d", "c": "c", "d": "d"}[state]
            elif control == "recover_initial" and state in {"a", "b"}:
                successor = "recovered"
            elif control == "recover_c" and state == "c":
                successor = "recovered"
            elif control == "recover_d" and state == "d":
                successor = "recovered"
            else:
                successor = "unsafe"
            transitions[(state, control)] = frozenset({successor})
    plant = FinitePlant(
        states=states,
        safe_states=states.difference({"unsafe"}),
        recovery_states=frozenset({"recovered"}),
        controls=controls,
        transitions=transitions,
    )
    belief = frozenset({"a", "b"})

    # This map separates the completion-time states c,d but not the captured
    # states a,b.  A stale label therefore propagates to {c,d} and cannot recover.
    stale_coarse = OutputPartition(
        "stale_coarse",
        (frozenset({"a", "b", "c"}), frozenset({"d", "recovered", "unsafe"})),
    )
    solved = GuaranteedCompletionEnvelope(
        plant,
        output_partition=stale_coarse,
        pending_controls=("advance",),
        recovery_controls=controls[1:],
        recovery_horizon=1,
        max_delay=1,
    ).solve()
    assert solved.supports_delay(belief, 0)
    assert not solved.supports_delay(belief, 1)
    assert not brute_force_guaranteed_completion(
        plant,
        belief,
        output_partition=stale_coarse,
        pending_controls=("advance",),
        recovery_controls=controls[1:],
        recovery_horizon=1,
        delay=1,
    )

    assert [
        minimum_symbols_via_hypergraph(
            plant,
            belief,
            pending_controls=("advance",),
            recovery_controls=controls[1:],
            recovery_horizon=1,
            delay=delay,
        ).minimum_symbols
        for delay in (0, 1)
    ] == [1, 2]


def test_minimum_symbols_matches_labeled_map_oracle() -> None:
    rng = random.Random(1701)
    for _ in range(12):
        safe = frozenset(range(3))
        states = safe.union({3})
        controls = (0, 1)
        transitions = {
            (state, control): frozenset({rng.randrange(4)})
            for state in states
            for control in controls
        }
        transitions[(3, 0)] = transitions[(3, 1)] = frozenset({3})
        plant = FinitePlant(
            states=states,
            safe_states=safe,
            recovery_states=frozenset({0}),
            controls=controls,
            transitions=transitions,
        )
        belief = safe
        for delay in range(3):
            dynamic = minimum_symbols_for_delay(
                plant,
                belief,
                pending_controls=controls,
                recovery_controls=controls,
                recovery_horizon=1,
                delay=delay,
            ).minimum_symbols
            exhaustive = brute_force_minimum_symbols(
                plant,
                belief,
                pending_controls=controls,
                recovery_controls=controls,
                recovery_horizon=1,
                delay=delay,
            )
            assert dynamic == exhaustive
            assert minimum_symbols_via_hypergraph(
                plant,
                belief,
                pending_controls=controls,
                recovery_controls=controls,
                recovery_horizon=1,
                delay=delay,
            ).minimum_symbols == exhaustive


def test_hypergraph_characterization_matches_partition_search() -> None:
    plant = unique_recovery_plant(5)
    belief = frozenset(f"x{index}" for index in range(5))
    assert sequence_information_requirement(
        plant,
        belief,
        (),
        recovery_controls=plant.controls,
        recovery_horizon=1,
    ) == 5
    assert minimum_symbols_for_delay(
        plant,
        belief,
        pending_controls=plant.controls,
        recovery_controls=plant.controls,
        recovery_horizon=1,
        delay=0,
    ).minimum_symbols == 5


def test_singleton_recovery_conflict_is_impossible() -> None:
    states = frozenset({"stuck", "recovered", "unsafe"})
    plant = FinitePlant(
        states=states,
        safe_states=frozenset({"stuck", "recovered"}),
        recovery_states=frozenset({"recovered"}),
        controls=("hold",),
        transitions={
            ("stuck", "hold"): frozenset({"stuck"}),
            ("recovered", "hold"): frozenset({"recovered"}),
            ("unsafe", "hold"): frozenset({"unsafe"}),
        },
    )
    belief = frozenset({"stuck"})
    assert sequence_information_requirement(
        plant,
        belief,
        (),
        recovery_controls=plant.controls,
        recovery_horizon=1,
    ) is None
    assert minimum_symbols_for_delay(
        plant,
        belief,
        pending_controls=plant.controls,
        recovery_controls=plant.controls,
        recovery_horizon=1,
        delay=0,
    ).minimum_symbols is None


def test_three_state_conflict_requires_two_symbols_not_three() -> None:
    states = frozenset({"a", "b", "c", "recovered", "unsafe"})
    controls = ("ab", "ac", "bc")
    covered = {
        "ab": frozenset({"a", "b"}),
        "ac": frozenset({"a", "c"}),
        "bc": frozenset({"b", "c"}),
    }
    transitions = {}
    for state in states:
        for control in controls:
            if state in {"recovered", "unsafe"}:
                successor = state
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

    assert sequence_information_requirement(
        plant,
        belief,
        (),
        recovery_controls=controls,
        recovery_horizon=1,
    ) == 2
    hypergraph = minimum_symbols_via_hypergraph(
        plant,
        belief,
        pending_controls=controls,
        recovery_controls=controls,
        recovery_horizon=1,
        delay=0,
    )
    assert hypergraph.minimum_symbols == 2
    assert hypergraph.conflict_hyperedges == (belief,)
    assert not any(len(edge) == 2 for edge in hypergraph.conflict_hyperedges)
    assert minimum_symbols_for_delay(
        plant,
        belief,
        pending_controls=controls,
        recovery_controls=controls,
        recovery_horizon=1,
        delay=0,
    ).minimum_symbols == 2
    assert brute_force_minimum_symbols(
        plant,
        belief,
        pending_controls=controls,
        recovery_controls=controls,
        recovery_horizon=1,
        delay=0,
    ) == 2


@pytest.mark.parametrize("max_delay", range(1, 6))
def test_deadline_escalation_family_has_exact_staircase(max_delay: int) -> None:
    from experiments.pc.information_completion_frontier import build_escalating_plant

    branches = max_delay + 1
    plant = build_escalating_plant(branches=branches, max_delay=max_delay)
    assert len(plant.states) == (max_delay + 1) ** 2 + 2
    assert len(plant.controls) == max_delay + 2
    belief = frozenset(f"x_0_{branch}" for branch in range(branches))
    recovery = tuple(f"recover_{index}" for index in range(branches))
    observed = [
        minimum_symbols_via_hypergraph(
            plant,
            belief,
            pending_controls=("advance",),
            recovery_controls=recovery,
            recovery_horizon=1,
            delay=delay,
        ).minimum_symbols
        for delay in range(max_delay + 1)
    ]
    assert observed == list(range(1, max_delay + 2))


def test_bounded_exhaustive_audit_counts_every_query() -> None:
    from experiments.pc.information_completion_frontier import (
        run_bounded_exhaustive_audit,
    )

    result = run_bounded_exhaustive_audit(
        {"safe_states": 1, "controls": 1, "recovery_horizon": 1, "max_delay": 1}
    )
    assert result == {"plants": 2, "queries": 4}


def test_refinement_and_deadline_monotonicity() -> None:
    plant = unique_recovery_plant(3)
    belief = frozenset({"x0", "x1", "x2"})
    coarse = OutputPartition("coarse", (plant.states,))
    fine = OutputPartition(
        "fine", tuple(frozenset({state}) for state in sorted(plant.states))
    )
    for partition in (coarse, fine):
        solved = GuaranteedCompletionEnvelope(
            plant,
            output_partition=partition,
            pending_controls=plant.controls,
            recovery_controls=plant.controls,
            recovery_horizon=1,
            max_delay=2,
        ).solve()
        assert solved.layers[2] <= solved.layers[1] <= solved.layers[0]
    assert belief not in GuaranteedCompletionEnvelope(
        plant,
        output_partition=coarse,
        recovery_horizon=1,
        max_delay=0,
    ).solve().layers[0]
    assert belief in GuaranteedCompletionEnvelope(
        plant,
        output_partition=fine,
        recovery_horizon=1,
        max_delay=0,
    ).solve().layers[0]

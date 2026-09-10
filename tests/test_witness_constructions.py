"""Feasible witness constructions and boundary countermodels.

Reference calculations read transition tables directly and minimize partitions
by subset dynamic programming. They do not call the production recovery,
hypergraph, candidate-generation, or partition-enumeration routines.
"""

from __future__ import annotations

import random
from fractions import Fraction
from functools import cache
from itertools import combinations, product
from math import prod

import pytest

from safe_compute.constructions import build_product_age_gap_plant
from safe_compute.finite import FinitePlant
from safe_compute.information_liveness import (
    minimum_symbols_for_delay,
    minimum_symbols_via_hypergraph,
)
from safe_compute.oracle import brute_force_minimum_symbols
from safe_compute.recovery_cover import (
    greedy_symbols_via_recovery_cover,
    sequence_recovery_cover,
)


def _forward(plant, belief, word):
    image = belief
    if not image or not image <= plant.safe_states:
        return None
    for control in word:
        image = frozenset().union(*(plant.transitions[(state, control)] for state in image))
        if not image <= plant.safe_states:
            return None
    return image


def _direct_feasible_cells(plant, belief, sequence, recovery, horizon):
    if _forward(plant, belief, sequence) is None:
        return ()
    words = tuple(
        word for length in range(horizon + 1)
        for word in product(recovery, repeat=length)
    )
    feasible = []
    for size in range(1, len(belief) + 1):
        for values in combinations(sorted(belief), size):
            image = cell = frozenset(values)
            for age in range(len(sequence) + 1):
                if not any(
                    (end := _forward(plant, image, word)) is not None
                    and end <= plant.recovery_states
                    for word in words
                ):
                    break
                if age < len(sequence):
                    image = _forward(plant, image, (sequence[age],))
                    assert image is not None
            else:
                feasible.append(cell)
    return tuple(feasible)


def _partition_optimum(belief, feasible):
    # Partition, not cover: each selected cell must fit wholly in the remainder.
    @cache
    def remaining_optimum(remaining):
        if not remaining:
            return 0
        anchor = min(remaining)
        return min(
            (1 + remaining_optimum(remaining - cell)
             for cell in feasible if anchor in cell and cell <= remaining),
            default=len(belief) + 1,
        )

    answer = remaining_optimum(belief)
    return answer if answer <= len(belief) else None


def _positive_greedy_trap(seed):
    # A is the unique largest set, but B+C is optimal. The second pending input
    # destroys A's future usefulness, making greedy choose the optimal two cells.
    vertices = list(range(6))
    random.Random(seed).shuffle(vertices)
    success = {
        "rA": set(vertices[:4]),
        "rB": {vertices[0], vertices[1], vertices[4]},
        "rC": {vertices[2], vertices[3], vertices[5]},
    }
    captured = frozenset(f"x_{vertex}" for vertex in range(6))
    duplicates = {f"y_{vertex}_{branch}" for vertex in range(6) for branch in range(2)}
    states = captured | duplicates | {"k", "unsafe"}
    pending, recovery = ("wait", "switch", "bad"), ("rA", "rB", "rC")
    transitions = {}
    for state in states:
        for control in (*pending, *recovery):
            if state in {"k", "unsafe"}:
                targets = {state}
            elif control == "bad":
                targets = {"unsafe"}
            elif control == "wait" or (control == "switch" and state.startswith("y_")):
                targets = {state}
            elif control == "switch":
                vertex = int(state.split("_")[1])
                targets = {f"y_{vertex}_0", f"y_{vertex}_1"}
            else:
                vertex = int(state.split("_")[1])
                succeeds = vertex in success[control] and not (
                    control == "rA" and state.startswith("y_")
                )
                targets = {"k" if succeeds else "unsafe"}
            transitions[(state, control)] = frozenset(targets)
    plant = FinitePlant(
        states, states - {"unsafe"}, frozenset({"k"}), (*pending, *recovery), transitions,
    )
    return plant, captured, pending, recovery


@pytest.mark.parametrize("seed", range(12))
def test_72_predetermined_positive_queries_and_nonoptimal_greedy(seed):
    plant, belief, pending, recovery = _positive_greedy_trap(seed)
    for horizon in (1, 2):
        for delay in range(3):
            exact_answers = []
            for sequence in product(pending, repeat=delay):
                feasible = _direct_feasible_cells(plant, belief, sequence, recovery, horizon)
                exact = _partition_optimum(belief, feasible)
                if exact is not None:
                    exact_answers.append(exact)
                candidates = sequence_recovery_cover(
                    plant, belief, sequence,
                    recovery_controls=recovery, recovery_horizon=horizon,
                )
                # Candidate coverage must equal the direct all-subset definition,
                # including unsafe sequences and cells crossing a nondeterministic branch.
                assert all(
                    any(cell <= candidate.captured_states for candidate in candidates)
                    for cell in feasible
                )
                assert all(candidate.captured_states in feasible for candidate in candidates)
            assert min(exact_answers) == 2
            result = greedy_symbols_via_recovery_cover(
                plant, belief, pending_controls=pending, recovery_controls=recovery,
                recovery_horizon=horizon, delay=delay,
            )
            assert result is not None
            assert result.symbols == (3 if delay == 0 else 2)
            if delay:
                assert "switch" in result.pending_sequence
                assert "bad" not in result.pending_sequence
            harmonic = sum(Fraction(1, i) for i in range(1, len(belief) + 1))
            assert 2 <= result.symbols <= 2 * harmonic
            assert frozenset().union(*(cell.captured_states for cell in result.cells)) == belief
            assert sum(len(cell.captured_states) for cell in result.cells) == len(belief)
            for cell in result.cells:
                assert len(cell.recovery_sequences) == delay + 1
                for age, word in enumerate(cell.recovery_sequences):
                    assert len(word) <= horizon and all(control in recovery for control in word)
                    image = _forward(plant, cell.captured_states, result.pending_sequence[:age])
                    assert image is not None
                    end = _forward(plant, image, word)
                    assert end is not None and end <= plant.recovery_states

            waiting = greedy_symbols_via_recovery_cover(
                plant, belief, pending_controls=("wait",), recovery_controls=recovery,
                recovery_horizon=horizon, delay=delay,
            )
            assert waiting is not None and waiting.symbols == 3


@pytest.mark.parametrize("alphabets", [
    (1,), (2,), (3,), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (1, 2, 2), (2, 2, 2),
])
def test_product_bound_is_sharp_at_every_prefix_and_each_known_age(alphabets):
    plant, belief, pending, recovery = build_product_age_gap_plant(alphabets)
    assert len(belief) == prod(alphabets)
    assert len(plant.states) == len(alphabets) * len(belief) + 2
    assert len(plant.controls) == 1 + sum(alphabets)
    for delay in range(len(alphabets)):
        sequence = pending * delay
        feasible = _direct_feasible_cells(plant, belief, sequence, recovery, 1)
        expected_static = prod(alphabets[:delay + 1])
        assert _partition_optimum(belief, feasible) == expected_static
        actual = minimum_symbols_via_hypergraph(
            plant, belief, pending_controls=pending, recovery_controls=recovery,
            recovery_horizon=1, delay=delay,
        )
        assert actual.minimum_symbols == expected_static
        known_age_counts = []
        for age in range(delay + 1):
            image = _forward(plant, belief, pending * age)
            assert image is not None
            age_cells = _direct_feasible_cells(plant, image, (), recovery, 1)
            age_count = _partition_optimum(image, age_cells)
            assert age_count == alphabets[age]
            known_age_counts.append(age_count)
        assert max(known_age_counts) <= expected_static <= min(
            len(belief), prod(known_age_counts)
        )


@pytest.mark.parametrize("alphabets", [(), (0,), (-1, 2), (1.5,), (True,)])
def test_product_construction_rejects_invalid_alphabets(alphabets):
    with pytest.raises(ValueError, match="positive integers"):
        build_product_age_gap_plant(alphabets)


def test_buffering_countermodel_does_not_contradict_forced_switch_theorem():
    plant = FinitePlant(
        frozenset({"x", "k"}), frozenset({"x", "k"}), frozenset({"k"}), ("wait",),
        {("x", "wait"): frozenset({"k"}), ("k", "wait"): frozenset({"k"})},
    )
    belief = frozenset({"x"})
    forced = minimum_symbols_via_hypergraph(
        plant, belief, pending_controls=("wait",), recovery_controls=("wait",),
        recovery_horizon=0, delay=1,
    )
    assert forced.minimum_symbols is None  # Immediate completion leaves no recovery steps.
    for arrival_age in (0, 1):
        # A different interface buffers any age-zero result and switches at age one.
        switch_age = max(arrival_age, 1)
        image = _forward(plant, belief, ("wait",) * switch_age)
        assert image is not None and image <= plant.recovery_states


def test_pending_control_choice_reduces_exact_alphabet():
    # A constructed finite example, not a physical plant or measurement.
    controls = ("advance", "merge", "r0", "r1", "r2")
    rows = {
        "a": ("c", "c", "k", "unsafe", "unsafe"),
        "b": ("d", "c", "k", "unsafe", "unsafe"),
        "c": ("c", "c", "unsafe", "k", "unsafe"),
        "d": ("d", "c", "unsafe", "unsafe", "k"),
        "k": ("k",) * 5,
        "unsafe": ("unsafe",) * 5,
    }
    states = frozenset(rows)
    plant = FinitePlant(
        states, states - {"unsafe"}, frozenset({"k"}), controls,
        {(state, control): frozenset({target})
         for state, targets in rows.items()
         for control, target in zip(controls, targets, strict=True)},
    )
    belief, recovery = frozenset({"a", "b"}), controls[2:]
    words = ((), *((control,) for control in recovery))
    # Direct lower certificate: after advance, no common H=1 word recovers {c,d}.
    separated = _forward(plant, belief, ("advance",))
    assert separated == frozenset({"c", "d"})
    assert not any(
        (end := _forward(plant, separated, word)) is not None
        and end <= plant.recovery_states for word in words
    )
    for delay in range(4):
        for pending in (("advance",), ("advance", "merge")):
            expected = 2 if delay > 0 and len(pending) == 1 else 1
            kwargs = dict(
                pending_controls=pending, recovery_controls=recovery,
                recovery_horizon=1, delay=delay,
            )
            direct = min(
                _partition_optimum(
                    belief, _direct_feasible_cells(plant, belief, sequence, recovery, 1)
                )
                for sequence in product(pending, repeat=delay)
            )
            partition = minimum_symbols_for_delay(plant, belief, **kwargs)
            assert direct == expected
            assert partition.minimum_symbols == expected
            hypergraph = minimum_symbols_via_hypergraph(plant, belief, **kwargs)
            assert hypergraph.minimum_symbols == expected
            assert brute_force_minimum_symbols(plant, belief, **kwargs) == expected
            assert partition.partition is not None and partition.pending_controls is not None
            if delay > 0 and expected == 1:
                assert partition.pending_controls[0] == "merge"
            for cell in partition.partition.cells:
                captured = cell & belief
                if not captured:
                    continue
                for age in range(delay + 1):
                    image = _forward(plant, captured, partition.pending_controls[:age])
                    assert image is not None
                    assert any(
                        (end := _forward(plant, image, word)) is not None
                        and end <= plant.recovery_states for word in words
                    )

from __future__ import annotations

import random
from collections import Counter
from itertools import combinations, product

import pytest

from safe_compute.envelope import SafeComputeEnvelope
from safe_compute.finite import FinitePlant
from safe_compute.information_liveness import GuaranteedCompletionEnvelope
from safe_compute.oracle import brute_force_minimum_symbols
from safe_compute.recovery_cover import (
    greedy_symbols_via_recovery_cover,
    recovery_word_success_set,
    sequence_recovery_cover,
)
from safe_compute.separation import OutputPartition


def _forward_success(plant, belief, word):
    image = belief
    if not image or not image <= plant.safe_states:
        return False
    for control in word:
        # Read transitions directly: no production predecessor or recovery solver.
        image = frozenset().union(*(plant.transitions[(state, control)] for state in image))
        if not image <= plant.safe_states:
            return False
    return image <= plant.recovery_states


def _random_plant(seed):
    rng = random.Random(seed)
    controls = ("p0", "p1", "r0", "r1")
    states = frozenset(range(4))
    transitions = {
        (state, control): frozenset(rng.sample(range(4), rng.choice((1, 1, 2))))
        for state in sorted(states) for control in controls
    }
    # Deliberately do not make K absorbing: words must stop at the right time.
    return FinitePlant(states, frozenset({0, 1, 2}), frozenset({0}), controls, transitions)


def _exact_cover_size(belief, candidates):
    for size in range(1, len(candidates) + 1):
        for chosen in combinations(candidates, size):
            if frozenset().union(*(cell.captured_states for cell in chosen)) == belief:
                return size
    return None


def test_word_domains_equal_forward_universal_safety_checks() -> None:
    for seed in range(24):
        plant = _random_plant(seed)
        for length in range(4):
            for word in product(plant.controls[2:], repeat=length):
                expected = frozenset(
                    state for state in plant.states
                    if _forward_success(plant, frozenset({state}), word)
                )
                assert recovery_word_success_set(plant, word) == expected, (seed, word)


def test_cover_construction_exactness_and_greedy_witnesses_on_216_queries() -> None:
    # 24 nondeterministic plants x H=0,1,2 x d=0,1,2. Every subset and every
    # pending sequence is checked, including infeasibility, rather than retaining
    # only queries for which the new method returns an attractive answer.
    outcomes = Counter()
    for seed in range(24):
        plant = _random_plant(seed)
        belief = plant.safe_states
        pending, recovery = plant.controls[:2], plant.controls[2:]
        for horizon in range(3):
            words = tuple(
                word for length in range(horizon + 1)
                for word in product(recovery, repeat=length)
            )
            for delay in range(3):
                cover_optima = []
                for sequence in product(pending, repeat=delay):
                    candidates = sequence_recovery_cover(
                        plant, belief, sequence,
                        recovery_controls=recovery, recovery_horizon=horizon,
                    )
                    assert len(candidates) <= len(words) ** (delay + 1)
                    whole_image = belief
                    safe_sequence = True
                    for control in sequence:
                        whole_image = plant.step(whole_image, control)
                        safe_sequence &= plant.is_safe(whole_image)
                    if not safe_sequence:
                        assert candidates == ()
                        continue
                    for size in range(1, len(belief) + 1):
                        for states in combinations(sorted(belief), size):
                            image = subset = frozenset(states)
                            feasible = any(_forward_success(plant, image, w) for w in words)
                            for control in sequence:
                                image = plant.step(image, control)
                                feasible &= any(
                                    _forward_success(plant, image, w) for w in words
                                )
                            assert feasible == any(
                                subset <= cell.captured_states for cell in candidates
                            ), (seed, horizon, sequence, subset)
                    optimum = _exact_cover_size(belief, candidates)
                    if optimum is not None:
                        cover_optima.append(optimum)
                expected = brute_force_minimum_symbols(
                    plant, belief, pending_controls=pending, recovery_controls=recovery,
                    recovery_horizon=horizon, delay=delay,
                )
                outcomes[expected] += 1
                assert min(cover_optima, default=None) == expected
                result = greedy_symbols_via_recovery_cover(
                    plant, belief, pending_controls=pending, recovery_controls=recovery,
                    recovery_horizon=horizon, delay=delay,
                )
                if expected is None:
                    assert result is None
                    continue
                assert result is not None
                harmonic = sum(1 / i for i in range(1, len(belief) + 1))
                assert expected <= result.symbols <= harmonic * expected
                assert frozenset().union(*(c.captured_states for c in result.cells)) == belief
                assert sum(len(c.captured_states) for c in result.cells) == len(belief)
                for cell in result.cells:
                    assert len(cell.recovery_sequences) == delay + 1
                    image = cell.captured_states
                    for age, word in enumerate(cell.recovery_sequences):
                        assert len(word) <= horizon
                        assert _forward_success(plant, image, word)
                        if age < delay:
                            image = plant.step(image, result.pending_sequence[age])
    # Preserve the original suite, but make its feasibility imbalance explicit.
    assert outcomes == {None: 205, 1: 1, 2: 1, 3: 9}


def test_multistep_recovery_and_nonabsorbing_terminal_are_handled() -> None:
    # x -> y -> k -> unsafe; the complete 3-step word is not a success for x.
    states = frozenset({"x", "y", "k", "unsafe"})
    successor = {"x": "y", "y": "k", "k": "unsafe", "unsafe": "unsafe"}
    controls = ("wait", "r")
    plant = FinitePlant(
        states, states - {"unsafe"}, frozenset({"k"}), controls,
        {(s, u): frozenset({s if u == "wait" else successor[s]})
         for s in states for u in controls},
    )
    assert recovery_word_success_set(plant, ()) == frozenset({"k"})
    assert recovery_word_success_set(plant, ("r", "r")) == frozenset({"x"})
    assert recovery_word_success_set(plant, ("r", "r", "r")) == frozenset()
    for horizon in (1, 2):
        result = greedy_symbols_via_recovery_cover(
            plant, frozenset({"x"}), pending_controls=("wait",), recovery_controls=("r",),
            recovery_horizon=horizon, delay=2,
        )
        assert (result is not None) == (horizon == 2)
        if result is not None:
            assert result.cells[0].recovery_sequences == (("r", "r"),) * 3


def test_unsafe_prefix_cannot_be_hidden_by_safe_final_state() -> None:
    plant = FinitePlant(
        frozenset({0, 1, 2}), frozenset({0, 2}), frozenset({2}), ("r",),
        {(0, "r"): frozenset({1}), (1, "r"): frozenset({2}),
         (2, "r"): frozenset({2})},
    )
    assert recovery_word_success_set(plant, ("r", "r")) == frozenset({2})


def test_constructive_cover_uses_no_belief_power_set_on_64_states(monkeypatch) -> None:
    def forbidden_power_set(_self):
        raise AssertionError("the constructive cover must not enumerate safe beliefs")

    monkeypatch.setattr(FinitePlant, "safe_beliefs", forbidden_power_set)
    captured = frozenset(range(64))
    states = captured | {64, 65}
    controls = ("wait", "r0", "r1", "r2", "r3")
    transitions = {}
    for state in sorted(states):
        for control in controls:
            if state >= 64 or control == "wait":
                target = state
            else:
                target = 64 if state % 4 == int(control[1:]) else 65
            transitions[(state, control)] = frozenset({target})
    plant = FinitePlant(states, states - {65}, frozenset({64}), controls, transitions)
    result = greedy_symbols_via_recovery_cover(
        plant, captured, pending_controls=("wait",), recovery_controls=controls[1:],
        recovery_horizon=1, delay=3,
    )
    assert result is not None and result.symbols == 4
    assert {cell.captured_states for cell in result.cells} == {
        frozenset(range(residue, 64, 4)) for residue in range(4)
    }


@pytest.mark.parametrize("change", [
    {"delay": -1}, {"recovery_horizon": -1}, {"pending_controls": ()},
    {"recovery_controls": ()}, {"pending_controls": ("missing",)},
    {"recovery_controls": ("missing",)},
])
def test_invalid_cover_contracts_are_rejected(change) -> None:
    plant = _random_plant(0)
    kwargs = dict(pending_controls=("p0",), recovery_controls=("r0",),
                  recovery_horizon=1, delay=1)
    kwargs.update(change)
    with pytest.raises(ValueError):
        greedy_symbols_via_recovery_cover(plant, plant.safe_states, **kwargs)


@pytest.mark.parametrize("argument", ["pending_controls", "recovery_controls"])
def test_exact_envelopes_do_not_replace_explicit_empty_alphabets(argument) -> None:
    plant = _random_plant(0)
    kwargs = {argument: ()}
    with pytest.raises(ValueError, match="nonempty"):
        SafeComputeEnvelope(plant, recovery_horizon=1, max_pending_ticks=1, **kwargs)
    with pytest.raises(ValueError, match="nonempty"):
        GuaranteedCompletionEnvelope(
            plant, output_partition=OutputPartition("all", (plant.states,)),
            recovery_horizon=1, max_delay=1, **kwargs,
        )


def test_zero_horizon_and_empty_terminal_set() -> None:
    for terminal in (frozenset({0}), frozenset()):
        plant = FinitePlant(
            frozenset({0}), frozenset({0}), terminal, ("wait",),
            {(0, "wait"): frozenset({0})},
        )
        result = greedy_symbols_via_recovery_cover(
            plant, plant.states, pending_controls=("wait",), recovery_controls=("wait",),
            recovery_horizon=0, delay=2,
        )
        assert (result is None) == (not terminal)
        if result is not None:
            assert result.symbols == 1
            assert result.cells[0].recovery_sequences == ((), (), ())

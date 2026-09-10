from __future__ import annotations

import random

from safe_compute.envelope import SafeComputeEnvelope
from safe_compute.finite import FinitePlant
from safe_compute.oracle import brute_force_budget


def _random_plant(seed: int) -> FinitePlant[int, str]:
    rng = random.Random(seed)
    states = frozenset({0, 1, 2, 3})
    controls = ("a", "b")
    transitions: dict[tuple[int, str], frozenset[int]] = {
        (0, "a"): frozenset({0}),
        (0, "b"): frozenset({0}),
    }
    ordered_states = tuple(sorted(states))
    for state in (1, 2, 3):
        for control in controls:
            size = rng.choice((1, 1, 2))
            transitions[(state, control)] = frozenset(rng.sample(ordered_states, size))
    return FinitePlant(
        states=states,
        safe_states=frozenset({0, 1, 2}),
        recovery_states=frozenset({0}),
        controls=controls,
        transitions=transitions,
    )


def test_layer_recursion_matches_exhaustive_sequences_on_100_random_plants() -> None:
    pending = ("a", "b")
    recovery = ("a", "b")
    recovery_horizon = 2
    cap = 3

    for seed in range(100):
        plant = _random_plant(seed)
        result = SafeComputeEnvelope(
            plant,
            pending_controls=pending,
            recovery_controls=recovery,
            recovery_horizon=recovery_horizon,
            max_pending_ticks=cap,
        ).solve()
        pointwise = SafeComputeEnvelope(
            plant,
            pending_controls=pending,
            recovery_controls=recovery,
            recovery_horizon=recovery_horizon,
            max_pending_ticks=cap,
        ).solve_for(plant.safe_beliefs())

        for later, earlier in zip(result.layers[1:], result.layers[:-1], strict=True):
            assert later <= earlier, f"nonnested layers for seed {seed}"

        for belief in plant.safe_beliefs():
            expected = brute_force_budget(
                plant,
                belief,
                pending_controls=pending,
                recovery_controls=recovery,
                recovery_horizon=recovery_horizon,
                max_pending_ticks=cap,
            )
            assert result.budget(belief) == expected, (seed, belief)
            assert pointwise.budget(belief) == expected, (seed, belief, "pointwise")

        for (belief, ticks), control in result.pending_policy.items():
            assert plant.step(belief, control) in result.layers[ticks - 1]

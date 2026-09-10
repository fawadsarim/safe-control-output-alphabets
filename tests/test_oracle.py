from __future__ import annotations

from safe_compute.envelope import SafeComputeEnvelope
from safe_compute.finite import FinitePlant
from safe_compute.oracle import brute_force_budget


def test_layer_recursion_matches_recursive_oracle(
    nonconstant_plant: FinitePlant[str, str],
) -> None:
    pending = ("hold",)
    recovery = ("recover",)
    horizon = 2
    cap = 4
    result = SafeComputeEnvelope(
        nonconstant_plant,
        pending_controls=pending,
        recovery_controls=recovery,
        recovery_horizon=horizon,
        max_pending_ticks=cap,
    ).solve()

    for belief in nonconstant_plant.safe_beliefs():
        assert result.budget(belief) == brute_force_budget(
            nonconstant_plant,
            belief,
            pending_controls=pending,
            recovery_controls=recovery,
            recovery_horizon=horizon,
            max_pending_ticks=cap,
        )

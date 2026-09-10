from __future__ import annotations

from safe_compute.envelope import SafeComputeEnvelope
from safe_compute.finite import FinitePlant


def test_envelope_is_state_dependent(nonconstant_plant: FinitePlant[str, str]) -> None:
    result = SafeComputeEnvelope(
        nonconstant_plant,
        pending_controls=("hold",),
        recovery_controls=("recover",),
        recovery_horizon=1,
        max_pending_ticks=4,
    ).solve()

    assert result.budget(frozenset({"b"})) == 0
    assert result.budget(frozenset({"a"})) == 1
    assert result.budget(frozenset({"k"})) == 4
    assert result.reaches_cap(frozenset({"k"}))
    assert result.pending_policy[(frozenset({"a"}), 1)] == "hold"


def test_uncertain_belief_can_have_less_slack(nonconstant_plant: FinitePlant[str, str]) -> None:
    result = SafeComputeEnvelope(
        nonconstant_plant,
        pending_controls=("hold",),
        recovery_controls=("recover",),
        recovery_horizon=1,
        max_pending_ticks=2,
    ).solve()

    assert result.budget(frozenset({"a"})) == 1
    assert result.budget(frozenset({"a", "b"})) == 0


def test_pointwise_solver_matches_full_power_set_solver(
    nonconstant_plant: FinitePlant[str, str],
) -> None:
    solver = SafeComputeEnvelope(
        nonconstant_plant,
        pending_controls=("hold",),
        recovery_controls=("recover",),
        recovery_horizon=1,
        max_pending_ticks=4,
    )
    beliefs = nonconstant_plant.safe_beliefs()
    full = solver.solve()
    pointwise = solver.solve_for(beliefs)

    assert {belief: full.budget(belief) for belief in beliefs} == pointwise.budgets

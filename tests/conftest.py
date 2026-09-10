from __future__ import annotations

import pytest

from safe_compute.finite import FinitePlant


@pytest.fixture
def nonconstant_plant() -> FinitePlant[str, str]:
    states = frozenset({"k", "a", "b", "unsafe"})
    controls = ("hold", "recover")
    transitions = {
        ("k", "hold"): frozenset({"k"}),
        ("k", "recover"): frozenset({"k"}),
        ("a", "hold"): frozenset({"b"}),
        ("a", "recover"): frozenset({"k"}),
        ("b", "hold"): frozenset({"unsafe"}),
        ("b", "recover"): frozenset({"k"}),
        ("unsafe", "hold"): frozenset({"unsafe"}),
        ("unsafe", "recover"): frozenset({"unsafe"}),
    }
    return FinitePlant(
        states=states,
        safe_states=frozenset({"k", "a", "b"}),
        recovery_states=frozenset({"k"}),
        controls=controls,
        transitions=transitions,
    )

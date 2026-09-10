"""Run the exact information--completion frontier experiments."""

from __future__ import annotations

import argparse
import json
import random
import tomllib
from itertools import product
from math import ceil, comb
from pathlib import Path

from safe_compute.constructions import (
    build_static_age_gap_plant,
    complete_uniform_conflict_plant,
)
from safe_compute.finite import FinitePlant
from safe_compute.information_liveness import (
    minimum_symbols_for_delay,
    minimum_symbols_via_hypergraph,
)
from safe_compute.oracle import brute_force_minimum_symbols


def run_bounded_exhaustive_audit(config: dict[str, int]) -> dict[str, int]:
    """Cross-check every deterministic plant in one frozen finite model class."""

    safe_count = config["safe_states"]
    control_count = config["controls"]
    if safe_count < 1 or control_count < 1:
        raise ValueError("bounded exhaustive audit requires positive sizes")
    safe = tuple(range(safe_count))
    unsafe = safe_count
    states = frozenset((*safe, unsafe))
    controls = tuple(range(control_count))
    positions = tuple((state, control) for state in safe for control in controls)
    plant_count = 0
    query_count = 0

    for successors in product(states, repeat=len(positions)):
        transitions = {
            position: frozenset({successor})
            for position, successor in zip(positions, successors, strict=True)
        }
        for control in controls:
            transitions[(unsafe, control)] = frozenset({unsafe})
        plant = FinitePlant(
            states=states,
            safe_states=frozenset(safe),
            recovery_states=frozenset({safe[0]}),
            controls=controls,
            transitions=transitions,
        )
        plant_count += 1
        for belief in plant.safe_beliefs():
            for delay in range(config["max_delay"] + 1):
                hypergraph = minimum_symbols_via_hypergraph(
                    plant,
                    belief,
                    pending_controls=controls,
                    recovery_controls=controls,
                    recovery_horizon=config["recovery_horizon"],
                    delay=delay,
                ).minimum_symbols
                partitions = minimum_symbols_for_delay(
                    plant,
                    belief,
                    pending_controls=controls,
                    recovery_controls=controls,
                    recovery_horizon=config["recovery_horizon"],
                    delay=delay,
                ).minimum_symbols
                canonical_maps = brute_force_minimum_symbols(
                    plant,
                    belief,
                    pending_controls=controls,
                    recovery_controls=controls,
                    recovery_horizon=config["recovery_horizon"],
                    delay=delay,
                )
                if not hypergraph == partitions == canonical_maps:
                    raise AssertionError(
                        "bounded exhaustive disagreement "
                        f"plant={plant_count - 1}, belief={belief}, delay={delay}: "
                        f"{hypergraph}, {partitions}, {canonical_maps}"
                    )
                query_count += 1

    return {"plants": plant_count, "queries": query_count}


def build_escalating_plant(branches: int, max_delay: int) -> FinitePlant[str, str]:
    """Create a deterministic family whose information need grows with delay."""

    if branches < 2 or max_delay < 0 or max_delay >= branches:
        raise ValueError("require branches >= 2 and 0 <= max_delay < branches")
    layer_states = tuple(
        f"x_{age}_{branch}"
        for age in range(max_delay + 1)
        for branch in range(branches)
    )
    states = frozenset((*layer_states, "recovered", "unsafe"))
    recovery_controls = tuple(f"recover_{index}" for index in range(branches))
    controls = ("advance", *recovery_controls)
    transitions: dict[tuple[str, str], frozenset[str]] = {}

    for state in sorted(states):
        for control in controls:
            if state in {"recovered", "unsafe"}:
                successor = state
            else:
                _, age_text, branch_text = state.split("_")
                age = int(age_text)
                branch = int(branch_text)
                if control == "advance":
                    successor = f"x_{min(age + 1, max_delay)}_{branch}"
                else:
                    required = f"recover_{min(branch, age)}"
                    successor = "recovered" if control == required else "unsafe"
            transitions[(state, control)] = frozenset({successor})

    return FinitePlant(
        states=states,
        safe_states=states.difference({"unsafe"}),
        recovery_states=frozenset({"recovered"}),
        controls=controls,
        transitions=transitions,
    )


def build_random_plant(
    rng: random.Random,
    *,
    uncertain_states: int,
    recovery_actions: int,
    pending_actions: int,
    pending_unsafe_probability: float,
) -> tuple[FinitePlant[str, str], frozenset[str], tuple[str, ...], tuple[str, ...]]:
    """Create a seeded nondeterministic plant with recoverable singleton modes."""

    uncertain = tuple(f"x_{index}" for index in range(uncertain_states))
    recovery = tuple(f"recover_{index}" for index in range(recovery_actions))
    pending = tuple(f"pending_{index}" for index in range(pending_actions))
    controls = (*pending, *recovery)
    states = frozenset((*uncertain, "recovered", "unsafe"))
    allowed = {
        state: frozenset(
            action
            for action in recovery
            if rng.random() < 0.5
        )
        for state in uncertain
    }
    for state in uncertain:
        if not allowed[state]:
            allowed[state] = frozenset({rng.choice(recovery)})

    transitions: dict[tuple[str, str], frozenset[str]] = {}
    for state in sorted(states):
        for control in controls:
            if state in {"recovered", "unsafe"}:
                successors = frozenset({state})
            elif control in recovery:
                successor = "recovered" if control in allowed[state] else "unsafe"
                successors = frozenset({successor})
            elif rng.random() < pending_unsafe_probability:
                successors = frozenset({"unsafe"})
            else:
                successors = frozenset(
                    {rng.choice(uncertain), rng.choice(uncertain)}
                )
            transitions[(state, control)] = successors

    plant = FinitePlant(
        states=states,
        safe_states=states.difference({"unsafe"}),
        recovery_states=frozenset({"recovered"}),
        controls=controls,
        transitions=transitions,
    )
    return plant, frozenset(uncertain), pending, recovery


def run(config_path: Path) -> dict[str, object]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    run_config = config["run"]
    exhaustive_config = config["exhaustive"]
    structured_config = config["structured"]
    random_config = config["random"]

    bounded_exhaustive = run_bounded_exhaustive_audit(exhaustive_config)

    branches = structured_config["branches"]
    max_structured_delay = structured_config["max_delay"]
    escalating = build_escalating_plant(branches, max_structured_delay)
    escalating_belief = frozenset(f"x_0_{branch}" for branch in range(branches))
    escalating_recovery = tuple(f"recover_{index}" for index in range(branches))
    structured_frontier = []
    for delay in range(max_structured_delay + 1):
        result = minimum_symbols_via_hypergraph(
            escalating,
            escalating_belief,
            pending_controls=("advance",),
            recovery_controls=escalating_recovery,
            recovery_horizon=1,
            delay=delay,
        )
        structured_frontier.append(
            {"delay": delay, "minimum_symbols": result.minimum_symbols}
        )
    expected = list(range(1, max_structured_delay + 2))
    observed = [row["minimum_symbols"] for row in structured_frontier]
    if observed != expected:
        raise AssertionError(f"escalating frontier mismatch: {observed} != {expected}")

    uniform_rank = structured_config["uniform_rank"]
    higher_order_gap = []
    for size in range(uniform_rank, structured_config["uniform_max_states"] + 1):
        plant, belief, recovery = complete_uniform_conflict_plant(
            size, rank=uniform_rank
        )
        result = minimum_symbols_via_hypergraph(
            plant,
            belief,
            pending_controls=recovery,
            recovery_controls=recovery,
            recovery_horizon=1,
            delay=0,
        )
        expected_symbols = ceil(size / (uniform_rank - 1))
        if result.minimum_symbols != expected_symbols:
            raise AssertionError("complete-uniform family missed its closed form")
        if not result.conflict_hyperedges or any(
            len(edge) != uniform_rank for edge in result.conflict_hyperedges
        ):
            raise AssertionError("complete-uniform family has an incorrect edge rank")
        higher_order_gap.append(
            {
                "uncertain_states": size,
                "rank": uniform_rank,
                "hyperedges": comb(size, uniform_rank),
                "pairwise_graph_estimate": 1,
                "minimum_symbols": expected_symbols,
            }
        )

    static_age_gap = []
    for size in range(2, structured_config["static_gap_max_states"] + 1):
        plant, belief, pending, recovery, matchings = build_static_age_gap_plant(size)
        result = minimum_symbols_via_hypergraph(
            plant,
            belief,
            pending_controls=pending,
            recovery_controls=recovery,
            recovery_horizon=1,
            delay=len(matchings) - 1,
        )
        if result.minimum_symbols != size:
            raise AssertionError("static-age family did not attain n static symbols")
        static_age_gap.append(
            {
                "uncertain_states": size,
                "completion_ages": len(matchings),
                "static_symbols": size,
                "known_age_symbols": 2,
            }
        )

    random_rng = random.Random(random_config["seed"])
    crosschecks = 0
    max_random_delay = random_config["max_delay"]
    for instance in range(run_config["crosscheck_instances"]):
        plant, belief, pending, recovery = build_random_plant(
            random_rng,
            uncertain_states=random_config["uncertain_states"],
            recovery_actions=random_config["recovery_actions"],
            pending_actions=random_config["pending_actions"],
            pending_unsafe_probability=random_config["pending_unsafe_probability"],
        )
        previous = 0
        for delay in range(max_random_delay + 1):
            result = minimum_symbols_via_hypergraph(
                plant,
                belief,
                pending_controls=pending,
                recovery_controls=recovery,
                recovery_horizon=random_config["recovery_horizon"],
                delay=delay,
            )
            symbols = result.minimum_symbols
            monotone_value = len(plant.safe_states) + 1 if symbols is None else symbols
            if monotone_value < previous:
                raise AssertionError("minimum information decreased with a later deadline")
            previous = monotone_value

            partition_result = minimum_symbols_for_delay(
                plant,
                belief,
                pending_controls=pending,
                recovery_controls=recovery,
                recovery_horizon=random_config["recovery_horizon"],
                delay=delay,
            ).minimum_symbols
            brute_result = brute_force_minimum_symbols(
                plant,
                belief,
                pending_controls=pending,
                recovery_controls=recovery,
                recovery_horizon=random_config["recovery_horizon"],
                delay=delay,
            )
            if not symbols == partition_result == brute_result:
                raise AssertionError(
                    "three-way frontier disagreement "
                    f"at instance={instance}, delay={delay}: "
                    f"{symbols}, {partition_result}, {brute_result}"
                )
            crosschecks += 1

    return {
        "status": run_config["status"],
        "configuration": config,
        "bounded_exhaustive": bounded_exhaustive,
        "structured_frontier": structured_frontier,
        "higher_order_gap": higher_order_gap,
        "static_age_gap": static_age_gap,
        "three_way_crosschecks": crosschecks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pc_information_completion.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/derived/information_completion_frontier.json"),
    )
    args = parser.parse_args()
    payload = run(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "bounded_exhaustive": payload["bounded_exhaustive"],
                "structured_frontier": payload["structured_frontier"],
                "higher_order_gap": payload["higher_order_gap"],
                "static_age_gap": payload["static_age_gap"],
                "three_way_crosschecks": payload["three_way_crosschecks"],
            }
        )
    )


if __name__ == "__main__":
    main()

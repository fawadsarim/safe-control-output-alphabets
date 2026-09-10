"""Deterministic witness families for the manuscript's structural results."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations, product

from .finite import FinitePlant


def _subsets(vertices: tuple[int, ...]) -> tuple[frozenset[int], ...]:
    return tuple(
        frozenset(subset)
        for size in range(len(vertices) + 1)
        for subset in combinations(vertices, size)
    )


def realize_conflict_clutter(
    vertex_count: int,
    hyperedges: Iterable[frozenset[int]],
) -> tuple[FinitePlant[str, str], frozenset[str], tuple[str, ...]]:
    """Realize a finite clutter as the exact one-step recovery conflicts.

    Each recovery control succeeds on one maximal independent set.  Consequently,
    a captured subset is recoverable exactly when it contains no supplied edge.
    """

    if vertex_count < 1:
        raise ValueError("vertex_count must be positive")
    vertices = tuple(range(vertex_count))
    vertex_set = frozenset(vertices)
    edges = tuple(sorted(set(hyperedges), key=lambda edge: (len(edge), tuple(edge))))
    if any(not edge or not edge <= vertex_set for edge in edges):
        raise ValueError("hyperedges must be nonempty subsets of the vertices")
    if any(left < right for left in edges for right in edges):
        raise ValueError("hyperedges must form an inclusion antichain")

    independent = tuple(
        subset
        for subset in _subsets(vertices)
        if not any(edge <= subset for edge in edges)
    )
    maximal = tuple(
        subset
        for subset in independent
        if all(
            subset.union({vertex}) not in independent
            for vertex in vertex_set.difference(subset)
        )
    )
    if not maximal:
        raise AssertionError("the empty set is independent and extends to a maximal set")

    uncertain = tuple(f"x_{vertex}" for vertex in vertices)
    recovery_controls = tuple(f"recover_{index}" for index in range(len(maximal)))
    states = frozenset((*uncertain, "recovered", "unsafe"))
    transitions: dict[tuple[str, str], frozenset[str]] = {}
    for state in states:
        for index, control in enumerate(recovery_controls):
            if state in {"recovered", "unsafe"}:
                successor = state
            else:
                vertex = int(state.removeprefix("x_"))
                successor = "recovered" if vertex in maximal[index] else "unsafe"
            transitions[(state, control)] = frozenset({successor})

    plant = FinitePlant(
        states=states,
        safe_states=states.difference({"unsafe"}),
        recovery_states=frozenset({"recovered"}),
        controls=recovery_controls,
        transitions=transitions,
    )
    return plant, frozenset(uncertain), recovery_controls


def complete_uniform_conflict_plant(
    vertex_count: int,
    rank: int,
) -> tuple[FinitePlant[str, str], frozenset[str], tuple[str, ...]]:
    """Realize the complete rank-uniform conflict hypergraph."""

    if rank < 2 or rank > vertex_count:
        raise ValueError("require 2 <= rank <= vertex_count")
    edges = tuple(
        frozenset(edge) for edge in combinations(range(vertex_count), rank)
    )
    return realize_conflict_clutter(vertex_count, edges)


def complete_graph_matching_decomposition(
    vertex_count: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Partition the edges of a complete graph into round-robin matchings."""

    if vertex_count < 2:
        raise ValueError("vertex_count must be at least two")
    dummy = vertex_count if vertex_count % 2 else None
    participants = list(range(vertex_count + (dummy is not None)))
    matchings: list[tuple[tuple[int, int], ...]] = []
    for _ in range(len(participants) - 1):
        pairs = []
        for index in range(len(participants) // 2):
            left = participants[index]
            right = participants[-index - 1]
            if dummy not in {left, right}:
                pairs.append(tuple(sorted((left, right))))
        matchings.append(tuple(sorted(pairs)))
        participants = [participants[0], participants[-1], *participants[1:-1]]

    expected = {
        tuple(pair) for pair in combinations(range(vertex_count), 2)
    }
    observed = {pair for matching in matchings for pair in matching}
    if observed != expected or sum(map(len, matchings)) != len(expected):
        raise AssertionError("round-robin construction must partition every edge once")
    return tuple(matchings)


def build_static_age_gap_plant(
    vertex_count: int,
) -> tuple[
    FinitePlant[str, str],
    frozenset[str],
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[tuple[int, int], ...], ...],
]:
    """Build the unbounded static-versus-known-age output-gap family."""

    matchings = complete_graph_matching_decomposition(vertex_count)
    layer_states = tuple(
        f"x_{age}_{vertex}"
        for age in range(len(matchings))
        for vertex in range(vertex_count)
    )
    states = frozenset((*layer_states, "recovered", "unsafe"))

    success_sets: dict[str, frozenset[str]] = {}
    for age, matching in enumerate(matchings):
        matched = {vertex for edge in matching for vertex in edge}
        unmatched = set(range(vertex_count)).difference(matched)
        for choice_index, choices in enumerate(product((0, 1), repeat=len(matching))):
            independent = unmatched.union(
                edge[choice] for edge, choice in zip(matching, choices, strict=True)
            )
            control = f"recover_{age}_{choice_index}"
            success_sets[control] = frozenset(
                f"x_{age}_{vertex}" for vertex in independent
            )

    recovery_controls = tuple(success_sets)
    controls = ("advance", *recovery_controls)
    transitions: dict[tuple[str, str], frozenset[str]] = {}
    for state in states:
        for control in controls:
            if state in {"recovered", "unsafe"}:
                successor = state
            elif control == "advance":
                _, age_text, vertex_text = state.split("_")
                age = int(age_text)
                successor = f"x_{min(age + 1, len(matchings) - 1)}_{vertex_text}"
            else:
                successor = "recovered" if state in success_sets[control] else "unsafe"
            transitions[(state, control)] = frozenset({successor})

    plant = FinitePlant(
        states=states,
        safe_states=states.difference({"unsafe"}),
        recovery_states=frozenset({"recovered"}),
        controls=controls,
        transitions=transitions,
    )
    belief = frozenset(f"x_0_{vertex}" for vertex in range(vertex_count))
    return plant, belief, ("advance",), recovery_controls, matchings


def build_product_age_gap_plant(
    age_alphabets: tuple[int, ...],
) -> tuple[FinitePlant[str, str], frozenset[str], tuple[str, ...], tuple[str, ...]]:
    """Attain the product bound for a static captured-state output alphabet.

    A captured vertex is a coordinate vector, with one coordinate per arrival
    age. At layer j only its j-th coordinate determines the successful recovery
    input. For N=product(age_alphabets) there are len(age_alphabets)*N+2 states
    and 1+sum(age_alphabets) controls; no maximal-independent-set enumeration
    is used. Recovery horizon is one and the sole pending input advances layers.
    """

    if not age_alphabets or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1
        for size in age_alphabets
    ):
        raise ValueError("age_alphabets must contain positive integers")
    coordinates = tuple(product(*(range(size) for size in age_alphabets)))
    layers = len(age_alphabets)
    metadata = {
        f"recover_{age}_{value}": (age, value)
        for age, size in enumerate(age_alphabets) for value in range(size)
    }
    recovery = tuple(metadata)
    controls = ("advance", *recovery)
    states = frozenset(
        [f"x_{age}_{vertex}" for age in range(layers) for vertex in range(len(coordinates))]
        + ["recovered", "unsafe"]
    )
    transitions = {}
    for state in states:
        for control in controls:
            if state in {"recovered", "unsafe"}:
                target = state
            else:
                _, age_text, vertex_text = state.split("_")
                age, vertex = int(age_text), int(vertex_text)
                if control == "advance":
                    target = f"x_{min(age + 1, layers - 1)}_{vertex}"
                else:
                    wanted_age, value = metadata[control]
                    succeeds = age == wanted_age and coordinates[vertex][age] == value
                    target = "recovered" if succeeds else "unsafe"
            transitions[(state, control)] = frozenset({target})
    plant = FinitePlant(
        states, states - {"unsafe"}, frozenset({"recovered"}), controls, transitions,
    )
    belief = frozenset(f"x_0_{vertex}" for vertex in range(len(coordinates)))
    return plant, belief, ("advance",), recovery

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import ceil

import pytest

from safe_compute.constructions import (
    build_static_age_gap_plant,
    complete_graph_matching_decomposition,
    complete_uniform_conflict_plant,
    realize_conflict_clutter,
)
from safe_compute.information_liveness import (
    greedy_feasible_cover,
    maximal_feasible_color_classes,
    minimum_hypergraph_colors,
    minimum_symbols_via_hypergraph,
)


def _nonempty_subsets(size: int) -> tuple[frozenset[int], ...]:
    return tuple(
        frozenset(subset)
        for cardinality in range(1, size + 1)
        for subset in combinations(range(size), cardinality)
    )


def _all_clutters(size: int):
    subsets = _nonempty_subsets(size)
    for mask in range(1 << len(subsets)):
        family = tuple(
            subset for index, subset in enumerate(subsets) if mask & (1 << index)
        )
        if not any(left < right for left in family for right in family):
            yield family


def _minimum_cover_size(
    vertices: frozenset[int], edges: tuple[frozenset[int], ...]
) -> int | None:
    classes = maximal_feasible_color_classes(vertices, edges)
    for count in range(1, len(vertices) + 1):
        if any(
            frozenset().union(*selected) == vertices
            for selected in combinations(classes, count)
        ):
            return count
    return None


def test_every_three_vertex_clutter_is_realized_exactly() -> None:
    count = 0
    for clutter in _all_clutters(3):
        plant, belief, recovery = realize_conflict_clutter(3, clutter)
        expected = {
            frozenset(f"x_{vertex}" for vertex in edge) for edge in clutter
        }
        result = minimum_symbols_via_hypergraph(
            plant,
            belief,
            pending_controls=recovery,
            recovery_controls=recovery,
            recovery_horizon=1,
            delay=0,
        )
        if any(len(edge) == 1 for edge in clutter):
            assert result.minimum_symbols is None
        else:
            assert set(result.conflict_hyperedges or ()) == expected

        for subset in _nonempty_subsets(3):
            states = frozenset(f"x_{vertex}" for vertex in subset)
            directly_recoverable = any(
                plant.step(states, control) <= plant.recovery_states
                for control in recovery
            )
            independent = not any(edge <= subset for edge in clutter)
            assert directly_recoverable == independent
        count += 1

    assert count == 19


def test_cover_dual_and_greedy_bound_on_every_four_vertex_clutter() -> None:
    vertices = frozenset(range(4))
    harmonic = sum(Fraction(1, index) for index in range(1, len(vertices) + 1))
    count = 0
    for clutter in _all_clutters(4):
        exact_coloring = minimum_hypergraph_colors(vertices, clutter)
        assert _minimum_cover_size(vertices, clutter) == exact_coloring
        greedy = greedy_feasible_cover(vertices, clutter)
        if exact_coloring is None:
            assert greedy is None
        else:
            assert greedy is not None
            assert frozenset().union(*greedy) == vertices
            assert sum(map(len, greedy)) == len(vertices)
            assert all(not any(edge <= cell for edge in clutter) for cell in greedy)
            assert len(greedy) <= harmonic * exact_coloring
        count += 1
    assert count == 167


@pytest.mark.parametrize("size", range(3, 9))
def test_complete_ternary_conflicts_have_unbounded_pairwise_gap(size: int) -> None:
    plant, belief, recovery = complete_uniform_conflict_plant(size, rank=3)
    result = minimum_symbols_via_hypergraph(
        plant,
        belief,
        pending_controls=recovery,
        recovery_controls=recovery,
        recovery_horizon=1,
        delay=0,
    )
    assert result.minimum_symbols == ceil(size / 2)
    assert len(result.conflict_hyperedges or ()) == len(tuple(combinations(range(size), 3)))
    assert all(len(edge) == 3 for edge in result.conflict_hyperedges or ())


@pytest.mark.parametrize("size", range(2, 8))
def test_complete_graph_matching_decomposition_covers_each_pair_once(size: int) -> None:
    matchings = complete_graph_matching_decomposition(size)
    flattened = [pair for matching in matchings for pair in matching]
    expected_matchings = size - 1 if size % 2 == 0 else size
    assert len(matchings) == expected_matchings
    assert len(flattened) == len(set(flattened)) == size * (size - 1) // 2
    assert set(flattened) == set(combinations(range(size), 2))
    assert all(
        len({vertex for pair in matching for vertex in pair}) == 2 * len(matching)
        for matching in matchings
    )


@pytest.mark.parametrize("size", range(2, 8))
def test_static_partition_gap_is_n_versus_two(size: int) -> None:
    plant, belief, pending, recovery, matchings = build_static_age_gap_plant(size)
    result = minimum_symbols_via_hypergraph(
        plant,
        belief,
        pending_controls=pending,
        recovery_controls=recovery,
        recovery_horizon=1,
        delay=len(matchings) - 1,
    )
    expected_edges = {
        frozenset({f"x_0_{left}", f"x_0_{right}"})
        for left, right in combinations(range(size), 2)
    }
    assert result.minimum_symbols == size
    assert set(result.conflict_hyperedges or ()) == expected_edges

    for age, matching in enumerate(matchings):
        matched = {vertex for edge in matching for vertex in edge}
        color_zero = set(range(size)).difference(matched)
        color_one: set[int] = set()
        for left, right in matching:
            color_zero.add(left)
            color_one.add(right)

        full_image = frozenset(f"x_{age}_{vertex}" for vertex in range(size))
        assert not any(
            plant.step(full_image, control) <= plant.recovery_states
            for control in recovery
        )
        for color_class in (color_zero, color_one):
            posterior = frozenset(f"x_{age}_{vertex}" for vertex in color_class)
            assert posterior
            assert any(
                plant.step(posterior, control) <= plant.recovery_states
                for control in recovery
            )

"""Static output partitions for finite information-state models."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Generic, TypeVar

StateT = TypeVar("StateT", bound=Hashable)


@dataclass(frozen=True)
class OutputPartition(Generic[StateT]):
    """A named, disjoint partition of a finite state space."""

    name: str
    cells: tuple[frozenset[StateT], ...]

    def validate(self, universe: frozenset[StateT]) -> None:
        if not self.name:
            raise ValueError("partition must be named")
        if not self.cells or any(not cell for cell in self.cells):
            raise ValueError("partition cells must be nonempty")
        covered: set[StateT] = set()
        for cell in self.cells:
            if not cell <= universe:
                raise ValueError("partition contains a state outside the plant")
            if covered.intersection(cell):
                raise ValueError("partition cells must be disjoint")
            covered.update(cell)
        if covered != set(universe):
            raise ValueError("partition cells must cover the plant state space")

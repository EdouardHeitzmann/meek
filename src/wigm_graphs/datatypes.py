from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional

import numpy as np
from numpy.typing import NDArray

try:
    from ..election_graphs.datatypes import (
        EdgeAction,
        EdgeRef,
        EdgeStatus,
        ElectionEdge,
        ElectionState,
        ElectionStatus,
        VertexRef,
    )
except ImportError:
    from election_graphs.datatypes import (
        EdgeAction,
        EdgeRef,
        EdgeStatus,
        ElectionEdge,
        ElectionState,
        ElectionStatus,
        VertexRef,
    )


@dataclass(frozen=True, slots=True)
class StateKey:
    seated_at: tuple[Optional[EdgeRef], ...]
    hopefuls: frozenset[int]


@dataclass(frozen=True, slots=True)
class BlackBoxStateKey:
    seated_at: tuple[Optional[EdgeRef], ...]
    hopefuls: frozenset[int]
    seed_id: int


@dataclass(slots=True)
class WIGMWeightScenarios:
    values: NDArray[np.float64]
    labels: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("Weight scenarios must have shape (scenarios, ballots).")
        if self.values.shape[0] != len(self.labels):
            raise ValueError("Weight scenario labels must match the scenario count.")


@dataclass(slots=True)
class PostSeedTallies:
    certain: NDArray[np.float64]
    uncertain: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.certain.ndim != 2:
            raise ValueError("Certain tallies must have shape (scenarios, candidates).")
        if self.uncertain.ndim != 1:
            raise ValueError("Uncertain tallies must be one-dimensional.")
        if self.certain.shape[1] != len(self.uncertain):
            raise ValueError("Certain and uncertain tally candidate counts must match.")

    @property
    def lower(self) -> NDArray[np.float64]:
        return self.certain.min(axis=0)

    @property
    def upper(self) -> NDArray[np.float64]:
        return self.certain.max(axis=0) + self.uncertain


@dataclass(frozen=True, slots=True)
class BlackBoxElectionData:
    candidate: int
    very_strong_candidates: frozenset[int]
    strong_candidates: frozenset[int]
    weak_candidates: frozenset[int]
    raw_maximum_surplus: float
    maximum_surplus: float
    maximum_transfer_value: float


@dataclass(slots=True)
class WIGMRuntimeCache:
    """
    Expansion-only cache. This should not be considered durable graph data.
    """
    bool_ballot_matrix: NDArray[np.bool_]
    pos_vec: NDArray[np.integer]
    fpv_vec: NDArray[np.integer] | None = None


__all__ = [
    "EdgeAction",
    "EdgeRef",
    "EdgeStatus",
    "ElectionEdge",
    "ElectionState",
    "ElectionStatus",
    "BlackBoxStateKey",
    "BlackBoxElectionData",
    "PostSeedTallies",
    "StateKey",
    "VertexRef",
    "WIGMRuntimeCache",
    "WIGMWeightScenarios",
]

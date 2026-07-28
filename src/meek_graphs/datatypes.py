from __future__ import annotations

from dataclasses import dataclass

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
class MeekStateKey:
    elected_candidates: tuple[int, ...]
    hopefuls: frozenset[int]


@dataclass(slots=True)
class MeekRuntimeCache:
    """
    Expansion-only cache. This should not be considered durable graph data.
    """
    bool_ballot_matrix: NDArray[np.bool_]
    pos_vec: NDArray[np.integer]
    fpv_vec: NDArray[np.integer]
    winner_comb_vec: NDArray[np.integer]
    winner_bitstring_vec: NDArray[np.integer]
    keep_factor_seed: NDArray[np.float64]


@dataclass(slots=True)
class KeepFactorCalibrationCache:
    support_comb_ids: NDArray[np.integer]
    ballot_to_support_row: NDArray[np.integer]
    sliced_winner_matrix: NDArray[np.integer]
    valid_mask: NDArray[np.bool_]
    permutation_lengths: NDArray[np.integer]
    support_total_mass: NDArray[np.float64]
    support_nonexhausted_mass: NDArray[np.float64]
    exhausted_mask: NDArray[np.bool_]
    fpv_vec: NDArray[np.integer]
    initial_wt_vec: NDArray[np.float64]


@dataclass(slots=True)
class MeekExpansionContext:
    quota: float
    iterations_used: int


__all__ = [
    "EdgeAction",
    "EdgeRef",
    "EdgeStatus",
    "ElectionEdge",
    "ElectionState",
    "ElectionStatus",
    "KeepFactorCalibrationCache",
    "MeekExpansionContext",
    "MeekRuntimeCache",
    "MeekStateKey",
    "VertexRef",
]

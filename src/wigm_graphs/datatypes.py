from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional
import numpy as np
from numpy.typing import NDArray
from enum import IntEnum, IntFlag

# ------------------------------ VERTICES ------------------------------

@dataclass(frozen=True, slots=True)
class VertexRef:
    """
    Stable reference to a vertex in the layered DAG.
    """
    layer: int
    local_id: int

@dataclass(frozen=True, slots=True)
class StateKey:
    seated_at: tuple[Optional[EdgeRef], ...]
    hopefuls: frozenset[int]

class ElectionStatus(IntEnum):
    UNEXPANDED = 0
    EXPANDED = 1
    TERMINAL = 2

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def description(self) -> str:
        return {
            ElectionStatus.UNEXPANDED: "created but not yet expanded",
            ElectionStatus.EXPANDED: "outgoing edges have been generated",
            ElectionStatus.TERMINAL: "leaf/final state; no outgoing edges",
        }[self]
    
@dataclass(slots=True, eq=False)
class ElectionState:
    ref: VertexRef
    key: StateKey
    degree: int

    tallies: NDArray[np.float64] | None = None
    status: ElectionStatus = ElectionStatus.UNEXPANDED

    color: int | None = None

    path_multiplicity: int = 0
    tightest_margin: float | None = None

    def __post_init__(self) -> None:
        if self.color is None:
            self.color = self.degree


# ------------------------------ EDGES ------------------------------


@dataclass(frozen=True, slots=True)
class EdgeRef:
    layer: int      # edge layer: k means layer k -> k+1
    local_id: int


class EdgeStatus(IntEnum):
    DEFAULT = 0
    NORMAL = 1
    CANONICAL = 2


class EdgeAction(IntEnum):
    ELECT = 0
    ELIMINATE = 1
    FORCE_ELECT = 2

@dataclass(slots=True)
class ElectionEdge:
    ref: EdgeRef
    src: VertexRef
    dst: VertexRef

    action: EdgeAction          # ELECT or ELIMINATE
    candidate: int
    status: EdgeStatus = EdgeStatus.DEFAULT

    margin: float | None = None

    # Meaningful mainly for ELECT edges
    transfer_value: float | None = None
    wt_vec: NDArray[np.float64] | None = None


@dataclass(slots=True)
class RuntimeCache:
    """
    Expansion-only cache. This should not be considered durable graph data.
    """
    bool_ballot_matrix: NDArray[np.bool_]
    pos_vec: NDArray[np.integer]
    fpv_vec: NDArray[np.integer] | None = None
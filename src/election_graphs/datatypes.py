from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ------------------------------ VERTICES ------------------------------

@dataclass(frozen=True, slots=True)
class VertexRef:
    """
    Stable reference to a vertex in the layered DAG.
    """
    layer: int
    local_id: int


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
    key: Any
    degree: int

    tallies: NDArray[np.float64] | None = None
    keep_factors: NDArray[np.float64] | None = None
    quota: float | None = None
    status: ElectionStatus = ElectionStatus.UNEXPANDED

    color: int | None = None

    path_multiplicity: int = 0
    tightest_margin: float | None = None
    next_margin: float | None = None

    def __post_init__(self) -> None:
        if self.color is None:
            self.color = self.degree


# ------------------------------ EDGES ------------------------------

@dataclass(frozen=True, slots=True)
class EdgeRef:
    layer: int
    local_id: int


class EdgeStatus(IntEnum):
    NOT_FACTUAL = -1
    DEFAULT = 0
    NORMAL = 1
    CANONICAL = 2


class EdgeAction(IntEnum):
    ELECT = 0
    ELIMINATE = 1
    FORCE_ELECT = 2
    SIMULTANEOUS_ELECT = 3
    BLACK_BOX_ELECT = 4

    @property
    def is_election(self) -> bool:
        return self in (
            EdgeAction.ELECT,
            EdgeAction.FORCE_ELECT,
            EdgeAction.SIMULTANEOUS_ELECT,
            EdgeAction.BLACK_BOX_ELECT,
        )

    @property
    def is_seating(self) -> bool:
        return self.is_election


@dataclass(slots=True)
class ElectionEdge:
    ref: EdgeRef
    src: VertexRef
    dst: VertexRef

    action: EdgeAction
    candidate: int
    status: EdgeStatus = EdgeStatus.DEFAULT

    margin: float | None = None

    # Meaningful mainly for WIGM ELECT edges.
    transfer_value: float | None = None
    wt_vec: NDArray[np.float64] | None = None
    fpv_vec: NDArray[np.integer] | None = None

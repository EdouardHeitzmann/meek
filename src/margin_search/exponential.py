from __future__ import annotations

import math
from typing import Any, Type

try:
    from ..election_graphs import AbstractGraphConstructor, IncoherentLeafError
    from ..meek_graphs import MeekGraphConstructor
except ImportError:
    from election_graphs import AbstractGraphConstructor, IncoherentLeafError
    from meek_graphs import MeekGraphConstructor


def exponential_search(
    profile: Any,
    *,
    m: int,
    starting_lam: float = 10.0,
    memory_lite: bool = True,
    constructor_cls: Type[AbstractGraphConstructor] = MeekGraphConstructor,
    **constructor_kwargs: Any,
) -> AbstractGraphConstructor:
    """
    Naive exponential LAM search.

    Rebuilds the graph from scratch at each doubled LAM until coherence fails.
    Then restricts the first incoherent graph to the smallest tightest_margin
    among terminal vertices whose winner set differs from the inferred recorded
    winner set, checks coherence again, and returns that restricted constructor.
    """
    if starting_lam < 0:
        raise ValueError("starting_lam must be non-negative.")

    lam = float(starting_lam)

    while True:
        constructor = constructor_cls(
            profile,
            m=m,
            LAM=lam,
            memory_lite=memory_lite,
            **constructor_kwargs,
        )
        constructor.build()
        constructor.add_natural_edges()
        constructor.assign_tightest_margins()

        if not constructor.coherence_check():
            break

        if lam == 0:
            lam = 1.0
        else:
            lam *= 2.0

    recorded_winner_set = _infer_recorded_winner_set(constructor)
    restriction_lam = _smallest_incoherent_terminal_margin(
        constructor,
        recorded_winner_set,
    )

    constructor.restrict_margin(restriction_lam)
    constructor.coherence_check()

    return constructor


def heap_based_search(
    profile: Any,
    *,
    m: int,
    memory_lite: bool = True,
    constructor_cls: Type[AbstractGraphConstructor] = MeekGraphConstructor,
    **constructor_kwargs: Any,
) -> AbstractGraphConstructor:
    """
    Naive next-margin expansion search.

    Starts at LAM 1, repeatedly expands to the smallest stored next_margin in
    the current graph, and stops when coherence fails. The first incoherent
    graph is then restricted to floor(smallest incoherent terminal margin).
    """
    constructor = constructor_cls(
        profile,
        m=m,
        LAM=1.0,
        memory_lite=memory_lite,
        **constructor_kwargs,
    )
    constructor.build()
    constructor.add_natural_edges()
    constructor.assign_tightest_margins()

    while constructor.coherence_check():
        next_lam = _smallest_next_margin(constructor)
        constructor.expand_margin(next_lam)
        constructor.add_natural_edges()
        constructor.assign_tightest_margins()

    recorded_winner_set = _infer_recorded_winner_set(constructor)
    incoherent_lam = _smallest_incoherent_terminal_margin(
        constructor,
        recorded_winner_set,
    )
    restriction_lam = math.floor(incoherent_lam)

    constructor.restrict_margin(restriction_lam)
    constructor.coherence_check()

    return constructor


def hybrid_search(
    profile: Any,
    *,
    m: int,
    starting_lam: float = 1.0,
    memory_lite: bool = True,
    constructor_cls: Type[AbstractGraphConstructor] = MeekGraphConstructor,
    **constructor_kwargs: Any,
) -> AbstractGraphConstructor:
    """
    Expansion-based hybrid LAM search.

    Starts with trip_when_incoherent enabled, doubles LAM in place with
    expand_margin until construction discovers the first incoherent terminal
    leaf, then restricts back to the last coherent LAM and finishes with the
    naive next-margin heap-style refinement.
    """
    if starting_lam < 0:
        raise ValueError("starting_lam must be non-negative.")

    constructor_kwargs.pop("trip_when_incoherent", None)
    last_coherent_lam: float | None = None

    constructor = constructor_cls(
        profile,
        m=m,
        LAM=float(starting_lam),
        memory_lite=memory_lite,
        trip_when_incoherent=True,
        **constructor_kwargs,
    )

    try:
        constructor.build()

        while True:
            next_lam = 1.0 if constructor.LAM == 0 else 2.0 * constructor.LAM
            last_coherent_lam = float(constructor.LAM)
            constructor.expand_margin(next_lam)
    except IncoherentLeafError:
        constructor.trip_when_incoherent = False
        constructor._pending_incoherent_leaf_error = None
        constructor._terminal_winner_set_tripwire = None

        if last_coherent_lam is not None:
            constructor.restrict_margin(last_coherent_lam)
        else:
            constructor = constructor_cls(
                profile,
                m=m,
                LAM=0.0,
                memory_lite=memory_lite,
                trip_when_incoherent=False,
                **constructor_kwargs,
            )
            constructor.build()

    return _heap_refinement_from_constructor(constructor)


def _heap_refinement_from_constructor(
    constructor: AbstractGraphConstructor,
) -> AbstractGraphConstructor:
    constructor.trip_when_incoherent = False
    constructor.add_natural_edges()
    constructor.assign_tightest_margins()

    while constructor.coherence_check():
        next_lam = _smallest_next_margin(constructor)
        constructor.expand_margin(next_lam)
        constructor.add_natural_edges()
        constructor.assign_tightest_margins()

    recorded_winner_set = _infer_recorded_winner_set(constructor)
    incoherent_lam = _smallest_incoherent_terminal_margin(
        constructor,
        recorded_winner_set,
    )
    restriction_lam = math.floor(incoherent_lam)

    constructor.restrict_margin(restriction_lam)
    constructor.coherence_check()

    return constructor


def _smallest_next_margin(
    constructor: AbstractGraphConstructor,
) -> float:
    margins = [
        vertex.next_margin
        for layer in constructor.layers
        for vertex in layer
        if vertex.next_margin is not None
        and vertex.next_margin > constructor.LAM
    ]

    if not margins:
        raise ValueError("No next_margin available before coherence failed.")

    return float(min(margins))


def _infer_recorded_winner_set(
    constructor: AbstractGraphConstructor,
) -> frozenset[int]:
    best_winner_set = None
    best_margin = float("inf")

    for winner_set, refs in constructor.terminal_vertices_by_winner_set.items():
        margins = [
            constructor.vertex(ref).tightest_margin
            for ref in refs
            if constructor.vertex(ref).tightest_margin is not None
        ]

        if not margins:
            continue

        winner_set_margin = min(margins)
        if winner_set_margin < best_margin:
            best_margin = winner_set_margin
            best_winner_set = winner_set

    if best_winner_set is None:
        raise ValueError("Could not infer recorded winner set.")

    return best_winner_set


def _smallest_incoherent_terminal_margin(
    constructor: AbstractGraphConstructor,
    recorded_winner_set: frozenset[int],
) -> float:
    best_margin = float("inf")

    for winner_set, refs in constructor.terminal_vertices_by_winner_set.items():
        if winner_set == recorded_winner_set:
            continue

        for ref in refs:
            margin = constructor.vertex(ref).tightest_margin
            if margin is not None and margin < best_margin:
                best_margin = margin

    if best_margin == float("inf"):
        raise ValueError("No incoherent terminal vertex found.")

    return float(best_margin)

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pytest

from src.election_graphs.datatypes import EdgeAction
from src.wigm_graphs.graph_wigm import WIGMGraphConstructor


@dataclass
class MatrixProfile:
    candidates: list[str]
    ballot_matrix: np.ndarray
    wt_vec: np.ndarray

    @property
    def total_ballot_wt(self) -> float:
        return float(self.wt_vec.sum())


def profile_from_first_preferences(tallies: list[float]) -> MatrixProfile:
    n_candidates = len(tallies)
    return MatrixProfile(
        candidates=[f"C{i}" for i in range(n_candidates)],
        ballot_matrix=np.array(
            [[candidate] for candidate in range(n_candidates)],
            dtype=np.int8,
        ),
        wt_vec=np.array(tallies, dtype=np.float64),
    )


def expand_root(constructor: WIGMGraphConstructor):
    root = constructor.initialize_root()
    constructor._expand_vertex(root)
    return root


def election_groups_from_root(
    constructor: WIGMGraphConstructor,
    root,
) -> list[tuple[int, ...]]:
    by_destination = defaultdict(list)
    for edge in constructor.outgoing_edges(root):
        if EdgeAction(edge.action).is_election:
            by_destination[edge.dst].append(edge.candidate)

    return sorted(tuple(sorted(group)) for group in by_destination.values())


def election_edges_from_root(
    constructor: WIGMGraphConstructor,
    root,
):
    return [
        edge
        for edge in constructor.outgoing_edges(root)
        if EdgeAction(edge.action).is_election
    ]


def test_simultaneous_optional_election_groups_match_quota_window_example():
    profile = profile_from_first_preferences([1100, 900, 1000, 500, 496])
    constructor = WIGMGraphConstructor(
        profile,
        m=3,
        LAM=150,
        simultaneous=True,
        memory_lite=True,
    )

    root = expand_root(constructor)

    assert constructor.quota == 1000
    assert election_groups_from_root(constructor, root) == [
        (0,),
        (0, 1),
        (0, 1, 2),
        (0, 2),
        (1,),
        (1, 2),
        (2,),
    ]
    assert len(election_edges_from_root(constructor, root)) == 12


def test_simultaneous_groups_do_not_exceed_remaining_seats():
    profile = profile_from_first_preferences([900, 900, 900, 900, 396])
    constructor = WIGMGraphConstructor(
        profile,
        m=3,
        LAM=150,
        simultaneous=True,
        memory_lite=True,
    )

    root = expand_root(constructor)

    assert constructor.quota == 1000
    assert election_groups_from_root(constructor, root) == [
        (0,),
        (0, 1),
        (0, 1, 2),
        (0, 1, 3),
        (0, 2),
        (0, 2, 3),
        (0, 3),
        (1,),
        (1, 2),
        (1, 2, 3),
        (1, 3),
        (2,),
        (2, 3),
        (3,),
    ]
    assert (0, 1, 2, 3) not in election_groups_from_root(constructor, root)


def test_forced_winner_is_required_in_simultaneous_groups():
    profile = profile_from_first_preferences([1200, 900, 800, 548, 548])
    constructor = WIGMGraphConstructor(
        profile,
        m=3,
        LAM=150,
        simultaneous=True,
        memory_lite=True,
    )

    root = expand_root(constructor)

    assert constructor.quota == 1000
    assert election_groups_from_root(constructor, root) == [(0,), (0, 1)]


def test_too_many_quota_window_candidates_restricts_to_highest_window():
    profile = profile_from_first_preferences([0, 0, 1100, 898])
    constructor = WIGMGraphConstructor(
        profile,
        m=1,
        LAM=150,
        simultaneous=True,
        memory_lite=True,
    )

    root = expand_root(constructor)

    assert constructor.quota == 1000
    assert election_groups_from_root(constructor, root) == [(2,)]


def test_simultaneous_child_cache_removes_every_elected_candidate():
    profile = MatrixProfile(
        candidates=["A", "B", "C", "D", "E"],
        ballot_matrix=np.array(
            [
                [0, 1, 2],
                [1, 0, 2],
                [2, 0, 1],
                [3, -127, -127],
                [4, -127, -127],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([1100, 900, 1000, 500, 496], dtype=np.float64),
    )
    constructor = WIGMGraphConstructor(
        profile,
        m=3,
        LAM=150,
        simultaneous=True,
        memory_lite=False,
    )
    root = expand_root(constructor)

    pair_child = None
    for edge in constructor.outgoing_edges(root):
        if EdgeAction(edge.action) != EdgeAction.SIMULTANEOUS_ELECT:
            continue

        group = sorted(
            other.candidate
            for other in constructor.outgoing_edges(root)
            if (
                other.dst == edge.dst
                and EdgeAction(other.action) == EdgeAction.SIMULTANEOUS_ELECT
            )
        )
        if group == [0, 1]:
            pair_child = edge.dst
            break

    assert pair_child is not None

    cache = constructor._cache_for_expansion(pair_child)

    assert cache.fpv_vec.tolist() == [2, 2, 2, 3, 4]


def test_full_build_runs_with_simultaneous_elections_enabled():
    profile = MatrixProfile(
        candidates=["A", "B", "C", "D"],
        ballot_matrix=np.array(
            [
                [0, 1, 2, 3],
                [1, 0, 2, 3],
                [2, 1, 0, 3],
                [3, 2, 1, 0],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([400, 350, 180, 70], dtype=np.float64),
    )
    constructor = WIGMGraphConstructor(
        profile,
        m=2,
        LAM=70,
        simultaneous=True,
        memory_lite=False,
    )

    constructor.build()

    assert sum(len(edge_layer) for edge_layer in constructor.edge_layers) == 18
    assert [len(layer) for layer in constructor.layers] == [1, 3, 6, 4, 2]


def test_seeded_very_strong_simultaneous_walk_waits_until_candidate_is_forced():
    profile = MatrixProfile(
        candidates=["Very0", "Very1", "Strong", "Weak"],
        ballot_matrix=np.array(
            [
                [0, 1, 2, 3],
                [1, 2, 3, -127],
                [2, 3, -127, -127],
                [3, -127, -127, -127],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([1200, 900, 800, 1097], dtype=np.float64),
    )
    constructor = WIGMGraphConstructor(
        profile,
        m=3,
        LAM=50,
        simultaneous=True,
        memory_lite=True,
    )

    constructor.seeded_build(
        very_strong_candidates={0, 1},
        strong_candidates={2},
        weak_candidates={3},
    )

    connector = constructor.vertex(constructor._seed_connector_ref)
    seated_edges = [edge_ref for edge_ref in connector.key.seated_at if edge_ref is not None]
    seating_edges = [constructor.edge(edge_ref) for edge_ref in seated_edges]

    assert [edge.candidate for edge in seating_edges] == [0, 1]
    assert seating_edges[0].src == constructor.root_ref
    assert seating_edges[1].src == seating_edges[0].dst
    assert seating_edges[0].dst != seating_edges[1].dst
    assert all(edge.action == EdgeAction.FORCE_ELECT for edge in seating_edges)


def test_seeded_build_reports_phase_and_memory_diagnostics(capsys):
    profile = MatrixProfile(
        candidates=["Strong", "Weak"],
        ballot_matrix=np.array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([10.0, 1.0]),
    )
    constructor = WIGMGraphConstructor(
        profile,
        m=1,
        LAM=0,
        memory_lite=False,
    )

    constructor.seeded_build(
        strong_candidates={0},
        weak_candidates={1},
        diagnostic_interval=1,
    )

    output = capsys.readouterr().out
    assert "[seeded_build:initialized]" in output
    assert "[seeded_build:seed allocation complete]" in output
    assert "[seeded_build:construction complete]" in output
    assert "runtime_caches=" in output
    assert "stencils=" in output
    assert "edge_weights=" in output
    assert "rss=" in output


def test_seeded_build_diagnostics_can_be_disabled(capsys):
    profile = MatrixProfile(
        candidates=["Strong", "Weak"],
        ballot_matrix=np.array([[0, 1], [1, 0]], dtype=np.int8),
        wt_vec=np.array([10.0, 1.0]),
    )
    constructor = WIGMGraphConstructor(profile, m=1, LAM=0)

    constructor.seeded_build(
        strong_candidates={0},
        weak_candidates={1},
        diagnostics=False,
    )

    assert "[seeded_build:" not in capsys.readouterr().out


def test_seeded_very_strong_candidates_not_forced_are_not_preseated():
    profile = MatrixProfile(
        candidates=["Very0", "Very1", "Strong", "Weak"],
        ballot_matrix=np.array(
            [
                [0, 1, 2, 3],
                [1, 2, 3, -127],
                [2, 3, -127, -127],
                [3, -127, -127, -127],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([1200, 900, 800, 1097], dtype=np.float64),
    )
    constructor = WIGMGraphConstructor(
        profile,
        m=3,
        LAM=150,
        simultaneous=True,
        memory_lite=True,
    )

    with pytest.warns(RuntimeWarning, match="not pre-seated"):
        constructor.seeded_build(
            very_strong_candidates={0, 1},
            strong_candidates={2},
            weak_candidates={3},
        )

    connector = constructor.vertex(constructor._seed_connector_ref)
    seating_edges = [
        constructor.edge(edge_ref)
        for edge_ref in connector.key.seated_at
        if edge_ref is not None
    ]

    assert [edge.candidate for edge in seating_edges] == [0]
    assert constructor.seed_very_strong_candidates == frozenset({0})


def test_seeded_very_strong_non_simultaneous_picks_highest_forced_first():
    profile = MatrixProfile(
        candidates=["Very0", "Very1", "Strong", "Weak"],
        ballot_matrix=np.array(
            [
                [0, -127, -127, -127],
                [1, -127, -127, -127],
                [2, 3, -127, -127],
                [3, -127, -127, -127],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([1200, 1190, 500, 1107], dtype=np.float64),
    )
    constructor = WIGMGraphConstructor(
        profile,
        m=3,
        LAM=50,
        simultaneous=False,
        memory_lite=True,
    )

    with pytest.warns(RuntimeWarning, match="within LAM of the maximum tally"):
        constructor.seeded_build(
            very_strong_candidates={0, 1},
            strong_candidates={2},
            weak_candidates={3},
        )

    connector = constructor.vertex(constructor._seed_connector_ref)
    seating_edges = [
        constructor.edge(edge_ref)
        for edge_ref in connector.key.seated_at
        if edge_ref is not None
    ]

    assert [edge.candidate for edge in seating_edges] == [0, 1]
    assert seating_edges[0].src == constructor.root_ref
    assert seating_edges[1].src == seating_edges[0].dst


def test_seeded_very_strong_preseed_vertices_store_tallies_for_audit_driver():
    profile = MatrixProfile(
        candidates=["Very0", "Very1", "Strong", "Weak"],
        ballot_matrix=np.array(
            [
                [0, 2, 3, -127],
                [1, 2, 3, -127],
                [2, 3, -127, -127],
                [3, -127, -127, -127],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([1200, 900, 300, 100], dtype=np.float64),
    )
    constructor = WIGMGraphConstructor(
        profile,
        m=3,
        LAM=50,
        simultaneous=True,
        memory_lite=True,
    )

    constructor.seeded_build(
        very_strong_candidates={0, 1},
        strong_candidates={2},
        weak_candidates={3},
    )

    preseed_refs = {
        vertex.ref
        for layer in constructor.layers
        for vertex in layer
        if vertex.path_multiplicity == 0
    }

    for ref in preseed_refs:
        vertex = constructor.vertex(ref)
        assert vertex.tallies is not None

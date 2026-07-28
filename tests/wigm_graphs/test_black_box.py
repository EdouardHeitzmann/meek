from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotting import plot_vertex_densities, plot_wigm_graph
from src.wigm_graphs.black_box import BlackBoxWIGMGraphConstructor
from src.wigm_graphs.datatypes import (
    BlackBoxStateKey,
    EdgeRef,
    EdgeAction,
    PostSeedTallies,
    VertexRef,
)
from votekit.pref_profile import NumpyRankProfile


def _profile() -> NumpyRankProfile:
    return NumpyRankProfile(
        ballot_matrix=np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 2, 3, 4],
                [1, 2, 0, 3, 4],
                [3, -127, -127, -127, -127],
                [4, -127, -127, -127, -127],
                [2, 0, 3, 4, -127],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([30.0, 5.0, 5.0, 30.0, 30.0, 20.0]),
        candidates=("W", "L", "B", "S1", "S2"),
        metadata={},
    )


def _profile_with_very_strong_candidate() -> NumpyRankProfile:
    return NumpyRankProfile(
        ballot_matrix=np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 0],
                [2, 1, 3, 4, 0],
                [3, 1, 2, 4, 0],
                [4, 1, 2, 3, 0],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([50.0, 20.0, 20.0, 20.0, 10.0]),
        candidates=("VS", "W", "S1", "S2", "L"),
        metadata={},
    )


def _build(*, memory_lite: bool = False) -> BlackBoxWIGMGraphConstructor:
    graph = BlackBoxWIGMGraphConstructor(
        _profile(),
        m=3,
        LAM=0,
        simultaneous=False,
        memory_lite=memory_lite,
    )
    return graph.seeded_build(
        strong_candidates=["S1", "S2"],
        uncertain_winner="W",
    )


def test_black_box_build_derives_weak_candidates_and_seed_specific_uncertainty():
    graph = _build()

    assert graph.seed_weak_candidates == frozenset({1})
    assert graph.seed_strong_candidates == frozenset({3, 4})
    assert graph.black_box_winner == 0
    assert len(graph._seed_refs) == 2

    seed_without_b = next(
        ref for ref in graph._seed_refs if 2 not in graph.vertex(ref).key.hopefuls
    )
    seed_with_b = next(
        ref for ref in graph._seed_refs if 2 in graph.vertex(ref).key.hopefuls
    )
    mask_without_b = graph.seed_uncertain_masks[
        graph.vertex(seed_without_b).key.seed_id
    ]
    mask_with_b = graph.seed_uncertain_masks[graph.vertex(seed_with_b).key.seed_id]

    np.testing.assert_array_equal(
        mask_without_b,
        np.array([False, True, True, False, False, True]),
    )
    np.testing.assert_array_equal(
        mask_with_b,
        np.array([False, True, False, False, False, False]),
    )

    without_b_tallies = graph.vertex_post_seed_tallies[seed_without_b]
    with_b_tallies = graph.vertex_post_seed_tallies[seed_with_b]
    assert without_b_tallies.certain.shape == (2, 5)
    assert with_b_tallies.certain.shape == (2, 5)
    assert without_b_tallies.uncertain[3] == 30.0
    assert with_b_tallies.uncertain[2] == 5.0


def test_descendants_keep_seed_identity_and_elections_double_scenarios():
    graph = _build()

    for edge_layer in graph.edge_layers:
        for edge in edge_layer:
            src_key = graph.vertex(edge.src).key
            dst_key = graph.vertex(edge.dst).key
            if (
                isinstance(src_key, BlackBoxStateKey)
                and isinstance(dst_key, BlackBoxStateKey)
                and src_key.seed_id >= 0
            ):
                assert dst_key.seed_id == src_key.seed_id

    post_seed_elections = [
        edge
        for edge_layer in graph.edge_layers
        for edge in edge_layer
        if edge.action in (EdgeAction.ELECT, EdgeAction.FORCE_ELECT)
        and edge.ref != graph.black_box_edge_ref
        and graph.vertex(edge.src).key.seed_id >= 0
    ]
    first_uncertain_election = next(
        edge
        for edge in post_seed_elections
        if graph.edge_weight_scenarios[edge.ref].values.shape[0] == 4
    )
    assert graph.edge_weight_scenarios[first_uncertain_election.ref].values.shape == (
        4,
        6,
    )

    second_uncertain_election = next(
        edge
        for edge in post_seed_elections
        if graph.edge_weight_scenarios[edge.ref].values.shape[0] == 8
    )
    assert graph.edge_weight_scenarios[second_uncertain_election.ref].values.shape == (
        8,
        6,
    )


def test_memory_lite_recomputes_seed_mask_without_storing_it():
    graph = _build(memory_lite=True)

    assert graph.seed_uncertain_masks == {}
    seed_ref = graph._seed_refs[0]
    cache = graph._materialize_cache_from_state(seed_ref)
    mask = graph._uncertain_mask_for_vertex(seed_ref, cache)
    assert mask.shape == (len(_profile().wt_vec),)
    assert mask.dtype == np.bool_


def test_seeded_build_accepts_progress_alias(capsys):
    graph = BlackBoxWIGMGraphConstructor(
        _profile(),
        m=3,
        LAM=0,
        simultaneous=False,
    )
    graph.seeded_build(
        strong_candidates=["S1", "S2"],
        uncertain_winner="W",
        shwo_progress=True,
    )

    captured = capsys.readouterr()
    assert "2 seeds will be constructed" in captured.out
    assert "2/2" in captured.err


def test_allow_seed_merges_uses_the_maximal_uncertainty_mask():
    separate_graph = _build()
    merged_graph = BlackBoxWIGMGraphConstructor(
        _profile(),
        m=3,
        LAM=0,
        simultaneous=False,
    ).seeded_build(
        strong_candidates=["S1", "S2"],
        uncertain_winner="W",
        allow_seed_merges=True,
    )

    separate_seed_without_b = next(
        ref
        for ref in separate_graph._seed_refs
        if 2 not in separate_graph.vertex(ref).key.hopefuls
    )
    maximal_mask = separate_graph.seed_uncertain_masks[
        separate_graph.vertex(separate_seed_without_b).key.seed_id
    ]

    assert {
        merged_graph.vertex(ref).key.seed_id for ref in merged_graph._seed_refs
    } == {0}
    assert len(merged_graph.seed_uncertain_masks) == 1
    np.testing.assert_array_equal(
        merged_graph.seed_uncertain_masks[0],
        maximal_mask,
    )
    assert sum(map(len, merged_graph.layers)) < sum(map(len, separate_graph.layers))
    assert sum(map(len, merged_graph.edge_layers)) < sum(
        map(len, separate_graph.edge_layers)
    )


def test_allow_seed_merges_recomputes_shared_mask_in_memory_lite_mode():
    graph = BlackBoxWIGMGraphConstructor(
        _profile(),
        m=3,
        LAM=0,
        simultaneous=False,
        memory_lite=True,
    ).seeded_build(
        strong_candidates=["S1", "S2"],
        uncertain_winner="W",
        allow_seed_merges=True,
    )

    assert graph.seed_uncertain_masks == {}
    for seed_ref in graph._seed_refs:
        cache = graph._materialize_cache_from_state(seed_ref)
        np.testing.assert_array_equal(
            graph._uncertain_mask_for_vertex(seed_ref, cache),
            np.array([False, True, True, False, False, True]),
        )


def test_lookup_helpers_expose_bounded_tallies_and_weight_scenarios(capsys):
    graph = _build()
    seed_ref = graph._seed_refs[0]
    seed_label = graph.vertex_label(seed_ref)

    vertex = graph.lookup_vertex(seed_label)
    bounded = graph.post_seed_tallies(seed_label)
    assert vertex.ref == seed_ref
    assert bounded is graph.vertex_post_seed_tallies[seed_ref]

    election_edge = next(
        edge
        for edge_layer in graph.edge_layers
        for edge in edge_layer
        if edge.action.is_election and edge.ref != graph.black_box_edge_ref
    )
    edge = graph.lookup_edge(
        f"E{election_edge.ref.layer}:{election_edge.ref.local_id}"
    )
    scenarios = graph.weight_scenarios(election_edge.ref)
    assert edge.ref == election_edge.ref
    assert scenarios is graph.edge_weight_scenarios[election_edge.ref]

    output = capsys.readouterr().out
    assert "uncertain tallies:" in output
    assert "lower tally bounds:" in output
    assert "weight scenarios:" in output


def test_edge_margins_never_mix_transfer_scenarios():
    graph = BlackBoxWIGMGraphConstructor(
        _profile(),
        m=3,
        LAM=0,
        simultaneous=False,
    )
    hopefuls = np.array([2, 3, 4], dtype=int)

    election_tallies = PostSeedTallies(
        certain=np.array(
            [
                [0.0, 0.0, 20.0, 10.0, 10.0],
                [0.0, 0.0, 40.0, 50.0, 50.0],
            ]
        ),
        uncertain=np.array([0.0, 0.0, 3.0, 0.0, 0.0]),
    )
    assert graph._election_margin_for_group(
        election_tallies,
        hopefuls,
        (2,),
    ) == 7.0

    graph.quota = 1000.0
    elimination_tallies = PostSeedTallies(
        certain=np.array(
            [
                [0.0, 0.0, 20.0, 10.0, 100.0],
                [0.0, 0.0, 100.0, 90.0, 110.0],
            ]
        ),
        uncertain=np.array([0.0, 0.0, 0.0, 2.0, 0.0]),
    )
    assert graph._elimination_margin(
        elimination_tallies,
        hopefuls,
        2,
    ) == 8.0


def test_simultaneous_margin_uses_one_scenario_for_the_whole_group():
    graph = BlackBoxWIGMGraphConstructor(
        _profile(),
        m=3,
        LAM=0,
        simultaneous=True,
    )
    graph.quota = 30.0
    tallies = PostSeedTallies(
        certain=np.array(
            [
                [0.0, 0.0, 40.0, 10.0, 0.0],
                [0.0, 0.0, 10.0, 40.0, 0.0],
            ]
        ),
        uncertain=np.zeros(5),
    )

    assert graph._election_margin_for_group(
        tallies,
        np.array([2, 3, 4], dtype=int),
        (2, 3),
    ) == 20.0


def test_plot_replaces_dummy_black_box_edge_with_labeled_seed_bar(monkeypatch):
    graph = _build()
    black_box_edge = graph.edge(graph.black_box_edge_ref)
    hidden_vertex_label = graph.vertex_label(black_box_edge.dst)
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_wigm_graph(
        graph,
        label_vertices=True,
        label_edges="literal",
    )

    axis = plt.gcf().axes[0]
    labels = [text.get_text() for text in axis.texts]
    assert labels.count("+ W") == 1
    assert hidden_vertex_label not in labels

    seed_bar = next(
        patch
        for patch in axis.patches
        if patch.get_facecolor()[:3] == (0.0, 0.0, 0.0)
    )
    assert seed_bar.get_height() >= 0.55
    plt.close("all")


def test_plot_can_highlight_zero_margin_edges(monkeypatch):
    graph = _build()
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_wigm_graph(
        graph,
        label_edges=None,
        highlight_natural_edges=True,
    )

    axis = plt.gcf().axes[0]
    red_arrows = [
        text
        for text in axis.texts
        if getattr(text, "arrow_patch", None) is not None
        and text.arrow_patch.get_edgecolor()[:3] == (1.0, 0.0, 0.0)
    ]
    assert red_arrows
    plt.close("all")


def test_plot_can_highlight_only_root_natural_path(monkeypatch):
    refs = [
        VertexRef(0, 0),
        VertexRef(1, 0),
        VertexRef(1, 1),
        VertexRef(2, 0),
    ]
    vertices = [SimpleNamespace(ref=ref, color=None, tightest_margin=0.0) for ref in refs]
    edges = [
        SimpleNamespace(ref=EdgeRef(0, 0), src=refs[0], dst=refs[1], margin=0.0),
        SimpleNamespace(ref=EdgeRef(0, 1), src=refs[0], dst=refs[2], margin=0.0),
        SimpleNamespace(ref=EdgeRef(1, 0), src=refs[1], dst=refs[3], margin=0.0),
        SimpleNamespace(ref=EdgeRef(1, 1), src=refs[2], dst=refs[3], margin=5.0),
    ]

    class Graph:
        layers = [[vertices[0]], [vertices[1], vertices[2]], [vertices[3]]]
        edge_layers = [[edges[0], edges[1]], [edges[2], edges[3]]]
        root_ref = refs[0]
        m = 1
        n_candidates = 2

        def outgoing_edges(self, ref):
            return [edge for layer in self.edge_layers for edge in layer if edge.src == ref]

    monkeypatch.setattr(plt, "show", lambda: None)

    plot_wigm_graph(
        Graph(),
        label_edges=None,
        highlight_natural_path=True,
    )

    axis = plt.gcf().axes[0]
    red_arrows = [
        text
        for text in axis.texts
        if getattr(text, "arrow_patch", None) is not None
        and text.arrow_patch.get_edgecolor()[:3] == (1.0, 0.0, 0.0)
    ]
    assert len(red_arrows) == 2
    plt.close("all")


def test_plot_vertex_densities_counts_eliminated_vertices(monkeypatch):
    refs = [
        VertexRef(0, 0),
        VertexRef(1, 0),
        VertexRef(1, 1),
        VertexRef(1, 2),
        VertexRef(2, 0),
        VertexRef(2, 1),
    ]
    seat_a_ref = EdgeRef(0, 0)
    vertices = [
        SimpleNamespace(
            ref=refs[0],
            key=SimpleNamespace(seated_at=(None,), hopefuls=frozenset({0, 1, 2})),
        ),
        SimpleNamespace(
            ref=refs[1],
            key=SimpleNamespace(seated_at=(None,), hopefuls=frozenset({0, 1})),
        ),
        SimpleNamespace(
            ref=refs[2],
            key=SimpleNamespace(seated_at=(seat_a_ref,), hopefuls=frozenset({1, 2})),
        ),
        SimpleNamespace(
            ref=refs[3],
            key=SimpleNamespace(seated_at=(None,), hopefuls=frozenset({1, 2})),
        ),
        SimpleNamespace(
            ref=refs[4],
            key=SimpleNamespace(seated_at=(None,), hopefuls=frozenset({1})),
        ),
        SimpleNamespace(
            ref=refs[5],
            key=SimpleNamespace(seated_at=(seat_a_ref,), hopefuls=frozenset({1})),
        ),
    ]

    class Graph:
        candidate_names = ("A", "B", "C")
        n_candidates = 3
        layers = [[vertices[0]], vertices[1:4], vertices[4:6]]

        def edge(self, ref):
            assert ref == seat_a_ref
            return SimpleNamespace(ref=seat_a_ref, candidate=0)

    monkeypatch.setattr(plt, "show", lambda: None)

    ax = plot_vertex_densities(Graph(), ["A"])

    np.testing.assert_allclose(
        [patch.get_height() for patch in ax.patches],
        [0.0, 1.0 / 3.0, 1.0 / 2.0],
    )
    assert [text.get_text() for text in ax.texts] == ["0/1", "1/3", "1/2"]
    plt.close("all")

    ax = plot_vertex_densities(
        Graph(),
        ["A"],
        show_complement_of_eliminated=True,
    )

    np.testing.assert_allclose(
        [patch.get_height() for patch in ax.patches],
        [1.0, 2.0 / 3.0, 1.0 / 2.0],
    )
    assert [text.get_text() for text in ax.texts] == ["1/1", "2/3", "1/2"]
    plt.close("all")


def test_black_box_build_with_very_strong_candidate_uses_preseed_margin_logic():
    graph = BlackBoxWIGMGraphConstructor(
        _profile_with_very_strong_candidate(),
        m=3,
        LAM=0,
        simultaneous=False,
    ).seeded_build(
        very_strong_candidates=["VS"],
        strong_candidates=["S1", "S2"],
        uncertain_winner="W",
    )

    assert graph.seed_very_strong_candidates == frozenset({0})
    assert graph.seed_weak_candidates == frozenset({4})
    assert graph.black_box_winner == 1
    assert graph.black_box_edge_ref is not None

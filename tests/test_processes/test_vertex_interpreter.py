from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from stv_partial_optimizer import evaluate_directional_derivative
from src.election_graphs.datatypes import EdgeAction
from src.test_processes import (
    CobraCompilerV2,
    CobraMentionsCompilerV2,
    CobraMentionsNoiseFilterCompiler,
    CobraNoiseFilterCompiler,
    CobraQuotaCompilerV2,
    CriticalMarginType,
    GlobalAuditDriverV2,
    ImplicitSampler,
    VertexInterpreter,
    plot_profiled_compiler,
)
from src.wigm_graphs.graph_wigm import WIGMGraphConstructor


@dataclass
class MatrixProfile:
    candidates: list[str]
    ballot_matrix: np.ndarray
    wt_vec: np.ndarray

    @property
    def total_ballot_wt(self) -> float:
        return float(self.wt_vec.sum())


def synthetic_profile() -> MatrixProfile:
    candidates = ["W1", "S2", "W3", "S3", "L1", "W2", "L2", "S1"]
    ballots = [
        (25_000, ["W1"]),
        (15_000, ["W1", "S2", "W3"]),
        (15_000, ["W1", "S2", "S3"]),
        (5_000, ["W1", "L1"]),
        (30_000, ["W1", "S3"]),
        (60_000, ["W1", "W3"]),
        (40_000, ["W2", "W3"]),
        (40_000, ["W2", "S3"]),
        (50_000, ["W3"]),
        (70_000, ["S3"]),
        (5_000, ["L1", "W3"]),
        (4_999, ["L2", "W3"]),
        (20_000, ["S1", "W2"]),
        (10_000, ["S2", "W3"]),
        (10_000, ["S2", "W2"]),
    ]
    ballot_matrix = np.full((len(ballots), 3), -127, dtype=np.int8)
    weights = []
    for row_idx, (weight, ranking) in enumerate(ballots):
        ballot_matrix[row_idx, : len(ranking)] = [
            candidates.index(candidate) for candidate in ranking
        ]
        weights.append(weight)

    return MatrixProfile(
        candidates=candidates,
        ballot_matrix=ballot_matrix,
        wt_vec=np.array(weights, dtype=np.float64),
    )


def build_synthetic_graph() -> WIGMGraphConstructor:
    graph = WIGMGraphConstructor(
        synthetic_profile(),
        m=3,
        LAM=9_000,
        memory_lite=True,
        simultaneous=True,
    )
    graph.build()
    return graph


def _basepoint_transfers(basepoint: np.ndarray, quota: float) -> np.ndarray:
    row_totals = np.asarray(basepoint, dtype=np.float64).sum(axis=1)
    degree = int(np.log2(len(row_totals)))
    transfers = np.empty(degree, dtype=np.float64)

    for winner in range(degree):
        tally = 0.0
        for row_mask, row_total in enumerate(row_totals):
            if not ((row_mask >> winner) & 1):
                continue
            weight = 1.0
            for earlier in range(winner):
                if (row_mask >> earlier) & 1:
                    weight *= transfers[earlier]
            tally += weight * row_total
        transfers[winner] = 1.0 - float(quota) / tally

    return transfers


def test_vertex_interpreter_degree_one_basepoint_and_margin():
    graph = build_synthetic_graph()
    l1 = graph.candidate_names.index("L1")
    vertex = next(v for v in graph.layers[1] if l1 in v.key.hopefuls)

    interpreter = VertexInterpreter(graph, vertex)
    base_point = interpreter.base_point("S2", "L1")

    np.testing.assert_array_equal(
        base_point,
        np.array(
            [
                [20_000, 5_000, 224_999],
                [30_000, 5_000, 115_000],
            ],
            dtype=np.float64,
        ),
    )

    evaluation = evaluate_directional_derivative(
        base_point,
        np.zeros_like(base_point),
        graph.quota,
    )
    assert evaluation.margin == pytest.approx(
        vertex.tallies[graph.candidate_names.index("S2")] - vertex.tallies[l1]
    )


def test_vertex_interpreter_degree_two_basepoint_and_margin():
    graph = build_synthetic_graph()
    vertex = graph.layers[6][0]

    assert [
        graph.candidate_names[graph.edge(edge_ref).candidate]
        for edge_ref in vertex.key.seated_at
        if edge_ref is not None
    ] == ["W1", "W2"]

    interpreter = VertexInterpreter(graph, vertex)
    base_point = interpreter.base_point("W3", "S3")

    np.testing.assert_array_equal(
        base_point,
        np.array(
            [
                [69_999, 70_000, 10_000],
                [75_000, 45_000, 30_000],
                [40_000, 40_000, 20_000],
                [0, 0, 0],
            ],
            dtype=np.float64,
        ),
    )

    evaluation = evaluate_directional_derivative(
        base_point,
        np.zeros_like(base_point),
        graph.quota,
    )
    assert evaluation.margin == pytest.approx(
        vertex.tallies[graph.candidate_names.index("W3")]
        - vertex.tallies[graph.candidate_names.index("S3")]
    )


def test_memory_lite_wigm_edges_store_source_fpv_vectors():
    graph = build_synthetic_graph()
    seating_edges = [
        edge
        for layer in graph.edge_layers
        for edge in layer
        if EdgeAction(edge.action).is_seating
    ]

    assert seating_edges
    assert all(edge.fpv_vec is not None for edge in seating_edges)
    assert all(edge.fpv_vec.shape == graph.root_wt_vec.shape for edge in seating_edges)


def test_winner_prefix_indices_match_edgewise_accumulation():
    graph = build_synthetic_graph()
    vertex = next(
        vertex
        for layer in graph.layers
        for vertex in layer
        if vertex.degree >= 2
    )
    interpreter = VertexInterpreter(graph, vertex)

    expected = np.zeros(graph.ballot_matrix.shape[0], dtype=np.int64)
    for bit, edge in enumerate(interpreter.seating_edges):
        expected |= (edge.fpv_vec == edge.candidate).astype(np.int64) << bit

    first = interpreter.winner_prefix_indices()
    cached = interpreter.winner_prefix_indices(copy=False)

    assert np.array_equal(first, expected)
    assert np.array_equal(cached, expected)
    assert cached is interpreter._prefix_cache


def test_seeded_basepoint_uses_unseeded_weights_for_transfer_recursion():
    class Profile:
        candidates = ["Very0", "Very1", "Strong", "Weak"]
        ballot_matrix = np.array(
            [
                [0, -127, -127, -127],
                [1, -127, -127, -127],
                [2, 3, -127, -127],
                [3, -127, -127, -127],
            ],
            dtype=np.int8,
        )
        wt_vec = np.array([1200, 900, 300, 100], dtype=np.float64)
        total_ballot_wt = float(wt_vec.sum())

    graph = WIGMGraphConstructor(
        Profile(),
        m=3,
        LAM=50,
        simultaneous=True,
        memory_lite=True,
    )
    graph.seeded_build(
        very_strong_candidates={0, 1},
        strong_candidates={2},
        weak_candidates={3},
    )
    assert not np.array_equal(graph.root_wt_vec, graph._unseeded_root_wt_vec)

    seed_vertex = next(
        vertex
        for layer in graph.layers
        for vertex in layer
        if frozenset(vertex.key.hopefuls) == {2}
    )
    interpreter = VertexInterpreter(graph, seed_vertex)
    basepoint = interpreter.base_point(2, 3)
    transfers = _basepoint_transfers(basepoint, graph.quota)

    assert np.all(np.isfinite(transfers))
    assert np.all(transfers >= -1.0)
    assert np.all(transfers <= 1.0)


def test_vertex_interpreter_projects_sampled_rows_and_optimizes_unique_thetas():
    graph = build_synthetic_graph()
    vertex = graph.layers[1][0]
    interpreter = VertexInterpreter(graph, vertex)

    theta = interpreter.theta(
        graph.ballot_matrix[0],
        graph.ballot_matrix[10],
        "S2",
        "L1",
    )
    assert theta.shape == (2, 3)
    assert theta.sum() == 0
    assert interpreter.theta_from_key((4, 1)).tolist() == [
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    theta_key = interpreter.theta_key(
        graph.ballot_matrix[0],
        graph.ballot_matrix[10],
        "S2",
        "L1",
    )
    assert interpreter.theta_from_key(theta_key).shape == theta.shape
    np.testing.assert_array_equal(interpreter.theta_from_key(theta_key), theta)

    impacts = interpreter.profile_theta_impacts(
        "S2",
        "L1",
        radius=4_000,
        maxiter=30,
    )
    assert (5, 1) in impacts
    assert all(key[0] != key[1] for key in impacts)
    assert impacts
    assert all(result.feasible for result in impacts.values())
    assert all(np.isfinite(result.minimum) for result in impacts.values())

    base_point = interpreter.base_point("S2", "L1")
    keyed_result = interpreter.optimize(base_point, theta_key, radius=4_000, maxiter=30)
    dense_result = interpreter.optimize(base_point, theta, radius=4_000, maxiter=30)
    assert keyed_result.minimum == pytest.approx(dense_result.minimum)


def test_cobra_compiler_v2_updates_with_cached_optimizer_outputs_and_plots():
    graph = build_synthetic_graph()
    interpreter = VertexInterpreter(graph, graph.layers[1][0])
    compiler = CobraCompilerV2(
        interpreter,
        "S2",
        "L1",
        radius=4_000,
        profile=True,
        optimizer_kwargs={"maxiter": 30},
    )

    sampler = ImplicitSampler.from_profile(
        graph.profile,
        noise_level=0.4,
        sample_size=25,
        with_replacement=True,
        seed=11,
    )
    compiler.run_sampler(sampler)

    assert compiler.num_updates > 0
    assert len(compiler.capital_history) == compiler.num_updates + 1
    assert len(compiler.w_history) == compiler.num_updates
    assert all(key[0] != key[1] for key in compiler.optimizer_cache)
    assert set(compiler.optimizer_cache) == set(compiler.optimizer_results)

    ax = plot_profiled_compiler(compiler)
    assert ax.get_title() == compiler.label


def test_plot_profiled_compiler_places_cropped_bar_labels_at_zero():
    graph = build_synthetic_graph()
    interpreter = VertexInterpreter(graph, graph.layers[1][0])
    compiler = CobraCompilerV2(
        interpreter,
        "S2",
        "L1",
        radius=4_000,
        profile=True,
        optimizer_kwargs={"maxiter": 5},
    )
    compiler.capital_history = [1.0, 1.1]
    compiler.theta_key_history = [(4, 1)]
    compiler.w_history = [-3.0]

    ax = plot_profiled_compiler(compiler)
    label = ax.figure.axes[1].texts[-1]

    assert label.get_position() == (0, 4)
    assert label.xy == (1, 0.0)


def test_cobra_quota_compiler_v2_candidate_below_quota_uses_dead_quota_column():
    graph = build_synthetic_graph()
    interpreter = VertexInterpreter(graph, graph.layers[1][0])
    compiler = CobraQuotaCompilerV2(
        interpreter,
        "L1",
        margin_type=CriticalMarginType.CANDIDATE_BELOW_QUOTA,
        radius=4_000,
        profile=True,
        optimizer_kwargs={"maxiter": 30},
    )

    assert compiler.dead_columns == (0,)
    assert np.all(compiler.base_point[:, 0] == 0.0)
    assert compiler.critical_margin == pytest.approx(
        graph.quota - graph.layers[1][0].tallies[graph.candidate_names.index("L1")]
    )

    theta_key = compiler._quota_theta_key(
        graph.ballot_matrix[10],
        graph.ballot_matrix[0],
    )
    assert theta_key[0] != theta_key[1]
    compiler.update(graph.ballot_matrix[0], graph.ballot_matrix[10])

    assert compiler.num_updates == 1
    assert theta_key in compiler.optimizer_cache
    assert len(compiler.capital_history) == 2


def test_cobra_quota_compiler_v2_candidate_above_quota_uses_dead_quota_column():
    graph = build_synthetic_graph()
    vertex = graph.layers[5][1]
    assert vertex.tallies[graph.candidate_names.index("W2")] > graph.quota

    interpreter = VertexInterpreter(graph, vertex)
    compiler = CobraQuotaCompilerV2(
        interpreter,
        "W2",
        margin_type=CriticalMarginType.CANDIDATE_ABOVE_QUOTA,
        radius=4_000,
        optimizer_kwargs={"maxiter": 30},
    )

    assert compiler.dead_columns == (1,)
    assert np.all(compiler.base_point[:, 1] == 0.0)
    assert compiler.critical_margin == pytest.approx(
        vertex.tallies[graph.candidate_names.index("W2")] - graph.quota
    )


def test_cobra_noise_filter_compiler_uses_half_radius_dilution():
    graph = build_synthetic_graph()
    interpreter = VertexInterpreter(graph, graph.layers[1][0])
    compiler = CobraNoiseFilterCompiler(
        interpreter,
        "S2",
        "L1",
        radius=4_000,
        profile=True,
    )

    assert compiler.critical_margin == 2_000
    assert compiler.v == pytest.approx(4_000 / (2 * graph.profile.total_ballot_wt))

    unchanged = compiler.update(graph.ballot_matrix[0], graph.ballot_matrix[0])
    noisy = compiler.update(graph.ballot_matrix[0], graph.ballot_matrix[10])

    assert compiler.num_updates == 2
    assert compiler.w_history == [0.0, 1.0]
    assert compiler.theta_key_history[0] is None
    assert compiler.theta_key_history[1] is not None
    assert compiler.optimizer_cache == {}
    assert noisy != unchanged


def test_cobra_mentions_compiler_v2_projects_weak_mentions_before_strong():
    graph = build_synthetic_graph()
    interpreter = VertexInterpreter(graph, graph.layers[1][0])
    strong = {
        graph.candidate_names.index("S2"),
        graph.candidate_names.index("W3"),
    }
    weak = graph.candidate_names.index("L1")
    frozen_mentions = np.zeros(graph.n_candidates, dtype=np.float64)
    frozen_mentions[weak] = 5_000.0

    compiler = CobraMentionsCompilerV2(
        interpreter,
        "L1",
        strong_candidates=strong,
        frozen_mentions=frozen_mentions,
        lowest_strong_candidate=graph.candidate_names.index("S2"),
        lowest_strong_tally=30_000,
        radius=4_000,
        profile=True,
        optimizer_kwargs={"maxiter": 30},
    )

    assert compiler.critical_margin == 25_000
    assert compiler._mentions_column(np.array([4, 1, -127])) == 1
    assert compiler._mentions_column(np.array([1, 4, -127])) == 0
    assert compiler._mentions_column(np.array([2, 4, -127])) == 2

    theta_key = compiler._mentions_theta_key(
        np.array([4, 1, -127]),
        np.array([1, 4, -127]),
    )
    assert theta_key[0] != theta_key[1]
    compiler.update(np.array([1, 4, -127]), np.array([4, 1, -127]))

    assert compiler.num_updates == 1
    assert theta_key in compiler.optimizer_cache
    assert len(compiler.capital_history) == 2


def test_cobra_mentions_noise_filter_compiler_uses_mentions_coordinates():
    graph = build_synthetic_graph()
    interpreter = VertexInterpreter(graph, graph.layers[1][0])
    strong = {
        graph.candidate_names.index("S2"),
        graph.candidate_names.index("W3"),
    }
    weak = graph.candidate_names.index("L1")
    frozen_mentions = np.zeros(graph.n_candidates, dtype=np.float64)
    frozen_mentions[weak] = 5_000.0

    compiler = CobraMentionsNoiseFilterCompiler(
        interpreter,
        "L1",
        strong_candidates=strong,
        frozen_mentions=frozen_mentions,
        lowest_strong_candidate=graph.candidate_names.index("S2"),
        lowest_strong_tally=30_000,
        radius=4_000,
        profile=True,
    )

    unchanged = compiler.update(np.array([1, 4, -127]), np.array([1, 4, -127]))
    noisy = compiler.update(np.array([1, 4, -127]), np.array([4, 1, -127]))

    assert compiler.critical_margin == 2_000
    assert compiler.v == pytest.approx(4_000 / (2 * graph.profile.total_ballot_wt))
    assert compiler.w_history == [0.0, 1.0]
    assert compiler.optimizer_cache == {}
    assert noisy != unchanged


def test_global_audit_driver_v2_initializes_pairs_and_tracks_discrepancies():
    graph = build_synthetic_graph()
    driver = GlobalAuditDriverV2(
        graph,
        noise_level=0.4,
        r=4_000,
        sample_size=20,
        seed=23,
        print_diagnostics_every=0,
        optimizer_kwargs={"maxiter": 30},
    )

    assert driver.compilers
    assert len(driver.compilers) == len(driver.compiler_info)
    assert len(driver.compilers) == len(driver.compiler_specs)
    assert any(spec.is_noise_filter for spec in driver.compiler_specs)
    assert len(driver.interpreters) < len(driver.compilers)

    driver.run()

    assert driver.i > 0
    assert driver.discrepancies
    assert all(len(item) == 5 for item in driver.discrepancies)


def test_global_audit_driver_v2_builds_one_forced_elimination_compiler_per_vertex(
    monkeypatch,
):
    graph = build_synthetic_graph()
    forced_elimination_calls = []
    forced_below_quota_election_calls = []
    original = GlobalAuditDriverV2._add_escape_compilers

    def spy(self, audit_graph, vertex, interpreter, candidate, action):
        if self._forced_above_quota_candidate(vertex) is not None:
            if action == EdgeAction.ELIMINATE:
                forced_elimination_calls.append(vertex.ref)
            elif action == EdgeAction.ELECT:
                margin = self._critical_margin_for_escape(vertex, candidate, action)
                if (
                    margin is not None
                    and margin["type"] == CriticalMarginType.CANDIDATE_BELOW_QUOTA
                ):
                    forced_below_quota_election_calls.append(vertex.ref)
        return original(self, audit_graph, vertex, interpreter, candidate, action)

    monkeypatch.setattr(GlobalAuditDriverV2, "_add_escape_compilers", spy)

    GlobalAuditDriverV2(
        graph,
        noise_level=0.1,
        r=4_000,
        sample_size=5,
        seed=31,
        print_diagnostics_every=0,
        optimizer_kwargs={"maxiter": 5},
        verbose=False,
    )

    assert forced_elimination_calls
    assert len(forced_elimination_calls) == len(set(forced_elimination_calls))
    assert forced_below_quota_election_calls == []


def test_global_audit_driver_v2_skips_final_reported_winner_seating_escapes():
    graph = build_synthetic_graph()
    driver = GlobalAuditDriverV2(
        graph,
        noise_level=0.1,
        r=4_000,
        sample_size=5,
        seed=7,
        print_diagnostics_every=0,
        optimizer_kwargs={"maxiter": 5},
    )

    assert driver.canonical_winner_set
    for info, spec in zip(driver.compiler_info, driver.compiler_specs):
        if spec.is_noise_filter or info.action != EdgeAction.ELECT:
            continue
        vertex = graph.vertex(spec.vertex_ref)
        assert not (
            vertex.degree == graph.m - 1
            and info.candidate in driver.canonical_winner_set
        )


def test_global_audit_driver_v2_passes_constant_lambda_to_compilers():
    graph = build_synthetic_graph()
    driver = GlobalAuditDriverV2(
        graph,
        noise_level=0.1,
        r=4_000,
        sample_size=5,
        seed=9,
        print_diagnostics_every=0,
        lambda_value=1.25,
        optimizer_kwargs={"maxiter": 5},
    )

    assert {compiler.lambda_value for compiler in driver.compilers} == {1.25}

    profiled = driver.plot_profile(num_compilers=1)
    assert profiled
    assert profiled[0].lambda_value == 1.25


def test_global_audit_driver_v2_syncs_manual_radius_and_lambda_changes_on_run():
    graph = build_synthetic_graph()
    driver = GlobalAuditDriverV2(
        graph,
        noise_level=0.1,
        r=4_000,
        sample_size=5,
        seed=17,
        print_diagnostics_every=0,
        optimizer_kwargs={"maxiter": 5},
        verbose=False,
    )
    driver.compilers[0].optimizer_cache[(0, 1)] = 1.0
    driver.compilers[0].optimizer_results[(0, 1)] = object()

    driver.r = 8_000
    driver.lambda_ = 1.4
    setattr(driver, "lambda", 1.6)
    driver.run(num_steps=0)

    assert driver.lambda_ == 1.6
    assert {compiler.radius for compiler in driver.compilers} == {8_000.0}
    assert {compiler.lambda_value for compiler in driver.compilers} == {1.6}
    assert driver.compilers[0].optimizer_cache == {}
    assert driver.compilers[0].optimizer_results == {}
    for compiler, spec in zip(driver.compilers, driver.compiler_specs):
        if spec.is_noise_filter:
            assert compiler.critical_margin == pytest.approx(4_000.0)
            assert compiler.v == pytest.approx(8_000.0 / (2.0 * compiler.N))


def test_global_audit_driver_v2_profile_reruns_slowest_compilers():
    graph = build_synthetic_graph()
    driver = GlobalAuditDriverV2(
        graph,
        noise_level=0.4,
        r=4_000,
        sample_size=15,
        seed=37,
        print_diagnostics_every=0,
        profile=True,
        optimizer_kwargs={"maxiter": 20},
    )

    driver.run()
    assert driver.profile_figure is None
    profiled = driver.plot_profile(num_compilers=2)

    if driver.certification_order or driver.pool:
        assert profiled
        assert all("(" in compiler.label and compiler.label.endswith(")") for compiler in profiled)
        assert driver.profile_figure is not None
        assert driver.profile_axes is not None

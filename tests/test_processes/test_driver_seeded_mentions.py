from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np

from src.test_processes.cobra import (
    CobraMentionsCompilerV2,
    CobraMentionsNoiseFilterCompiler,
    CobraNoiseFilterCompiler,
    CobraQuotaNoiseFilterCompiler,
)
from src.test_processes.driver import GlobalAuditDriver, GlobalAuditDriverV2
from src.wigm_graphs.graph_wigm import WIGMGraphConstructor


def test_seeded_mentions_use_post_very_strong_wt_vec_not_unseeded_root():
    graph = SimpleNamespace(
        n_candidates=4,
        ballot_matrix=np.array(
            [
                [3, 0, 2, -127],
                [0, 2, -127, -127],
                [2, 0, -127, -127],
                [1, 2, -127, -127],
            ],
            dtype=np.int8,
        ),
        _unseeded_root_wt_vec=np.array([100.0, 10.0, 10.0, 10.0]),
        root_wt_vec=np.array([20.0, 10.0, 10.0, 10.0]),
    )
    driver = GlobalAuditDriver.__new__(GlobalAuditDriver)
    driver.seed_very_strong_candidates = frozenset({3})
    driver.seed_strong_candidates = frozenset({0, 1})
    driver.seed_frozen_mentions = None
    driver.seed_prebatch_strong_tallies = None

    driver._initialize_seeded_mentions_data(graph)

    assert driver.seed_prebatch_strong_tallies.tolist() == [30.0, 10.0, 10.0, 0.0]
    assert driver.seed_frozen_mentions.tolist() == [0.0, 0.0, 10.0, 0.0]


def test_driver_initializes_from_seeded_graph_with_very_strong_preseed_path():
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
    rows = np.array([[0, 2, 3, -127]], dtype=np.int8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        driver = GlobalAuditDriver(
            graph,
            noise_level=0.0,
            BAL=rows,
            CVR=rows,
            print_diagnostics_every=0,
            simultaneous=True,
        )

    assert len(driver.compilers) > 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        noise_driver = GlobalAuditDriver(
            graph,
            noise_level=0.0,
            BAL=rows,
            CVR=rows,
            compiler_type="noise",
            print_diagnostics_every=0,
            simultaneous=True,
        )

    assert all(
        isinstance(
            compiler,
            (
                CobraNoiseFilterCompiler,
                CobraQuotaNoiseFilterCompiler,
                CobraMentionsNoiseFilterCompiler,
            ),
        )
        for compiler in noise_driver.compilers
    )
    assert any(
        isinstance(compiler, CobraMentionsNoiseFilterCompiler)
        for compiler in noise_driver.compilers
    )
    assert {compiler.LAM for compiler in noise_driver.compilers} == {
        float(graph.LAM)
    }
    assert all(
        compiler.radius == 2.0 * compiler.critical_margin
        for compiler in noise_driver.compilers
    )
    mentions_compiler = next(
        compiler
        for compiler in noise_driver.compilers
        if isinstance(compiler, CobraMentionsNoiseFilterCompiler)
    )
    assert mentions_compiler.critical_margin == (
        mentions_compiler.lowest_strong_tally
        - mentions_compiler.frozen_mentions[mentions_compiler.weak_candidate]
    )


def test_v2_driver_initializes_batch_seeded_mentions_compilers():
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
    rows = np.array([[0, 2, 3, -127]], dtype=np.int8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        driver = GlobalAuditDriverV2(
            graph,
            noise_level=0.0,
            BAL=rows,
            CVR=rows,
            print_diagnostics_every=0,
            simultaneous=True,
        )

    assert any(isinstance(c, CobraMentionsCompilerV2) for c in driver.compilers)
    assert any(isinstance(c, CobraMentionsNoiseFilterCompiler) for c in driver.compilers)
    assert driver.seed_frozen_mentions is not None
    assert driver.seed_prebatch_strong_tallies is not None
    mentions_info = next(info for info in driver.compiler_info if info.escape_id.endswith("-M3"))
    assert driver.lookup_compiler(mentions_info.escape_id).weak_candidate == 3

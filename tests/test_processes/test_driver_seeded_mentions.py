from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np

from src.test_processes.cobra import (
    CobraMentionsCompilerV2,
    CobraMentionsNoiseFilterCompiler,
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

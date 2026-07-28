from __future__ import annotations

import warnings

import numpy as np

from src.election_graphs.datatypes import EdgeAction, ElectionState, VertexRef
from src.test_processes.cobra import CobraCompiler, CriticalMarginType
from src.wigm_graphs.datatypes import StateKey


def make_mentions_compiler() -> CobraCompiler:
    base = ElectionState(
        ref=VertexRef(0, 0),
        key=StateKey(seated_at=(), hopefuls=frozenset({0, 1})),
        degree=0,
        tallies=np.array([100.0, 90.0, 0.0, 0.0]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return CobraCompiler(
            base,
            2,
            ["Strong0", "Strong1", "Weak", "VeryStrong"],
            EdgeAction.ELIMINATE,
            LAM=5,
            quota=150,
            N=200,
            noise_level_guess=0.0,
            use_fallback=True,
            critical_margin_type=CriticalMarginType.CANDIDATE_TO_MENTIONS,
            strong_candidates={0, 1},
            very_strong_candidates={3},
            frozen_mentions=np.array([0.0, 0.0, 20.0, 0.0]),
            lowest_strong_candidate=1,
            lowest_strong_tally=90,
        )


def test_candidate_to_mentions_initializes_margin_from_frozen_mentions():
    compiler = make_mentions_compiler()

    assert compiler.critical_margin_type == CriticalMarginType.CANDIDATE_TO_MENTIONS
    assert compiler.critical_margin == 70
    assert compiler.canonical_victor == 1
    assert compiler.canonical_loser == 2


def test_candidate_to_mentions_assorter_uses_strong_vs_weak_fpv_effect():
    compiler = make_mentions_compiler()

    assert compiler.assorter(np.array([1, -127]), np.array([2, -127])) == 0.0
    assert compiler._last_w == 2

    assert compiler.assorter(np.array([2, -127]), np.array([1, -127])) == 2 * compiler.a
    assert compiler._last_w == -2


def test_candidate_to_mentions_prefix_difference_is_conservative():
    compiler = make_mentions_compiler()

    value = compiler.assorter(
        np.array([3, 1, -127]),
        np.array([1, -127, -127]),
    )

    assert compiler._last_w == 1
    assert value == compiler.a / 2

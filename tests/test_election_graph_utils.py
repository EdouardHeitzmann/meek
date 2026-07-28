import numpy as np

from src.election_graphs.utils import analyze_black_box_seed
from votekit.pref_profile import NumpyRankProfile


def test_analyze_black_box_seed_seats_and_bounds_candidates():
    profile = NumpyRankProfile(
        ballot_matrix=np.array(
            [
                [0, 2, 1, 3],
                [0, 1, 3, 2],
                [1, 2, 3, 0],
                [2, 3, 1, 0],
                [3, 2, 1, 0],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([40.0, 30.0, 15.0, 10.0, 5.0]),
        candidates=("A", "B", "C", "D"),
        metadata={},
    )

    result = analyze_black_box_seed(
        profile,
        very_strong=[0],
        strong=["B"],
        w=2,
        LAM=0,
        simultaneous=True,
        m=2,
        quota=30,
        verbose=False,
    )

    np.testing.assert_allclose(
        result["base_wt_vec"],
        np.array([40.0, 30.0, 15.0, 10.0, 5.0]) * np.array(
            [4 / 7, 4 / 7, 1, 1, 1]
        ),
    )
    assert result["maximum_surplus"] == 5.0
    assert result["maximum_transfer_value"] == 1 / 7
    assert result["maximum_tallies"]["D"] == 45 / 7
    assert result["candidate_comparisons"]["D"] == {
        "maximum_possible_tally": 45 / 7,
        "lowest_strong_fpv": 15.0 + 30.0 * 4 / 7,
        "is_weak": True,
    }
    assert result["weak"] == ("D",)
    assert result["in_between"] == ()
    assert result["seed_count"] == 1


def test_analyze_black_box_seed_requires_quota_information():
    profile = NumpyRankProfile(
        ballot_matrix=np.array([[0, 1]], dtype=np.int8),
        wt_vec=np.array([1.0]),
        candidates=("A", "B"),
        metadata={},
    )

    try:
        analyze_black_box_seed(
            profile,
            very_strong=[],
            strong=["A"],
            w="B",
            LAM=0,
            verbose=False,
        )
    except ValueError as error:
        assert "Quota cannot be inferred" in str(error)
    else:
        raise AssertionError("Expected missing quota information to be rejected.")


def test_lam_inflated_surplus_propagates_to_transfer_and_candidate_tallies():
    profile = NumpyRankProfile(
        ballot_matrix=np.array(
            [
                [0, 1, 2],
                [1, 2, 0],
                [2, 1, 0],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([10.0, 20.0, 5.0]),
        candidates=("Strong", "W", "Other"),
        metadata={},
    )

    result = analyze_black_box_seed(
        profile,
        very_strong=[],
        strong=["Strong"],
        w="W",
        LAM=5,
        quota=20,
        verbose=False,
    )

    assert result["raw_maximum_surplus"] == 5.0
    assert result["maximum_surplus"] == 10.0
    assert result["maximum_transfer_value"] == 1 / 3
    assert np.isclose(result["maximum_tallies"]["Other"], 5.0 + 20.0 / 3.0)


def test_maximum_tally_gives_uncertain_rows_full_weight():
    profile = NumpyRankProfile(
        ballot_matrix=np.array(
            [
                [0, 2, 3, 1],
                [1, 0, 2, 3],
                [3, 2, 1, 0],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([20.0, 10.0, 5.0]),
        candidates=("W", "Predecessor", "Weak", "Strong"),
        metadata={},
    )

    result = analyze_black_box_seed(
        profile,
        very_strong=[],
        strong=["Strong"],
        w="W",
        LAM=0,
        m=1,
        quota=20,
        verbose=False,
    )

    assert result["maximum_transfer_value"] == 1 / 3
    assert result["uncertain_weight_by_candidate"]["Weak"] == 10.0
    assert np.isclose(result["maximum_tallies"]["Weak"], 10.0 + 20.0 / 3.0)

from __future__ import annotations

import numpy as np

from src.test_processes.noise import ImplicitSampler


def test_aggressive_target_indices_are_deterministic_and_front_loaded():
    sampler_a = ImplicitSampler(
        noise_level=1.0,
        n_cands=10,
        wt_vec=np.array([20_000], dtype=np.float64),
        sample_size=10_000,
        with_replacement=True,
        seed=101,
    )
    sampler_b = ImplicitSampler(
        noise_level=1.0,
        n_cands=10,
        wt_vec=np.array([20_000], dtype=np.float64),
        sample_size=10_000,
        with_replacement=True,
        seed=101,
    )

    np.testing.assert_array_equal(sampler_a.target_indices, sampler_b.target_indices)
    assert 0.87 <= np.mean(sampler_a.target_indices <= sampler_a.n_cands) <= 0.93


def test_aggressive_repeated_serials_share_target_indices_with_replacement():
    sampler = ImplicitSampler(
        noise_level=1.0,
        n_cands=5,
        wt_vec=np.array([1_000], dtype=np.float64),
        sample_size=200,
        with_replacement=True,
        seed=17,
    )

    serial_targets: dict[int, int] = {}
    duplicates_seen = False
    for serial, target in zip(sampler.sampled_serials, sampler.target_indices):
        serial_int = int(serial)
        if serial_int in serial_targets:
            duplicates_seen = True
        previous = serial_targets.setdefault(serial_int, int(target))
        assert previous == int(target)
    assert duplicates_seen


def test_aggressive_operations_target_early_positions():
    sampler = ImplicitSampler(
        noise_level=0.0,
        n_cands=6,
        wt_vec=np.array([2], dtype=np.float64),
        sample_size=1,
        with_replacement=True,
        seed=13,
    )
    rng = np.random.default_rng(23)
    row = np.array([0, 1, 2, -127, -127], dtype=np.int8)

    replaced = sampler._replace_candidate(row, rng, target_index=0)
    assert replaced[0] not in {0, 1, 2}
    np.testing.assert_array_equal(replaced[1:], row[1:])

    inserted = sampler._insert_candidate(row, rng, target_index=0)
    assert inserted[1:4].tolist() == [0, 1, 2]

    deleted = sampler._delete_candidate(row, rng, target_index=0)
    assert deleted.tolist() == [1, 2, -127, -127, -127]

    swapped = sampler._swap_candidates(row, rng, target_index=0)
    assert swapped[0] != 0
    assert set(swapped[:3]) == {0, 1, 2}


def test_ballot_completion_noise_ignores_aggressive_target_index():
    sampler = ImplicitSampler(
        noise_level=0.0,
        n_cands=6,
        wt_vec=np.array([2], dtype=np.float64),
        sample_size=1,
        with_replacement=True,
        seed=19,
    )
    rng_a = np.random.default_rng(29)
    rng_b = np.random.default_rng(29)
    row = np.array([0, 1, -127, -127, -127], dtype=np.int8)

    direct = sampler._append_candidates(row, rng_a)
    via_dispatch = sampler._apply_noise_operation(row, 4, rng_b, target_index=0)

    np.testing.assert_array_equal(via_dispatch, direct)

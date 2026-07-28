from __future__ import annotations

from collections.abc import Iterable
from math import comb

import numpy as np
from numpy.typing import NDArray


SENTINEL = -127


def _normalize_candidates(candidates: Iterable[int] | None) -> frozenset[int]:
    if candidates is None:
        return frozenset()
    return frozenset(int(candidate) for candidate in candidates)


def fpv_tallies_from_matrix(
    ballot_matrix: NDArray[np.integer],
    wt_vec: NDArray[np.float64],
    n_candidates: int,
    *,
    allowed_candidates: Iterable[int] | None = None,
    masked_candidates: Iterable[int] | None = None,
) -> NDArray[np.float64]:
    """
    Compute first-preference tallies after removing masked candidates.

    If allowed_candidates is provided, only those candidates can receive first
    preferences; this is useful for checking tallies after condensing to a
    proposed strong set.
    """
    allowed = None
    if allowed_candidates is not None:
        allowed = _normalize_candidates(allowed_candidates)

    masked = _normalize_candidates(masked_candidates)
    tallies = np.zeros(n_candidates, dtype=np.float64)

    for row, weight in zip(ballot_matrix, wt_vec):
        for raw_candidate in row:
            candidate = int(raw_candidate)
            if candidate == SENTINEL:
                break
            if candidate in masked:
                continue
            if allowed is not None and candidate not in allowed:
                continue

            tallies[candidate] += float(weight)
            break

    return tallies


def frozen_mentions_from_matrix(
    ballot_matrix: NDArray[np.integer],
    wt_vec: NDArray[np.float64],
    n_candidates: int,
    strong_candidates: Iterable[int],
    *,
    masked_candidates: Iterable[int] | None = None,
) -> NDArray[np.float64]:
    """
    Count mentions before the first strong candidate in each condensed row.
    """
    strong = _normalize_candidates(strong_candidates)
    masked = _normalize_candidates(masked_candidates)
    mentions = np.zeros(n_candidates, dtype=np.float64)

    for row, weight in zip(ballot_matrix, wt_vec):
        for raw_candidate in row:
            candidate = int(raw_candidate)
            if candidate == SENTINEL:
                break
            if candidate in masked:
                continue
            if candidate in strong:
                break

            mentions[candidate] += float(weight)

    return mentions


def weak_candidates_from_strong(
    ballot_matrix: NDArray[np.integer],
    wt_vec: NDArray[np.float64],
    n_candidates: int,
    strong_candidates: Iterable[int],
    *,
    LAM: float,
    quota: float,
    verify_strong: bool = False,
    masked_candidates: Iterable[int] | None = None,
) -> tuple[frozenset[int], NDArray[np.float64]]:
    """
    Determine the weak set induced by a prescribed strong set.

    A non-strong candidate is weak when its frozen mentions are at least LAM
    below the smallest current first-preference tally among strong candidates.
    """
    strong = _normalize_candidates(strong_candidates)
    masked = _normalize_candidates(masked_candidates)
    active_strong = strong - masked
    if not active_strong:
        raise ValueError("Cannot derive weak candidates from an empty strong set.")

    if verify_strong:
        strong_only_tallies = fpv_tallies_from_matrix(
            ballot_matrix,
            wt_vec,
            n_candidates,
            allowed_candidates=active_strong,
            masked_candidates=masked,
        )
        bad_strong = [
            candidate
            for candidate in active_strong
            if strong_only_tallies[candidate] + LAM >= quota
        ]
        if bad_strong:
            details = ", ".join(
                f"{candidate}: {strong_only_tallies[candidate]}"
                for candidate in sorted(bad_strong)
            )
            raise ValueError(
                "Strong candidates must remain below quota minus LAM when "
                f"standing alone; violating tallies are {details}."
            )

    current_fpv = fpv_tallies_from_matrix(
        ballot_matrix,
        wt_vec,
        n_candidates,
        masked_candidates=masked,
    )
    smallest_strong_fpv = min(float(current_fpv[candidate]) for candidate in active_strong)
    frozen_mentions = frozen_mentions_from_matrix(
        ballot_matrix,
        wt_vec,
        n_candidates,
        active_strong,
        masked_candidates=masked,
    )

    weak = frozenset(
        candidate
        for candidate in range(n_candidates)
        if (
            candidate not in active_strong
            and candidate not in masked
            and frozen_mentions[candidate] + LAM <= smallest_strong_fpv
        )
    )

    return weak, frozen_mentions


def search_strong_weak_candidates(
    ballot_matrix: NDArray[np.integer],
    wt_vec: NDArray[np.float64],
    n_candidates: int,
    *,
    remaining_seats: int,
    LAM: float,
    quota: float,
    masked_candidates: Iterable[int] | None = None,
) -> tuple[frozenset[int], frozenset[int], NDArray[np.float64]]:
    """
    Greedily search for a strong set inducing the largest weak set.

    Candidates are added in current FPV order, beginning with enough candidates
    to fill the remaining seats. The search stops once a valid expansion no
    longer grows the induced weak set.
    """
    masked = _normalize_candidates(masked_candidates)
    if remaining_seats <= 0:
        return frozenset(), frozenset(), np.zeros(n_candidates, dtype=np.float64)

    current_fpv = fpv_tallies_from_matrix(
        ballot_matrix,
        wt_vec,
        n_candidates,
        masked_candidates=masked,
    )
    candidate_order = [
        candidate
        for candidate in np.argsort(-current_fpv)
        if int(candidate) not in masked
    ]
    if len(candidate_order) < remaining_seats:
        raise ValueError(
            "Cannot search strong candidates: fewer active candidates than "
            "remaining seats."
        )

    best_strong: frozenset[int] | None = None
    best_weak: frozenset[int] = frozenset()
    best_mentions = np.zeros(n_candidates, dtype=np.float64)

    for size in range(remaining_seats, len(candidate_order) + 1):
        strong = frozenset(int(candidate) for candidate in candidate_order[:size])
        strong_only_tallies = fpv_tallies_from_matrix(
            ballot_matrix,
            wt_vec,
            n_candidates,
            allowed_candidates=strong,
            masked_candidates=masked,
        )
        if any(strong_only_tallies[candidate] + LAM >= quota for candidate in strong):
            continue

        weak, mentions = weak_candidates_from_strong(
            ballot_matrix,
            wt_vec,
            n_candidates,
            strong,
            LAM=LAM,
            quota=quota,
            verify_strong=False,
            masked_candidates=masked,
        )

        if best_strong is None or len(weak) > len(best_weak):
            best_strong = strong
            best_weak = weak
            best_mentions = mentions
            continue

        break

    if best_strong is None:
        raise ValueError(
            "Could not find a valid strong set whose condensed tallies stay "
            "below quota minus LAM."
        )

    return best_strong, best_weak, best_mentions


def analyze_black_box_seed(
    profile,
    very_strong: Iterable[str | int],
    strong: Iterable[str | int],
    w: str | int,
    LAM: float,
    simultaneous: bool = True,
    *,
    m: int | None = None,
    quota: float | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """
    Compute the exploratory bounds used by a black-boxed election seed.

    Candidate indices always refer to the input profile. Results use candidate
    names because VoteKit reindexes candidates whenever a profile is condensed.
    """
    from votekit.pref_profile.numpy_profile import (
        numpy_profile_fpv,
        remove_and_reweigh_and_condense,
    )

    candidates = tuple(profile.candidates)
    candidate_to_index = profile.candidate_to_index

    def candidate_name(candidate: str | int) -> str:
        if isinstance(candidate, str):
            if candidate not in candidate_to_index:
                raise ValueError(f"Unknown candidate: {candidate!r}")
            return candidate

        index = int(candidate)
        if index < 0 or index >= len(candidates):
            raise ValueError(f"Candidate index out of range: {index}")
        return candidates[index]

    very_strong_names = tuple(
        dict.fromkeys(candidate_name(candidate) for candidate in very_strong)
    )
    strong_names = tuple(
        dict.fromkeys(candidate_name(candidate) for candidate in strong)
    )
    w_name = candidate_name(w)

    very_strong_set = set(very_strong_names)
    strong_set = set(strong_names)
    if very_strong_set & strong_set:
        raise ValueError("very_strong and strong must be disjoint.")
    if w_name in very_strong_set or w_name in strong_set:
        raise ValueError("w must be disjoint from very_strong and strong.")
    if not strong_names:
        raise ValueError("strong must contain at least one candidate.")

    if m is None:
        m = profile.metadata.get("m", profile.metadata.get("seats"))
    if quota is None:
        quota = profile.metadata.get("quota")
    if quota is None:
        if m is None:
            raise ValueError(
                "Quota cannot be inferred from a NumpyRankProfile. Pass m or "
                "quota, or store one of them in profile.metadata."
            )
        quota = np.floor(profile.total_ballot_wt / (int(m) + 1)) + 1
    quota = float(quota)

    base_profile = profile
    pending = set(very_strong_names)
    seating_rounds: list[dict[str, object]] = []

    while pending:
        tallies = numpy_profile_fpv(base_profile)
        eligible = [
            candidate
            for candidate in pending
            if tallies[candidate] > quota + LAM
        ]
        eligible.sort(
            key=lambda candidate: (-tallies[candidate], candidates.index(candidate))
        )

        if not eligible:
            details = ", ".join(
                f"{candidate}={tallies[candidate]:.6g}"
                for candidate in sorted(pending, key=candidates.index)
            )
            raise ValueError(
                "The remaining very-strong candidates are not more than LAM "
                f"above quota {quota:.6g}: {details}."
            )

        elected = eligible if simultaneous else eligible[:1]
        transfer_values = [
            (tallies[candidate] - quota) / tallies[candidate]
            for candidate in elected
        ]
        seating_rounds.append(
            {
                "candidates": tuple(elected),
                "tallies": tuple(tallies[candidate] for candidate in elected),
                "transfer_values": tuple(transfer_values),
            }
        )
        base_profile = remove_and_reweigh_and_condense(
            base_profile,
            elected,
            transfer_values,
            remove_empty_ballots=False,
            remove_zero_weight_ballots=False,
        )
        pending.difference_update(elected)

    bounded_candidates = [
        candidate
        for candidate in base_profile.candidates
        if candidate not in strong_set and candidate != w_name
    ]
    base_matrix = base_profile.ballot_matrix
    base_indices = base_profile.candidate_to_index
    fixed_indices = np.fromiter(
        (base_indices[candidate] for candidate in strong_set | {w_name}),
        dtype=np.intp,
    )
    fixed_mask = np.isin(base_matrix, fixed_indices)
    w_index = base_indices[w_name]
    base_fpv = base_matrix[:, 0]

    def first_two_with(candidate: str) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
        kept_mask = fixed_mask | (base_matrix == base_indices[candidate])
        kept_rank = np.cumsum(kept_mask, axis=1)
        first = np.where(
            kept_mask & (kept_rank == 1),
            base_matrix,
            SENTINEL,
        ).max(axis=1)
        second = np.where(
            kept_mask & (kept_rank == 2),
            base_matrix,
            SENTINEL,
        ).max(axis=1)
        return first, second

    surplus_by_candidate: dict[str, float] = {}
    direct_tally_by_candidate: dict[str, float] = {}
    w_transfer_weight_by_candidate: dict[str, float] = {}
    uncertain_weight_by_candidate: dict[str, float] = {}

    for candidate in bounded_candidates:
        first, second = first_two_with(candidate)
        candidate_index = base_indices[candidate]
        surplus_mask = (first == candidate_index) & (second == w_index)
        surplus_by_candidate[candidate] = float(
            base_profile.wt_vec[surplus_mask].sum()
        )
        direct_tally_by_candidate[candidate] = float(
            base_profile.wt_vec[first == candidate_index].sum()
        )
        w_transfer_weight_by_candidate[candidate] = float(
            base_profile.wt_vec[
                (base_fpv == w_index)
                & (first == w_index)
                & (second == candidate_index)
            ].sum()
        )
        uncertain_weight_by_candidate[candidate] = float(
            base_profile.wt_vec[
                (base_fpv != w_index)
                & (first == w_index)
                & (second == candidate_index)
            ].sum()
        )

    raw_maximum_surplus = max(surplus_by_candidate.values(), default=0.0)
    maximum_surplus = raw_maximum_surplus + LAM
    maximum_transfer_value = maximum_surplus / (quota + maximum_surplus)
    maximum_tallies: dict[str, float] = {}

    for candidate in bounded_candidates:
        maximum_tallies[candidate] = float(
            direct_tally_by_candidate[candidate]
            + maximum_transfer_value * w_transfer_weight_by_candidate[candidate]
            + uncertain_weight_by_candidate[candidate]
        )

    base_tallies = numpy_profile_fpv(base_profile)
    strong_base_tallies = {
        candidate: base_tallies[candidate] for candidate in strong_names
    }
    lowest_strong_tally = min(strong_base_tallies.values())
    weak = tuple(
        candidate
        for candidate in bounded_candidates
        if maximum_tallies[candidate] + LAM <= lowest_strong_tally
    )
    in_between = tuple(
        candidate for candidate in bounded_candidates if candidate not in weak
    )
    seed_count: int | None = None
    if m is not None:
        remaining_seats = int(m) - len(very_strong_names) - 1
        minimum_in_between = max(remaining_seats - len(strong_names), 0)
        seed_count = sum(
            comb(len(in_between), subset_size)
            for subset_size in range(minimum_in_between, len(in_between) + 1)
        )
    candidate_comparisons = {
        candidate: {
            "maximum_possible_tally": maximum_tallies[candidate],
            "lowest_strong_fpv": lowest_strong_tally,
            "is_weak": candidate in weak,
        }
        for candidate in bounded_candidates
    }

    result = {
        "quota": quota,
        "base_profile": base_profile,
        "base_wt_vec": base_profile.wt_vec.copy(),
        "seating_rounds": tuple(seating_rounds),
        "surplus_by_candidate": surplus_by_candidate,
        "uncertain_weight_by_candidate": uncertain_weight_by_candidate,
        "raw_maximum_surplus": raw_maximum_surplus,
        "maximum_surplus": maximum_surplus,
        "maximum_transfer_value": maximum_transfer_value,
        "strong_base_tallies": strong_base_tallies,
        "lowest_strong_tally": lowest_strong_tally,
        "maximum_tallies": maximum_tallies,
        "candidate_comparisons": candidate_comparisons,
        "weak": weak,
        "in_between": in_between,
        "seed_count": seed_count,
    }

    if verbose:
        print(f"Maximum surplus of {w_name}: {maximum_surplus:.6g}")
        print(f"Maximum transfer value of {w_name}: {maximum_transfer_value:.6g}")
        print(f"Candidate comparisons: {candidate_comparisons}")
        print(f"Weak candidates: {list(weak)}")
        print(
            f"Found {len(weak)} weak and {len(in_between)} in-between candidates."
        )
        if seed_count is not None:
            print(f"Black-box construction will require {seed_count} seeds.")

    return result

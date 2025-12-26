from votekit.pref_profile import PreferenceProfile
import numpy as np
from numpy.typing import NDArray
import random

def convert_pf_to_numpy_arrays(
        pf: PreferenceProfile
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        This converts the profile into a numpy matrix with some helper arrays for faster iteration.

        Args:
            profile (RankProfile): The preference profile to convert.

        Returns:
            tuple[NDArray, NDArray, NDArray]: The ballot matrix, weights vector, and
                first-preference vector.
        """
        df = pf.df
        candidate_to_index = {
            frozenset([name]): i for i, name in enumerate(pf.candidates)
        }
        candidate_to_index[frozenset(["~"])] = -127

        ranking_columns = [c for c in df.columns if c.startswith("Ranking")]
        num_rows = len(df)
        num_cols = len(ranking_columns)
        cells = df[ranking_columns].to_numpy()

        def map_cell(cell):
            try:
                return candidate_to_index[cell]
            except KeyError:
                raise TypeError("Ballots must have rankings.")

        mapped = np.frompyfunc(map_cell, 1, 1)(cells).astype(np.int8)

        # Add padding
        ballot_matrix: NDArray = np.full((num_rows, num_cols + 1), -126, dtype=np.int8)
        ballot_matrix[:, :num_cols] = mapped

        wt_vec: NDArray = df["Weight"].astype(np.float64).to_numpy()
        fpv_vec: NDArray = ballot_matrix[:, 0].copy()

        # Reject ballots that have no rankings at all (all -127)
        empty_rows = np.where(np.all(ballot_matrix == -127, axis=1))[0]
        if empty_rows.size:
            raise TypeError("Ballots must have rankings.")

        return ballot_matrix, wt_vec, fpv_vec


def single_ballot_matrix(ballot_matrix: NDArray, wt_vec: NDArray) -> NDArray:
    """
    Expand weighted ballots into a single ballot matrix. (NOT CURRENTLY USED -- this is not the right noising philosophy).
    (Instead we will use an implicit pool of noised indices.)
    Args:
        ballot_matrix (NDArray): Matrix where each row is a ballot.
        wt_vec (NDArray): Weights indicating how many times each ballot should appear.

    Returns:
        NDArray: Matrix where each ballot is duplicated according to its weight.
    """
    ballot_matrix = np.asarray(ballot_matrix)
    wt_vec = np.asarray(wt_vec)

    if ballot_matrix.shape[0] != wt_vec.shape[0]:
        raise ValueError("ballot_matrix rows and wt_vec length must match.")

    if np.any(wt_vec < 0):
        raise ValueError("Weights must be non-negative.")

    rounded_weights = np.rint(wt_vec).astype(np.int64)
    if not np.allclose(wt_vec, rounded_weights):
        raise ValueError("Weights must be whole numbers to expand ballots.")

    return np.repeat(ballot_matrix, rounded_weights, axis=0)

def sample_rows_from_mult_vec(
    mult_vec: NDArray,  _amount:int = None, frac:float = None, return_positions: bool = False
) -> NDArray:
    """
    Samples s row indices to transfer from an implicit pool,
    where each row index i appears wt_vec[i] times if fpv_vec[i] == winner.
    Returns a counts vector where counts[i] is the number of times row i was sampled.
    Ensures sum(counts) == s and counts[i] <= wt_vec[i].

    Args:
        fpv_vec (NDArray): First-preference vector.
        wt_vec (NDArray): Integer weights vector.
        winner (int): Candidate code whose ballots are to be transferred.
        surplus (int): Number of surplus votes to transfer.
    """
    rng = np.random.default_rng()

    # running example: assume that candidate 2 just won.
    # assume the fpv_vec looks like [2,5,3,2]
    # then eligible looks like [True, False, False, True]
    # and winner_row_indices looks like [0, 3]
    #eligible = mult_vec > 0
    #row_indices = np.flatnonzero(eligible)
    vec_length = mult_vec.shape[0]

    # assume the original weight vector was [200, 100, 50, 25]
    # then wts looks like [200, 25]
    wts = mult_vec.copy().astype(np.int64)

    # assume that quota was 220, so winner 2 had 5 surplus votes and 225 transferable votes
    transferable = int(wts.sum())

    # this deals with cases where there are fewer than surplus votes to transfer
    # (lots of exhausted ballots)
    if frac is None and _amount is None:
        raise ValueError("Either frac or _amount must be specified.")
    if frac is not None and _amount is not None:
        raise ValueError("Only one of frac or _amount may be specified.")
    if _amount is not None:
        amount = _amount
    else:
        amount = int(frac*sum(mult_vec))
    amount = min(amount, transferable)

    # Sample surplus distinct positions in the implicit pool [0, transferable)
    # in our example: we sample 5 distinct numbers from [0, 225)
    positions_to_transfer = rng.choice(transferable, size=amount, replace=False)
    positions_to_transfer.sort()

    # Say we sampled the numbers 12, 50, 178, 200, and 201
    # numbers 0 through 199 inclusive get mapped to the first bin, so the first three sampled
    # votes go to winner_row_index[0]
    # numbers 200 and 201 get mapped to the second bin, so they go to our second
    #  winner_row_index[1]
    bins = np.cumsum(wts)  # len = len(idx)
    owners = np.searchsorted(
        bins, positions_to_transfer, side="right"
    )  # values in winner_row_indices

    # Accumulate counts back to global rows
    counts = np.bincount(owners, minlength=vec_length)
    if return_positions:
        return counts, positions_to_transfer
    return counts


def build_noise_submatrix(
    ballot_matrix: NDArray, counts: NDArray, num_cands: int
) -> NDArray:
    """
    Build a matrix of noised ballots sampled according to counts.

    Each row in the output is a noised copy of a row in ballot_matrix,
    selected according to counts produced by sample_rows_from_mult_vec.

    Args:
        ballot_matrix (NDArray): Original ballots (each row is a ballot).
        counts (NDArray): How many times to sample each row of ballot_matrix.
        num_cands (int): Maximum candidate code (exclusive upper bound).

    Returns:
        NDArray: Noised ballots with shape (counts.sum(), ballot_matrix.shape[1]).
    """
    rng = np.random.default_rng()
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty((0, ballot_matrix.shape[1]), dtype=ballot_matrix.dtype)

    noised = np.empty((total, ballot_matrix.shape[1]), dtype=ballot_matrix.dtype)
    insert_pos = 0

    for row_idx, count in enumerate(counts):
        if count == 0:
            continue
        source_row = ballot_matrix[row_idx]
        for _ in range(int(count)):
            noised[insert_pos] = _noise_row(source_row, num_cands, rng)
            insert_pos += 1

    return noised


def _noise_row(row: NDArray, num_cands: int, rng: np.random.Generator) -> NDArray:
    """
    Apply one of four noise operations to a ballot row, with specified fallbacks.
    """
    row = np.asarray(row)
    num_pos = sum(row >= 0)
    num_neg = sum(row < 0)
    if num_pos == 0:
        noise_type = 2
    elif num_pos == 1:
        noise_type = rng.choice([1, 2, 4])
    elif num_neg == 1:
        noise_type = rng.choice([1, 3])
    else:
        noise_type = rng.integers(1, 5)  # [1, 4]

    def available_candidates():
        ranked = set(int(x) for x in row if x >= 0)
        return [c for c in range(num_cands) if c not in ranked]

    def apply_option1():
        # Delete a random ranking.
        nonneg_positions = np.flatnonzero(row[:-1] >= 0)
        if nonneg_positions.size == 0:
            return None
        remove_idx = int(rng.choice(nonneg_positions))
        remaining = [int(val) for i, val in enumerate(row[:-1]) if i != remove_idx and val >= 0]
        new_row = np.full_like(row, -127)
        new_row[-1] = -126
        new_row[: len(remaining)] = remaining
        return new_row

    def apply_option2():
        # Insert a new ranking if there is space before the final sentinel.
        if row[-2] >= 0:
            return None
        avail = available_candidates()
        cand = int(rng.choice(avail if avail else range(num_cands)))

        # Keep only current rankings, remove the designated negative slot,
        # then insert the new candidate among existing rankings.
        rankings = [int(v) for v in row[:-1] if v >= 0]
        insert_idx = int(rng.integers(0, len(rankings) + 1))
        rankings.insert(insert_idx, cand)

        new_row = np.full_like(row, -127)
        new_row[-1] = -126
        new_row[: len(rankings)] = rankings
        return new_row

    def apply_option3():
        # Swap two rankings.
        nonneg_positions = np.flatnonzero(row >= 0)
        if nonneg_positions.size <= 1:
            return None
        # If swapping equal values would be a no-op, retry a few times.
        for _ in range(4):
            idx1, idx2 = rng.choice(nonneg_positions, size=2, replace=False)
            if row[idx1] == row[idx2]:
                continue
            new_row = row.copy()
            new_row[idx1], new_row[idx2] = new_row[idx2], new_row[idx1]
            return new_row
        return None

    def apply_option4():
        # Replace one ranking with an unused candidate.
        nonneg_positions = np.flatnonzero(row >= 0)
        if nonneg_positions.size == 0:
            return None
        avail = available_candidates()
        if not avail:
            return None
        target_pos = int(rng.choice(nonneg_positions))
        replacement = int(rng.choice(avail))
        new_row = row.copy()
        new_row[target_pos] = replacement
        return new_row

    # Choose the primary option, then fall back following the specified rules.
    if noise_type == 1:
        candidate = apply_option1()
    elif noise_type == 2:
        candidate = apply_option2()
    elif noise_type == 3:
        candidate = apply_option3()
    else:
        candidate = apply_option4()
        
    if candidate is None:
        raise ValueError(f"Failed to generate a noised ballot: row={row}, noise_type={noise_type}, num_pos={num_pos}, num_neg={num_neg}")
    if np.array_equal(candidate, row):
        raise ValueError("Noised ballot should differ from original but did not.")
    return candidate


def sample_bal_cvr(
    ballot_matrix: NDArray,
    mult_vec: NDArray,
    noised_counts: NDArray,
    noised_positions: NDArray,
    noised_submatrix: NDArray,
    n: int,
) -> tuple[NDArray, NDArray]:
    """
    Sample n ballots to form BAL (possibly noised) and CVR (original) matrices.

    Args:
        ballot_matrix (NDArray): Original ballots.
        mult_vec (NDArray): Weight vector used for sampling.
        noised_counts (NDArray): Counts passed to build_noise_submatrix.
        noised_positions (NDArray): Implicit pool positions that were noised.
        noised_submatrix (NDArray): Noised rows produced by build_noise_submatrix.
        n (int): Number of rows to sample.

    Returns:
        tuple[NDArray, NDArray]: BAL (with noise substitutions) and CVR (original) matrices.
    """

    mult_vec = np.asarray(mult_vec, dtype=np.int64)
    total_weight = mult_vec.sum()

    #frac = n / float(total_weight)
    sampled_counts, sampled_positions = sample_rows_from_mult_vec(
        mult_vec, _amount=n, return_positions=True
    )
    if sampled_counts.sum() != n or sampled_positions.shape[0] != n:
        raise ValueError("Sampling did not produce the requested number of rows.")

    # Map sampled positions to their owning row indices in the implicit pool.
    bins = np.cumsum(mult_vec.astype(np.int64))
    sampled_owners = np.searchsorted(bins, sampled_positions, side="right")

    noised_counts = np.asarray(noised_counts, dtype=np.int64)
    if noised_counts.sum() != noised_submatrix.shape[0]:
        raise ValueError("noised_counts and noised_submatrix size mismatch.")

    noised_positions = np.asarray(noised_positions, dtype=np.int64)
    if noised_positions.ndim != 1:
        raise ValueError("noised_positions must be a 1-D array of positions.")

    # Build mapping from implicit pool position -> row in noised_submatrix.
    noised_owners = np.searchsorted(bins, noised_positions, side="right")
    starts = np.cumsum(noised_counts) - noised_counts
    seen = np.zeros_like(noised_counts, dtype=np.int64)
    pos_to_noised_idx = {}
    for pos, owner in zip(noised_positions, noised_owners):
        idx = starts[owner] + seen[owner]
        pos_to_noised_idx[pos] = idx
        seen[owner] += 1

    bal = np.empty((n, ballot_matrix.shape[1]), dtype=ballot_matrix.dtype)
    cvr = np.empty((n, ballot_matrix.shape[1]), dtype=ballot_matrix.dtype)

    for i, (row_idx, pos) in enumerate(zip(sampled_owners, sampled_positions)):
        cvr[i] = ballot_matrix[row_idx]
        noised_idx = pos_to_noised_idx.get(pos)
        if noised_idx is not None:
            bal[i] = noised_submatrix[noised_idx]
        else:
            bal[i] = ballot_matrix[row_idx]

    return bal, cvr


def append_ghosts(
    ballot_matrix: NDArray, mult_vec: NDArray, num_ghosts: int
) -> tuple[NDArray, NDArray]:
    """
    Append a single ghost ballot row and its multiplicity.

    The ghost row is all -127 except for the trailing -126 sentinel.
    """
    ghost_row = np.full((1, ballot_matrix.shape[1]), -127, dtype=ballot_matrix.dtype)
    ghost_row[0, -1] = -126

    new_ballot_matrix = np.vstack([ballot_matrix, ghost_row])
    new_mult_vec = np.concatenate([np.asarray(mult_vec), np.array([num_ghosts], dtype=np.int64)])
    return new_ballot_matrix, new_mult_vec

def bal_cvr_sample_constructor(bal_matrix, mult_vec, noise_level, sample_size, num_ghosts):
    bal_matrix, mult_vec = append_ghosts(bal_matrix, mult_vec, num_ghosts)
    #print(f"Total multiplicity after adding ghosts: {np.sum(mult_vec)}")

    num_cands = np.max(bal_matrix)+1
    indices_to_noise, positions_to_noise = sample_rows_from_mult_vec(mult_vec=mult_vec, frac = noise_level, return_positions=True)
    noised_submatrix = build_noise_submatrix(bal_matrix, indices_to_noise, num_cands)
    
    bal, cvr = sample_bal_cvr(
        ballot_matrix=bal_matrix,
        mult_vec=mult_vec,
        noised_counts=indices_to_noise,
        noised_positions=positions_to_noise,
        noised_submatrix=noised_submatrix,
        n=sample_size,
    )
    return bal, cvr

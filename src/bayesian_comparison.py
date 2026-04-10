"""
Bayesian ballot-comparison audit for STV elections.

This module implements a naive first-pass version of Rivest's Bayesian ballot-comparison
audit framework, adapted for STV elections using VoteKit preference profiles.

The approach uses row-wise Dirichlet conjugacy with direct gamma-based posterior simulation.
For each reported ballot type, we maintain a Dirichlet prior/posterior over the actual
ballot type distribution. Sequential ballot comparisons update these posteriors, and we
can generate posterior predictive samples of the full election tally by sampling from
each row's posterior and summing.

This is exploratory code for research purposes, not a certified risk-limiting audit
implementation. Correctness, clarity, and modularity are prioritized over aggressive
optimization.

Key concepts:
- t: number of possible ballot types (including an invalid/other bucket)
- n: total number of ballots in the CVR
- R_j: number of ballots in the CVR with reported type j
- C_{j,k}: observed count of sampled ballots reported as j but actually k
- alpha_{j,k}: prior hyperparameter for the (j,k) cell

Posterior predictive simulation for a single draw:
1. For each reported type j, sample a Dirichlet draw for the composition of unaudited
   ballots in row j using Gamma(alpha_{j,:} + C_{j,:}).
2. Scale by the number of unaudited ballots (R_j - sum(C_{j,:})).
3. Add back the observed counts C_{j,:}.
4. Sum over all rows to get a full-election ballot type tally.
5. Convert to a VoteKit profile and run FastSTV.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
from src.utils import convert_pf_to_numpy_arrays
from src.edouard.FastSTV import FastCore
from src.edouard.noise import bal_cvr_sample_constructor


# ============================================================================
# Ballot Type Enumeration
# ============================================================================

@dataclass(frozen=True)
class BallotType:
    """
    Represents a single ballot type for an STV election.
    
    A ballot type is defined by its ranking vector, where each position contains
    a candidate index (-1 for exhausted/unranked, -2 for invalid/other).
    
    For efficiency with small-to-medium candidate sets, we represent partial rankings
    as tuples. Two ballots have the same type if they rank the same candidates in
    the same order up to exhaustion.
    """
    ranking: tuple[int, ...]
    
    def __hash__(self):
        return hash(self.ranking)
    
    def __eq__(self, other):
        if not isinstance(other, BallotType):
            return False
        return self.ranking == other.ranking
    
    def is_invalid(self) -> bool:
        """Returns True if this ballot type represents an invalid/other ballot."""
        return len(self.ranking) > 0 and self.ranking[0] == -2


def ballot_row_to_type(ballot_row: NDArray, truncate_at: int = -127) -> BallotType:
    """
    Convert a single ballot row (from the numpy ballot matrix) to a BallotType.
    
    Args:
        ballot_row: 1D array of candidate indices from the ballot matrix.
        truncate_at: Sentinel value indicating exhaustion/padding (default -127).
    
    Returns:
        BallotType object representing this ballot's ranking.
    """
    # Truncate at the first exhaustion marker or padding
    ranking_list = []
    for cand_idx in ballot_row:
        if cand_idx == truncate_at or cand_idx == -126:
            break
        ranking_list.append(int(cand_idx))
    
    # If the ballot is completely empty or invalid, mark it as invalid type
    if len(ranking_list) == 0:
        return BallotType((-2,))
    
    return BallotType(tuple(ranking_list))


def enumerate_ballot_types(
    ballot_matrix: NDArray,
    include_invalid: bool = True
) -> tuple[list[BallotType], dict[BallotType, int], dict[int, BallotType]]:
    """
    Enumerate all unique ballot types present in a ballot matrix.
    
    Args:
        ballot_matrix: numpy array of shape (n_ballots, max_ranking_length).
        include_invalid: if True, ensure there is at least one invalid/other type.
    
    Returns:
        types_list: list of BallotType objects (index matters for type_to_idx).
        type_to_idx: dict mapping BallotType -> integer index in [0, t-1].
        idx_to_type: dict mapping integer index -> BallotType.
    """
    unique_types = set()
    for i in range(ballot_matrix.shape[0]):
        bt = ballot_row_to_type(ballot_matrix[i, :])
        unique_types.add(bt)
    
    # Ensure we have an invalid type bucket
    invalid_type = BallotType((-2,))
    if include_invalid and invalid_type not in unique_types:
        unique_types.add(invalid_type)
    
    # Sort for deterministic ordering (convert to string for sorting)
    types_list = sorted(unique_types, key=lambda bt: bt.ranking)
    
    type_to_idx = {bt: i for i, bt in enumerate(types_list)}
    idx_to_type = {i: bt for i, bt in enumerate(types_list)}
    
    return types_list, type_to_idx, idx_to_type


def profile_to_reported_types(
    pf: PreferenceProfile,
    type_to_idx: dict[BallotType, int]
) -> tuple[NDArray, NDArray]:
    """
    Convert a VoteKit PreferenceProfile to reported ballot type arrays.
    
    Args:
        pf: VoteKit preference profile (the reported CVR).
        type_to_idx: mapping from BallotType to integer index.
    
    Returns:
        reported_type_by_ballot: length-n array of type indices (one per ballot, with
            ballots expanded according to weights).
        R: length-t array of reported type totals.
    """
    ballot_matrix, wt_vec, _ = convert_pf_to_numpy_arrays(pf)
    
    # Expand ballots according to weights (assumes integer weights)
    wt_int = np.rint(wt_vec).astype(np.int64)
    if not np.allclose(wt_vec, wt_int):
        raise ValueError("Ballot weights must be integers for ballot-comparison audit.")
    
    expanded_ballot_matrix = np.repeat(ballot_matrix, wt_int, axis=0)
    n = expanded_ballot_matrix.shape[0]
    
    # Map each ballot to its type index
    reported_type_by_ballot = np.zeros(n, dtype=np.int32)
    for i in range(n):
        bt = ballot_row_to_type(expanded_ballot_matrix[i, :])
        reported_type_by_ballot[i] = type_to_idx[bt]
    
    # Compute reported type totals
    t = len(type_to_idx)
    R = np.zeros(t, dtype=np.int64)
    for type_idx in range(t):
        R[type_idx] = np.sum(reported_type_by_ballot == type_idx)
    
    return reported_type_by_ballot, R


# ============================================================================
# Audit State Management
# ============================================================================

@dataclass
class AuditState:
    """
    State of a sequential ballot-comparison audit for STV.
    
    Attributes:
        t: number of observed ballot types.
        n: total number of ballots.
        R: length-t array of reported type totals.
        C: (t, t+1) matrix of observed counts, where C[j, k] is the number of sampled
            ballots reported as type j but actually type k. Column t is for unseen types.
        alpha: (t, t+1) matrix of prior hyperparameters. Column t has unseen_prior_mass.
        sample_size: total number of ballots sampled so far.
        type_to_idx: mapping from BallotType to integer index.
        idx_to_type: mapping from integer index to BallotType.
        reported_outcome: the reported election outcome (tuple of frozensets of candidate names).
        candidates: list of candidate names from the original profile.
        num_winners: number of seats (m).
        unseen_prior_mass: prior probability mass allocated to unseen ballot types.
        profile_for_stv: the original reported PreferenceProfile (for reconstruction).
    """
    t: int
    n: int
    R: NDArray  # shape (t,)
    C: NDArray  # shape (t, t+1)
    alpha: NDArray  # shape (t, t+1)
    sample_size: int
    type_to_idx: dict[BallotType, int]
    idx_to_type: dict[int, BallotType]
    reported_outcome: tuple[frozenset[str], ...]
    candidates: list[str]
    num_winners: int
    unseen_prior_mass: float
    ballot_matrix_template: NDArray  # template ballot matrix for reconstruction
    profile_for_stv: PreferenceProfile
    
    def row_sample_sizes(self) -> NDArray:
        """Return the number of sampled ballots for each reported type."""
        return np.sum(self.C, axis=1)
    
    def unaudited_counts(self) -> NDArray:
        """Return the number of unaudited ballots for each reported type."""
        return self.R - self.row_sample_sizes()
    
    def posterior_hyperparameters(self) -> NDArray:
        """Return the posterior hyperparameters alpha + C."""
        return self.alpha + self.C


def initialize_audit_state(
    reported_profile: PreferenceProfile,
    prior: Literal["haldane", "laplace", "custom"] = "laplace",
    custom_alpha: Optional[NDArray] = None,
    num_winners: int = 1,
    unseen_prior_mass: float = 1.0
) -> AuditState:
    """
    Initialize an audit state from a reported CVR profile.
    
    Args:
        reported_profile: VoteKit PreferenceProfile representing the reported CVR.
        prior: one of "haldane" (all zeros), "laplace" (all ones), or "custom".
        custom_alpha: if prior=="custom", a (t, t+1) array of prior hyperparameters.
        num_winners: number of seats (m) in the STV election.
        unseen_prior_mass: prior probability mass for unseen ballot types (added to column t).
        
    Returns:
        AuditState object ready for sequential ballot comparisons.
    """
    # Enumerate ballot types
    ballot_matrix, wt_vec, _ = convert_pf_to_numpy_arrays(reported_profile)
    types_list, type_to_idx, idx_to_type = enumerate_ballot_types(ballot_matrix)
    t = len(types_list)
    
    # Convert profile to reported type arrays
    reported_type_by_ballot, R = profile_to_reported_types(reported_profile, type_to_idx)
    n = len(reported_type_by_ballot)
    
    # Initialize C matrix (all zeros) - includes column for unseen types
    C = np.zeros((t, t + 1), dtype=np.int64)
    
    # Initialize alpha matrix - includes column for unseen types
    if prior == "haldane":
        alpha = np.zeros((t, t + 1), dtype=np.float64)
        # Still give unseen types some prior mass to avoid division by zero
        alpha[:, t] = unseen_prior_mass
    elif prior == "laplace":
        alpha = np.ones((t, t + 1), dtype=np.float64)
        # Scale unseen column by unseen_prior_mass
        alpha[:, t] = unseen_prior_mass
    elif prior == "custom":
        if custom_alpha is None:
            raise ValueError("custom_alpha must be provided when prior='custom'.")
        if custom_alpha.shape != (t, t + 1):
            raise ValueError(f"custom_alpha must have shape ({t}, {t + 1}).")
        alpha = custom_alpha.astype(np.float64)
    else:
        raise ValueError(f"Unknown prior type: {prior}")
    
    # Compute reported outcome by running STV on the reported profile
    try:
        fast_stv = create_fast_stv(
            ballot_matrix=ballot_matrix,
            wt_vec=wt_vec,
            candidates=reported_profile.candidates,
            num_winners=num_winners,
            profile=reported_profile
        )
        reported_outcome = fast_stv.get_elected(no_fsets=False)
    except Exception as e:
        # Fallback: use top m candidates by first preference
        raise RuntimeError(f"Failed to compute reported outcome: {e}")
    
    return AuditState(
        t=t,
        n=n,
        R=R,
        C=C,
        alpha=alpha,
        sample_size=0,
        type_to_idx=type_to_idx,
        idx_to_type=idx_to_type,
        reported_outcome=reported_outcome,
        candidates=reported_profile.candidates,
        num_winners=num_winners,
        unseen_prior_mass=unseen_prior_mass,
        ballot_matrix_template=ballot_matrix,
        profile_for_stv=reported_profile
    )


def record_comparison(
    state: AuditState,
    reported_type_idx: int,
    actual_type_idx: int
) -> None:
    """
    Record a single ballot comparison (mutates the state in-place).
    
    Args:
        state: the current AuditState.
        reported_type_idx: the reported type index for this ballot.
        actual_type_idx: the actual (hand-inspected) type index for this ballot.
    """
    if reported_type_idx < 0 or reported_type_idx >= state.t:
        raise ValueError(f"reported_type_idx {reported_type_idx} out of range [0, {state.t}).")
    if actual_type_idx < 0 or actual_type_idx >= state.t:
        raise ValueError(f"actual_type_idx {actual_type_idx} out of range [0, {state.t}).")
    
    state.C[reported_type_idx, actual_type_idx] += 1
    state.sample_size += 1


def record_comparisons_batch(
    state: AuditState,
    reported_type_indices: NDArray,
    actual_type_indices: NDArray
) -> None:
    """
    Record multiple ballot comparisons at once (mutates the state in-place).
    
    Args:
        state: the current AuditState.
        reported_type_indices: array of reported type indices.
        actual_type_indices: array of actual type indices.
    """
    if len(reported_type_indices) != len(actual_type_indices):
        raise ValueError("reported and actual type arrays must have the same length.")
    
    for r_idx, a_idx in zip(reported_type_indices, actual_type_indices):
        record_comparison(state, int(r_idx), int(a_idx))


# ============================================================================
# Posterior Simulation
# ============================================================================

def sample_row_posterior(
    state: AuditState,
    row_idx: int,
    rng: np.random.Generator
) -> NDArray:
    """
    Sample a posterior predictive row tally for reported type row_idx.
    
    Uses the gamma/Dirichlet method:
    1. Sample gamma variates with shapes (alpha[row_idx, :] + C[row_idx, :]).
    2. Normalize to get a Dirichlet draw (composition of unaudited ballots).
    3. Scale by the number of unaudited ballots in this row.
    4. Add back the observed counts C[row_idx, :].
    
    Note: Column t (last column) represents unseen ballot types.
    
    Args:
        state: the current AuditState.
        row_idx: the reported type index to sample for.
        rng: numpy random generator.
    
    Returns:
        length-(t+1) array of simulated actual type counts for this row.
    """
    posterior_params = state.alpha[row_idx, :] + state.C[row_idx, :]
    unaudited_count = state.R[row_idx] - np.sum(state.C[row_idx, :])
    
    if unaudited_count < 0:
        raise ValueError(f"Row {row_idx} has more sampled ballots than reported ballots.")
    
    if unaudited_count == 0:
        # No unaudited ballots; just return the observed counts
        return state.C[row_idx, :].astype(np.float64)
    
    # Sample gamma variates (including unseen type column)
    # For Haldane prior (alpha=0), gamma(0) is improper, but if we have any observations,
    # the posterior is proper. If no observations and alpha=0, we treat it as uniform.
    gamma_samples = np.zeros(state.t + 1, dtype=np.float64)
    for k in range(state.t + 1):
        shape = posterior_params[k]
        if shape > 0:
            gamma_samples[k] = rng.gamma(shape, 1.0)
        else:
            # Haldane prior with no observations: treat as zero (will normalize to uniform if all zero)
            gamma_samples[k] = 0.0
    
    # Normalize to get Dirichlet draw (composition)
    total_gamma = np.sum(gamma_samples)
    if total_gamma > 0:
        composition = gamma_samples / total_gamma
    else:
        # All gamma samples are zero (only happens if all posterior params are zero)
        # Fall back to uniform (including unseen types)
        composition = np.ones(state.t + 1, dtype=np.float64) / (state.t + 1)
    
    # Scale by unaudited count and add observed counts
    unaudited_tally = composition * unaudited_count
    row_tally = unaudited_tally + state.C[row_idx, :].astype(np.float64)
    
    return row_tally


def sample_full_election_tally(
    state: AuditState,
    rng: np.random.Generator
) -> NDArray:
    """
    Sample a full-election ballot type tally from the posterior.
    
    For each reported type row, sample its posterior row tally and sum over all rows.
    The last element (index t) represents unseen types.
    
    Args:
        state: the current AuditState.
        rng: numpy random generator.
    
    Returns:
        length-(t+1) array of simulated ballot type counts (last element is unseen types).
    """
    full_tally = np.zeros(state.t + 1, dtype=np.float64)
    for j in range(state.t):
        row_tally = sample_row_posterior(state, j, rng)
        full_tally += row_tally
    return full_tally


# ============================================================================
# STV Outcome Integration
# ============================================================================

def filter_invalid_ballots(
    ballot_matrix: NDArray,
    wt_vec: NDArray
) -> tuple[NDArray, NDArray]:
    """
    Filter out invalid/empty ballots from a ballot matrix and normalize for FastCore.
    
    Invalid ballots are those where the first preference is negative (< 0),
    which includes exhausted ballots (all -127) and invalid markers (-2).
    
    Also cleans any ballot that has negative candidate indices (other than -127/-126)
    in any position, as these will cause issues when STV advances through preferences.
    
    FastCore expects only candidate indices (>=0) or exhausted (-127). We normalize
    padding values (-126) to exhausted (-127) for compatibility.
    
    Args:
        ballot_matrix: ballot matrix (n x max_length)
        wt_vec: weight vector (length n)
    
    Returns:
        (filtered_ballot_matrix, filtered_wt_vec) with invalid ballots removed and normalized
    """
    # Get first preference for each ballot
    fpv = ballot_matrix[:, 0]
    
    # Check for any negative values (other than -127/-126) in any position
    # These indicate invalid ballot structures
    has_invalid_values = np.any(
        (ballot_matrix < -1) & (ballot_matrix != -127) & (ballot_matrix != -126),
        axis=1
    )
    
    # Keep only ballots with valid first preference (>= 0) and no invalid internal structure
    valid_mask = (fpv >= 0) & (~has_invalid_values)
    
    filtered_ballot_matrix = ballot_matrix[valid_mask].copy()
    filtered_wt_vec = wt_vec[valid_mask]
    
    # Normalize the ballot matrix: replace -126 (padding) with -127 (exhausted)
    # This ensures FastCore only sees candidates (>=0) or exhausted (-127)
    filtered_ballot_matrix[filtered_ballot_matrix == -126] = -127
    
    return filtered_ballot_matrix, filtered_wt_vec


def create_fast_stv(
    ballot_matrix: NDArray,
    wt_vec: NDArray,
    candidates: list[str],
    num_winners: int,
    profile: PreferenceProfile
) -> FastCore:
    """
    Create a FastCore instance with proper setup.
    
    FastCore expects self.candidates and self.profile to be set, but doesn't
    initialize them. This helper sets them before the main initialization runs.
    
    Invalid/empty ballots (first preference < 0) are filtered out before running STV,
    as they don't participate in the election.
    
    Args:
        ballot_matrix: ballot matrix
        wt_vec: weight vector
        candidates: list of candidate names
        num_winners: number of winners (m)
        profile: original PreferenceProfile
    
    Returns:
        Initialized FastCore instance
    """
    # Filter out invalid/empty ballots
    filtered_matrix, filtered_wt = filter_invalid_ballots(ballot_matrix, wt_vec)
    
    # Handle edge case where all ballots are invalid
    if len(filtered_matrix) == 0:
        raise ValueError("All ballots are invalid/empty; cannot run STV.")
    
    # Monkey-patch FastCore to accept candidates
    fast_stv = object.__new__(FastCore)
    fast_stv.candidates = candidates
    fast_stv.profile = profile
    # Now call __init__
    FastCore.__init__(
        fast_stv,
        ballot_matrix=filtered_matrix,
        mult_vec=filtered_wt,
        num_cands=len(candidates),
        m=num_winners,
        transfer="fractional",
        quota="droop"
    )
    return fast_stv


def ballot_type_to_ranking_row(bt: BallotType, num_cands: int, max_length: int) -> NDArray:
    """
    Convert a BallotType back to a ballot matrix row.
    
    Args:
        bt: BallotType object.
        num_cands: number of candidates in the contest.
        max_length: maximum ranking length (number of columns in ballot matrix).
    
    Returns:
        1D array of shape (max_length,) with candidate indices and padding.
    """
    row = np.full(max_length, -127, dtype=np.int8)
    for i, cand_idx in enumerate(bt.ranking):
        if i >= max_length:
            break
        row[i] = cand_idx
    return row


def tally_to_profile(
    type_tally: NDArray,
    state: AuditState
) -> PreferenceProfile:
    """
    Convert a ballot type tally to a VoteKit PreferenceProfile.
    
    Args:
        type_tally: length-t array of ballot type counts (can be fractional).
        state: the current AuditState (for type metadata).
    
    Returns:
        PreferenceProfile suitable for running FastSTV.
    """
    # Construct ballots with weights
    ballots = []
    weights = []
    
    max_length = state.ballot_matrix_template.shape[1]
    
    for type_idx in range(state.t):
        count = type_tally[type_idx]
        if count <= 0:
            continue
        
        bt = state.idx_to_type[type_idx]
        if bt.is_invalid():
            # Skip invalid ballots (or handle appropriately)
            continue
        
        # Build ranking tuple with candidate names
        ranking = []
        for cand_idx in bt.ranking:
            if cand_idx >= 0 and cand_idx < len(state.candidates):
                ranking.append(frozenset([state.candidates[cand_idx]]))
        
        if len(ranking) == 0:
            continue
        
        ballot = Ballot(ranking=tuple(ranking))
        ballots.append(ballot)
        weights.append(count)
    
    if len(ballots) == 0:
        # Fallback: create a dummy ballot
        ballots = [Ballot(ranking=(frozenset([state.candidates[0]]),))]
        weights = [1.0]
    
    profile = PreferenceProfile(ballots=ballots, candidates=state.candidates)
    profile.df["Weight"] = weights
    
    return profile


def run_stv_on_tally(
    type_tally: NDArray,
    state: AuditState
) -> tuple[frozenset[str], ...]:
    """
    Run FastSTV on a simulated ballot type tally and return the winner set.
    
    Invalid/empty ballots and unseen types are excluded from the STV calculation.
    Unseen types (index t, last element) are treated as exhausted/invalid.
    
    Args:
        type_tally: length-(t+1) array of ballot type counts (last element is unseen types).
        state: the current AuditState.
    
    Returns:
        Tuple of frozensets of elected candidate names.
    """
    # Build a ballot matrix from the type tally
    # We need to construct a weighted ballot matrix
    # Invalid ballots and unseen types are filtered out (they don't participate in STV)
    ballot_rows = []
    weights = []
    
    max_length = state.ballot_matrix_template.shape[1]
    
    # Only iterate over observed types (not unseen type at index t)
    for type_idx in range(state.t):
        count = type_tally[type_idx]
        if count <= 0:
            continue
        
        bt = state.idx_to_type[type_idx]
        # Skip invalid ballots - they don't participate in the election
        if bt.is_invalid():
            continue
        
        row = ballot_type_to_ranking_row(bt, len(state.candidates), max_length)
        ballot_rows.append(row)
        weights.append(count)
    
    # Note: unseen types (type_tally[state.t]) are implicitly excluded here
    
    if len(ballot_rows) == 0:
        # All ballots are invalid; return empty outcome
        return (frozenset(),)
    
    ballot_matrix = np.vstack(ballot_rows)
    wt_vec = np.array(weights, dtype=np.float64)
    
    try:
        fast_stv = create_fast_stv(
            ballot_matrix=ballot_matrix,
            wt_vec=wt_vec,
            candidates=state.candidates,
            num_winners=state.num_winners,
            profile=state.profile_for_stv
        )
        outcome = fast_stv.get_elected(no_fsets=False)
        return outcome
    except Exception as e:
        # If STV fails (e.g., not enough candidates), return empty
        # TODO: better error handling for large M
        return (frozenset(),)


# ============================================================================
# Audit Stopping and Reporting
# ============================================================================

def run_posterior_simulations(
    state: AuditState,
    num_simulations: int,
    seed: Optional[int] = None
) -> dict:
    """
    Run many posterior simulations and estimate upset probability.
    
    Args:
        state: the current AuditState.
        num_simulations: number of posterior draws to generate.
        seed: random seed for reproducibility.
    
    Returns:
        dict with keys:
            - "upset_probability": estimated P(simulated outcome != reported outcome).
            - "winner_set_counts": dict mapping outcome tuples to counts.
            - "num_simulations": number of simulations run.
    """
    rng = np.random.default_rng(seed)
    
    winner_set_counts = {}
    upset_count = 0
    
    for _ in range(num_simulations):
        type_tally = sample_full_election_tally(state, rng)
        outcome = run_stv_on_tally(type_tally, state)
        
        # Canonicalize outcome for comparison
        outcome_key = tuple(sorted([tuple(sorted(fs)) for fs in outcome]))
        winner_set_counts[outcome_key] = winner_set_counts.get(outcome_key, 0) + 1
        
        # Check if this is an upset
        if outcome != state.reported_outcome:
            upset_count += 1
    
    upset_probability = upset_count / num_simulations if num_simulations > 0 else 0.0
    
    return {
        "upset_probability": upset_probability,
        "winner_set_counts": winner_set_counts,
        "num_simulations": num_simulations
    }


def sequential_stopping_check(
    state: AuditState,
    risk_limit: float,
    num_simulations: int = 1000,
    seed: Optional[int] = None
) -> tuple[bool, dict]:
    """
    Check if the audit can stop at the current sample size.
    
    Stops if the posterior upset probability <= risk_limit.
    
    Args:
        state: the current AuditState.
        risk_limit: desired risk limit (e.g., 0.05).
        num_simulations: number of posterior draws for estimating upset probability.
        seed: random seed.
    
    Returns:
        (can_stop, results) where can_stop is True if upset_prob <= risk_limit,
        and results is the output of run_posterior_simulations.
    """
    results = run_posterior_simulations(state, num_simulations, seed)
    can_stop = results["upset_probability"] <= risk_limit
    return can_stop, results


# ============================================================================
# Simulation and Noise Helpers for Testing
# ============================================================================

def generate_noised_sample(
    true_profile: PreferenceProfile,
    sample_size: int,
    noise_level: float,
    num_ghosts: int = 0,
    seed: Optional[int] = None
) -> tuple[NDArray, NDArray]:
    """
    Generate a noised ballot comparison sample using the existing noise infrastructure.
    
    This uses the sophisticated noise model from src.edouard.noise, which:
    - Adds ghost ballots (empty ballots with all -127)
    - Pre-selects which positions to noise at the specified noise_level
    - Returns both BAL (true ballots) and CVR (reported ballots) matrices
    
    Args:
        true_profile: the ground-truth PreferenceProfile.
        sample_size: number of ballots to sample.
        noise_level: fraction of the population to pre-noise (e.g., 0.02 for 2%).
        num_ghosts: number of empty ghost ballots to add to the population.
        seed: random seed (note: bal_cvr_sample_constructor uses its own RNG).
    
    Returns:
        (BAL, CVR) where:
            BAL is the sample of true ballots (possibly noised from CVR)
            CVR is the sample of reported ballots
            Both are numpy arrays of shape (sample_size, max_ranking_length)
    """
    # Note: bal_cvr_sample_constructor uses numpy's default_rng internally
    # which doesn't accept a seed parameter. For reproducibility, we'd need
    # to modify the source or accept non-deterministic noise.
    if seed is not None:
        np.random.seed(seed)  # Set global seed as workaround
    
    ballot_matrix, wt_vec, _ = convert_pf_to_numpy_arrays(true_profile)
    
    BAL, CVR = bal_cvr_sample_constructor(
        bal_matrix=ballot_matrix,
        mult_vec=wt_vec,
        noise_level=noise_level,
        sample_size=sample_size,
        num_ghosts=num_ghosts
    )
    
    return BAL, CVR


def simulate_sequential_audit(
    true_profile: PreferenceProfile,
    reported_profile: PreferenceProfile,
    sample_indices: list[int],
    ballot_idx_to_true_type: dict[int, BallotType],
    prior: Literal["haldane", "laplace", "custom"] = "laplace",
    num_winners: int = 1,
    seed: Optional[int] = None
) -> AuditState:
    """
    Simulate a sequential audit given true and reported profiles.
    
    NOTE: This function is deprecated in favor of the simulate_audit_from_sample
    function which uses the more sophisticated bal_cvr_sample_constructor noise model.
    
    Args:
        true_profile: ground-truth PreferenceProfile.
        reported_profile: noised CVR PreferenceProfile.
        sample_indices: list of ballot indices to sample (in expanded ballot order).
        ballot_idx_to_true_type: mapping from ballot index to true BallotType.
        prior: prior choice for the audit.
        num_winners: number of seats.
        seed: random seed.
    
    Returns:
        AuditState after recording all sampled comparisons.
    """
    # Initialize audit state
    state = initialize_audit_state(reported_profile, prior=prior, num_winners=num_winners)
    
    # Convert reported profile to type indices
    reported_type_by_ballot, _ = profile_to_reported_types(reported_profile, state.type_to_idx)
    
    # Record each sampled ballot
    for ballot_idx in sample_indices:
        if ballot_idx < 0 or ballot_idx >= state.n:
            continue
        
        reported_type_idx = reported_type_by_ballot[ballot_idx]
        true_type = ballot_idx_to_true_type[ballot_idx]
        actual_type_idx = state.type_to_idx[true_type]
        
        record_comparison(state, reported_type_idx, actual_type_idx)
    
    return state


def simulate_audit_from_sample(
    CVR: NDArray,
    BAL: NDArray,
    candidates: list[str],
    prior: Literal["haldane", "laplace", "custom"] = "laplace",
    num_winners: int = 1,
    unseen_prior_mass: float = 1.0,
    audit_fraction: float = 1.0,
    seed: int | None = None
) -> AuditState:
    """
    Run a sequential audit from a CVR/BAL ballot comparison sample.
    
    This function takes CVR (reported) and BAL (true) ballot matrices from
    bal_cvr_sample_constructor and runs an audit, recording comparisons for
    a subset of ballots.
    
    Args:
        CVR: reported ballot matrix from bal_cvr_sample_constructor (sample_size x max_length)
        BAL: true ballot matrix from bal_cvr_sample_constructor (sample_size x max_length)
        candidates: list of candidate names (for converting indices to names)
        prior: prior choice for the audit ("haldane", "laplace", or "custom")
        num_winners: number of seats (m) in the election
        unseen_prior_mass: prior probability mass for unseen ballot types
        audit_fraction: fraction of ballots to audit (0 < audit_fraction <= 1.0)
                       Default 1.0 audits all ballots (for backward compatibility)
        seed: random seed for selecting which ballots to audit (if audit_fraction < 1.0)
    
    Returns:
        AuditState with comparisons recorded for audited ballots
    """
    # First, enumerate all ballot types present in CVR (including invalid ones)
    # This ensures R correctly reflects the CVR distribution
    cvr_types = []
    for i in range(CVR.shape[0]):
        bt = ballot_row_to_type(CVR[i, :])
        cvr_types.append(bt)
    
    # Enumerate unique types
    unique_types = sorted(set(cvr_types), key=lambda bt: bt.ranking)
    type_to_idx = {bt: i for i, bt in enumerate(unique_types)}
    idx_to_type = {i: bt for i, bt in enumerate(unique_types)}
    t = len(unique_types)
    
    # Build R from the CVR sample (including invalid ballots)
    R = np.zeros(t, dtype=np.int64)
    for bt in cvr_types:
        R[type_to_idx[bt]] += 1
    
    # Build a reported profile for determining the outcome
    # Filter to valid ballots only for the STV calculation
    from collections import defaultdict
    cvr_type_to_weight = defaultdict(float)
    for bt in cvr_types:
        if not bt.is_invalid():
            cvr_type_to_weight[bt] += 1.0
    
    # Build the reported profile from valid CVR ballots
    reported_ballots = []
    reported_weights = []
    for bt, wt in cvr_type_to_weight.items():
        ranking = []
        for cand_idx in bt.ranking:
            if 0 <= cand_idx < len(candidates):
                ranking.append(frozenset([candidates[cand_idx]]))
        if len(ranking) > 0:
            reported_ballots.append(Ballot(ranking=tuple(ranking)))
            reported_weights.append(wt)
    
    if len(reported_ballots) == 0:
        # All ballots are invalid; create a minimal valid profile
        reported_ballots = [Ballot(ranking=(frozenset([candidates[0]]),))]
        reported_weights = [1.0]
    
    reported_profile = PreferenceProfile(ballots=reported_ballots, candidates=candidates)
    reported_profile.df["Weight"] = reported_weights
    
    # Compute reported outcome
    ballot_matrix_for_stv, wt_vec_for_stv, _ = convert_pf_to_numpy_arrays(reported_profile)
    fast_stv = create_fast_stv(
        ballot_matrix=ballot_matrix_for_stv,
        wt_vec=wt_vec_for_stv,
        candidates=candidates,
        num_winners=num_winners,
        profile=reported_profile
    )
    reported_outcome = fast_stv.get_elected(no_fsets=False)
    
    # Initialize C and alpha matrices - includes column for unseen types
    C = np.zeros((t, t + 1), dtype=np.int64)
    
    if prior == "haldane":
        alpha = np.zeros((t, t + 1), dtype=np.float64)
        # Still give unseen types some prior mass
        alpha[:, t] = unseen_prior_mass
    elif prior == "laplace":
        alpha = np.ones((t, t + 1), dtype=np.float64)
        # Scale unseen column
        alpha[:, t] = unseen_prior_mass
    else:
        raise ValueError(f"Unsupported prior: {prior}")
    
    # Create the AuditState (not yet populated with comparisons)
    state = AuditState(
        t=t,
        n=len(CVR),  # Total sample size
        R=R,
        C=C,
        alpha=alpha,
        sample_size=0,
        type_to_idx=type_to_idx,
        idx_to_type=idx_to_type,
        reported_outcome=reported_outcome,
        candidates=candidates,
        num_winners=num_winners,
        unseen_prior_mass=unseen_prior_mass,
        ballot_matrix_template=CVR,  # Use CVR as template
        profile_for_stv=reported_profile
    )
    
    # Determine which ballots to audit based on audit_fraction
    num_to_audit = int(np.ceil(audit_fraction * CVR.shape[0]))
    
    if audit_fraction < 1.0:
        # Randomly select which ballots to audit
        rng = np.random.default_rng(seed)
        audited_indices = rng.choice(CVR.shape[0], size=num_to_audit, replace=False)
    else:
        # Audit all ballots
        audited_indices = np.arange(CVR.shape[0])
    
    # Now record comparisons only for audited ballots
    for i in audited_indices:
        cvr_type = ballot_row_to_type(CVR[i, :])
        bal_type = ballot_row_to_type(BAL[i, :])
        
        cvr_type_idx = type_to_idx[cvr_type]
        
        # Check if BAL type is in our CVR type system
        if bal_type in type_to_idx:
            # BAL type was also in CVR - record as normal
            bal_type_idx = type_to_idx[bal_type]
            record_comparison(state, cvr_type_idx, bal_type_idx)
        else:
            # BAL has a type not in CVR - record as "unseen type" (column t)
            # This is exactly what we added the unseen column for!
            state.C[cvr_type_idx, state.t] += 1
            state.sample_size += 1
    
    return state


# ============================================================================
# Basic Tests / Self-Checks
# ============================================================================

def run_basic_tests():
    """
    Run basic sanity checks on the audit machinery.
    
    This includes:
    1. Tiny toy contest with enumerable ballot types.
    2. Verify row totals and posterior draws have correct dimensions.
    3. Verify full-sample audit (n sampled) produces low upset probability for correct outcome.
    4. Verify sequential updates only change one row of C.
    """
    print("=" * 60)
    print("Running basic tests for bayesian_comparison module")
    print("=" * 60)
    
    # Test 1: Tiny toy contest
    print("\nTest 1: Tiny toy contest with 3 candidates, 2 seats")
    from votekit.ballot import Ballot
    
    # Create a simple profile
    ballots = [
        Ballot(ranking=(frozenset(["A"]), frozenset(["B"]), frozenset(["C"]))),
        Ballot(ranking=(frozenset(["B"]), frozenset(["A"]), frozenset(["C"]))),
        Ballot(ranking=(frozenset(["C"]), frozenset(["A"]), frozenset(["B"]))),
        Ballot(ranking=(frozenset(["A"]), frozenset(["C"]), frozenset(["B"]))),
        Ballot(ranking=(frozenset(["B"]), frozenset(["C"]), frozenset(["A"]))),
    ]
    candidates = ["A", "B", "C"]
    weights = [10, 8, 5, 7, 6]
    
    pf = PreferenceProfile(ballots=ballots, candidates=candidates)
    pf.df["Weight"] = weights
    
    state = initialize_audit_state(pf, prior="laplace", num_winners=2)
    
    print(f"  Number of ballot types: {state.t}")
    print(f"  Total ballots: {state.n}")
    print(f"  Reported outcome: {state.reported_outcome}")
    assert state.t >= 3, "Should have at least 3 ballot types (one per ballot pattern)"
    assert state.n == sum(weights), "Total ballots should match sum of weights"
    
    # Test 2: Verify posterior dimensions
    print("\nTest 2: Verify posterior simulation dimensions")
    rng = np.random.default_rng(42)
    
    row_tally = sample_row_posterior(state, 0, rng)
    assert row_tally.shape == (state.t + 1,), f"Row tally shape mismatch: {row_tally.shape}"
    
    full_tally = sample_full_election_tally(state, rng)
    assert full_tally.shape == (state.t + 1,), f"Full tally shape mismatch: {full_tally.shape}"
    assert np.isclose(np.sum(full_tally), state.n, atol=1e-6), "Full tally sum should equal n"
    
    print(f"  Row tally shape: {row_tally.shape} ✓")
    print(f"  Full tally shape: {full_tally.shape} ✓")
    print(f"  Full tally sum: {np.sum(full_tally):.2f} (expected {state.n}) ✓")
    
    # Test 3: Full-sample audit (zero errors)
    print("\nTest 3: Full-sample audit with zero errors should have low upset probability")
    
    # Simulate a full audit with zero errors
    ballot_matrix, wt_vec, _ = convert_pf_to_numpy_arrays(pf)
    wt_int = np.rint(wt_vec).astype(np.int64)
    expanded_matrix = np.repeat(ballot_matrix, wt_int, axis=0)
    
    reported_type_by_ballot, _ = profile_to_reported_types(pf, state.type_to_idx)
    
    # Sample all ballots with zero errors
    for i in range(state.n):
        reported_type_idx = reported_type_by_ballot[i]
        actual_type_idx = reported_type_idx  # No error
        record_comparison(state, reported_type_idx, actual_type_idx)
    
    results = run_posterior_simulations(state, num_simulations=100, seed=42)
    print(f"  Upset probability: {results['upset_probability']:.4f}")
    
    # With full sample and zero errors, upset probability should be very low
    # (may not be exactly zero due to simulation noise)
    assert results['upset_probability'] < 0.2, "Full zero-error audit should have low upset prob"
    
    # Test 4: Verify sequential updates
    print("\nTest 4: Verify sequential updates only change one row of C")
    
    # Create a fresh state
    state2 = initialize_audit_state(pf, prior="laplace", num_winners=2)
    C_before = state2.C.copy()
    
    record_comparison(state2, 0, 1)
    
    C_after = state2.C.copy()
    diff = C_after - C_before
    
    # Only row 0 should have changed
    assert diff[0, 1] == 1, "C[0, 1] should increase by 1"
    assert np.sum(diff) == 1, "Only one cell should change"
    
    print("  Sequential update test passed ✓")
    
    print("\nAll basic tests passed! ✓")
    print("=" * 60)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Run basic tests
    run_basic_tests()
    
    print("\n" + "=" * 60)
    print("Example: Synthetic audit with noise using bal_cvr_sample_constructor")
    print("=" * 60)
    
    # Create a toy election
    from votekit.ballot import Ballot
    
    ballots = [
        Ballot(ranking=(frozenset(["Alice"]), frozenset(["Bob"]), frozenset(["Carol"]))),
        Ballot(ranking=(frozenset(["Bob"]), frozenset(["Alice"]), frozenset(["Carol"]))),
        Ballot(ranking=(frozenset(["Carol"]), frozenset(["Alice"]), frozenset(["Bob"]))),
        Ballot(ranking=(frozenset(["Alice"]), frozenset(["Carol"]), frozenset(["Bob"]))),
    ]
    candidates = ["Alice", "Bob", "Carol"]
    weights = [100, 80, 60, 70]
    
    true_pf = PreferenceProfile(ballots=ballots, candidates=candidates)
    true_pf.df["Weight"] = weights
    
    print(f"\nTrue profile: {sum(weights)} ballots, {len(ballots)} types")
    
    # Generate noised sample using the existing noise infrastructure
    sample_size = 50
    BAL, CVR = generate_noised_sample(
        true_pf, 
        sample_size=sample_size,
        noise_level=0.02, 
        num_ghosts=10,
        seed=123
    )
    
    print(f"Noised sample generated:")
    print(f"  Sample size: {sample_size}")
    print(f"  Noise level: 0.02")
    print(f"  Ghost ballots: 10")
    
    # Run audit from the sample
    state = simulate_audit_from_sample(
        CVR=CVR,
        BAL=BAL,
        candidates=candidates,
        prior="laplace",
        num_winners=1
    )
    
    print(f"\nAudit state initialized from sample:")
    print(f"  t={state.t} ballot types")
    print(f"  Sampled {state.sample_size} ballots")
    print(f"  Reported outcome: {state.reported_outcome}")
    
    # Count discrepancies
    num_discrepant = np.sum(np.any(BAL != CVR, axis=1))
    print(f"  Discrepancies in sample: {num_discrepant}/{sample_size}")
    
    # Run posterior simulations
    can_stop, results = sequential_stopping_check(state, risk_limit=0.05, num_simulations=500, seed=789)
    
    print(f"\nPosterior simulation results:")
    print(f"  Upset probability: {results['upset_probability']:.4f}")
    print(f"  Can stop at risk_limit=0.05? {can_stop}")
    print(f"  Winner set counts: {len(results['winner_set_counts'])} unique outcomes")
    
    print("\n" + "=" * 60)

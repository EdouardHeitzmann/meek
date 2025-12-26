import numpy as np
import math
from src.edouard.permutations import update_perm_idx_vectorized

def project_matrix_onto_election_state(
    sample_matrix,
    num_cands,
    m,
    fpv_vec = None,
    winner_comb_vec = None,
    winner_to_cand = [],
    losers = []):
    num_ballots = sample_matrix.shape[0]
    cand_to_winner = np.zeros(num_cands, dtype=np.int8) -1
    for idx, cand in enumerate(winner_to_cand):
        cand_to_winner[cand] = idx
    pos_vec = np.zeros(num_ballots, dtype=np.int8)
    if fpv_vec is None:
        fpv_vec = sample_matrix[np.arange(sample_matrix.shape[0]), pos_vec]
    if winner_comb_vec is None:
        winner_comb_vec = np.zeros(num_ballots)
    bool_ballot_matrix = sample_matrix != -127
    if len(losers)>0:
        bool_ballot_matrix &= ~np.isin(sample_matrix, losers)
        needs_update = np.isin(fpv_vec, losers)
        pos_vec[needs_update] = np.argmax(bool_ballot_matrix[needs_update, :], axis = 1)
        fpv_vec[needs_update] = sample_matrix[needs_update, pos_vec[needs_update]]
    for _ in range(len(winner_to_cand)):
        needs_update = np.isin(fpv_vec, winner_to_cand)
        winner_comb_vec[needs_update] = update_perm_idx_vectorized(
            winner_comb_vec[needs_update],
            cand_to_winner[fpv_vec[needs_update]],
            m+1
        )
        bool_ballot_matrix[needs_update, pos_vec[needs_update]] = False
        pos_vec[needs_update] = np.argmax(bool_ballot_matrix[needs_update,:], axis = 1)
        fpv_vec[needs_update] = sample_matrix[needs_update, pos_vec[needs_update]]
    return fpv_vec, winner_comb_vec

def log_comb(n: int, k: int) -> float:
    """Logarithm of binomial coefficient C(n, k) using log-gamma."""
    if k < 0 or k > n:
        return float('-inf')  # log(0)
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def hypergeom_log_p_zero(N: int, K: int, n: int) -> float:
    """
    log P(T = 0 | population size N, K nonzeros, sample size n),
    where T is the number of nonzeros observed in a sample without replacement.
    """
    # P(T=0 | K) = C(N-K, n) / C(N, n)
    if K > N - n:  # then C(N-K, n) = 0, so probability is 0
        return float('-inf')
    return log_comb(N - K, n) - log_comb(N, n)

def K_upper(N: int, n: int, alpha: float = 0.05) -> int:
    """
    One-sided (1 - alpha) upper confidence bound on K
    given that we observed T = 0 nonzeros in a simple random sample
    without replacement of size n from a population of size N.

    Finds the largest integer K such that:
        P(T = 0 | K) >= alpha
    i.e.
        C(N-K, n) / C(N, n) >= alpha.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if not (1 <= n <= N):
        raise ValueError("Need 1 <= n <= N.")

    log_alpha = math.log(alpha)

    # If K = 0, probability is 1; if K > N-n, probability is 0.
    # So K is in [0, N-n] when alpha > 0.
    lo, hi = 0, N - n
    best = 0  # at least K=0 works

    while lo <= hi:
        mid = (lo + hi) // 2
        log_p0 = hypergeom_log_p_zero(N, mid, n)
        if log_p0 >= log_alpha:
            # T=0 is still not too rare; we can increase K
            best = mid
            lo = mid + 1
        else:
            # T=0 too unlikely; decrease K
            hi = mid - 1

    return best

def deg0_cand_to_cand(
        i,
        j, # note: j should always be the recorded loser
        K_upper,
        fpv_scores,
        delta_vecs,
        N,
        z_alpha, # corresponds to one-sided interval
        verbose=False
        ):
    T_ij = fpv_scores[i]-fpv_scores[j]
    d_ij = delta_vecs[i].astype(int) - delta_vecs[j].astype(int)
    if verbose:
        print(f"d_ij for candidate {i} vs Candidate {j}: {d_ij}")
        print(f"Variance of d_ij: {np.var(d_ij)}")
        print(f"T_ij: {T_ij}")
    if np.all(d_ij == 0):
        sample_variance = 4*K_upper/N
    else:
        sample_variance = np.var(d_ij)
    fpc = (N - len(d_ij)) / (N - 1)
    st_dev = math.sqrt(sample_variance * fpc)
    M_lower = T_ij - N* z_alpha * st_dev
    if verbose:
        print(f"Candidate {i} vs Candidate {j}: M_lower = {M_lower}")
    if M_lower > 0:
        return True
    else:
        return False
    
def deg0_cand_to_quota(
        i,
        m,
        K_upper,
        fpv_scores,
        g,
        delta_vecs,
        N,
        z_alpha, # corresponds to one-sided interval
        epsilon=1e-6,
        verbose=False
        ):
    T_0 = fpv_scores[i] - (N - g)/(m+1) - epsilon
    d_iq = delta_vecs[i].astype(int) + delta_vecs[-127].astype(int)/(m+1)
    if np.all(d_iq == 0):
        sample_variance = 4*K_upper/N
    else:
        sample_variance = np.var(d_iq)
    fpc = (N - len(d_iq)) / (N - 1)
    st_dev = math.sqrt(sample_variance * fpc)
    M_upper = T_0 + N* z_alpha * st_dev
    if verbose:
        print(f"Candidate {i} to quota: M_upper = {M_upper}")
    if M_upper < 0:
        return True
    else:
        return False
    
def log_comb(n, k):
    if k < 0 or k > n:
        return float('-inf')
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def hypergeom_log_cdf_leq(N: int, K: int, n: int, t: int) -> float:
    """
    log P(T <= t | population size N, K nonzeros, sample size n),
    where T is the number of nonzeros observed in a sample without replacement.

    T ~ Hypergeom(N, K, n).
    """
    # Feasible range of T for this (N, K, n):
    #   max(0, n - (N - K)) <= T <= min(n, K)
    t_min = max(0, n - (N - K))
    t_max = min(n, K)

    if t < t_min:
        # Then P(T <= t) = 0
        return float('-inf')

    # We only sum over j in [t_min, min(t, t_max)].
    upper = min(t, t_max)
    if upper < t_min:
        return float('-inf')

    # Compute numerator: sum_{j=t_min}^{upper} C(K, j) C(N-K, n-j)
    # Denominator is C(N, n).
    log_terms = []
    for j in range(t_min, upper + 1):
        log_terms.append(log_comb(K, j) + log_comb(N - K, n - j))

    # log-sum-exp to stay stable
    m = max(log_terms)
    log_num = m + math.log(sum(math.exp(x - m) for x in log_terms))
    log_den = log_comb(N, n)

    return log_num - log_den


def alternative_K_upper(_N: int, _n: int, _t: int, alpha: float = 0.05) -> int:
    """
    One-sided (1 - alpha) upper confidence bound on K
    given that we observed T = t nonzeros in a simple random sample
    without replacement of size n from a population of size N.

    We find the largest integer K (number of nonzeros in the population) such that:
        P(T <= t | N, K, n) >= alpha,
    where T ~ Hypergeom(N, K, n).

    For t = 0, this reduces to your original K_upper() definition.
    """
    _n = int(_n)
    _N = int(_N)
    _t = int(_t)
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if not (1 <= _n <= _N):
        raise ValueError("Need 1 <= n <= N.")
    if not (0 <= _t <= _n):
        raise ValueError(f"Need 0 <= t <= n, got t={_t}, n={_n}.")

    log_alpha = math.log(alpha)

    # Logical bounds for K:
    # - Must have K >= t (can't observe more successes than exist in population).
    # - Must have enough failures to allow at least n - t failures in the sample:
    #       N - K >= n - t  =>  K <= N - (n - t)
    lo = _t
    hi = _N - (_n - _t)
    if hi < lo:
        # Observed t is impossible for any K in [0, N] with this (N, n),
        # but that's a pathological case. We just return t.
        return _t

    best = lo  # at least K = t is always logically possible

    while lo <= hi:
        mid = (lo + hi) // 2
        log_cdf = hypergeom_log_cdf_leq(_N, mid, _n, _t)
        if log_cdf >= log_alpha:
            # Observing T <= t is still not too rare; we can increase K
            best = mid
            lo = mid + 1
        else:
            # T <= t would be too unlikely; decrease K
            hi = mid - 1

    return best
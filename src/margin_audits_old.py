from statistics import NormalDist
import numpy as np
import math

def mu_upper_from_nonzero_indicator(x_sample, N: int, alpha: float = 0.05,
                                    prior_q_max: float | None = None) -> float:
    """
    X in {-2,-1,0,1,2}. Let q = Pr(|X|>0). For any distribution, E[X] <= 2 q.
    Uses Wilson+FPC UB for q (or min with prior_q_max if provided).
    """
    xs = list(x_sample); n = len(xs)
    qhat = sum(1 for x in xs if x != 0) / n
    q_ub = _wilson_upper(qhat, n, N, alpha)
    if prior_q_max is not None:
        q_ub = min(q_ub, float(prior_q_max))
    return min(1.0, max(-1.0, 2.0 * q_ub))  # mu in [-2,2] so mu<=2*q<=2, divide by 2? No: mu<=2*q; then clip to [0,2]; but as a mean of X it's in [-2,2]. Here we return mu upper, so clip to 2 via min(2.0, 2*q_ub). For safety, clamp to [-2,2].

def _wilson_upper(phat: float, n: int, N: int, alpha: float) -> float:
    """
    One-sided Wilson UPPER bound for a binomial proportion under SRSWOR via
    FPC-adjusted score test. Returns UB at level 1-alpha.
    """
    phat = min(1.0, max(0.0, phat))
    z = NormalDist().inv_cdf(1 - alpha)
    a = (z*z) * (1.0 - n / N) / n  # FPC-adjusted z^2/n
    center = phat + 0.5 * a
    rad = math.sqrt(a * phat * (1.0 - phat) + 0.25 * a * a)
    denom = 1.0 + a
    ub = (center + rad) / denom
    return min(1.0, max(0.0, ub))

def cand_to_quota_deg0(deg0_samples, i, sentinel, Ti, Tg, N, m, n, epsilon=1e-6, alpha = 0.05, noise_level=0.05):
    SRSWOR_adjustment = (1 - n/N)/n
    C = Ti- epsilon - (N-Tg)/(m+1)
    M_iq_samples = deg0_samples[i] - deg0_samples[sentinel]/(m+1)
    var_iq = np.var(M_iq_samples, ddof=1)
    # if the variance is zero, use the known upper bound on variance
    if var_iq == 0:
        mu_iq_upper = mu_upper_from_nonzero_indicator(
            M_iq_samples, N, alpha, prior_q_max=1.0 - noise_level
        )
        print("Using nonzero indicator mu upper:", mu_iq_upper)
        return C + N*mu_iq_upper
    var_iq *= SRSWOR_adjustment
    mean_iq = np.mean(M_iq_samples)
    z = NormalDist().inv_cdf(1 - alpha/2)
    mu_iq_upper = mean_iq + z * np.sqrt(var_iq)
    mu_iq_lower = mean_iq - z * np.sqrt(var_iq)
    upper_bound = C + N*mu_iq_upper
    lower_bound = C + N*mu_iq_lower
    return lower_bound, upper_bound

def cand_to_cand_deg0(deg0_samples, i, j, Ti, Tj, N, n, alpha = 0.05, noise_level=0.05):
    """
    Audit the upper bound on M_ij = T_i - T_j at degree 0.
    """
    SRSWOR_adjustment = (1 - n/N)/n
    M_ij_samples = deg0_samples[i] - deg0_samples[j]
    var_ij = np.var(M_ij_samples, ddof=1)
    # if the variance is zero, use the known upper bound on variance
    if var_ij == 0:
        mu_ij_upper = mu_upper_from_nonzero_indicator(
            M_ij_samples, N, alpha, prior_q_max=1.0 - noise_level
        )
        print("Using nonzero indicator mu upper:", mu_ij_upper)
        return (Ti - Tj) + N*mu_ij_upper
    var_ij *= SRSWOR_adjustment
    mean_ij = np.mean(M_ij_samples)
    z = NormalDist().inv_cdf(1 - alpha)
    mu_ij_upper = mean_ij + z * np.sqrt(var_ij)
    upper_bound = (Ti - Tj) + N*mu_ij_upper
    return upper_bound

def cand_to_cand_deg1(deg0_samples, deg1_samples, w, i, j, Ti, Tj, Twi, Twj, Tg, Tw, Twg, N, m, n,epsilon=1e-6, alpha = 0.05, noise_level=0.05,sentinel=-1):
    SRSWOR_adjustment = (1 - n/N)/n
    C0 = Ti - Tj
    C1 = Twi - Twj
    Cu=(N-(Twg+Tg)+(m+1)*epsilon)/N
    Cv = ((m+1)*Tw-Twg)/N

    mu_u_sample = deg1_samples[sentinel] + deg0_samples[sentinel]
    mu_v_sample = (m+1)*deg0_samples[w] - deg1_samples[sentinel]
    mu0_sample = deg0_samples[i] - deg0_samples[j]
    mu1_sample = deg1_samples[i] - deg1_samples[j]

    mu_u = np.mean(mu_u_sample)
    mu_v = np.mean(mu_v_sample)
    mu0 = np.mean(mu0_sample)
    mu1 = np.mean(mu1_sample)

    s_uu = np.var(mu_u_sample, ddof=1)
    s_vv = np.var(mu_v_sample, ddof=1)
    s_uv = np.cov(mu_u_sample, mu_v_sample, ddof=1)[0][1]
    s_0u = np.cov(mu0_sample, mu_u_sample, ddof=1)[0][1]
    s_0v = np.cov(mu0_sample, mu_v_sample, ddof=1)[0][1]
    s_1u = np.cov(mu1_sample, mu_u_sample, ddof=1)[0][1]
    s_1v = np.cov(mu1_sample, mu_v_sample, ddof=1)[0][1]
    s_00 = np.var(mu0_sample, ddof=1)
    s_11 = np.var(mu1_sample, ddof=1)
    s_01 = np.cov(mu0_sample, mu1_sample, ddof=1)[0][1]

    S_theta = np.array([[s_uu, s_uv, s_0u, s_0v],
                        [s_uv, s_vv, s_0v, s_1v],
                        [s_0u, s_0v, s_00, s_01],
                        [s_1u, s_1v, s_01, s_11]])
    S_theta *= SRSWOR_adjustment

    k_hat = (Cu-mu_u)/(Cv + mu_v)

    grad_T = np.array([(C1+N * mu1)/(Cv + mu_v), k_hat*(C1+ N * mu1)/(Cv + mu_v),N, (1-k_hat)*N])
    var_T = grad_T @ S_theta @ grad_T.T
    SE_T = math.sqrt(var_T)

    if var_T == 0:
        mu0_upper = mu_upper_from_nonzero_indicator(
            mu0_sample, N, alpha, prior_q_max=1.0 - noise_level
        )
        mu1_upper = mu_upper_from_nonzero_indicator(
            mu1_sample, N, alpha, prior_q_max=1.0 - noise_level
        )
        print("Using nonzero indicator mu upper:", mu0_upper, mu1_upper)
        upper_bound = C0 + (1 - k_hat) * C1 + N * mu0_upper + (1 - k_hat) * N * mu1_upper
        return upper_bound

    z = NormalDist().inv_cdf(1 - alpha)
    upper_bound = C0 + (1 - k_hat) * C1 + N * mu0 + (1 - k_hat) * N * mu1 + z * SE_T
    return upper_bound

def cand_to_quota_deg1(deg0_samples, deg1_samples, w, i, Ti, Twi, Tg, Tw, Twg, N, m, n,epsilon=1e-6, alpha = 0.05, noise_level=0.05,sentinel=-1):
    SRSWOR_adjustment = (1 - n/N)/n
    C0 = Ti + Twi
    C1 = Twi + Tw
    Cu=(N-(Twg+Tg)+(m+1)*epsilon)/N
    Cv = ((m+1)*Tw-Twg)/N

    mu_u_sample = deg1_samples[sentinel] + deg0_samples[sentinel]
    mu_v_sample = (m+1)*deg0_samples[w] - deg1_samples[sentinel]
    mu0_sample = deg0_samples[i] + deg1_samples[i]
    mu1_sample = deg1_samples[i] + deg0_samples[w]

    mu_u = np.mean(mu_u_sample)
    mu_v = np.mean(mu_v_sample)
    mu0 = np.mean(mu0_sample)
    mu1 = np.mean(mu1_sample)

    s_uu = np.var(mu_u_sample, ddof=1)
    s_vv = np.var(mu_v_sample, ddof=1)
    s_uv = np.cov(mu_u_sample, mu_v_sample, ddof=1)[0][1]
    s_0u = np.cov(mu0_sample, mu_u_sample, ddof=1)[0][1]
    s_0v = np.cov(mu0_sample, mu_v_sample, ddof=1)[0][1]
    s_1u = np.cov(mu1_sample, mu_u_sample, ddof=1)[0][1]
    s_1v = np.cov(mu1_sample, mu_v_sample, ddof=1)[0][1]
    s_00 = np.var(mu0_sample, ddof=1)
    s_11 = np.var(mu1_sample, ddof=1)
    s_01 = np.cov(mu0_sample, mu1_sample, ddof=1)[0][1]

    S_theta = np.array([[s_uu, s_uv, s_0u, s_0v],
                        [s_uv, s_vv, s_0v, s_1v],
                        [s_0u, s_0v, s_00, s_01],
                        [s_1u, s_1v, s_01, s_11]])
    S_theta *= SRSWOR_adjustment

    k_hat = (Cu-mu_u)/(Cv + mu_v)

    grad_T = np.array([(C1 + N * mu1)/(Cv + mu_v), k_hat*(C1+ N * mu1)/(Cv + mu_v),N, -N*k_hat])
    var_T = grad_T @ S_theta @ grad_T.T
    SE_T = math.sqrt(var_T)

    if var_T == 0:
        mu0_upper = mu_upper_from_nonzero_indicator(
            mu0_sample, N, alpha, prior_q_max=1.0 - noise_level
        )
        mu1_upper = mu_upper_from_nonzero_indicator(
            mu1_sample, N, alpha, prior_q_max=1.0 - noise_level
        )
        print("Using nonzero indicator mu upper:", mu0_upper, mu1_upper)
        upper_bound = C0 + (1 - k_hat) * C1 + N * mu0_upper + (1 - k_hat) * N * mu1_upper
        return upper_bound

    z = NormalDist().inv_cdf(1 - alpha)
    upper_bound = C0 + N * mu0 + - k_hat * (C1 + N * mu1) + z * SE_T
    return upper_bound

def project_sample(sample_array, elected, hopeful, exhaust_sentinel=-1):
    """
    Project the BAL and CVR samples onto the space defined by the elected and hopeful candidates.
    Also compacts non-sentinel entries to the front of each row.
    """
    # mask all entries in the samples that are not in elected or hopeful with a -1
    valid_candidates = elected.union(hopeful)
    valid_candidates = np.array(list(valid_candidates), dtype=np.int8)
    projected_sample = np.where(np.isin(sample_array, valid_candidates), sample_array, exhaust_sentinel)
    
    # Compact non-sentinel entries to the front of each row
    compacted_sample = np.full_like(projected_sample, exhaust_sentinel)
    
    for i in range(projected_sample.shape[0]):
        row = projected_sample[i]
        non_sentinel_mask = row != exhaust_sentinel
        non_sentinel_values = row[non_sentinel_mask]
        compacted_sample[i, :len(non_sentinel_values)] = non_sentinel_values
    
    return compacted_sample

def deg0_sampler(_BAL, _CVR, hopeful, exhaust_sentinel=-1):
    hopeful_with_sentinel = hopeful.union({exhaust_sentinel})
    projected_BAL = project_sample(_BAL, elected=set(), hopeful=hopeful, exhaust_sentinel=exhaust_sentinel)
    projected_CVR = project_sample(_CVR, elected=set(), hopeful=hopeful, exhaust_sentinel=exhaust_sentinel)
    BAL_FPV = projected_BAL[:, 0]
    CVR_FPV = projected_CVR[:, 0]
    discrepant_mask = BAL_FPV != CVR_FPV
    discrepant_idx = np.where(discrepant_mask)[0]
    
    pi_samples = {i: np.zeros(len(BAL_FPV), dtype=np.int8) for i in hopeful_with_sentinel}
    
    for candidate in hopeful_with_sentinel:
        # Add 1 where BAL has this candidate and there's a discrepancy
        pi_samples[candidate][discrepant_idx] += (BAL_FPV[discrepant_idx] == candidate).astype(np.int8)
        # Subtract 1 where CVR has this candidate and there's a discrepancy
        pi_samples[candidate][discrepant_idx] -= (CVR_FPV[discrepant_idx] == candidate).astype(np.int8)
    
    return pi_samples

def deg1_sampler(_BAL, _CVR, hopeful, w, exhaust_sentinel=-1):
    hopeful_with_sentinel = hopeful.union({exhaust_sentinel})
    projected_BAL = project_sample(_BAL, elected={w}, hopeful=hopeful, exhaust_sentinel=exhaust_sentinel)
    projected_CVR = project_sample(_CVR, elected={w}, hopeful=hopeful, exhaust_sentinel=exhaust_sentinel)
    BAL_FPV = projected_BAL[:, 0]
    CVR_FPV = projected_CVR[:, 0]
    BAL_SPV = projected_BAL[:, 1]
    CVR_SPV = projected_CVR[:, 1]
    winner_fpv_mask = (BAL_FPV == w) & (CVR_FPV == w)
    discrepant_mask = (BAL_SPV != CVR_SPV) & winner_fpv_mask
    discrepant_idx = np.where(discrepant_mask)[0]

    pi_samples = {i: np.zeros(len(BAL_FPV), dtype=np.int8) for i in hopeful_with_sentinel}

    for candidate in hopeful_with_sentinel:
        # Add 1 where BAL has this candidate and there's a discrepancy
        pi_samples[candidate][discrepant_idx] += (BAL_SPV[discrepant_idx] == candidate).astype(np.int8)
        # Subtract 1 where CVR has this candidate and there's a discrepancy
        pi_samples[candidate][discrepant_idx] -= (CVR_SPV[discrepant_idx] == candidate).astype(np.int8)

    return pi_samples
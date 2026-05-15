from statistics import NormalDist
from ..audit_machinery import project_matrix_onto_election_state, K_upper, alternative_K_upper
from ..utils import convert_pf_to_numpy_arrays
from ..noise import bal_cvr_sample_constructor
from ..ranking_election.meek import MeekCore
import numpy as np
import sympy as sp
import math

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

def deg0_node_audit(str_node_ID, graph, deg0_kwargs, verbose = False):
    node_info = graph.lookup_node(str_node_ID)
    depth = node_info['layer']
    profile = deg0_kwargs['profile']
    canonical_loser = node_info.get('canonical_loser', None)
    next_layer_codes = graph.nodes_by_layer[depth + 1]
    allowable_losers, allowable_winners = [], []
    for code in next_layer_codes:
        decoded = graph.decode_node_string(code)
        if set(decoded["winner_to_cand"]) >= set(node_info["winner_to_cand"]) and set(decoded["initial_losers"]) >= set(node_info["initial_losers"]):
            if len(decoded["winner_to_cand"]) == len(node_info["winner_to_cand"]) + 1:
                new_winner = list(set(decoded["winner_to_cand"]) - set(node_info["winner_to_cand"]))
                allowable_winners.append(new_winner[0])
            elif len(decoded["initial_losers"]) == len(node_info["initial_losers"]) + 1:
                new_loser = list(set(decoded["initial_losers"]) - set(node_info["initial_losers"]))
                allowable_losers.append(new_loser[0])
    projected_BAL_fpv, _ = project_matrix_onto_election_state(
        deg0_kwargs['BAL'],
        num_cands=len(profile.candidates),
        m=deg0_kwargs['m'],
        losers=node_info['initial_losers'])
    projected_CVR_fpv, _ = project_matrix_onto_election_state(
        deg0_kwargs['CVR'],
        num_cands=len(profile.candidates),
        m=deg0_kwargs['m'],
        losers=node_info['initial_losers'])
    core = MeekCore(
        profile=deg0_kwargs['profile'],
        m=deg0_kwargs['m'],
        candidates=list(deg0_kwargs['profile'].candidates),
        initial_losers=node_info['initial_losers']
    )
    fpv_scores, helper_vecs, play, tiebreak, quota = core._run_first_round()
    if verbose:
        print(f"First round results at node {str_node_ID}:")
        print(f"FPV scores: {fpv_scores}")
        print(f"Quota: {quota}")
        print(f"Initial losers: {node_info['initial_losers']}")
        print(f"Initial winners: {node_info['winner_to_cand']}")
        print(f"Canonical loser: {canonical_loser}")
        print(f"Allowable winners: {allowable_winners}")
        print(f"Allowable losers: {allowable_losers}")
    fpc = (deg0_kwargs['N'] - deg0_kwargs['n']) / (deg0_kwargs['n'] * (deg0_kwargs['N'] - 1)) #lol
    g = deg0_kwargs['N']- sum(fpv_scores)
    #print(fpv_scores)
    delta_vecs = {i: (projected_BAL_fpv == i).astype(int) - (projected_CVR_fpv == i).astype(int) for i in range(len(profile.candidates))}
    delta_vecs[-127] = (projected_BAL_fpv < 0).astype(int) - (projected_CVR_fpv < 0).astype(int)
    node_succesfully_audited = True
    hopeful_candidates = [c for c in range(len(profile.candidates)) if c not in node_info['initial_losers'] and c not in node_info['winner_to_cand']]
    if node_info.get('canonical_winner', None) is not None:
        canonical_winner = node_info['canonical_winner']
        # just do a cand-to-quota audit for the canonical winner
        T_0 = fpv_scores[canonical_winner] - (deg0_kwargs['N'] - g)/(deg0_kwargs['m']+1) - deg0_kwargs['epsilon']
        d_iq = delta_vecs[canonical_winner].astype(int) + delta_vecs[-127].astype(int)/(deg0_kwargs['m']+1)
        mean_diq = np.mean(d_iq)
        sample_variance = np.var(d_iq, ddof=1) if np.any(d_iq != 0) else 4 * deg0_kwargs['K'] / deg0_kwargs['N']
        st_dev = math.sqrt(sample_variance * fpc)
        M_lower = T_0 + deg0_kwargs['N'] * (mean_diq + deg0_kwargs['z'] * st_dev)
        if M_lower < 0:
            print(f"AUDIT FAILED at node {str_node_ID}: canonical winner {profile.candidates[canonical_winner]} might not have quota, M_cq > {M_lower} and T_cq = {T_0}")
            print(M_lower)
            node_succesfully_audited = False
        elif verbose:
            print(f"Succesfully checked that candidate {profile.candidates[canonical_winner]} must have quota at node {str_node_ID}, M_cq > {M_lower} and T_cq = {T_0}")
        return node_succesfully_audited
    for c in hopeful_candidates:
        if c not in allowable_winners:
            T_0 = fpv_scores[c] - (deg0_kwargs['N'] - g)/(deg0_kwargs['m']+1) - deg0_kwargs['epsilon']
            d_iq = delta_vecs[c].astype(int) + delta_vecs[-127].astype(int)/(deg0_kwargs['m']+1)
            mean_diq = np.mean(d_iq)
            sample_variance = np.var(d_iq, ddof=1) if np.any(d_iq != 0) else 4 * deg0_kwargs['K'] / deg0_kwargs['N']
            st_dev = math.sqrt(sample_variance * fpc)
            M_upper = T_0 + deg0_kwargs['N'] * (mean_diq + deg0_kwargs['z'] * st_dev)
            if M_upper > 0:
                print(f"AUDIT FAILED at node {str_node_ID}: candidate {profile.candidates[c]} might have quota.")
                print(M_upper)
                node_succesfully_audited = False
            elif verbose:
                print(f"Succesfully checked that candidate {profile.candidates[c]} does not have quota at node {str_node_ID}, M_cq < {M_upper} and T_cq = {T_0}")
        if c not in allowable_losers: # check the M_{cl} is positive
            deltas = delta_vecs[c] - delta_vecs[canonical_loser]
            delta_mean = np.mean(deltas)
            delta_sample_var = np.var(deltas, ddof=1) if np.any(deltas != 0) else 4 * deg0_kwargs['K'] / deg0_kwargs['N']
            st_dev = math.sqrt(delta_sample_var * fpc)
            T_cl = fpv_scores[c] - fpv_scores[canonical_loser]
            M_lower = T_cl + deg0_kwargs['N'] * (delta_mean - deg0_kwargs['z'] * st_dev)
            if M_lower <= 0:
                print(f"AUDIT FAILED at node {str_node_ID}: candidate {profile.candidates[c]} might lose to {profile.candidates[canonical_loser]}: M_cl < {M_lower} and T_cl = {T_cl}")
                print(M_lower)
                node_succesfully_audited = False
            elif verbose:
                print(f"Succesfully checked that candidate {profile.candidates[c]} does not lose to {profile.candidates[canonical_loser]} at node {str_node_ID}, M_cl > {M_lower} and T_cl = {T_cl}")
    return node_succesfully_audited

def deg1_node_audit(str_node_ID, graph, deg1_kwargs, verbose = False):
    N = deg1_kwargs['N']
    m = deg1_kwargs['m']
    BAL = deg1_kwargs['BAL']
    CVR = deg1_kwargs['CVR']
    epsilon = deg1_kwargs['epsilon']
    K = deg1_kwargs['K']
    n = deg1_kwargs['n']
    z = deg1_kwargs['z']
    num_ghosts = deg1_kwargs['num_ghosts']
    profile = deg1_kwargs['profile']

    node_info = graph.lookup_node(str_node_ID)
    print(node_info)
    depth = node_info['layer']
    canonical_loser = node_info.get('canonical_loser', None)
    if canonical_loser is None:
        raise ValueError("deg1_node_audit requires a canonical_loser in node_info")
    next_layer_codes = graph.nodes_by_layer[depth + 1]
    allowable_losers, allowable_winners = [], []
    for code in next_layer_codes:
        decoded = graph.decode_node_string(code)
        if set(decoded["winner_to_cand"]) >= set(node_info["winner_to_cand"]) and set(decoded["initial_losers"]) >= set(node_info["initial_losers"]):
            if len(decoded["winner_to_cand"]) == len(node_info["winner_to_cand"]) + 1:
                new_winner = list(set(decoded["winner_to_cand"]) - set(node_info["winner_to_cand"]))
                allowable_winners.append(new_winner[0])
            elif len(decoded["initial_losers"]) == len(node_info["initial_losers"]) + 1:
                new_loser = list(set(decoded["initial_losers"]) - set(node_info["initial_losers"]))
                allowable_losers.append(new_loser[0])
    projected_BAL_fpv, BAL_winner_comb = project_matrix_onto_election_state(
        BAL,
        num_cands=len(profile.candidates),
        m=m,
        losers=node_info['initial_losers'],
        winner_to_cand=node_info['winner_to_cand'])
    projected_CVR_fpv, CVR_winner_comb = project_matrix_onto_election_state(
        CVR,
        num_cands=len(profile.candidates),
        m=m,
        losers=node_info['initial_losers'],
        winner_to_cand=node_info['winner_to_cand'])
    #core = MeekCore(
    #    profile=deg1_kwargs['profile'],
    #    m=m,
    #    candidates=list(deg1_kwargs['profile'].candidates),
    #    initial_losers=node_info['initial_losers'],
    #    winner_to_cand=node_info['winner_to_cand']
    #)
    helper_fpv_vec, helper_winner_comb = project_matrix_onto_election_state(deg1_kwargs['ballot_matrix'], num_cands=len(profile.candidates), m=m, losers=node_info['initial_losers'], winner_to_cand=node_info['winner_to_cand'])

    fpc = (N - n) / (n * (N- 1))
    mult_vec = deg1_kwargs['mult_vec']
    g = np.sum(mult_vec[(helper_fpv_vec < 0) & (helper_winner_comb == 0)]) + num_ghosts

    deg0_scores = {i: sum(mult_vec[(helper_fpv_vec == i) & (helper_winner_comb == 0)]) for i in range(len(profile.candidates))}
    deg0_scores[-1] = sum(mult_vec[(helper_fpv_vec < 0) & (helper_winner_comb == 0)])
    deg1_scores = {i: sum(mult_vec[(helper_fpv_vec == i) & (helper_winner_comb == 1)]) for i in range(len(profile.candidates))}
    deg1_scores[-1] = sum(mult_vec[(helper_fpv_vec < 0) & (helper_winner_comb == 1)])

    Tw = sum(mult_vec[helper_winner_comb == 1])
    print(f"Tw: {Tw}, g: {g}")

    deg0_deltas = {i: ((projected_BAL_fpv == i) & (BAL_winner_comb == 0)).astype(int) - ((projected_CVR_fpv == i) & (CVR_winner_comb == 0)).astype(int) for i in range(len(profile.candidates))}
    deg0_deltas[-1] = ((projected_BAL_fpv < 0) & (BAL_winner_comb == 0)).astype(int) - ((projected_CVR_fpv < 0) & (CVR_winner_comb == 0)).astype(int)
    deg1_deltas = {i: ((projected_BAL_fpv == i) & (BAL_winner_comb == 1)).astype(int) - ((projected_CVR_fpv == i) & (CVR_winner_comb == 1)).astype(int) for i in range(len(profile.candidates))}
    deg1_deltas[-1] = ((projected_BAL_fpv < 0) & (BAL_winner_comb == 1)).astype(int) - ((projected_CVR_fpv < 0) & (CVR_winner_comb == 1)).astype(int)
    w_deltas = (BAL_winner_comb.astype(int) - CVR_winner_comb.astype(int))

    node_succesfully_audited = True
    hopeful_candidates = [c for c in range(len(profile.candidates)) if c not in node_info['initial_losers'] and c not in node_info['winner_to_cand']]

    Cu = N - g - deg1_scores[-1] + (m+1)*epsilon
    Cv = (m+1)*Tw - deg1_scores[-1]

    du_sample = deg0_deltas[-1].astype(int) + deg1_deltas[-1].astype(int)
    dv_sample = (m+1)*w_deltas - deg1_deltas[-1].astype(int)
    mu_u = np.mean(du_sample)
    mu_v = np.mean(dv_sample)
    k_hat = (Cu-N*mu_u)/(Cv+N*mu_v)

    if verbose:
        print(f"info at node {str_node_ID}: deg0_scores= {deg0_scores}, deg1_scores= {deg1_scores}, Tw = {Tw}, g = {g}, k_hat= {k_hat}")
        print(f"Cu= {Cu}, Cv= {Cv}, mu_u= {mu_u}, mu_v= {mu_v}")

    if node_info.get('canonical_winner', None) is not None:
        canonical_winner = node_info['canonical_winner']
        # just do a cand-to-quota audit for the canonical winner
        C0 = deg0_scores[canonical_winner] + deg1_scores[canonical_winner] 
        C1 = deg1_scores[canonical_winner] + Tw

        d0_sample = deg0_deltas[canonical_winner].astype(int) + deg1_deltas[canonical_winner].astype(int)
        d1_sample = deg1_deltas[canonical_winner].astype(int) + w_deltas

        mu_0 = np.mean(d0_sample)
        mu_1 = np.mean(d1_sample)

        grad_M = np.array([N,-k_hat * N,N*(C1+ N*mu_1)/(Cv+N*mu_v), -k_hat * N* (C1+ N*mu_1)/(Cv+N*mu_v)])
        min_variances = [4*K/N, 4*K/N, 4*K/N, (m+2)**2*K/N]
        covariance_matrix = np.cov([np.array(d0_sample), np.array(d1_sample), np.array(du_sample), np.array(dv_sample)], ddof=1)
        for i in range(len(min_variances)):
            if covariance_matrix[i,i] < min_variances[i]:
                covariance_matrix[i,i] = min_variances[i]
        delta_variance = grad_M @ covariance_matrix @ grad_M.T
        delta_variance *= fpc

        M = C0 + N*mu_0 - k_hat* (C1 + N*mu_1)
        M_lower = M - z * math.sqrt(delta_variance)
        if M_lower < 0:
            print(f"AUDIT FAILED at node {str_node_ID}: canonical winner {profile.candidates[canonical_winner]} might not have quota: M_cq > {M_lower} and estimate M_cq= {M}, k_hat= {k_hat}.")
            print(M_lower)
            node_succesfully_audited = False
        elif verbose:
            print(f"Succesfully checked that candidate {profile.candidates[canonical_winner]} must have quota at node {str_node_ID}, M_cq > {M_lower} and estimate M_cq= {M}, k_hat= {k_hat}.")
        return node_succesfully_audited
        
    for c in hopeful_candidates:
        if c not in allowable_winners:
            C0 = deg0_scores[c] + deg1_scores[c] 
            C1 = deg1_scores[c] + Tw
            
            d0_sample = deg0_deltas[c].astype(int) + deg1_deltas[c].astype(int)
            d1_sample = deg1_deltas[c].astype(int) + w_deltas

            mu_0 = np.mean(d0_sample)
            mu_1 = np.mean(d1_sample)

            grad_M = np.array([N,-k_hat * N,N*(C1+ N*mu_1)/(Cv+N*mu_v), -k_hat * N* (C1+ N*mu_1)/(Cv+N*mu_v)])
            min_variances = [4*K/N, 4*K/N, 4*K/N, (m+2)**2*K/N]
            covariance_matrix = np.cov([np.array(d0_sample), np.array(d1_sample), np.array(du_sample), np.array(dv_sample)], ddof=1)
            for i in range(len(min_variances)):
                if covariance_matrix[i,i] < min_variances[i]:
                    covariance_matrix[i,i] = min_variances[i]
            delta_variance = grad_M @ covariance_matrix @ grad_M.T
            delta_variance *= fpc

            M = C0 + N*mu_0 - k_hat* (C1 + N*mu_1)
            M_upper = M + z * math.sqrt(delta_variance)
            if M_upper > 0:
                print(f"AUDIT FAILED at node {str_node_ID}: candidate {profile.candidates[c]} might have quota: M_cq < {M_upper} and estimate M_cq= {M}, k_hat= {k_hat}.")
                print(M_upper)
                node_succesfully_audited = False
            elif verbose:
                print(f"Succesfully checked that candidate {profile.candidates[c]} does not have quota at node {str_node_ID}, M_cq < {M_upper} and estimate M_cq= {M}, k_hat= {k_hat}")
        if c not in allowable_losers:
            C0 = deg0_scores[c] - deg0_scores[canonical_loser] + deg1_scores[c] - deg1_scores[canonical_loser]
            C1 = deg1_scores[c] - deg1_scores[canonical_loser]

            d0_sample = deg0_deltas[c].astype(int) - deg0_deltas[canonical_loser].astype(int) + deg1_deltas[c].astype(int) - deg1_deltas[canonical_loser].astype(int)
            d1_sample = deg1_deltas[c].astype(int) - deg1_deltas[canonical_loser].astype(int) 
            mu_0 = np.mean(d0_sample)
            mu_1 = np.mean(d1_sample)
            grad_M = np.array([N,-k_hat * N,N*(C1+ N*mu_1)/(Cv+N*mu_v), -k_hat * N* (C1+ N*mu_1)/(Cv+N*mu_v)])
            min_variances = [4*K/N, 4*K/N, 4*K/N, (m+2)**2*K/N]
            covariance_matrix = np.cov([np.array(d0_sample), np.array(d1_sample), np.array(du_sample), np.array(dv_sample)], ddof=1)
            for i in range(len(min_variances)):
                if covariance_matrix[i,i] < min_variances[i]:
                    covariance_matrix[i,i] = min_variances[i]
            delta_variance = grad_M @ covariance_matrix @ grad_M.T
            delta_variance *= fpc
            print(f"delta_variance: {delta_variance}")

            M = C0 + N*mu_0 - k_hat* (C1 + N*mu_1)
            M_lower = M - z * math.sqrt(delta_variance)
            if M_lower <= 0:
                print(f"AUDIT FAILED at node {str_node_ID}: candidate {profile.candidates[c]} might lose to {profile.candidates[canonical_loser]}. M_cl < {M_lower} and estimate M_cl= {M}, k_hat= {k_hat}.")
                print(M_lower)
                node_succesfully_audited = False
            elif verbose:
                print(f"Succesfully checked that candidate {profile.candidates[c]} does not lose to {profile.candidates[canonical_loser]} at node {str_node_ID}, M_cl > {M_lower} and estimate M_cl= {M}, k_hat= {k_hat}")
    return node_succesfully_audited


# --------------- Symbols and other setup for deg 2 ---------------------------------


#
# 0. Symbols
#

# Parameters that appear in A2,B2,C2 (you can treat these as the θ_i if you like)
T1, T2, T12, T21 = sp.symbols('T1 T2 T12 T21', real=True)
mu1, mu2, mu12, mu21, mug = sp.symbols('mu1 mu2 mu12 mu21 mug', real=True)
t1, t2, t12, t21 = sp.symbols('t1 t2 t12 t21', real=True)
nu1, nu2, nu12, nu21 = sp.symbols('nu1 nu2 nu12 nu21', real=True)
m, N, g, epsilon = sp.symbols('m N g epsilon', real=True)

# Keep factors
k1, k2 = sp.symbols('k1 k2', real=True)

#
# 1. A2, B2, C2 and the symmetric A1, B1, C1
#    (Equations (1)–(3) in your pdf.)
#

A2 = (
    (T2+N*mu2)*(t12+N*nu12)
    - (T12+N*mu12)*(t2+N*nu2)
    + (T2+N*mu2)*(t21+N*nu21)
    + (T21+N*mu21)*(t12+N*nu12)
    + (T21+N*mu21)*(t2+N*nu2)
    + (T21+N*mu21)*(t21+N*nu21)
    - (m + 1)*((T12+N*mu12)*(T21+N*mu21) + (T2+N*mu2)*(T21+N*mu21))
)

B2 = (
    - (T1+N*mu1)*(t12+N*nu12)
    - (T1+N*mu1)*(t2+N*nu2)
    + (T12+N*mu12)*(t2+N*nu2)
    - (T2+N*mu2)*(t1+N*nu1)
    - (T2+N*mu2)*(t12+N*nu12)
    - (T1+N*mu1)*(t21+N*nu21)
    - (T2+N*mu2)*(t21+N*nu21)
    + (N - (g+N*mug))*((T21+N*mu21) - (T12+N*mu12))
    - (T21+N*mu21)*(t1+N*nu1)
    - 2*((T21+N*mu21)*(t12+N*nu12) + (T21+N*mu21)*(t2+N*nu2) + (T21+N*mu21)*(t21+N*nu21))
    + (m + 1)*((T1+N*mu1)*(T12+N*mu12) + (T1+N*mu1)*(T2+N*mu2) + (T12+N*mu12)*(T21+N*mu21) + (T2+N*mu2)*(T21+N*mu21) - (T12+N*mu12)*epsilon + (T21+N*mu21)*epsilon)
)

C2 = - (T1 + N*mu1 + T21 + N*mu21) * (
    N - (g + N*mug) - t1 - N*nu1 - t12 - N*nu12 - t2 - N*nu2 - t21 - N*nu21 + (m + 1)*epsilon
)

# Helper: swap indices 1 <-> 2 to get A1,B1,C1
_swap_pairs = [(T1, T2), (t1, t2), (T12, T21), (t12, t21), (mu1, mu2), (nu1, nu2), (mu12, mu21), (nu12, nu21)]

def _swap_1_2(expr: sp.Expr) -> sp.Expr:
    """Swap indices 1 and 2 in an expression (T1<->T2, t1<->t2, 12<->21)."""
    out = expr
    for a, b in _swap_pairs:
        tmp = sp.Symbol(f'__tmp_{a.name}')
        # a -> tmp, b -> a, then tmp -> b
        out = out.subs({a: tmp, b: a})
        out = out.subs({tmp: b})
    return out

A1 = _swap_1_2(A2)
B1 = _swap_1_2(B2)
C1 = _swap_1_2(C2)

# Polynomials P1(θ,k1) and P2(θ,k2)
P1 = A1*k1**2 + B1*k1 + C1
P2 = A2*k2**2 + B2*k2 + C2

#
# 2. ∂θ A, ∂θ B, ∂θ C and implicit ∂θ k via IFT
#

# Choose which symbols you want to treat as θ_i for the keep factors.
# You can modify this list to match your actual θ-vector.
poly_constants = [T1, T2, T12, T21,
              t1, t2, t12, t21, g, N, m, epsilon]
theta_poly = [mug, mu1, mu2, mu12, mu21,
              nu1, nu2, nu12, nu21]

# Derivatives of A1,B1,C1 and A2,B2,C2 w.r.t each θ_i
dA1 = {th: sp.diff(A1, th) for th in theta_poly}
dB1 = {th: sp.diff(B1, th) for th in theta_poly}
dC1 = {th: sp.diff(C1, th) for th in theta_poly}

dA2 = {th: sp.diff(A2, th) for th in theta_poly}
dB2 = {th: sp.diff(B2, th) for th in theta_poly}
dC2 = {th: sp.diff(C2, th) for th in theta_poly}

print(dA2)

# Implicit Function Theorem:
# ∂k/∂θ_i = - ( (∂θ A) k^2 + (∂θ B) k + (∂θ C) ) / (2 A k + B)

dk1 = {
    th: - (dA1[th]*k1**2 + dB1[th]*k1 + dC1[th]) / (2*A1*k1 + B1)
    for th in theta_poly
}

dk2 = {
    th: - (dA2[th]*k2**2 + dB2[th]*k2 + dC2[th]) / (2*A2*k2 + B2)
    for th in theta_poly
}

#
# 3. Symbolic M_ij(θ, k1, k2)
#    Using the formula in the pdf (with check T's and μ's as separate symbols).
#

# Check-T totals (treated as constants w.r.t θ unless you add them to θ)
T_i, T_j = sp.symbols('T_i T_j', real=True)
T_w1i, T_w1j = sp.symbols('T_w1i T_w1j', real=True)
T_w2i, T_w2j = sp.symbols('T_w2i T_w2j', real=True)
T_w1w2i, T_w1w2j = sp.symbols('T_w1w2i T_w1w2j', real=True)

# μ parameters that appear directly in the margin
mu_i, mu_j = sp.symbols('mu_i mu_j', real=True)
mu_w1i, mu_w1j = sp.symbols('mu_w1i mu_w1j', real=True)
mu_w2i, mu_w2j = sp.symbols('mu_w2i mu_w2j', real=True)
mu_w1w2i, mu_w1w2j = sp.symbols('mu_w1w2i mu_w1w2j', real=True)

M_ij = (
    (T_i - T_j)
    + N*(mu_i - mu_j)
    + (1 - k1)*(1 - k2)*(
        T_w1w2i - T_w1w2j + N*(mu_w1w2i - mu_w1w2j)
    )
    + (1 - k1)*(
        T_w1i - T_w1j + N*(mu_w1i - mu_w1j)
    )
    + (1 - k2)*(
        T_w2i - T_w2j + N*(mu_w2i - mu_w2j)
    )
)

#
# 4. ∇_θ M_ij, using the precomputed ∂θ k1, ∂θ k2
#

# Extend θ to include whichever μ's you want derivatives for.
# This is the θ-vector for the margin. Adjust to match your preferred ordering.
theta_margin_extra = [
    mu_i, mu_j,
    mu_w1i, mu_w1j,
    mu_w2i, mu_w2j,
    mu_w1w2i, mu_w1w2j,
] #NB: these could be combined pairwise, e.g., mu_i - mu_j, decreasing the number of parameters.

theta_full = theta_poly + theta_margin_extra

# Gradient entries: ∂θ M = ∂θ M |_k fixed + (∂M/∂k1) ∂θ k1 + (∂M/∂k2) ∂θ k2
grad_M_exprs = []
dM_dk1 = sp.diff(M_ij, k1)
dM_dk2 = sp.diff(M_ij, k2)

for th in theta_full:
    dM_direct = sp.diff(M_ij, th)
    dM_via_k1 = dM_dk1 * dk1.get(th, 0)
    dM_via_k2 = dM_dk2 * dk2.get(th, 0)
    grad_M_exprs.append(dM_direct + dM_via_k1 + dM_via_k2)

# grad_M_exprs[i] is ∂M/∂θ_full[i]


#
# 5. Numerical evaluation of ∇_θ M_ij
#

# All symbols that need numeric values for evaluation
all_syms_for_eval = (
    theta_full
    + poly_constants     
    + [T_i, T_j,
       T_w1i, T_w1j,
       T_w2i, T_w2j,
       T_w1w2i, T_w1w2j,
       k1, k2]
)

_grad_M_func = sp.lambdify(all_syms_for_eval, grad_M_exprs, 'numpy')


def eval_grad_M(
    mu_values: dict,
    Tcheck_values: dict,
    k_values: dict,
) -> np.ndarray:
    """
    Evaluate ∇_θ M_ij at numerical values.

    Parameters
    ----------
    mu_values : dict
        Numerical values for μ parameters in the margin:
        keys: mu_i, mu_j, mu_w1i, mu_w1j, mu_w2i, mu_w2j, mu_w1w2i, mu_w1w2j, mu1, mu2, mu12, mu21, mug, nu1, nu2, nu12, nu21.
    Tcheck_values : dict
        Numerical values for check-* totals:
        keys: T_i, T_j, T_w1i, T_w1j, T_w2i, T_w2j, T_w1w2i, T_w1w2j, T1, T2, T12, T21, t1, t2, t12, t21, g, N, m, epsilon.
    k_values : dict
        Numerical values for the keep factors:
        keys: k1, k2 (the roots you solved for separately).

    Returns
    -------
    grad : np.ndarray, shape (len(theta_full),)
        Gradient vector ∇_θ M_ij in the order given by `theta_full`.
    """
    # Merge all dicts keyed by SymPy symbols
    subs = {}
    subs.update(mu_values)
    subs.update(Tcheck_values)
    subs.update(k_values)

    # Build argument list in the correct order
    args = [subs[sym] for sym in all_syms_for_eval]
    grad_vals = _grad_M_func(*args)
    return np.asarray(grad_vals, dtype=float)

# --- Lambdify for M_ij itself (no variance) ---

# All symbols that M_ij actually depends on
M_syms_for_eval = [
    mu_i, mu_j,
    mu_w1i, mu_w1j,
    mu_w2i, mu_w2j,
    mu_w1w2i, mu_w1w2j,
    T_i, T_j,
    T_w1i, T_w1j,
    T_w2i, T_w2j,
    T_w1w2i, T_w1w2j,
    N, k1, k2,
]

_M_func = sp.lambdify(M_syms_for_eval, M_ij, 'numpy')


def eval_M(
    mu_values: dict,
    Tcheck_values: dict,
    k_values: dict,
) -> float:
    """
    Evaluate the margin M_ij at numerical values.

    Parameters
    ----------
    mu_values : dict
        Numerical values for μ parameters in the margin, e.g.:
        mu_i, mu_j, mu_w1i, mu_w1j, mu_w2i, mu_w2j, mu_w1w2i, mu_w1w2j, ...
    Tcheck_values : dict
        Numerical values for check-* totals and N (and friends if you like), e.g.:
        Tc_i, Tc_j, Tc_w1i, Tc_w1j, Tc_w2i, Tc_w2j, Tc_w1w2i, Tc_w1w2j, N, ...
    k_values : dict
        Numerical values for the keep factors:
        keys: k1, k2.

    Returns
    -------
    M : float
        The scalar value of M_ij at the given parameters.
    """
    subs = {}
    subs.update(mu_values)
    subs.update(Tcheck_values)
    subs.update(k_values)

    args = [subs[sym] for sym in M_syms_for_eval]
    M_val = _M_func(*args)
    return float(M_val)

# 6. Quadratic roots for P1 and P2 via lambdify

# A1,B1,C1,A2,B2,C2 depend only on theta_poly
_P_coeff_syms = poly_constants + theta_poly
_P_coeffs_func = sp.lambdify(
    _P_coeff_syms,
    (A1, B1, C1, A2, B2, C2),
    'numpy'
)


def _real_roots_quadratic(A, B, C, tol=1e-12):
    """
    Return list of real roots of A x^2 + B x + C = 0 (0, 1, or 2 roots).
    Handles near-linear/degenerate cases in a simple way.
    """
    roots = []

    # Degenerate to linear?
    if abs(A) < tol:
        if abs(B) < tol:
            return roots  # constant equation: either no root or all x; ignore
        roots.append(-C / B)
        return roots

    disc = B * B - 4.0 * A * C
    if disc < -tol:
        return roots  # no real roots

    # Clamp very small negative discriminants to zero
    if disc < 0.0:
        disc = 0.0

    sqrt_disc = np.sqrt(disc)
    r1 = (-B - sqrt_disc) / (2.0 * A)
    r2 = (-B + sqrt_disc) / (2.0 * A)
    roots.extend([r1, r2])
    return roots


def eval_least_positive_root(mu_values, Tcheck_values,
                             tol=1e-12):
    """
    Using the same dict interface as eval_grad_M, evaluate A1,B1,C1,A2,B2,C2
    and return the least positive real root of P1(k1)=0 or P2(k2)=0.

        P1(k1) = A1 k1^2 + B1 k1 + C1
        P2(k2) = A2 k2^2 + B2 k2 + C2

    Parameters
    ----------
    theta_values : dict
        Numerical values for T1, T2, T12, T21, t1, t2, t12, t21, m, N, g, epsilon.
    mu_values : dict
        Ignored here; accepted for signature compatibility with eval_grad_M.
    Tcheck_values : dict
        Ignored here; accepted for signature compatibility with eval_grad_M.
    k_values : dict
        Ignored here; accepted for signature compatibility with eval_grad_M.
    tol : float
        Tolerance for detecting degeneracy / positivity.

    Returns
    -------
    k_min : float
        Least positive real root among roots of P1 and P2.

    Raises
    ------
    ValueError
        If no positive real roots are found.
    """
    # We only need theta_values, but we accept the other dicts for compatibility.
    subs = {}
    subs.update(Tcheck_values)
    subs.update(mu_values)

    # Build argument list for lambdified coeff function
    args = [subs[sym] for sym in _P_coeff_syms]
    A1v, B1v, C1v, A2v, B2v, C2v = _P_coeffs_func(*args)

    roots1 = []
    roots2 = []
    roots1.extend(_real_roots_quadratic(A1v, B1v, C1v, tol=tol))
    roots2.extend(_real_roots_quadratic(A2v, B2v, C2v, tol=tol))

    pos_roots_k1 = [float(r) for r in roots1 if r > tol]
    pos_roots_k2 = [float(r) for r in roots2 if r > tol]

    if not pos_roots_k1 or not pos_roots_k2:
        raise ValueError("No positive real roots for P1 or P2 at given parameters.")

    return min(pos_roots_k1), min(pos_roots_k2)

def deg2_node_audit(str_node_ID, graph, deg2_kwargs, verbose = False):
    _N = deg2_kwargs['N']
    _m = deg2_kwargs['m']
    BAL = deg2_kwargs['BAL']
    CVR = deg2_kwargs['CVR']
    _epsilon = deg2_kwargs['epsilon']
    K = deg2_kwargs['K']
    _n = deg2_kwargs['n']
    z = deg2_kwargs['z']
    profile = deg2_kwargs['profile']
    candidate_strs = list(profile.candidates)
    num_cands = len(candidate_strs)

    if _m!= 3:
        raise ValueError("deg2_node_audit currently only supports m=3")
    
    node_info = graph.lookup_node(str_node_ID)
    remaining_candidates = [c for c in range(num_cands) if c not in node_info['initial_losers'] and c not in node_info['winner_to_cand']]
    
    if len(remaining_candidates) == 1:
        if verbose:
            print(f"Node {str_node_ID} only has one remaining candidate {candidate_strs[remaining_candidates[0]]}, automatically passing audit.")
        return True
    elif len(remaining_candidates) == 0:
        raise ValueError(f"Node {str_node_ID} has no remaining candidates, something went wrong.")
    else:
        canonical_loser = node_info.get('canonical_loser', None)
        if canonical_loser is None:
            raise ValueError("deg2_node_audit requires a canonical_loser in node_info")
        
    
    depth = node_info['layer']
    next_layer_codes = graph.nodes_by_layer[depth + 1]
    allowable_losers= []
    for code in next_layer_codes:
        decoded = graph.decode_node_string(code)
        if set(decoded["winner_to_cand"]) >= set(node_info["winner_to_cand"]) and set(decoded["initial_losers"]) >= set(node_info["initial_losers"]):
            if len(decoded["winner_to_cand"]) == len(node_info["winner_to_cand"]) + 1 and verbose:
                print(f"Skipping edge from node {str_node_ID} to {code} since deg2_node_audit only handles loser expansions.")
            elif len(decoded["initial_losers"]) == len(node_info["initial_losers"]) + 1:
                new_loser = list(set(decoded["initial_losers"]) - set(node_info["initial_losers"]))
                allowable_losers.append(new_loser[0])

    projected_BAL_fpv, BAL_winner_comb = project_matrix_onto_election_state(
        BAL,
        num_cands=len(profile.candidates),
        m=_m,
        losers=node_info['initial_losers'],
        winner_to_cand=node_info['winner_to_cand'])
    projected_CVR_fpv, CVR_winner_comb = project_matrix_onto_election_state(
        CVR,
        num_cands=len(profile.candidates),
        m=_m,
        losers=node_info['initial_losers'],
        winner_to_cand=node_info['winner_to_cand'])
    
    ## RESERVED SYMBOLS: DO NOT USE FOR OTHER PURPOSES
    #T1, T2, T12, T21 = sp.symbols('T1 T2 T12 T21', real=True)
    #mu1, mu2, mu12, mu21, mug = sp.symbols('mu1 mu2 mu12 mu21 mug', real=True)
    #t1, t2, t12, t21 = sp.symbols('t1 t2 t12 t21', real=True)
    #nu1, nu2, nu12, nu21 = sp.symbols('nu1 nu2 nu12 nu21', real=True)
    #m, N, g, epsilon = sp.symbols('m N g epsilon', real=True)
    #
    #k1, k2 = sp.symbols('k1 k2', real=True)
    #
    ##CANDIDATE SPECIFIC SYMBOLS
    #T_i, T_j = sp.symbols('T_i T_j', real=True)
    #T_w1i, T_w1j = sp.symbols('T_w1i T_w1j', real=True)
    #T_w2i, T_w2j = sp.symbols('T_w2i T_w2j', real=True)
    #T_w1w2i, T_w1w2j = sp.symbols('T_w1w2i T_w1w2j', real=True)
    #mu_i, mu_j = sp.symbols('mu_i mu_j', real=True)
    #mu_w1i, mu_w1j = sp.symbols('mu_w1i mu_w1j', real=True)
    #mu_w2i, mu_w2j = sp.symbols('mu_w2i mu_w2j', real=True)
    #mu_w1w2i, mu_w1w2j = sp.symbols('mu_w1w2i mu_w1w2j', real=True)

    # dicts with symbolic keys
    mu_values = {}
    Tcheck_values = {N: _N, m: _m, epsilon: _epsilon} 
    k_values = {}
    
    helper_fpv_vec, helper_winner_comb = project_matrix_onto_election_state(deg2_kwargs['ballot_matrix'], num_cands=len(profile.candidates), m=_m, losers=node_info['initial_losers'], winner_to_cand=node_info['winner_to_cand'])

    fpc = (_N - _n) / (_n * (_N- 1))
    mult_vec = deg2_kwargs['mult_vec']

    score_transfer_masks = {i: helper_winner_comb == i for i in [0,1,2,4,6]}
    score_fpv_masks = {i: helper_fpv_vec == i for i in remaining_candidates}
    score_fpv_masks[-1] = helper_fpv_vec < 0

    Tcheck_values[g] = np.sum(mult_vec[score_fpv_masks[-1] & (score_transfer_masks[0])]) + deg2_kwargs['num_ghosts']

    deg0_scores = {i: sum(mult_vec[score_fpv_masks[i] & score_transfer_masks[0]]) for i in remaining_candidates}
    deg0_scores[-1] = sum(mult_vec[score_fpv_masks[-1] & score_transfer_masks[0]])
    w1_scores = {i: sum(mult_vec[score_fpv_masks[i] & score_transfer_masks[1]]) for i in remaining_candidates}
    w1_scores[-1] = sum(mult_vec[score_fpv_masks[-1] & score_transfer_masks[1]])
    w2_scores = {i: sum(mult_vec[score_fpv_masks[i] & score_transfer_masks[2]]) for i in remaining_candidates}
    w2_scores[-1] = sum(mult_vec[score_fpv_masks[-1] & score_transfer_masks[2]])
    w1w2_scores = {i: sum(mult_vec[score_fpv_masks[i] & (score_transfer_masks[4]| score_transfer_masks[6])]) for i in remaining_candidates} # we don't care what order the transfer happens in
    #w1w2_scores[-1] = sum(mult_vec[score_fpv_masks[-1] & (score_transfer_masks[4]| score_transfer_masks[6])])

    Tcheck_values[T1] = sum(mult_vec[score_transfer_masks[1] | score_transfer_masks[4]])
    Tcheck_values[T2] = sum(mult_vec[score_transfer_masks[2]| score_transfer_masks[6]])
    Tcheck_values[T12] = sum(mult_vec[score_transfer_masks[4]])
    Tcheck_values[T21] = sum(mult_vec[score_transfer_masks[6]])
    Tcheck_values[t1] = w1_scores[-1]
    Tcheck_values[t2] = w2_scores[-1]
    Tcheck_values[t12] = sum(mult_vec[score_fpv_masks[-1] & score_transfer_masks[4]])
    Tcheck_values[t21] = sum(mult_vec[score_fpv_masks[-1] & score_transfer_masks[6]])

    # j is always the canonical loser, so fill those values in now
    Tcheck_values[T_j] = deg0_scores[canonical_loser] 
    Tcheck_values[T_w1j] = w1_scores[canonical_loser]
    Tcheck_values[T_w2j] = w2_scores[canonical_loser]
    Tcheck_values[T_w1w2j] = w1w2_scores[canonical_loser]

    BAL_fpv_masks = {i: (projected_BAL_fpv == i) for i in remaining_candidates}
    BAL_fpv_masks[-1] = (projected_BAL_fpv < 0)
    CVR_fpv_masks = {i: (projected_CVR_fpv == i) for i in remaining_candidates}
    CVR_fpv_masks[-1] = (projected_CVR_fpv < 0)
    BAL_transfer_masks = {i: (BAL_winner_comb == i) for i in [0,1,2,4,6]}
    CVR_transfer_masks = {i: (CVR_winner_comb == i) for i in [0,1,2,4,6]}

    deg0_deltas = {i: (BAL_fpv_masks[i] & BAL_transfer_masks[0]).astype(int) - (CVR_fpv_masks[i] & CVR_transfer_masks[0]).astype(int) for i in remaining_candidates}
    deg0_deltas[-1] = (BAL_fpv_masks[-1] & BAL_transfer_masks[0]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[0]).astype(int)
    w1_deltas = {i: (BAL_fpv_masks[i] & BAL_transfer_masks[1]).astype(int) - (CVR_fpv_masks[i] & CVR_transfer_masks[1]).astype(int) for i in remaining_candidates}
    w1_deltas[-1] = (BAL_fpv_masks[-1] & BAL_transfer_masks[1]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[1]).astype(int)
    w2_deltas = {i: (BAL_fpv_masks[i] & BAL_transfer_masks[2]).astype(int) - (CVR_fpv_masks[i] & CVR_transfer_masks[2]).astype(int) for i in remaining_candidates}
    w2_deltas[-1] = (BAL_fpv_masks[-1] & BAL_transfer_masks[2]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[2]).astype(int)
    w1w2_deltas = {i: ((BAL_fpv_masks[i] & (BAL_transfer_masks[4] | BAL_transfer_masks[6])).astype(int) - (CVR_fpv_masks[i] & (CVR_transfer_masks[4] | CVR_transfer_masks[6])).astype(int)) for i in remaining_candidates}
    
    mug_deltas = (BAL_fpv_masks[-1] & BAL_transfer_masks[0]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[0]).astype(int)
    mu1_deltas = BAL_transfer_masks[1].astype(int) - CVR_transfer_masks[1].astype(int)
    mu2_deltas = BAL_transfer_masks[2].astype(int) - CVR_transfer_masks[2].astype(int)
    nu1_deltas = (BAL_fpv_masks[-1] & BAL_transfer_masks[1]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[1]).astype(int)
    nu2_deltas = (BAL_fpv_masks[-1] & BAL_transfer_masks[2]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[2]).astype(int)
    mu12_deltas = BAL_transfer_masks[4].astype(int) - CVR_transfer_masks[4].astype(int)
    mu21_deltas = BAL_transfer_masks[6].astype(int) - CVR_transfer_masks[6].astype(int)
    nu12_deltas = (BAL_fpv_masks[-1] & BAL_transfer_masks[4]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[4]).astype(int)
    nu21_deltas = (BAL_fpv_masks[-1] & BAL_transfer_masks[6]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[6]).astype(int)
    
    mu_values[mug] = np.mean(mug_deltas.astype(int))
    mu_values[mu1] = np.mean(mu1_deltas.astype(int))
    mu_values[mu2] = np.mean(mu2_deltas.astype(int))
    mu_values[mu12] = np.mean(mu12_deltas.astype(int))
    mu_values[mu21] = np.mean(mu21_deltas.astype(int))
    mu_values[nu1] = np.mean(nu1_deltas.astype(int))
    mu_values[nu2] = np.mean(nu2_deltas.astype(int))
    mu_values[nu12] = np.mean(nu12_deltas.astype(int))
    mu_values[nu21] = np.mean(nu21_deltas.astype(int))

    print("Parameters for K: Tcheclk_values:")
    for key, value in Tcheck_values.items():
        print(f"  {key}: {value}")
    _k1, _k2 = eval_least_positive_root(mu_values, Tcheck_values)
    k_values[k1] = _k1
    k_values[k2] = _k2

    # again, j is always the canonical loser, so fill out the mu_*j
    mu_values[mu_j] = np.mean(deg0_deltas[canonical_loser].astype(int))
    mu_values[mu_w1j] = np.mean(w1_deltas[canonical_loser].astype(int))
    mu_values[mu_w2j] = np.mean(w2_deltas[canonical_loser].astype(int))
    mu_values[mu_w1w2j] = np.mean(w1w2_deltas[canonical_loser].astype(int))

    node_succesfully_audited = True

    if verbose:
        print(f"info at node {str_node_ID}: k1= {_k1}, k2= {_k2}")
    
    for c in remaining_candidates:
        if c not in allowable_losers and c != canonical_loser:
            Tcheck_values[T_i] = deg0_scores[c] 
            Tcheck_values[T_w1i] = w1_scores[c]
            Tcheck_values[T_w2i] = w2_scores[c]
            Tcheck_values[T_w1w2i] = w1w2_scores[c]

            mu_values[mu_i] = np.mean(deg0_deltas[c].astype(int))
            mu_values[mu_w1i] = np.mean(w1_deltas[c].astype(int))
            mu_values[mu_w2i] = np.mean(w2_deltas[c].astype(int))
            mu_values[mu_w1w2i] = np.mean(w1w2_deltas[c].astype(int))

            min_variance = K / _N
            # theta is: mug, mu1, mu2, mu12, mu21, nu1, nu2, nu12, nu21, mu_i, mu_j, mu_w1i, mu_w1j, mu_w2i, mu_w2j, mu_w1w2i, mu_w1w2j

            data = np.vstack([
                mug_deltas,
                mu1_deltas,
                mu2_deltas,
                mu12_deltas,
                mu21_deltas,
                nu1_deltas,
                nu2_deltas,
                nu12_deltas,
                nu21_deltas,
                deg0_deltas[c],
                deg0_deltas[canonical_loser],
                w1_deltas[c],
                w1_deltas[canonical_loser],
                w2_deltas[c],
                w2_deltas[canonical_loser],
                w1w2_deltas[c],
                w1w2_deltas[canonical_loser],
            ]).astype(float)

            covariance_matrix = np.cov(data, ddof=1)
            grad_M = eval_grad_M(mu_values, Tcheck_values, k_values)
            for i in range(covariance_matrix.shape[0]):
                if covariance_matrix[i,i] == 0:
                    covariance_matrix[i,i] = min_variance
            delta_variance = grad_M @ covariance_matrix @ grad_M.T
            delta_variance *= fpc
            print(f"delta_variance: {delta_variance}")

            M = eval_M(mu_values, Tcheck_values, k_values)

            M_lower = M - z * math.sqrt(abs(delta_variance))
            if M_lower <= 0:
                print(f"AUDIT FAILED at node {str_node_ID}: candidate {profile.candidates[c]} might lose to {profile.candidates[canonical_loser]}. M_cl < {M_lower} and estimate M_cl= {M}, k1= {k_values[k1]}, k2= {k_values[k2]}.")
                print(M_lower)
                node_succesfully_audited = False
            elif verbose:
                print(f"Succesfully checked that candidate {profile.candidates[c]} does not lose to {profile.candidates[canonical_loser]} at node {str_node_ID}, M_cl > {M_lower} and estimate M_cl= {M}, k1= {k_values[k1]}, k2= {k_values[k2]}.")
    return node_succesfully_audited

def end_to_end_synthetic_audit_3seat(profile, graph, num_ghosts, sample_size_fraction, alpha, noise_level, hypergeo_var_bounds = [],epsilon=1e-6, verbose=False):
    ballot_matrix, mult_vec, _ = convert_pf_to_numpy_arrays(profile)
    num_original_ballots = np.sum(mult_vec)
    _N = int(num_original_ballots+num_ghosts)
    sample_size =int(_N * sample_size_fraction)
    BAL, CVR = bal_cvr_sample_constructor(
        ballot_matrix,
        mult_vec,
        noise_level=noise_level,
        sample_size=sample_size,    
        num_ghosts=num_ghosts,
    )
    print("Number of ballots N:", _N)
    print("Sample size n:", BAL.shape[0])
    _general_kwargs = {
        'profile': profile,
        'm': 3,
        'n': BAL.shape[0],
        'alpha': alpha,
        'num_ghosts': num_ghosts, #(profile.total_ballot_wt//10),
        'z': NormalDist().inv_cdf(1 - alpha),
        'K': K_upper(_N, BAL.shape[0], alpha),
        'N': _N,
        'BAL': BAL,
        'CVR': CVR,
        'epsilon': epsilon,
        'ballot_matrix': ballot_matrix,
        'mult_vec': mult_vec
    }
    overall_pass = True
    failures = []
    if len(hypergeo_var_bounds) == 0:
        hypergeo_var_bounds = [alternative_K_upper(_N, BAL.shape[0], i, alpha)/_N for i in range(int(sample_size*noise_level))]
        print("using the following hypergeo_var_bounds:", hypergeo_var_bounds)
    for node_str in graph.list_nodes():
        node_degree = len(graph.lookup_node(node_str)['winner_to_cand'])
        print(f"-------------------------- Auditing node {node_str} with degree {node_degree} ------------------")
        if node_degree == 0:
            if not deg0_node_audit_alternative(node_str, graph, deg0_kwargs=_general_kwargs,hypergeo_var_bounds = hypergeo_var_bounds, verbose=verbose):
                overall_pass = False
                failures.append(node_str)
        elif node_degree == 1:
            if not deg1_node_audit_alternative(node_str, graph, deg1_kwargs=_general_kwargs,hypergeo_var_bounds = hypergeo_var_bounds, verbose=verbose):
                overall_pass = False
                failures.append(node_str)
        elif node_degree == 2:
            if not deg2_node_audit_alternative(node_str, graph, deg2_kwargs=_general_kwargs,hypergeo_var_bounds = hypergeo_var_bounds, verbose=verbose):
                overall_pass = False
                failures.append(node_str)
        elif node_degree == 3:
            if verbose:
                print(f"Skipping deg3 node {node_str} audit (automatic pass).")
        else:
            raise ValueError(f"Unsupported node degree {node_degree} at node {node_str}.")
    if overall_pass:
        print("Overall audit PASSED.")
        return True
    else:
        print(f"Overall audit FAILED: failed nodes: {failures}")
        
    # audit the failed nodes again to print their output again 
    for node_str in failures:
        node_degree = len(graph.lookup_node(node_str)['winner_to_cand'])
        print(f"-------------------------- Re-Auditing failed node {node_str} with degree {node_degree} ------------------")
        if node_degree == 0:
            deg0_node_audit_alternative(node_str, graph, deg0_kwargs=_general_kwargs,hypergeo_var_bounds = hypergeo_var_bounds, verbose=True)
        elif node_degree == 1:
            deg1_node_audit_alternative(node_str, graph, deg1_kwargs=_general_kwargs,hypergeo_var_bounds = hypergeo_var_bounds, verbose=True)
        elif node_degree == 2:
            deg2_node_audit_alternative(node_str, graph, deg2_kwargs=_general_kwargs,hypergeo_var_bounds = hypergeo_var_bounds, verbose=True)
        elif node_degree == 3:
            print(f"Skipping deg3 node {node_str} audit (automatic pass).")
    return False

def deg0_node_audit_alternative(str_node_ID, graph, deg0_kwargs, K_factor_alpha=.005, hypergeo_var_bounds=[], verbose = False):
    node_info = graph.lookup_node(str_node_ID)
    depth = node_info['layer']
    canonical_loser = node_info.get('canonical_loser', None)
    next_layer_codes = graph.nodes_by_layer[depth + 1]
    allowable_losers, allowable_winners = [], []
    profile = deg0_kwargs['profile']
    alpha = deg0_kwargs['alpha'] - K_factor_alpha
    if alpha <= 0:
        raise ValueError("alpha - K_factor_alpha must be positive in deg0_node_audit_alternative")
    N = deg0_kwargs['N']
    n = deg0_kwargs['n']
    for code in next_layer_codes:
        decoded = graph.decode_node_string(code)
        if set(decoded["winner_to_cand"]) >= set(node_info["winner_to_cand"]) and set(decoded["initial_losers"]) >= set(node_info["initial_losers"]):
            if len(decoded["winner_to_cand"]) == len(node_info["winner_to_cand"]) + 1:
                new_winner = list(set(decoded["winner_to_cand"]) - set(node_info["winner_to_cand"]))
                allowable_winners.append(new_winner[0])
            elif len(decoded["initial_losers"]) == len(node_info["initial_losers"]) + 1:
                new_loser = list(set(decoded["initial_losers"]) - set(node_info["initial_losers"]))
                allowable_losers.append(new_loser[0])
    projected_BAL_fpv, _ = project_matrix_onto_election_state(
        deg0_kwargs['BAL'],
        num_cands=len(profile.candidates),
        m=deg0_kwargs['m'],
        losers=node_info['initial_losers'])
    projected_CVR_fpv, _ = project_matrix_onto_election_state(
        deg0_kwargs['CVR'],
        num_cands=len(profile.candidates),
        m=deg0_kwargs['m'],
        losers=node_info['initial_losers'])
    
    mult_vec = deg0_kwargs['mult_vec']
    m = deg0_kwargs['m']
    epsilon = deg0_kwargs['epsilon']

    helper_fpv_vec, helper_winner_comb = project_matrix_onto_election_state(deg0_kwargs['ballot_matrix'], num_cands=len(profile.candidates), m=m, losers=node_info['initial_losers'], winner_to_cand=node_info['winner_to_cand'])
    deltas = {i: (projected_BAL_fpv == i).astype(int) - (projected_CVR_fpv == i).astype(int) for i in range(len(profile.candidates))}
    deltas[-127] = (projected_BAL_fpv < 0).astype(int) - (projected_CVR_fpv <0).astype(int)
    scores = {i: sum(mult_vec[helper_fpv_vec == i]) for i in range(len(profile.candidates))}

    quota = sum(scores.values())/(m+1)+epsilon
    #if len(hypergeo_var_bounds) != 21:
    #    hypergeo_var_bounds = [alternative_K_upper(N, n, i, alpha)/N for i in range(21)]
    if verbose:
        print(f"First round results at node {str_node_ID}:")
        print(f"FPV scores: {scores}")
        print(f"Quota: {quota}")
        print(f"Initial losers: {node_info['initial_losers']}")
        print(f"Initial winners: {node_info['winner_to_cand']}")
        print(f"Canonical loser: {canonical_loser}")
        print(f"Allowable winners: {allowable_winners}")
        print(f"Allowable losers: {allowable_losers}")
    fpc = (deg0_kwargs['N'] - deg0_kwargs['n']) / (deg0_kwargs['n'] * (deg0_kwargs['N'] - 1)) #lol

    g = deg0_kwargs['N']- sum(scores)
    #print(fpv_scores)
    #delta_vecs = {i: (projected_BAL_fpv == i).astype(int) - (projected_CVR_fpv == i).astype(int) for i in range(len(profile.candidates))}
    #delta_vecs[-127] = (projected_BAL_fpv < 0).astype(int) - (projected_CVR_fpv < 0).astype(int)
    node_succesfully_audited = True
    hopeful_candidates = [c for c in range(len(profile.candidates)) if c not in node_info['initial_losers'] and c not in node_info['winner_to_cand']]
    if len(hopeful_candidates) == m:
        print(f"Skipping node {str_node_ID} audit since there are as many hopeful candidates as seats.")
        return True
    if node_info.get('canonical_winner', None) is not None:
        canonical_winner = node_info['canonical_winner']
        # just do a cand-to-quota audit for the canonical winner
        C0 = scores[canonical_winner] - quota
        #d_iq = delta_vecs[canonical_winner].astype(int) + delta_vecs[-127].astype(int)/(deg0_kwargs['m']+1)
        d_i = deltas[canonical_winner].astype(int) 
        d_q = deltas[-127].astype(int)
        mean_diq = np.mean(d_i + d_q/(m+1))
        #print("mean_diq", mean_diq) 
        #sample_variance = np.var(d_iq, ddof=1) if np.any(d_iq != 0) else 4 * deg0_kwargs['K'] / deg0_kwargs['N']
        grad_M = np.array([N,N/(m+1)])
        samples = [np.array(d_i), np.array(d_q)]
        relevant_noisiness = np.sum((samples[0] != 0) | (samples[1] != 0)).astype(int)
        K_u = hypergeo_var_bounds[relevant_noisiness] if len(hypergeo_var_bounds)> relevant_noisiness else alternative_K_upper(deg0_kwargs['N'], deg0_kwargs['n'], relevant_noisiness, alpha=K_factor_alpha)/deg0_kwargs['N']
        covariance_matrix = np.cov(samples, ddof=1)
        for i in range(len(samples)):
            successes = sum(samples[i] != 0)
            if successes <=20:
                #covariance_matrix[i,i] = hypergeo_var_bounds[successes]
                covariance_matrix[i,i] = K_u
        delta_variance = grad_M @ covariance_matrix @ grad_M.T
        delta_variance *= fpc
        st_dev = math.sqrt(delta_variance)

        M_lower = C0 + deg0_kwargs['N'] * (mean_diq) - deg0_kwargs['z'] * st_dev
        if M_lower < 0:
            print(f"AUDIT FAILED at node {str_node_ID}: canonical winner {profile.candidates[canonical_winner]} might not have quota, M_cq > {M_lower} and M_cq = {C0}")
            node_succesfully_audited = False
        elif verbose:
            print(f"Succesfully checked that candidate {profile.candidates[canonical_winner]} must have quota at node {str_node_ID}, M_cq > {M_lower} and M_cq = {C0}")
        #return node_succesfully_audited
        # check that every other candidate not in other_allowable_winners loses to the canonical winner
        other_allowable_winners = node_info['other_plausible_winners']
        for h in hopeful_candidates:
            if h not in other_allowable_winners and h != canonical_winner:
                d_i = deltas[canonical_winner] 
                d_l = deltas[h]
                delta_mean = np.mean(d_i - d_l)
                grad_M = np.array([N,-N])
                samples = [np.array(d_i), np.array(d_l)]
                relevant_noisiness = np.sum((samples[0] != 0) | (samples[1] != 0)).astype(int)
                K_u = hypergeo_var_bounds[relevant_noisiness] if len(hypergeo_var_bounds)> relevant_noisiness else alternative_K_upper(deg0_kwargs['N'], deg0_kwargs['n'], relevant_noisiness, alpha=K_factor_alpha)/deg0_kwargs['N']
                covariance_matrix = np.cov(samples, ddof=1)
                for i in range(len(samples)):
                    successes = sum(samples[i] != 0)
                    if successes <=20:
                        #covariance_matrix[i,i] = hypergeo_var_bounds[successes]
                        covariance_matrix[i,i] = K_u
                delta_variance = grad_M @ covariance_matrix @ grad_M.T
                delta_variance *= fpc
                st_dev = math.sqrt(delta_variance)
                C_0 = scores[canonical_winner] - scores[h]
                M_lower = C_0 + deg0_kwargs['N'] * (delta_mean) - deg0_kwargs['z'] * st_dev
                if M_lower <= 0:
                    print(f"AUDIT FAILED at node {str_node_ID}: canonical winner {profile.candidates[canonical_winner]} might lose to {profile.candidates[h]}: M_cl < {M_lower} and M_cl = {C_0}")
                    node_succesfully_audited = False
                elif verbose:
                    print(f"Succesfully checked that canonical winner {profile.candidates[canonical_winner]}  does not lose to {profile.candidates[h]} at node {str_node_ID}, M_cl > {M_lower} and M_cl = {C_0}")
        return node_succesfully_audited
    
    rejected_winners = node_info.get('rejected_winners', None)
    if rejected_winners is None:
        rejected_winners = []
    for c in hopeful_candidates:
        if c not in allowable_winners and c not in rejected_winners:
            # just do a cand-to-quota audit for the canonical winner
            C0 = scores[c] - quota
            #d_iq = delta_vecs[canonical_winner].astype(int) + delta_vecs[-127].astype(int)/(deg0_kwargs['m']+1)
            d_i = deltas[c].astype(int) 
            d_q = deltas[-127].astype(int)
            mean_diq = np.mean(d_i + d_q/(m+1))
            #print("mean_diq", mean_diq) 
            #sample_variance = np.var(d_iq, ddof=1) if np.any(d_iq != 0) else 4 * deg0_kwargs['K'] / deg0_kwargs['N']
            grad_M = np.array([N,N/(m+1)])
            samples = [np.array(d_i), np.array(d_q)]
            relevant_noisiness = np.sum((samples[0] != 0) | (samples[1] != 0)).astype(int)
            #print("n:", deg0_kwargs['n'], "relevant_noisiness:", relevant_noisiness)
            K_u = hypergeo_var_bounds[relevant_noisiness] if len(hypergeo_var_bounds)> relevant_noisiness else alternative_K_upper(deg0_kwargs['N'], deg0_kwargs['n'], relevant_noisiness, alpha=K_factor_alpha)/deg0_kwargs['N']
            covariance_matrix = np.cov(samples, ddof=1)
            for i in range(len(samples)):
                successes = sum(samples[i] != 0)
                if successes <=20:
                    #covariance_matrix[i,i] = hypergeo_var_bounds[successes]
                    covariance_matrix[i,i] = K_u
            delta_variance = grad_M @ covariance_matrix @ grad_M.T
            delta_variance *= fpc
            #print(delta_variance, covariance_matrix)
            st_dev = math.sqrt(delta_variance)

            M_upper = C0 + deg0_kwargs['N'] * (mean_diq) + deg0_kwargs['z'] * st_dev
            if M_upper > 0:
                print(f"AUDIT FAILED at node {str_node_ID}: candidate {profile.candidates[c]} might have quota: M_cq < {M_upper} and M_cq = {C0}.")
                node_succesfully_audited = False
            elif verbose:
                print(f"Succesfully checked that candidate {profile.candidates[c]} does not have quota at node {str_node_ID}, M_cq < {M_upper} and M_cq = {C0}")
        if c in rejected_winners: # check the M_{cw} is negative
            #deltas = delta_vecs[c] - delta_vecs[canonical_loser]
            w = node_info['probable_winner']
            d_i = deltas[c] 
            d_l = deltas[w]
            delta_mean = np.mean(d_i - d_l)
            grad_M = np.array([N,-N])
            samples = [np.array(d_i), np.array(d_l)]
            relevant_noisiness = np.sum((samples[0] != 0) | (samples[1] != 0)).astype(int)
            K_u = hypergeo_var_bounds[relevant_noisiness] if len(hypergeo_var_bounds)> relevant_noisiness else alternative_K_upper(deg0_kwargs['N'], deg0_kwargs['n'], relevant_noisiness, alpha=K_factor_alpha)/deg0_kwargs['N']
            covariance_matrix = np.cov(samples, ddof=1)
            for i in range(len(samples)):
                successes = sum(samples[i] != 0)
                if successes <=20:
                    #covariance_matrix[i,i] = hypergeo_var_bounds[successes]
                    covariance_matrix[i,i] = K_u
            delta_variance = grad_M @ covariance_matrix @ grad_M.T
            delta_variance *= fpc
            st_dev = math.sqrt(delta_variance)
            C_0 = scores[c] - scores[w]
            M_upper = C_0 + deg0_kwargs['N'] * (delta_mean) + deg0_kwargs['z'] * st_dev
            if M_upper >= 0:
                print(f"AUDIT FAILED at node {str_node_ID}: rejected winner {profile.candidates[c]} might not lose to probable winner {profile.candidates[w]}: M_cw < {M_upper} and M_cw = {C_0}")
                node_succesfully_audited = False
            elif verbose:
                print(f"Succesfully checked that rejected winner {profile.candidates[c]} loses to probable winner {profile.candidates[w]} at node {str_node_ID}, M_cw < {M_upper} and T_cw = {C_0}")
        if c not in allowable_losers and node_info.get('canonical_winner', None) is None: # check the M_{cl} is positive
            #deltas = delta_vecs[c] - delta_vecs[canonical_loser]
            d_i = deltas[c] 
            d_l = deltas[canonical_loser]
            delta_mean = np.mean(d_i - d_l)
            grad_M = np.array([N,-N])
            samples = [np.array(d_i), np.array(d_l)]
            relevant_noisiness = np.sum((samples[0] != 0) | (samples[1] != 0)).astype(int)
            K_u = hypergeo_var_bounds[relevant_noisiness] if len(hypergeo_var_bounds)> relevant_noisiness else alternative_K_upper(deg0_kwargs['N'], deg0_kwargs['n'], relevant_noisiness, alpha=K_factor_alpha)/deg0_kwargs['N']
            covariance_matrix = np.cov(samples, ddof=1)
            for i in range(len(samples)):
                successes = sum(samples[i] != 0)
                if successes <=20:
                    #covariance_matrix[i,i] = hypergeo_var_bounds[successes]
                    covariance_matrix[i,i] = K_u
            delta_variance = grad_M @ covariance_matrix @ grad_M.T
            delta_variance *= fpc
            st_dev = math.sqrt(delta_variance)
            C_0 = scores[c] - scores[canonical_loser]
            M_lower = C_0 + deg0_kwargs['N'] * (delta_mean) - deg0_kwargs['z'] * st_dev
            if M_lower <= 0:
                print(f"AUDIT FAILED at node {str_node_ID}: candidate {profile.candidates[c]} might lose to {profile.candidates[canonical_loser]}: M_cl < {M_lower} and M_cl = {C_0}")
                node_succesfully_audited = False
            elif verbose:
                print(f"Succesfully checked that candidate {profile.candidates[c]} does not lose to {profile.candidates[canonical_loser]} at node {str_node_ID}, M_cl > {M_lower} and T_cl = {C_0}")
    return node_succesfully_audited

def deg1_node_audit_alternative(str_node_ID, graph, deg1_kwargs, K_factor_alpha = .005, hypergeo_var_bounds = [], verbose = False):
    N = deg1_kwargs['N']
    m = deg1_kwargs['m']
    BAL = deg1_kwargs['BAL']
    CVR = deg1_kwargs['CVR']
    epsilon = deg1_kwargs['epsilon']
    K = deg1_kwargs['K']
    n = deg1_kwargs['n']
    z = deg1_kwargs['z']
    num_ghosts = deg1_kwargs['num_ghosts']
    profile = deg1_kwargs['profile']
    alpha = deg1_kwargs['alpha'] - K_factor_alpha
    if alpha <= 0:
        raise ValueError("alpha - K_factor_alpha must be positive in deg1_node_audit_alternative")

    node_info = graph.lookup_node(str_node_ID)
    depth = node_info['layer']
    canonical_loser = node_info.get('canonical_loser', None)
    if canonical_loser is None:
        raise ValueError("deg1_node_audit requires a canonical_loser in node_info")
    next_layer_codes = graph.nodes_by_layer[depth + 1]
    allowable_losers, allowable_winners = [], []
    for code in next_layer_codes:
        decoded = graph.decode_node_string(code)
        if set(decoded["winner_to_cand"]) >= set(node_info["winner_to_cand"]) and set(decoded["initial_losers"]) >= set(node_info["initial_losers"]):
            if len(decoded["winner_to_cand"]) == len(node_info["winner_to_cand"]) + 1:
                new_winner = list(set(decoded["winner_to_cand"]) - set(node_info["winner_to_cand"]))
                allowable_winners.append(new_winner[0])
            elif len(decoded["initial_losers"]) == len(node_info["initial_losers"]) + 1:
                new_loser = list(set(decoded["initial_losers"]) - set(node_info["initial_losers"]))
                allowable_losers.append(new_loser[0])
    projected_BAL_fpv, BAL_winner_comb = project_matrix_onto_election_state(
        BAL,
        num_cands=len(profile.candidates),
        m=m,
        losers=node_info['initial_losers'],
        winner_to_cand=node_info['winner_to_cand'])
    projected_CVR_fpv, CVR_winner_comb = project_matrix_onto_election_state(
        CVR,
        num_cands=len(profile.candidates),
        m=m,
        losers=node_info['initial_losers'],
        winner_to_cand=node_info['winner_to_cand'])
    
    helper_fpv_vec, helper_winner_comb = project_matrix_onto_election_state(deg1_kwargs['ballot_matrix'], num_cands=len(profile.candidates), m=m, losers=node_info['initial_losers'], winner_to_cand=node_info['winner_to_cand'])

    fpc = (N - n) / (n * (N- 1))
    mult_vec = deg1_kwargs['mult_vec']
    g = np.sum(mult_vec[(helper_fpv_vec < 0) & (helper_winner_comb == 0)]) + num_ghosts

    deg0_scores = {i: sum(mult_vec[(helper_fpv_vec == i) & (helper_winner_comb == 0)]) for i in range(len(profile.candidates))}
    deg0_scores[-1] = sum(mult_vec[(helper_fpv_vec < 0) & (helper_winner_comb == 0)])
    deg1_scores = {i: sum(mult_vec[(helper_fpv_vec == i) & (helper_winner_comb == 1)]) for i in range(len(profile.candidates))}
    deg1_scores[-1] = sum(mult_vec[(helper_fpv_vec < 0) & (helper_winner_comb == 1)])

    Tw = sum(mult_vec[helper_winner_comb == 1])

    deg0_deltas = {i: ((projected_BAL_fpv == i) & (BAL_winner_comb == 0)).astype(int) - ((projected_CVR_fpv == i) & (CVR_winner_comb == 0)).astype(int) for i in range(len(profile.candidates))}
    deg0_deltas[-1] = ((projected_BAL_fpv < 0) & (BAL_winner_comb == 0)).astype(int) - ((projected_CVR_fpv < 0) & (CVR_winner_comb == 0)).astype(int)
    deg1_deltas = {i: ((projected_BAL_fpv == i) & (BAL_winner_comb == 1)).astype(int) - ((projected_CVR_fpv == i) & (CVR_winner_comb == 1)).astype(int) for i in range(len(profile.candidates))}
    deg1_deltas[-1] = ((projected_BAL_fpv < 0) & (BAL_winner_comb == 1)).astype(int) - ((projected_CVR_fpv < 0) & (CVR_winner_comb == 1)).astype(int)
    w_deltas = (BAL_winner_comb.astype(int) - CVR_winner_comb.astype(int))

    node_succesfully_audited = True
    hopeful_candidates = [c for c in range(len(profile.candidates)) if c not in node_info['initial_losers'] and c not in node_info['winner_to_cand']]
    if len(hopeful_candidates) == m-1:
        print(f"Skipping node {str_node_ID} audit since there are as many hopeful candidates as seats.")
        return True

    Cu = N - g - deg1_scores[-1] + (m+1)*epsilon
    Cv = (m+1)*Tw - deg1_scores[-1]

    dg_sample = deg0_deltas[-1].astype(int) 
    dwg_sample = deg1_deltas[-1].astype(int)
    dw_sample = (BAL_winner_comb.astype(int) - CVR_winner_comb.astype(int))

    mu_u = np.mean(dg_sample + dwg_sample)
    mu_v = np.mean((m+1)*dw_sample - dwg_sample)
    k_hat = (Cu-N*mu_u)/(Cv+N*mu_v)

    #if len(hypergeo_var_bounds) != 21:
    #    hypergeo_var_bounds = [alternative_K_upper(N, n, i, alpha)/N for i in range(21)]

    if verbose:
        print(f"info at node {str_node_ID}: deg0_scores= {deg0_scores}, deg1_scores= {deg1_scores}, Tw = {Tw}, g = {g}, k_hat= {k_hat}")
        print(f"Cu= {Cu}, Cv= {Cv}, mu_u= {mu_u}, mu_v= {mu_v}")

    if node_info.get('canonical_winner', None) is not None:
        c = node_info['canonical_winner']
        # just do a cand-to-quota audit for the canonical winner
        C0 = deg0_scores[c] + deg1_scores[c] 
        C1 = deg1_scores[c] + Tw
        
        #d0_sample = deg0_deltas[c].astype(int) + deg1_deltas[c].astype(int)
        di_sample = deg0_deltas[c].astype(int)
        dwi_sample = deg1_deltas[c].astype(int)
        #d1_sample = deg1_deltas[c].astype(int) + w_deltas
        #dw_sample = w_deltas

        mu_0 = np.mean(di_sample+dwi_sample)
        mu_1 = np.mean(dwi_sample + dw_sample)

        grad_M = np.array([N,N-k_hat * N,N*k_hat*(m+1)*(C1+ N*mu_1)/(Cv+N*mu_v)-N*k_hat, N*(C1+ N*mu_1)/(Cv+N*mu_v), (N-k_hat*N)*(C1+ N*mu_1)/(Cv+N*mu_v)])
        samples = [np.array(di_sample), np.array(dwi_sample), np.array(dw_sample), np.array(dg_sample), np.array(dwg_sample)]
        relevant_noisiness = np.sum((samples[0]!=0) | (samples[1]!=0) | (samples[2]!=0) | (samples[3]!=0) | (samples[4]!=0)).astype(int)
        K_u = hypergeo_var_bounds[relevant_noisiness] if len(hypergeo_var_bounds)> relevant_noisiness else alternative_K_upper(N, n, relevant_noisiness, alpha=K_factor_alpha)/N
        covariance_matrix = np.cov(samples, ddof=1)
        for i in range(len(samples)):
            successes = sum(samples[i] != 0)
            if successes <=20:
                #covariance_matrix[i,i] = hypergeo_var_bounds[successes]
                covariance_matrix[i,i] = K_u
        delta_variance = grad_M @ covariance_matrix @ grad_M.T
        delta_variance *= fpc

        M = C0 + N*mu_0 - k_hat* (C1 + N*mu_1)
        M_lower = M - z * math.sqrt(delta_variance)
        if M_lower < 0:
            print(f"AUDIT FAILED at node {str_node_ID}: canonical winner {profile.candidates[c]} might not have quota: M_cq < {M_lower} and estimate M_cq= {M}, k_hat= {k_hat}.")
            node_succesfully_audited = False
        elif verbose:
            print(f"Succesfully checked that candidate {profile.candidates[c]} must have quota at node {str_node_ID}, M_cq > {M_lower} and estimate M_cq= {M}, k_hat= {k_hat}.")
        other_allowable_winners = node_info['other_plausible_winners']
        # check that every other candidate not in other_allowable_winners loses to the canonical winner
        for h in hopeful_candidates:
            if h not in other_allowable_winners and h != c:
                C0 = deg0_scores[h] - deg0_scores[c] + deg1_scores[h] - deg1_scores[c]
                C1 = deg1_scores[h] - deg1_scores[c]

                #d0_sample = deg0_deltas[h].astype(int) - deg0_deltas[c].astype(int) + deg1_deltas[h].astype(int) - deg1_deltas[c].astype(int)
                di_sample = deg0_deltas[h].astype(int)
                dl_sample =deg0_deltas[c].astype(int) 
                dwc_sample = deg1_deltas[h].astype(int) 
                dwl_sample =deg1_deltas[c].astype(int)
                #d1_sample = deg1_deltas[h].astype(int) - deg1_deltas[c].astype(int) 
                mu_0 = np.mean(di_sample - dl_sample+ dwc_sample - dwl_sample)
                mu_1 = np.mean(dwc_sample - dwl_sample)
                grad_M = np.array([N,-N,N-k_hat * N,k_hat*N-N, N*(C1+ N*mu_1)/(Cv+N*mu_v),(N-k_hat*N)*(C1+ N*mu_1)/(Cv+N*mu_v), k_hat * N* (m+1) *(C1+ N*mu_1)/(Cv+N*mu_v)])
                #min_variances = [K/N, K/N, K/N,K/N,K/N,K/N,K/N]
                samples = [np.array(di_sample), np.array(dl_sample), np.array(dwc_sample), np.array(dwl_sample),np.array(dg_sample),np.array(dwg_sample), np.array(dw_sample)]
                relevant_noisiness = np.sum((samples[0] != 0) | (samples[1] != 0) | (samples[2] != 0) | (samples[3] != 0) | (samples[4] != 0) | (samples[5] != 0) | (samples[6] != 0)).astype(int)
                K_u = hypergeo_var_bounds[relevant_noisiness] if len(hypergeo_var_bounds)> relevant_noisiness else alternative_K_upper(N, n, relevant_noisiness, alpha=K_factor_alpha)/N
                covariance_matrix = np.cov(samples, ddof=1)
                for i in range(len(samples)):
                    successes = sum(samples[i] != 0)
                    if successes <=20:
                        covariance_matrix[i,i] = K_u
                delta_variance = grad_M @ covariance_matrix @ grad_M.T
                delta_variance *= fpc

                M = C0 + N*mu_0 - k_hat* (C1 + N*mu_1)
                M_higher = M + z * math.sqrt(delta_variance)
                if M_higher >= 0:
                    print(f"AUDIT FAILED at node {str_node_ID}: candidate {profile.candidates[h]} might not lose to canonical winner {profile.candidates[c]}. M_cl > {M_higher} and estimate M_cl= {M}, k_hat= {k_hat}.")
                    node_succesfully_audited = False
                elif verbose:
                    print(f"Succesfully checked that candidate {profile.candidates[h]} loses to canonical winner {profile.candidates[c]} at node {str_node_ID}, M_cl < {M_higher} and estimate M_cl= {M}, k_hat= {k_hat}.")
        return node_succesfully_audited

    rejected_winners = node_info.get('rejected_winners', None)
    if rejected_winners is None:
        rejected_winners = []
    for c in hopeful_candidates:
        if c not in allowable_winners and c not in rejected_winners:
            C0 = deg0_scores[c] + deg1_scores[c] 
            C1 = deg1_scores[c] + Tw
            
            #d0_sample = deg0_deltas[c].astype(int) + deg1_deltas[c].astype(int)
            di_sample = deg0_deltas[c].astype(int)
            dwi_sample = deg1_deltas[c].astype(int)
            #d1_sample = deg1_deltas[c].astype(int) + w_deltas
            #dw_sample = w_deltas

            mu_0 = np.mean(di_sample+dwi_sample)
            mu_1 = np.mean(dwi_sample + dw_sample)

            grad_M = np.array([N,-N-k_hat * N,N*k_hat*(m+1)*(C1+ N*mu_1)/(Cv+N*mu_v)-N*k_hat, N*(C1+ N*mu_1)/(Cv+N*mu_v), (N-k_hat*N)*(C1+ N*mu_1)/(Cv+N*mu_v)])
            samples = [np.array(di_sample), np.array(dwi_sample), np.array(dw_sample), np.array(dg_sample), np.array(dwg_sample)]
            relevant_noisiness = np.sum((samples[0] != 0) | (samples[1] != 0) | (samples[2] != 0) | (samples[3] != 0) | (samples[4] != 0)).astype(int)
            K_u = hypergeo_var_bounds[relevant_noisiness] if len(hypergeo_var_bounds)> relevant_noisiness else alternative_K_upper(N, n, relevant_noisiness, alpha=K_factor_alpha)/N
            covariance_matrix = np.cov(samples, ddof=1)
            #if verbose:
            #    print("covariance_matrix", covariance_matrix)
            for i in range(len(samples)):
                successes = sum(samples[i] != 0)
                if successes <=20:
                    covariance_matrix[i,i] = K_u
            delta_variance = grad_M @ covariance_matrix @ grad_M.T
            delta_variance *= fpc

            M = C0 + N*mu_0 - k_hat* (C1 + N*mu_1)
            M_upper = M + z * math.sqrt(delta_variance)
            if M_upper > 0:
                print(f"AUDIT FAILED at node {str_node_ID}: candidate {profile.candidates[c]} aka {c} might have quota: M_cq < {M_upper} and estimate M_cq= {M}, k_hat= {k_hat}, allowable_winners = {allowable_winners}, rejected_winners= {rejected_winners}")
                node_succesfully_audited = False
            elif verbose:
                print(f"Succesfully checked that candidate {profile.candidates[c]} does not have quota at node {str_node_ID}, M_cq < {M_upper} and estimate M_cq= {M}, k_hat= {k_hat}")
        if c in rejected_winners:
            w = node_info['probable_winner']
            C0 = deg0_scores[c] - deg0_scores[w] + deg1_scores[c] - deg1_scores[w]
            C1 = deg1_scores[c] - deg1_scores[w]

            #d0_sample = deg0_deltas[c].astype(int) - deg0_deltas[canonical_loser].astype(int) + deg1_deltas[c].astype(int) - deg1_deltas[canonical_loser].astype(int)
            di_sample = deg0_deltas[c].astype(int)
            dl_sample =deg0_deltas[w].astype(int) 
            dwc_sample = deg1_deltas[c].astype(int) 
            dwl_sample =deg1_deltas[w].astype(int)
            #d1_sample = deg1_deltas[c].astype(int) - deg1_deltas[canonical_loser].astype(int) 
            mu_0 = np.mean(di_sample - dl_sample+ dwc_sample - dwl_sample)
            mu_1 = np.mean(dwc_sample - dwl_sample)
            grad_M = np.array([N,-N,N-k_hat * N,k_hat*N-N, N*(C1+ N*mu_1)/(Cv+N*mu_v),(N-k_hat*N)*(C1+ N*mu_1)/(Cv+N*mu_v), k_hat * N* (m+1) *(C1+ N*mu_1)/(Cv+N*mu_v)])
            #min_variances = [K/N, K/N, K/N,K/N,K/N,K/N,K/N]
            samples = [np.array(di_sample), np.array(dl_sample), np.array(dwc_sample), np.array(dwl_sample),np.array(dg_sample),np.array(dwg_sample), np.array(dw_sample)]
            relevant_noisiness = np.sum((samples[0] != 0) | (samples[1] != 0) | (samples[2] != 0) | (samples[3] != 0) | (samples[4] != 0) | (samples[5] != 0) | (samples[6] != 0)).astype(int)
            K_u = hypergeo_var_bounds[relevant_noisiness] if len(hypergeo_var_bounds)> relevant_noisiness else alternative_K_upper(N, n, relevant_noisiness, alpha=K_factor_alpha)/N
            covariance_matrix = np.cov(samples, ddof=1)
            for i in range(len(samples)):
                successes = sum(samples[i] != 0)
                if successes <=20:
                    #covariance_matrix[i,i] = hypergeo_var_bounds[successes]
                    covariance_matrix[i,i] = K_u
            delta_variance = grad_M @ covariance_matrix @ grad_M.T
            delta_variance *= fpc

            M = C0 + N*mu_0 - k_hat* (C1 + N*mu_1)
            M_higher = M + z * math.sqrt(delta_variance)
            if M_higher >= 0:
                print(f"AUDIT FAILED at node {str_node_ID}: rejected winner {profile.candidates[c]} might not lose to probable {profile.candidates[w]}. M_cw < {M_higher} and estimate M_cw= {M}, k_hat= {k_hat}.")
                node_succesfully_audited = False
            elif verbose:
                print(f"Succesfully checked that rejected winner {profile.candidates[c]} loses to probable winner {profile.candidates[w]} at node {str_node_ID}, M_cw < {M_higher} and estimate M_cw= {M}, k_hat= {k_hat}")
        if c not in allowable_losers and node_info.get('canonical_winner', None) is  None:
            C0 = deg0_scores[c] - deg0_scores[canonical_loser] + deg1_scores[c] - deg1_scores[canonical_loser]
            C1 = deg1_scores[c] - deg1_scores[canonical_loser]

            #d0_sample = deg0_deltas[c].astype(int) - deg0_deltas[canonical_loser].astype(int) + deg1_deltas[c].astype(int) - deg1_deltas[canonical_loser].astype(int)
            di_sample = deg0_deltas[c].astype(int)
            dl_sample =deg0_deltas[canonical_loser].astype(int) 
            dwc_sample = deg1_deltas[c].astype(int) 
            dwl_sample =deg1_deltas[canonical_loser].astype(int)
            #d1_sample = deg1_deltas[c].astype(int) - deg1_deltas[canonical_loser].astype(int) 
            mu_0 = np.mean(di_sample - dl_sample+ dwc_sample - dwl_sample)
            mu_1 = np.mean(dwc_sample - dwl_sample)
            grad_M = np.array([N,-N,N-k_hat * N,k_hat*N-N, N*(C1+ N*mu_1)/(Cv+N*mu_v),(N-k_hat*N)*(C1+ N*mu_1)/(Cv+N*mu_v), k_hat * N* (m+1) *(C1+ N*mu_1)/(Cv+N*mu_v)])
            #min_variances = [K/N, K/N, K/N,K/N,K/N,K/N,K/N]
            samples = [np.array(di_sample), np.array(dl_sample), np.array(dwc_sample), np.array(dwl_sample),np.array(dg_sample),np.array(dwg_sample), np.array(dw_sample)]
            relevant_noisiness = np.sum((samples[0] != 0) | (samples[1] != 0) | (samples[2] != 0) | (samples[3] != 0) | (samples[4] != 0) | (samples[5] != 0) | (samples[6] != 0)).astype(int)
            K_u = hypergeo_var_bounds[relevant_noisiness] if len(hypergeo_var_bounds)> relevant_noisiness else alternative_K_upper(N, n, relevant_noisiness, alpha=K_factor_alpha)/N
            covariance_matrix = np.cov(samples, ddof=1)
            for i in range(len(samples)):
                successes = sum(samples[i] != 0)
                if successes <=20:
                    #covariance_matrix[i,i] = hypergeo_var_bounds[successes]
                    covariance_matrix[i,i] = K_u
            delta_variance = grad_M @ covariance_matrix @ grad_M.T
            delta_variance *= fpc

            M = C0 + N*mu_0 - k_hat* (C1 + N*mu_1)
            M_lower = M - z * math.sqrt(delta_variance)
            if M_lower <= 0:
                print(f"AUDIT FAILED at node {str_node_ID}: candidate {profile.candidates[c]} might lose to {profile.candidates[canonical_loser]}. M_cl > {M_lower} and estimate M_cl= {M}, k_hat= {k_hat}.")
                node_succesfully_audited = False
            elif verbose:
                print(f"Succesfully checked that candidate {profile.candidates[c]} does not lose to {profile.candidates[canonical_loser]} at node {str_node_ID}, M_cl > {M_lower} and estimate M_cl= {M}, k_hat= {k_hat}")
    return node_succesfully_audited

def deg2_node_audit_alternative(str_node_ID, graph, deg2_kwargs, K_factor_alpha=0.005, hypergeo_var_bounds=[], verbose = False):
    _N = deg2_kwargs['N']
    _m = deg2_kwargs['m']
    BAL = deg2_kwargs['BAL']
    CVR = deg2_kwargs['CVR']
    _epsilon = deg2_kwargs['epsilon']
    K = deg2_kwargs['K']
    _n = deg2_kwargs['n']
    z = deg2_kwargs['z']
    alpha = deg2_kwargs['alpha']- K_factor_alpha
    if alpha <= 0:
        raise ValueError("alpha - K_factor_alpha must be positive in deg2_node_audit_alternative")
    profile = deg2_kwargs['profile']
    candidate_strs = profile.candidates
    num_cands = len(candidate_strs)

    if _m!= 3:
        raise ValueError("deg2_node_audit currently only supports m=3")
    
    node_info = graph.lookup_node(str_node_ID)
    remaining_candidates = [c for c in range(num_cands) if c not in node_info['initial_losers'] and c not in node_info['winner_to_cand']]
    
    if len(remaining_candidates) == 1:
        if verbose:
            print(f"Node {str_node_ID} only has one remaining candidate {candidate_strs[remaining_candidates[0]]}, automatically passing audit.")
        return True
    elif len(remaining_candidates) == 0:
        raise ValueError(f"Node {str_node_ID} has no remaining candidates, something went wrong.")
    else:
        canonical_loser = node_info.get('canonical_loser', None)
        if canonical_loser is None:
            raise ValueError("deg2_node_audit requires a canonical_loser in node_info")
        
    
    depth = node_info['layer']
    next_layer_codes = graph.nodes_by_layer[depth + 1]
    allowable_losers= []
    for code in next_layer_codes:
        decoded = graph.decode_node_string(code)
        if set(decoded["winner_to_cand"]) >= set(node_info["winner_to_cand"]) and set(decoded["initial_losers"]) >= set(node_info["initial_losers"]):
            if len(decoded["winner_to_cand"]) == len(node_info["winner_to_cand"]) + 1 and verbose:
                print(f"Skipping edge from node {str_node_ID} to {code} since deg2_node_audit only handles loser expansions.")
            elif len(decoded["initial_losers"]) == len(node_info["initial_losers"]) + 1:
                new_loser = list(set(decoded["initial_losers"]) - set(node_info["initial_losers"]))
                allowable_losers.append(new_loser[0])

    projected_BAL_fpv, BAL_winner_comb = project_matrix_onto_election_state(
        BAL,
        num_cands=len(profile.candidates),
        m=_m,
        losers=node_info['initial_losers'],
        winner_to_cand=node_info['winner_to_cand'])
    projected_CVR_fpv, CVR_winner_comb = project_matrix_onto_election_state(
        CVR,
        num_cands=len(profile.candidates),
        m=_m,
        losers=node_info['initial_losers'],
        winner_to_cand=node_info['winner_to_cand'])
    
    ## RESERVED SYMBOLS: DO NOT USE FOR OTHER PURPOSES
    #T1, T2, T12, T21 = sp.symbols('T1 T2 T12 T21', real=True)
    #mu1, mu2, mu12, mu21, mug = sp.symbols('mu1 mu2 mu12 mu21 mug', real=True)
    #t1, t2, t12, t21 = sp.symbols('t1 t2 t12 t21', real=True)
    #nu1, nu2, nu12, nu21 = sp.symbols('nu1 nu2 nu12 nu21', real=True)
    #m, N, g, epsilon = sp.symbols('m N g epsilon', real=True)
    #
    #k1, k2 = sp.symbols('k1 k2', real=True)
    #
    ##CANDIDATE SPECIFIC SYMBOLS
    #T_i, T_j = sp.symbols('T_i T_j', real=True)
    #T_w1i, T_w1j = sp.symbols('T_w1i T_w1j', real=True)
    #T_w2i, T_w2j = sp.symbols('T_w2i T_w2j', real=True)
    #T_w1w2i, T_w1w2j = sp.symbols('T_w1w2i T_w1w2j', real=True)
    #mu_i, mu_j = sp.symbols('mu_i mu_j', real=True)
    #mu_w1i, mu_w1j = sp.symbols('mu_w1i mu_w1j', real=True)
    #mu_w2i, mu_w2j = sp.symbols('mu_w2i mu_w2j', real=True)
    #mu_w1w2i, mu_w1w2j = sp.symbols('mu_w1w2i mu_w1w2j', real=True)

    # dicts with symbolic keys
    mu_values = {}
    Tcheck_values = {N: _N, m: _m, epsilon: _epsilon} 
    k_values = {}
    
    helper_fpv_vec, helper_winner_comb = project_matrix_onto_election_state(deg2_kwargs['ballot_matrix'], num_cands=len(profile.candidates), m=_m, losers=node_info['initial_losers'], winner_to_cand=node_info['winner_to_cand'])

    fpc = (_N - _n) / (_n * (_N- 1))
    mult_vec = deg2_kwargs['mult_vec']

    score_transfer_masks = {i: helper_winner_comb == i for i in [0,1,2,4,6]}
    score_fpv_masks = {i: helper_fpv_vec == i for i in remaining_candidates}
    score_fpv_masks[-1] = helper_fpv_vec < 0

    Tcheck_values[g] = np.sum(mult_vec[score_fpv_masks[-1] & (score_transfer_masks[0])]) + deg2_kwargs['num_ghosts']

    deg0_scores = {i: sum(mult_vec[score_fpv_masks[i] & score_transfer_masks[0]]) for i in remaining_candidates}
    deg0_scores[-1] = sum(mult_vec[score_fpv_masks[-1] & score_transfer_masks[0]])
    w1_scores = {i: sum(mult_vec[score_fpv_masks[i] & score_transfer_masks[1]]) for i in remaining_candidates}
    w1_scores[-1] = sum(mult_vec[score_fpv_masks[-1] & score_transfer_masks[1]])
    w2_scores = {i: sum(mult_vec[score_fpv_masks[i] & score_transfer_masks[2]]) for i in remaining_candidates}
    w2_scores[-1] = sum(mult_vec[score_fpv_masks[-1] & score_transfer_masks[2]])
    w1w2_scores = {i: sum(mult_vec[score_fpv_masks[i] & (score_transfer_masks[4]| score_transfer_masks[6])]) for i in remaining_candidates} # we don't care what order the transfer happens in
    #w1w2_scores[-1] = sum(mult_vec[score_fpv_masks[-1] & (score_transfer_masks[4]| score_transfer_masks[6])])

    Tcheck_values[T1] = sum(mult_vec[score_transfer_masks[1] | score_transfer_masks[4]])
    Tcheck_values[T2] = sum(mult_vec[score_transfer_masks[2]| score_transfer_masks[6]])
    Tcheck_values[T12] = sum(mult_vec[score_transfer_masks[4]])
    Tcheck_values[T21] = sum(mult_vec[score_transfer_masks[6]])
    Tcheck_values[t1] = w1_scores[-1]
    Tcheck_values[t2] = w2_scores[-1]
    Tcheck_values[t12] = sum(mult_vec[score_fpv_masks[-1] & score_transfer_masks[4]])
    Tcheck_values[t21] = sum(mult_vec[score_fpv_masks[-1] & score_transfer_masks[6]])

    # j is always the canonical loser, so fill those values in now
    Tcheck_values[T_j] = deg0_scores[canonical_loser] 
    Tcheck_values[T_w1j] = w1_scores[canonical_loser]
    Tcheck_values[T_w2j] = w2_scores[canonical_loser]
    Tcheck_values[T_w1w2j] = w1w2_scores[canonical_loser]

    BAL_fpv_masks = {i: (projected_BAL_fpv == i) for i in remaining_candidates}
    BAL_fpv_masks[-1] = (projected_BAL_fpv < 0)
    CVR_fpv_masks = {i: (projected_CVR_fpv == i) for i in remaining_candidates}
    CVR_fpv_masks[-1] = (projected_CVR_fpv < 0)
    BAL_transfer_masks = {i: (BAL_winner_comb == i) for i in [0,1,2,4,6]}
    CVR_transfer_masks = {i: (CVR_winner_comb == i) for i in [0,1,2,4,6]}

    deg0_deltas = {i: (BAL_fpv_masks[i] & BAL_transfer_masks[0]).astype(int) - (CVR_fpv_masks[i] & CVR_transfer_masks[0]).astype(int) for i in remaining_candidates}
    deg0_deltas[-1] = (BAL_fpv_masks[-1] & BAL_transfer_masks[0]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[0]).astype(int)
    w1_deltas = {i: (BAL_fpv_masks[i] & BAL_transfer_masks[1]).astype(int) - (CVR_fpv_masks[i] & CVR_transfer_masks[1]).astype(int) for i in remaining_candidates}
    w1_deltas[-1] = (BAL_fpv_masks[-1] & BAL_transfer_masks[1]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[1]).astype(int)
    w2_deltas = {i: (BAL_fpv_masks[i] & BAL_transfer_masks[2]).astype(int) - (CVR_fpv_masks[i] & CVR_transfer_masks[2]).astype(int) for i in remaining_candidates}
    w2_deltas[-1] = (BAL_fpv_masks[-1] & BAL_transfer_masks[2]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[2]).astype(int)
    w1w2_deltas = {i: ((BAL_fpv_masks[i] & (BAL_transfer_masks[4] | BAL_transfer_masks[6])).astype(int) - (CVR_fpv_masks[i] & (CVR_transfer_masks[4] | CVR_transfer_masks[6])).astype(int)) for i in remaining_candidates}
    
    mug_deltas = (BAL_fpv_masks[-1] & BAL_transfer_masks[0]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[0]).astype(int)
    mu1_deltas = BAL_transfer_masks[1].astype(int) - CVR_transfer_masks[1].astype(int)
    mu2_deltas = BAL_transfer_masks[2].astype(int) - CVR_transfer_masks[2].astype(int)
    nu1_deltas = (BAL_fpv_masks[-1] & BAL_transfer_masks[1]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[1]).astype(int)
    nu2_deltas = (BAL_fpv_masks[-1] & BAL_transfer_masks[2]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[2]).astype(int)
    mu12_deltas = BAL_transfer_masks[4].astype(int) - CVR_transfer_masks[4].astype(int)
    mu21_deltas = BAL_transfer_masks[6].astype(int) - CVR_transfer_masks[6].astype(int)
    nu12_deltas = (BAL_fpv_masks[-1] & BAL_transfer_masks[4]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[4]).astype(int)
    nu21_deltas = (BAL_fpv_masks[-1] & BAL_transfer_masks[6]).astype(int) - (CVR_fpv_masks[-1] & CVR_transfer_masks[6]).astype(int)
    
    mu_values[mug] = np.mean(mug_deltas.astype(int))
    mu_values[mu1] = np.mean(mu1_deltas.astype(int))
    mu_values[mu2] = np.mean(mu2_deltas.astype(int))
    mu_values[mu12] = np.mean(mu12_deltas.astype(int))
    mu_values[mu21] = np.mean(mu21_deltas.astype(int))
    mu_values[nu1] = np.mean(nu1_deltas.astype(int))
    mu_values[nu2] = np.mean(nu2_deltas.astype(int))
    mu_values[nu12] = np.mean(nu12_deltas.astype(int))
    mu_values[nu21] = np.mean(nu21_deltas.astype(int))

    if verbose:
        print("Parameters for K: Tcheclk_values:")
        for key, value in Tcheck_values.items():
            print(f"  {key}: {value}")
    _k1, _k2 = eval_least_positive_root(mu_values, Tcheck_values)
    k_values[k1] = _k1
    k_values[k2] = _k2

    # again, j is always the canonical loser, so fill out the mu_*j
    mu_values[mu_j] = np.mean(deg0_deltas[canonical_loser].astype(int))
    mu_values[mu_w1j] = np.mean(w1_deltas[canonical_loser].astype(int))
    mu_values[mu_w2j] = np.mean(w2_deltas[canonical_loser].astype(int))
    mu_values[mu_w1w2j] = np.mean(w1w2_deltas[canonical_loser].astype(int))


    #if len(hypergeo_var_bounds) != 21:
    #    hypergeo_var_bounds = [alternative_K_upper(_N, _n, i, alpha)/_N for i in range(21)]

    node_succesfully_audited = True

    if verbose:
        print(f"info at node {str_node_ID}: k1= {_k1}, k2= {_k2}")
    
    for c in remaining_candidates:
        if c not in allowable_losers and c != canonical_loser:
            Tcheck_values[T_i] = deg0_scores[c] 
            Tcheck_values[T_w1i] = w1_scores[c]
            Tcheck_values[T_w2i] = w2_scores[c]
            Tcheck_values[T_w1w2i] = w1w2_scores[c]

            mu_values[mu_i] = np.mean(deg0_deltas[c].astype(int))
            mu_values[mu_w1i] = np.mean(w1_deltas[c].astype(int))
            mu_values[mu_w2i] = np.mean(w2_deltas[c].astype(int))
            mu_values[mu_w1w2i] = np.mean(w1w2_deltas[c].astype(int))

            # theta is: mug, mu1, mu2, mu12, mu21, nu1, nu2, nu12, nu21, mu_i, mu_j, mu_w1i, mu_w1j, mu_w2i, mu_w2j, mu_w1w2i, mu_w1w2j

            data = np.vstack([
                mug_deltas,
                mu1_deltas,
                mu2_deltas,
                mu12_deltas,
                mu21_deltas,
                nu1_deltas,
                nu2_deltas,
                nu12_deltas,
                nu21_deltas,
                deg0_deltas[c],
                deg0_deltas[canonical_loser],
                w1_deltas[c],
                w1_deltas[canonical_loser],
                w2_deltas[c],
                w2_deltas[canonical_loser],
                w1w2_deltas[c],
                w1w2_deltas[canonical_loser],
            ]).astype(float)
            relevant_noisiness = sum((data != 0).any(axis=0).astype(int))
            K_u = hypergeo_var_bounds[relevant_noisiness] if len(hypergeo_var_bounds)> relevant_noisiness else alternative_K_upper(_N, _n, relevant_noisiness, alpha=K_factor_alpha)/_N

            covariance_matrix = np.cov(data, ddof=1)
            for i in range(len(data)):
                successes = sum(data[i] != 0)
                if successes <=20:
                    #covariance_matrix[i,i] = hypergeo_var_bounds[successes]
                    covariance_matrix[i,i] = K_u
            grad_M = eval_grad_M(mu_values, Tcheck_values, k_values)
            delta_variance = grad_M @ covariance_matrix @ grad_M.T
            delta_variance *= fpc

            M = eval_M(mu_values, Tcheck_values, k_values)

            M_lower = M - z * math.sqrt(abs(delta_variance))
            if M_lower <= 0:
                print(f"AUDIT FAILED at node {str_node_ID}: candidate {profile.candidates[c]} might lose to {profile.candidates[canonical_loser]}. M_cl < {M_lower} and estimate M_cl= {M}, k1= {k_values[k1]}, k2= {k_values[k2]}.")
                print(M_lower)
                node_succesfully_audited = False
            elif verbose:
                print(f"Succesfully checked that candidate {profile.candidates[c]} does not lose to {profile.candidates[canonical_loser]} at node {str_node_ID}, M_cl > {M_lower} and estimate M_cl= {M}, k1= {k_values[k1]}, k2= {k_values[k2]}.")
    return node_succesfully_audited
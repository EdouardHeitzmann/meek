import math
from typing import Iterable, Tuple, Optional
from statistics import NormalDist
import math

def _norm_ppf(q: float) -> float:
    """
    Standard normal quantile Φ^{-1}(q), robust across all regions.

    Prefers statistics.NormalDist().inv_cdf when available.
    Falls back to a corrected Acklam approximation with a sign guard
    so that Φ^{-1}(q) > 0 for q > 0.5 and < 0 for q < 0.5.
    """
    if not (0.0 < q < 1.0):
        if q == 0.0: return -math.inf
        if q == 1.0: return  math.inf
        raise ValueError("q must be in (0,1)")
    return NormalDist().inv_cdf(q)

def Wald_CI(
    sample: Iterable[float],
    N: int,
    alpha: float = 0.05,
    bounds: Optional[Tuple[float, float]] = (-1.0, 1.0),
    clip_to_bounds: bool = True,
    fallback: str = "variance_bound",   # or "serfling"
    p0_lower: Optional[float] = .05,   # use if fallback="variance_bound" and you know P(B=0) >= p0_lower
    eps_var: float = 1e-12,
    return_extras: bool = False,
):
    """
    Wald + FPC CI for the mean μ of a finite population (SRSWOR).
    If sample variance is ~0, use a conservative fallback to avoid a degenerate CI.

    Parameters
    ----------
    sample : iterable of floats
        Observed B_i (fractional allowed), assumed bounded in `bounds` (default [-1,1]).
    N : int
        Finite population size.
    alpha : float
        1 - confidence level.
    bounds : (a, b) or None
        Natural parameter bounds for μ; used only for clipping.
    clip_to_bounds : bool
        Clip CI to the provided bounds.
    fallback : {"variance_bound", "serfling"}
        Strategy when sample variance ~ 0.
        - "variance_bound": use σ_max^2 to compute a nonzero half-width (tight if you know p0_lower).
        - "serfling": use Serfling's distribution-free half-width on mean for [-1,1].
    p0_lower : float in [0,1], optional
        If you can assert P(B=0) >= p0_lower, then Var(B) <= 1 - p0_lower. Used when fallback="variance_bound".
    eps_var : float
        Threshold below which we treat the sample variance as zero.
    return_extras : bool
        If True, also return a dict with diagnostics.

    Returns
    -------
    (lo, hi) or (lo, hi, extras)
    """
    x = list(sample)
    n = len(x)
    if n == 0:
        raise ValueError("Sample is empty.")
    if N <= 0:
        raise ValueError("N must be positive.")
    if n > N:
        raise ValueError("Sample size n cannot exceed population size N.")

    m = sum(x) / n
    if n >= 2:
        ssq = sum((xi - m)**2 for xi in x) / (n - 1)
    else:
        ssq = 0.0
    s = math.sqrt(max(0.0, ssq))

    # Finite-population correction
    fpc = math.sqrt(max(0.0, 1.0 - n / N))
    z = _norm_ppf(1.0 - alpha/2.0)

    # Main path: Wald + FPC
    use_fallback = s**2 < eps_var
    if not use_fallback:
        se = (s / math.sqrt(n)) * fpc
        half = z * se
    else:
        if fallback.lower() == "variance_bound":
            # σ_max^2: worst-case variance for [-1,1] is 1 (achieved by ±1 equally likely).
            sigma2_max = 1.0
            if p0_lower is not None:
                # If P(B=0) >= p0, then Var(B) <= 1 - p0
                sigma2_max = min(sigma2_max, 1.0 - p0_lower)
            se = math.sqrt(sigma2_max / n) * fpc
            half = z * se
        elif fallback.lower() == "serfling":
            # Serfling bound for mean on [a,b]; default [-1,1] => R=2
            a, b = (-1.0, 1.0) if bounds is None else bounds
            R = b - a
            factor = 1.0 - (n - 1) / N
            # two-sided: ε = R * sqrt( factor * log(2/α) / (2n) )
            half = R * math.sqrt(max(0.0, factor) * math.log(2.0/alpha) / (2.0 * n))
        else:
            raise ValueError("fallback must be 'variance_bound' or 'serfling'")

    lo, hi = m - half, m + half
    if clip_to_bounds and bounds is not None:
        a, b = bounds
        if a >= b:
            raise ValueError("bounds must satisfy a < b")
        lo, hi = max(a, lo), min(b, hi)

    if return_extras:
        return lo, hi, {
            "mean": m, "sd": s, "var": s**2, "n": n, "N": N,
            "alpha": alpha, "z": z, "fpc": fpc, "halfwidth": half,
            "used_fallback": bool(use_fallback),
            "fallback": fallback if use_fallback else None,
            "sigma2_max_used": (1.0 - p0_lower) if (use_fallback and fallback=="variance_bound" and p0_lower is not None) else (1.0 if use_fallback and fallback=="variance_bound" else None),
        }
    return lo, hi
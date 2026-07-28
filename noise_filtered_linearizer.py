"""Low-dimensional symbolic prototype for noise-filtered STV linearizers.

The symbolic part is exact (SymPy). Numerical optimization uses SciPy
differential evolution, so a reported minimum is a candidate global minimum
rather than a mathematically certified lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Mapping, Sequence

import numpy as np
import sympy as sp
from scipy.optimize import NonlinearConstraint, OptimizeResult, differential_evolution

__version__ = "0.2.0-l1"

__version__ = "0.2.0-l1"

COLUMNS = ("c", "l", "o")


@dataclass(frozen=True)
class SymbolicMarginModel:
    degree: int
    symbols: np.ndarray
    quota: sp.Expr
    winner_tallies: tuple[sp.Expr, ...]
    transfer_values: tuple[sp.Expr, ...]
    margin: sp.Expr

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        """All ballot-total variables in row-major order."""
        return tuple(self.symbols.reshape(-1))


@dataclass(frozen=True)
class BoxMinimum:
    expression: sp.Expr
    minimum: float
    point: np.ndarray
    bounds: tuple[tuple[float, float], ...]
    feasible: bool
    optimizer_result: OptimizeResult


@dataclass(frozen=True)
class L1BallMinimum:
    expression: sp.Expr
    minimum: float
    point: np.ndarray
    center: np.ndarray
    radius: float
    l1_distance: float
    feasible: bool
    optimizer_result: OptimizeResult
    restart_minima: tuple[float, ...]


def _subset_label(mask: int, degree: int) -> str:
    members = [str(j) for j in range(degree) if mask & (1 << j)]
    return "".join(members) if members else "empty"


def _infer_degree(symbol_array: np.ndarray) -> int:
    arr = np.asarray(symbol_array, dtype=object)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Expected an array of shape (2**d, 3); received {arr.shape}."
        )
    rows = arr.shape[0]
    degree = int(log2(rows)) if rows > 0 else -1
    if degree < 0 or 2**degree != rows:
        raise ValueError(f"The number of rows ({rows}) is not a power of two.")
    return degree


def make_symbolic_array(
    degree: int,
    *,
    prefix: str = "t",
    nonnegative: bool = True,
) -> np.ndarray:
    """Construct the (2**degree, 3) array [c, l, o] of ballot totals.

    Row ``mask`` represents the winner subset encoded by the set bits of
    ``mask``.  Thus, at degree 3, row 4 == 0b100 represents {w_2}, and row
    6 == 0b110 represents {w_1, w_2}.
    """
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")

    out = np.empty((2**degree, 3), dtype=object)
    for mask in range(2**degree):
        subset = _subset_label(mask, degree)
        for col, candidate in enumerate(COLUMNS):
            out[mask, col] = sp.Symbol(
                f"{prefix}_{subset}_{candidate}", nonnegative=nonnegative
            )
    return out


def hardcoded_margin(
    symbol_array: np.ndarray,
    quota: sp.Expr,
) -> SymbolicMarginModel:
    """Return the explicitly written degree-0 or degree-1 model."""
    t = np.asarray(symbol_array, dtype=object)
    degree = _infer_degree(t)
    q = sp.sympify(quota)

    if degree == 0:
        margin = t[0, 0] - t[0, 1]
        return SymbolicMarginModel(0, t, q, (), (), margin)

    if degree == 1:
        tally_0 = sp.Add(*t[1, :])
        tau_0 = (tally_0 - q) / tally_0
        margin = t[0, 0] - t[0, 1] + tau_0 * (t[1, 0] - t[1, 1])
        return SymbolicMarginModel(1, t, q, (tally_0,), (tau_0,), margin)

    raise ValueError("hardcoded_margin is defined only for degree 0 or degree 1")


def build_recursive_margin(
    symbol_array: np.ndarray,
    quota: sp.Expr,
) -> SymbolicMarginModel:
    """Recursively construct winner tallies, transfer values, and M_cl.

    Winners are ordered w_0, ..., w_{d-1}.  For winner w_j, a final-prefix
    row contributes iff bit j is set.  Its weight at the time w_j is seated is
    the product of transfer values for the *earlier* set bits only.

    The final c-vs-l margin uses the product of transfer values for every set
    bit in the row.
    """
    t = np.asarray(symbol_array, dtype=object)
    degree = _infer_degree(t)
    q = sp.sympify(quota)
    row_totals = tuple(sp.Add(*t[mask, :]) for mask in range(2**degree))

    tallies: list[sp.Expr] = []
    taus: list[sp.Expr] = []

    for j in range(degree):
        terms: list[sp.Expr] = []
        for mask in range(2**degree):
            if not (mask & (1 << j)):
                continue
            earlier_weight = sp.prod(
                taus[i] for i in range(j) if mask & (1 << i)
            )
            terms.append(earlier_weight * row_totals[mask])

        tally_j = sp.Add(*terms)
        tau_j = (tally_j - q) / tally_j
        tallies.append(tally_j)
        taus.append(tau_j)

    margin_terms: list[sp.Expr] = []
    for mask in range(2**degree):
        transfer_weight = sp.prod(
            taus[j] for j in range(degree) if mask & (1 << j)
        )
        margin_terms.append(transfer_weight * (t[mask, 0] - t[mask, 1]))

    margin = sp.Add(*margin_terms)
    return SymbolicMarginModel(
        degree=degree,
        symbols=t,
        quota=q,
        winner_tallies=tuple(tallies),
        transfer_values=tuple(taus),
        margin=margin,
    )


def resolve_coordinate(
    model: SymbolicMarginModel,
    coordinate: sp.Symbol | tuple[int, int],
) -> sp.Symbol:
    """Resolve either a Symbol or a (row, column) index to a model variable."""
    if isinstance(coordinate, tuple):
        if len(coordinate) != 2:
            raise ValueError("A coordinate index must be a (row, column) pair.")
        variable = model.symbols[coordinate]
    else:
        variable = coordinate

    if variable not in model.variables:
        raise ValueError(f"{variable!r} is not a coordinate of this model.")
    return variable


def margin_partial(
    model: SymbolicMarginModel,
    coordinate: sp.Symbol | tuple[int, int],
    *,
    simplify: bool = False,
    print_expression: bool = True,
) -> sp.Expr:
    """Take and optionally pretty-print one coordinate partial of M_cl."""
    variable = resolve_coordinate(model, coordinate)
    derivative = sp.diff(model.margin, variable)
    if simplify:
        derivative = sp.factor(sp.cancel(derivative))

    if print_expression:
        print(f"∂M_cl/∂{variable} =")
        sp.pprint(derivative, use_unicode=True)
    return derivative


def directional_derivative(
    model: SymbolicMarginModel,
    direction: np.ndarray,
    *,
    simplify: bool = False,
) -> sp.Expr:
    """Return grad(M_cl) dot direction for an array shaped like the totals."""
    theta = np.asarray(direction, dtype=object)
    if theta.shape != model.symbols.shape:
        raise ValueError(
            f"direction has shape {theta.shape}; expected {model.symbols.shape}"
        )

    expression = sp.Add(
        *(
            sp.sympify(theta[index]) * sp.diff(model.margin, model.symbols[index])
            for index in np.ndindex(model.symbols.shape)
        )
    )
    return sp.factor(sp.cancel(expression)) if simplify else expression


def _numeric_expression(
    model: SymbolicMarginModel,
    expression: sp.Expr,
    parameter_values: Mapping[sp.Symbol, float] | None,
) -> sp.Expr:
    substitutions = dict(parameter_values or {})
    numeric = sp.sympify(expression).subs(substitutions)
    extra_symbols = numeric.free_symbols.difference(model.variables)
    if extra_symbols:
        names = ", ".join(sorted(map(str, extra_symbols)))
        raise ValueError(
            "The optimized expression still has non-coordinate symbols: "
            f"{names}. Supply them through parameter_values."
        )
    return numeric


def minimize_expression_over_box(
    model: SymbolicMarginModel,
    expression: sp.Expr,
    base_point: np.ndarray,
    radius: float | np.ndarray,
    *,
    parameter_values: Mapping[sp.Symbol, float] | None = None,
    nonnegative: bool = True,
    require_winner_quota: bool = False,
    seed: int = 0,
    maxiter: int = 600,
    popsize: int = 20,
    tol: float = 1e-9,
) -> BoxMinimum:
    """Numerically minimize a symbolic expression over a coordinate box.

    Bounds are base_point +/- radius, intersected with the nonnegative orthant
    by default.  If ``require_winner_quota`` is true, every recursively defined
    seating tally is constrained to be at least q.

    This uses differential evolution followed by SciPy's polishing step.  It is
    useful for exploration, but it is not a rigorous interval certificate.
    """
    base = np.asarray(base_point, dtype=float)
    if base.shape != model.symbols.shape:
        raise ValueError(f"base_point has shape {base.shape}; expected {model.symbols.shape}")

    radii = np.asarray(radius, dtype=float)
    if radii.ndim == 0:
        radii = np.full(base.shape, float(radii))
    else:
        radii = np.broadcast_to(radii, base.shape).copy()
    if np.any(radii < 0):
        raise ValueError("radius must be nonnegative")

    lower = base - radii
    upper = base + radii
    if nonnegative:
        lower = np.maximum(lower, 0.0)
    if np.any(lower > upper):
        raise ValueError("At least one box coordinate has lower bound above upper bound.")

    flat_lower = lower.reshape(-1)
    flat_upper = upper.reshape(-1)
    bounds = tuple(zip(flat_lower, flat_upper, strict=True))
    numeric_expr = _numeric_expression(model, expression, parameter_values)
    objective_lambda = sp.lambdify(model.variables, numeric_expr, "numpy", cse=True)

    numeric_tallies: tuple[sp.Expr, ...] = ()
    quota_value: float | None = None
    if require_winner_quota and model.degree > 0:
        numeric_tallies = tuple(
            _numeric_expression(model, tally, parameter_values)
            for tally in model.winner_tallies
        )
        quota_expr = model.quota.subs(dict(parameter_values or {}))
        if quota_expr.free_symbols:
            names = ", ".join(sorted(map(str, quota_expr.free_symbols)))
            raise ValueError(f"Quota is still symbolic ({names}); supply parameter_values.")
        quota_value = float(quota_expr)

    # Drop coordinates that affect neither the objective nor the constraints.
    # This matters in low-degree examples: flat dimensions can prevent a global
    # optimizer's convergence test from firing even when the minimum is clear.
    active_symbols = set(numeric_expr.free_symbols)
    for tally in numeric_tallies:
        active_symbols.update(tally.free_symbols)
    active_indices = tuple(
        i for i, variable in enumerate(model.variables) if variable in active_symbols
    )
    full_template = np.clip(base.reshape(-1), flat_lower, flat_upper)

    def expand(active_x: np.ndarray) -> np.ndarray:
        full_x = full_template.copy()
        if active_indices:
            full_x[np.asarray(active_indices, dtype=int)] = active_x
        return full_x

    def objective(active_x: np.ndarray) -> float:
        full_x = expand(active_x)
        try:
            value = float(np.asarray(objective_lambda(*full_x)))
        except (FloatingPointError, OverflowError, ZeroDivisionError, TypeError, ValueError):
            return 1e100
        return value if np.isfinite(value) else 1e100

    constraints: tuple[NonlinearConstraint, ...] = ()
    tally_lambda = None
    if numeric_tallies:
        tally_lambda = sp.lambdify(model.variables, numeric_tallies, "numpy", cse=True)

        def tally_vector(active_x: np.ndarray) -> np.ndarray:
            full_x = expand(active_x)
            values = np.asarray(tally_lambda(*full_x), dtype=float).reshape(-1)
            return np.where(np.isfinite(values), values, -1e100)

        constraints = (
            NonlinearConstraint(
                tally_vector,
                lb=np.full(model.degree, quota_value),
                ub=np.full(model.degree, np.inf),
            ),
        )

    if active_indices:
        active_bounds = tuple(bounds[i] for i in active_indices)
        result = differential_evolution(
            objective,
            bounds=active_bounds,
            constraints=constraints,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            polish=True,
            updating="immediate",
            workers=1,
        )
        active_x = np.asarray(result.x, dtype=float)
        full_x = expand(active_x)
        result["active_x"] = active_x
        result["active_indices"] = active_indices
        result["x"] = full_x
    else:
        full_x = full_template
        value = objective(np.empty(0, dtype=float))
        result = OptimizeResult(
            x=full_x,
            fun=value,
            success=True,
            message="The expression and constraints are constant on this box.",
            active_x=np.empty(0, dtype=float),
            active_indices=(),
        )

    point = full_x.reshape(model.symbols.shape)
    feasible = bool(np.all(point >= lower - 1e-8) and np.all(point <= upper + 1e-8))
    if feasible and tally_lambda is not None:
        final_tallies = np.asarray(tally_lambda(*full_x), dtype=float).reshape(-1)
        feasible = bool(np.all(final_tallies >= float(quota_value) - 1e-7))

    return BoxMinimum(
        expression=numeric_expr,
        minimum=float(result.fun),
        point=point,
        bounds=bounds,
        feasible=feasible,
        optimizer_result=result,
    )


def minimize_coordinate_partial_over_box(
    model: SymbolicMarginModel,
    base_point: np.ndarray,
    radius: float | np.ndarray,
    coordinate: sp.Symbol | tuple[int, int],
    **optimizer_kwargs,
) -> BoxMinimum:
    """Minimize one coordinate partial of M_cl over the induced box."""
    derivative = margin_partial(
        model, coordinate, simplify=False, print_expression=False
    )
    return minimize_expression_over_box(
        model, derivative, base_point, radius, **optimizer_kwargs
    )


def minimize_directional_derivative_over_box(
    model: SymbolicMarginModel,
    base_point: np.ndarray,
    radius: float | np.ndarray,
    direction: np.ndarray,
    **optimizer_kwargs,
) -> BoxMinimum:
    """Minimize grad(M_cl) dot direction over the induced box."""
    derivative = directional_derivative(model, direction)
    return minimize_expression_over_box(
        model, derivative, base_point, radius, **optimizer_kwargs
    )



def _contract_active_point_to_l1_ball(
    candidate: np.ndarray,
    anchor: np.ndarray,
    expand,
    center_flat: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Contract ``candidate`` toward a feasible anchor until it enters the ball."""
    if np.sum(np.abs(expand(candidate) - center_flat)) <= radius:
        return candidate

    lo, hi = 0.0, 1.0
    for _ in range(64):
        alpha = 0.5 * (lo + hi)
        trial = anchor + alpha * (candidate - anchor)
        if np.sum(np.abs(expand(trial) - center_flat)) <= radius:
            lo = alpha
        else:
            hi = alpha
    return anchor + lo * (candidate - anchor)


def minimize_expression_over_l1_ball(
    model: SymbolicMarginModel,
    expression: sp.Expr,
    base_point: np.ndarray,
    radius: float,
    *,
    parameter_values: Mapping[sp.Symbol, float] | None = None,
    nonnegative: bool = True,
    require_winner_quota: bool = False,
    seed: int = 0,
    restarts: int = 3,
    maxiter: int = 600,
    popsize: int = 20,
    tol: float = 1e-9,
    polish: bool = True,
    feasibility_tol: float = 1e-7,
) -> L1BallMinimum:
    """Numerically minimize a symbolic expression over an L1 ball.

    The domain is

        sum_i abs(x_i - base_point_i) <= radius.

    It is intersected with the nonnegative orthant by default. Winner-tally
    quota constraints are disabled by default; hence the search may include
    points with ``T_j < q``. This is a stochastic numerical search, not a
    certified global lower bound.
    """
    center = np.asarray(base_point, dtype=float)
    if center.shape != model.symbols.shape:
        raise ValueError(
            f"base_point has shape {center.shape}; expected {model.symbols.shape}"
        )
    if np.ndim(radius) != 0 or not np.isfinite(radius) or float(radius) < 0:
        raise ValueError("radius must be a finite nonnegative scalar")
    if restarts < 1:
        raise ValueError("restarts must be at least 1")
    radius = float(radius)

    center_flat = center.reshape(-1)
    flat_lower = center_flat - radius
    flat_upper = center_flat + radius
    if nonnegative:
        flat_lower = np.maximum(flat_lower, 0.0)
    if np.any(flat_lower > flat_upper):
        raise ValueError("The L1 ball does not intersect the requested bounds.")

    # Nearest point satisfying coordinate bounds; this is a feasible L1 anchor
    # exactly when the bounded domain intersects the ball.
    full_template = np.clip(center_flat, flat_lower, flat_upper)
    if np.sum(np.abs(full_template - center_flat)) > radius + feasibility_tol:
        raise ValueError("The L1 ball does not intersect the nonnegative orthant.")

    numeric_expr = _numeric_expression(model, expression, parameter_values)
    objective_lambda = sp.lambdify(
        model.variables, numeric_expr, "numpy", cse=True
    )

    numeric_tallies: tuple[sp.Expr, ...] = ()
    quota_value: float | None = None
    if require_winner_quota and model.degree > 0:
        numeric_tallies = tuple(
            _numeric_expression(model, tally, parameter_values)
            for tally in model.winner_tallies
        )
        quota_expr = model.quota.subs(dict(parameter_values or {}))
        if quota_expr.free_symbols:
            names = ", ".join(sorted(map(str, quota_expr.free_symbols)))
            raise ValueError(
                f"Quota is still symbolic ({names}); supply parameter_values."
            )
        quota_value = float(quota_expr)

    # Coordinates absent from the objective and enabled constraints can be
    # fixed at their nearest admissible values: moving them only spends budget.
    active_symbols = set(numeric_expr.free_symbols)
    for tally in numeric_tallies:
        active_symbols.update(tally.free_symbols)
    active_indices = tuple(
        i for i, variable in enumerate(model.variables) if variable in active_symbols
    )
    active_index_array = np.asarray(active_indices, dtype=int)

    def expand(active_x: np.ndarray) -> np.ndarray:
        full_x = full_template.copy()
        if active_indices:
            full_x[active_index_array] = active_x
        return full_x

    def objective(active_x: np.ndarray) -> float:
        full_x = expand(active_x)
        try:
            value = float(np.asarray(objective_lambda(*full_x)))
        except (FloatingPointError, OverflowError, ZeroDivisionError, TypeError, ValueError):
            return 1e100
        return value if np.isfinite(value) else 1e100

    def l1_distance(active_x: np.ndarray) -> float:
        return float(np.sum(np.abs(expand(active_x) - center_flat)))

    constraints: list[NonlinearConstraint] = [
        NonlinearConstraint(l1_distance, lb=-np.inf, ub=radius)
    ]

    tally_lambda = None
    if numeric_tallies:
        tally_lambda = sp.lambdify(
            model.variables, numeric_tallies, "numpy", cse=True
        )

        def tally_vector(active_x: np.ndarray) -> np.ndarray:
            try:
                values = np.asarray(
                    tally_lambda(*expand(active_x)), dtype=float
                ).reshape(-1)
            except (FloatingPointError, OverflowError, ZeroDivisionError, TypeError, ValueError):
                return np.full(model.degree, -1e100)
            return np.where(np.isfinite(values), values, -1e100)

        constraints.append(
            NonlinearConstraint(
                tally_vector,
                lb=np.full(model.degree, quota_value),
                ub=np.full(model.degree, np.inf),
            )
        )

    def independently_feasible(full_x: np.ndarray) -> bool:
        if np.sum(np.abs(full_x - center_flat)) > radius + feasibility_tol:
            return False
        if nonnegative and np.any(full_x < -feasibility_tol):
            return False
        if tally_lambda is not None:
            try:
                values = np.asarray(tally_lambda(*full_x), dtype=float).reshape(-1)
            except Exception:
                return False
            if not np.all(np.isfinite(values)):
                return False
            if np.any(values < float(quota_value) - feasibility_tol):
                return False
        return True

    if not active_indices:
        full_x = full_template
        value = objective(np.empty(0, dtype=float))
        feasible = independently_feasible(full_x)
        result = OptimizeResult(
            x=full_x,
            fun=value,
            success=feasible,
            message="The expression and enabled constraints are constant on the domain.",
            active_x=np.empty(0, dtype=float),
            active_indices=(),
        )
        return L1BallMinimum(
            expression=numeric_expr,
            minimum=value,
            point=full_x.reshape(model.symbols.shape),
            center=center.copy(),
            radius=radius,
            l1_distance=float(np.sum(np.abs(full_x - center_flat))),
            feasible=feasible,
            optimizer_result=result,
            restart_minima=(value if feasible else np.inf,),
        )

    active_bounds = tuple(
        (float(flat_lower[i]), float(flat_upper[i])) for i in active_indices
    )
    anchor = full_template[active_index_array].copy()
    population_size = max(5, popsize * len(active_indices))

    runs: list[tuple[OptimizeResult, np.ndarray, bool, float]] = []
    restart_minima: list[float] = []

    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart)
        init = np.empty((population_size, len(active_indices)), dtype=float)
        init[0] = anchor
        cursor = 1

        # Include the coordinate-axis tips of the diamond where possible.
        for j, (lo, hi) in enumerate(active_bounds):
            for endpoint in (lo, hi):
                if cursor >= population_size:
                    break
                candidate = anchor.copy()
                candidate[j] = endpoint
                init[cursor] = _contract_active_point_to_l1_ball(
                    candidate, anchor, expand, center_flat, radius
                )
                cursor += 1

        while cursor < population_size:
            candidate = np.asarray(
                [rng.uniform(lo, hi) for lo, hi in active_bounds], dtype=float
            )
            init[cursor] = _contract_active_point_to_l1_ball(
                candidate, anchor, expand, center_flat, radius
            )
            cursor += 1

        result = differential_evolution(
            objective,
            bounds=active_bounds,
            constraints=tuple(constraints),
            init=init,
            seed=seed + restart,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            polish=polish,
            updating="immediate",
            workers=1,
        )
        active_x = np.asarray(result.x, dtype=float)
        full_x = expand(active_x)
        feasible = independently_feasible(full_x)
        value = objective(active_x)

        result["active_x"] = active_x
        result["active_indices"] = active_indices
        result["x"] = full_x
        result["fun"] = value
        result["independent_feasible_check"] = feasible

        runs.append((result, full_x, feasible, value))
        restart_minima.append(value if feasible else np.inf)

    feasible_runs = [run for run in runs if run[2]]
    chosen = min(feasible_runs or runs, key=lambda run: run[3])
    result, full_x, feasible, value = chosen

    return L1BallMinimum(
        expression=numeric_expr,
        minimum=float(value),
        point=full_x.reshape(model.symbols.shape),
        center=center.copy(),
        radius=radius,
        l1_distance=float(np.sum(np.abs(full_x - center_flat))),
        feasible=bool(feasible),
        optimizer_result=result,
        restart_minima=tuple(float(x) for x in restart_minima),
    )


def minimize_coordinate_partial(
    model: SymbolicMarginModel,
    base_point: np.ndarray,
    radius: float,
    coordinate: sp.Symbol | tuple[int, int],
    **optimizer_kwargs,
) -> L1BallMinimum:
    """Minimize one coordinate partial of ``M_cl`` over an L1 ball."""
    derivative = margin_partial(
        model, coordinate, simplify=False, print_expression=False
    )
    return minimize_expression_over_l1_ball(
        model, derivative, base_point, radius, **optimizer_kwargs
    )


def minimize_directional_derivative(
    model: SymbolicMarginModel,
    base_point: np.ndarray,
    radius: float,
    direction: np.ndarray,
    **optimizer_kwargs,
) -> L1BallMinimum:
    """Minimize ``grad(M_cl) dot direction`` over an L1 ball."""
    derivative = directional_derivative(model, direction)
    return minimize_expression_over_l1_ball(
        model, derivative, base_point, radius, **optimizer_kwargs
    )

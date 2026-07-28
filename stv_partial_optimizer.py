"""Fast numerical optimization of STV margin directional derivatives.

The public entry point is :func:`minimize_partial`.  Inputs remain in the
original ``(2**degree, 3)`` coordinate system, whose columns are ``c, l, o``.
The optimization domain is the nonnegative L1 ball

    {x >= 0 : ||x - base_point||_1 <= radius}.

Two numerical paths are provided:

1. ``single_row_reduced`` (default when applicable)
   If theta is supported on one row and preserves that row total, then

       D_theta M = (theta_c - theta_l) * w_S(s),

   where ``w_S`` is one transfer weight and ``s`` is the vector of row totals.
   The full diamond projects exactly to a nonnegative L1 ball in row-total
   space.  Irrelevant row totals are fixed at their centre values.

2. ``general_recursive``
   Every other direction is handled by a recursive numerical evaluator that
   computes both

       D_theta M(x)

   and

       grad_x D_theta M(x) = Hessian(M)(x) @ theta

   exactly up to floating-point arithmetic.  No symbolic directional
   derivative, symbolic Hessian, or finite differencing is used.

Both paths use scale-normalized projected gradient descent (PGD).  Callers may
mark rows as ``dead_rows`` and columns as ``dead_columns``; those coordinates
are required to be zero in both the base point and direction and are removed
from the optimization variables entirely.  No winner quota constraint is
imposed: paths may pass through points with T_j < q.  The only singularity is
T_j = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Generic, Literal, TypeVar

import numpy as np

__version__ = "1.2.0"

PathName = Literal["zero_direction", "single_row_reduced", "general_recursive"]


@dataclass(frozen=True)
class DirectionalEvaluation:
    """Value and exact numerical gradient of ``D_theta M`` at one point."""

    value: float
    gradient: np.ndarray
    margin: float
    winner_tallies: np.ndarray
    transfer_values: np.ndarray


@dataclass(frozen=True)
class PartialMinimum:
    """Result returned by :func:`minimize_partial`."""

    path: PathName
    degree: int
    minimum: float
    point: np.ndarray
    center: np.ndarray
    radius: float
    l1_distance: float
    feasible: bool
    success: bool
    message: str
    nit: int
    nfev: int
    njev: int
    projected_gradient_norm: float
    start_minima: tuple[float, ...]
    start_iterations: tuple[int, ...]
    winner_tallies: np.ndarray
    active_dimension: int
    row_mask: int | None = None
    theta_d: float | None = None

    def __repr__(self) -> str:
        return repr(self.minimum)


@dataclass(frozen=True)
class SingleRowDirection:
    degree: int
    row_mask: int
    theta_d: float


@dataclass(frozen=True)
class _PGDRun:
    value: float
    variable: np.ndarray
    metadata: object
    converged: bool
    nit: int
    nfev: int
    njev: int
    stationarity: float


_Metadata = TypeVar("_Metadata")
_Evaluator = Callable[[np.ndarray], tuple[float, np.ndarray, _Metadata]]


# ---------------------------------------------------------------------------
# Input validation and direction structure
# ---------------------------------------------------------------------------


def _degree_from_rows(rows: int) -> int:
    degree = rows.bit_length() - 1
    if rows < 1 or 2**degree != rows:
        raise ValueError("The number of rows must be a positive power of two")
    return degree


def _normalize_dead_rows(
    dead_rows: object,
    rows: int,
) -> np.ndarray:
    """Return a boolean mask identifying rows fixed permanently at zero.

    ``dead_rows`` may be:

    - ``None``;
    - an iterable of integer row indices; or
    - a one-dimensional boolean mask of length ``rows``.
    """
    mask = np.zeros(rows, dtype=bool)
    if dead_rows is None:
        return mask

    array = np.asarray(dead_rows)
    if array.dtype == bool:
        if array.ndim != 1 or array.size != rows:
            raise ValueError(
                f"Boolean dead_rows mask must have shape ({rows},); "
                f"got {array.shape}"
            )
        return array.astype(bool, copy=True)

    try:
        indices = np.asarray(list(dead_rows), dtype=int).reshape(-1)
    except TypeError as exc:
        raise ValueError(
            "dead_rows must be None, an iterable of row indices, or a "
            "boolean row mask"
        ) from exc

    if indices.size == 0:
        return mask
    if np.any(indices < 0) or np.any(indices >= rows):
        raise ValueError(f"dead row indices must lie in [0, {rows})")
    mask[indices] = True
    return mask



def _normalize_dead_columns(
    dead_columns: object,
) -> np.ndarray:
    """Return a boolean mask identifying columns fixed permanently at zero.

    ``dead_columns`` may be:

    - ``None``;
    - an iterable of integer column indices in ``{0, 1, 2}``; or
    - a one-dimensional boolean mask of length 3.
    """
    mask = np.zeros(3, dtype=bool)
    if dead_columns is None:
        return mask

    array = np.asarray(dead_columns)
    if array.dtype == bool:
        if array.ndim != 1 or array.size != 3:
            raise ValueError(
                "Boolean dead_columns mask must have shape (3,); "
                f"got {array.shape}"
            )
        return array.astype(bool, copy=True)

    try:
        indices = np.asarray(list(dead_columns), dtype=int).reshape(-1)
    except TypeError as exc:
        raise ValueError(
            "dead_columns must be None, an iterable of column indices, or a "
            "boolean column mask"
        ) from exc

    if indices.size == 0:
        return mask
    if np.any(indices < 0) or np.any(indices >= 3):
        raise ValueError("dead column indices must lie in [0, 3)")
    mask[indices] = True
    return mask


def _validate_dead_coordinates(
    point: np.ndarray,
    direction: np.ndarray,
    dead_row_mask: np.ndarray,
    dead_column_mask: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> None:
    """Require every dead row or dead column coordinate to be zero."""
    dead_coordinate_mask = (
        dead_row_mask[:, None] | dead_column_mask[None, :]
    )

    if np.any(np.abs(point[dead_coordinate_mask]) > tolerance):
        bad = np.argwhere(
            (np.abs(point) > tolerance) & dead_coordinate_mask
        )
        raise ValueError(
            "base_point must be zero on every dead row/column coordinate; "
            f"nonzero coordinates: {bad.tolist()}"
        )
    if np.any(np.abs(direction[dead_coordinate_mask]) > tolerance):
        bad = np.argwhere(
            (np.abs(direction) > tolerance) & dead_coordinate_mask
        )
        raise ValueError(
            "direction must be zero on every dead row/column coordinate; "
            f"nonzero coordinates: {bad.tolist()}"
        )

def _validate_point_and_direction(
    point: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    t = np.asarray(point, dtype=float)
    theta = np.asarray(direction, dtype=float)
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError(f"point must have shape (2**degree, 3); got {t.shape}")
    if theta.shape != t.shape:
        raise ValueError(f"direction must have shape {t.shape}; got {theta.shape}")
    rows = t.shape[0]
    degree = _degree_from_rows(rows)
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(theta)):
        raise ValueError("point and direction must be finite")
    return t, theta, rows, degree


def identify_single_row_direction(
    direction: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> SingleRowDirection | None:
    """Return the reduced-direction description when the fast path applies.

    The fast path requires exactly one nonzero row and

        theta[row, c] + theta[row, l] + theta[row, o] = 0.

    A zero direction returns ``None``; it is handled separately by the public
    dispatcher.
    """
    theta = np.asarray(direction, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 3:
        raise ValueError(
            f"direction must have shape (2**degree, 3); got {theta.shape}"
        )
    degree = _degree_from_rows(theta.shape[0])
    nonzero_rows = np.flatnonzero(np.any(np.abs(theta) > tolerance, axis=1))
    if nonzero_rows.size != 1:
        return None

    row_mask = int(nonzero_rows[0])
    row = theta[row_mask]
    if abs(float(np.sum(row))) > tolerance:
        return None

    return SingleRowDirection(
        degree=degree,
        row_mask=row_mask,
        theta_d=float(row[0] - row[1]),
    )


def two_entry_theta_key(
    direction: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> tuple[int, int]:
    """Return ``(plus_flat_index, minus_flat_index)`` for a +1/-1 theta."""
    theta = np.asarray(direction, dtype=float).reshape(-1)
    plus = np.flatnonzero(np.abs(theta - 1.0) <= tolerance)
    minus = np.flatnonzero(np.abs(theta + 1.0) <= tolerance)
    other = np.flatnonzero(
        (np.abs(theta) > tolerance)
        & (np.abs(theta - 1.0) > tolerance)
        & (np.abs(theta + 1.0) > tolerance)
    )
    if plus.size != 1 or minus.size != 1 or other.size:
        raise ValueError("direction is not exactly one +1 entry and one -1 entry")
    return int(plus[0]), int(minus[0])


# ---------------------------------------------------------------------------
# Exact projection onto a nonnegative L1 ball
# ---------------------------------------------------------------------------


def project_nonnegative_l1_ball(
    y: np.ndarray,
    center: np.ndarray,
    radius: float,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Euclidean projection onto ``{x >= 0, ||x-center||_1 <= radius}``."""
    center_array = np.asarray(center, dtype=float)
    original_shape = center_array.shape
    y_flat = np.asarray(y, dtype=float).reshape(-1)
    center_flat = center_array.reshape(-1)

    if y_flat.shape != center_flat.shape:
        raise ValueError("y and center must have the same number of entries")
    if np.any(center_flat < -tolerance):
        raise ValueError("center must be nonnegative")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be a finite nonnegative scalar")

    clipped = np.maximum(y_flat, 0.0)
    if np.sum(np.abs(clipped - center_flat)) <= radius + tolerance:
        return clipped.reshape(original_shape)

    displacement = y_flat - center_flat

    def point(lam: float) -> np.ndarray:
        delta = np.sign(displacement) * np.maximum(
            np.abs(displacement) - lam,
            0.0,
        )
        # Enforce x = center + delta >= 0.
        delta = np.maximum(delta, -center_flat)
        return center_flat + delta

    lo = 0.0
    hi = max(1.0, float(np.max(np.abs(displacement))))
    while np.sum(np.abs(point(hi) - center_flat)) > radius:
        hi *= 2.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.sum(np.abs(point(mid) - center_flat)) > radius:
            lo = mid
        else:
            hi = mid

    return point(hi).reshape(original_shape)


def _make_normalized_starts(
    center: np.ndarray,
    radius: float,
    n_starts: int,
    seed: int,
) -> list[np.ndarray]:
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")

    shape = center.shape
    dimension = center.size
    starts = [np.zeros(shape, dtype=float)]
    if radius == 0 or n_starts == 1:
        return starts

    # Deterministic axis tips before random interior points.
    for coordinate in range(dimension):
        for sign in (-1.0, 1.0):
            if len(starts) >= n_starts:
                return starts
            u = np.zeros(dimension, dtype=float)
            u[coordinate] = sign
            candidate = project_nonnegative_l1_ball(
                center.reshape(-1) + radius * u,
                center.reshape(-1),
                radius,
            )
            starts.append(((candidate - center.reshape(-1)) / radius).reshape(shape))

    rng = np.random.default_rng(seed)
    while len(starts) < n_starts:
        weights = rng.dirichlet(np.ones(dimension))
        signs = rng.choice((-1.0, 1.0), size=dimension)
        radial_fraction = rng.random() ** (1.0 / dimension)
        u = radial_fraction * signs * weights
        candidate = project_nonnegative_l1_ball(
            center.reshape(-1) + radius * u,
            center.reshape(-1),
            radius,
        )
        starts.append(((candidate - center.reshape(-1)) / radius).reshape(shape))

    return starts


# ---------------------------------------------------------------------------
# Generic normalized PGD driver
# ---------------------------------------------------------------------------


def _run_normalized_pgd(
    center: np.ndarray,
    radius: float,
    evaluator: _Evaluator[_Metadata],
    *,
    n_starts: int,
    seed: int,
    maxiter: int,
    initial_step: float,
    maximum_step: float,
    backtrack_factor: float,
    armijo_constant: float,
    stationarity_tol: float,
    movement_tol: float,
) -> tuple[_PGDRun, tuple[_PGDRun, ...]]:
    """Run PGD in normalized coordinates ``u=(x-center)/radius``."""
    center = np.asarray(center, dtype=float)

    if radius == 0:
        value, gradient, metadata = evaluator(center)
        run = _PGDRun(
            value=float(value),
            variable=center.copy(),
            metadata=metadata,
            converged=np.isfinite(value) and np.all(np.isfinite(gradient)),
            nit=0,
            nfev=1,
            njev=1,
            stationarity=0.0,
        )
        return run, (run,)

    starts = _make_normalized_starts(center, radius, n_starts, seed)
    runs: list[_PGDRun] = []

    def x_from_u(u: np.ndarray) -> np.ndarray:
        return center + radius * u

    def project_u(v: np.ndarray) -> np.ndarray:
        projected_x = project_nonnegative_l1_ball(x_from_u(v), center, radius)
        return (projected_x - center) / radius

    for start in starts:
        u = start.copy()
        value, gradient_x, metadata = evaluator(x_from_u(u))
        value = float(value)
        gradient_x = np.asarray(gradient_x, dtype=float)
        nfev = 1
        njev = 1
        step = initial_step
        converged = False
        stationarity = np.inf
        iteration = 0

        for iteration in range(1, maxiter + 1):
            if not np.isfinite(value) or not np.all(np.isfinite(gradient_x)):
                break

            gradient_u = radius * gradient_x
            projected = project_u(u - gradient_u)
            stationarity = float(np.linalg.norm(u - projected))
            if stationarity <= stationarity_tol:
                converged = True
                break

            accepted = False
            trial_step = step
            for _ in range(60):
                candidate_u = project_u(u - trial_step * gradient_u)
                movement = candidate_u - u
                if np.linalg.norm(movement) <= movement_tol:
                    break

                candidate_value, candidate_gradient_x, candidate_metadata = evaluator(
                    x_from_u(candidate_u)
                )
                candidate_value = float(candidate_value)
                candidate_gradient_x = np.asarray(candidate_gradient_x, dtype=float)
                nfev += 1
                njev += 1

                armijo_rhs = value + armijo_constant * float(
                    gradient_u.reshape(-1) @ movement.reshape(-1)
                )
                if np.isfinite(candidate_value) and candidate_value <= armijo_rhs:
                    accepted = True
                    break

                trial_step *= backtrack_factor

            if not accepted:
                break

            u = candidate_u
            value = candidate_value
            gradient_x = candidate_gradient_x
            metadata = candidate_metadata
            step = min(maximum_step, 1.5 * trial_step)

        if np.isfinite(value) and np.all(np.isfinite(gradient_x)):
            gradient_u = radius * gradient_x
            stationarity = float(np.linalg.norm(u - project_u(u - gradient_u)))

        runs.append(
            _PGDRun(
                value=float(value),
                variable=x_from_u(u),
                metadata=metadata,
                converged=converged,
                nit=int(iteration),
                nfev=int(nfev),
                njev=int(njev),
                stationarity=float(stationarity),
            )
        )

    finite_runs = [run for run in runs if np.isfinite(run.value)]
    if not finite_runs:
        raise RuntimeError("All PGD starts encountered nonfinite objective values")

    best = min(finite_runs, key=lambda run: run.value)
    return best, tuple(runs)


# ---------------------------------------------------------------------------
# General recursive directional-derivative evaluator
# ---------------------------------------------------------------------------


# A transfer-weight jet contains
#
#   value,
#   gradient with respect to row totals s,
#   directional derivative in the row-total direction alpha,
#   gradient of that directional derivative (Hessian-vector product).
#
# Working only in row-total coordinates here halves the derivative-vector
# dimension relative to a generic (s,d) jet.  The margin's d-dependence is
# inserted analytically at the end of the recursion.
_WeightJet = tuple[float, np.ndarray, float, np.ndarray]


def _weight_one(rows: int) -> _WeightJet:
    zero = np.zeros(rows, dtype=float)
    return 1.0, zero, 0.0, zero.copy()


def _weight_mul(a: _WeightJet, b: _WeightJet) -> _WeightJet:
    av, ag, ad, ah = a
    bv, bg, bd, bh = b
    return (
        av * bv,
        bv * ag + av * bg,
        ad * bv + av * bd,
        bv * ah + ad * bg + bd * ag + av * bh,
    )


def _tau_weight_jet(
    tally_value: float,
    tally_gradient: np.ndarray,
    tally_directional: float,
    tally_directional_gradient: np.ndarray,
    quota: float,
    singularity_tol: float,
) -> _WeightJet:
    if not np.isfinite(tally_value) or abs(tally_value) <= singularity_tol:
        nan_vector = np.full_like(tally_gradient, np.nan)
        return np.nan, nan_vector, np.nan, nan_vector.copy()

    first = quota / (tally_value * tally_value)
    second = -2.0 * quota / (tally_value * tally_value * tally_value)
    return (
        1.0 - quota / tally_value,
        first * tally_gradient,
        first * tally_directional,
        second * tally_directional * tally_gradient
        + first * tally_directional_gradient,
    )


def evaluate_directional_derivative(
    point: np.ndarray,
    direction: np.ndarray,
    quota: float,
    *,
    singularity_tol: float = 1e-14,
) -> DirectionalEvaluation:
    """Evaluate ``D_theta M`` and its exact gradient numerically.

    The recursion propagates transfer-weight values, gradients, directional
    derivatives, and Hessian-vector products in row-total coordinates.  It
    then inserts the margin's affine dependence on row differences directly.
    """
    t, theta, rows, degree = _validate_point_and_direction(point, direction)
    if not np.isfinite(quota):
        raise ValueError("quota must be finite")

    s = t.sum(axis=1)
    d = t[:, 0] - t[:, 1]
    alpha = theta.sum(axis=1)
    beta = theta[:, 0] - theta[:, 1]

    # prefix_weights[mask] is the transfer-weight jet for the set of already
    # processed winners encoded by mask.
    prefix_weights: list[_WeightJet] = [_weight_one(rows)]
    tallies = np.empty(degree, dtype=float)
    taus = np.empty(degree, dtype=float)

    for winner in range(degree):
        tally_value = 0.0
        tally_gradient = np.zeros(rows, dtype=float)
        tally_directional = 0.0
        tally_directional_gradient = np.zeros(rows, dtype=float)

        lower_count = 1 << winner
        higher_count = 1 << (degree - winner - 1)

        for earlier_mask in range(lower_count):
            masks = [
                earlier_mask
                | (1 << winner)
                | (higher << (winner + 1))
                for higher in range(higher_count)
            ]
            aggregate = float(np.sum(s[masks]))
            aggregate_directional = float(np.sum(alpha[masks]))

            weight_value, weight_gradient, weight_directional, weight_hvp = (
                prefix_weights[earlier_mask]
            )

            tally_value += weight_value * aggregate
            tally_gradient += weight_gradient * aggregate
            tally_gradient[masks] += weight_value

            tally_directional += (
                weight_directional * aggregate
                + weight_value * aggregate_directional
            )
            tally_directional_gradient += (
                weight_hvp * aggregate
                + weight_gradient * aggregate_directional
            )
            tally_directional_gradient[masks] += weight_directional

        tallies[winner] = tally_value
        tau = _tau_weight_jet(
            tally_value,
            tally_gradient,
            tally_directional,
            tally_directional_gradient,
            float(quota),
            singularity_tol,
        )
        taus[winner] = tau[0]
        if not np.isfinite(tau[0]):
            return DirectionalEvaluation(
                value=np.nan,
                gradient=np.full_like(t, np.nan),
                margin=np.nan,
                winner_tallies=tallies.copy(),
                transfer_values=taus.copy(),
            )

        old_weights = prefix_weights
        prefix_weights = old_weights + [
            _weight_mul(weight, tau) for weight in old_weights
        ]

    margin_value = 0.0
    directional_value = 0.0
    gradient_s = np.zeros(rows, dtype=float)
    gradient_d = np.zeros(rows, dtype=float)

    for mask, (weight_value, weight_gradient, weight_directional, weight_hvp) in enumerate(prefix_weights):
        margin_value += weight_value * d[mask]
        directional_value += (
            weight_directional * d[mask]
            + weight_value * beta[mask]
        )
        gradient_s += (
            weight_hvp * d[mask]
            + weight_gradient * beta[mask]
        )
        gradient_d[mask] = weight_directional

    # Chain rule from (s,d) back to the original (c,l,o) coordinates.
    gradient_t = np.empty_like(t)
    gradient_t[:, 0] = gradient_s + gradient_d
    gradient_t[:, 1] = gradient_s - gradient_d
    gradient_t[:, 2] = gradient_s

    return DirectionalEvaluation(
        value=float(directional_value),
        gradient=gradient_t,
        margin=float(margin_value),
        winner_tallies=tallies,
        transfer_values=taus,
    )


# ---------------------------------------------------------------------------
# Specialized transfer-weight evaluator for one-row directions
# ---------------------------------------------------------------------------


def _product_and_gradient(
    values: list[float],
    gradients: list[np.ndarray],
    dimension: int,
) -> tuple[float, np.ndarray]:
    count = len(values)
    if count == 0:
        return 1.0, np.zeros(dimension, dtype=float)
    if count == 1:
        return float(values[0]), np.asarray(gradients[0], dtype=float).copy()

    prefix = np.ones(count + 1, dtype=float)
    suffix = np.ones(count + 1, dtype=float)
    for index in range(count):
        prefix[index + 1] = prefix[index] * float(values[index])
    for index in range(count - 1, -1, -1):
        suffix[index] = suffix[index + 1] * float(values[index])

    gradient = np.zeros(dimension, dtype=float)
    for index in range(count):
        gradient += prefix[index] * suffix[index + 1] * gradients[index]
    return float(prefix[count]), gradient


def _evaluate_transfer_weight(
    degree: int,
    row_mask: int,
    row_totals: np.ndarray,
    quota: float,
    singularity_tol: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return one transfer weight, its row-total gradient, tallies, and taus."""
    totals = np.asarray(row_totals, dtype=float).reshape(-1)
    rows = 2**degree
    if totals.size != rows:
        raise ValueError(f"Expected {rows} row totals; got {totals.size}")

    tallies = np.empty(degree, dtype=float)
    taus: list[float] = []
    tau_gradients: list[np.ndarray] = []

    for winner in range(degree):
        tally = 0.0
        tally_gradient = np.zeros(rows, dtype=float)

        for mask in range(rows):
            if not (mask & (1 << winner)):
                continue
            included = [
                earlier
                for earlier in range(winner)
                if mask & (1 << earlier)
            ]
            weight, weight_gradient = _product_and_gradient(
                [taus[index] for index in included],
                [tau_gradients[index] for index in included],
                rows,
            )
            tally += weight * totals[mask]
            tally_gradient += totals[mask] * weight_gradient
            tally_gradient[mask] += weight

        tallies[winner] = tally
        if not np.isfinite(tally) or abs(tally) <= singularity_tol:
            return (
                np.nan,
                np.full(rows, np.nan),
                tallies,
                np.asarray(taus + [np.nan], dtype=float),
            )

        tau = 1.0 - quota / tally
        tau_gradient = (quota / (tally * tally)) * tally_gradient
        taus.append(float(tau))
        tau_gradients.append(tau_gradient)

    included_winners = [
        winner for winner in range(degree) if row_mask & (1 << winner)
    ]
    value, gradient = _product_and_gradient(
        [taus[winner] for winner in included_winners],
        [tau_gradients[winner] for winner in included_winners],
        rows,
    )
    return value, gradient, tallies, np.asarray(taus, dtype=float)


@lru_cache(maxsize=None)
def _structural_active_indices(degree: int, row_mask: int) -> tuple[int, ...]:
    """Row totals that can influence transfer weight ``w_row_mask``."""
    if row_mask == 0:
        return ()
    highest_winner = row_mask.bit_length() - 1
    earlier_or_equal_bits = (1 << (highest_winner + 1)) - 1
    return tuple(
        mask
        for mask in range(2**degree)
        if mask & earlier_or_equal_bits
    )


def lift_row_totals_to_point(
    base_point: np.ndarray,
    target_row_totals: np.ndarray,
    *,
    dead_columns: object = None,
) -> np.ndarray:
    """Lift row totals at minimum L1 distance while preserving dead columns.

    Dead columns must already be zero in ``base_point``.  Any mass added to a
    zero-total row is placed in the first live column.
    """
    base = np.asarray(base_point, dtype=float)
    target = np.asarray(target_row_totals, dtype=float).reshape(-1)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("base_point must have shape (2**degree, 3)")
    dead_column_mask = _normalize_dead_columns(dead_columns)
    if np.any(np.abs(base[:, dead_column_mask]) > 1e-12):
        raise ValueError("base_point must be zero on every dead column")
    live_columns = np.flatnonzero(~dead_column_mask)
    if live_columns.size == 0 and np.any(np.abs(target) > 1e-12):
        raise ValueError("nonzero target totals are impossible when all columns are dead")

    center_totals = base.sum(axis=1)
    if target.shape != center_totals.shape:
        raise ValueError("target_row_totals has the wrong length")
    if np.any(target < -1e-10):
        raise ValueError("target_row_totals must be nonnegative")

    result = np.zeros_like(base)
    positive = center_totals > 0
    result[positive] = (
        base[positive]
        * (target[positive] / center_totals[positive])[:, None]
    )
    if live_columns.size:
        result[~positive, live_columns[0]] = target[~positive]
    result[:, dead_column_mask] = 0.0
    return result


# ---------------------------------------------------------------------------
# Public optimization functions
# ---------------------------------------------------------------------------


def _assemble_result(
    *,
    path: PathName,
    degree: int,
    center: np.ndarray,
    point: np.ndarray,
    radius: float,
    best: _PGDRun,
    runs: tuple[_PGDRun, ...],
    winner_tallies: np.ndarray,
    active_dimension: int,
    row_mask: int | None = None,
    theta_d: float | None = None,
) -> PartialMinimum:
    l1_distance = float(np.sum(np.abs(point - center)))
    feasibility_tolerance = max(1e-7, 1e-10 * max(1.0, radius))
    feasible = bool(
        np.all(point >= -1e-7)
        and l1_distance <= radius + feasibility_tolerance
    )
    success = bool(best.converged and feasible)
    message = (
        f"Projected-gradient residual {best.stationarity:.3e}."
        if best.converged
        else f"Stopped with projected-gradient residual {best.stationarity:.3e}."
    )

    return PartialMinimum(
        path=path,
        degree=degree,
        minimum=float(best.value),
        point=point,
        center=center.copy(),
        radius=float(radius),
        l1_distance=l1_distance,
        feasible=feasible,
        success=success,
        message=message,
        nit=best.nit,
        nfev=best.nfev,
        njev=best.njev,
        projected_gradient_norm=best.stationarity,
        start_minima=tuple(float(run.value) for run in runs),
        start_iterations=tuple(int(run.nit) for run in runs),
        winner_tallies=np.asarray(winner_tallies, dtype=float).copy(),
        active_dimension=int(active_dimension),
        row_mask=row_mask,
        theta_d=theta_d,
    )


def minimize_single_row_direction_pgd(
    base_point: np.ndarray,
    direction: np.ndarray,
    radius: float,
    quota: float,
    *,
    dead_rows: object = None,
    dead_columns: object = None,
    n_starts: int = 1,
    seed: int = 0,
    maxiter: int = 1000,
    initial_step: float = 1.0,
    maximum_step: float = 1e6,
    backtrack_factor: float = 0.5,
    armijo_constant: float = 1e-4,
    stationarity_tol: float = 1e-8,
    movement_tol: float = 1e-12,
    singularity_tol: float = 1e-14,
) -> PartialMinimum:
    """Optimize a one-row, row-total-preserving direction in row-total space."""
    center, theta, rows, degree = _validate_point_and_direction(
        base_point, direction
    )
    if np.any(center < 0):
        raise ValueError("base_point must be nonnegative")
    dead_mask = _normalize_dead_rows(dead_rows, rows)
    dead_column_mask = _normalize_dead_columns(dead_columns)
    _validate_dead_coordinates(
        center, theta, dead_mask, dead_column_mask
    )
    info = identify_single_row_direction(theta)
    if info is None:
        raise ValueError(
            "The specialized path requires exactly one nonzero row whose "
            "three theta entries sum to zero."
        )

    center_totals = center.sum(axis=1)
    live_columns = np.flatnonzero(~dead_column_mask)
    active_indices = tuple(
        index
        for index in _structural_active_indices(degree, info.row_mask)
        if not dead_mask[index] and live_columns.size > 0
    )
    active_array = np.asarray(active_indices, dtype=int)
    active_center = center_totals[active_array]

    if info.theta_d == 0.0 or not active_indices:
        value, _, tallies, _ = _evaluate_transfer_weight(
            degree,
            info.row_mask,
            center_totals,
            quota,
            singularity_tol,
        )
        minimum = info.theta_d * value
        run = _PGDRun(
            value=float(minimum),
            variable=active_center.copy(),
            metadata=tallies,
            converged=np.isfinite(minimum),
            nit=0,
            nfev=1,
            njev=1,
            stationarity=0.0,
        )
        return _assemble_result(
            path="single_row_reduced",
            degree=degree,
            center=center,
            point=center.copy(),
            radius=radius,
            best=run,
            runs=(run,),
            winner_tallies=tallies,
            active_dimension=len(active_indices),
            row_mask=info.row_mask,
            theta_d=info.theta_d,
        )

    def evaluator(active_totals: np.ndarray):
        full_totals = center_totals.copy()
        full_totals[active_array] = np.asarray(active_totals, dtype=float)
        value, gradient, tallies, taus = _evaluate_transfer_weight(
            degree,
            info.row_mask,
            full_totals,
            quota,
            singularity_tol,
        )
        metadata = (full_totals, tallies, taus)
        return (
            info.theta_d * value,
            info.theta_d * gradient[active_array],
            metadata,
        )

    best, runs = _run_normalized_pgd(
        active_center,
        radius,
        evaluator,
        n_starts=n_starts,
        seed=seed,
        maxiter=maxiter,
        initial_step=initial_step,
        maximum_step=maximum_step,
        backtrack_factor=backtrack_factor,
        armijo_constant=armijo_constant,
        stationarity_tol=stationarity_tol,
        movement_tol=movement_tol,
    )

    full_totals, tallies, _ = best.metadata
    point = lift_row_totals_to_point(
        center, full_totals, dead_columns=dead_column_mask
    )
    return _assemble_result(
        path="single_row_reduced",
        degree=degree,
        center=center,
        point=point,
        radius=radius,
        best=best,
        runs=runs,
        winner_tallies=tallies,
        active_dimension=len(active_indices),
        row_mask=info.row_mask,
        theta_d=info.theta_d,
    )


def minimize_general_direction_pgd(
    base_point: np.ndarray,
    direction: np.ndarray,
    radius: float,
    quota: float,
    *,
    dead_rows: object = None,
    dead_columns: object = None,
    n_starts: int = 1,
    seed: int = 0,
    maxiter: int = 1000,
    initial_step: float = 1.0,
    maximum_step: float = 1e6,
    backtrack_factor: float = 0.5,
    armijo_constant: float = 1e-4,
    stationarity_tol: float = 1e-8,
    movement_tol: float = 1e-12,
    singularity_tol: float = 1e-14,
) -> PartialMinimum:
    """Optimize an arbitrary direction using recursive objective/gradient PGD."""
    center, theta, rows, degree = _validate_point_and_direction(
        base_point, direction
    )
    if np.any(center < 0):
        raise ValueError("base_point must be nonnegative")

    dead_mask = _normalize_dead_rows(dead_rows, rows)
    dead_column_mask = _normalize_dead_columns(dead_columns)
    _validate_dead_coordinates(
        center, theta, dead_mask, dead_column_mask
    )
    active_coordinate_mask = (
        (~dead_mask)[:, None] & (~dead_column_mask)[None, :]
    )
    active_center = center[active_coordinate_mask].copy()

    def evaluator(active_point: np.ndarray):
        full_point = np.zeros_like(center)
        full_point[active_coordinate_mask] = np.asarray(
            active_point, dtype=float
        ).reshape(-1)
        evaluation = evaluate_directional_derivative(
            full_point,
            theta,
            quota,
            singularity_tol=singularity_tol,
        )
        return (
            evaluation.value,
            evaluation.gradient[active_coordinate_mask],
            evaluation,
        )

    best, runs = _run_normalized_pgd(
        active_center,
        radius,
        evaluator,
        n_starts=n_starts,
        seed=seed,
        maxiter=maxiter,
        initial_step=initial_step,
        maximum_step=maximum_step,
        backtrack_factor=backtrack_factor,
        armijo_constant=armijo_constant,
        stationarity_tol=stationarity_tol,
        movement_tol=movement_tol,
    )

    evaluation = best.metadata
    point = np.zeros_like(center)
    point[active_coordinate_mask] = np.asarray(
        best.variable, dtype=float
    ).reshape(-1)
    return _assemble_result(
        path="general_recursive",
        degree=degree,
        center=center,
        point=point,
        radius=radius,
        best=best,
        runs=runs,
        winner_tallies=evaluation.winner_tallies,
        active_dimension=int(np.count_nonzero(active_coordinate_mask)),
    )


def minimize_partial(
    base_point: np.ndarray,
    direction: np.ndarray,
    radius: float,
    quota: float,
    *,
    dead_rows: object = None,
    dead_columns: object = None,
    specialize_single_row: bool = True,
    n_starts: int = 1,
    seed: int = 0,
    maxiter: int = 1000,
    initial_step: float = 1.0,
    maximum_step: float = 1e6,
    backtrack_factor: float = 0.5,
    armijo_constant: float = 1e-4,
    stationarity_tol: float = 1e-8,
    movement_tol: float = 1e-12,
    singularity_tol: float = 1e-14,
) -> PartialMinimum:
    """Minimize ``D_direction M`` over a nonnegative L1 ball.

    By default, a one-row row-total-preserving direction uses the exact reduced
    row-total path.  Every other direction uses the unified recursive path.
    ``dead_rows`` may be an iterable of row indices or a boolean row mask.
    ``dead_columns`` may similarly identify any of the three columns ``c, l, o``
    by index or by a length-3 boolean mask.  Every dead coordinate must be zero
    in both ``base_point`` and ``direction`` and is held fixed at zero throughout
    optimization.  Set ``specialize_single_row=False`` to force the general
    path for testing.
    """
    center, theta, rows, degree = _validate_point_and_direction(
        base_point, direction
    )
    if np.any(center < 0):
        raise ValueError("base_point must be nonnegative")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be a finite nonnegative scalar")
    if not np.isfinite(quota):
        raise ValueError("quota must be finite")

    dead_mask = _normalize_dead_rows(dead_rows, rows)
    dead_column_mask = _normalize_dead_columns(dead_columns)
    _validate_dead_coordinates(
        center, theta, dead_mask, dead_column_mask
    )

    common = dict(
        dead_rows=dead_mask,
        dead_columns=dead_column_mask,
        n_starts=n_starts,
        seed=seed,
        maxiter=maxiter,
        initial_step=initial_step,
        maximum_step=maximum_step,
        backtrack_factor=backtrack_factor,
        armijo_constant=armijo_constant,
        stationarity_tol=stationarity_tol,
        movement_tol=movement_tol,
        singularity_tol=singularity_tol,
    )

    if not np.any(np.abs(theta) > 1e-12):
        evaluation = evaluate_directional_derivative(
            center,
            theta,
            quota,
            singularity_tol=singularity_tol,
        )
        run = _PGDRun(
            value=0.0,
            variable=center.copy(),
            metadata=evaluation,
            converged=True,
            nit=0,
            nfev=1,
            njev=1,
            stationarity=0.0,
        )
        return _assemble_result(
            path="zero_direction",
            degree=degree,
            center=center,
            point=center.copy(),
            radius=radius,
            best=run,
            runs=(run,),
            winner_tallies=evaluation.winner_tallies,
            active_dimension=0,
        )

    if specialize_single_row and identify_single_row_direction(theta) is not None:
        return minimize_single_row_direction_pgd(
            center,
            theta,
            radius,
            quota,
            **common,
        )

    return minimize_general_direction_pgd(
        center,
        theta,
        radius,
        quota,
        **common,
    )


# Backwards-friendly descriptive alias.
minimize_directional_derivative = minimize_partial


__all__ = [
    "DirectionalEvaluation",
    "PartialMinimum",
    "SingleRowDirection",
    "evaluate_directional_derivative",
    "identify_single_row_direction",
    "lift_row_totals_to_point",
    "minimize_directional_derivative",
    "minimize_general_direction_pgd",
    "minimize_partial",
    "minimize_single_row_direction_pgd",
    "project_nonnegative_l1_ball",
    "two_entry_theta_key",
]

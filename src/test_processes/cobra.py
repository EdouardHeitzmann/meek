from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray

try:
    from ..election_graphs.datatypes import (
        EdgeAction,
        ElectionState,
    )
except ImportError:
    from election_graphs.datatypes import (
        EdgeAction,
        ElectionState,
    )
    from test_processes.interpreter import ThetaKey, VertexInterpreter
else:
    from .interpreter import ThetaKey, VertexInterpreter


SENTINEL = np.int16(-127)
DEFAULT_LAMBDA_VALUE = 1.9
MIN_NUMERICAL_LAMBDA_VALUE = 0.1


class CriticalMarginType(str, Enum):
    CANDIDATE_TO_CANDIDATE = "candidate-to-candidate"
    CANDIDATE_ABOVE_QUOTA = "candidate-above-quota"
    CANDIDATE_BELOW_QUOTA = "candidate-below-quota"
    CANDIDATE_TO_MENTIONS = "candidate-to-mentions"


class CobraCompilerV2Base:
    """Shared state and update loop for COBRA v2 compilers."""

    def __init__(
        self,
        interpreter: VertexInterpreter,
        *,
        critical_margin: int | float | None = None,
        radius: int | float | None = None,
        LAM: int | float | None = None,
        N: int | float | None = None,
        alpha: float = 0.05,
        lambda_value: float | None = None,
        use_fallback: bool = True,
        optimizer_kwargs: dict[str, Any] | None = None,
        profile: bool = False,
        label: str | None = None,
    ) -> None:
        self.interpreter = interpreter
        self.graph = interpreter.graph
        self.base_vertex = interpreter.vertex
        self.LAM = float(getattr(self.graph, "LAM") if LAM is None else LAM)
        self.radius = float(2.0 * self.LAM if radius is None else radius)
        self.N = float(self._total_ballot_weight() if N is None else N)
        self.alpha = float(alpha)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        self.threshold = 1.0 / self.alpha
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.profile = bool(profile)
        self.dead_rows = self.interpreter.simultaneous_dead_rows()
        self.label = label or self._default_label()
        self.critical_margin = float(
            self._recorded_margin() if critical_margin is None else critical_margin
        )
        self.v = self._diluted_margin()
        if self.critical_margin <= 0.0:
            raise ValueError(
                f"critical_margin must be positive, got {self.critical_margin}."
            )
        self.M = 1.0
        self.num_updates = 0
        self.optimizer_cache: dict[ThetaKey, float] = {}
        self.optimizer_results: dict[ThetaKey, Any] = {}
        self.discrepancies: list[tuple[int, ThetaKey, float]] = []
        self.capital_history: list[float] = [self.M] if self.profile else []
        self.w_history: list[float] = [] if self.profile else []
        self.theta_key_history: list[ThetaKey | None] = [] if self.profile else []
        self.used_fallback = False

        self.lambda_value = (
            self._default_lambda()
            if lambda_value is None and use_fallback
            else float(lambda_value if lambda_value is not None else self._default_lambda())
        )
        if lambda_value is None and use_fallback:
            self.used_fallback = True

    @property
    def compiler_kind(self) -> str:
        return self.__class__.__name__

    def update(
        self,
        row_c: NDArray[np.integer],
        row_b: NDArray[np.integer],
        verbose: bool = False,
    ) -> float:
        relevant_key, w = self._discrepancy_bound(row_c, row_b)
        x = self.assorter_from_w(w)
        self.num_updates += 1
        self.M *= 1.0 + self.lambda_value * (x - 0.5)

        if self.profile:
            self.capital_history.append(self.M)
            self.w_history.append(w)
            self.theta_key_history.append(relevant_key)

        if verbose:
            print(
                f"{self.label}: update={self.num_updates}, "
                f"theta={relevant_key}, w={w}, M={self.M}"
            )
        return self.M

    def run_sampler(
        self,
        sampler: Any,
        ballot_matrix: NDArray[np.integer] | None = None,
        *,
        threshold: float | None = None,
        max_updates: int | None = None,
        verbose: bool = False,
    ) -> int:
        matrix = self.graph.ballot_matrix if ballot_matrix is None else ballot_matrix
        stop = self.threshold if threshold is None else float(threshold)
        limit = sampler.sample_size if max_updates is None else min(
            int(max_updates),
            sampler.sample_size,
        )
        start = self.num_updates
        for i in range(limit):
            row_c, row_b = sampler.sample(i, matrix)
            self.update(row_c, row_b, verbose=verbose)
            if self.M >= stop:
                break
        return self.num_updates - start

    def assorter_from_w(self, w: float) -> float:
        bounded_w = float(np.clip(w, -2.0, 2.0))
        return (1.0 + bounded_w / 2.0) / (2.0 - self.v)

    def reset(self) -> None:
        self.M = 1.0
        self.num_updates = 0
        self.discrepancies.clear()
        if self.profile:
            self.capital_history = [self.M]
            self.w_history = []
            self.theta_key_history = []

    def get_info(self) -> str:
        return (
            f"{self.compiler_kind}("
            f"label={self.label}, "
            f"critical_margin={self.critical_margin}, "
            f"v={self.v}, "
            f"radius={self.radius}, "
            f"lambda_value={self.lambda_value}, "
            f"M={self.M}, "
            f"num_updates={self.num_updates}, "
            f"cached_thetas={len(self.optimizer_cache)}"
            ")"
        )

    def __repr__(self) -> str:
        return self.get_info()

    def _discrepancy_bound(
        self,
        row_c: NDArray[np.integer],
        row_b: NDArray[np.integer],
    ) -> tuple[ThetaKey | None, float]:
        raise NotImplementedError

    def _recorded_margin(self) -> float:
        raise NotImplementedError

    def _diluted_margin(self) -> float:
        return self.critical_margin / self.N

    def _default_label(self) -> str:
        return self.graph.vertex_label(self.base_vertex.ref)

    def _w_for_theta(self, theta_key: ThetaKey) -> float:
        if theta_key in self.optimizer_cache:
            return self.optimizer_cache[theta_key]

        result = self.interpreter.optimize(
            self.base_point,
            theta_key,
            self.radius,
            dead_rows=self.dead_rows,
            **self.optimizer_kwargs,
        )
        w = float(result.minimum)
        self.optimizer_cache[theta_key] = w
        self.optimizer_results[theta_key] = result
        return w

    def _total_ballot_weight(self) -> float:
        profile = getattr(self.graph, "profile", None)
        if profile is not None and hasattr(profile, "total_ballot_wt"):
            return float(profile.total_ballot_wt)
        return float(np.asarray(self.graph.root_wt_vec, dtype=np.float64).sum())

    def _candidate_label(self, idx: int) -> str:
        return str(self.graph.candidate_names[int(idx)])

    def _default_lambda(self) -> float:
        value = DEFAULT_LAMBDA_VALUE
        if value <= 0.0 or value >= 2.0:
            return float(np.nextafter(2.0, 0.0))
        return value

    def _fallback_lambda(self, reason: str) -> float:
        self.used_fallback = True
        warnings.warn(
            f"{self.label}: {reason} Using fallback lambda={DEFAULT_LAMBDA_VALUE}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return self._default_lambda()


class CobraCompilerV2(CobraCompilerV2Base):
    """
    Noise-filtered COBRA compiler for vertex-local candidate-to-candidate margins.
    """

    def __init__(
        self,
        interpreter: VertexInterpreter,
        c: int | str,
        l: int | str,
        **kwargs: Any,
    ) -> None:
        self.c = interpreter.candidate_index(c)
        self.l = interpreter.candidate_index(l)
        if self.c == self.l:
            raise ValueError("c and l must be distinct candidates.")
        self.c_label = str(interpreter.graph.candidate_names[self.c])
        self.l_label = str(interpreter.graph.candidate_names[self.l])
        self.base_point = interpreter.base_point(self.c, self.l)
        self.dead_columns: tuple[int, ...] = ()
        super().__init__(interpreter, **kwargs)

    def _default_label(self) -> str:
        return (
            f"{self.graph.vertex_label(self.base_vertex.ref)}: "
            f"{self.c_label}>{self.l_label}"
        )

    def _discrepancy_bound(
        self,
        row_c: NDArray[np.integer],
        row_b: NDArray[np.integer],
    ) -> tuple[ThetaKey | None, float]:
        theta_key = self.interpreter.theta_key(row_b, row_c, self.c, self.l)
        if theta_key[0] == theta_key[1]:
            return None, 0.0
        w = self._w_for_theta(theta_key)
        self.discrepancies.append((self.num_updates + 1, theta_key, w))
        return theta_key, w

    def _recorded_margin(self) -> float:
        if self.base_vertex.tallies is None:
            raise ValueError("base_vertex.tallies must be computed.")
        tallies = np.asarray(self.base_vertex.tallies, dtype=np.float64)
        return float(tallies[self.c] - tallies[self.l])


class CobraQuotaCompilerV2(CobraCompilerV2Base):
    """Noise-filtered COBRA compiler for candidate-to-quota margins."""

    def __init__(
        self,
        interpreter: VertexInterpreter,
        candidate: int | str,
        *,
        margin_type: CriticalMarginType | str,
        quota: int | float | None = None,
        **kwargs: Any,
    ) -> None:
        self.candidate = interpreter.candidate_index(candidate)
        self.candidate_label = str(interpreter.graph.candidate_names[self.candidate])
        self.margin_type = CriticalMarginType(margin_type)
        if self.margin_type == CriticalMarginType.CANDIDATE_ABOVE_QUOTA:
            self.candidate_column = 0
            self.dead_columns = (1,)
        elif self.margin_type == CriticalMarginType.CANDIDATE_BELOW_QUOTA:
            self.candidate_column = 1
            self.dead_columns = (0,)
        else:
            raise ValueError(
                "margin_type must be CANDIDATE_ABOVE_QUOTA or "
                "CANDIDATE_BELOW_QUOTA."
            )
        self.quota = float(getattr(interpreter.graph, "quota") if quota is None else quota)
        self.base_point = self._quota_base_point(interpreter)
        super().__init__(interpreter, **kwargs)

    def _default_label(self) -> str:
        relation = "above quota" if self.candidate_column == 0 else "below quota"
        return (
            f"{self.graph.vertex_label(self.base_vertex.ref)}: "
            f"{self.candidate_label} {relation}"
        )

    def _discrepancy_bound(
        self,
        row_c: NDArray[np.integer],
        row_b: NDArray[np.integer],
    ) -> tuple[ThetaKey | None, float]:
        theta_key = self._quota_theta_key(row_b, row_c)
        if theta_key[0] == theta_key[1]:
            return None, 0.0
        w = self._w_for_theta(theta_key)
        self.discrepancies.append((self.num_updates + 1, theta_key, w))
        return theta_key, w

    def _recorded_margin(self) -> float:
        if self.base_vertex.tallies is None:
            raise ValueError("base_vertex.tallies must be computed.")
        tally = float(np.asarray(self.base_vertex.tallies)[self.candidate])
        if self.margin_type == CriticalMarginType.CANDIDATE_ABOVE_QUOTA:
            return tally - self.quota
        return self.quota - tally

    def _w_for_theta(self, theta_key: ThetaKey) -> float:
        if theta_key in self.optimizer_cache:
            return self.optimizer_cache[theta_key]
        result = self.interpreter.optimize(
            self.base_point,
            theta_key,
            self.radius,
            dead_rows=self.dead_rows,
            dead_columns=self.dead_columns,
            **self.optimizer_kwargs,
        )
        w = float(result.minimum)
        self.optimizer_cache[theta_key] = w
        self.optimizer_results[theta_key] = result
        return w

    def _quota_base_point(self, interpreter: VertexInterpreter) -> NDArray[np.float64]:
        weights = interpreter.profile_wt_vec()
        fpv = interpreter.current_fpv_vec(copy=False)
        prefixes = interpreter.winner_prefix_indices(copy=False)
        columns = np.full(len(fpv), 2, dtype=np.int8)
        columns[fpv == self.candidate] = self.candidate_column

        point = np.zeros(interpreter.shape, dtype=np.float64)
        np.add.at(point, (prefixes, columns), weights)
        return point

    def _quota_theta_key(
        self,
        ballot_row: NDArray[np.integer],
        cvr_row: NDArray[np.integer],
    ) -> ThetaKey:
        return (
            self._quota_flat_coordinate(ballot_row),
            self._quota_flat_coordinate(cvr_row),
        )

    def _quota_flat_coordinate(self, row: NDArray[np.integer]) -> int:
        ballot_row = np.asarray(row, dtype=int)
        prefix = 0
        for bit, edge in enumerate(self.interpreter.seating_edges):
            source = self.graph.vertex(edge.src)
            fpv = self.interpreter._row_fpv(ballot_row, source.key.hopefuls)
            if fpv == edge.candidate:
                prefix |= 1 << bit

        current_fpv = self.interpreter._row_fpv(
            ballot_row,
            self.base_vertex.key.hopefuls,
        )
        column = self.candidate_column if current_fpv == self.candidate else 2
        return prefix * self.interpreter.shape[1] + column


class CobraNoiseFilterCompiler(CobraCompilerV2Base):
    """COBRA compiler verifying the local noise allowance for a margin."""

    def __init__(
        self,
        interpreter: VertexInterpreter,
        c: int | str,
        l: int | str,
        **kwargs: Any,
    ) -> None:
        self.c = interpreter.candidate_index(c)
        self.l = interpreter.candidate_index(l)
        if self.c == self.l:
            raise ValueError("c and l must be distinct candidates.")
        self.c_label = str(interpreter.graph.candidate_names[self.c])
        self.l_label = str(interpreter.graph.candidate_names[self.l])
        super().__init__(interpreter, **kwargs)

    def _default_label(self) -> str:
        return (
            f"{self.graph.vertex_label(self.base_vertex.ref)} noise: "
            f"{self.c_label}>{self.l_label}"
        )

    def _discrepancy_bound(
        self,
        row_c: NDArray[np.integer],
        row_b: NDArray[np.integer],
    ) -> tuple[ThetaKey | None, float]:
        theta_key = self.interpreter.theta_key(row_b, row_c, self.c, self.l)
        if theta_key[0] == theta_key[1]:
            return None, 0.0
        self.discrepancies.append((self.num_updates + 1, theta_key, 1.0))
        return theta_key, 1.0

    def _recorded_margin(self) -> float:
        return self.radius / 2.0

    def _diluted_margin(self) -> float:
        return self.radius / (2.0 * self.N)

    def assorter_from_w(self, w: float) -> float:
        a = 1.0 / (2.0 - self.v)
        return a / 2.0 if w else a


class CobraQuotaNoiseFilterCompiler(CobraCompilerV2Base):
    """Noise-filter compiler for candidate-to-quota local coordinates."""

    def __init__(
        self,
        interpreter: VertexInterpreter,
        candidate: int | str,
        *,
        margin_type: CriticalMarginType | str,
        **kwargs: Any,
    ) -> None:
        self.candidate = interpreter.candidate_index(candidate)
        self.candidate_label = str(interpreter.graph.candidate_names[self.candidate])
        self.margin_type = CriticalMarginType(margin_type)
        if self.margin_type == CriticalMarginType.CANDIDATE_ABOVE_QUOTA:
            self.candidate_column = 0
        elif self.margin_type == CriticalMarginType.CANDIDATE_BELOW_QUOTA:
            self.candidate_column = 1
        else:
            raise ValueError(
                "margin_type must be CANDIDATE_ABOVE_QUOTA or "
                "CANDIDATE_BELOW_QUOTA."
            )
        super().__init__(interpreter, **kwargs)

    def _default_label(self) -> str:
        relation = "above quota" if self.candidate_column == 0 else "below quota"
        return (
            f"{self.graph.vertex_label(self.base_vertex.ref)} noise: "
            f"{self.candidate_label} {relation}"
        )

    def _discrepancy_bound(
        self,
        row_c: NDArray[np.integer],
        row_b: NDArray[np.integer],
    ) -> tuple[ThetaKey | None, float]:
        theta_key = self._quota_theta_key(row_b, row_c)
        if theta_key[0] == theta_key[1]:
            return None, 0.0
        self.discrepancies.append((self.num_updates + 1, theta_key, 1.0))
        return theta_key, 1.0

    def _recorded_margin(self) -> float:
        return self.radius / 2.0

    def _diluted_margin(self) -> float:
        return self.radius / (2.0 * self.N)

    def assorter_from_w(self, w: float) -> float:
        a = 1.0 / (2.0 - self.v)
        return a / 2.0 if w else a

    def _quota_theta_key(
        self,
        ballot_row: NDArray[np.integer],
        cvr_row: NDArray[np.integer],
    ) -> ThetaKey:
        return (
            self._quota_flat_coordinate(ballot_row),
            self._quota_flat_coordinate(cvr_row),
        )

    def _quota_flat_coordinate(self, row: NDArray[np.integer]) -> int:
        ballot_row = np.asarray(row, dtype=int)
        prefix = 0
        for bit, edge in enumerate(self.interpreter.seating_edges):
            source = self.graph.vertex(edge.src)
            fpv = self.interpreter._row_fpv(ballot_row, source.key.hopefuls)
            if fpv == edge.candidate:
                prefix |= 1 << bit

        current_fpv = self.interpreter._row_fpv(
            ballot_row,
            self.base_vertex.key.hopefuls,
        )
        column = self.candidate_column if current_fpv == self.candidate else 2
        return prefix * self.interpreter.shape[1] + column


class CobraMentionsCompilerV2(CobraCompilerV2Base):
    """Noise-filtered compiler for a strong-candidate tally vs weak mentions."""

    def __init__(
        self,
        interpreter: VertexInterpreter,
        weak_candidate: int | str,
        *,
        strong_candidates: Iterable[int],
        frozen_mentions: Sequence[float] | NDArray[np.float64],
        lowest_strong_candidate: int | None = None,
        lowest_strong_tally: int | float | None = None,
        **kwargs: Any,
    ) -> None:
        self.weak_candidate = interpreter.candidate_index(weak_candidate)
        self.weak_label = str(interpreter.graph.candidate_names[self.weak_candidate])
        self.strong_candidates = frozenset(int(candidate) for candidate in strong_candidates)
        if not self.strong_candidates:
            raise ValueError("strong_candidates cannot be empty.")
        if self.weak_candidate in self.strong_candidates:
            raise ValueError("weak_candidate cannot also be a strong candidate.")

        self.frozen_mentions = np.asarray(frozen_mentions, dtype=np.float64)
        if self.frozen_mentions.ndim != 1:
            raise ValueError("frozen_mentions must be one-dimensional.")
        if len(self.frozen_mentions) <= self.weak_candidate:
            raise ValueError("frozen_mentions must be indexed by candidate.")

        if lowest_strong_candidate is None:
            if interpreter.vertex.tallies is None:
                raise ValueError(
                    "lowest_strong_candidate is required when vertex tallies are missing."
                )
            tallies = np.asarray(interpreter.vertex.tallies, dtype=np.float64)
            self.strong_candidate = min(
                self.strong_candidates,
                key=lambda candidate: tallies[candidate],
            )
        else:
            self.strong_candidate = int(lowest_strong_candidate)
            if self.strong_candidate not in self.strong_candidates:
                raise ValueError("lowest_strong_candidate must be in strong_candidates.")

        self.strong_label = str(interpreter.graph.candidate_names[self.strong_candidate])
        if lowest_strong_tally is None:
            if interpreter.vertex.tallies is None:
                raise ValueError(
                    "lowest_strong_tally is required when vertex tallies are missing."
                )
            self.lowest_strong_tally = float(
                np.asarray(interpreter.vertex.tallies, dtype=np.float64)[
                    self.strong_candidate
                ]
            )
        else:
            self.lowest_strong_tally = float(lowest_strong_tally)

        self.base_point = self._mentions_base_point(interpreter)
        super().__init__(interpreter, **kwargs)

    def _default_label(self) -> str:
        return (
            f"{self.graph.vertex_label(self.base_vertex.ref)}: "
            f"{self.strong_label}>{self.weak_label} mentions"
        )

    def _discrepancy_bound(
        self,
        row_c: NDArray[np.integer],
        row_b: NDArray[np.integer],
    ) -> tuple[ThetaKey | None, float]:
        theta_key = self._mentions_theta_key(row_b, row_c)
        if theta_key[0] == theta_key[1]:
            return None, 0.0
        w = self._w_for_theta(theta_key)
        self.discrepancies.append((self.num_updates + 1, theta_key, w))
        return theta_key, w

    def _recorded_margin(self) -> float:
        return self.lowest_strong_tally - float(
            self.frozen_mentions[self.weak_candidate]
        )

    def _mentions_base_point(
        self,
        interpreter: VertexInterpreter,
    ) -> NDArray[np.float64]:
        weights = interpreter.profile_wt_vec()
        prefixes = interpreter.winner_prefix_indices(copy=False)
        columns = np.array(
            [self._mentions_column(row) for row in interpreter.graph.ballot_matrix],
            dtype=np.int8,
        )

        point = np.zeros(interpreter.shape, dtype=np.float64)
        np.add.at(point, (prefixes, columns), weights)
        return point

    def _mentions_theta_key(
        self,
        ballot_row: NDArray[np.integer],
        cvr_row: NDArray[np.integer],
    ) -> ThetaKey:
        return (
            self._mentions_flat_coordinate(ballot_row),
            self._mentions_flat_coordinate(cvr_row),
        )

    def _mentions_flat_coordinate(self, row: NDArray[np.integer]) -> int:
        ballot_row = np.asarray(row, dtype=int)
        prefix = 0
        for bit, edge in enumerate(self.interpreter.seating_edges):
            source = self.graph.vertex(edge.src)
            fpv = self.interpreter._row_fpv(ballot_row, source.key.hopefuls)
            if fpv == edge.candidate:
                prefix |= 1 << bit
        return prefix * self.interpreter.shape[1] + self._mentions_column(ballot_row)

    def _mentions_column(self, row: NDArray[np.integer]) -> int:
        for value in row:
            candidate = int(value)
            if candidate < 0:
                break
            if candidate == self.weak_candidate:
                return 1
            if candidate in self.strong_candidates:
                return 0 if candidate == self.strong_candidate else 2
        return 2


class CobraMentionsNoiseFilterCompiler(CobraMentionsCompilerV2):
    """Noise-filter compiler for candidate-to-mentions local coordinates."""

    def _discrepancy_bound(
        self,
        row_c: NDArray[np.integer],
        row_b: NDArray[np.integer],
    ) -> tuple[ThetaKey | None, float]:
        theta_key = self._mentions_theta_key(row_b, row_c)
        if theta_key[0] == theta_key[1]:
            return None, 0.0
        self.discrepancies.append((self.num_updates + 1, theta_key, 1.0))
        return theta_key, 1.0

    def _recorded_margin(self) -> float:
        return self.radius / 2.0

    def _diluted_margin(self) -> float:
        return self.radius / (2.0 * self.N)

    def assorter_from_w(self, w: float) -> float:
        a = 1.0 / (2.0 - self.v)
        return a / 2.0 if w else a


def plot_profiled_compiler(compiler: CobraCompilerV2Base, *, ax=None):
    if not compiler.profile:
        raise ValueError("Compiler was initialized with profile=False.")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to plot compiler profiles.") from exc

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    xs = np.arange(len(compiler.capital_history))
    ax.scatter(xs, compiler.capital_history, s=14, color="tab:blue")
    ax.axhline(compiler.threshold, color="tab:blue", linestyle="--", linewidth=1)
    ax.set_xlabel("update")
    ax.set_ylabel("Martingale Capital")
    ax.set_title(compiler.label)

    bar_ax = ax.twinx()
    discrepancy_points = [
        (idx, theta_key, float(w))
        for idx, (theta_key, w) in enumerate(
            zip(compiler.theta_key_history, compiler.w_history),
            start=1,
        )
        if theta_key is not None
    ]
    if discrepancy_points:
        bar_x = np.asarray([point[0] for point in discrepancy_points], dtype=float)
        bar_w = np.asarray([point[2] for point in discrepancy_points], dtype=float)
        bar_ax.vlines(
            bar_x,
            0.0,
            bar_w,
            color="tab:orange",
            alpha=0.65,
            linewidth=1.3,
        )
        bar_ax.scatter(
            bar_x,
            bar_w,
            s=14,
            color="tab:orange",
            alpha=0.85,
            zorder=3,
        )
    bar_ax.axhline(0.0, color="tab:orange", linewidth=0.8)
    bar_ax.set_ylim(-2.25, 2.25)
    bar_ax.set_ylabel("Worst-Case Discrepancy Bound")

    visible_low, visible_high = -2.0, 2.0
    for x, theta_key, w in discrepancy_points:
        label_y = 0.0 if w < visible_low or w > visible_high else w
        offset_y = 4 if label_y >= 0.0 else -4
        bar_ax.annotate(
            str(theta_key),
            xy=(x, label_y),
            xytext=(0, offset_y),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset_y >= 0 else "top",
            fontsize=7,
            rotation=0,
            clip_on=False,
        )

    return ax


class CobraCompiler:
    def __init__(
        self,
        base_vertex: ElectionState,
        candidate_index: int,
        candidate_strings: Sequence[str] | EdgeAction | None = None,
        edge_type: EdgeAction | None = None,
        LAM: int | float | None = None,
        quota: int | float | None = None,
        N: int | float | None = None,
        noise_level_guess: float = 0.0,
        simultaneous: bool = False,
        verbose: bool = False,
        remember_discrepancies: bool = True,
        use_fallback: bool = False,
        compiler_label: str | None = None,
        audit_graph: Any | None = None,
        critical_margin_type: CriticalMarginType | str | None = None,
        strong_candidates: Iterable[int] | None = None,
        very_strong_candidates: Iterable[int] | None = None,
        frozen_mentions: Sequence[float] | NDArray[np.float64] | None = None,
        lowest_strong_candidate: int | None = None,
        lowest_strong_tally: int | float | None = None,
    ) -> None:
        if edge_type is None and isinstance(candidate_strings, EdgeAction):
            edge_type = candidate_strings
            candidate_strings = None

        self.base_vertex = base_vertex
        self.base_vertex_label = self._base_vertex_label_from_graph(
            base_vertex,
            audit_graph,
        )
        self.compiler_label = compiler_label
        self.candidate_index = int(candidate_index)
        self.candidate_strings = self._candidate_strings_from(
            candidate_strings,
            base_vertex,
            audit_graph,
        )
        if edge_type is None:
            raise ValueError("edge_type is required.")
        self.edge_type = EdgeAction(edge_type)
        self.LAM = float(self._init_value_from_graph("LAM", LAM, audit_graph))
        self.quota = float(self._init_value_from_graph("quota", quota, audit_graph))
        self.N = float(self._init_total_ballot_weight(N, audit_graph))
        self.noise_level_guess = float(noise_level_guess)
        self.simultaneous = bool(simultaneous)
        self.verbose = bool(verbose)
        self.remember_discrepancies = bool(remember_discrepancies)
        self.use_fallback = bool(use_fallback)
        self.num_updates = 0
        self.discrepancies: list[tuple[int, int]] = []
        self._last_w = 0
        self.used_fallback = False
        self.requested_critical_margin_type = (
            None
            if critical_margin_type is None
            else CriticalMarginType(critical_margin_type)
        )
        self.strong_candidates = frozenset(
            int(candidate) for candidate in (() if strong_candidates is None else strong_candidates)
        )
        self.very_strong_candidates = tuple(
            sorted(
                int(candidate)
                for candidate in (
                    () if very_strong_candidates is None else very_strong_candidates
                )
            )
        )
        self.frozen_mentions = (
            None
            if frozen_mentions is None
            else np.asarray(frozen_mentions, dtype=np.float64)
        )
        self.lowest_strong_candidate = (
            None if lowest_strong_candidate is None else int(lowest_strong_candidate)
        )
        self.lowest_strong_tally = (
            None if lowest_strong_tally is None else float(lowest_strong_tally)
        )

        self._validate_inputs()

        self.tallies = np.asarray(base_vertex.tallies, dtype=np.float64)
        self.hopefuls = frozenset(int(c) for c in base_vertex.key.hopefuls)
        seated_winners_list, winner_context_list = self._seated_winner_data(audit_graph)
        self.seated_winners = np.asarray(
            seated_winners_list,
            dtype=np.int16,
        )
        self.winner_context = winner_context_list
        seated_winner_set = set(self.seated_winners.tolist())
        self.eliminated_candidates = [
            idx
            for idx in range(len(self.candidate_strings))
            if idx not in self.hopefuls and idx not in seated_winner_set
        ]
        self._eliminated_set = frozenset(self.eliminated_candidates)
        self._seated_set = frozenset(int(c) for c in self.seated_winners.tolist())
        self._current_unavailable_set = frozenset(
            set(range(len(self.candidate_strings))) - set(self.hopefuls)
        )

        self.critical_margin_type: CriticalMarginType
        self.critical_margin: float
        self.canonical_victor: int | None = None
        self.canonical_loser: int | None = None
        self.canonical_winner: int | None = None
        self.canonical_non_winner: int | None = None

        self._select_critical_margin()

        if self.critical_margin <= 0:
            raise ValueError(
                f"Critical margin must be positive, got {self.critical_margin}."
            )

        self.v = self.critical_margin / self.N
        self.a = 1.0 / (2.0 - self.v)
        self.M = 1.0
        self.used_fallback = False
        if self.use_fallback:
            self.lambda_value = self._naive_lambda_fallback(
                self.noise_level_guess,
                "Compiler configured to use fallback lambda.",
            )
        else:
            self.lambda_value = self.find_lambda(self.noise_level_guess)

        if self.verbose:
            self.print_info()

    def _candidate_strings_from(
        self,
        candidate_strings: Sequence[str] | EdgeAction | None,
        base_vertex: ElectionState,
        audit_graph: Any | None,
    ) -> list[str]:
        if candidate_strings is not None:
            if isinstance(candidate_strings, EdgeAction):
                raise ValueError("edge_type is required.")
            return list(candidate_strings)
        if audit_graph is not None and hasattr(audit_graph, "candidate_names"):
            return list(audit_graph.candidate_names)
        if base_vertex.tallies is None:
            raise ValueError(
                "candidate_strings can be omitted only when audit_graph exposes "
                "candidate_names or base_vertex.tallies is already available."
            )
        return [str(idx) for idx in range(len(base_vertex.tallies))]

    def _base_vertex_label_from_graph(
        self,
        base_vertex: ElectionState,
        audit_graph: Any | None,
    ) -> str | None:
        if audit_graph is None or not hasattr(audit_graph, "vertex_label"):
            return None
        return audit_graph.vertex_label(base_vertex.ref)

    def _init_value_from_graph(
        self,
        name: str,
        explicit: int | float | None,
        audit_graph: Any | None,
    ) -> int | float:
        if explicit is not None:
            return explicit
        if audit_graph is None or not hasattr(audit_graph, name):
            raise ValueError(f"{name} is required when audit_graph.{name} is unavailable.")
        return getattr(audit_graph, name)

    def _init_total_ballot_weight(
        self,
        explicit: int | float | None,
        audit_graph: Any | None,
    ) -> int | float:
        if explicit is not None:
            return explicit
        if audit_graph is None or not hasattr(audit_graph, "profile"):
            raise ValueError(
                "N is required when audit_graph.profile.total_ballot_wt is unavailable."
            )
        profile = audit_graph.profile
        if not hasattr(profile, "total_ballot_wt"):
            raise ValueError(
                "N is required when audit_graph.profile.total_ballot_wt is unavailable."
            )
        return profile.total_ballot_wt

    def _validate_inputs(self) -> None:
        if not 0 <= self.candidate_index < len(self.candidate_strings):
            raise ValueError(f"Invalid candidate_index: {self.candidate_index}.")
        if self.base_vertex.tallies is None:
            raise ValueError("base_vertex.tallies must be computed before initializing.")
        if not hasattr(self.base_vertex.key, "hopefuls"):
            raise ValueError("base_vertex.key must expose a hopefuls set.")
        is_mentions_margin = (
            self.requested_critical_margin_type
            == CriticalMarginType.CANDIDATE_TO_MENTIONS
        )
        if not is_mentions_margin and self.candidate_index not in self.base_vertex.key.hopefuls:
            raise ValueError(
                "Escape-edge candidate is not hopeful at the base vertex: "
                f"{self._candidate_label(self.candidate_index)}."
            )
        tallies = np.asarray(self.base_vertex.tallies)
        if tallies.ndim != 1 or len(tallies) != len(self.candidate_strings):
            raise ValueError("base_vertex.tallies must be one-dimensional by candidate.")
        if self.LAM <= 0 or self.N <= 0 or self.quota <= 0:
            raise ValueError("LAM, N, and quota must all be positive.")
        if not 0.0 <= self.noise_level_guess <= 1.0:
            raise ValueError("noise_level_guess must be in [0, 1].")
        if self.LAM >= 2.0 * self.quota / 3.0:
            self._warn("LAM is at least 2*quota/3; COBRA assumptions may be weak.")
        if self.edge_type not in (
            EdgeAction.ELIMINATE,
            EdgeAction.ELECT,
            EdgeAction.FORCE_ELECT,
            EdgeAction.SIMULTANEOUS_ELECT,
        ):
            raise ValueError(f"Unsupported edge_type: {self.edge_type}.")

    def _seated_winner_data(
        self,
        audit_graph: Any | None,
    ) -> tuple[list[int], list[frozenset[int]]]:
        if hasattr(self.base_vertex, "seated_at"):
            seated_at = getattr(self.base_vertex, "seated_at")
        elif hasattr(self.base_vertex.key, "elected_candidates"):
            seated_at = self.base_vertex.key.elected_candidates
        elif hasattr(self.base_vertex.key, "seated_at"):
            seated_at = self.base_vertex.key.seated_at
        else:
            seated_at = ()

        seated: list[int] = []
        winner_context: list[frozenset[int]] = []
        for item in seated_at:
            if item is None:
                continue
            if isinstance(item, (int, np.integer)):
                seated.append(int(item))
                winner_context.append(self._base_vertex_unavailable_context())
                continue
            if audit_graph is None:
                raise ValueError(
                    "base_vertex.key.seated_at contains edge refs, so audit_graph "
                    "is required to recover seated winners and winner contexts."
                )

            edge = audit_graph.edge(item)
            action = EdgeAction(edge.action)
            if not action.is_election:
                raise ValueError(
                    "base_vertex.key.seated_at referenced a non-election edge: "
                    f"{item} has action {action.name}."
                )
            seated.append(int(edge.candidate))
            winner_context.append(
                self._winner_context_from_seating_edge(edge, audit_graph)
            )

        return seated, winner_context

    def _winner_context_from_seating_edge(
        self,
        seating_edge: Any,
        audit_graph: Any,
    ) -> frozenset[int]:
        parent = audit_graph.vertex(seating_edge.src)
        if not hasattr(parent.key, "hopefuls"):
            raise ValueError(
                "Cannot recover winner context: seating edge parent key does "
                "not expose hopefuls."
            )
        return frozenset(
            set(range(len(self.candidate_strings))) - set(parent.key.hopefuls)
        )

    def _base_vertex_unavailable_context(self) -> frozenset[int]:
        return frozenset(
            set(range(len(self.candidate_strings))) - set(self.base_vertex.key.hopefuls)
        )

    def _select_critical_margin(self) -> None:
        if self.requested_critical_margin_type == CriticalMarginType.CANDIDATE_TO_MENTIONS:
            self._select_candidate_to_mentions_margin()
            return
        if self.edge_type == EdgeAction.ELIMINATE:
            self._select_elimination_escape_margin()
        else:
            self._select_election_escape_margin()

    def _select_candidate_to_mentions_margin(self) -> None:
        if not self.strong_candidates:
            raise ValueError("candidate-to-mentions compilers require strong_candidates.")
        if self.frozen_mentions is None:
            raise ValueError("candidate-to-mentions compilers require frozen_mentions.")
        if len(self.frozen_mentions) != len(self.candidate_strings):
            raise ValueError("frozen_mentions must be indexed by candidate.")

        if self.lowest_strong_candidate is None:
            self.lowest_strong_candidate = min(
                self.strong_candidates,
                key=lambda candidate: self.tallies[candidate],
            )
        if self.lowest_strong_tally is None:
            self.lowest_strong_tally = float(self.tallies[self.lowest_strong_candidate])

        self.critical_margin_type = CriticalMarginType.CANDIDATE_TO_MENTIONS
        self.canonical_victor = int(self.lowest_strong_candidate)
        self.canonical_loser = int(self.candidate_index)
        self.critical_margin = (
            float(self.lowest_strong_tally)
            - float(self.frozen_mentions[self.candidate_index])
        )

    def _select_elimination_escape_margin(self) -> None:
        forced = np.where(self.tallies >= self.quota + self.LAM)[0]
        if len(forced) > 0:
            winner = int(forced[np.argmax(self.tallies[forced])])
            self.critical_margin_type = CriticalMarginType.CANDIDATE_ABOVE_QUOTA
            self.canonical_winner = winner
            self.critical_margin = float(self.tallies[winner] - self.quota)
            return

        hopefuls = np.asarray(sorted(self.hopefuls), dtype=int)
        lowest = int(hopefuls[np.argmin(self.tallies[hopefuls])])
        if lowest == self.candidate_index:
            raise ValueError(
                "Elimination escape edge is incoherent: candidate is already the "
                f"lowest hopeful. candidate={self._candidate_label(self.candidate_index)}, "
                f"tallies={self.tallies}."
            )

        c_tally = float(self.tallies[self.candidate_index])
        lowest_tally = float(self.tallies[lowest])
        if lowest_tally + self.LAM >= c_tally:
            self._warn(
                "Elimination escape margin is not more than LAM: "
                f"candidate={self._candidate_label(self.candidate_index)} "
                f"tally={c_tally}; lowest={self._candidate_label(lowest)} "
                f"tally={lowest_tally}."
            )

        self.critical_margin_type = CriticalMarginType.CANDIDATE_TO_CANDIDATE
        self.canonical_victor = self.candidate_index
        self.canonical_loser = lowest
        self.critical_margin = c_tally - lowest_tally

    def _select_election_escape_margin(self) -> None:
        c_tally = float(self.tallies[self.candidate_index])

        if not self.simultaneous:
            challengers = np.where(self.tallies > c_tally + self.LAM)[0]
            challengers = np.asarray(
                [idx for idx in challengers if idx != self.candidate_index],
                dtype=int,
            )
            if len(challengers) > 0:
                winner = int(challengers[np.argmax(self.tallies[challengers])])
                self.critical_margin_type = CriticalMarginType.CANDIDATE_TO_CANDIDATE
                self.canonical_victor = winner
                self.canonical_loser = self.candidate_index
                self.critical_margin = float(self.tallies[winner] - c_tally)
                return

        if c_tally + self.LAM < self.quota:
            self.critical_margin_type = CriticalMarginType.CANDIDATE_BELOW_QUOTA
            self.canonical_non_winner = self.candidate_index
            self.critical_margin = float(self.quota - c_tally)
            return

        raise ValueError(
            "Could not identify a critical margin for election escape edge: "
            f"candidate={self._candidate_label(self.candidate_index)}, "
            f"tally={c_tally}, quota={self.quota}, LAM={self.LAM}, "
            f"simultaneous={self.simultaneous}."
        )

    def naive_lambda(self, noise_level: float | None = None) -> float:
        p_0 = (
            1.0 - self.noise_level_guess
            if noise_level is None
            else 1.0 - float(noise_level)
        )
        denom = 1.0 - 2.0 * self.a
        if denom == 0.0:
            return self._fallback_lambda("Naive lambda denominator is zero.")
        return self._lambda_in_range((2.0 - 4.0 * self.a * p_0) / denom)

    def find_lambda(self, noise_level: float | None = None) -> float:
        noise = self.noise_level_guess if noise_level is None else float(noise_level)
        n_candidates = max(len(self.candidate_strings), 1)
        p_2 = noise / n_candidates
        p_1 = noise - p_2
        p_0 = 1.0 - p_1 - p_2

        def objective(lam: float) -> float:
            return (
                ((self.a - 0.5) * p_0) / (1.0 + lam * (self.a - 0.5))
                + ((self.a - 1.0) * p_1) / (2.0 - lam * (1.0 - self.a))
                - p_2 / (2.0 - lam)
            )

        lo = np.nextafter(0.0, 1.0)
        hi = np.nextafter(2.0, 0.0)
        f_lo = objective(0.0)
        if abs(f_lo) <= 1e-12:
            return self._fallback_lambda(
                "COBRA lambda objective has only a lower-endpoint root."
            )
        if not np.isfinite(f_lo):
            return self._naive_lambda_fallback(
                noise,
                "COBRA lambda objective is non-finite at lower endpoint.",
            )

        roots = self._find_objective_roots(objective, lo, hi)
        decreasing_roots = [
            root
            for root in roots
            if self._numerical_derivative(objective, root, lo, hi) < 0.0
        ]

        if not decreasing_roots:
            self._warn(
                "No decreasing COBRA lambda critical point found in [0, 2); "
                "using naive lambda."
            )
            return self._naive_lambda_fallback(noise, "No decreasing critical point.")

        best_root = min(
            decreasing_roots,
            key=lambda root: self._numerical_derivative(objective, root, lo, hi),
        )
        return self._numerical_lambda_or_backup(best_root, noise)

    def _find_objective_roots(
        self,
        objective,
        lo: float,
        hi: float,
        n_grid: int = 512,
    ) -> list[float]:
        grid = np.linspace(lo, hi, n_grid + 1)
        values = np.asarray([objective(float(x)) for x in grid], dtype=np.float64)
        roots: list[float] = []

        for i in range(n_grid):
            left = float(grid[i])
            right = float(grid[i + 1])
            f_left = float(values[i])
            f_right = float(values[i + 1])
            if not np.isfinite(f_left) or not np.isfinite(f_right):
                continue
            if abs(f_left) <= 1e-12:
                roots.append(left)
                continue
            if f_left * f_right > 0.0:
                continue
            roots.append(self._bisect_root(objective, left, right, f_left, f_right))

        deduped: list[float] = []
        for root in roots:
            if root <= 0.0 or root >= 2.0:
                continue
            if all(abs(root - existing) > 1e-8 for existing in deduped):
                deduped.append(root)
        return deduped

    def _bisect_root(
        self,
        objective,
        lo: float,
        hi: float,
        f_lo: float,
        f_hi: float,
    ) -> float:
        if abs(f_hi) < abs(f_lo):
            best = hi
        else:
            best = lo

        for _ in range(80):
            mid = (lo + hi) / 2.0
            f_mid = float(objective(mid))
            if abs(f_mid) < abs(objective(best)):
                best = mid
            if abs(f_mid) <= 1e-14:
                return mid
            if f_lo * f_mid <= 0.0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid

        return best

    def _numerical_derivative(
        self,
        objective,
        x: float,
        lo: float,
        hi: float,
    ) -> float:
        h = min(1e-5, max((x - lo) / 2.0, 0.0), max((hi - x) / 2.0, 0.0))
        if h <= 0.0:
            h = 1e-8
        left = max(lo, x - h)
        right = min(hi, x + h)
        if right <= left:
            return float("nan")
        return float((objective(right) - objective(left)) / (right - left))

    def _lambda_in_range(self, value: float) -> float:
        if not np.isfinite(value):
            return self._fallback_lambda("Non-finite lambda.")
        if value <= 0.0:
            return self._fallback_lambda(f"Lambda {value} is not positive.")
        if value >= 2.0:
            self._warn(f"Lambda {value} outside [0, 2); clipping below 2.")
            return float(np.nextafter(2.0, 0.0))
        return float(value)

    def _fallback_lambda(self, reason: str) -> float:
        self.used_fallback = True
        self._warn(f"{reason} Using fallback lambda={DEFAULT_LAMBDA_VALUE}.")
        return DEFAULT_LAMBDA_VALUE

    def _naive_lambda_fallback(self, noise_level: float, reason: str) -> float:
        self.used_fallback = True
        self._warn(f"{reason} Using naive lambda.")
        return self.naive_lambda(noise_level)

    def _numerical_lambda_or_backup(
        self,
        value: float,
        noise_level: float,
    ) -> float:
        lambda_value = self._lambda_in_range(value)
        if lambda_value < MIN_NUMERICAL_LAMBDA_VALUE:
            self._warn(
                "Numerical COBRA lambda is below "
                f"{MIN_NUMERICAL_LAMBDA_VALUE}; using naive lambda."
            )
            return self._naive_lambda_fallback(
                noise_level,
                "Numerical COBRA lambda is too small.",
            )
        return lambda_value

    def _warning_label(self) -> str:
        if self.compiler_label is not None:
            return self.compiler_label
        if self.base_vertex_label is not None:
            return self.base_vertex_label
        return "unlabeled compiler"

    def _warn(self, message: str) -> None:
        warnings.warn(
            f"{self._warning_label()}: {message}",
            RuntimeWarning,
            stacklevel=2,
        )

    def assorter(
        self,
        row_c: NDArray[np.integer],
        row_b: NDArray[np.integer],
    ) -> float:
        self._last_w = 0
        row_c = np.asarray(row_c)
        row_b = np.asarray(row_b)
        if row_c.shape != row_b.shape:
            raise ValueError("CVR and ballot rows must have the same shape.")
        if np.array_equal(row_c, row_b):
            return self.a

        if self.critical_margin_type == CriticalMarginType.CANDIDATE_TO_MENTIONS:
            winner_prefix_c, fpv_c = self._project_mentions_row(row_c)
            winner_prefix_b, fpv_b = self._project_mentions_row(row_b)
        else:
            winner_prefix_c, fpv_c = self._project_row(row_c)
            winner_prefix_b, fpv_b = self._project_row(row_b)

        if np.array_equal(winner_prefix_c, winner_prefix_b) and fpv_c == fpv_b:
            return self.a

        w = self._raw_overstatement(fpv_c, fpv_b)
        if np.any(winner_prefix_c) or np.any(winner_prefix_b):
            w = max(1, w)
        self._last_w = int(w)

        if self.verbose:
            print(
                "Detected discrepancy: "
                f"fpv_c={self._candidate_label(fpv_c)}, "
                f"fpv_b={self._candidate_label(fpv_b)}, w={w}."
            )

        return {
            2: 0.0,
            1: self.a / 2.0,
            0: self.a,
            -1: 3.0 * self.a / 2.0,
            -2: 2.0 * self.a,
        }[int(w)]

    def _project_row(
        self,
        row: NDArray[np.integer],
    ) -> tuple[NDArray[np.int8], int]:
        winner_prefix = np.zeros(len(self.seated_winners), dtype=np.int8)

        for i, winner in enumerate(self.seated_winners.tolist()):
            fpv_at_seating = self._first_unmasked_candidate(
                row,
                self.winner_context[i],
            )
            if fpv_at_seating == int(winner):
                winner_prefix[i] = 1

        fpv = self._first_unmasked_candidate(row, self._current_unavailable_set)
        return winner_prefix, fpv

    def _project_mentions_row(
        self,
        row: NDArray[np.integer],
    ) -> tuple[NDArray[np.int8], int]:
        allowed = self.strong_candidates | {self.candidate_index} | set(
            self.very_strong_candidates
        )
        projected = []
        for value in row:
            candidate = int(value)
            if candidate < 0:
                break
            if candidate in allowed:
                projected.append(candidate)

        winner_prefix = np.zeros(len(self.very_strong_candidates), dtype=np.int8)
        very_strong_index = {
            candidate: idx
            for idx, candidate in enumerate(self.very_strong_candidates)
        }

        pos = 0
        while pos < len(projected):
            candidate = projected[pos]
            idx = very_strong_index.get(candidate)
            if idx is None:
                break
            winner_prefix[idx] = 1
            pos += 1

        if pos >= len(projected):
            return winner_prefix, int(SENTINEL)
        return winner_prefix, int(projected[pos])

    def _first_unmasked_candidate(
        self,
        row: NDArray[np.integer],
        masked_candidates: frozenset[int],
    ) -> int:
        for value in row:
            candidate = int(value)
            if candidate < 0:
                return int(SENTINEL)
            if candidate in masked_candidates:
                continue
            return candidate

        return int(SENTINEL)

    def _raw_overstatement(self, fpv_c: int, fpv_b: int) -> int:
        if self.critical_margin_type in (
            CriticalMarginType.CANDIDATE_TO_CANDIDATE,
            CriticalMarginType.CANDIDATE_TO_MENTIONS,
        ):
            assert self.canonical_victor is not None
            assert self.canonical_loser is not None
            return (
                int(fpv_c == self.canonical_victor)
                - int(fpv_b == self.canonical_victor)
                + int(fpv_b == self.canonical_loser)
                - int(fpv_c == self.canonical_loser)
            )

        if self.critical_margin_type == CriticalMarginType.CANDIDATE_ABOVE_QUOTA:
            assert self.canonical_winner is not None
            return (
                int(fpv_c == self.canonical_winner)
                - int(fpv_b == self.canonical_winner)
            )

        assert self.canonical_non_winner is not None
        return (
            int(fpv_b == self.canonical_non_winner)
            - int(fpv_c == self.canonical_non_winner)
        )

    def update(
        self,
        row_c: NDArray[np.integer],
        row_b: NDArray[np.integer],
        verbose: bool | None = None,
    ) -> float:
        x = self.assorter(row_c, row_b)
        self.num_updates += 1
        if self.remember_discrepancies and self._last_w != 0:
            self.discrepancies.append((self.num_updates, self._last_w))
        self.M *= 1.0 + self.lambda_value * (x - 0.5)
        should_print = self.verbose if verbose is None else bool(verbose)
        if should_print:
            print(f"COBRA capital M={self.M}")
        return self.M

    def reset(self) -> None:
        self.M = 1.0
        self.num_updates = 0
        self.discrepancies.clear()
        self._last_w = 0

    def get_info(self) -> str:
        label = self._warning_label()
        margin = self._critical_margin_info()
        return (
            "CobraCompiler("
            f"label={label}, "
            f"edge_type={self.edge_type.name}, "
            f"candidate={self._candidate_label(self.candidate_index)}, "
            f"critical_margin_type={self.critical_margin_type.value}, "
            f"critical_margin={self.critical_margin}, "
            f"margin={margin}, "
            f"lambda_value={self.lambda_value}, "
            f"used_fallback={self.used_fallback}, "
            f"M={self.M}, "
            f"num_updates={self.num_updates}"
            ")"
        )

    def __repr__(self) -> str:
        return self.get_info()

    def print_info(self) -> None:
        print("COBRA compiler")
        if self.compiler_label is not None:
            print(f"  label: {self.compiler_label}")
        elif self.base_vertex_label is not None:
            print(f"  base_vertex: {self.base_vertex_label}")
        print(f"  edge_type: {self.edge_type.name}")
        print(f"  critical_margin_type: {self.critical_margin_type.value}")
        print(f"  critical_margin: {self.critical_margin}")
        for attr in (
            "canonical_victor",
            "canonical_loser",
            "canonical_winner",
            "canonical_non_winner",
        ):
            value = getattr(self, attr)
            if value is not None:
                print(f"  {attr}: {self._candidate_label(value)}")
        print(f"  hopefuls: {self._candidate_labels(sorted(self.hopefuls))}")
        print(f"  seated_winners: {self._candidate_labels(self.seated_winners)}")
        print(
            "  eliminated_candidates: "
            f"{self._candidate_labels(self.eliminated_candidates)}"
        )
        print(f"  v: {self.v}")
        print(f"  a: {self.a}")
        print(f"  lambda_value: {self.lambda_value}")
        print(f"  used_fallback: {self.used_fallback}")
        print(f"  M: {self.M}")
        print(f"  num_updates: {self.num_updates}")
        if self.remember_discrepancies:
            print(f"  discrepancies: {self.discrepancies}")

    def _candidate_label(self, idx: int) -> str:
        if idx < 0:
            return "exhausted"
        return f"{idx}:{self.candidate_strings[int(idx)]}"

    def _candidate_labels(self, indices: Iterable[int]) -> list[str]:
        return [self._candidate_label(int(idx)) for idx in indices]

    def _critical_margin_info(self) -> str:
        if self.canonical_victor is not None and self.canonical_loser is not None:
            return (
                f"{self._candidate_label(self.canonical_victor)} > "
                f"{self._candidate_label(self.canonical_loser)}"
            )
        if self.canonical_winner is not None:
            return f"{self._candidate_label(self.canonical_winner)} >= quota"
        if self.canonical_non_winner is not None:
            return f"{self._candidate_label(self.canonical_non_winner)} < quota"
        if self.critical_margin_type == CriticalMarginType.CANDIDATE_TO_MENTIONS:
            return "mentions margin"
        return "unknown"

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .cobra import CobraCompiler
from .cobra import (
    CobraCompilerV2,
    CobraCompilerV2Base,
    CobraMentionsCompilerV2,
    CobraMentionsNoiseFilterCompiler,
    CobraNoiseFilterCompiler,
    CobraQuotaCompilerV2,
    CobraQuotaNoiseFilterCompiler,
    CriticalMarginType,
    plot_profiled_compiler,
)
from .interpreter import VertexInterpreter
from .noise import ImplicitSampler

try:
    from ..election_graphs.datatypes import EdgeAction, ElectionStatus
    from ..election_graphs.utils import (
        fpv_tallies_from_matrix,
        frozen_mentions_from_matrix,
    )
except ImportError:
    from election_graphs.datatypes import EdgeAction, ElectionStatus
    from election_graphs.utils import (
        fpv_tallies_from_matrix,
        frozen_mentions_from_matrix,
    )


@dataclass(frozen=True, slots=True)
class EscapeCompilerInfo:
    escape_id: str
    base_label: str
    base_layer: int
    base_local_id: int
    action: EdgeAction
    candidate: int
    candidate_name: str


class GlobalAuditDriver:
    """Audit every missing graph edge with COBRA assertions or noise filters.

    Set ``compiler_type="noise"`` to replace each assertion compiler with a
    local noise-filter compiler. Candidate-to-candidate noise budgets are half
    the escape margin; all other margin types use the full escape margin.
    """

    def __init__(
        self,
        audit_graph: Any,
        noise_level: float,
        noise_guess: float | None = None,
        sample_size: int | None = None,
        fractional_sample_size: float | None = None,
        seed: int | None = None,
        compiler_type: str = "cobra",
        print_diagnostics_every: int = 30,
        alpha: float = 0.05,
        keep_certified_compilers: bool = False,
        cache_noised_rows: bool = False,
        remember_discrepancies: bool = True,
        simultaneous: bool = False,
        use_fallback: bool = True,
        BAL: NDArray[np.integer] | None = None,
        CVR: NDArray[np.integer] | None = None,
    ) -> None:
        compiler_type = str(compiler_type).strip().lower()
        if compiler_type not in {"cobra", "noise"}:
            raise ValueError(
                f"Unsupported compiler_type: {compiler_type}. "
                "Expected 'cobra' or 'noise'."
            )
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")

        self.verbose = False
        self.audit_graph = audit_graph
        self.compiler_type = compiler_type
        self.noise_level = float(noise_level)
        self.noise_guess = self.noise_level if noise_guess is None else float(noise_guess)
        self.print_diagnostics_every = int(print_diagnostics_every)
        self.alpha = float(alpha)
        self.threshold = 1.0 / self.alpha
        self.keep_certified_compilers = bool(keep_certified_compilers)
        self.use_fallback = bool(use_fallback)
        self.i = 0
        self.m = int(audit_graph.m)
        self.sampler: ImplicitSampler | None = None
        self.BAL: NDArray[np.integer] | None = None
        self.CVR: NDArray[np.integer] | None = None
        self.seeded_graph = bool(getattr(audit_graph, "used_seeded_build", False))
        self.seed_very_strong_candidates = frozenset(
            getattr(audit_graph, "seed_very_strong_candidates", frozenset())
        )
        self.seed_strong_candidates = frozenset(
            getattr(audit_graph, "seed_strong_candidates", frozenset())
        )
        self.seed_weak_candidates = frozenset(
            getattr(audit_graph, "seed_weak_candidates", frozenset())
        )
        self.seed_frozen_mentions: NDArray[np.float64] | None = None
        self.seed_prebatch_strong_tallies: NDArray[np.float64] | None = None
        self.sample_size = self._initialize_sample_source(
            audit_graph=audit_graph,
            sample_size=sample_size,
            fractional_sample_size=fractional_sample_size,
            seed=seed,
            cache_noised_rows=cache_noised_rows,
            BAL=BAL,
            CVR=CVR,
        )

        self.interpreters: dict[Any, VertexInterpreter] = {}
        self.compilers: list[CobraCompilerV2Base | CobraCompiler] = []
        self.compiler_info: list[EscapeCompilerInfo] = []
        self.compiler_index_by_escape_id: dict[str, int] = {}
        self._initialize_compilers(
            audit_graph=audit_graph,
            remember_discrepancies=remember_discrepancies,
            simultaneous=simultaneous,
            use_fallback=self.use_fallback,
        )
        if self.seeded_graph:
            self._initialize_seeded_graph_compilers(
                audit_graph=audit_graph,
                remember_discrepancies=remember_discrepancies,
                simultaneous=simultaneous,
                use_fallback=self.use_fallback,
            )
        self.deduplicate_compilers()

        self.pool: list[int] = list(range(len(self.compilers)))
        self.certified: set[int] = set()
        self.certification_order: list[int] = []

    def _log(self, message: str) -> None:
        if getattr(self, "verbose", False):
            print(message)

    def _wt_vec_from_graph(self, audit_graph: Any) -> NDArray[np.float64]:
        if hasattr(audit_graph, "_unseeded_root_wt_vec"):
            return np.asarray(audit_graph._unseeded_root_wt_vec, dtype=np.float64)
        if hasattr(audit_graph, "root_wt_vec"):
            return np.asarray(audit_graph.root_wt_vec, dtype=np.float64)
        if hasattr(audit_graph, "profile"):
            return ImplicitSampler._wt_vec_from_profile(audit_graph.profile)
        raise ValueError("Could not recover wt_vec from audit_graph.")

    def _initialize_sample_source(
        self,
        audit_graph: Any,
        sample_size: int | None,
        fractional_sample_size: float | None,
        seed: int | None,
        cache_noised_rows: bool,
        BAL: NDArray[np.integer] | None,
        CVR: NDArray[np.integer] | None,
    ) -> int:
        if (BAL is None) != (CVR is None):
            raise ValueError("BAL and CVR must be provided together.")

        if BAL is not None and CVR is not None:
            if sample_size is not None or fractional_sample_size is not None:
                raise ValueError(
                    "sample_size and fractional_sample_size should not be provided "
                    "when BAL/CVR sample matrices are provided."
                )
            bal = np.asarray(BAL)
            cvr = np.asarray(CVR)
            if bal.ndim != 2 or cvr.ndim != 2:
                raise ValueError("BAL and CVR must be two-dimensional arrays.")
            if bal.shape != cvr.shape:
                raise ValueError("BAL and CVR must have the same shape.")
            self.BAL = bal
            self.CVR = cvr
            self._log(
                f"  sample source: provided BAL/CVR matrices with {cvr.shape[0]} rows"
            )
            return int(cvr.shape[0])

        wt_vec = self._wt_vec_from_graph(audit_graph)
        self.sampler = ImplicitSampler(
            noise_level=self.noise_level,
            n_cands=int(audit_graph.n_candidates),
            wt_vec=wt_vec,
            sample_size=sample_size,
            fractional_sample_size=fractional_sample_size,
            with_replacement=True,
            seed=seed,
            cache_rows=cache_noised_rows,
        )
        return int(self.sampler.sample_size)

    def _initialize_compilers(
        self,
        audit_graph: Any,
        remember_discrepancies: bool,
        simultaneous: bool,
        use_fallback: bool,
    ) -> None:
        for layer in audit_graph.layers:
            for vertex in layer:
                if vertex.status == ElectionStatus.TERMINAL:
                    continue
                if not hasattr(vertex.key, "hopefuls"):
                    raise ValueError("Audit graph vertex key does not expose hopefuls.")
                if len(vertex.key.hopefuls) + int(vertex.degree) == self.m:
                    continue

                outgoing = audit_graph.outgoing_edges(vertex.ref)
                if not outgoing and getattr(vertex, "path_multiplicity", 1) == 0:
                    continue
                if vertex.tallies is None:
                    label = (
                        audit_graph.vertex_label(vertex.ref)
                        if hasattr(audit_graph, "vertex_label")
                        else str(vertex.ref)
                    )
                    raise ValueError(
                        "Cannot initialize COBRA compilers from vertex "
                        f"{label}: base_vertex.tallies is not computed."
                    )

                for candidate in sorted(vertex.key.hopefuls):
                    if not self._has_outgoing_elimination(outgoing, candidate):
                        self._add_compiler(
                            audit_graph=audit_graph,
                            vertex=vertex,
                            candidate=int(candidate),
                            action=EdgeAction.ELIMINATE,
                            remember_discrepancies=remember_discrepancies,
                            simultaneous=simultaneous,
                            use_fallback=use_fallback,
                        )
                    if not self._has_outgoing_election(outgoing, candidate):
                        self._add_compiler(
                            audit_graph=audit_graph,
                            vertex=vertex,
                            candidate=int(candidate),
                            action=EdgeAction.ELECT,
                            remember_discrepancies=remember_discrepancies,
                            simultaneous=simultaneous,
                            use_fallback=use_fallback,
                        )

    def _has_outgoing_elimination(
        self,
        outgoing: list[Any],
        candidate: int,
    ) -> bool:
        return any(
            edge.candidate == candidate and edge.action == EdgeAction.ELIMINATE
            for edge in outgoing
        )

    def _has_outgoing_election(
        self,
        outgoing: list[Any],
        candidate: int,
    ) -> bool:
        return any(
            edge.candidate == candidate
            and EdgeAction(edge.action).is_election
            for edge in outgoing
        )

    def _add_compiler(
        self,
        audit_graph: Any,
        vertex: Any,
        candidate: int,
        action: EdgeAction,
        remember_discrepancies: bool,
        simultaneous: bool,
        use_fallback: bool,
    ) -> None:
        info = self._make_compiler_info(
            audit_graph=audit_graph,
            vertex=vertex,
            candidate=candidate,
            action=action,
        )
        compiler: CobraCompilerV2Base | CobraCompiler
        if self.compiler_type == "noise":
            compiler = self._make_noise_filter_compiler(
                audit_graph=audit_graph,
                vertex=vertex,
                candidate=candidate,
                action=action,
                simultaneous=simultaneous,
                label=info.escape_id,
            )
        else:
            compiler = CobraCompiler(
                vertex,
                candidate,
                action,
                audit_graph=audit_graph,
                noise_level_guess=self.noise_guess,
                simultaneous=simultaneous,
                remember_discrepancies=remember_discrepancies,
                use_fallback=use_fallback,
                compiler_label=info.escape_id,
            )
        self.compilers.append(compiler)
        self.compiler_info.append(info)
        self.compiler_index_by_escape_id[info.escape_id.upper()] = (
            len(self.compilers) - 1
        )

    def _make_noise_filter_compiler(
        self,
        audit_graph: Any,
        vertex: Any,
        candidate: int,
        action: EdgeAction,
        simultaneous: bool,
        label: str,
    ) -> CobraCompilerV2Base:
        interpreter = self._interpreter_for_vertex(audit_graph, vertex)
        margin = self._noise_filter_margin(vertex, candidate, action, simultaneous)
        margin_type = CriticalMarginType(margin["type"])
        escape_margin = float(margin["margin"])
        noise_budget = (
            escape_margin / 2.0
            if margin_type == CriticalMarginType.CANDIDATE_TO_CANDIDATE
            else escape_margin
        )
        common: dict[str, Any] = {
            "LAM": float(audit_graph.LAM),
            "critical_margin": noise_budget,
            "radius": 2.0 * noise_budget,
            "alpha": self.alpha,
            "label": label,
        }

        compiler: CobraCompilerV2Base
        if margin_type == CriticalMarginType.CANDIDATE_TO_CANDIDATE:
            compiler = CobraNoiseFilterCompiler(
                interpreter,
                int(margin["c"]),
                int(margin["l"]),
                **common,
            )
        else:
            compiler = CobraQuotaNoiseFilterCompiler(
                interpreter,
                int(margin["candidate"]),
                margin_type=margin_type,
                **common,
            )
        return compiler

    def _interpreter_for_vertex(
        self,
        audit_graph: Any,
        vertex: Any,
    ) -> VertexInterpreter:
        existing = self.interpreters.get(vertex.ref)
        if existing is not None:
            return existing
        interpreter = VertexInterpreter(audit_graph, vertex)
        self.interpreters[vertex.ref] = interpreter
        return interpreter

    def _noise_filter_margin(
        self,
        vertex: Any,
        candidate: int,
        action: EdgeAction,
        simultaneous: bool,
    ) -> dict[str, Any]:
        """Identify the local coordinates whose noise can enable an escape edge."""
        tallies = np.asarray(vertex.tallies, dtype=np.float64)
        graph = self.audit_graph

        if action == EdgeAction.ELIMINATE:
            forced = np.where(tallies >= graph.quota + graph.LAM)[0]
            if len(forced) > 0:
                winner = int(forced[np.argmax(tallies[forced])])
                return {
                    "type": CriticalMarginType.CANDIDATE_ABOVE_QUOTA,
                    "candidate": winner,
                    "margin": float(tallies[winner] - graph.quota),
                }

            hopefuls = np.asarray(sorted(vertex.key.hopefuls), dtype=int)
            lowest = int(hopefuls[np.argmin(tallies[hopefuls])])
            if lowest == candidate:
                raise ValueError(
                    "Elimination escape edge is incoherent: candidate is already "
                    f"the lowest hopeful. candidate={candidate}, tallies={tallies}."
                )
            return {
                "type": CriticalMarginType.CANDIDATE_TO_CANDIDATE,
                "c": int(candidate),
                "l": lowest,
                "margin": float(tallies[candidate] - tallies[lowest]),
            }

        c_tally = float(tallies[candidate])
        if not simultaneous:
            challengers = np.where(tallies > c_tally + graph.LAM)[0]
            challengers = np.asarray(
                [idx for idx in challengers if idx != candidate],
                dtype=int,
            )
            if len(challengers) > 0:
                winner = int(challengers[np.argmax(tallies[challengers])])
                return {
                    "type": CriticalMarginType.CANDIDATE_TO_CANDIDATE,
                    "c": winner,
                    "l": int(candidate),
                    "margin": float(tallies[winner] - c_tally),
                }

        if c_tally + graph.LAM < graph.quota:
            return {
                "type": CriticalMarginType.CANDIDATE_BELOW_QUOTA,
                "candidate": int(candidate),
                "margin": float(graph.quota - c_tally),
            }

        raise ValueError(
            "Could not identify a noise-filter margin for election escape edge: "
            f"candidate={candidate}, tally={c_tally}, quota={graph.quota}, "
            f"LAM={graph.LAM}, simultaneous={simultaneous}."
        )

    def add_compiler(
        self,
        compiler: CobraCompilerV2Base | CobraCompiler,
        info: EscapeCompilerInfo | None = None,
    ) -> None:
        if info is None:
            if not isinstance(compiler, CobraCompiler):
                raise ValueError("info is required when adding a noise-filter compiler.")
            label = getattr(compiler, "compiler_label", None)
            if label is None:
                label = f"MANUAL{len(self.compilers)}"
                compiler.compiler_label = label
            info = EscapeCompilerInfo(
                escape_id=label,
                base_label=compiler.base_vertex_label or "unknown",
                base_layer=int(getattr(compiler.base_vertex.ref, "layer", -1)),
                base_local_id=int(getattr(compiler.base_vertex.ref, "local_id", -1)),
                action=compiler.edge_type,
                candidate=compiler.candidate_index,
                candidate_name=compiler.candidate_strings[compiler.candidate_index],
            )

        self.compilers.append(compiler)
        self.compiler_info.append(info)
        self.compiler_index_by_escape_id[info.escape_id.upper()] = (
            len(self.compilers) - 1
        )

    def _initialize_seeded_graph_compilers(
        self,
        audit_graph: Any,
        remember_discrepancies: bool,
        simultaneous: bool,
        use_fallback: bool,
    ) -> None:
        self._print_seeded_graph_flag(audit_graph)
        if not self.seed_strong_candidates:
            raise ValueError("Seeded audit graph did not expose seed_strong_candidates.")

        strong_vertex = self._find_strong_only_seed_vertex(audit_graph)
        self._validate_no_election_edges_from_strong_vertex(audit_graph, strong_vertex)
        self._initialize_seeded_mentions_data(audit_graph)
        assert self.seed_frozen_mentions is not None
        assert self.seed_prebatch_strong_tallies is not None
        frozen_mentions = self.seed_frozen_mentions
        strong_tallies = self.seed_prebatch_strong_tallies

        for candidate in sorted(self.seed_strong_candidates):
            self._add_compiler(
                audit_graph=audit_graph,
                vertex=strong_vertex,
                candidate=int(candidate),
                action=EdgeAction.ELECT,
                remember_discrepancies=remember_discrepancies,
                simultaneous=simultaneous,
                use_fallback=use_fallback,
            )

        lowest_strong_candidate = min(
            self.seed_strong_candidates,
            key=lambda candidate: strong_tallies[candidate],
        )
        lowest_strong_tally = float(strong_tallies[lowest_strong_candidate])

        for weak_candidate in sorted(self.seed_weak_candidates):
            info = self._make_seed_mentions_compiler_info(
                audit_graph=audit_graph,
                vertex=strong_vertex,
                candidate=int(weak_candidate),
            )
            compiler: CobraCompilerV2Base | CobraCompiler
            if self.compiler_type == "noise":
                interpreter = self._interpreter_for_vertex(audit_graph, strong_vertex)
                critical_margin = (
                    lowest_strong_tally - float(frozen_mentions[weak_candidate])
                )
                compiler = CobraMentionsNoiseFilterCompiler(
                    interpreter,
                    int(weak_candidate),
                    strong_candidates=self.seed_strong_candidates,
                    frozen_mentions=frozen_mentions,
                    lowest_strong_candidate=int(lowest_strong_candidate),
                    lowest_strong_tally=lowest_strong_tally,
                    LAM=float(audit_graph.LAM),
                    critical_margin=critical_margin,
                    radius=2.0 * critical_margin,
                    alpha=self.alpha,
                    label=info.escape_id,
                )
            else:
                compiler = CobraCompiler(
                    strong_vertex,
                    int(weak_candidate),
                    EdgeAction.ELIMINATE,
                    audit_graph=audit_graph,
                    noise_level_guess=self.noise_guess,
                    simultaneous=simultaneous,
                    remember_discrepancies=remember_discrepancies,
                    use_fallback=use_fallback,
                    compiler_label=info.escape_id,
                    critical_margin_type=CriticalMarginType.CANDIDATE_TO_MENTIONS,
                    strong_candidates=self.seed_strong_candidates,
                    very_strong_candidates=self.seed_very_strong_candidates,
                    frozen_mentions=frozen_mentions,
                    lowest_strong_candidate=int(lowest_strong_candidate),
                    lowest_strong_tally=lowest_strong_tally,
                )
            self.add_compiler(compiler, info)

    def _print_seeded_graph_flag(self, audit_graph: Any) -> None:
        def names(candidates: frozenset[int]) -> list[str]:
            return [audit_graph.candidate_names[candidate] for candidate in sorted(candidates)]

        print("Detected seeded audit graph.")
        print(f"  very strong: {names(self.seed_very_strong_candidates)}")
        print(f"  strong: {names(self.seed_strong_candidates)}")
        print(f"  weak: {names(self.seed_weak_candidates)}")

    def _find_strong_only_seed_vertex(self, audit_graph: Any) -> Any:
        matches = [
            vertex
            for layer in audit_graph.layers
            for vertex in layer
            if (
                hasattr(vertex.key, "hopefuls")
                and frozenset(vertex.key.hopefuls) == self.seed_strong_candidates
            )
        ]
        if not matches:
            raise ValueError(
                "Seeded audit graph does not contain a vertex whose hopefuls are "
                "exactly the seed strong candidates."
            )
        return min(matches, key=lambda vertex: (vertex.ref.layer, vertex.ref.local_id))

    def _validate_no_election_edges_from_strong_vertex(
        self,
        audit_graph: Any,
        strong_vertex: Any,
    ) -> None:
        election_edges = [
            edge
            for edge in audit_graph.outgoing_edges(strong_vertex.ref)
            if (
                EdgeAction(edge.action).is_election
                and not self._is_forced_fill_edge(strong_vertex, edge)
            )
        ]
        if election_edges:
            label = audit_graph.vertex_label(strong_vertex.ref)
            raise ValueError(
                "Strong-only seed vertex has election edges, so strong candidates "
                f"are not safely below quota. vertex={label}, edges={election_edges}"
            )

    def _is_forced_fill_edge(self, vertex: Any, edge: Any) -> bool:
        return (
            EdgeAction(edge.action) == EdgeAction.FORCE_ELECT
            and hasattr(vertex.key, "hopefuls")
            and len(vertex.key.hopefuls) + int(vertex.degree) == self.m
        )

    def _initialize_seeded_mentions_data(self, audit_graph: Any) -> None:
        wt_vec = self._seeded_prebatch_wt_vec_from_graph(audit_graph)
        masked = self.seed_very_strong_candidates
        self.seed_prebatch_strong_tallies = fpv_tallies_from_matrix(
            audit_graph.ballot_matrix,
            wt_vec,
            int(audit_graph.n_candidates),
            masked_candidates=masked,
        )

        graph_mentions = getattr(audit_graph, "seed_frozen_mentions", None)
        if graph_mentions is None:
            self.seed_frozen_mentions = frozen_mentions_from_matrix(
                audit_graph.ballot_matrix,
                wt_vec,
                int(audit_graph.n_candidates),
                self.seed_strong_candidates,
                masked_candidates=masked,
            )
        else:
            self.seed_frozen_mentions = np.asarray(graph_mentions, dtype=np.float64)

    def _seeded_prebatch_wt_vec_from_graph(self, audit_graph: Any) -> NDArray[np.float64]:
        """
        Return the wt_vec after any very-strong/pre-seed elections.

        For seeded WIGM graphs, seeded_build stores this post-election,
        pre-batch-elimination vector in root_wt_vec. This is intentionally
        different from _wt_vec_from_graph, which prefers _unseeded_root_wt_vec
        for sampling from the original ballot population.
        """
        if not hasattr(audit_graph, "root_wt_vec"):
            raise ValueError(
                "Seeded audit graph must expose root_wt_vec for seeded assertions."
            )
        return np.asarray(audit_graph.root_wt_vec, dtype=np.float64)

    def _make_seed_mentions_compiler_info(
        self,
        audit_graph: Any,
        vertex: Any,
        candidate: int,
    ) -> EscapeCompilerInfo:
        base_label = audit_graph.vertex_label(vertex.ref)
        candidate_name = audit_graph.candidate_names[candidate]
        return EscapeCompilerInfo(
            escape_id=f"{base_label}-M{candidate}",
            base_label=base_label,
            base_layer=int(vertex.ref.layer),
            base_local_id=int(vertex.ref.local_id),
            action=EdgeAction.ELIMINATE,
            candidate=int(candidate),
            candidate_name=candidate_name,
        )

    def deduplicate_compilers(self) -> int:
        seen: dict[tuple[Any, ...], int] = {}
        keep_indices: list[int] = []
        removed = 0

        for idx, compiler in enumerate(self.compilers):
            signature = self._compiler_signature(compiler)
            if signature in seen:
                removed += 1
                continue
            seen[signature] = idx
            keep_indices.append(idx)

        if removed == 0:
            return 0

        self.compilers = [self.compilers[idx] for idx in keep_indices]
        self.compiler_info = [self.compiler_info[idx] for idx in keep_indices]
        self.compiler_index_by_escape_id = {
            info.escape_id.upper(): idx
            for idx, info in enumerate(self.compiler_info)
        }
        print(f"Removed {removed} duplicate compilers.")
        return removed

    def _compiler_signature(
        self,
        compiler: CobraCompilerV2Base | CobraCompiler,
    ) -> tuple[Any, ...]:
        if isinstance(compiler, CobraNoiseFilterCompiler):
            return (
                "noise-filter",
                compiler.base_vertex.ref,
                compiler.c,
                compiler.l,
                round(float(compiler.LAM), 12),
                round(float(compiler.critical_margin), 12),
            )
        if isinstance(compiler, CobraQuotaNoiseFilterCompiler):
            return (
                "quota-noise-filter",
                compiler.base_vertex.ref,
                compiler.candidate,
                compiler.margin_type,
                round(float(compiler.LAM), 12),
                round(float(compiler.critical_margin), 12),
            )
        if isinstance(compiler, CobraMentionsNoiseFilterCompiler):
            return (
                "mentions-noise-filter",
                compiler.base_vertex.ref,
                compiler.weak_candidate,
                compiler.strong_candidate,
                round(float(compiler.LAM), 12),
                round(float(compiler.critical_margin), 12),
            )
        if not isinstance(compiler, CobraCompiler):
            raise TypeError(f"Unsupported compiler type: {type(compiler).__name__}.")
        return (
            compiler.base_vertex_label,
            compiler.critical_margin_type,
            compiler.canonical_victor,
            compiler.canonical_loser,
            compiler.canonical_winner,
            compiler.canonical_non_winner,
            round(float(compiler.critical_margin), 12),
        )

    def _make_compiler_info(
        self,
        audit_graph: Any,
        vertex: Any,
        candidate: int,
        action: EdgeAction,
    ) -> EscapeCompilerInfo:
        base_label = audit_graph.vertex_label(vertex.ref)
        action_code = "E" if action == EdgeAction.ELIMINATE else "W"
        candidate_name = audit_graph.candidate_names[candidate]
        return EscapeCompilerInfo(
            escape_id=f"{base_label}-{action_code}{candidate}",
            base_label=base_label,
            base_layer=int(vertex.ref.layer),
            base_local_id=int(vertex.ref.local_id),
            action=action,
            candidate=int(candidate),
            candidate_name=candidate_name,
        )

    def run(
        self,
        ballot_matrix: NDArray[np.integer] | None = None,
        num_steps: int | None = None,
        reset_compilers: bool = True,
    ) -> bool:
        if reset_compilers:
            self.reset()

        max_i = self.sample_size
        if num_steps is not None:
            max_i = min(max_i, self.i + int(num_steps))

        while self.i < max_i and not self.is_done:
            row_c, row_b = self._sample_pair(self.i, ballot_matrix)
            self.update_pool(row_c, row_b)
            self.i += 1

            if (
                self.print_diagnostics_every > 0
                and self.i > 0
                and self.i % self.print_diagnostics_every == 0
            ):
                self.print_diagnostics()

        finished = self.is_done
        if finished or self.i >= self.sample_size:
            self.print_summary(success=finished)
        return finished

    def _sample_pair(
        self,
        i: int,
        ballot_matrix: NDArray[np.integer] | None,
    ) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
        if self.BAL is not None and self.CVR is not None:
            return self.CVR[i], self.BAL[i]

        if self.sampler is None:
            raise ValueError("Driver has no sampler and no BAL/CVR matrices.")
        if ballot_matrix is None:
            raise ValueError("ballot_matrix is required when using implicit sampling.")
        return self.sampler.sample(i, ballot_matrix)

    def reset(self) -> None:
        for compiler in self.compilers:
            compiler.reset()
        self.pool = list(range(len(self.compilers)))
        self.certified = set()
        self.certification_order = []
        self.i = 0

    def lookup_compiler(self, escape_id: str) -> CobraCompilerV2Base | CobraCompiler:
        normalized = escape_id.strip().upper()
        try:
            return self.compilers[self.compiler_index_by_escape_id[normalized]]
        except KeyError:
            known = ", ".join(info.escape_id for info in self.compiler_info[:10])
            if len(self.compiler_info) > 10:
                known += ", ..."
            raise KeyError(
                f"Unknown compiler escape_id {escape_id!r}. Known IDs include: {known}"
            ) from None

    @property
    def is_done(self) -> bool:
        if self.keep_certified_compilers:
            return len(self.certified) == len(self.compilers)
        return len(self.pool) == 0

    def update_pool(
        self,
        row_c: NDArray[np.integer],
        row_b: NDArray[np.integer],
    ) -> None:
        active = list(self.pool)
        still_active: list[int] = []

        for idx in active:
            compiler = self.compilers[idx]
            compiler.update(row_c, row_b)
            if compiler.M >= self.threshold:
                if idx not in self.certified:
                    self.certified.add(idx)
                    self.certification_order.append(idx)
                if self.keep_certified_compilers:
                    still_active.append(idx)
            else:
                still_active.append(idx)

        self.pool = still_active

    def print_diagnostics(self, limit: int = 5) -> None:
        remaining = self._remaining_indices()
        print(
            f"Audit diagnostic after {self.i} samples: "
            f"{len(remaining)} compilers remain below threshold "
            f"({len(self.certified)}/{len(self.compilers)} certified)."
        )
        for idx in self._lowest_capital_indices(remaining, limit):
            self._print_compiler_brief(idx)

    def print_summary(self, success: bool) -> None:
        if success:
            print(
                "Audit certified all compilers. "
                f"Final sample size: {self.i}."
            )
            recent = self.certification_order[-3:]
            if recent:
                print("Last compilers to pass threshold:")
                for idx in recent:
                    self._print_compiler_brief(idx)
            return

        print(
            "Audit did not certify all compilers before sample exhaustion. "
            f"Samples used: {self.i}/{self.sample_size}."
        )
        remaining = self._remaining_indices()
        for idx in self._lowest_capital_indices(remaining, 3):
            self._print_compiler_brief(idx)
            compiler = self.compilers[idx]
            if isinstance(compiler, CobraCompiler):
                compiler.print_info()
            else:
                print(compiler.get_info())

    def _remaining_indices(self) -> list[int]:
        if self.keep_certified_compilers:
            return [
                idx
                for idx in range(len(self.compilers))
                if idx not in self.certified
            ]
        return list(self.pool)

    def _lowest_capital_indices(
        self,
        indices: list[int],
        limit: int,
    ) -> list[int]:
        return sorted(indices, key=lambda idx: self.compilers[idx].M)[:limit]

    def _print_compiler_brief(self, idx: int) -> None:
        compiler = self.compilers[idx]
        info = self.compiler_info[idx]
        if not isinstance(compiler, CobraCompiler):
            print(
                f"  {info.escape_id}: M={compiler.M:.6g}, "
                f"escape={info.action.name} {info.candidate}:{info.candidate_name}, "
                f"margin={compiler.critical_margin}, kind={compiler.compiler_kind}"
            )
            return
        print(
            f"  {info.escape_id}: M={compiler.M:.6g}, "
            f"escape={info.action.name} {info.candidate}:{info.candidate_name}, "
            f"margin={compiler.critical_margin_type.value} "
            f"{self._critical_margin_label(compiler)}"
        )

    def _critical_margin_label(self, compiler: CobraCompiler) -> str:
        if compiler.canonical_victor is not None and compiler.canonical_loser is not None:
            return (
                f"{compiler._candidate_label(compiler.canonical_victor)} > "
                f"{compiler._candidate_label(compiler.canonical_loser)}"
            )
        if compiler.canonical_winner is not None:
            return f"{compiler._candidate_label(compiler.canonical_winner)} >= quota"
        if compiler.canonical_non_winner is not None:
            return (
                f"{compiler._candidate_label(compiler.canonical_non_winner)} "
                "< quota"
            )
        return "(unknown)"


@dataclass(frozen=True, slots=True)
class V2CompilerSpec:
    info: EscapeCompilerInfo
    kind: str
    vertex_ref: Any
    c: int | None = None
    l: int | None = None
    candidate: int | None = None
    margin_type: CriticalMarginType | None = None
    is_noise_filter: bool = False
    strong_candidates: frozenset[int] | None = None
    frozen_mentions: NDArray[np.float64] | None = None
    lowest_strong_candidate: int | None = None
    lowest_strong_tally: float | None = None


class GlobalAuditDriverV2:
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"lambda", "lambda_"}:
            object.__setattr__(
                self,
                "lambda_value",
                None if value is None else float(value),
            )
            return
        if name == "r" and value is not None:
            object.__setattr__(self, name, float(value))
            return
        object.__setattr__(self, name, value)

    @property
    def lambda_(self) -> float | None:
        return self.lambda_value

    @lambda_.setter
    def lambda_(self, value: float | None) -> None:
        self.lambda_value = None if value is None else float(value)

    def __init__(
        self,
        audit_graph: Any,
        noise_level: float,
        *,
        r: int | float | None = None,
        sample_size: int | None = None,
        fractional_sample_size: float | None = None,
        seed: int | None = None,
        print_diagnostics_every: int = 30,
        alpha: float = 0.05,
        keep_certified_compilers: bool = False,
        cache_noised_rows: bool = False,
        profile: bool = False,
        lambda_value: float | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        simultaneous: bool = False,
        verbose: bool = True,
        BAL: NDArray[np.integer] | None = None,
        CVR: NDArray[np.integer] | None = None,
    ) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        self.verbose = bool(verbose)
        self.seeded_graph = bool(getattr(audit_graph, "used_seeded_build", False))
        if self.seeded_graph and self._is_black_box_seeded_graph(audit_graph):
            raise NotImplementedError(
                "GlobalAuditDriverV2 supports seeded batch-elimination graphs, "
                "but not black-box seeded graphs yet."
            )

        self.audit_graph = audit_graph
        self.noise_level = float(noise_level)
        self.r = float(2.0 * audit_graph.LAM if r is None else r)
        self.print_diagnostics_every = int(print_diagnostics_every)
        self.alpha = float(alpha)
        self.threshold = 1.0 / self.alpha
        self.keep_certified_compilers = bool(keep_certified_compilers)
        self.profile = bool(profile)
        self.lambda_value = None if lambda_value is None else float(lambda_value)
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.simultaneous = bool(simultaneous)
        self.seed = seed
        self.i = 0
        self.m = int(audit_graph.m)
        self.seed_very_strong_candidates = frozenset(
            getattr(audit_graph, "seed_very_strong_candidates", frozenset())
        )
        self.seed_strong_candidates = frozenset(
            getattr(audit_graph, "seed_strong_candidates", frozenset())
        )
        self.seed_weak_candidates = frozenset(
            getattr(audit_graph, "seed_weak_candidates", frozenset())
        )
        self.seed_frozen_mentions: NDArray[np.float64] | None = None
        self.seed_prebatch_strong_tallies: NDArray[np.float64] | None = None
        self.canonical_winner_set = self._canonical_winner_set_from_terminal(
            audit_graph
        )
        self.sampler: ImplicitSampler | None = None
        self.BAL: NDArray[np.integer] | None = None
        self.CVR: NDArray[np.integer] | None = None
        self.profile_figure = None
        self.profile_axes = None
        self.discrepancies: list[tuple[int, int, str, tuple[int, int], float]] = []

        self._log("Initializing GlobalAuditDriverV2.")
        self._log(
            f"  graph: candidates={audit_graph.n_candidates}, seats={self.m}, "
            f"LAM={audit_graph.LAM}, seeded={self.seeded_graph}"
        )
        self.sample_size = self._initialize_sample_source(
            audit_graph=audit_graph,
            sample_size=sample_size,
            fractional_sample_size=fractional_sample_size,
            seed=seed,
            cache_noised_rows=cache_noised_rows,
            BAL=BAL,
            CVR=CVR,
        )

        self.interpreters: dict[Any, VertexInterpreter] = {}
        self.compilers: list[CobraCompilerV2Base] = []
        self.compiler_info: list[EscapeCompilerInfo] = []
        self.compiler_specs: list[V2CompilerSpec] = []
        self.compiler_index_by_escape_id: dict[str, int] = {}
        self._initialize_compilers(audit_graph)
        if self.seeded_graph:
            self._initialize_seeded_graph_compilers(audit_graph)
        self.deduplicate_compilers()
        self._log(
            f"Initialized {len(self.compilers)} V2 compilers across "
            f"{len(self.interpreters)} vertex interpreters."
        )

        self.pool: list[int] = list(range(len(self.compilers)))
        self.certified: set[int] = set()
        self.certification_order: list[int] = []

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _is_black_box_seeded_graph(self, audit_graph: Any) -> bool:
        return (
            audit_graph.__class__.__name__ == "BlackBoxWIGMGraphConstructor"
            or hasattr(audit_graph, "vertex_post_seed_tallies")
            or hasattr(audit_graph, "seed_weight_scenarios")
        )

    def _wt_vec_from_graph(self, audit_graph: Any) -> NDArray[np.float64]:
        if hasattr(audit_graph, "_unseeded_root_wt_vec"):
            return np.asarray(audit_graph._unseeded_root_wt_vec, dtype=np.float64)
        if hasattr(audit_graph, "root_wt_vec"):
            return np.asarray(audit_graph.root_wt_vec, dtype=np.float64)
        if hasattr(audit_graph, "profile"):
            return ImplicitSampler._wt_vec_from_profile(audit_graph.profile)
        raise ValueError("Could not recover wt_vec from audit_graph.")

    def _initialize_sample_source(
        self,
        audit_graph: Any,
        sample_size: int | None,
        fractional_sample_size: float | None,
        seed: int | None,
        cache_noised_rows: bool,
        BAL: NDArray[np.integer] | None,
        CVR: NDArray[np.integer] | None,
    ) -> int:
        if (BAL is None) != (CVR is None):
            raise ValueError("BAL and CVR must be provided together.")
        if BAL is not None and CVR is not None:
            if sample_size is not None or fractional_sample_size is not None:
                raise ValueError(
                    "sample_size and fractional_sample_size should not be provided "
                    "when BAL/CVR sample matrices are provided."
                )
            bal = np.asarray(BAL)
            cvr = np.asarray(CVR)
            if bal.ndim != 2 or cvr.ndim != 2:
                raise ValueError("BAL and CVR must be two-dimensional arrays.")
            if bal.shape != cvr.shape:
                raise ValueError("BAL and CVR must have the same shape.")
            self.BAL = bal
            self.CVR = cvr
            self._log(
                f"  sample source: provided BAL/CVR matrices with {cvr.shape[0]} rows"
            )
            return int(cvr.shape[0])

        wt_vec = self._wt_vec_from_graph(audit_graph)
        self.sampler = ImplicitSampler(
            noise_level=self.noise_level,
            n_cands=int(audit_graph.n_candidates),
            wt_vec=wt_vec,
            sample_size=sample_size,
            fractional_sample_size=fractional_sample_size,
            with_replacement=True,
            seed=seed,
            cache_rows=cache_noised_rows,
        )
        self._log(
            f"  sample source: implicit sampler with {self.sampler.sample_size} rows "
            f"and noise_level={self.noise_level}"
        )
        return int(self.sampler.sample_size)

    def _initialize_compilers(self, audit_graph: Any) -> None:
        self._log("Initializing escape-edge compilers.")
        for layer in audit_graph.layers:
            before = len(self.compilers)
            visited = 0
            for vertex in layer:
                visited += 1
                if vertex.status == ElectionStatus.TERMINAL:
                    continue
                if not hasattr(vertex.key, "hopefuls"):
                    raise ValueError("Audit graph vertex key does not expose hopefuls.")
                if len(vertex.key.hopefuls) + int(vertex.degree) == self.m:
                    continue

                outgoing = audit_graph.outgoing_edges(vertex.ref)
                if not outgoing and getattr(vertex, "path_multiplicity", 1) == 0:
                    continue
                if vertex.tallies is None:
                    raise ValueError(
                        "Cannot initialize V2 compilers from vertex "
                        f"{audit_graph.vertex_label(vertex.ref)}: tallies missing."
                    )

                interpreter = self._interpreter_for_vertex(audit_graph, vertex)
                forced_winner = self._forced_above_quota_candidate(vertex)
                if forced_winner is not None and any(
                    not self._has_outgoing_elimination(outgoing, candidate)
                    for candidate in vertex.key.hopefuls
                ):
                    self._add_escape_compilers(
                        audit_graph=audit_graph,
                        vertex=vertex,
                        interpreter=interpreter,
                        candidate=forced_winner,
                        action=EdgeAction.ELIMINATE,
                    )
                for candidate in sorted(vertex.key.hopefuls):
                    if (
                        forced_winner is None
                        and not self._has_outgoing_elimination(outgoing, candidate)
                    ):
                        self._add_escape_compilers(
                            audit_graph=audit_graph,
                            vertex=vertex,
                            interpreter=interpreter,
                            candidate=int(candidate),
                            action=EdgeAction.ELIMINATE,
                        )
                    if not self._has_outgoing_election(outgoing, candidate):
                        if self._skip_final_reported_winner_seating_escape(
                            vertex,
                            int(candidate),
                        ):
                            continue
                        if (
                            forced_winner is not None
                            and self._forced_election_escape_is_redundant(
                                vertex,
                                int(candidate),
                            )
                        ):
                            continue
                        self._add_escape_compilers(
                            audit_graph=audit_graph,
                            vertex=vertex,
                            interpreter=interpreter,
                            candidate=int(candidate),
                            action=EdgeAction.ELECT,
                        )
            added = len(self.compilers) - before
            if added:
                self._log(
                    f"  layer {getattr(layer[0].ref, 'layer', '?') if layer else '?'}: "
                    f"visited {visited} vertices, added {added} compilers"
                )

    def _forced_above_quota_candidate(self, vertex: Any) -> int | None:
        tallies = np.asarray(vertex.tallies, dtype=np.float64)
        hopefuls = np.asarray(sorted(vertex.key.hopefuls), dtype=int)
        forced = hopefuls[
            tallies[hopefuls] >= self.audit_graph.quota + self.audit_graph.LAM
        ]
        if len(forced) == 0:
            return None
        return int(forced[np.argmax(tallies[forced])])

    def _forced_election_escape_is_redundant(
        self,
        vertex: Any,
        candidate: int,
    ) -> bool:
        tallies = np.asarray(vertex.tallies, dtype=np.float64)
        c_tally = float(tallies[candidate])
        if not self.simultaneous:
            challengers = np.where(tallies > c_tally + self.audit_graph.LAM)[0]
            challengers = np.asarray(
                [idx for idx in challengers if idx != candidate],
                dtype=int,
            )
            if len(challengers) > 0:
                return False
        return bool(c_tally + self.audit_graph.LAM < self.audit_graph.quota)

    def _interpreter_for_vertex(
        self,
        audit_graph: Any,
        vertex: Any,
    ) -> VertexInterpreter:
        existing = self.interpreters.get(vertex.ref)
        if existing is not None:
            return existing
        interpreter = VertexInterpreter(audit_graph, vertex)
        interpreter.compilers = []
        self.interpreters[vertex.ref] = interpreter
        return interpreter

    def _canonical_winner_set_from_terminal(self, audit_graph: Any) -> frozenset[int]:
        terminals = [
            vertex
            for layer in audit_graph.layers
            for vertex in layer
            if vertex.status == ElectionStatus.TERMINAL
        ]
        if not terminals:
            return frozenset()

        terminal = min(terminals, key=lambda v: (v.ref.layer, v.ref.local_id))
        if hasattr(audit_graph, "_winner_set_for_vertex"):
            return frozenset(int(c) for c in audit_graph._winner_set_for_vertex(terminal))

        winners: set[int] = set()
        seated_at = getattr(terminal.key, "seated_at", ())
        for edge_ref in seated_at:
            if edge_ref is not None:
                winners.add(int(audit_graph.edge(edge_ref).candidate))
        return frozenset(winners)

    def _skip_final_reported_winner_seating_escape(
        self,
        vertex: Any,
        candidate: int,
    ) -> bool:
        return (
            int(vertex.degree) == self.m - 1
            and int(candidate) in self.canonical_winner_set
        )

    def _has_outgoing_elimination(self, outgoing: list[Any], candidate: int) -> bool:
        return any(
            edge.candidate == candidate and edge.action == EdgeAction.ELIMINATE
            for edge in outgoing
        )

    def _has_outgoing_election(self, outgoing: list[Any], candidate: int) -> bool:
        return any(
            edge.candidate == candidate and EdgeAction(edge.action).is_election
            for edge in outgoing
        )

    def _add_escape_compilers(
        self,
        audit_graph: Any,
        vertex: Any,
        interpreter: VertexInterpreter,
        candidate: int,
        action: EdgeAction,
    ) -> None:
        margin = self._critical_margin_for_escape(vertex, candidate, action)
        if margin is None:
            return
        info = self._make_compiler_info(audit_graph, vertex, candidate, action)

        if margin["type"] == CriticalMarginType.CANDIDATE_TO_CANDIDATE:
            c = int(margin["c"])
            l = int(margin["l"])
            main = CobraCompilerV2(
                interpreter,
                c,
                l,
                critical_margin=float(margin["margin"]),
                radius=self.r,
                alpha=self.alpha,
                lambda_value=self.lambda_value,
                optimizer_kwargs=self.optimizer_kwargs,
                label=info.escape_id,
            )
            noise = CobraNoiseFilterCompiler(
                interpreter,
                c,
                l,
                radius=self.r,
                alpha=self.alpha,
                lambda_value=self.lambda_value,
                label=f"{info.escape_id}-N",
            )
            spec = V2CompilerSpec(
                info=info,
                kind="candidate-to-candidate",
                vertex_ref=vertex.ref,
                c=c,
                l=l,
            )
            noise_spec = V2CompilerSpec(
                info=self._noise_info(info),
                kind="noise-filter",
                vertex_ref=vertex.ref,
                c=c,
                l=l,
                is_noise_filter=True,
            )
        else:
            margin_type = CriticalMarginType(margin["type"])
            quota_candidate = int(margin["candidate"])
            main = CobraQuotaCompilerV2(
                interpreter,
                quota_candidate,
                margin_type=margin_type,
                critical_margin=float(margin["margin"]),
                radius=self.r,
                alpha=self.alpha,
                lambda_value=self.lambda_value,
                optimizer_kwargs=self.optimizer_kwargs,
                label=info.escape_id,
            )
            noise = CobraQuotaNoiseFilterCompiler(
                interpreter,
                quota_candidate,
                margin_type=margin_type,
                radius=self.r,
                alpha=self.alpha,
                lambda_value=self.lambda_value,
                label=f"{info.escape_id}-N",
            )
            spec = V2CompilerSpec(
                info=info,
                kind=margin_type.value,
                vertex_ref=vertex.ref,
                candidate=quota_candidate,
                margin_type=margin_type,
            )
            noise_spec = V2CompilerSpec(
                info=self._noise_info(info),
                kind="quota-noise-filter",
                vertex_ref=vertex.ref,
                candidate=quota_candidate,
                margin_type=margin_type,
                is_noise_filter=True,
            )

        self._append_compiler(main, spec.info, spec)
        self._append_compiler(noise, noise_spec.info, noise_spec)
        interpreter.compilers.extend((main, noise))

    def _initialize_seeded_graph_compilers(self, audit_graph: Any) -> None:
        self._print_seeded_graph_flag(audit_graph)
        if not self.seed_strong_candidates:
            raise ValueError("Seeded audit graph did not expose seed_strong_candidates.")

        strong_vertex = self._find_strong_only_seed_vertex(audit_graph)
        self._log(
            "Initializing seeded batch-elimination assertions from "
            f"{audit_graph.vertex_label(strong_vertex.ref)}."
        )
        self._validate_no_election_edges_from_strong_vertex(audit_graph, strong_vertex)
        self._initialize_seeded_mentions_data(audit_graph)
        if self.seed_frozen_mentions is None:
            raise ValueError("Seeded mentions were not initialized.")
        if self.seed_prebatch_strong_tallies is None:
            raise ValueError("Seeded strong tallies were not initialized.")

        interpreter = self._interpreter_for_vertex(audit_graph, strong_vertex)

        for candidate in sorted(self.seed_strong_candidates):
            self._add_escape_compilers(
                audit_graph=audit_graph,
                vertex=strong_vertex,
                interpreter=interpreter,
                candidate=int(candidate),
                action=EdgeAction.ELECT,
            )

        lowest_strong_candidate = min(
            self.seed_strong_candidates,
            key=lambda candidate: self.seed_prebatch_strong_tallies[candidate],
        )
        lowest_strong_tally = float(
            self.seed_prebatch_strong_tallies[lowest_strong_candidate]
        )

        for weak_candidate in sorted(self.seed_weak_candidates):
            info = self._make_seed_mentions_compiler_info(
                audit_graph=audit_graph,
                vertex=strong_vertex,
                candidate=int(weak_candidate),
            )
            main = CobraMentionsCompilerV2(
                interpreter,
                int(weak_candidate),
                strong_candidates=self.seed_strong_candidates,
                frozen_mentions=self.seed_frozen_mentions,
                lowest_strong_candidate=int(lowest_strong_candidate),
                lowest_strong_tally=lowest_strong_tally,
                radius=self.r,
                alpha=self.alpha,
                lambda_value=self.lambda_value,
                optimizer_kwargs=self.optimizer_kwargs,
                label=info.escape_id,
            )
            noise_info = self._noise_info(info)
            noise = CobraMentionsNoiseFilterCompiler(
                interpreter,
                int(weak_candidate),
                strong_candidates=self.seed_strong_candidates,
                frozen_mentions=self.seed_frozen_mentions,
                lowest_strong_candidate=int(lowest_strong_candidate),
                lowest_strong_tally=lowest_strong_tally,
                radius=self.r,
                alpha=self.alpha,
                lambda_value=self.lambda_value,
                label=noise_info.escape_id,
            )
            spec = V2CompilerSpec(
                info=info,
                kind="candidate-to-mentions",
                vertex_ref=strong_vertex.ref,
                candidate=int(weak_candidate),
                margin_type=CriticalMarginType.CANDIDATE_TO_MENTIONS,
                strong_candidates=self.seed_strong_candidates,
                frozen_mentions=self.seed_frozen_mentions,
                lowest_strong_candidate=int(lowest_strong_candidate),
                lowest_strong_tally=lowest_strong_tally,
            )
            noise_spec = V2CompilerSpec(
                info=noise_info,
                kind="mentions-noise-filter",
                vertex_ref=strong_vertex.ref,
                candidate=int(weak_candidate),
                margin_type=CriticalMarginType.CANDIDATE_TO_MENTIONS,
                is_noise_filter=True,
                strong_candidates=self.seed_strong_candidates,
                frozen_mentions=self.seed_frozen_mentions,
                lowest_strong_candidate=int(lowest_strong_candidate),
                lowest_strong_tally=lowest_strong_tally,
            )
            self._append_compiler(main, info, spec)
            self._append_compiler(noise, noise_info, noise_spec)
            interpreter.compilers.extend((main, noise))
        self._log(
            f"  seeded assertions: added {2 * len(self.seed_weak_candidates)} "
            "candidate-to-mentions compilers"
        )

    def _print_seeded_graph_flag(self, audit_graph: Any) -> None:
        def names(candidates: frozenset[int]) -> list[str]:
            return [audit_graph.candidate_names[candidate] for candidate in sorted(candidates)]

        self._log("Detected seeded audit graph.")
        self._log(f"  very strong: {names(self.seed_very_strong_candidates)}")
        self._log(f"  strong: {names(self.seed_strong_candidates)}")
        self._log(f"  weak: {names(self.seed_weak_candidates)}")

    def _find_strong_only_seed_vertex(self, audit_graph: Any) -> Any:
        matches = [
            vertex
            for layer in audit_graph.layers
            for vertex in layer
            if (
                hasattr(vertex.key, "hopefuls")
                and frozenset(vertex.key.hopefuls) == self.seed_strong_candidates
            )
        ]
        if not matches:
            raise ValueError(
                "Seeded audit graph does not contain a vertex whose hopefuls are "
                "exactly the seed strong candidates."
            )
        return min(matches, key=lambda vertex: (vertex.ref.layer, vertex.ref.local_id))

    def _validate_no_election_edges_from_strong_vertex(
        self,
        audit_graph: Any,
        strong_vertex: Any,
    ) -> None:
        election_edges = [
            edge
            for edge in audit_graph.outgoing_edges(strong_vertex.ref)
            if (
                EdgeAction(edge.action).is_election
                and not self._is_forced_fill_edge(strong_vertex, edge)
            )
        ]
        if election_edges:
            label = audit_graph.vertex_label(strong_vertex.ref)
            raise ValueError(
                "Strong-only seed vertex has election edges, so strong candidates "
                f"are not safely below quota. vertex={label}, edges={election_edges}"
            )

    def _is_forced_fill_edge(self, vertex: Any, edge: Any) -> bool:
        return (
            EdgeAction(edge.action) == EdgeAction.FORCE_ELECT
            and hasattr(vertex.key, "hopefuls")
            and len(vertex.key.hopefuls) + int(vertex.degree) == self.m
        )

    def _initialize_seeded_mentions_data(self, audit_graph: Any) -> None:
        wt_vec = self._seeded_prebatch_wt_vec_from_graph(audit_graph)
        masked = self.seed_very_strong_candidates
        self.seed_prebatch_strong_tallies = fpv_tallies_from_matrix(
            audit_graph.ballot_matrix,
            wt_vec,
            int(audit_graph.n_candidates),
            masked_candidates=masked,
        )

        graph_mentions = getattr(audit_graph, "seed_frozen_mentions", None)
        if graph_mentions is None:
            self.seed_frozen_mentions = frozen_mentions_from_matrix(
                audit_graph.ballot_matrix,
                wt_vec,
                int(audit_graph.n_candidates),
                self.seed_strong_candidates,
                masked_candidates=masked,
            )
        else:
            self.seed_frozen_mentions = np.asarray(graph_mentions, dtype=np.float64)

    def _seeded_prebatch_wt_vec_from_graph(self, audit_graph: Any) -> NDArray[np.float64]:
        if not hasattr(audit_graph, "root_wt_vec"):
            raise ValueError(
                "Seeded audit graph must expose root_wt_vec for seeded assertions."
            )
        return np.asarray(audit_graph.root_wt_vec, dtype=np.float64)

    def _make_seed_mentions_compiler_info(
        self,
        audit_graph: Any,
        vertex: Any,
        candidate: int,
    ) -> EscapeCompilerInfo:
        base_label = audit_graph.vertex_label(vertex.ref)
        candidate_name = audit_graph.candidate_names[candidate]
        return EscapeCompilerInfo(
            escape_id=f"{base_label}-M{candidate}",
            base_label=base_label,
            base_layer=int(vertex.ref.layer),
            base_local_id=int(vertex.ref.local_id),
            action=EdgeAction.ELIMINATE,
            candidate=int(candidate),
            candidate_name=candidate_name,
        )

    def deduplicate_compilers(self) -> int:
        seen: dict[tuple[Any, ...], int] = {}
        keep_indices: list[int] = []
        removed = 0

        for idx, spec in enumerate(self.compiler_specs):
            signature = self._compiler_signature(spec, self.compilers[idx])
            if signature in seen:
                removed += 1
                continue
            seen[signature] = idx
            keep_indices.append(idx)

        if removed == 0:
            return 0

        self.compilers = [self.compilers[idx] for idx in keep_indices]
        self.compiler_info = [self.compiler_info[idx] for idx in keep_indices]
        self.compiler_specs = [self.compiler_specs[idx] for idx in keep_indices]
        self.compiler_index_by_escape_id = {
            info.escape_id.upper(): idx
            for idx, info in enumerate(self.compiler_info)
        }
        for interpreter in self.interpreters.values():
            interpreter.compilers = [
                compiler
                for compiler in getattr(interpreter, "compilers", [])
                if compiler in self.compilers
            ]
        self._log(f"Removed {removed} duplicate V2 compilers.")
        return removed

    def _compiler_signature(
        self,
        spec: V2CompilerSpec,
        compiler: CobraCompilerV2Base,
    ) -> tuple[Any, ...]:
        return (
            spec.kind,
            spec.vertex_ref,
            spec.c,
            spec.l,
            spec.candidate,
            spec.margin_type,
            spec.is_noise_filter,
            spec.lowest_strong_candidate,
            round(float(compiler.critical_margin), 12),
        )

    def _critical_margin_for_escape(
        self,
        vertex: Any,
        candidate: int,
        action: EdgeAction,
    ) -> dict[str, Any] | None:
        tallies = np.asarray(vertex.tallies, dtype=np.float64)
        if action == EdgeAction.ELIMINATE:
            forced = np.where(tallies >= self.audit_graph.quota + self.audit_graph.LAM)[0]
            if len(forced) > 0:
                winner = int(forced[np.argmax(tallies[forced])])
                margin = float(tallies[winner] - self.audit_graph.quota)
                return {
                    "type": CriticalMarginType.CANDIDATE_ABOVE_QUOTA,
                    "candidate": winner,
                    "margin": margin,
                }

            hopefuls = np.asarray(sorted(vertex.key.hopefuls), dtype=int)
            lowest = int(hopefuls[np.argmin(tallies[hopefuls])])
            if lowest == candidate:
                return None
            return {
                "type": CriticalMarginType.CANDIDATE_TO_CANDIDATE,
                "c": int(candidate),
                "l": lowest,
                "margin": float(tallies[candidate] - tallies[lowest]),
            }

        c_tally = float(tallies[candidate])
        if not self.simultaneous:
            challengers = np.where(tallies > c_tally + self.audit_graph.LAM)[0]
            challengers = np.asarray(
                [idx for idx in challengers if idx != candidate],
                dtype=int,
            )
            if len(challengers) > 0:
                winner = int(challengers[np.argmax(tallies[challengers])])
                return {
                    "type": CriticalMarginType.CANDIDATE_TO_CANDIDATE,
                    "c": winner,
                    "l": int(candidate),
                    "margin": float(tallies[winner] - c_tally),
                }

        if c_tally + self.audit_graph.LAM < self.audit_graph.quota:
            return {
                "type": CriticalMarginType.CANDIDATE_BELOW_QUOTA,
                "candidate": int(candidate),
                "margin": float(self.audit_graph.quota - c_tally),
            }
        return None

    def _append_compiler(
        self,
        compiler: CobraCompilerV2Base,
        info: EscapeCompilerInfo,
        spec: V2CompilerSpec,
    ) -> None:
        self.compilers.append(compiler)
        self.compiler_info.append(info)
        self.compiler_specs.append(spec)
        self.compiler_index_by_escape_id[info.escape_id.upper()] = (
            len(self.compilers) - 1
        )

    def _noise_info(self, info: EscapeCompilerInfo) -> EscapeCompilerInfo:
        return EscapeCompilerInfo(
            escape_id=f"{info.escape_id}-N",
            base_label=info.base_label,
            base_layer=info.base_layer,
            base_local_id=info.base_local_id,
            action=info.action,
            candidate=info.candidate,
            candidate_name=f"{info.candidate_name} noise",
        )

    def _make_compiler_info(
        self,
        audit_graph: Any,
        vertex: Any,
        candidate: int,
        action: EdgeAction,
    ) -> EscapeCompilerInfo:
        base_label = audit_graph.vertex_label(vertex.ref)
        action_code = "E" if action == EdgeAction.ELIMINATE else "W"
        candidate_name = audit_graph.candidate_names[candidate]
        return EscapeCompilerInfo(
            escape_id=f"{base_label}-{action_code}{candidate}",
            base_label=base_label,
            base_layer=int(vertex.ref.layer),
            base_local_id=int(vertex.ref.local_id),
            action=action,
            candidate=int(candidate),
            candidate_name=candidate_name,
        )

    def run(
        self,
        ballot_matrix: NDArray[np.integer] | None = None,
        num_steps: int | None = None,
        reset_compilers: bool = True,
    ) -> bool:
        self._synchronize_compiler_parameters()
        if reset_compilers:
            self._log("Resetting V2 audit state before run.")
            self.reset()

        max_i = self.sample_size
        if num_steps is not None:
            max_i = min(max_i, self.i + int(num_steps))
        source = "provided BAL/CVR matrices" if self.BAL is not None else "implicit sampler"
        self._log(
            "Starting V2 audit run: "
            f"active_compilers={len(self.pool)}, threshold={self.threshold:g}, "
            f"samples={self.i}->{max_i}, source={source}"
        )

        while self.i < max_i and not self.is_done:
            row_c, row_b = self._sample_pair(self.i, ballot_matrix)
            self.update_pool(row_c, row_b)
            self.i += 1
            if (
                self.print_diagnostics_every > 0
                and self.i > 0
                and self.i % self.print_diagnostics_every == 0
            ):
                self.print_diagnostics()

        finished = self.is_done
        self._log(
            "Finished V2 audit run: "
            f"certified={len(self.certified)}/{len(self.compilers)}, "
            f"remaining={len(self._remaining_indices())}, samples_used={self.i}"
        )
        if finished or self.i >= self.sample_size:
            self.print_summary(success=finished)
        return finished

    def _synchronize_compiler_parameters(self) -> None:
        if not hasattr(self, "compilers"):
            return

        target_radius = float(self.r)
        target_lambda = self.lambda_value
        changed_radius = 0
        changed_lambda = 0

        for compiler, spec in zip(self.compilers, self.compiler_specs):
            if float(compiler.radius) != target_radius:
                compiler.radius = target_radius
                if spec.is_noise_filter:
                    compiler.critical_margin = float(compiler._recorded_margin())
                compiler.v = float(compiler._diluted_margin())
                compiler.optimizer_cache.clear()
                compiler.optimizer_results.clear()
                changed_radius += 1

            if target_lambda is not None and float(compiler.lambda_value) != float(target_lambda):
                compiler.lambda_value = float(target_lambda)
                compiler.used_fallback = False
                changed_lambda += 1

        if changed_radius:
            self._log(
                f"Updated radius r={target_radius:g} on {changed_radius} compilers "
                "and cleared affected optimizer caches."
            )
        if changed_lambda:
            self._log(
                f"Updated lambda={float(target_lambda):g} on {changed_lambda} compilers."
            )

    def _sample_pair(
        self,
        i: int,
        ballot_matrix: NDArray[np.integer] | None,
    ) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
        if self.BAL is not None and self.CVR is not None:
            return self.CVR[i], self.BAL[i]
        if self.sampler is None:
            raise ValueError("Driver has no sampler and no BAL/CVR matrices.")
        matrix = self.audit_graph.ballot_matrix if ballot_matrix is None else ballot_matrix
        return self.sampler.sample(i, matrix)

    def reset(self) -> None:
        for compiler in self.compilers:
            compiler.reset()
        self.pool = list(range(len(self.compilers)))
        self.certified = set()
        self.certification_order = []
        self.discrepancies = []
        self.i = 0

    def lookup_compiler(self, escape_id: str) -> CobraCompilerV2Base:
        normalized = escape_id.strip().upper()
        try:
            return self.compilers[self.compiler_index_by_escape_id[normalized]]
        except KeyError:
            known = ", ".join(info.escape_id for info in self.compiler_info[:10])
            if len(self.compiler_info) > 10:
                known += ", ..."
            raise KeyError(
                f"Unknown compiler escape_id {escape_id!r}. Known IDs include: {known}"
            ) from None

    @property
    def is_done(self) -> bool:
        if self.keep_certified_compilers:
            return len(self.certified) == len(self.compilers)
        return len(self.pool) == 0

    def update_pool(
        self,
        row_c: NDArray[np.integer],
        row_b: NDArray[np.integer],
    ) -> None:
        active = list(self.pool)
        still_active: list[int] = []
        for idx in active:
            compiler = self.compilers[idx]
            before = len(compiler.discrepancies)
            compiler.update(row_c, row_b)
            for update_num, theta_key, w in compiler.discrepancies[before:]:
                self.discrepancies.append(
                    (self.i + 1, idx, self.compiler_info[idx].escape_id, theta_key, w)
                )
            if compiler.M >= self.threshold:
                if idx not in self.certified:
                    self.certified.add(idx)
                    self.certification_order.append(idx)
                if self.keep_certified_compilers:
                    still_active.append(idx)
            else:
                still_active.append(idx)
        self.pool = still_active

    def plot_profile(
        self,
        num_compilers: int = 3,
        *,
        ballot_matrix: NDArray[np.integer] | None = None,
    ):
        limit = int(num_compilers)
        if limit <= 0:
            raise ValueError("num_compilers must be positive.")

        selected = self.certification_order[-limit:]
        if not selected:
            selected = self._lowest_capital_indices(self._remaining_indices(), limit)
        if not selected:
            return None

        profiled = [self._rebuild_profiled_compiler(idx) for idx in selected]
        active = list(range(len(profiled)))
        step = 0
        while step < self.sample_size and active:
            row_c, row_b = self._sample_pair(step, ballot_matrix)
            still = []
            for local_idx in active:
                compiler = profiled[local_idx]
                compiler.update(row_c, row_b)
                if compiler.M < compiler.threshold:
                    still.append(local_idx)
            active = still
            step += 1

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.profile_figure = None
            self.profile_axes = None
            return profiled

        fig, axes = plt.subplots(
            len(profiled),
            1,
            figsize=(12, 4 * len(profiled)),
            constrained_layout=True,
        )
        if len(profiled) == 1:
            axes = [axes]
        for compiler, ax in zip(profiled, axes):
            plot_profiled_compiler(compiler, ax=ax)
        self.profile_figure = fig
        self.profile_axes = axes
        return profiled

    def profile_slowest_compilers(
        self,
        *,
        ballot_matrix: NDArray[np.integer] | None = None,
        limit: int = 3,
    ):
        return self.plot_profile(
            num_compilers=limit,
            ballot_matrix=ballot_matrix,
        )

    def _rebuild_profiled_compiler(self, idx: int) -> CobraCompilerV2Base:
        spec = self.compiler_specs[idx]
        interpreter = self._interpreter_for_vertex(
            self.audit_graph,
            self.audit_graph.vertex(spec.vertex_ref),
        )
        info = self.compiler_info[idx]
        common = dict(
            radius=self.r,
            alpha=self.alpha,
            lambda_value=self.lambda_value,
            profile=True,
            optimizer_kwargs=self.optimizer_kwargs,
            label=self._profile_compiler_label(info),
        )
        if spec.kind == "candidate-to-candidate":
            return CobraCompilerV2(interpreter, spec.c, spec.l, **common)
        if spec.kind == "noise-filter":
            return CobraNoiseFilterCompiler(interpreter, spec.c, spec.l, **common)
        if spec.kind == "candidate-to-mentions":
            return CobraMentionsCompilerV2(
                interpreter,
                spec.candidate,
                strong_candidates=spec.strong_candidates,
                frozen_mentions=spec.frozen_mentions,
                lowest_strong_candidate=spec.lowest_strong_candidate,
                lowest_strong_tally=spec.lowest_strong_tally,
                **common,
            )
        if spec.kind == "mentions-noise-filter":
            return CobraMentionsNoiseFilterCompiler(
                interpreter,
                spec.candidate,
                strong_candidates=spec.strong_candidates,
                frozen_mentions=spec.frozen_mentions,
                lowest_strong_candidate=spec.lowest_strong_candidate,
                lowest_strong_tally=spec.lowest_strong_tally,
                **common,
            )
        if spec.kind == "quota-noise-filter":
            return CobraQuotaNoiseFilterCompiler(
                interpreter,
                spec.candidate,
                margin_type=spec.margin_type,
                **common,
            )
        return CobraQuotaCompilerV2(
            interpreter,
            spec.candidate,
            margin_type=spec.margin_type,
            **common,
        )

    def _profile_compiler_label(self, info: EscapeCompilerInfo) -> str:
        candidate_name = info.candidate_name
        if candidate_name.endswith(" noise"):
            candidate_name = candidate_name[: -len(" noise")]
        return f"{info.escape_id} ({candidate_name})"

    def print_diagnostics(self, limit: int = 5) -> None:
        remaining = self._remaining_indices()
        print(
            f"V2 audit diagnostic after {self.i} samples: "
            f"{len(remaining)} compilers remain below threshold "
            f"({len(self.certified)}/{len(self.compilers)} certified)."
        )
        for idx in self._lowest_capital_indices(remaining, limit):
            self._print_compiler_brief(idx)

    def print_summary(self, success: bool) -> None:
        if success:
            print(
                "V2 audit certified all compilers. "
                f"Final sample size: {self.i}."
            )
            recent = self.certification_order[-3:]
            if recent:
                print("Last compilers to pass threshold:")
                for idx in recent:
                    self._print_compiler_brief(idx)
            return

        print(
            "V2 audit did not certify all compilers before sample exhaustion. "
            f"Samples used: {self.i}/{self.sample_size}."
        )
        remaining = self._remaining_indices()
        for idx in self._lowest_capital_indices(remaining, 3):
            self._print_compiler_brief(idx)
            print(self.compilers[idx].get_info())

    def _remaining_indices(self) -> list[int]:
        if self.keep_certified_compilers:
            return [
                idx
                for idx in range(len(self.compilers))
                if idx not in self.certified
            ]
        return list(self.pool)

    def _lowest_capital_indices(self, indices: list[int], limit: int) -> list[int]:
        return sorted(indices, key=lambda idx: self.compilers[idx].M)[:limit]

    def _print_compiler_brief(self, idx: int) -> None:
        compiler = self.compilers[idx]
        info = self.compiler_info[idx]
        print(
            f"  {info.escape_id}: M={compiler.M:.6g}, "
            f"escape={info.action.name} {info.candidate}:{info.candidate_name}, "
            f"margin={compiler.critical_margin}, kind={compiler.compiler_kind}"
        )


__all__ = ["EscapeCompilerInfo", "GlobalAuditDriver", "GlobalAuditDriverV2"]

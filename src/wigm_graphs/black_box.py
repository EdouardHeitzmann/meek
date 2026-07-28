from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

try:
    from ..election_graphs import ChildProposal
    from ..election_graphs.utils import analyze_black_box_seed
except ImportError:
    from election_graphs import ChildProposal
    from election_graphs.utils import analyze_black_box_seed

from .datatypes import (
    BlackBoxElectionData,
    BlackBoxStateKey,
    EdgeAction,
    EdgeRef,
    EdgeStatus,
    ElectionEdge,
    ElectionState,
    ElectionStatus,
    PostSeedTallies,
    StateKey,
    VertexRef,
    WIGMRuntimeCache,
    WIGMWeightScenarios,
)
from .graph_wigm import WIGMGraphConstructor


@dataclass(slots=True)
class BlackBoxExpansionContext:
    weights: WIGMWeightScenarios
    uncertain_mask: NDArray[np.bool_]


class BlackBoxWIGMGraphConstructor(WIGMGraphConstructor):
    """
    WIGM constructor seeded after one election with an uncertain transfer.

    The graph keeps exact candidate availability in its ordinary runtime cache,
    while weight scenarios and uncertain-vote tallies are stored separately.
    """

    def _initialize_graph_specific_storage(self) -> None:
        super()._initialize_graph_specific_storage()
        self.edge_weight_scenarios: dict[EdgeRef, WIGMWeightScenarios] = {}
        self.vertex_post_seed_tallies: dict[VertexRef, PostSeedTallies] = {}
        self.seed_weight_scenarios: dict[int, WIGMWeightScenarios] = {}
        self.seed_uncertain_masks: dict[int, NDArray[np.bool_]] = {}
        self.seed_refs_by_id: dict[int, VertexRef] = {}
        self.black_box_elections: dict[EdgeRef, BlackBoxElectionData] = {}
        self.black_box_edge_ref: EdgeRef | None = None
        self.black_box_winner: int | None = None
        self.black_box_base_wt_vec: NDArray[np.float64] | None = None
        self.allow_seed_merges = False
        self.shared_seed_hopefuls: frozenset[int] | None = None

    def _make_state_key(
        self,
        *,
        seated_at: tuple[EdgeRef | None, ...],
        hopefuls: frozenset[int],
        parent_key: StateKey | BlackBoxStateKey | None = None,
    ) -> BlackBoxStateKey:
        seed_id = parent_key.seed_id if isinstance(parent_key, BlackBoxStateKey) else -1
        return BlackBoxStateKey(
            seated_at=seated_at,
            hopefuls=hopefuls,
            seed_id=seed_id,
        )

    def _same_seated_index_key(self, key: BlackBoxStateKey):
        return key.seated_at, key.seed_id

    def _remap_key_refs(
        self,
        key: BlackBoxStateKey,
        edge_ref_map: dict[EdgeRef, EdgeRef],
    ) -> BlackBoxStateKey:
        return BlackBoxStateKey(
            seated_at=tuple(
                None if edge_ref is None else edge_ref_map[edge_ref]
                for edge_ref in key.seated_at
            ),
            hopefuls=key.hopefuls,
            seed_id=key.seed_id,
        )

    def _state_key_signature(self, key: BlackBoxStateKey):
        return super()._state_key_signature(key), key.seed_id

    def seeded_build(
        self,
        *,
        strong_candidates: Iterable[int | str],
        uncertain_winner: int | str,
        very_strong_candidates: Iterable[int | str] | None = None,
        show_progress: bool = False,
        shwo_progress: bool | None = None,
        allow_seed_merges: bool = False,
    ):
        if shwo_progress is not None:
            if show_progress and bool(shwo_progress) != show_progress:
                raise ValueError(
                    "show_progress and shwo_progress specify conflicting values."
                )
            show_progress = bool(shwo_progress)
        allow_seed_merges = bool(allow_seed_merges)

        very_strong = self._candidate_indices(very_strong_candidates or ())
        strong = self._candidate_indices(strong_candidates)
        w = self._candidate_index(uncertain_winner)

        analysis = analyze_black_box_seed(
            self.profile,
            [self.candidate_names[candidate] for candidate in very_strong],
            [self.candidate_names[candidate] for candidate in strong],
            self.candidate_names[w],
            self.LAM,
            simultaneous=self.simultaneous,
            m=self.m,
            quota=self.quota,
            verbose=False,
        )
        weak = frozenset(
            self.candidate_names.index(candidate)
            for candidate in analysis["weak"]
        )
        in_between = frozenset(
            self.candidate_names.index(candidate)
            for candidate in analysis["in_between"]
        )
        print(
            f"Black-box seed analysis found {len(weak)} weak and "
            f"{len(in_between)} in-between candidates; "
            f"{analysis['seed_count']} seeds will be constructed."
        )

        self._validate_seed_candidate_sets(weak, strong, very_strong | {w})
        self._reset_seeded_graph_storage()
        self.allow_seed_merges = allow_seed_merges

        if very_strong:
            base_wt_vec, seated_at, elected_groups = self._initialize_very_strong_seed(
                very_strong
            )
            seated_very_strong = frozenset(
                candidate for group in elected_groups for candidate in group
            )
            if seated_very_strong != very_strong:
                raise ValueError(
                    "Not all requested very-strong candidates were force-elected."
                )
        else:
            base_wt_vec = self._unseeded_root_wt_vec.copy()
            seated_at = (None,) * self.m

        if not np.allclose(base_wt_vec, analysis["base_wt_vec"]):
            raise ValueError(
                "Very-strong seating produced weights inconsistent with the "
                "black-box seed analysis."
            )
        self.root_wt_vec = base_wt_vec
        self.black_box_base_wt_vec = base_wt_vec.copy()
        self.black_box_winner = w

        seated_at, raw_weight_scenarios = self._make_black_box_connector(
            seated_at=seated_at,
            degree=len(very_strong),
            winner=w,
            maximum_transfer_value=float(analysis["maximum_transfer_value"]),
        )

        self.used_seeded_build = True
        self.seed_very_strong_candidates = very_strong
        self.seed_strong_candidates = strong
        self.seed_weak_candidates = weak
        self.seed_frozen_mentions = None

        metadata = BlackBoxElectionData(
            candidate=w,
            very_strong_candidates=very_strong,
            strong_candidates=strong,
            weak_candidates=weak,
            raw_maximum_surplus=float(analysis["raw_maximum_surplus"]),
            maximum_surplus=float(analysis["maximum_surplus"]),
            maximum_transfer_value=float(analysis["maximum_transfer_value"]),
        )
        assert self.black_box_edge_ref is not None
        self.black_box_elections[self.black_box_edge_ref] = metadata

        initial_degree = len(very_strong) + 1
        remaining_seats = self.m - initial_degree
        flexible = tuple(
            candidate
            for candidate in range(self.n_candidates)
            if candidate not in very_strong | strong | weak | {w}
        )
        shared_uncertain_mask = None
        if self.allow_seed_merges:
            self.shared_seed_hopefuls = strong
            shared_cache = self._cache_for_hopefuls(strong)
            shared_uncertain_mask = self._uncertain_mask_for_seed(
                cache=shared_cache,
                very_strong=very_strong,
                strong=strong,
                winner=w,
            )

        seed_refs: list[VertexRef] = []
        next_seed_id = 0
        for subset_size in range(len(flexible) + 1):
            for subset in combinations(flexible, subset_size):
                hopefuls = strong | frozenset(subset)
                if len(hopefuls) < remaining_seats:
                    continue
                seed_id = 0 if self.allow_seed_merges else next_seed_id

                key = BlackBoxStateKey(
                    seated_at=seated_at,
                    hopefuls=hopefuls,
                    seed_id=seed_id,
                )
                layer = self.n_candidates - len(hopefuls)
                ref, is_new = self._get_or_create_vertex(
                    layer=layer,
                    key=key,
                    degree=initial_degree,
                )
                if not is_new:
                    raise RuntimeError("Black-box seed IDs must create unique vertices.")

                cache = self._materialize_cache_from_state(ref)
                uncertain_mask = (
                    shared_uncertain_mask
                    if shared_uncertain_mask is not None
                    else self._uncertain_mask_for_seed(
                        cache=cache,
                        very_strong=very_strong,
                        strong=strong,
                        winner=w,
                    )
                )
                scenarios = WIGMWeightScenarios(
                    values=np.where(
                        uncertain_mask[np.newaxis, :],
                        0.0,
                        raw_weight_scenarios.values,
                    ),
                    labels=raw_weight_scenarios.labels,
                )
                self.seed_weight_scenarios[seed_id] = scenarios
                self.seed_refs_by_id.setdefault(seed_id, ref)
                if not self.memory_lite:
                    self.seed_uncertain_masks[seed_id] = uncertain_mask
                    self.runtime_cache[ref] = cache

                vertex = self.vertex(ref)
                vertex.path_multiplicity = 1
                self._enqueue(ref)
                seed_refs.append(ref)
                next_seed_id += 1

        if not seed_refs:
            raise ValueError("Black-box seeded construction produced no viable seeds.")

        self._seed_refs = seed_refs
        root_key = self._make_root_key()
        root_local = self.layer_index[0].get(root_key) if self.layers else None
        self.root_ref = None if root_local is None else VertexRef(0, root_local)
        self._expand_seed_subtrees(show_progress=show_progress)

        self.coherence_checked = False
        self.tightest_margins_assigned = False
        self.terminal_vertices_by_winner_set = {}
        return self

    def _expand_seed_subtrees(self, *, show_progress: bool) -> None:
        progress = None
        if show_progress:
            try:
                from tqdm.auto import tqdm
            except ImportError as exc:
                raise ImportError(
                    "show_progress=True requires the optional tqdm package."
                ) from exc
            progress = tqdm(
                total=len(self._seed_refs),
                desc="Constructing black-box seeds",
                unit="seed",
            )

        active_seed_id: int | None = None
        seed_refs = set(self._seed_refs)
        completed_seed_refs: set[VertexRef] = set()
        try:
            while self.stack:
                ref = self.stack.pop()
                vertex = self.vertex(ref)
                seed_id = vertex.key.seed_id

                if not self.allow_seed_merges:
                    if active_seed_id is None:
                        active_seed_id = seed_id
                    elif seed_id != active_seed_id:
                        if progress is not None:
                            progress.update(1)
                        active_seed_id = seed_id

                if vertex.status == ElectionStatus.UNEXPANDED:
                    self._expand_vertex(ref)

                if (
                    self.allow_seed_merges
                    and ref in seed_refs
                    and ref not in completed_seed_refs
                ):
                    completed_seed_refs.add(ref)
                    if progress is not None:
                        progress.update(1)

            if (
                not self.allow_seed_merges
                and active_seed_id is not None
                and progress is not None
            ):
                progress.update(1)
        finally:
            if progress is not None:
                progress.close()

    def _candidate_index(self, candidate: int | str) -> int:
        if isinstance(candidate, str):
            try:
                return self.candidate_names.index(candidate)
            except ValueError as exc:
                raise ValueError(f"Unknown candidate: {candidate!r}") from exc
        index = int(candidate)
        if index < 0 or index >= self.n_candidates:
            raise ValueError(f"Candidate index out of range: {index}")
        return index

    def _candidate_indices(
        self,
        candidates: Iterable[int | str],
    ) -> frozenset[int]:
        return frozenset(self._candidate_index(candidate) for candidate in candidates)

    def _make_black_box_connector(
        self,
        *,
        seated_at: tuple[EdgeRef | None, ...],
        degree: int,
        winner: int,
        maximum_transfer_value: float,
    ) -> tuple[tuple[EdgeRef | None, ...], WIGMWeightScenarios]:
        parent_ref = self._seed_connector_ref
        if parent_ref is None:
            parent_ref = self._make_already_elected_root_anchor()
        parent = self.vertex(parent_ref)

        parent_cache = self._materialize_cache_from_state(parent_ref)
        fpv = parent_cache.fpv_vec
        if fpv is None:
            raise ValueError("Black-box connector parent has no FPV vector.")

        values = np.repeat(
            self.black_box_base_wt_vec[np.newaxis, :],
            2,
            axis=0,
        )
        definite_winner_rows = fpv == winner
        values[0, definite_winner_rows] = 0.0
        values[1, definite_winner_rows] *= maximum_transfer_value
        scenarios = WIGMWeightScenarios(
            values=values,
            labels=(("black_box", "tau_zero"), ("black_box", "tau_high")),
        )

        edge_ref = self._next_edge_ref(parent_ref.layer)
        next_seated_at = list(seated_at)
        next_seated_at[degree] = edge_ref
        child_ref = self._create_vertex_no_dedupe(
            layer=parent_ref.layer + 1,
            key=BlackBoxStateKey(
                seated_at=tuple(next_seated_at),
                hopefuls=parent.key.hopefuls - {winner},
                seed_id=-1,
            ),
            degree=degree + 1,
        )
        child = self.vertex(child_ref)
        child.status = ElectionStatus.EXPANDED
        child.path_multiplicity = 0

        created_ref, _ = self._add_edge(
            src=parent_ref,
            dst=child_ref,
            action=EdgeAction.BLACK_BOX_ELECT,
            candidate=winner,
            status=EdgeStatus.CANONICAL,
            transfer_value=maximum_transfer_value,
            wt_vec=values[1].copy(),
            fpv_vec=(
                None
                if parent_cache.fpv_vec is None or not self.store_fpv_vec
                else parent_cache.fpv_vec.copy()
            ),
            margin=0.0,
            forced_ref=edge_ref,
            skip_dedupe=True,
        )
        self.black_box_edge_ref = created_ref
        self.edge_weight_scenarios[created_ref] = scenarios
        self._seed_connector_ref = child_ref
        return tuple(next_seated_at), scenarios

    def _uncertain_mask_for_seed(
        self,
        *,
        cache: WIGMRuntimeCache,
        very_strong: frozenset[int],
        strong: frozenset[int],
        winner: int,
    ) -> NDArray[np.bool_]:
        current_pos = cache.pos_vec
        columns = np.arange(self.ballot_matrix.shape[1])[np.newaxis, :]

        base_available = np.ones_like(self.ballot_matrix, dtype=np.bool_)
        for candidate in very_strong:
            base_available &= self.candidate_stencils[candidate]
        base_pos = base_available.argmax(axis=1)
        base_fpv = self.ballot_matrix[np.arange(len(base_pos)), base_pos]

        winner_positions = np.where(
            self.ballot_matrix == winner,
            columns,
            self.ballot_matrix.shape[1],
        ).min(axis=1)
        prefix = columns < current_pos[:, np.newaxis]
        strong_in_prefix = (
            prefix & np.isin(self.ballot_matrix, np.fromiter(strong, dtype=np.intp))
        ).any(axis=1)

        return (
            (base_fpv != winner)
            & (winner_positions < current_pos)
            & ~strong_in_prefix
        )

    def _uncertain_mask_for_vertex(
        self,
        ref: VertexRef,
        cache: WIGMRuntimeCache,
    ) -> NDArray[np.bool_]:
        key = self.vertex(ref).key
        seed_id = key.seed_id
        stored = self.seed_uncertain_masks.get(seed_id)
        if stored is not None:
            return stored

        if self.allow_seed_merges:
            seed_cache = self._cache_for_hopefuls(self.shared_seed_hopefuls)
        else:
            seed_ref = self.seed_refs_by_id[seed_id]
            seed_cache = self._materialize_cache_from_state(seed_ref)
        return self._uncertain_mask_for_seed(
            cache=seed_cache,
            very_strong=self.seed_very_strong_candidates,
            strong=self.seed_strong_candidates,
            winner=self.black_box_winner,
        )

    def _cache_for_hopefuls(
        self,
        hopefuls: frozenset[int],
    ) -> WIGMRuntimeCache:
        bool_ballot_matrix = np.ones_like(self.ballot_matrix, dtype=np.bool_)
        unavailable = set(range(self.n_candidates)) - set(hopefuls)
        for candidate in unavailable:
            bool_ballot_matrix &= self.candidate_stencils[candidate]
        pos_vec = bool_ballot_matrix.argmax(axis=1).astype(np.int8)
        fpv_vec = self.ballot_matrix[
            np.arange(self.ballot_matrix.shape[0]),
            pos_vec,
        ]
        return WIGMRuntimeCache(
            bool_ballot_matrix=bool_ballot_matrix,
            pos_vec=pos_vec,
            fpv_vec=fpv_vec,
        )

    def _weight_scenarios_for_vertex(
        self,
        vertex: ElectionState,
    ) -> WIGMWeightScenarios:
        latest = self._latest_seating_edge(vertex.key)
        if latest is None or latest == self.black_box_edge_ref:
            return self.seed_weight_scenarios[vertex.key.seed_id]
        return self.edge_weight_scenarios[latest]

    def _expansion_context(
        self,
        ref: VertexRef,
        v: ElectionState,
        cache: WIGMRuntimeCache,
        incoming_edge: ElectionEdge | None,
    ) -> BlackBoxExpansionContext:
        return BlackBoxExpansionContext(
            weights=self._weight_scenarios_for_vertex(v),
            uncertain_mask=self._uncertain_mask_for_vertex(ref, cache),
        )

    def _compute_tallies(
        self,
        v: ElectionState,
        cache: WIGMRuntimeCache,
        context: BlackBoxExpansionContext,
    ) -> NDArray[np.float64]:
        if cache.fpv_vec is None:
            raise ValueError("cache.fpv_vec has not been computed.")

        valid = cache.fpv_vec >= 0
        certain = np.vstack(
            [
                np.bincount(
                    cache.fpv_vec[valid],
                    weights=weights[valid],
                    minlength=self.n_candidates,
                )
                for weights in context.weights.values
            ]
        ).astype(np.float64)
        uncertain_rows = valid & context.uncertain_mask
        uncertain = np.bincount(
            cache.fpv_vec[uncertain_rows],
            weights=self.black_box_base_wt_vec[uncertain_rows],
            minlength=self.n_candidates,
        ).astype(np.float64)
        bounded = PostSeedTallies(certain=certain, uncertain=uncertain)
        self.vertex_post_seed_tallies[v.ref] = bounded

        # Preserve the existing one-dimensional diagnostic surface.
        return bounded.lower

    def _next_margin_for_vertex(
        self,
        v: ElectionState,
        context: BlackBoxExpansionContext,
    ) -> float | None:
        if v.ref not in self.vertex_post_seed_tallies:
            return super()._next_margin_for_vertex(v, context)

        proposals = list(self._bounded_proposals(v, context, include_all=True))
        excluded = [
            float(proposal.margin)
            for proposal in proposals
            if proposal.margin is not None and proposal.margin > self.LAM
        ]
        return min(excluded, default=None)

    def _propose_children(
        self,
        v: ElectionState,
        cache: WIGMRuntimeCache,
        context: BlackBoxExpansionContext,
    ):
        for proposal in self._bounded_proposals(v, context, include_all=False):
            if proposal.action.is_election:
                group = proposal.candidates or (proposal.candidate,)
                if proposal.transfer_values == (0.0,):
                    scenarios = context.weights
                    low_values = (0.0,)
                    high_values = (0.0,)
                else:
                    scenarios, low_values, high_values = self._election_scenarios(
                        group=group,
                        cache=cache,
                        context=context,
                        tallies=self.vertex_post_seed_tallies[v.ref],
                    )
                proposal.payload = scenarios
                proposal.wt_vec = scenarios.values[0].copy()
                proposal.transfer_value = high_values[0]
                proposal.transfer_values = high_values
                proposal.fpv_vec = self._edge_fpv_vec_from_cache(cache)
            yield proposal

    def _bounded_proposals(
        self,
        v: ElectionState,
        context: BlackBoxExpansionContext,
        *,
        include_all: bool,
    ):
        bounded = self.vertex_post_seed_tallies[v.ref]
        hopefuls = np.asarray(sorted(v.key.hopefuls), dtype=int)
        if len(hopefuls) == 0:
            return

        if len(hopefuls) + v.degree == self.m:
            for candidate in hopefuls:
                yield ChildProposal(
                    action=EdgeAction.FORCE_ELECT,
                    candidate=int(candidate),
                    transfer_value=0.0,
                    transfer_values=(0.0,),
                    margin=0.0,
                    candidates=(int(candidate),),
                )
            return

        forced = frozenset(
            int(candidate)
            for candidate in hopefuls
            if np.all(
                bounded.certain[:, candidate] > self.quota + self.LAM
            )
        )
        election_margin = {
            int(candidate): self._election_margin_for_group(
                bounded,
                hopefuls,
                (int(candidate),),
            )
            for candidate in hopefuls
        }
        eligible = tuple(
            int(candidate)
            for candidate in hopefuls
            if include_all or election_margin[int(candidate)] <= self.LAM
        )

        if self.simultaneous:
            max_size = min(len(eligible), self.m - v.degree)
            groups = (
                group
                for size in range(1, max_size + 1)
                for group in combinations(eligible, size)
                if not forced or forced <= frozenset(group)
            )
        else:
            groups = (
                (candidate,)
                for candidate in eligible
                if not forced or candidate in forced
            )

        for group in groups:
            margin = self._election_margin_for_group(
                bounded,
                hopefuls,
                tuple(group),
            )
            if not include_all and margin > self.LAM:
                continue
            yield ChildProposal(
                action=(
                    EdgeAction.FORCE_ELECT
                    if forced and len(group) == 1
                    else (
                        EdgeAction.ELECT
                        if len(group) == 1
                        else EdgeAction.SIMULTANEOUS_ELECT
                    )
                ),
                candidate=group[0],
                candidates=tuple(group),
                margin=margin,
            )

        if forced:
            return

        for candidate in hopefuls:
            margin = self._elimination_margin(
                bounded,
                hopefuls,
                int(candidate),
            )
            if include_all or margin <= self.LAM:
                yield ChildProposal(
                    action=EdgeAction.ELIMINATE,
                    candidate=int(candidate),
                    margin=margin,
                )

    def _election_margin_for_group(
        self,
        tallies: PostSeedTallies,
        hopefuls: NDArray[np.integer],
        group: tuple[int, ...],
    ) -> float:
        scenario_margins = []
        group_set = frozenset(group)

        for scenario in tallies.certain:
            member_margins = []
            for candidate in group:
                other_tallies = [
                    scenario[other]
                    for other in hopefuls
                    if int(other) not in group_set
                ]
                highest_other = max(other_tallies, default=0.0)
                raw_margin = max(
                    self.quota - scenario[candidate],
                    highest_other - scenario[candidate],
                    0.0,
                )
                member_margins.append(
                    max(raw_margin - tallies.uncertain[candidate], 0.0)
                )

            scenario_margins.append(max(member_margins, default=0.0))

        return float(min(scenario_margins, default=0.0))

    def _elimination_margin(
        self,
        tallies: PostSeedTallies,
        hopefuls: NDArray[np.integer],
        candidate: int,
    ) -> float:
        scenario_margins = []

        for scenario in tallies.certain:
            others = [
                int(other) for other in hopefuls if int(other) != candidate
            ]
            if not others:
                scenario_margins.append(0.0)
                continue

            comparison_candidate = min(
                others,
                key=lambda other: scenario[other],
            )
            raw_margin = max(
                scenario[candidate] - scenario[comparison_candidate],
                float(np.max(scenario[hopefuls])) - self.quota,
                0.0,
            )
            scenario_margins.append(
                max(
                    raw_margin - tallies.uncertain[comparison_candidate],
                    0.0,
                )
            )

        return float(min(scenario_margins, default=0.0))

    def _election_scenarios(
        self,
        *,
        group: tuple[int, ...],
        cache: WIGMRuntimeCache,
        context: BlackBoxExpansionContext,
        tallies: PostSeedTallies,
    ) -> tuple[WIGMWeightScenarios, tuple[float, ...], tuple[float, ...]]:
        low_values = tuple(
            self._transfer_value(float(tallies.lower[candidate]))
            for candidate in group
        )
        high_values = tuple(
            self._transfer_value(float(tallies.upper[candidate]))
            for candidate in group
        )

        rows: list[NDArray[np.float64]] = []
        labels = []
        for parent_label, parent_weights in zip(
            context.weights.labels,
            context.weights.values,
        ):
            for choices in product((0, 1), repeat=len(group)):
                updated = parent_weights.copy()
                choice_labels = []
                for candidate, choice, low, high in zip(
                    group,
                    choices,
                    low_values,
                    high_values,
                ):
                    transfer_value = high if choice else low
                    updated[cache.fpv_vec == candidate] *= transfer_value
                    choice_labels.append(
                        (candidate, "tau_high" if choice else "tau_low")
                    )
                rows.append(updated)
                labels.append((parent_label, tuple(choice_labels)))

        values = np.vstack(rows)
        _, unique_indices = np.unique(values, axis=0, return_index=True)
        unique_indices.sort()
        scenarios = WIGMWeightScenarios(
            values=values[unique_indices],
            labels=tuple(labels[index] for index in unique_indices),
        )
        return scenarios, low_values, high_values

    def _transfer_value(self, tally: float) -> float:
        if tally == 0:
            return 0.0
        return float((tally - self.quota) / tally)

    def _add_child_from_proposal(
        self,
        parent_ref: VertexRef,
        proposal: ChildProposal,
    ) -> VertexRef:
        child_ref = super()._add_child_from_proposal(parent_ref, proposal)
        if proposal.action.is_election:
            edge_ref = self.primary_parent_edge[child_ref]
            if not isinstance(proposal.payload, WIGMWeightScenarios):
                raise ValueError("Black-box election proposal is missing scenarios.")
            self.edge_weight_scenarios[edge_ref] = proposal.payload
            if proposal.action == EdgeAction.SIMULTANEOUS_ELECT:
                for seated_ref in self.vertex(child_ref).key.seated_at:
                    if seated_ref is not None:
                        edge = self.edge(seated_ref)
                        if edge.src == parent_ref and edge.dst == child_ref:
                            self.edge_weight_scenarios[seated_ref] = proposal.payload
        return child_ref

    def lookup_vertex(self, label: str) -> ElectionState:
        vertex = super().lookup_vertex(label)
        bounded = self.vertex_post_seed_tallies.get(vertex.ref)
        if bounded is None:
            print("  post-seed tallies: unavailable")
            return vertex

        print("  certain tally scenarios:")
        for scenario_index, tallies in enumerate(bounded.certain):
            print(f"    scenario {scenario_index}:")
            self._print_candidate_array(tallies, indent="      ")

        print("  uncertain tallies:")
        self._print_candidate_array(bounded.uncertain, indent="    ")
        print("  lower tally bounds:")
        self._print_candidate_array(bounded.lower, indent="    ")
        print("  upper tally bounds:")
        self._print_candidate_array(bounded.upper, indent="    ")
        return vertex

    def lookup_edge(
        self,
        ref: EdgeRef | tuple[int, int] | str,
    ) -> ElectionEdge:
        edge_ref = self._normalize_edge_ref(ref)
        edge = self.edge(edge_ref)
        print(f"Edge {edge_ref.layer}:{edge_ref.local_id}")
        self._print_edge_summary(edge)

        scenarios = self.edge_weight_scenarios.get(edge_ref)
        if scenarios is None:
            print("  weight scenarios: unavailable")
        else:
            print(f"  weight scenarios: {len(scenarios.labels)}")
            for index, (label, weights) in enumerate(
                zip(scenarios.labels, scenarios.values)
            ):
                print(
                    f"    scenario {index}: label={label!r}; "
                    f"total_weight={float(weights.sum()):.6g}"
                )

        bounded = self.vertex_post_seed_tallies.get(edge.dst)
        if bounded is not None:
            print(
                "  destination bounded tallies: "
                f"{self.vertex_label(edge.dst)}"
            )
            print("    uncertain:")
            self._print_candidate_array(bounded.uncertain, indent="      ")
            print("    lower:")
            self._print_candidate_array(bounded.lower, indent="      ")
            print("    upper:")
            self._print_candidate_array(bounded.upper, indent="      ")

        return edge

    def post_seed_tallies(
        self,
        vertex: VertexRef | str,
    ) -> PostSeedTallies:
        ref = self.parse_vertex_label(vertex) if isinstance(vertex, str) else vertex
        try:
            return self.vertex_post_seed_tallies[ref]
        except KeyError as exc:
            raise ValueError(f"Vertex {ref} has no post-seed tallies.") from exc

    def weight_scenarios(
        self,
        edge: EdgeRef | tuple[int, int] | str,
    ) -> WIGMWeightScenarios:
        ref = self._normalize_edge_ref(edge)
        try:
            return self.edge_weight_scenarios[ref]
        except KeyError as exc:
            raise ValueError(f"Edge {ref} has no weight scenarios.") from exc

    def _normalize_edge_ref(
        self,
        ref: EdgeRef | tuple[int, int] | str,
    ) -> EdgeRef:
        if isinstance(ref, EdgeRef):
            return ref
        if isinstance(ref, tuple) and len(ref) == 2:
            return EdgeRef(layer=int(ref[0]), local_id=int(ref[1]))
        if isinstance(ref, str):
            normalized = ref.strip().upper().removeprefix("E")
            separator = ":" if ":" in normalized else ","
            parts = normalized.split(separator)
            if len(parts) == 2:
                return EdgeRef(layer=int(parts[0]), local_id=int(parts[1]))
        raise ValueError(
            "Edge reference must be EdgeRef, (layer, local_id), or "
            "a string like 'E2:4'."
        )

    def _print_candidate_array(
        self,
        values: NDArray[np.float64],
        *,
        indent: str,
    ) -> None:
        for candidate, value in enumerate(values):
            print(
                f"{indent}{candidate}: {self.candidate_names[candidate]} "
                f"-> {float(value):.6g}"
            )


__all__ = ["BlackBoxWIGMGraphConstructor"]

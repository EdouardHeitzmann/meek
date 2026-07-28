from itertools import combinations
from math import comb

from .datatypes import (
    EdgeAction,
    EdgeStatus,
    ElectionStatus,
    ElectionState,
    ElectionEdge,
    StateKey,
    EdgeRef,
    VertexRef,)
from .datatypes import WIGMRuntimeCache as RuntimeCache
try:
    from ..election_graphs import AbstractGraphConstructor, ChildProposal
    from ..election_graphs.utils import (
        frozen_mentions_from_matrix,
        search_strong_weak_candidates,
        weak_candidates_from_strong,
    )
except ImportError:
    from election_graphs import AbstractGraphConstructor, ChildProposal
    from election_graphs.utils import (
        frozen_mentions_from_matrix,
        search_strong_weak_candidates,
        weak_candidates_from_strong,
    )

import numpy as np
from numpy.typing import NDArray
from typing import Iterable, Sequence
from collections import defaultdict
import warnings


class WIGMGraphConstructor(AbstractGraphConstructor):
    def __init__(
        self,
        profile,
        m: int,
        LAM: float,
        *,
        memory_lite: bool = False,
        trip_when_incoherent: bool = False,
        simultaneous: bool = False,
        store_fpv_vec: bool = True,
    ) -> None:
        self.simultaneous = bool(simultaneous)
        self.store_fpv_vec = bool(store_fpv_vec)
        super().__init__(
            profile,
            m,
            LAM,
            memory_lite=memory_lite,
            trip_when_incoherent=trip_when_incoherent,
        )

    # -----------------------------------------------------------------
    # Initial setup
    # -----------------------------------------------------------------

    def _initialize_profile_data(self, profile) -> None:
        if self._has_numpy_profile_arrays(profile):
            self.ballot_matrix, self.root_wt_vec = (
                self._make_padded_numpy_profile_arrays(profile)
            )
        else:
            self.ballot_matrix, self.root_wt_vec = self._make_ballot_matrix(profile)
        self.candidate_stencils = self._make_candidate_stencils(self.ballot_matrix)
        self._unseeded_root_wt_vec = self.root_wt_vec.copy()

        # Droop quota.
        self.quota = np.floor(self.root_wt_vec.sum() / (self.m + 1)) + 1

    def _initialize_graph_specific_storage(self) -> None:
        # Index for post-build natural-edge completion.
        self.same_seated_index: list[
            dict[tuple[EdgeRef | None, ...], set[VertexRef]]
        ] = []
        self._seed_connector_ref: VertexRef | None = None
        self.used_seeded_build = False
        self.seed_very_strong_candidates = frozenset()
        self.seed_strong_candidates = frozenset()
        self.seed_weak_candidates = frozenset()
        self.seed_frozen_mentions: NDArray[np.float64] | None = None

    def _on_new_layer(self, layer: int) -> None:
        self.same_seated_index.append(defaultdict(set))

    def _record_vertex_index(self, layer: int, key: StateKey, ref: VertexRef) -> None:
        self.same_seated_index[layer][self._same_seated_index_key(key)].add(ref)

    def _same_seated_index_key(self, key: StateKey):
        return key.seated_at

    def _make_state_key(
        self,
        *,
        seated_at: tuple[EdgeRef | None, ...],
        hopefuls: frozenset[int],
        parent_key: StateKey | None = None,
    ) -> StateKey:
        return StateKey(seated_at=seated_at, hopefuls=hopefuls)

    def _remap_key_refs(
        self,
        key: StateKey,
        edge_ref_map: dict[EdgeRef, EdgeRef],
    ) -> StateKey:
        seated_at = tuple(
            None if edge_ref is None else edge_ref_map[edge_ref]
            for edge_ref in key.seated_at
        )
        return self._make_state_key(
            seated_at=seated_at,
            hopefuls=key.hopefuls,
            parent_key=key,
        )

    def _state_key_signature(
        self,
        key: StateKey,
    ) -> tuple[tuple[tuple[int, object, int, int] | None, ...], frozenset[int]]:
        seated_edges = tuple(
            None if edge_ref is None else self._seating_edge_signature(edge_ref)
            for edge_ref in key.seated_at
        )
        return seated_edges, key.hopefuls

    def _seating_edge_signature(
        self,
        edge_ref: EdgeRef,
    ) -> tuple[int, object, int, int]:
        edge = self.edge(edge_ref)
        parent = self.vertex(edge.src)
        return (
            edge.src.layer,
            self._state_key_signature(parent.key),
            int(edge.action),
            int(edge.candidate),
        )

    def _make_ballot_matrix(
        self,
        pf,
    ) -> tuple[NDArray[np.integer], NDArray[np.float64]]:
        df = pf.df.copy()

        candidate_to_index = {
            frozenset([name]): i
            for i, name in enumerate(self.candidate_names)
        }
        candidate_to_index[frozenset(["~"])] = int(-127)

        ranking_columns = [c for c in df.columns if c.startswith("Ranking")]
        num_rows = len(df)
        num_cols = len(ranking_columns)

        if num_cols > len(pf.candidates):
            ranking_columns = ranking_columns[: len(pf.candidates)]
            num_cols = len(ranking_columns)

        cells = df[ranking_columns].to_numpy()

        def map_cell(cell):
            try:
                return candidate_to_index[cell]
            except KeyError:
                raise TypeError(f"Found invalid entry: {cell}")

        mapped = np.frompyfunc(map_cell, 1, 1)(cells).astype(np.int8)

        # Padding gives every row an eventual exhausted / sentinel value.
        ballot_matrix: NDArray = np.full(
            (num_rows, num_cols + 1),
            -127,
            dtype=np.int8,
        )
        ballot_matrix[:, :num_cols] = mapped

        wt_vec: NDArray = df["Weight"].astype(np.float64).to_numpy()

        return ballot_matrix, wt_vec

    def _make_candidate_stencils(
        self,
        ballot_matrix: NDArray[np.integer],
    ) -> list[NDArray[np.bool_]]:
        stencil_list = []

        for idx in range(self.n_candidates):
            stencil = ballot_matrix == idx
            stencil_list.append(~stencil)

        return stencil_list

    def _make_root_key(self) -> StateKey:
        return self._make_state_key(
            seated_at=(None,) * self.m,
            hopefuls=frozenset(range(self.n_candidates)),
        )

    def _make_root_cache(self) -> RuntimeCache:
        bool_ballot_matrix = np.ones_like(self.ballot_matrix, dtype=np.bool_)
        pos_vec = np.zeros(bool_ballot_matrix.shape[0], dtype=np.int8)
        fpv_vec = self.ballot_matrix[np.arange(bool_ballot_matrix.shape[0]), pos_vec]

        return RuntimeCache(
            bool_ballot_matrix=bool_ballot_matrix,
            pos_vec=pos_vec,
            fpv_vec=fpv_vec,
        )

    def _derive_child_cache(
        self,
        parent_cache: RuntimeCache,
        incoming_edge: ElectionEdge,
    ) -> RuntimeCache:
        bool_ballot_matrix = parent_cache.bool_ballot_matrix.copy()
        if incoming_edge.action == EdgeAction.SIMULTANEOUS_ELECT:
            removed_candidates = self._simultaneous_group_for_edge(incoming_edge)
        else:
            removed_candidates = (incoming_edge.candidate,)

        for candidate in removed_candidates:
            bool_ballot_matrix &= self.candidate_stencils[candidate]

        needs_update = np.isin(parent_cache.fpv_vec, removed_candidates)

        pos_vec = parent_cache.pos_vec.copy()
        pos_vec[needs_update] = bool_ballot_matrix[needs_update].argmax(axis=1)

        fpv_vec = self.ballot_matrix[np.arange(bool_ballot_matrix.shape[0]), pos_vec]

        return RuntimeCache(
            bool_ballot_matrix=bool_ballot_matrix,
            pos_vec=pos_vec,
            fpv_vec=fpv_vec,
        )

    def _simultaneous_group_for_edge(
        self,
        incoming_edge: ElectionEdge,
    ) -> tuple[int, ...]:
        if incoming_edge.action != EdgeAction.SIMULTANEOUS_ELECT:
            return (incoming_edge.candidate,)

        group = [
            edge.candidate
            for edge in self.edge_layers[incoming_edge.ref.layer]
            if (
                edge.src == incoming_edge.src
                and edge.dst == incoming_edge.dst
                and edge.action == EdgeAction.SIMULTANEOUS_ELECT
            )
        ]
        if not group:
            raise ValueError(
                f"Could not recover simultaneous election group for {incoming_edge.ref}."
            )

        return tuple(group)

    def _materialize_cache_from_state(self, ref: VertexRef) -> RuntimeCache:
        v = self.vertex(ref)
        bool_ballot_matrix = np.ones_like(self.ballot_matrix, dtype=np.bool_)

        unavailable = set(range(self.n_candidates)) - set(v.key.hopefuls)
        for candidate in unavailable:
            bool_ballot_matrix &= self.candidate_stencils[candidate]

        pos_vec = bool_ballot_matrix.argmax(axis=1).astype(np.int8)
        fpv_vec = self.ballot_matrix[np.arange(bool_ballot_matrix.shape[0]), pos_vec]

        return RuntimeCache(
            bool_ballot_matrix=bool_ballot_matrix,
            pos_vec=pos_vec,
            fpv_vec=fpv_vec,
        )

    # -----------------------------------------------------------------
    # Weight-vector / tally logic
    # -----------------------------------------------------------------

    def _wt_vec_for_vertex(
        self,
        v: ElectionState,
        incoming_edge: ElectionEdge | None,
    ) -> NDArray[np.float64]:
        if incoming_edge is not None and incoming_edge.action.is_election:
            if incoming_edge.wt_vec is None:
                raise ValueError("Election edge is missing wt_vec.")
            return incoming_edge.wt_vec

        latest = self._latest_seating_edge(v.key)

        if latest is None:
            return self.root_wt_vec

        e = self.edge(latest)
        if e.wt_vec is None:
            raise ValueError(f"Seating edge {latest} is missing wt_vec.")

        return e.wt_vec

    def _expansion_context(
        self,
        ref: VertexRef,
        v: ElectionState,
        cache: RuntimeCache,
        incoming_edge: ElectionEdge | None,
    ) -> NDArray[np.float64]:
        return self._wt_vec_for_vertex(v, incoming_edge)

    def _latest_seating_edge(self, key: StateKey) -> EdgeRef | None:
        seating_edges = [e for e in key.seated_at if e is not None]
        if not seating_edges:
            return None

        return max(seating_edges, key=lambda e: (e.layer, e.local_id))

    def _compute_tallies(
        self,
        v: ElectionState,
        cache: RuntimeCache,
        wt_vec: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if cache.fpv_vec is None:
            raise ValueError("cache.fpv_vec has not been computed.")

        not_exhausted_mask = cache.fpv_vec >= 0

        return np.bincount(
            cache.fpv_vec[not_exhausted_mask],
            weights=wt_vec[not_exhausted_mask],
            minlength=self.n_candidates,
        ).astype(np.float64)

    def _compute_tallies_from_fpv(
        self,
        fpv_vec: NDArray[np.integer],
        wt_vec: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        not_exhausted_mask = fpv_vec >= 0
        return np.bincount(
            fpv_vec[not_exhausted_mask],
            weights=wt_vec[not_exhausted_mask],
            minlength=self.n_candidates,
        ).astype(np.float64)

    def _edge_fpv_vec_from_cache(
        self,
        cache: RuntimeCache,
    ) -> NDArray[np.integer] | None:
        if not self.store_fpv_vec:
            return None
        if cache.fpv_vec is None:
            return None
        return cache.fpv_vec.copy()

    def _next_margin_for_vertex(
        self,
        v: ElectionState,
        wt_vec: NDArray[np.float64],
    ) -> float | None:
        if v.tallies is None or v.degree == self.m:
            return None

        margins: list[float] = []

        if np.any(v.tallies > self.quota + self.LAM):
            highest_tally = np.max(v.tallies)
            margins.extend(
                float(highest_tally - tally)
                for tally in v.tallies
                if highest_tally - tally > self.LAM
            )
        elif len(v.key.hopefuls) + v.degree == self.m:
            return None
        else:
            highest_tally = np.max(v.tallies)
            election_margins = np.maximum(
                np.maximum(highest_tally, self.quota) - v.tallies,
                0.0,
            )
            margins.extend(float(m) for m in election_margins if m > self.LAM)

            hopefuls = np.array(list(v.key.hopefuls), dtype=int)
            if len(hopefuls) > 0:
                lowest_tally = np.min(v.tallies[hopefuls])
                elimination_margin_floor = float(max(highest_tally - self.quota, 0.0))
                elimination_margins = np.maximum(
                    v.tallies[hopefuls] - lowest_tally,
                    elimination_margin_floor,
                )
                margins.extend(float(m) for m in elimination_margins if m > self.LAM)

        if not margins:
            return None
        return min(margins)

    # -----------------------------------------------------------------
    # Child proposal logic
    # -----------------------------------------------------------------

    def _updated_wt_vec_for_election_group(
        self,
        group: tuple[int, ...],
        cache: RuntimeCache,
        wt_vec: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], tuple[float, ...]]:
        updated_wt_vec = wt_vec.copy()
        transfer_values: list[float] = []

        for candidate in group:
            tally = float(wt_vec[cache.fpv_vec == candidate].sum())
            if tally == 0:
                raise ValueError(
                    f"Cannot compute transfer value for candidate {candidate} "
                    "with zero first-preference tally."
                )
            transfer_value = float((tally - self.quota) / tally)
            transfer_values.append(transfer_value)
            updated_wt_vec[cache.fpv_vec == candidate] *= transfer_value

        return updated_wt_vec, tuple(transfer_values)

    def _election_groups_within_lam(
        self,
        v: ElectionState,
        *,
        forced: bool,
    ) -> Iterable[tuple[int, ...]]:
        if v.tallies is None:
            raise ValueError("Cannot choose election groups before tallies exist.")

        remaining_seats = self.m - v.degree
        if remaining_seats <= 0:
            return ()

        hopefuls = np.array(sorted(v.key.hopefuls), dtype=int)
        if len(hopefuls) == 0:
            return ()

        highest_tally = float(np.max(v.tallies[hopefuls]))
        within_quota = [
            int(candidate)
            for candidate in hopefuls
            if v.tallies[candidate] >= self.quota - self.LAM
        ]

        if len(within_quota) > remaining_seats:
            within_quota = [
                candidate
                for candidate in within_quota
                if v.tallies[candidate] >= highest_tally - self.LAM
            ]

        required = frozenset()
        if forced:
            required = frozenset(
                int(candidate)
                for candidate in hopefuls
                if v.tallies[candidate] > self.quota + self.LAM
            )
            if len(required) > remaining_seats:
                return ()

        max_group_size = min(len(within_quota), remaining_seats)
        groups = []
        for group_size in range(1, max_group_size + 1):
            for group in combinations(within_quota, group_size):
                group_set = frozenset(group)
                if required and not required <= group_set:
                    continue
                groups.append(tuple(sorted(group)))

        return tuple(groups)

    def _propose_children(
        self,
        v: ElectionState,
        cache: RuntimeCache,
        wt_vec: NDArray[np.float64],
    ) -> Iterable[ChildProposal]:
        if v.tallies is None:
            raise ValueError("Cannot propose children before tallies are computed.")

        hopefuls = np.array(sorted(v.key.hopefuls), dtype=int)
        if len(hopefuls) == 0:
            return

        # Forced election: someone is safely above quota + LAM.
        if np.any(v.tallies[hopefuls] > self.quota + self.LAM):
            edge_fpv_vec = self._edge_fpv_vec_from_cache(cache)
            if self.simultaneous:
                for group in self._election_groups_within_lam(v, forced=True):
                    updated_wt_vec, transfer_values = (
                        self._updated_wt_vec_for_election_group(group, cache, wt_vec)
                    )
                    action = (
                        EdgeAction.FORCE_ELECT
                        if len(group) == 1
                        else EdgeAction.SIMULTANEOUS_ELECT
                    )
                    yield ChildProposal(
                        action=action,
                        candidate=int(group[0]),
                        transfer_value=transfer_values[0],
                        transfer_values=transfer_values,
                        wt_vec=updated_wt_vec,
                        fpv_vec=edge_fpv_vec,
                        margin=0.0,
                        candidates=group,
                    )
                return

            highest_tally = np.max(v.tallies[hopefuls])
            winner_idx_within_lam = hopefuls[
                np.where(v.tallies[hopefuls] >= highest_tally - self.LAM)[0]
            ]

            for candidate in winner_idx_within_lam:
                group = (int(candidate),)
                updated_wt_vec, transfer_values = (
                    self._updated_wt_vec_for_election_group(group, cache, wt_vec)
                )
                yield ChildProposal(
                    action=EdgeAction.FORCE_ELECT,
                    candidate=int(group[0]),
                    transfer_value=transfer_values[0],
                    transfer_values=transfer_values,
                    wt_vec=updated_wt_vec,
                    fpv_vec=edge_fpv_vec,
                    margin=float(highest_tally - v.tallies[candidate]),
                    candidates=group,
                )

        # All remaining hopefuls must be seated.
        elif len(v.key.hopefuls) + v.degree == self.m:
            edge_fpv_vec = self._edge_fpv_vec_from_cache(cache)
            for candidate in hopefuls:
                updated_wt_vec = wt_vec.copy()

                yield ChildProposal(
                    action=EdgeAction.FORCE_ELECT,
                    candidate=int(candidate),
                    transfer_value=0.0,
                    wt_vec=updated_wt_vec,
                    fpv_vec=edge_fpv_vec,
                    margin=0.0,
                )

        else:
            # Optional election edges for candidates within LAM of highest tally.
            highest_tally = np.max(v.tallies)
            elimination_margin_floor = float(max(highest_tally - self.quota, 0.0))
            if highest_tally > self.quota - self.LAM:
                edge_fpv_vec = self._edge_fpv_vec_from_cache(cache)
                if self.simultaneous:
                    group_iter = self._election_groups_within_lam(v, forced=False)
                else:
                    winner_idx_within_lam = hopefuls[
                        np.where(
                            v.tallies[hopefuls]
                            >= max(highest_tally, self.quota) - self.LAM
                        )[0]
                    ]
                    group_iter = tuple((int(candidate),) for candidate in winner_idx_within_lam)

                for group in group_iter:
                    updated_wt_vec, transfer_values = (
                        self._updated_wt_vec_for_election_group(group, cache, wt_vec)
                    )
                    group_margin = max(
                        max(highest_tally, self.quota) - min(v.tallies[list(group)]),
                        0.0,
                    )
                    action = (
                        EdgeAction.ELECT
                        if len(group) == 1
                        else EdgeAction.SIMULTANEOUS_ELECT
                    )

                    yield ChildProposal(
                        action=action,
                        candidate=int(group[0]),
                        transfer_value=transfer_values[0],
                        transfer_values=transfer_values,
                        wt_vec=updated_wt_vec,
                        fpv_vec=edge_fpv_vec,
                        margin=float(group_margin),
                        candidates=group,
                    )

            # Optional elimination edges for candidates within LAM of lowest tally.
            if len(hopefuls) > 0:
                lowest_tally = np.min(v.tallies[hopefuls])
                loser_idx_within_lam = hopefuls[
                    np.where(v.tallies[hopefuls] <= lowest_tally + self.LAM)[0]
                ]

                for candidate in loser_idx_within_lam:
                    margin = max(
                        v.tallies[candidate] - lowest_tally,
                        elimination_margin_floor,
                    )

                    yield ChildProposal(
                        action=EdgeAction.ELIMINATE,
                        candidate=int(candidate),
                        margin=float(margin),
                    )

    # -----------------------------------------------------------------
    # Child insertion
    # -----------------------------------------------------------------

    def _add_election_child(
        self,
        parent_ref: VertexRef,
        proposal: ChildProposal,
    ) -> VertexRef:
        if proposal.candidates is not None and len(proposal.candidates) > 1:
            return self._add_simultaneous_election_child(parent_ref, proposal)

        parent = self.vertex(parent_ref)
        edge_layer = parent_ref.layer

        # WIGM StateKey stores this incoming EdgeRef, so reserve it first.
        edge_ref = self._next_edge_ref(edge_layer)

        child_key, child_degree = self._make_child_key_and_degree(
            parent=parent,
            proposal=proposal,
            incoming_edge_ref=edge_ref,
        )

        # Election children are guaranteed new for WIGM because the child key
        # includes the reserved incoming seating edge.
        child_ref = self._create_vertex_no_dedupe(
            layer=parent_ref.layer + 1,
            key=child_key,
            degree=child_degree,
        )

        created_edge_ref, _ = self._add_edge(
            src=parent_ref,
            dst=child_ref,
            action=proposal.action,
            candidate=proposal.candidate,
            status=proposal.status,
            transfer_value=proposal.transfer_value,
            wt_vec=proposal.wt_vec,
            fpv_vec=proposal.fpv_vec,
            margin=proposal.margin,
            forced_ref=edge_ref,
            skip_dedupe=True,
        )

        assert created_edge_ref == edge_ref

        self.primary_parent_edge[child_ref] = edge_ref

        if self.vertex(child_ref).status != ElectionStatus.TERMINAL:
            self.pending_primary_children[parent_ref] += 1
            self._enqueue(child_ref)
        else:
            self._check_incoherent_leaf_tripwire(child_ref)

        return child_ref

    def _add_simultaneous_election_child(
        self,
        parent_ref: VertexRef,
        proposal: ChildProposal,
    ) -> VertexRef:
        if proposal.wt_vec is None:
            raise ValueError("Simultaneous election proposal is missing wt_vec.")
        if proposal.candidates is None or len(proposal.candidates) <= 1:
            raise ValueError(
                "Simultaneous election proposal must include multiple candidates."
            )

        parent = self.vertex(parent_ref)
        group = tuple(int(candidate) for candidate in proposal.candidates)
        edge_layer = parent_ref.layer

        first_edge_ref = self._next_edge_ref(edge_layer)
        edge_refs = [
            EdgeRef(edge_layer, first_edge_ref.local_id + offset)
            for offset in range(len(group))
        ]

        new_seated_at = list(parent.key.seated_at)
        for offset, edge_ref in enumerate(edge_refs):
            new_seated_at[parent.degree + offset] = edge_ref

        child_key = self._make_state_key(
            seated_at=tuple(new_seated_at),
            hopefuls=parent.key.hopefuls - set(group),
            parent_key=parent.key,
        )
        child_degree = parent.degree + len(group)
        child_ref = self._create_vertex_no_dedupe(
            layer=parent_ref.layer + len(group),
            key=child_key,
            degree=child_degree,
        )

        transfer_values = self._transfer_values_for_simultaneous_proposal(
            group=group,
            proposal=proposal,
        )

        for candidate, edge_ref, transfer_value in zip(
            group,
            edge_refs,
            transfer_values,
        ):
            created_edge_ref, _ = self._add_edge(
                src=parent_ref,
                dst=child_ref,
                action=EdgeAction.SIMULTANEOUS_ELECT,
                candidate=candidate,
                status=proposal.status,
                transfer_value=transfer_value,
                wt_vec=proposal.wt_vec,
                fpv_vec=proposal.fpv_vec,
                margin=proposal.margin,
                forced_ref=edge_ref,
                skip_dedupe=True,
            )
            assert created_edge_ref == edge_ref

        child = self.vertex(child_ref)
        child.path_multiplicity = parent.path_multiplicity
        self.primary_parent_edge[child_ref] = edge_refs[0]

        if child.status != ElectionStatus.TERMINAL:
            self.pending_primary_children[parent_ref] += 1
            self._enqueue(child_ref)
        else:
            self._check_incoherent_leaf_tripwire(child_ref)

        return child_ref

    def _transfer_values_for_simultaneous_proposal(
        self,
        *,
        group: tuple[int, ...],
        proposal: ChildProposal,
    ) -> tuple[float, ...]:
        if len(group) == 0:
            return ()

        if proposal.transfer_values is None:
            raise ValueError("Simultaneous election proposal is missing transfer values.")

        transfer_values = proposal.transfer_values
        if len(transfer_values) != len(group):
            raise ValueError("Simultaneous transfer value count does not match group.")

        return tuple(float(value) for value in transfer_values)

    def _make_child_key_and_degree(
        self,
        parent: ElectionState,
        proposal: ChildProposal,
        incoming_edge_ref: EdgeRef | None,
    ) -> tuple[StateKey, int]:
        if proposal.action == EdgeAction.ELIMINATE:
            return (
                self._make_state_key(
                    seated_at=parent.key.seated_at,
                    hopefuls=parent.key.hopefuls - {proposal.candidate},
                    parent_key=parent.key,
                ),
                parent.degree,
            )

        if proposal.action.is_election:
            new_seated_at = list(parent.key.seated_at)
            new_seated_at[parent.degree] = incoming_edge_ref

            return (
                self._make_state_key(
                    seated_at=tuple(new_seated_at),
                    hopefuls=parent.key.hopefuls - {proposal.candidate},
                    parent_key=parent.key,
                ),
                parent.degree + 1,
            )

        raise ValueError(f"Unknown proposal action: {proposal.action}")

    # -----------------------------------------------------------------
    # Seeded construction
    # -----------------------------------------------------------------

    def seeded_build(
        self,
        *,
        weak_candidates: Iterable[int] | None = None,
        strong_candidates: Iterable[int] | None = None,
        very_strong_candidates: Iterable[int] | None = None,
        already_elected: Sequence[Sequence[int | str]] | None = None,
        transfer_values: Sequence[Sequence[float]] | None = None,
        diagnostics: bool = True,
        diagnostic_interval: int = 1000,
    ):
        """
        Build forward from all viable seeds matching candidate filters.

        A seed excludes every weak and already-elected candidate from hopefuls,
        includes every strong candidate in hopefuls, and chooses any subset of
        the remaining candidates as hopeful. Seeds with too few hopeful
        candidates to fill the remaining seats are skipped.
        """
        if diagnostic_interval <= 0:
            raise ValueError("diagnostic_interval must be positive.")

        weak = (
            None
            if weak_candidates is None
            else frozenset(int(candidate) for candidate in weak_candidates)
        )
        strong = (
            None
            if strong_candidates is None
            else frozenset(int(candidate) for candidate in strong_candidates)
        )
        very_strong = frozenset(
            int(candidate)
            for candidate in (
                () if very_strong_candidates is None else very_strong_candidates
            )
        )

        if very_strong and already_elected is not None:
            raise ValueError(
                "very_strong_candidates cannot currently be combined with "
                "already_elected; use one pre-seating mechanism at a time."
            )
        if very_strong and transfer_values is not None:
            raise ValueError(
                "transfer_values cannot currently be combined with "
                "very_strong_candidates."
            )

        elected_groups = self._normalize_already_elected(already_elected)
        already_elected_set = frozenset(
            candidate
            for group in elected_groups
            for candidate in group
        ) | very_strong
        self._validate_seed_candidate_sets(
            frozenset() if weak is None else weak,
            frozenset() if strong is None else strong,
            already_elected_set,
        )
        self._reset_seeded_graph_storage()
        if diagnostics:
            self._print_seeded_build_diagnostics(
                "initialized",
                detail=(
                    f"memory_lite={self.memory_lite}, "
                    f"candidates={self.n_candidates}, ballots={len(self.root_wt_vec)}"
                ),
            )

        if very_strong:
            seeded_wt_vec, seated_at, elected_groups = (
                self._initialize_very_strong_seed(very_strong)
            )
            already_elected_set = frozenset(
                candidate
                for group in elected_groups
                for candidate in group
            )
        else:
            seeded_wt_vec, seated_at = self._initialize_already_elected_seed(
                elected_groups=elected_groups,
                transfer_values=transfer_values,
            )
        self.root_wt_vec = seeded_wt_vec
        initial_degree = len(already_elected_set)
        remaining_seats = self.m - initial_degree
        if diagnostics:
            self._print_seeded_build_diagnostics(
                "pre-seating complete",
                detail=(
                    f"already_elected={len(already_elected_set)}, "
                    f"remaining_seats={remaining_seats}"
                ),
            )

        if strong is None:
            strong, weak, frozen_mentions = search_strong_weak_candidates(
                self.ballot_matrix,
                seeded_wt_vec,
                self.n_candidates,
                remaining_seats=remaining_seats,
                LAM=float(self.LAM),
                quota=float(self.quota),
                masked_candidates=already_elected_set,
            )
        elif weak is None:
            weak, frozen_mentions = weak_candidates_from_strong(
                self.ballot_matrix,
                seeded_wt_vec,
                self.n_candidates,
                strong,
                LAM=float(self.LAM),
                quota=float(self.quota),
                verify_strong=True,
                masked_candidates=already_elected_set,
            )
        else:
            frozen_mentions = frozen_mentions_from_matrix(
                self.ballot_matrix,
                seeded_wt_vec,
                self.n_candidates,
                strong,
                masked_candidates=already_elected_set,
            )

        self._validate_seed_candidate_sets(weak, strong, already_elected_set)
        self.used_seeded_build = True
        self.seed_very_strong_candidates = already_elected_set & very_strong
        self.seed_strong_candidates = strong
        self.seed_weak_candidates = weak
        self.seed_frozen_mentions = frozen_mentions
        if diagnostics:
            self._print_seeded_build_diagnostics(
                "candidate partition complete",
                detail=f"strong={len(strong)}, weak={len(weak)}",
            )

        if remaining_seats == 0:
            flexible = ()
        else:
            flexible = tuple(
                candidate
                for candidate in range(self.n_candidates)
                if (
                    candidate not in weak
                    and candidate not in strong
                    and candidate not in already_elected_set
                )
            )

        minimum_flexible = max(remaining_seats - len(strong), 0)
        expected_seed_count = sum(
            comb(len(flexible), subset_size)
            for subset_size in range(minimum_flexible, len(flexible) + 1)
        )
        seed_report_interval = max(
            diagnostic_interval,
            max(expected_seed_count // 20, 1),
        )
        if diagnostics:
            self._print_seeded_build_diagnostics(
                "allocating seeds",
                detail=(
                    f"flexible={len(flexible)}, "
                    f"expected_viable_seeds={expected_seed_count}"
                ),
            )

        seed_refs: list[VertexRef] = []
        for subset_size in range(len(flexible) + 1):
            for subset in combinations(flexible, subset_size):
                hopefuls = strong | frozenset(subset)
                if len(hopefuls) < remaining_seats:
                    continue

                key = self._make_state_key(
                    seated_at=seated_at,
                    hopefuls=frozenset(hopefuls),
                )
                layer = self.n_candidates - len(hopefuls)
                ref, is_new = self._get_or_create_vertex(
                    layer=layer,
                    key=key,
                    degree=initial_degree,
                )

                if not is_new:
                    continue

                vertex = self.vertex(ref)
                vertex.path_multiplicity = 1
                if not self.memory_lite:
                    self.runtime_cache[ref] = self._materialize_cache_from_state(ref)
                self._enqueue(ref)
                seed_refs.append(ref)
                if (
                    diagnostics
                    and len(seed_refs) % seed_report_interval == 0
                ):
                    self._print_seeded_build_diagnostics(
                        "allocating seeds",
                        detail=(
                            f"created={len(seed_refs)}/{expected_seed_count}"
                        ),
                    )

        if not seed_refs:
            raise ValueError("seeded_build did not produce any viable seed vertices.")

        self._seed_refs = seed_refs
        if diagnostics:
            self._print_seeded_build_diagnostics(
                "seed allocation complete",
                detail=f"created={len(seed_refs)}",
            )
        root_key = self._make_root_key()
        root_local = self.layer_index[0].get(root_key) if self.layers else None
        self.root_ref = None if root_local is None else VertexRef(0, root_local)

        expanded_count = 0
        while self.stack:
            ref = self.stack.pop()
            v = self.vertex(ref)

            if v.status != ElectionStatus.UNEXPANDED:
                continue

            self._expand_vertex(ref)
            expanded_count += 1
            if diagnostics and expanded_count % diagnostic_interval == 0:
                self._print_seeded_build_diagnostics(
                    "expanding graph",
                    detail=f"expanded={expanded_count}",
                )

        if diagnostics:
            self._print_seeded_build_diagnostics(
                "construction complete",
                detail=f"expanded={expanded_count}",
            )

        self.coherence_checked = False
        self.tightest_margins_assigned = False
        self.terminal_vertices_by_winner_set = {}

        return self

    def _print_seeded_build_diagnostics(
        self,
        phase: str,
        *,
        detail: str | None = None,
    ) -> None:
        memory = self._seeded_build_array_memory()
        memory_text = ", ".join(
            f"{name}={size / (1024 ** 2):.1f} MiB"
            for name, size in memory.items()
        )
        rss = self._current_rss_mib()
        rss_text = "unknown" if rss is None else f"{rss:.1f} MiB"
        vertex_count = sum(len(layer) for layer in self.layers)
        edge_count = sum(len(layer) for layer in self.edge_layers)

        suffix = "" if detail is None else f"; {detail}"
        print(
            f"[seeded_build:{phase}] vertices={vertex_count}, edges={edge_count}, "
            f"stack={len(self.stack)}, runtime_caches={len(self.runtime_cache)}, "
            f"rss={rss_text}; {memory_text}{suffix}"
        )
        if (
            phase == "seed allocation complete"
            and not self.memory_lite
            and self.runtime_cache
        ):
            print(
                "[seeded_build:memory-note] Each seed currently retains a "
                "ballot-sized runtime cache. Use memory_lite=True to reconstruct "
                "these caches on demand instead."
            )

    def _seeded_build_array_memory(self) -> dict[str, int]:
        def unique_nbytes(arrays) -> int:
            seen: set[int] = set()
            total = 0
            for array in arrays:
                if not isinstance(array, np.ndarray):
                    continue
                identity = id(array)
                if identity in seen:
                    continue
                seen.add(identity)
                total += int(array.nbytes)
            return total

        profile_arrays = unique_nbytes(
            (self.ballot_matrix, self.root_wt_vec, self._unseeded_root_wt_vec)
        )
        stencil_arrays = unique_nbytes(self.candidate_stencils)
        cache_arrays = unique_nbytes(
            array
            for cache in self.runtime_cache.values()
            for array in (
                cache.bool_ballot_matrix,
                cache.pos_vec,
                cache.fpv_vec,
            )
        )
        edge_arrays = unique_nbytes(
            edge.wt_vec
            for layer in self.edge_layers
            for edge in layer
        )
        tally_arrays = unique_nbytes(
            vertex.tallies
            for layer in self.layers
            for vertex in layer
        )
        return {
            "profile": profile_arrays,
            "stencils": stencil_arrays,
            "caches": cache_arrays,
            "edge_weights": edge_arrays,
            "tallies": tally_arrays,
        }

    def _current_rss_mib(self) -> float | None:
        try:
            with open("/proc/self/status", encoding="ascii") as status_file:
                for line in status_file:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except OSError:
            return None
        return None

    def _initialize_very_strong_seed(
        self,
        very_strong: frozenset[int],
    ) -> tuple[
        NDArray[np.float64],
        tuple[EdgeRef | None, ...],
        tuple[tuple[int, ...], ...],
    ]:
        wt_vec = self._unseeded_root_wt_vec.copy()
        bool_ballot_matrix = np.ones_like(self.ballot_matrix, dtype=np.bool_)
        pos_vec = np.zeros(self.ballot_matrix.shape[0], dtype=np.int8)
        fpv_vec = self.ballot_matrix[np.arange(self.ballot_matrix.shape[0]), pos_vec]
        seated_at: list[EdgeRef | None] = [None] * self.m
        seat_idx = 0
        anchor_ref: VertexRef | None = None
        remaining = set(very_strong)
        elected_groups: list[tuple[int, ...]] = []

        while remaining and seat_idx < self.m:
            tallies = self._compute_tallies_from_fpv(fpv_vec, wt_vec)
            forced = [
                candidate
                for candidate in sorted(remaining)
                if tallies[candidate] > self.quota + self.LAM
            ]
            if not forced:
                break

            if anchor_ref is None:
                anchor_ref = self._make_already_elected_root_anchor()
            anchor = self.vertex(anchor_ref)
            anchor.tallies = tallies
            anchor.next_margin = self._next_margin_for_vertex(anchor, wt_vec)

            remaining_seats = self.m - seat_idx
            if self.simultaneous:
                ranked_forced = sorted(
                    forced,
                    key=lambda candidate: (-tallies[candidate], candidate),
                )
                if len(ranked_forced) > remaining_seats:
                    warnings.warn(
                        "More forced very-strong candidates than remaining seats; "
                        "pre-seating the highest current tallies only.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                group = tuple(sorted(ranked_forced[:remaining_seats]))
            else:
                highest = max(forced, key=lambda candidate: tallies[candidate])
                near_highest = [
                    candidate
                    for candidate in forced
                    if (
                        candidate != highest
                        and tallies[highest] - tallies[candidate] <= self.LAM
                    )
                ]
                if near_highest:
                    warnings.warn(
                        "Multiple very-strong candidates are forced and within "
                        "LAM of the maximum tally; pre-seating only the highest "
                        f"current tally candidate {highest}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                group = (int(highest),)

            group_transfer_values = self._compute_simultaneous_transfer_values(
                group=group,
                fpv_vec=fpv_vec,
                wt_vec=wt_vec,
            )
            for candidate, transfer_value in zip(group, group_transfer_values):
                wt_vec[fpv_vec == candidate] *= transfer_value

            for candidate in group:
                bool_ballot_matrix &= self.candidate_stencils[candidate]
                remaining.remove(candidate)

            child_ref, edge_refs = self._make_already_elected_group_vertex(
                parent_ref=anchor_ref,
                seated_at=seated_at,
                group=group,
                transfer_values=group_transfer_values,
                wt_vec=wt_vec,
                fpv_vec=fpv_vec,
                degree=seat_idx + len(group),
            )

            for edge_ref in edge_refs:
                seated_at[seat_idx] = edge_ref
                seat_idx += 1

            elected_groups.append(group)
            anchor_ref = child_ref
            pos_vec = bool_ballot_matrix.argmax(axis=1).astype(np.int8)
            fpv_vec = self.ballot_matrix[
                np.arange(self.ballot_matrix.shape[0]),
                pos_vec,
            ]
            child = self.vertex(anchor_ref)
            child.tallies = self._compute_tallies_from_fpv(fpv_vec, wt_vec)
            child.next_margin = self._next_margin_for_vertex(child, wt_vec)

        if remaining:
            warnings.warn(
                "Some very_strong_candidates were not pre-seated because their "
                f"current tallies did not force election: {sorted(remaining)}.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._seed_connector_ref = anchor_ref
        return wt_vec, tuple(seated_at), tuple(elected_groups)

    def _normalize_seed_elected_groups(
        self,
        *,
        very_strong_candidates: frozenset[int],
        already_elected: Sequence[Sequence[int | str]] | None,
        transfer_values: Sequence[Sequence[float]] | None,
    ) -> tuple[tuple[int, ...], ...]:
        user_groups = self._normalize_already_elected(already_elected)
        if not very_strong_candidates:
            return user_groups

        if transfer_values is not None:
            raise ValueError(
                "transfer_values cannot currently be combined with "
                "very_strong_candidates; pass the corresponding candidates via "
                "already_elected instead."
            )

        if self.simultaneous:
            very_strong_groups = (tuple(sorted(very_strong_candidates)),)
        else:
            very_strong_groups = tuple(
                (candidate,)
                for candidate in sorted(very_strong_candidates)
            )

        overlap = very_strong_candidates & frozenset(
            candidate
            for group in user_groups
            for candidate in group
        )
        if overlap:
            raise ValueError(
                "very_strong_candidates cannot overlap already_elected: "
                f"{sorted(overlap)}"
            )

        return very_strong_groups + user_groups

    def _validate_seed_candidate_sets(
        self,
        weak: frozenset[int],
        strong: frozenset[int],
        already_elected: frozenset[int] = frozenset(),
    ) -> None:
        invalid = sorted(
            candidate
            for candidate in weak | strong | already_elected
            if candidate < 0 or candidate >= self.n_candidates
        )
        if invalid:
            raise ValueError(f"seeded_build has invalid candidate indices: {invalid}")

        overlap = weak & strong
        if overlap:
            raise ValueError(
                "seeded_build candidates cannot be both weak and strong: "
                f"{sorted(overlap)}"
            )

        elected_overlap = already_elected & (weak | strong)
        if elected_overlap:
            raise ValueError(
                "seeded_build already_elected candidates cannot also be weak "
                f"or strong: {sorted(elected_overlap)}"
            )

        if len(already_elected) > self.m:
            raise ValueError(
                "seeded_build already_elected contains more candidates than "
                f"available seats: {len(already_elected)} > {self.m}"
            )

    def _normalize_already_elected(
        self,
        already_elected: Sequence[Sequence[int | str]] | None,
    ) -> tuple[tuple[int, ...], ...]:
        if already_elected is None:
            return ()

        candidate_to_index = {
            name: idx
            for idx, name in enumerate(self.candidate_names)
        }
        seen: set[int] = set()
        groups: list[tuple[int, ...]] = []

        for group in already_elected:
            normalized_group = []
            for candidate in group:
                if isinstance(candidate, str):
                    try:
                        candidate_idx = candidate_to_index[candidate]
                    except KeyError as exc:
                        raise ValueError(
                            f"Unknown already_elected candidate name: {candidate}"
                        ) from exc
                else:
                    candidate_idx = int(candidate)

                if candidate_idx in seen:
                    raise ValueError(
                        "already_elected candidates cannot be repeated: "
                        f"{candidate_idx}"
                    )

                seen.add(candidate_idx)
                normalized_group.append(candidate_idx)

            if not normalized_group:
                raise ValueError("already_elected cannot contain an empty group.")

            groups.append(tuple(normalized_group))

        return tuple(groups)

    def _initialize_already_elected_seed(
        self,
        *,
        elected_groups: tuple[tuple[int, ...], ...],
        transfer_values: Sequence[Sequence[float]] | None,
    ) -> tuple[NDArray[np.float64], tuple[EdgeRef | None, ...]]:
        if (
            transfer_values is not None
            and len(transfer_values) != len(elected_groups)
        ):
            raise ValueError(
                "transfer_values must have the same number of groups as "
                "already_elected."
            )

        wt_vec = self._unseeded_root_wt_vec.copy()
        bool_ballot_matrix = np.ones_like(self.ballot_matrix, dtype=np.bool_)
        pos_vec = np.zeros(self.ballot_matrix.shape[0], dtype=np.int8)
        fpv_vec = self.ballot_matrix[np.arange(self.ballot_matrix.shape[0]), pos_vec]
        seated_at: list[EdgeRef | None] = [None] * self.m
        seat_idx = 0
        anchor_ref: VertexRef | None = None

        for group_idx, group in enumerate(elected_groups):
            current_tallies = self._compute_tallies_from_fpv(fpv_vec, wt_vec)
            if transfer_values is None:
                group_transfer_values = self._compute_simultaneous_transfer_values(
                    group=group,
                    fpv_vec=fpv_vec,
                    wt_vec=wt_vec,
                )
            else:
                raw_group_transfer_values = transfer_values[group_idx]
                if len(raw_group_transfer_values) != len(group):
                    raise ValueError(
                        "Each transfer_values group must match the corresponding "
                        "already_elected group length."
                    )
                group_transfer_values = tuple(
                    float(transfer_value)
                    for transfer_value in raw_group_transfer_values
                )

            for candidate, transfer_value in zip(group, group_transfer_values):
                wt_vec[fpv_vec == candidate] *= transfer_value

            for candidate in group:
                bool_ballot_matrix &= self.candidate_stencils[candidate]

            if anchor_ref is None:
                anchor_ref = self._make_already_elected_root_anchor()
            anchor = self.vertex(anchor_ref)
            anchor.tallies = current_tallies
            anchor.next_margin = self._next_margin_for_vertex(anchor, wt_vec)

            child_ref, edge_refs = self._make_already_elected_group_vertex(
                parent_ref=anchor_ref,
                seated_at=seated_at,
                group=group,
                transfer_values=group_transfer_values,
                wt_vec=wt_vec,
                fpv_vec=fpv_vec,
                degree=seat_idx + len(group),
            )

            for edge_ref in edge_refs:
                seated_at[seat_idx] = edge_ref
                seat_idx += 1

            anchor_ref = child_ref

            pos_vec = bool_ballot_matrix.argmax(axis=1).astype(np.int8)
            fpv_vec = self.ballot_matrix[
                np.arange(self.ballot_matrix.shape[0]),
                pos_vec,
            ]
            child = self.vertex(anchor_ref)
            child.tallies = self._compute_tallies_from_fpv(fpv_vec, wt_vec)
            child.next_margin = self._next_margin_for_vertex(child, wt_vec)

        self._seed_connector_ref = anchor_ref

        return wt_vec, tuple(seated_at)

    def _make_already_elected_root_anchor(self) -> VertexRef:
        anchor_ref, _ = self._get_or_create_vertex(
            layer=0,
            key=self._make_root_key(),
            degree=0,
        )
        anchor = self.vertex(anchor_ref)
        anchor.status = ElectionStatus.EXPANDED
        anchor.path_multiplicity = 0
        return anchor_ref

    def _make_already_elected_group_vertex(
        self,
        *,
        parent_ref: VertexRef,
        seated_at: list[EdgeRef | None],
        group: tuple[int, ...],
        transfer_values: tuple[float, ...],
        wt_vec: NDArray[np.float64],
        fpv_vec: NDArray[np.integer] | None = None,
        degree: int,
    ) -> tuple[VertexRef, list[EdgeRef]]:
        parent = self.vertex(parent_ref)
        next_seated_at = list(seated_at)
        first_edge_ref = self._next_edge_ref(parent_ref.layer)
        edge_refs = [
            EdgeRef(parent_ref.layer, first_edge_ref.local_id + offset)
            for offset in range(len(group))
        ]

        for offset, edge_ref in enumerate(edge_refs):
            next_seated_at[parent.degree + offset] = edge_ref

        child_ref = self._create_vertex_no_dedupe(
            layer=parent_ref.layer + len(group),
            key=self._make_state_key(
                seated_at=tuple(next_seated_at),
                hopefuls=parent.key.hopefuls - set(group),
                parent_key=parent.key,
            ),
            degree=degree,
        )
        child = self.vertex(child_ref)
        child.status = ElectionStatus.EXPANDED
        child.path_multiplicity = 0

        action = (
            EdgeAction.FORCE_ELECT
            if len(group) == 1
            else EdgeAction.SIMULTANEOUS_ELECT
        )

        for candidate, transfer_value, edge_ref in zip(
            group,
            transfer_values,
            edge_refs,
        ):
            edge_ref, _ = self._add_edge(
                src=parent_ref,
                dst=child_ref,
                action=action,
                candidate=candidate,
                status=EdgeStatus.CANONICAL,
                transfer_value=transfer_value,
                wt_vec=wt_vec.copy(),
                fpv_vec=(
                    None
                    if fpv_vec is None or not self.store_fpv_vec
                    else fpv_vec.copy()
                ),
                margin=0.0,
                forced_ref=edge_ref,
                skip_dedupe=True,
            )

        child.path_multiplicity = 0
        return child_ref, edge_refs

    def _compute_simultaneous_transfer_values(
        self,
        *,
        group: tuple[int, ...],
        fpv_vec: NDArray[np.integer],
        wt_vec: NDArray[np.float64],
    ) -> tuple[float, ...]:
        transfer_values = []

        for candidate in group:
            tally = float(wt_vec[fpv_vec == candidate].sum())
            if tally == 0:
                raise ValueError(
                    "Cannot compute transfer value for already_elected candidate "
                    f"{candidate} with zero first-preference tally."
                )

            transfer_values.append(float((tally - self.quota) / tally))

        return tuple(transfer_values)

    def _reset_seeded_graph_storage(self) -> None:
        self.layers = []
        self.edge_layers = []
        self.layer_index = []
        self.edge_by_ref = {}
        self.edge_lookup = {}
        self.stack.clear()
        self.enqueued = set()
        self._deferred_enqueue_refs = None
        self.runtime_cache = {}
        self.primary_parent_edge = {}
        self.pending_primary_children.clear()
        self.root_ref = None
        self.terminal_vertices_by_winner_set = {}
        self.coherence_checked = False
        self.tightest_margins_assigned = False
        self._terminal_winner_set_tripwire = None
        self._pending_incoherent_leaf_error = None
        self._seed_refs = []
        self._seed_connector_ref = None
        self._initialize_graph_specific_storage()

    # -----------------------------------------------------------------
    # Post-build natural edges and tightest margins
    # -----------------------------------------------------------------

    def add_natural_edges(self) -> int:
        """
        Add missing natural elimination edges between adjacent layers.

        These are parent -> child edges where seated_at agrees and the child
        has exactly one fewer hopeful candidate.

        Since these edges were missed by construction, assign them margin LAM.
        Returns the number of new edges added.
        """
        n_added = 0

        for layer_idx, layer in enumerate(self.layers[:-1]):
            next_layer_idx = layer_idx + 1

            for parent in layer:
                candidates = self.same_seated_index[next_layer_idx].get(
                    self._same_seated_index_key(parent.key),
                    set(),
                )

                for child_ref in candidates:
                    child = self.vertex(child_ref)

                    diff = parent.key.hopefuls ^ child.key.hopefuls
                    if len(diff) != 1:
                        continue

                    if not child.key.hopefuls < parent.key.hopefuls:
                        continue

                    eliminated = int(next(iter(diff)))

                    _, edge_is_new = self._add_edge(
                        src=parent.ref,
                        dst=child_ref,
                        action=EdgeAction.ELIMINATE,
                        candidate=eliminated,
                        status=EdgeStatus.DEFAULT,
                        margin=float(self.LAM),
                    )

                    if edge_is_new:
                        n_added += 1
                        if (
                            parent.status == ElectionStatus.TERMINAL
                            and parent.degree < self.m
                        ):
                            parent.status = ElectionStatus.EXPANDED

        return n_added

    # -----------------------------------------------------------------
    # Post-Construction analysis and utilities
    # -----------------------------------------------------------------

    def _winner_set_for_vertex(self, v: ElectionState) -> frozenset[int]:
        winners = set()

        for edge_ref in v.key.seated_at:
            if edge_ref is None:
                continue

            edge = self.edge(edge_ref)
            winners.add(edge.candidate)

        return frozenset(winners)

    def _transfer_value_for_seating_edge(self, edge_ref: EdgeRef) -> float | None:
        """
        Return the transfer value associated with a seating edge.

        Prefer the stored edge.transfer_value. If missing, recompute as
        (tally - quota) / tally using the parent tallies.
        """
        edge = self.edge(edge_ref)

        if edge.transfer_value is not None:
            return edge.transfer_value

        parent = self.vertex(edge.src)

        if parent.tallies is None:
            return None

        tally = parent.tallies[edge.candidate]

        if tally == 0:
            return None

        return float((tally - self.quota) / tally)


    def _elected_summary_for_vertex(self, v: ElectionState) -> list[tuple[int, float | None]]:
        """
        Return [(candidate_index, transfer_value), ...] for candidates already seated
        at this vertex.
        """
        elected = []

        for edge_ref in v.key.seated_at:
            if edge_ref is None:
                continue

            edge = self.edge(edge_ref)
            tv = self._transfer_value_for_seating_edge(edge_ref)
            elected.append((edge.candidate, tv))

        return sorted(elected, key=lambda x: x[0])

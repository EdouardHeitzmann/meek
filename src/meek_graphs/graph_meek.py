from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Any, Iterable

import numpy as np
from numpy.typing import DTypeLike, NDArray

try:
    from ..election_graphs import AbstractGraphConstructor, ChildProposal
except ImportError:
    from election_graphs import AbstractGraphConstructor, ChildProposal

from .datatypes import (
    EdgeAction,
    EdgeRef,
    EdgeStatus,
    ElectionEdge,
    ElectionState,
    ElectionStatus,
    KeepFactorCalibrationCache,
    MeekExpansionContext,
    MeekRuntimeCache,
    MeekStateKey,
    VertexRef,
)


class MeekGraphConstructor(AbstractGraphConstructor):
    def __init__(
        self,
        profile,
        m: int,
        LAM: float,
        *,
        memory_lite: bool = False,
        trip_when_incoherent: bool = False,
        tolerance: float | None = 1e-6,
        epsilon: float | None = 1e-6,
        max_iterations: int | None = 500,
    ) -> None:
        self.tolerance = 1e-6 if tolerance is None else float(tolerance)
        self.epsilon = 1e-6 if epsilon is None else float(epsilon)
        self.max_iterations = 500 if max_iterations is None else int(max_iterations)
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
        self.max_winner_combination_length = min(
            self.m,
            self.ballot_matrix.shape[1] - 1,
            self.n_candidates,
        )
        self.permutation_matrix = permutation_matrix_constructor(
            self.m,
            self.max_winner_combination_length,
            dtype=np.dtype(np.int8),
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

    def _make_root_key(self) -> MeekStateKey:
        return MeekStateKey(
            elected_candidates=(),
            hopefuls=frozenset(range(self.n_candidates)),
        )

    def _make_root_cache(self) -> MeekRuntimeCache:
        bool_ballot_matrix = np.ones_like(self.ballot_matrix, dtype=np.bool_)
        pos_vec = np.zeros(bool_ballot_matrix.shape[0], dtype=np.int8)
        fpv_vec = self.ballot_matrix[np.arange(bool_ballot_matrix.shape[0]), pos_vec]
        winner_comb_vec = np.zeros(bool_ballot_matrix.shape[0], dtype=np.int64)
        winner_bitstring_vec = np.zeros(bool_ballot_matrix.shape[0], dtype=np.int64)
        keep_factor_seed = np.array([], dtype=np.float64)

        return MeekRuntimeCache(
            bool_ballot_matrix=bool_ballot_matrix,
            pos_vec=pos_vec,
            fpv_vec=fpv_vec,
            winner_comb_vec=winner_comb_vec,
            winner_bitstring_vec=winner_bitstring_vec,
            keep_factor_seed=keep_factor_seed,
        )

    # -----------------------------------------------------------------
    # Runtime cache handling
    # -----------------------------------------------------------------

    def _derive_child_cache(
        self,
        parent_cache: MeekRuntimeCache,
        incoming_edge: ElectionEdge,
    ) -> MeekRuntimeCache:
        bool_ballot_matrix = parent_cache.bool_ballot_matrix.copy()
        pos_vec = parent_cache.pos_vec.copy()
        fpv_vec = parent_cache.fpv_vec.copy()
        winner_comb_vec = parent_cache.winner_comb_vec.copy()
        winner_bitstring_vec = parent_cache.winner_bitstring_vec.copy()
        keep_factor_seed = parent_cache.keep_factor_seed.copy()

        if incoming_edge.action == EdgeAction.ELIMINATE:
            bool_ballot_matrix &= self.candidate_stencils[incoming_edge.candidate]
            needs_update = fpv_vec == incoming_edge.candidate
            if np.any(needs_update):
                pos_vec[needs_update] = bool_ballot_matrix[needs_update].argmax(axis=1)
                fpv_vec[needs_update] = self.ballot_matrix[
                    needs_update,
                    pos_vec[needs_update],
                ]

        elif incoming_edge.action.is_election:
            parent = self.vertex(incoming_edge.src)
            insertion_idx = self._elected_insert_index(
                parent.key.elected_candidates,
                incoming_edge.candidate,
            )
            keep_factor_seed = np.insert(keep_factor_seed, insertion_idx, 1.0)

            child_cache = self._materialize_cache_from_state(incoming_edge.dst)
            child_cache.keep_factor_seed = keep_factor_seed
            return child_cache

        else:
            raise ValueError(f"Unknown edge action: {incoming_edge.action}")

        child = self.vertex(incoming_edge.dst)
        elected_candidates = list(child.key.elected_candidates)

        if len(elected_candidates) > 0:
            self._update_winner_comb_vec(
                winner_comb_vec=winner_comb_vec,
                winner_bitstring_vec=winner_bitstring_vec,
                fpv_vec=fpv_vec,
                bool_ballot_matrix=bool_ballot_matrix,
                pos_vec=pos_vec,
                all_winners=elected_candidates,
            )

        return MeekRuntimeCache(
            bool_ballot_matrix=bool_ballot_matrix,
            pos_vec=pos_vec,
            fpv_vec=fpv_vec,
            winner_comb_vec=winner_comb_vec,
            winner_bitstring_vec=winner_bitstring_vec,
            keep_factor_seed=keep_factor_seed,
        )

    def _materialize_cache_from_state(self, ref: VertexRef) -> MeekRuntimeCache:
        v = self.vertex(ref)
        elected_candidates = set(v.key.elected_candidates)
        eliminated_candidates = (
            set(range(self.n_candidates)) - set(v.key.hopefuls) - elected_candidates
        )

        bool_ballot_matrix = np.ones_like(self.ballot_matrix, dtype=np.bool_)
        for candidate in eliminated_candidates:
            bool_ballot_matrix &= self.candidate_stencils[candidate]

        pos_vec = bool_ballot_matrix.argmax(axis=1).astype(np.int8)
        fpv_vec = self.ballot_matrix[np.arange(bool_ballot_matrix.shape[0]), pos_vec]
        winner_comb_vec = np.zeros(bool_ballot_matrix.shape[0], dtype=np.int64)
        winner_bitstring_vec = np.zeros(bool_ballot_matrix.shape[0], dtype=np.int64)

        self._update_winner_comb_vec(
            winner_comb_vec=winner_comb_vec,
            winner_bitstring_vec=winner_bitstring_vec,
            fpv_vec=fpv_vec,
            bool_ballot_matrix=bool_ballot_matrix,
            pos_vec=pos_vec,
            all_winners=list(v.key.elected_candidates),
        )

        if v.keep_factors is None:
            keep_factor_seed = np.ones(v.degree, dtype=np.float64)
        else:
            keep_factor_seed = v.keep_factors.copy()

        return MeekRuntimeCache(
            bool_ballot_matrix=bool_ballot_matrix,
            pos_vec=pos_vec,
            fpv_vec=fpv_vec,
            winner_comb_vec=winner_comb_vec,
            winner_bitstring_vec=winner_bitstring_vec,
            keep_factor_seed=keep_factor_seed,
        )

    # -----------------------------------------------------------------
    # Tally logic
    # -----------------------------------------------------------------

    def _expansion_context(
        self,
        ref: VertexRef,
        v: ElectionState,
        cache: MeekRuntimeCache,
        incoming_edge: ElectionEdge | None,
    ) -> MeekExpansionContext:
        return MeekExpansionContext(quota=0.0, iterations_used=0)

    def _compute_tallies(
        self,
        v: ElectionState,
        cache: MeekRuntimeCache,
        context: MeekExpansionContext,
    ) -> NDArray[np.float64]:
        tallies, keep_factors, quota, iterations = self._keep_factor_calibrator(
            initial_wt_vec=self.root_wt_vec,
            fpv_vec=cache.fpv_vec,
            winners=list(v.key.elected_candidates),
            winner_comb_vec=cache.winner_comb_vec,
            keep_factor_seed=cache.keep_factor_seed,
            vertex=v,
        )
        v.keep_factors = keep_factors
        v.quota = quota
        cache.keep_factor_seed = keep_factors.copy()
        context.quota = quota
        context.iterations_used = iterations
        return tallies

    def _next_margin_for_vertex(
        self,
        v: ElectionState,
        context: MeekExpansionContext,
    ) -> float | None:
        if v.tallies is None or v.degree == self.m:
            return None

        hopefuls = np.array(sorted(v.key.hopefuls), dtype=int)
        if len(hopefuls) == 0:
            return None

        hopeful_tallies = v.tallies[hopefuls]
        margins: list[float] = []

        if np.any(hopeful_tallies > context.quota + self.LAM):
            highest_tally = np.max(hopeful_tallies)
            margins.extend(
                float(highest_tally - v.tallies[candidate])
                for candidate in hopefuls
                if highest_tally - v.tallies[candidate] > self.LAM
            )
        elif len(v.key.hopefuls) + v.degree == self.m:
            return None
        else:
            highest_tally = np.max(hopeful_tallies)
            election_margins = np.maximum(
                np.maximum(highest_tally, context.quota) - hopeful_tallies,
                0.0,
            )
            margins.extend(float(m) for m in election_margins if m > self.LAM)

            lowest_tally = np.min(hopeful_tallies)
            elimination_margin_floor = float(max(highest_tally - context.quota, 0.0))
            elimination_margins = np.maximum(
                hopeful_tallies - lowest_tally,
                elimination_margin_floor,
            )
            margins.extend(float(m) for m in elimination_margins if m > self.LAM)

        if not margins:
            return None
        return min(margins)

    def _get_threshold(
        self,
        total_ballot_wt: float,
        *,
        floor: bool = False,
    ) -> float:
        fractional_quota = total_ballot_wt / (self.m + 1)
        if floor:
            return int(fractional_quota) + self.epsilon
        return fractional_quota + self.epsilon

    def _keep_factor_calibrator(
        self,
        *,
        initial_wt_vec: NDArray[np.float64],
        fpv_vec: NDArray[np.integer],
        winners: list[int],
        winner_comb_vec: NDArray[np.integer],
        keep_factor_seed: NDArray[np.float64],
        vertex: ElectionState,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, int]:
        if len(winners) == 0:
            exhausted_mask = fpv_vec < 0
            active_votes = float(initial_wt_vec[~exhausted_mask].sum())
            quota = self._get_threshold(active_votes, floor=False)
            keep_factors = np.array([], dtype=np.float64)
            tallies = np.bincount(
                fpv_vec[~exhausted_mask],
                weights=initial_wt_vec[~exhausted_mask],
                minlength=self.n_candidates,
            ).astype(np.float64)
            return tallies, keep_factors, quota, 0

        calibration_cache = self._build_keep_factor_calibration_cache(
            initial_wt_vec=initial_wt_vec,
            fpv_vec=fpv_vec,
            winner_comb_vec=winner_comb_vec,
        )

        keep_factors, winner_tallies, leftover_factor, quota, iterations = (
            self._iterate_keep_factors_from_cache(
                cache=calibration_cache,
                keep_factor_seed=keep_factor_seed,
                num_winners=len(winners),
                vertex=vertex,
            )
        )

        if np.any(keep_factors > 1.0):
            print(
                "Warning: Meek keep-factor calibration returned keep_factors > 1 "
                f"at vertex {vertex.ref}; hopefuls={sorted(vertex.key.hopefuls)}; "
                f"keep_factors={keep_factors}"
            )

        tallies = self._finalize_tallies_from_cache(
            cache=calibration_cache,
            winners=winners,
            winner_tallies=winner_tallies,
            leftover_factor_by_support=leftover_factor,
        )

        return tallies, keep_factors, quota, iterations

    def _build_keep_factor_calibration_cache(
        self,
        *,
        initial_wt_vec: NDArray[np.float64],
        fpv_vec: NDArray[np.integer],
        winner_comb_vec: NDArray[np.integer],
    ) -> KeepFactorCalibrationCache:
        support_comb_ids, ballot_to_support_row = np.unique(
            winner_comb_vec,
            return_inverse=True,
        )

        exhausted_mask = fpv_vec < 0

        support_total_mass = np.bincount(
            ballot_to_support_row,
            weights=initial_wt_vec,
            minlength=len(support_comb_ids),
        ).astype(np.float64)

        support_nonexhausted_mass = np.bincount(
            ballot_to_support_row[~exhausted_mask],
            weights=initial_wt_vec[~exhausted_mask],
            minlength=len(support_comb_ids),
        ).astype(np.float64)

        raw_view = self.permutation_matrix[support_comb_ids]

        valid_mask = raw_view >= 0
        permutation_lengths = valid_mask.sum(axis=1).astype(np.int8)

        sliced_winner_matrix = raw_view.copy()
        sliced_winner_matrix[~valid_mask] = 0

        return KeepFactorCalibrationCache(
            support_comb_ids=support_comb_ids,
            ballot_to_support_row=ballot_to_support_row,
            sliced_winner_matrix=sliced_winner_matrix,
            valid_mask=valid_mask,
            permutation_lengths=permutation_lengths,
            support_total_mass=support_total_mass,
            support_nonexhausted_mass=support_nonexhausted_mass,
            exhausted_mask=exhausted_mask,
            fpv_vec=fpv_vec,
            initial_wt_vec=initial_wt_vec,
        )

    def _iterate_keep_factors_from_cache(
        self,
        *,
        cache: KeepFactorCalibrationCache,
        keep_factor_seed: NDArray[np.float64],
        num_winners: int,
        vertex: ElectionState,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, int]:
        keep_factors = np.ones(num_winners, dtype=np.float64)
        seed_len = min(len(keep_factor_seed), num_winners)
        if seed_len > 0:
            keep_factors[:seed_len] = keep_factor_seed[:seed_len]

        P, Lmax = cache.sliced_winner_matrix.shape

        k_mat = np.zeros((P, Lmax), dtype=np.float64)
        prefix = np.ones((P, Lmax), dtype=np.float64)
        carry_before = np.zeros((P, Lmax), dtype=np.float64)
        contrib = np.zeros((P, Lmax), dtype=np.float64)

        winner_tallies = np.zeros(num_winners, dtype=np.float64)
        leftover_factor = np.ones(P, dtype=np.float64)

        nonempty_rows = cache.permutation_lengths > 0
        nonempty_idx = np.where(nonempty_rows)[0]

        previous_keep_factors = keep_factors.copy()
        quota = 0.0
        iterations_used = 0

        for iteration in range(self.max_iterations):
            iterations_used = iteration + 1
            previous_keep_factors[:] = keep_factors

            k_mat.fill(0.0)
            k_mat[cache.valid_mask] = keep_factors[
                cache.sliced_winner_matrix[cache.valid_mask]
            ]

            prefix.fill(1.0)
            prefix[cache.valid_mask] -= k_mat[cache.valid_mask]
            np.cumprod(prefix, axis=1, out=prefix)

            carry_before.fill(0.0)
            carry_before[:, 0] = 1.0
            if Lmax > 1:
                carry_before[:, 1:] = prefix[:, :-1]
            carry_before[~cache.valid_mask] = 0.0

            contrib[:] = carry_before
            contrib *= k_mat
            contrib *= cache.support_total_mass[:, None]
            contrib[~cache.valid_mask] = 0.0

            winner_tallies.fill(0.0)
            for j in range(Lmax):
                not_empty_position = cache.valid_mask[:, j]
                if np.any(not_empty_position):
                    np.add.at(
                        winner_tallies,
                        cache.sliced_winner_matrix[not_empty_position, j],
                        contrib[not_empty_position, j],
                    )

            leftover_factor.fill(1.0)
            if len(nonempty_idx) > 0:
                leftover_factor[nonempty_idx] = prefix[
                    nonempty_idx,
                    cache.permutation_lengths[nonempty_idx] - 1,
                ]

            active_votes = float(
                winner_tallies.sum()
                + np.dot(cache.support_nonexhausted_mass, leftover_factor)
            )
            quota = self._get_threshold(active_votes, floor=False)

            if np.all(winner_tallies <= quota + self.tolerance):
                break

            scale = np.ones_like(keep_factors)
            positive = winner_tallies > 0.0
            scale[positive] = np.minimum(quota / winner_tallies[positive], 1.0)
            keep_factors *= scale

        if iterations_used == self.max_iterations:
            print(
                "Warning: Meek keep-factor calibration used max_iterations "
                f"at vertex {vertex.ref}; hopefuls={sorted(vertex.key.hopefuls)}"
            )
            ratio = np.ones_like(keep_factors)
            nonzero_previous = previous_keep_factors != 0.0
            ratio[nonzero_previous] = keep_factors[nonzero_previous] / previous_keep_factors[
                nonzero_previous
            ]
            if np.any((ratio < 0.9) | (ratio > 1.1)):
                raise RuntimeError(
                    "Degenerate Meek calibration state: final keep-factor ratio "
                    "between the last two iterations is outside [0.9, 1.1]. "
                    f"vertex={vertex.ref}; hopefuls={sorted(vertex.key.hopefuls)}; "
                    f"elected_candidates={list(vertex.key.elected_candidates)}; "
                    f"ratio={ratio}; keep_factors={keep_factors}; "
                    f"previous_keep_factors={previous_keep_factors}"
                )

        return keep_factors, winner_tallies, leftover_factor, quota, iterations_used

    def _finalize_tallies_from_cache(
        self,
        *,
        cache: KeepFactorCalibrationCache,
        winners: list[int],
        winner_tallies: NDArray[np.float64],
        leftover_factor_by_support: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        actual_wt_vec = (
            cache.initial_wt_vec * leftover_factor_by_support[cache.ballot_to_support_row]
        )

        tallies = np.bincount(
            cache.fpv_vec[~cache.exhausted_mask],
            weights=actual_wt_vec[~cache.exhausted_mask],
            minlength=self.n_candidates,
        ).astype(np.float64)

        if len(winners) > 0:
            tallies[np.asarray(winners, dtype=int)] = winner_tallies

        return tallies

    # -----------------------------------------------------------------
    # Child proposal and insertion
    # -----------------------------------------------------------------

    def _propose_children(
        self,
        v: ElectionState,
        cache: MeekRuntimeCache,
        context: MeekExpansionContext,
    ) -> Iterable[ChildProposal]:
        if v.tallies is None:
            raise ValueError("Cannot propose children before tallies are computed.")

        hopefuls = np.array(sorted(v.key.hopefuls), dtype=int)
        if len(hopefuls) == 0:
            return

        hopeful_tallies = v.tallies[hopefuls]

        if np.any(hopeful_tallies > context.quota + self.LAM):
            highest_tally = np.max(hopeful_tallies)
            winner_idx_within_lam = hopefuls[
                np.where(hopeful_tallies >= highest_tally - self.LAM)[0]
            ]

            for candidate in winner_idx_within_lam:
                yield ChildProposal(
                    action=EdgeAction.FORCE_ELECT,
                    candidate=int(candidate),
                    margin=float(highest_tally - v.tallies[candidate]),
                )

        elif len(v.key.hopefuls) + v.degree == self.m:
            for candidate in hopefuls:
                yield ChildProposal(
                    action=EdgeAction.FORCE_ELECT,
                    candidate=int(candidate),
                    margin=0.0,
                )

        else:
            highest_tally = np.max(hopeful_tallies)
            elimination_margin_floor = float(max(highest_tally - context.quota, 0.0))
            if highest_tally > context.quota - self.LAM:
                winner_idx_within_lam = hopefuls[
                    np.where(
                        hopeful_tallies >= max(highest_tally, context.quota) - self.LAM
                    )[0]
                ]

                for candidate in winner_idx_within_lam:
                    yield ChildProposal(
                        action=EdgeAction.ELECT,
                        candidate=int(candidate),
                        margin=float(
                            max(max(highest_tally, context.quota) - v.tallies[candidate], 0.0)
                        ),
                    )

            lowest_tally = np.min(hopeful_tallies)
            loser_idx_within_lam = hopefuls[
                np.where(hopeful_tallies <= lowest_tally + self.LAM)[0]
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

    def _add_election_child(
        self,
        parent_ref: VertexRef,
        proposal: ChildProposal,
    ) -> VertexRef:
        parent = self.vertex(parent_ref)

        child_key, child_degree = self._make_child_key_and_degree(
            parent=parent,
            proposal=proposal,
            incoming_edge_ref=None,
        )

        child_ref, child_is_new = self._get_or_create_vertex(
            layer=parent_ref.layer + 1,
            key=child_key,
            degree=child_degree,
        )

        edge_ref, _ = self._add_edge(
            src=parent_ref,
            dst=child_ref,
            action=proposal.action,
            candidate=proposal.candidate,
            status=proposal.status,
            margin=proposal.margin,
        )

        if child_is_new:
            self.primary_parent_edge[child_ref] = edge_ref
            child = self.vertex(child_ref)
            if child.status != ElectionStatus.TERMINAL:
                self.pending_primary_children[parent_ref] += 1
                self._enqueue(child_ref)
            else:
                self._check_incoherent_leaf_tripwire(child_ref)

        return child_ref

    def _make_child_key_and_degree(
        self,
        parent: ElectionState,
        proposal: ChildProposal,
        incoming_edge_ref: Any,
    ) -> tuple[MeekStateKey, int]:
        if proposal.action == EdgeAction.ELIMINATE:
            return (
                MeekStateKey(
                    elected_candidates=parent.key.elected_candidates,
                    hopefuls=parent.key.hopefuls - {proposal.candidate},
                ),
                parent.degree,
            )

        if proposal.action.is_election:
            return (
                MeekStateKey(
                    elected_candidates=self._normalized_elected_candidates(
                        parent.key.elected_candidates,
                        proposal.candidate,
                    ),
                    hopefuls=parent.key.hopefuls - {proposal.candidate},
                ),
                parent.degree + 1,
            )

        raise ValueError(f"Unknown proposal action: {proposal.action}")

    # -----------------------------------------------------------------
    # Post-build natural edges
    # -----------------------------------------------------------------

    def add_natural_edges(self) -> int:
        """
        Add missing natural edges between adjacent layers.

        For Meek keys, adjacent natural edges are recognized when either:
        1. elected_candidates are identical and hopefuls differ by one candidate;
        2. hopefuls are identical and elected_candidates differ by one candidate.

        Since these edges were missed by construction, assign them margin LAM.
        Returns the number of new edges added.
        """
        n_added = 0

        for layer_idx, layer in enumerate(self.layers[:-1]):
            next_layer = self.layers[layer_idx + 1]

            for parent in layer:
                for child in next_layer:
                    if child.key == parent.key:
                        continue

                    same_elected = child.key.elected_candidates == parent.key.elected_candidates
                    same_hopefuls = child.key.hopefuls == parent.key.hopefuls

                    if same_elected:
                        diff = parent.key.hopefuls ^ child.key.hopefuls
                        if len(diff) != 1:
                            continue

                        if not child.key.hopefuls < parent.key.hopefuls:
                            continue

                        candidate = int(next(iter(diff)))
                        action = EdgeAction.ELIMINATE

                    elif same_hopefuls:
                        parent_elected = set(parent.key.elected_candidates)
                        child_elected = set(child.key.elected_candidates)
                        diff = parent_elected ^ child_elected

                        if len(diff) != 1:
                            continue

                        if not parent_elected < child_elected:
                            continue

                        candidate = int(next(iter(diff)))
                        action = EdgeAction.ELECT

                    else:
                        continue

                    _, edge_is_new = self._add_edge(
                        src=parent.ref,
                        dst=child.ref,
                        action=action,
                        candidate=candidate,
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
    # Seeded construction
    # -----------------------------------------------------------------

    def seeded_build(
        self,
        *,
        weak_candidates: Iterable[int],
        strong_candidates: Iterable[int],
    ):
        """
        Build forward from all viable degree-zero seeds matching candidate filters.

        A seed has no elected candidates, excludes every weak candidate from
        hopefuls, includes every strong candidate in hopefuls, and chooses any
        subset of the remaining candidates as hopeful. Seeds with too few
        hopeful candidates to fill all seats are skipped.
        """
        weak = frozenset(int(candidate) for candidate in weak_candidates)
        strong = frozenset(int(candidate) for candidate in strong_candidates)
        self._validate_seed_candidate_sets(weak, strong)
        self._reset_seeded_graph_storage()

        flexible = tuple(
            candidate
            for candidate in range(self.n_candidates)
            if candidate not in weak and candidate not in strong
        )

        seed_refs: list[VertexRef] = []
        for subset_size in range(len(flexible) + 1):
            for subset in combinations(flexible, subset_size):
                hopefuls = strong | frozenset(subset)
                if len(hopefuls) < self.m:
                    continue

                key = MeekStateKey(
                    elected_candidates=(),
                    hopefuls=frozenset(hopefuls),
                )
                layer = self.n_candidates - len(hopefuls)
                ref, is_new = self._get_or_create_vertex(
                    layer=layer,
                    key=key,
                    degree=0,
                )

                if not is_new:
                    continue

                vertex = self.vertex(ref)
                vertex.path_multiplicity = 1
                if not self.memory_lite:
                    self.runtime_cache[ref] = self._materialize_cache_from_state(ref)
                self._enqueue(ref)
                seed_refs.append(ref)

        if not seed_refs:
            raise ValueError("seeded_build did not produce any viable seed vertices.")

        self._seed_refs = seed_refs
        root_key = self._make_root_key()
        root_local = self.layer_index[0].get(root_key) if self.layers else None
        self.root_ref = None if root_local is None else VertexRef(0, root_local)

        while self.stack:
            ref = self.stack.pop()
            v = self.vertex(ref)

            if v.status != ElectionStatus.UNEXPANDED:
                continue

            self._expand_vertex(ref)

        self.coherence_checked = False
        self.tightest_margins_assigned = False
        self.terminal_vertices_by_winner_set = {}

        return self

    def _validate_seed_candidate_sets(
        self,
        weak: frozenset[int],
        strong: frozenset[int],
    ) -> None:
        invalid = sorted(
            candidate
            for candidate in weak | strong
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
        self._initialize_graph_specific_storage()

    # -----------------------------------------------------------------
    # Reverse construction
    # -----------------------------------------------------------------

    def reverse_build(self, winner_set: Iterable[int]):
        """
        Build a Meek graph backward from a target ordered winner set.

        The seed leaf is the fully resolved state in which every candidate has
        been removed from hopefuls and the supplied winners have been elected in
        the supplied order. Reverse traversal inserts ordinary forward edges
        from each discovered ancestor to its child. If the edge margin is below
        self.LAM, the ancestor is expanded backward; otherwise the ancestor is
        retained as a terminal frontier vertex.
        """
        print(f"Selected winners are: {[self.candidate_names[c] for c in winner_set]}")
        winners = self._normalized_elected_candidates(
            tuple(int(candidate) for candidate in winner_set),
        )
        self._validate_reverse_winner_set(winners)
        self._reset_reverse_graph_storage()

        leaf_key = MeekStateKey(
            elected_candidates=winners,
            hopefuls=frozenset(),
        )
        leaf_ref, _ = self._get_or_create_vertex(
            layer=self.n_candidates,
            key=leaf_key,
            degree=len(winners),
        )
        leaf = self.vertex(leaf_ref)
        leaf.status = ElectionStatus.TERMINAL
        if not self.memory_lite:
            self.runtime_cache[leaf_ref] = self._materialize_cache_from_state(leaf_ref)
        self._reverse_seed_leaf_ref = leaf_ref

        reverse_stack: list[VertexRef] = [leaf_ref]
        reverse_seen: set[VertexRef] = {leaf_ref}

        while reverse_stack:
            child_ref = reverse_stack.pop()
            child = self.vertex(child_ref)
            if self.memory_lite:
                child_cache = self._materialize_cache_from_state(child_ref)
            else:
                child_cache = self.runtime_cache[child_ref]

            for parent_key, action, candidate in self._reverse_parent_specs(child):
                parent_ref, parent_is_new = self._get_or_create_vertex(
                    layer=child_ref.layer - 1,
                    key=parent_key,
                    degree=len(parent_key.elected_candidates),
                )
                parent = self.vertex(parent_ref)

                if self.memory_lite:
                    parent_cache = self._derive_parent_cache(
                        child_cache=child_cache,
                        parent_ref=parent_ref,
                        action=action,
                        candidate=candidate,
                    )
                else:
                    if parent_ref not in self.runtime_cache:
                        self.runtime_cache[parent_ref] = self._derive_parent_cache(
                            child_cache=child_cache,
                            parent_ref=parent_ref,
                            action=action,
                            candidate=candidate,
                        )
                    parent_cache = self.runtime_cache[parent_ref]
                incoming_edge = self._primary_incoming_edge(parent_ref)
                context = self._expansion_context(
                    parent_ref,
                    parent,
                    parent_cache,
                    incoming_edge,
                )
                parent.tallies = self._compute_tallies(parent, parent_cache, context)
                parent.next_margin = self._next_margin_for_vertex(parent, context)

                proposal = self._reverse_edge_proposal(
                    parent=parent,
                    requested_action=action,
                    candidate=candidate,
                    context=context,
                )
                if proposal is None:
                    continue

                edge_ref, edge_is_new = self._add_edge(
                    src=parent_ref,
                    dst=child_ref,
                    action=proposal.action,
                    candidate=proposal.candidate,
                    status=proposal.status,
                    margin=proposal.margin,
                )

                self.primary_parent_edge.setdefault(child_ref, edge_ref)

                if proposal.margin is not None and proposal.margin < self.LAM:
                    if (
                        parent.status == ElectionStatus.TERMINAL
                        and parent.degree < self.m
                    ):
                        parent.status = ElectionStatus.UNEXPANDED

                    if parent_ref.layer > 0 and parent_ref not in reverse_seen:
                        reverse_stack.append(parent_ref)
                        reverse_seen.add(parent_ref)
                elif parent_is_new:
                    parent.status = ElectionStatus.TERMINAL

                if edge_is_new and parent.status != ElectionStatus.TERMINAL:
                    parent.status = ElectionStatus.EXPANDED

        self._finalize_reverse_build()
        return self

    def _validate_reverse_winner_set(self, winners: tuple[int, ...]) -> None:
        if len(winners) != self.m:
            raise ValueError(
                f"reverse_build expected {self.m} winners, got {len(winners)}."
            )

        if len(set(winners)) != len(winners):
            raise ValueError("reverse_build winner_set contains duplicates.")

        invalid = [
            candidate
            for candidate in winners
            if candidate < 0 or candidate >= self.n_candidates
        ]
        if invalid:
            raise ValueError(
                f"reverse_build winner_set has invalid candidates: {invalid}"
            )

    def _normalized_elected_candidates(
        self,
        elected_candidates: tuple[int, ...],
        added_candidate: int | None = None,
    ) -> tuple[int, ...]:
        if added_candidate is None:
            candidates = elected_candidates
        else:
            candidates = elected_candidates + (int(added_candidate),)

        return tuple(sorted(candidates))

    def _elected_insert_index(
        self,
        elected_candidates: tuple[int, ...],
        added_candidate: int,
    ) -> int:
        normalized = self._normalized_elected_candidates(
            elected_candidates,
            added_candidate,
        )
        return normalized.index(int(added_candidate))

    def _reset_reverse_graph_storage(self) -> None:
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
        self._reverse_seed_leaf_ref = None
        self._initialize_graph_specific_storage()

    def _reverse_parent_specs(
        self,
        child: ElectionState,
    ) -> Iterable[tuple[MeekStateKey, EdgeAction, int]]:
        child_elected = tuple(child.key.elected_candidates)
        child_elected_set = set(child_elected)
        child_hopefuls = set(child.key.hopefuls)

        if child.ref.layer == 0:
            return

        if child.degree < self.m:
            removed_candidates = (
                set(range(self.n_candidates)) - child_elected_set - child_hopefuls
            )
            for candidate in sorted(removed_candidates):
                yield (
                    MeekStateKey(
                        elected_candidates=child_elected,
                        hopefuls=frozenset(child_hopefuls | {candidate}),
                    ),
                    EdgeAction.ELIMINATE,
                    int(candidate),
                )

        for idx, candidate in enumerate(child_elected):
            parent_elected = child_elected[:idx] + child_elected[idx + 1:]
            yield (
                MeekStateKey(
                    elected_candidates=parent_elected,
                    hopefuls=frozenset(child_hopefuls | {candidate}),
                ),
                EdgeAction.ELECT,
                int(candidate),
            )

    def _derive_parent_cache(
        self,
        *,
        child_cache: MeekRuntimeCache,
        parent_ref: VertexRef,
        action: EdgeAction,
        candidate: int,
    ) -> MeekRuntimeCache:
        parent = self.vertex(parent_ref)
        bool_ballot_matrix = child_cache.bool_ballot_matrix.copy()

        if action == EdgeAction.ELIMINATE:
            bool_ballot_matrix |= ~self.candidate_stencils[candidate]
        elif action.is_election:
            bool_ballot_matrix |= ~self.candidate_stencils[candidate]
        else:
            raise ValueError(f"Unknown reverse action: {action}")

        return self._cache_from_bool_matrix_and_key(
            key=parent.key,
            bool_ballot_matrix=bool_ballot_matrix,
            keep_factor_seed=parent.keep_factors,
        )

    def _cache_from_bool_matrix_and_key(
        self,
        *,
        key: MeekStateKey,
        bool_ballot_matrix: NDArray[np.bool_],
        keep_factor_seed: NDArray[np.float64] | None,
    ) -> MeekRuntimeCache:
        normalized_bool_ballot_matrix = np.ones_like(
            self.ballot_matrix,
            dtype=np.bool_,
        )
        eliminated_candidates = (
            set(range(self.n_candidates))
            - set(key.hopefuls)
            - set(key.elected_candidates)
        )
        for eliminated_candidate in eliminated_candidates:
            normalized_bool_ballot_matrix &= self.candidate_stencils[
                eliminated_candidate
            ]

        pos_vec = normalized_bool_ballot_matrix.argmax(axis=1).astype(np.int8)
        fpv_vec = self.ballot_matrix[
            np.arange(normalized_bool_ballot_matrix.shape[0]),
            pos_vec,
        ]
        winner_comb_vec = np.zeros(
            normalized_bool_ballot_matrix.shape[0],
            dtype=np.int64,
        )
        winner_bitstring_vec = np.zeros(
            normalized_bool_ballot_matrix.shape[0],
            dtype=np.int64,
        )

        self._update_winner_comb_vec(
            winner_comb_vec=winner_comb_vec,
            winner_bitstring_vec=winner_bitstring_vec,
            fpv_vec=fpv_vec,
            bool_ballot_matrix=normalized_bool_ballot_matrix,
            pos_vec=pos_vec,
            all_winners=list(key.elected_candidates),
        )

        if keep_factor_seed is None:
            seed = np.ones(len(key.elected_candidates), dtype=np.float64)
        else:
            seed = keep_factor_seed.copy()

        return MeekRuntimeCache(
            bool_ballot_matrix=normalized_bool_ballot_matrix,
            pos_vec=pos_vec,
            fpv_vec=fpv_vec,
            winner_comb_vec=winner_comb_vec,
            winner_bitstring_vec=winner_bitstring_vec,
            keep_factor_seed=seed,
        )

    def _reverse_edge_proposal(
        self,
        *,
        parent: ElectionState,
        requested_action: EdgeAction,
        candidate: int,
        context: MeekExpansionContext,
    ) -> ChildProposal | None:
        if parent.tallies is None:
            raise ValueError("Cannot compute reverse edge before parent tallies.")

        if candidate not in parent.key.hopefuls:
            return None

        hopefuls = np.array(sorted(parent.key.hopefuls), dtype=int)
        if len(hopefuls) == 0:
            return None

        hopeful_tallies = parent.tallies[hopefuls]
        highest_tally = float(np.max(hopeful_tallies))
        elected_deficit_margin = self._elected_deficit_margin(parent, context)

        if requested_action.is_election:
            if len(parent.key.hopefuls) + parent.degree == self.m:
                return ChildProposal(
                    action=EdgeAction.FORCE_ELECT,
                    candidate=candidate,
                    status=(
                        EdgeStatus.NOT_FACTUAL
                        if elected_deficit_margin > self.LAM
                        else EdgeStatus.DEFAULT
                    ),
                    margin=elected_deficit_margin,
                )

            election_margin = float(
                max(
                    max(highest_tally, context.quota) - parent.tallies[candidate],
                    0.0,
                )
            )
            margin = max(election_margin, elected_deficit_margin)
            action = EdgeAction.ELECT
            if parent.tallies[candidate] > context.quota:
                action = EdgeAction.FORCE_ELECT
            status = (
                EdgeStatus.NOT_FACTUAL
                if margin > self.LAM
                else EdgeStatus.DEFAULT
            )

            return ChildProposal(
                action=action,
                candidate=candidate,
                status=status,
                margin=margin,
            )

        if requested_action == EdgeAction.ELIMINATE:
            if len(parent.key.hopefuls) + parent.degree == self.m:
                return None

            lowest_tally = float(np.min(hopeful_tallies))
            elimination_margin_floor = float(max(highest_tally - context.quota, 0.0))
            elimination_margin = max(
                float(parent.tallies[candidate] - lowest_tally),
                elimination_margin_floor,
            )
            margin = max(elimination_margin, elected_deficit_margin)
            return ChildProposal(
                action=EdgeAction.ELIMINATE,
                candidate=candidate,
                status=(
                    EdgeStatus.NOT_FACTUAL
                    if margin > self.LAM
                    else EdgeStatus.DEFAULT
                ),
                margin=float(margin),
            )

        raise ValueError(f"Unknown reverse action: {requested_action}")

    def _elected_deficit_margin(
        self,
        parent: ElectionState,
        context: MeekExpansionContext,
    ) -> float:
        if parent.tallies is None or not parent.key.elected_candidates:
            return 0.0

        elected = np.asarray(parent.key.elected_candidates, dtype=int)
        deficits = context.quota - parent.tallies[elected]
        if not np.any(deficits > 0.0):
            return 0.0

        return float(np.max(deficits))

    def _finalize_reverse_build(self) -> None:
        root_key = self._make_root_key()
        root_local = None
        if self.layers:
            root_local = self.layer_index[0].get(root_key)
        self.root_ref = None if root_local is None else VertexRef(0, root_local)

        for layer in self.layers:
            for vertex in layer:
                vertex.path_multiplicity = 0

        if self.root_ref is not None:
            self.vertex(self.root_ref).path_multiplicity = 1
            for edge_layer in self.edge_layers:
                for edge in edge_layer:
                    self.vertex(edge.dst).path_multiplicity += self.vertex(
                        edge.src
                    ).path_multiplicity

        self.coherence_checked = False
        self.tightest_margins_assigned = False
        self.terminal_vertices_by_winner_set = {}

    def reverse_coherence_check(self) -> bool:
        """
        Return False exactly when reverse construction reached the ordinary root.

        In reverse construction, the seed leaf is an alternate winner set. A
        coherent LAM is one where the reverse search fails to connect that leaf
        back to the all-hopeful root state.
        """
        root_included = self.root_ref is not None
        self.coherence_checked = True

        if root_included:
            print("Reverse coherence check failed.")
            print(f"  root reached: {self.vertex_label(self.root_ref)}")
            return False

        print("Reverse coherence check passed.")
        print("  root was not reached.")
        return True

    def find_reverse_upset_path(self) -> list[EdgeRef]:
        """
        Return one root-to-seed-leaf edge sequence after reverse incoherence.
        """
        if self.root_ref is None:
            raise ValueError("Reverse graph does not include the root.")

        target_ref = getattr(self, "_reverse_seed_leaf_ref", None)
        if target_ref is None:
            target_ref = self._infer_reverse_seed_leaf_ref()

        stack: list[tuple[VertexRef, list[EdgeRef]]] = [(self.root_ref, [])]
        visited: set[VertexRef] = set()

        while stack:
            ref, path = stack.pop()
            if ref == target_ref:
                print("Reverse upset path found.")
                print(f"  root vertex: {self.vertex_label(self.root_ref)}")
                print(f"  seed leaf:   {self.vertex_label(target_ref)}")
                print(f"  steps:       {len(path)}")
                return path

            if ref in visited:
                continue
            visited.add(ref)

            for edge in reversed(self.outgoing_edges(ref)):
                if edge.dst in visited:
                    continue
                stack.append((edge.dst, path + [edge.ref]))

        raise ValueError("No path from root to reverse seed leaf found.")

    def find_minimal_reverse_upset_path(self) -> list[EdgeRef]:
        return self.find_reverse_upset_path()

    def _infer_reverse_seed_leaf_ref(self) -> VertexRef:
        candidates = [
            vertex.ref
            for layer in self.layers
            for vertex in layer
            if vertex.degree == self.m and len(vertex.key.hopefuls) == 0
        ]

        if not candidates:
            raise ValueError("Could not infer reverse seed leaf.")

        return max(candidates, key=lambda ref: (ref.layer, ref.local_id))

    # -----------------------------------------------------------------
    # Winner combination updates
    # -----------------------------------------------------------------

    def _update_winner_comb_vec(
        self,
        *,
        winner_comb_vec: NDArray[np.integer],
        winner_bitstring_vec: NDArray[np.integer],
        fpv_vec: NDArray[np.integer],
        bool_ballot_matrix: NDArray[np.bool_],
        pos_vec: NDArray[np.integer],
        all_winners: list[int],
    ) -> None:
        if len(all_winners) == 0:
            return

        winner_array = np.asarray(all_winners, dtype=int)

        cand_to_winner_pos = np.full(self.n_candidates, -1, dtype=np.int32)
        cand_to_winner_pos[winner_array] = np.arange(winner_array.size, dtype=np.int32)

        max_passes = winner_array.size
        num_passes = 0

        while True:
            current_winner_pos = np.full(fpv_vec.shape, -1, dtype=np.int32)
            active_rows = fpv_vec >= 0
            current_winner_pos[active_rows] = cand_to_winner_pos[fpv_vec[active_rows]]

            needs_update = current_winner_pos >= 0
            if not np.any(needs_update):
                break

            num_passes += 1
            if num_passes > max_passes:
                print(np.nonzero(needs_update))
                raise RuntimeError(
                    "Winner-combination update exceeded the expected number of passes. "
                    "This suggests some ballots are repeatedly landing on seated winners "
                    "without making progress."
                )

            updated_comb, updated_bits = vectorized_perm_updater(
                winner_comb_vec[needs_update],
                self.m,
                self.max_winner_combination_length,
                winner_bitstring_vec[needs_update],
                current_winner_pos[needs_update],
            )
            winner_comb_vec[needs_update] = updated_comb
            winner_bitstring_vec[needs_update] = updated_bits

            bool_ballot_matrix[needs_update, pos_vec[needs_update]] = False

            pos_vec[needs_update] = bool_ballot_matrix[needs_update].argmax(axis=1)
            fpv_vec[needs_update] = self.ballot_matrix[
                needs_update,
                pos_vec[needs_update],
            ]

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def _winner_set_for_vertex(self, v: ElectionState) -> frozenset[int]:
        return frozenset(v.key.elected_candidates)


@lru_cache(maxsize=None)
def build_section_list(m: int, L: int) -> list[int]:
    section = 1
    section_list = [1]
    for k in range(L - 1, -1, -1):
        section = section * (m - k) + 1
        section_list.append(section)
    section_list.reverse()
    return section_list


def vectorized_perm_updater(
    winner_comb_vec: NDArray[np.integer],
    m: int,
    L: int,
    winner_bitstring_vec: NDArray[np.integer],
    winner_vec: NDArray[np.integer],
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    winner_mask_array = np.left_shift(1, winner_vec.astype(np.int64)) - 1
    truncated_winner_mask_array = np.bitwise_and(
        winner_mask_array,
        winner_bitstring_vec,
    )
    no_update_needed = np.bitwise_and(winner_mask_array + 1, winner_bitstring_vec) != 0
    update_needed = ~no_update_needed
    if np.any(no_update_needed):
        raise ValueError(
            "Winner-combination update attempted to add a winner already present "
            f"at row offsets {np.where(no_update_needed)[0]}."
        )
    sections = build_section_list(m, L)
    length_vec = np.bitwise_count(winner_bitstring_vec[update_needed])
    section_vec = np.array(sections)[length_vec + 1]
    shift_vec = np.bitwise_count(truncated_winner_mask_array[update_needed])
    return (
        winner_comb_vec + section_vec * (winner_vec - shift_vec) + 1,
        np.bitwise_or(winner_bitstring_vec, winner_mask_array + 1),
    )


def reverse_perm_updater(
    winner_comb_vec: NDArray[np.integer],
    m: int,
    L: int,
    winner_bitstring_vec: NDArray[np.integer],
    winner_vec: NDArray[np.integer],
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    winner_mask_array = np.left_shift(1, winner_vec.astype(np.int64)) - 1
    winner_bit_array = winner_mask_array + 1
    winner_present = np.bitwise_and(winner_bit_array, winner_bitstring_vec) != 0

    if np.any(~winner_present):
        raise ValueError(
            "Winner-combination reverse update attempted to remove a winner "
            f"not present at row offsets {np.where(~winner_present)[0]}."
        )

    updated_bitstring_vec = np.bitwise_and(
        winner_bitstring_vec,
        np.bitwise_not(winner_bit_array),
    )
    truncated_winner_mask_array = np.bitwise_and(
        winner_mask_array,
        updated_bitstring_vec,
    )

    sections = build_section_list(m, L)
    length_vec = np.bitwise_count(updated_bitstring_vec)
    section_vec = np.array(sections)[length_vec + 1]
    shift_vec = np.bitwise_count(truncated_winner_mask_array)
    updated_comb_vec = (
        winner_comb_vec
        - section_vec * (winner_vec - shift_vec)
        - 1
    )

    if np.any(updated_comb_vec < 0):
        raise ValueError(
            "Winner-combination reverse update produced a negative row id "
            f"at row offsets {np.where(updated_comb_vec < 0)[0]}."
        )

    return updated_comb_vec, updated_bitstring_vec


def permutation_matrix_constructor(
    m: int,
    L: int,
    sections: list[int] | None = None,
    dtype: DTypeLike | None = None,
) -> NDArray[np.integer]:
    if sections is None:
        sections = build_section_list(m, L)

    if len(sections) != L + 1:
        raise ValueError("sections must have length L+1")

    if dtype is None:
        out_dtype: np.dtype[Any] = np.dtype(
            np.int16 if m > np.iinfo(np.int8).max else np.int8
        )
    else:
        out_dtype = np.dtype(dtype)

    A = np.full((sections[0], L), -1, dtype=out_dtype)
    used = np.zeros(m, dtype=bool)

    def fill(start, depth):
        if depth == L:
            return

        section = sections[depth + 1]
        row = start + 1

        for x in range(m):
            if used[x]:
                continue

            A[row : row + section, depth] = x
            used[x] = True
            fill(row, depth + 1)
            used[x] = False

            row += section

    fill(0, 0)
    return A

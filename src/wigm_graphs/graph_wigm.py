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

import numpy as np
from numpy.typing import NDArray
from typing import Iterable
from collections import deque, defaultdict
from dataclasses import dataclass


@dataclass(slots=True)
class ChildProposal:
    """
    Produced by election logic when expanding a vertex.

    `margin` is edge-local: it is the LAM threshold needed for this parent
    to justify this child edge. It does not incorporate the parent's own
    tightest_margin.
    """
    action: EdgeAction
    candidate: int

    status: EdgeStatus = EdgeStatus.DEFAULT

    # Mainly for ELECT / FORCE_ELECT proposals.
    transfer_value: float | None = None
    wt_vec: NDArray[np.float64] | None = None

    # Edge-local margin.
    margin: float | None = None


class WIGMGraphConstructor:
    def __init__(
        self,
        profile,
        m: int,
        LAM: float,
        *,
        memory_lite: bool = False,
    ) -> None:
        self.profile = profile
        self.candidate_names = list(profile.candidates)
        self.n_candidates = len(self.candidate_names)
        self.m = m
        self.LAM = LAM
        self.memory_lite = memory_lite

        self.ballot_matrix, self.root_wt_vec = self._make_ballot_matrix(profile)
        self.candidate_stencils = self._make_candidate_stencils(self.ballot_matrix)

        # Droop quota.
        self.quota = np.floor(self.root_wt_vec.sum() / (self.m + 1)) + 1

        # Layered graph storage.
        self.layers: list[list[ElectionState]] = []
        self.edge_layers: list[list[ElectionEdge]] = []

        # Per-layer state-key deduplication.
        self.layer_index: list[dict[StateKey, int]] = []

        # Edge lookup / access.
        self.edge_by_ref: dict[EdgeRef, ElectionEdge] = {}
        self.edge_lookup: dict[
            tuple[VertexRef, VertexRef, EdgeAction, int],
            EdgeRef,
        ] = {}

        # DFS stack.
        self.stack: deque[VertexRef] = deque()
        self.enqueued: set[VertexRef] = set()

        # Cache reconstruction bookkeeping.
        self.runtime_cache: dict[VertexRef, RuntimeCache] = {}
        self.primary_parent_edge: dict[VertexRef, EdgeRef] = {}
        self.pending_primary_children: dict[VertexRef, int] = defaultdict(int)

        # Index for post-build natural-edge completion.
        self.same_seated_index: list[
            dict[tuple[EdgeRef | None, ...], set[VertexRef]]
        ] = []

        self.root_ref: VertexRef | None = None
        self.terminal_vertices_by_winner_set: dict[frozenset[int], list[VertexRef]] = {}
        self.coherence_checked: bool = False
        self.tightest_margins_assigned: bool = False

    # -----------------------------------------------------------------
    # Initial setup
    # -----------------------------------------------------------------

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

    def _make_root_cache(self) -> RuntimeCache:
        bool_ballot_matrix = np.ones_like(self.ballot_matrix, dtype=np.bool_)
        pos_vec = np.zeros(bool_ballot_matrix.shape[0], dtype=np.int8)
        fpv_vec = self.ballot_matrix[np.arange(bool_ballot_matrix.shape[0]), pos_vec]

        return RuntimeCache(
            bool_ballot_matrix=bool_ballot_matrix,
            pos_vec=pos_vec,
            fpv_vec=fpv_vec,
        )

    def initialize_root(self) -> VertexRef:
        if self.root_ref is not None:
            return self.root_ref

        root_key = StateKey(
            seated_at=(None,) * self.m,
            hopefuls=frozenset(range(self.n_candidates)),
        )

        root_ref, _ = self._get_or_create_vertex(
            layer=0,
            key=root_key,
            degree=0,
        )

        root = self.vertex(root_ref)
        root.path_multiplicity = 1
        root.tightest_margin = None

        self.runtime_cache[root_ref] = self._make_root_cache()

        self.root_ref = root_ref
        self._enqueue(root_ref)

        return root_ref

    # -----------------------------------------------------------------
    # Main DFS construction
    # -----------------------------------------------------------------

    def build(self) -> None:
        """
        Build vertices, proposal edges, tallies, and runtime caches.

        This does not add all post-hoc natural edges and does not assign
        final vertex tightest_margin values.
        """
        self.initialize_root()

        while self.stack:
            ref = self.stack.pop()
            v = self.vertex(ref)

            if v.status != ElectionStatus.UNEXPANDED:
                continue

            self._materialize_cache(ref)
            self._expand_vertex(ref)

    def _expand_vertex(self, ref: VertexRef) -> None:
        v = self.vertex(ref)
        print(f"Expanding vertex {ref} with degree {v.degree} and key {v.key}")

        cache = self.runtime_cache[ref]
        incoming_edge = self._primary_incoming_edge(ref)

        wt_vec = self._wt_vec_for_vertex(v, incoming_edge)
        v.tallies = self._compute_tallies(cache, wt_vec)

        proposals = list(self._propose_children(v, cache, wt_vec))

        for proposal in proposals:
            self._add_child_from_proposal(ref, proposal)

        if proposals:
            v.status = ElectionStatus.EXPANDED
        else:
            v.status = ElectionStatus.TERMINAL

        self._maybe_drop_cache(ref)

    # -----------------------------------------------------------------
    # Runtime cache handling
    # -----------------------------------------------------------------

    def _materialize_cache(self, ref: VertexRef) -> None:
        if ref in self.runtime_cache:
            return

        parent_edge_ref = self.primary_parent_edge[ref]
        parent_edge = self.edge(parent_edge_ref)
        parent_ref = parent_edge.src

        parent_cache = self.runtime_cache[parent_ref]

        self.runtime_cache[ref] = self._derive_child_cache(
            parent_cache=parent_cache,
            incoming_edge=parent_edge,
        )

        self.pending_primary_children[parent_ref] -= 1
        self._maybe_drop_cache(parent_ref)

    def _derive_child_cache(
        self,
        parent_cache: RuntimeCache,
        incoming_edge: ElectionEdge,
    ) -> RuntimeCache:
        bool_ballot_matrix = parent_cache.bool_ballot_matrix.copy()
        bool_ballot_matrix &= self.candidate_stencils[incoming_edge.candidate]

        needs_update = parent_cache.fpv_vec == incoming_edge.candidate

        pos_vec = parent_cache.pos_vec.copy()
        pos_vec[needs_update] = bool_ballot_matrix[needs_update].argmax(axis=1)

        fpv_vec = self.ballot_matrix[np.arange(bool_ballot_matrix.shape[0]), pos_vec]

        return RuntimeCache(
            bool_ballot_matrix=bool_ballot_matrix,
            pos_vec=pos_vec,
            fpv_vec=fpv_vec,
        )

    def _maybe_drop_cache(self, ref: VertexRef) -> None:
        if not self.memory_lite:
            return

        v = self.vertex(ref)

        if v.status == ElectionStatus.UNEXPANDED:
            return

        if self.pending_primary_children[ref] > 0:
            return

        self.runtime_cache.pop(ref, None)

    # -----------------------------------------------------------------
    # Weight-vector / tally logic
    # -----------------------------------------------------------------

    def _primary_incoming_edge(self, ref: VertexRef) -> ElectionEdge | None:
        edge_ref = self.primary_parent_edge.get(ref)
        if edge_ref is None:
            return None
        return self.edge(edge_ref)

    def _wt_vec_for_vertex(
        self,
        v: ElectionState,
        incoming_edge: ElectionEdge | None,
    ) -> NDArray[np.float64]:
        if incoming_edge is not None and incoming_edge.action in (
            EdgeAction.ELECT,
            EdgeAction.FORCE_ELECT,
        ):
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

    def _latest_seating_edge(self, key: StateKey) -> EdgeRef | None:
        seating_edges = [e for e in key.seated_at if e is not None]
        if not seating_edges:
            return None

        return max(seating_edges, key=lambda e: (e.layer, e.local_id))

    def _compute_tallies(
        self,
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

    # -----------------------------------------------------------------
    # Child proposal logic
    # -----------------------------------------------------------------

    def _propose_children(
        self,
        v: ElectionState,
        cache: RuntimeCache,
        wt_vec: NDArray[np.float64],
    ) -> Iterable[ChildProposal]:
        if v.tallies is None:
            raise ValueError("Cannot propose children before tallies are computed.")

        # Forced election: someone is safely above quota + LAM.
        if np.any(v.tallies > self.quota + self.LAM):
            highest_tally = np.max(v.tallies)
            winner_idx_within_lam = np.where(
                v.tallies >= highest_tally - self.LAM
            )[0]

            for candidate in winner_idx_within_lam:
                tally = v.tallies[candidate]
                transfer_value = (tally - self.quota) / tally

                updated_wt_vec = wt_vec.copy()
                updated_wt_vec[cache.fpv_vec == candidate] *= transfer_value

                yield ChildProposal(
                    action=EdgeAction.FORCE_ELECT,
                    candidate=int(candidate),
                    transfer_value=transfer_value,
                    wt_vec=updated_wt_vec,
                    margin=float(highest_tally - tally),
                )

        # All remaining hopefuls must be seated.
        elif len(v.key.hopefuls) + v.degree == self.m:
            for candidate in np.array(list(v.key.hopefuls), dtype=int):
                updated_wt_vec = wt_vec.copy()

                yield ChildProposal(
                    action=EdgeAction.FORCE_ELECT,
                    candidate=int(candidate),
                    transfer_value=0.0,
                    wt_vec=updated_wt_vec,
                    margin=0.0,
                )

        else:
            # Optional election edges for candidates within LAM of highest tally.
            highest_tally = np.max(v.tallies)
            if highest_tally > self.quota - self.LAM:
                winner_idx_within_lam = np.where(
                    (v.tallies >= max(highest_tally, self.quota) - self.LAM)
                )[0]

                for candidate in winner_idx_within_lam:
                    tally = v.tallies[candidate]
                    transfer_value = (tally - self.quota) / tally

                    updated_wt_vec = wt_vec.copy()
                    updated_wt_vec[cache.fpv_vec == candidate] *= transfer_value

                    yield ChildProposal(
                        action=EdgeAction.ELECT,
                        candidate=int(candidate),
                        transfer_value=transfer_value,
                        wt_vec=updated_wt_vec,
                        margin=float(max(self.quota - tally, 0.0)),
                    )

            # Optional elimination edges for candidates within LAM of lowest tally.
            hopefuls = np.array(list(v.key.hopefuls), dtype=int)

            if len(hopefuls) > 0:
                lowest_tally = np.min(v.tallies[hopefuls])
                loser_idx_within_lam = hopefuls[
                    np.where(v.tallies[hopefuls] <= lowest_tally + self.LAM)[0]
                ]

                for candidate in loser_idx_within_lam:
                    margin = v.tallies[candidate] - lowest_tally

                    yield ChildProposal(
                        action=EdgeAction.ELIMINATE,
                        candidate=int(candidate),
                        margin=float(margin),
                    )

    # -----------------------------------------------------------------
    # Child insertion
    # -----------------------------------------------------------------

    def _add_child_from_proposal(
        self,
        parent_ref: VertexRef,
        proposal: ChildProposal,
    ) -> VertexRef:
        if proposal.action in (EdgeAction.ELECT, EdgeAction.FORCE_ELECT):
            return self._add_election_child(parent_ref, proposal)

        return self._add_elimination_child(parent_ref, proposal)

    def _add_election_child(
        self,
        parent_ref: VertexRef,
        proposal: ChildProposal,
    ) -> VertexRef:
        parent = self.vertex(parent_ref)
        edge_layer = parent_ref.layer

        # Child StateKey stores this incoming EdgeRef, so reserve it first.
        edge_ref = self._next_edge_ref(edge_layer)

        child_key, child_degree = self._make_child_key_and_degree(
            parent=parent,
            proposal=proposal,
            incoming_edge_ref=edge_ref,
        )

        # Election children are guaranteed new.
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
            margin=proposal.margin,
            forced_ref=edge_ref,
            skip_dedupe=True,
        )

        assert created_edge_ref == edge_ref

        self.primary_parent_edge[child_ref] = edge_ref

        if self.vertex(child_ref).status != ElectionStatus.TERMINAL:
            self.pending_primary_children[parent_ref] += 1
            self._enqueue(child_ref)

        return child_ref

    def _add_elimination_child(
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
            self.pending_primary_children[parent_ref] += 1
            self._enqueue(child_ref)

        return child_ref

    def _make_child_key_and_degree(
        self,
        parent: ElectionState,
        proposal: ChildProposal,
        incoming_edge_ref: EdgeRef | None,
    ) -> tuple[StateKey, int]:
        if proposal.action == EdgeAction.ELIMINATE:
            return (
                StateKey(
                    seated_at=parent.key.seated_at,
                    hopefuls=parent.key.hopefuls - {proposal.candidate},
                ),
                parent.degree,
            )

        if proposal.action in (EdgeAction.ELECT, EdgeAction.FORCE_ELECT):
            new_seated_at = list(parent.key.seated_at)
            new_seated_at[parent.degree] = incoming_edge_ref

            return (
                StateKey(
                    seated_at=tuple(new_seated_at),
                    hopefuls=parent.key.hopefuls - {proposal.candidate},
                ),
                parent.degree + 1,
            )

        raise ValueError(f"Unknown proposal action: {proposal.action}")

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
                    parent.key.seated_at,
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
                        if parent.status == ElectionStatus.TERMINAL:
                            parent.status = ElectionStatus.EXPANDED

        return n_added

    def assign_tightest_margins(self) -> None:
        """
        Fill vertex.tightest_margin layer-by-layer from edge margins.

        For each incoming edge e: parent -> child, the value offered to child is

            max(e.margin, parent.tightest_margin)

        The child takes the minimum over all incoming edges.
        """
        if self.root_ref is None:
            raise ValueError("Cannot assign margins before root is initialized.")

        for layer in self.layers:
            for v in layer:
                v.tightest_margin = None

        root = self.vertex(self.root_ref)
        root.tightest_margin = 0.0

        for edge_layer in self.edge_layers:
            for edge in edge_layer:
                parent = self.vertex(edge.src)
                child = self.vertex(edge.dst)

                if parent.tightest_margin is None:
                    continue

                if edge.margin is None:
                    raise ValueError(f"Edge {edge.ref} has no margin assigned.")

                candidate_margin = max(edge.margin, parent.tightest_margin)

                if child.tightest_margin is None:
                    child.tightest_margin = candidate_margin
                else:
                    child.tightest_margin = min(
                        child.tightest_margin,
                        candidate_margin,
                    )

        self.tightest_margins_assigned = True

    # -----------------------------------------------------------------
    # Low-level graph storage
    # -----------------------------------------------------------------

    def _get_or_create_vertex(
        self,
        layer: int,
        key: StateKey,
        degree: int,
    ) -> tuple[VertexRef, bool]:
        self._ensure_layer(layer)

        existing = self.layer_index[layer].get(key)
        if existing is not None:
            return VertexRef(layer, existing), False

        ref = VertexRef(layer, len(self.layers[layer]))

        v = ElectionState(
            ref=ref,
            key=key,
            degree=degree,
            tightest_margin=None,
        )

        self.layers[layer].append(v)
        self.layer_index[layer][key] = ref.local_id
        self.same_seated_index[layer][key.seated_at].add(ref)

        return ref, True

    def _initial_status_for_degree(self, degree: int) -> ElectionStatus:
        if degree == self.m:
            return ElectionStatus.TERMINAL
        return ElectionStatus.UNEXPANDED

    def _create_vertex_no_dedupe(
        self,
        layer: int,
        key: StateKey,
        degree: int,
    ) -> VertexRef:
        self._ensure_layer(layer)

        if key in self.layer_index[layer]:
            raise ValueError("Election child unexpectedly already exists.")

        ref = VertexRef(layer, len(self.layers[layer]))

        v = ElectionState(
            ref=ref,
            key=key,
            degree=degree,
            status=self._initial_status_for_degree(degree),
            tightest_margin=None,
        )

        self.layers[layer].append(v)
        self.layer_index[layer][key] = ref.local_id
        self.same_seated_index[layer][key.seated_at].add(ref)

        return ref

    def _add_edge(
        self,
        *,
        src: VertexRef,
        dst: VertexRef,
        action: EdgeAction,
        candidate: int,
        status: EdgeStatus = EdgeStatus.DEFAULT,
        transfer_value: float | None = None,
        wt_vec: NDArray[np.float64] | None = None,
        margin: float | None = None,
        forced_ref: EdgeRef | None = None,
        skip_dedupe: bool = False,
    ) -> tuple[EdgeRef, bool]:
        lookup_key = (src, dst, action, candidate)

        if not skip_dedupe:
            existing = self.edge_lookup.get(lookup_key)
            if existing is not None:
                edge = self.edge(existing)
                edge.status = max(edge.status, status)

                if edge.margin is None and margin is not None:
                    edge.margin = margin

                return existing, False

        layer = src.layer
        self._ensure_edge_layer(layer)

        ref = forced_ref if forced_ref is not None else self._next_edge_ref(layer)

        if ref.local_id != len(self.edge_layers[layer]):
            raise ValueError("forced_ref is not the next available edge ref.")

        edge = ElectionEdge(
            ref=ref,
            src=src,
            dst=dst,
            action=action,
            candidate=candidate,
            status=status,
            transfer_value=transfer_value,
            wt_vec=wt_vec,
            margin=margin,
        )

        self.edge_layers[layer].append(edge)
        self.edge_by_ref[ref] = edge
        self.edge_lookup[lookup_key] = ref

        self.vertex(dst).path_multiplicity += self.vertex(src).path_multiplicity

        return ref, True

    def _next_edge_ref(self, layer: int) -> EdgeRef:
        self._ensure_edge_layer(layer)
        return EdgeRef(layer=layer, local_id=len(self.edge_layers[layer]))

    def _enqueue(self, ref: VertexRef) -> None:
        if ref in self.enqueued:
            return

        self.stack.append(ref)
        self.enqueued.add(ref)

    def _ensure_layer(self, layer: int) -> None:
        while len(self.layers) <= layer:
            self.layers.append([])
            self.layer_index.append({})
            self.same_seated_index.append(defaultdict(set))

    def _ensure_edge_layer(self, layer: int) -> None:
        while len(self.edge_layers) <= layer:
            self.edge_layers.append([])

    def vertex(self, ref: VertexRef) -> ElectionState:
        return self.layers[ref.layer][ref.local_id]

    def edge(self, ref: EdgeRef) -> ElectionEdge:
        return self.edge_by_ref[ref]
    
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


    def coherence_check(self) -> bool:
        """
        Check whether all TERMINAL vertices have the same unordered winner set.

        Also stores:
            self.terminal_vertices_by_winner_set
        """
        terminal_winner_sets: dict[frozenset[int], list[VertexRef]] = defaultdict(list)

        for layer in self.layers:
            for v in layer:
                if v.status != ElectionStatus.TERMINAL:
                    continue

                winner_set = self._winner_set_for_vertex(v)
                terminal_winner_sets[winner_set].append(v.ref)

        # Sort refs for stable reporting.
        self.terminal_vertices_by_winner_set = {
            winner_set: sorted(refs, key=lambda r: (r.layer, r.local_id))
            for winner_set, refs in terminal_winner_sets.items()
        }

        self.coherence_checked = True

        if not self.terminal_vertices_by_winner_set:
            print("Coherence check failed: no TERMINAL vertices found.")
            return False

        if len(self.terminal_vertices_by_winner_set) == 1:
            winner_set = next(iter(self.terminal_vertices_by_winner_set))
            winner_names = [self.candidate_names[i] for i in sorted(winner_set)]
            n_terminal = sum(len(refs) for refs in self.terminal_vertices_by_winner_set.values())

            print("Coherence check passed.")
            print(f"  terminal vertices: {n_terminal}")
            print(f"  winner set: {winner_names}")
            return True

        winner_sets = list(self.terminal_vertices_by_winner_set.keys())

        shared = set(winner_sets[0])
        for winner_set in winner_sets[1:]:
            shared &= set(winner_set)

        shared_names = [self.candidate_names[i] for i in sorted(shared)]

        print("Coherence check failed.")
        print(f"  distinct terminal winner sets: {len(self.terminal_vertices_by_winner_set)}")
        print(f"  candidates shared by all winner sets: {shared_names}")

        for winner_set, refs in self.terminal_vertices_by_winner_set.items():
            full_names = [self.candidate_names[i] for i in sorted(winner_set)]
            completion = sorted(set(winner_set) - shared)
            completion_names = [self.candidate_names[i] for i in completion]
            labels = [self.vertex_label(ref) for ref in refs]

            print()
            print(f"  winner set: {full_names}")
            print(f"    completion beyond shared core: {completion_names}")
            print(f"    terminal node count: {len(refs)}")
            print(f"    terminal nodes: {labels}")

        return False
    
    def _layer_alpha(self, layer: int) -> str:
        """
        Convert internal layer index to an Excel-style label.

        Internal layer 0 -> A
        Internal layer 1 -> B
        ...
        Internal layer 25 -> Z
        Internal layer 26 -> AA
        """
        n = layer + 1
        letters = []

        while n > 0:
            n, rem = divmod(n - 1, 26)
            letters.append(chr(ord("A") + rem))

        return "".join(reversed(letters))


    def _alpha_layer(self, alpha: str) -> int:
        """
        Inverse of _layer_alpha.
        """
        n = 0

        for ch in alpha.upper():
            if not ("A" <= ch <= "Z"):
                raise ValueError(f"Invalid layer label character: {ch}")
            n = 26 * n + (ord(ch) - ord("A") + 1)

        return n - 1


    def vertex_label(self, ref: VertexRef) -> str:
        """
        Canonical visual label for a vertex.
        """
        return f"{self._layer_alpha(ref.layer)}{ref.local_id}"


    def parse_vertex_label(self, label: str) -> VertexRef:
        """
        Parse labels like A0, B13, AA4 into VertexRef objects.
        """
        label = label.strip().upper()

        split = 0
        while split < len(label) and label[split].isalpha():
            split += 1

        if split == 0 or split == len(label):
            raise ValueError(f"Invalid vertex label: {label}")

        alpha = label[:split]
        local_id_str = label[split:]

        layer = self._alpha_layer(alpha)
        local_id = int(local_id_str)

        return VertexRef(layer=layer, local_id=local_id)


    def lookup_vertex(self, label: str) -> ElectionState:
        """
        Print and return the vertex corresponding to a canonical label like A0 or AA12.
        """
        ref = self.parse_vertex_label(label)
        v = self.vertex(ref)

        print(f"Vertex {label.upper()}")
        print(f"  ref: {v.ref}")
        print(f"  status: {v.status.name if hasattr(v.status, 'name') else v.status}")
        print(f"  degree: {v.degree}")
        print(f"  color: {v.color}")
        print(f"  path_multiplicity: {v.path_multiplicity}")
        print(f"  tightest_margin: {v.tightest_margin}")

        winners = []
        for edge_ref in v.key.seated_at:
            if edge_ref is None:
                continue
            edge = self.edge(edge_ref)
            winners.append((edge.candidate, self.candidate_names[edge.candidate], edge_ref))

        print("  seated winners:")
        for cand_idx, cand_name, edge_ref in winners:
            print(f"    {cand_idx}: {cand_name} via {edge_ref}")

        hopefuls = sorted(v.key.hopefuls)
        print("  hopefuls:")
        for cand_idx in hopefuls:
            print(f"    {cand_idx}: {self.candidate_names[cand_idx]}")

        if v.tallies is not None:
            print("  tallies:")
            for i, tally in enumerate(v.tallies):
                print(f"    {i}: {self.candidate_names[i]} -> {tally}")

        return v

    def _zero_margin_edge_from_parent(
        self,
        parent_ref: VertexRef,
        *,
        preferred_action: EdgeAction | None = None,
        tol: float = 1e-9,
    ) -> ElectionEdge | None:
        edges = self.outgoing_edges(parent_ref)

        zero_edges = [
            edge
            for edge in edges
            if edge.margin is not None and abs(edge.margin) <= tol
        ]

        if preferred_action is not None:
            same_action = [
                edge for edge in zero_edges
                if edge.action == preferred_action
            ]
            if same_action:
                zero_edges = same_action

        if not zero_edges:
            return None

        # Prefer canonical/normal if those statuses exist.
        canonical = [
            edge for edge in zero_edges
            if edge.status == EdgeStatus.CANONICAL
        ]
        if canonical:
            return canonical[0]

        normal = [
            edge for edge in zero_edges
            if edge.status == EdgeStatus.NORMAL
        ]
        if normal:
            return normal[0]

        return zero_edges[0]

    def incoming_edges(self, ref: VertexRef) -> list[ElectionEdge]:
        if ref.layer == 0:
            return []

        edge_layer_idx = ref.layer - 1
        if edge_layer_idx >= len(self.edge_layers):
            return []

        return [
            edge
            for edge in self.edge_layers[edge_layer_idx]
            if edge.dst == ref
        ]


    def outgoing_edges(self, ref: VertexRef) -> list[ElectionEdge]:
        if ref.layer >= len(self.edge_layers):
            return []

        return [
            edge
            for edge in self.edge_layers[ref.layer]
            if edge.src == ref
        ]


    def _action_word(self, action: EdgeAction) -> str:
        if action in (EdgeAction.ELECT, EdgeAction.FORCE_ELECT):
            return "elected"
        if action == EdgeAction.ELIMINATE:
            return "eliminated"
        return str(action)
    
    def _action_symbol(self, action: EdgeAction) -> str:
        if action in (EdgeAction.ELECT, EdgeAction.FORCE_ELECT):
            return "+"
        if action == EdgeAction.ELIMINATE:
            return "x"
        return "?"
    
    def _elected_indices_for_vertex(self, v: ElectionState) -> list[int]:
        elected = []

        for edge_ref in v.key.seated_at:
            if edge_ref is None:
                continue
            elected.append(self.edge(edge_ref).candidate)

        return sorted(elected)
    
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

    def find_minimal_upset_path(
        self,
        recorded_winner_set: set[int] | frozenset[int] | None = None,
        *,
        tol: float = 1e-9,
        verbose = False
    ) -> list[EdgeRef]:
        """
        Find a root-to-terminal path to a terminal vertex with a different winner
        set than the recorded one, minimizing terminal tightest_margin.

        If recorded_winner_set is None, infer it as the terminal winner set with
        the smallest tightest_margin. In normal use this should be the recorded
        winner set with margin 0.
        """
        if not self.coherence_checked:
            self.coherence_check()

        if not self.tightest_margins_assigned:
            self.assign_tightest_margins()

        if not self.terminal_vertices_by_winner_set:
            raise ValueError("No terminal vertices available.")

        # Infer recorded winner set if needed.
        if recorded_winner_set is None:
            best_recorded_key = None
            best_recorded_margin = float("inf")

            for winner_set, refs in self.terminal_vertices_by_winner_set.items():
                margins = [
                    self.vertex(ref).tightest_margin
                    for ref in refs
                    if self.vertex(ref).tightest_margin is not None
                ]

                if not margins:
                    continue

                winner_set_margin = min(margins)

                if winner_set_margin < best_recorded_margin:
                    best_recorded_margin = winner_set_margin
                    best_recorded_key = winner_set

            if best_recorded_key is None:
                raise ValueError("Could not infer recorded winner set.")

            recorded_winner_set = best_recorded_key
        else:
            recorded_winner_set = frozenset(recorded_winner_set)

        # Find terminal upset vertex with smallest tightest_margin.
        best_ref = None
        best_margin = float("inf")
        best_winner_set = None

        for winner_set, refs in self.terminal_vertices_by_winner_set.items():
            if winner_set == recorded_winner_set:
                continue

            for ref in refs:
                v = self.vertex(ref)

                if v.tightest_margin is None:
                    continue

                if v.tightest_margin < best_margin:
                    best_margin = v.tightest_margin
                    best_ref = ref
                    best_winner_set = winner_set

        if best_ref is None:
            raise ValueError("No upset terminal vertex found.")

        # Walk backwards from terminal to root.
        path_edges_reversed: list[EdgeRef] = []
        current_ref = best_ref

        while current_ref != self.root_ref:
            incoming = self.incoming_edges(current_ref)

            if not incoming:
                raise ValueError(f"Vertex {current_ref} has no incoming edges.")

            def edge_score(edge: ElectionEdge) -> float:
                parent = self.vertex(edge.src)

                if parent.tightest_margin is None:
                    return float("inf")

                if edge.margin is None:
                    return float("inf")

                return max(edge.margin, parent.tightest_margin)

            best_edge = min(incoming, key=edge_score)
            path_edges_reversed.append(best_edge.ref)
            current_ref = best_edge.src

        path_edges = list(reversed(path_edges_reversed))

        recorded_names = [
            self.candidate_names[i]
            for i in sorted(recorded_winner_set)
        ]
        upset_names = [
            self.candidate_names[i]
            for i in sorted(best_winner_set)
        ]

        print("Minimal upset path found.")
        print(f"  recorded winner set: {recorded_names}")
        print(f"  upset winner set:    {upset_names}")
        print(f"  terminal vertex:     {self.vertex_label(best_ref)}")
        print(f"  tightest_margin:     {best_margin}")

        print()
        print("Nonzero-margin steps:")

        any_nonzero = False

        for i, edge_ref in enumerate(path_edges, start=1):
            edge = self.edge(edge_ref)

            if edge.margin is None or abs(edge.margin) <= tol:
                continue

            any_nonzero = True

            parent_ref = edge.src
            comparison_edge = self._zero_margin_edge_from_parent(
                parent_ref,
                preferred_action=edge.action,
                tol=tol,
            )

            if verbose:
                x_name = self.candidate_names[edge.candidate]
                x_action = self._action_word(edge.action)

                if comparison_edge is None:
                    y_phrase = "the zero-margin candidate could not be identified"
                else:
                    y_name = self.candidate_names[comparison_edge.candidate]
                    y_action = self._action_word(comparison_edge.action)
                    y_phrase = f"candidate {y_name} should have been {y_action}"

                print(
                    f"  in round {i}, candidate {x_name} was {x_action}, "
                    f"whereas {y_phrase}, "
                    f"corresponding to an upset of {edge.margin} ballots"
                )

            else:
                x_sym = self._action_symbol(edge.action)
                x_idx = edge.candidate

                if comparison_edge is None:
                    y_part = "? ?"
                else:
                    y_sym = self._action_symbol(comparison_edge.action)
                    y_idx = comparison_edge.candidate
                    y_part = f"{y_sym} {y_idx}"

                parent = self.vertex(edge.src)

                hopefuls = sorted(parent.key.hopefuls)
                elected_summary = self._elected_summary_for_vertex(parent)

                print(
                    f"round {i}: {x_sym} {x_idx} instead of {y_part}. "
                    f"M = {edge.margin}"
                )
                print(f"  hopefuls = {hopefuls}")
                print(f"  elected = {elected_summary}")

        if not any_nonzero:
            print("  none")

        return path_edges
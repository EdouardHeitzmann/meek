from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .datatypes import (
    EdgeAction,
    EdgeRef,
    EdgeStatus,
    ElectionEdge,
    ElectionState,
    ElectionStatus,
    VertexRef,
)


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

    # Mainly for WIGM ELECT / FORCE_ELECT proposals.
    transfer_value: float | None = None
    transfer_values: tuple[float, ...] | None = None
    wt_vec: NDArray[np.float64] | None = None
    fpv_vec: NDArray[np.integer] | None = None
    candidates: tuple[int, ...] | None = None
    payload: Any = None

    # Edge-local margin.
    margin: float | None = None


class IncoherentLeafError(RuntimeError):
    """
    Raised when graph construction discovers terminal leaves with different
    winner sets while trip_when_incoherent is enabled.
    """

    def __init__(
        self,
        first_winner_set: frozenset[int],
        first_ref: VertexRef,
        second_winner_set: frozenset[int],
        second_ref: VertexRef,
    ) -> None:
        self.first_winner_set = first_winner_set
        self.first_ref = first_ref
        self.second_winner_set = second_winner_set
        self.second_ref = second_ref
        super().__init__(
            "Incoherent terminal leaves discovered: "
            f"{sorted(first_winner_set)} at {first_ref} and "
            f"{sorted(second_winner_set)} at {second_ref}."
        )


class AbstractGraphConstructor(ABC):
    """
    Shared layered-election-graph construction skeleton.

    Concrete graph families own state-key semantics, runtime-cache contents,
    tally computation, and proposal generation. This base class owns the
    deterministic graph traversal, vertex/edge storage, primary-parent cache
    reconstruction, and path-margin propagation.
    """

    def __init__(
        self,
        profile: Any,
        m: int,
        LAM: float,
        *,
        memory_lite: bool = False,
        trip_when_incoherent: bool = False,
    ) -> None:
        self.profile = profile
        self.candidate_names = list(profile.candidates)
        self.n_candidates = len(self.candidate_names)
        self.m = m
        self.LAM = LAM
        self.memory_lite = memory_lite
        self.trip_when_incoherent = trip_when_incoherent

        self._initialize_profile_data(profile)

        # Layered graph storage.
        self.layers: list[list[ElectionState]] = []
        self.edge_layers: list[list[ElectionEdge]] = []

        # Per-layer state-key deduplication.
        self.layer_index: list[dict[Any, int]] = []

        # Edge lookup / access.
        self.edge_by_ref: dict[EdgeRef, ElectionEdge] = {}
        self.edge_lookup: dict[
            tuple[VertexRef, VertexRef, EdgeAction, int],
            EdgeRef,
        ] = {}

        # DFS stack.
        self.stack: deque[VertexRef] = deque()
        self.enqueued: set[VertexRef] = set()
        self._deferred_enqueue_refs: list[VertexRef] | None = None

        # Cache reconstruction bookkeeping.
        self.runtime_cache: dict[VertexRef, Any] = {}
        self.primary_parent_edge: dict[VertexRef, EdgeRef] = {}
        self.pending_primary_children: dict[VertexRef, int] = defaultdict(int)

        self._initialize_graph_specific_storage()

        self.root_ref: VertexRef | None = None
        self.terminal_vertices_by_winner_set: dict[frozenset[int], list[VertexRef]] = {}
        self.coherence_checked: bool = False
        self.tightest_margins_assigned: bool = False
        self._terminal_winner_set_tripwire: tuple[frozenset[int], VertexRef] | None = None
        self._pending_incoherent_leaf_error: IncoherentLeafError | None = None

    def _has_numpy_profile_arrays(self, profile: Any) -> bool:
        return hasattr(profile, "ballot_matrix") and hasattr(profile, "wt_vec")

    def _make_padded_numpy_profile_arrays(
        self,
        profile: Any,
    ) -> tuple[NDArray[np.integer], NDArray[np.float64]]:
        mapped = np.asarray(profile.ballot_matrix)
        wt_vec = np.asarray(profile.wt_vec, dtype=np.float64)

        if mapped.ndim != 2:
            raise TypeError("NumpyRankProfile ballot_matrix must be two-dimensional.")

        if not np.issubdtype(mapped.dtype, np.integer):
            raise TypeError("NumpyRankProfile ballot_matrix must contain integers.")

        if wt_vec.ndim != 1:
            raise TypeError("NumpyRankProfile wt_vec must be one-dimensional.")

        if mapped.shape[0] != wt_vec.shape[0]:
            raise TypeError(
                "NumpyRankProfile ballot_matrix and wt_vec must have the same "
                "number of rows."
            )

        if np.any((mapped != -127) & ((mapped < 0) | (mapped >= self.n_candidates))):
            raise TypeError(
                "NumpyRankProfile ballot_matrix contains entries outside the "
                "candidate index range."
            )

        ballot_matrix: NDArray = np.full(
            (mapped.shape[0], mapped.shape[1] + 1),
            -127,
            dtype=np.int8,
        )
        ballot_matrix[:, : mapped.shape[1]] = mapped.astype(np.int8, copy=False)

        return ballot_matrix, wt_vec.copy()

    def __eq__(self, other: object) -> bool:
        """
        Compare graph constructors up to vertex and edge ref relabeling.

        Vertices are matched by identical state keys within the same layer.
        For each matched vertex, outgoing edges must have the same action,
        candidate, and matched destination state key.
        """
        if not isinstance(other, AbstractGraphConstructor):
            return NotImplemented

        if self.m != other.m or self.n_candidates != other.n_candidates:
            print(
                "Graph equality failed: constructors have different election sizes "
                f"(self: m={self.m}, n_candidates={self.n_candidates}; "
                f"other: m={other.m}, n_candidates={other.n_candidates})."
            )
            return False

        if len(self.layers) != len(other.layers):
            print(
                "Graph equality failed: constructors have different layer counts "
                f"(self: {len(self.layers)}, other: {len(other.layers)})."
            )
            return False

        for layer_idx, (self_layer, other_layer) in enumerate(
            zip(self.layers, other.layers)
        ):
            if len(self_layer) != len(other_layer):
                print(
                    "Graph equality failed: layer has different vertex counts "
                    f"(layer={layer_idx}, self={len(self_layer)}, "
                    f"other={len(other_layer)})."
                )
                return False

            self_keys = {
                self._state_key_signature(vertex.key)
                for vertex in self_layer
            }
            other_keys = {
                other._state_key_signature(vertex.key)
                for vertex in other_layer
            }

            if len(self_keys) != len(self_layer):
                print(
                    "Graph equality failed: duplicate vertex StateKey in self "
                    f"at layer {layer_idx}."
                )
                return False

            if len(other_keys) != len(other_layer):
                print(
                    "Graph equality failed: duplicate vertex StateKey in other "
                    f"at layer {layer_idx}."
                )
                return False

            if self_keys != other_keys:
                missing_from_other = self_keys - other_keys
                missing_from_self = other_keys - self_keys
                if missing_from_other:
                    print(
                        "Graph equality failed: vertex StateKey from self was not "
                        f"found in other at layer {layer_idx}: "
                        f"{next(iter(missing_from_other))}"
                    )
                else:
                    print(
                        "Graph equality failed: vertex StateKey from other was not "
                        f"found in self at layer {layer_idx}: "
                        f"{next(iter(missing_from_self))}"
                    )
                return False

        for layer_idx, self_layer in enumerate(self.layers):
            other_vertices_by_key = {
                other._state_key_signature(vertex.key): vertex
                for vertex in other.layers[layer_idx]
            }

            for self_vertex in self_layer:
                self_key = self._state_key_signature(self_vertex.key)
                other_vertex = other_vertices_by_key[self_key]

                self_signature = self._outgoing_edge_signature(self_vertex.ref)
                other_signature = other._outgoing_edge_signature(other_vertex.ref)

                if self_signature != other_signature:
                    print(
                        "Graph equality failed: outgoing edges differ for matched "
                        f"vertex at layer {layer_idx} with key {self_key}."
                    )
                    print(f"  self outgoing:  {self_signature}")
                    print(f"  other outgoing: {other_signature}")
                    return False

        return True

    def _outgoing_edge_signature(
        self,
        ref: VertexRef,
    ) -> tuple[tuple[int, int, int, Any], ...]:
        signature = []

        for edge in self.outgoing_edges(ref):
            dst = self.vertex(edge.dst)
            signature.append(
                (
                    int(edge.action),
                    int(edge.candidate),
                    int(edge.dst.layer),
                    self._state_key_signature(dst.key),
                )
            )

        return tuple(sorted(signature))

    def _state_key_signature(self, key: Any) -> Any:
        """
        Return the equality signature for a state key.

        Concrete graph families can override this when keys contain graph-local
        refs that should not participate directly in constructor equality.
        """
        return key

    # -----------------------------------------------------------------
    # Graph-family hooks
    # -----------------------------------------------------------------

    @abstractmethod
    def _initialize_profile_data(self, profile: Any) -> None:
        """Initialize concrete profile-derived arrays and constants."""

    def _initialize_graph_specific_storage(self) -> None:
        """Initialize concrete side indexes before any layers are created."""

    def _on_new_layer(self, layer: int) -> None:
        """Extend concrete per-layer indexes when a new vertex layer appears."""

    def _record_vertex_index(self, layer: int, key: Any, ref: VertexRef) -> None:
        """Record concrete side indexes for a newly-created vertex."""

    def _remap_key_refs(
        self,
        key: Any,
        edge_ref_map: dict[EdgeRef, EdgeRef],
    ) -> Any:
        """Remap graph-family references embedded in a state key."""
        return key

    @abstractmethod
    def _make_root_key(self) -> Any:
        """Return the graph-family root state key."""

    @abstractmethod
    def _make_root_cache(self) -> Any:
        """Return the graph-family root runtime cache."""

    @abstractmethod
    def _derive_child_cache(
        self,
        parent_cache: Any,
        incoming_edge: ElectionEdge,
    ) -> Any:
        """Derive a child's runtime cache from its primary parent edge."""

    @abstractmethod
    def _expansion_context(
        self,
        ref: VertexRef,
        v: ElectionState,
        cache: Any,
        incoming_edge: ElectionEdge | None,
    ) -> Any:
        """Compute concrete per-expansion context used by tallies/proposals."""

    @abstractmethod
    def _compute_tallies(
        self,
        v: ElectionState,
        cache: Any,
        context: Any,
    ) -> NDArray[np.float64]:
        """Compute candidate tallies for a vertex."""

    @abstractmethod
    def _propose_children(
        self,
        v: ElectionState,
        cache: Any,
        context: Any,
    ) -> Iterable[ChildProposal]:
        """Yield graph-family child proposals for an expanded vertex."""

    @abstractmethod
    def _next_margin_for_vertex(
        self,
        v: ElectionState,
        context: Any,
    ) -> float | None:
        """Return the smallest edge margin not currently included by self.LAM."""

    @abstractmethod
    def _materialize_cache_from_state(self, ref: VertexRef) -> Any:
        """Materialize a runtime cache directly from a vertex state key."""

    @abstractmethod
    def _make_child_key_and_degree(
        self,
        parent: ElectionState,
        proposal: ChildProposal,
        incoming_edge_ref: EdgeRef | None,
    ) -> tuple[Any, int]:
        """Return the child key and elected-candidate degree."""

    @abstractmethod
    def _winner_set_for_vertex(self, v: ElectionState) -> frozenset[int]:
        """Return the unordered winner set represented by a vertex."""

    # -----------------------------------------------------------------
    # Root setup and DFS construction
    # -----------------------------------------------------------------

    def initialize_root(self) -> VertexRef:
        if self.root_ref is not None:
            return self.root_ref

        root_ref, _ = self._get_or_create_vertex(
            layer=0,
            key=self._make_root_key(),
            degree=0,
        )

        root = self.vertex(root_ref)
        root.path_multiplicity = 1
        root.tightest_margin = None

        if not self.memory_lite:
            self.runtime_cache[root_ref] = self._make_root_cache()

        self.root_ref = root_ref
        self._enqueue(root_ref)

        return root_ref

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

            self._expand_vertex(ref)

    def _expand_vertex(self, ref: VertexRef) -> None:
        v = self.vertex(ref)
        #print(f"Expanding vertex {ref} with degree {v.degree} and key {v.key}")

        cache = self._cache_for_expansion(ref)
        incoming_edge = self._primary_incoming_edge(ref)

        context = self._expansion_context(ref, v, cache, incoming_edge)
        v.tallies = self._compute_tallies(v, cache, context)
        v.next_margin = self._next_margin_for_vertex(v, context)

        proposals = list(self._propose_children(v, cache, context))

        previous_deferred = self._deferred_enqueue_refs
        self._deferred_enqueue_refs = []
        try:
            for proposal in proposals:
                self._add_child_from_proposal(ref, proposal)
            deferred_enqueue_refs = self._deferred_enqueue_refs
        finally:
            self._deferred_enqueue_refs = previous_deferred

        for child_ref in reversed(deferred_enqueue_refs):
            self._enqueue_now(child_ref)

        if proposals:
            v.status = ElectionStatus.EXPANDED
        else:
            v.status = ElectionStatus.TERMINAL
            self._check_incoherent_leaf_tripwire(v.ref)

        self._maybe_drop_cache(ref)
        self._raise_pending_incoherent_leaf_error()

    def expand_margin(self, new_lam: float):
        """
        Expand this graph in-place from its current LAM to a larger LAM.

        Existing vertices are scanned from oldest layer to newest. Vertices
        whose stored next_margin is newly admitted by new_lam are re-expanded
        from a freshly materialized cache, and ordinary DFS then fills the new
        branches.
        """
        new_lam = float(new_lam)
        if new_lam < self.LAM:
            raise ValueError("expand_margin requires new_lam >= current LAM.")

        old_lam = float(self.LAM)
        self.LAM = new_lam

        for layer in list(self.layers):
            for vertex in list(layer):
                if vertex.status == ElectionStatus.TERMINAL:
                    continue

                if vertex.next_margin is None or vertex.next_margin > new_lam:
                    continue

                self._expand_vertex_margin_window(
                    vertex.ref,
                    old_lam=old_lam,
                    new_lam=new_lam,
                )

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

    def _expand_vertex_margin_window(
        self,
        ref: VertexRef,
        *,
        old_lam: float,
        new_lam: float,
    ) -> None:
        v = self.vertex(ref)
        cache = self._materialize_cache_from_state(ref)
        if not self.memory_lite:
            self.runtime_cache[ref] = cache

        incoming_edge = self._primary_incoming_edge(ref)
        context = self._expansion_context(ref, v, cache, incoming_edge)
        v.tallies = self._compute_tallies(v, cache, context)
        v.next_margin = self._next_margin_for_vertex(v, context)

        proposals = [
            proposal
            for proposal in self._propose_children(v, cache, context)
            if proposal.margin is None or old_lam < proposal.margin <= new_lam
        ]

        previous_deferred = self._deferred_enqueue_refs
        self._deferred_enqueue_refs = []
        try:
            for proposal in proposals:
                self._add_child_from_proposal(ref, proposal)
            deferred_enqueue_refs = self._deferred_enqueue_refs
        finally:
            self._deferred_enqueue_refs = previous_deferred

        for child_ref in reversed(deferred_enqueue_refs):
            self._enqueue_now(child_ref)

        if self.outgoing_edges(ref):
            v.status = ElectionStatus.EXPANDED
        else:
            v.status = ElectionStatus.TERMINAL
            self._check_incoherent_leaf_tripwire(v.ref)

        self._maybe_drop_cache(ref)
        self._raise_pending_incoherent_leaf_error()

    # -----------------------------------------------------------------
    # Runtime cache handling
    # -----------------------------------------------------------------

    def _materialize_cache(self, ref: VertexRef) -> None:
        if self.memory_lite:
            return

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

    def _cache_for_expansion(self, ref: VertexRef) -> Any:
        if self.memory_lite:
            return self._materialize_cache_from_state(ref)

        self._materialize_cache(ref)
        return self.runtime_cache[ref]

    def _maybe_drop_cache(self, ref: VertexRef) -> None:
        if self.memory_lite:
            self.runtime_cache.pop(ref, None)
            return

        v = self.vertex(ref)

        if v.status == ElectionStatus.UNEXPANDED:
            return

        if self.pending_primary_children[ref] > 0:
            return

        self.runtime_cache.pop(ref, None)

    def _primary_incoming_edge(self, ref: VertexRef) -> ElectionEdge | None:
        edge_ref = self.primary_parent_edge.get(ref)
        if edge_ref is None:
            return None
        return self.edge(edge_ref)

    # -----------------------------------------------------------------
    # Child insertion
    # -----------------------------------------------------------------

    def _add_child_from_proposal(
        self,
        parent_ref: VertexRef,
        proposal: ChildProposal,
    ) -> VertexRef:
        if proposal.action.is_election:
            return self._add_election_child(parent_ref, proposal)

        return self._add_elimination_child(parent_ref, proposal)

    @abstractmethod
    def _add_election_child(
        self,
        parent_ref: VertexRef,
        proposal: ChildProposal,
    ) -> VertexRef:
        """Insert an ELECT or FORCE_ELECT child edge and vertex."""

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

            if self.vertex(child_ref).status != ElectionStatus.TERMINAL:
                self.pending_primary_children[parent_ref] += 1
                self._enqueue(child_ref)
            else:
                self._check_incoherent_leaf_tripwire(child_ref)

        return child_ref

    # -----------------------------------------------------------------
    # Tightest margins
    # -----------------------------------------------------------------

    def assign_tightest_margins(self) -> None:
        """
        Fill vertex.tightest_margin layer-by-layer from edge margins.

        For each incoming edge e: parent -> child, the value offered to child is

            max(e.margin, parent.tightest_margin)

        The child takes the minimum over all incoming edges.
        """
        for layer in self.layers:
            for v in layer:
                v.tightest_margin = None

        source_refs = self._margin_source_refs()
        if not source_refs:
            raise ValueError("Cannot assign margins: graph has no source vertices.")

        for source_ref in source_refs:
            self.vertex(source_ref).tightest_margin = 0.0

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

    def _margin_source_refs(self) -> set[VertexRef]:
        """
        Return vertices that should act as zero-margin path sources.

        Ordinary constructions have one source, the root. Seeded constructions
        can have several source vertices, usually in different layers.
        """
        sources: set[VertexRef] = set()

        for layer in self.layers:
            for vertex in layer:
                if vertex.path_multiplicity <= 0:
                    continue

                if vertex.ref in self.primary_parent_edge:
                    continue

                sources.add(vertex.ref)

        if sources:
            return sources

        if self.root_ref is not None:
            return {self.root_ref}

        return set()

    def _check_incoherent_leaf_tripwire(self, ref: VertexRef) -> None:
        if not self.trip_when_incoherent:
            return

        vertex = self.vertex(ref)
        if vertex.status != ElectionStatus.TERMINAL:
            return

        winner_set = self._winner_set_for_vertex(vertex)

        if self._terminal_winner_set_tripwire is None:
            self._terminal_winner_set_tripwire = (winner_set, ref)
            return

        first_winner_set, first_ref = self._terminal_winner_set_tripwire
        if (
            winner_set != first_winner_set
            and self._pending_incoherent_leaf_error is None
        ):
            self._pending_incoherent_leaf_error = IncoherentLeafError(
                first_winner_set=first_winner_set,
                first_ref=first_ref,
                second_winner_set=winner_set,
                second_ref=ref,
            )

    def _raise_pending_incoherent_leaf_error(self) -> None:
        if self._pending_incoherent_leaf_error is None:
            return

        error = self._pending_incoherent_leaf_error
        self._pending_incoherent_leaf_error = None
        raise error

    def restrict_margin(self, new_lam: float):
        """
        Restrict a completed graph to edges with margin at most new_lam.

        This removes vertices no longer reachable from the root, compacts vertex
        and edge refs, rebuilds lookup indexes, and recomputes path
        multiplicities and tightest margins.
        """
        if self.root_ref is None:
            raise ValueError("Cannot restrict margin before root is initialized.")

        new_lam = float(new_lam)

        kept_old_edges: list[ElectionEdge] = []
        adjacency: dict[VertexRef, list[ElectionEdge]] = defaultdict(list)

        for edge_layer in self.edge_layers:
            for edge in edge_layer:
                if edge.margin is not None and edge.margin > new_lam:
                    continue

                kept_old_edges.append(edge)
                adjacency[edge.src].append(edge)

        reachable: set[VertexRef] = {self.root_ref}
        queue: deque[VertexRef] = deque([self.root_ref])

        while queue:
            ref = queue.popleft()

            for edge in adjacency.get(ref, []):
                if edge.dst in reachable:
                    continue

                reachable.add(edge.dst)
                queue.append(edge.dst)

        vertex_ref_map: dict[VertexRef, VertexRef] = {}
        new_layers: list[list[ElectionState]] = []

        for old_layer_idx, old_layer in enumerate(self.layers):
            new_layer: list[ElectionState] = []

            for old_vertex in old_layer:
                if old_vertex.ref not in reachable:
                    continue

                new_ref = VertexRef(
                    layer=old_layer_idx,
                    local_id=len(new_layer),
                )
                vertex_ref_map[old_vertex.ref] = new_ref
                old_vertex.ref = new_ref
                new_layer.append(old_vertex)

            if new_layer or old_layer_idx == 0:
                new_layers.append(new_layer)

        kept_old_edges = [
            edge
            for edge in kept_old_edges
            if edge.src in vertex_ref_map and edge.dst in vertex_ref_map
        ]

        edge_ref_map: dict[EdgeRef, EdgeRef] = {}
        edge_counts_by_layer: dict[int, int] = defaultdict(int)

        for old_edge in kept_old_edges:
            layer = old_edge.ref.layer
            new_ref = EdgeRef(
                layer=layer,
                local_id=edge_counts_by_layer[layer],
            )
            edge_counts_by_layer[layer] += 1
            edge_ref_map[old_edge.ref] = new_ref

        for layer in new_layers:
            for vertex in layer:
                vertex.key = self._remap_key_refs(vertex.key, edge_ref_map)

        new_edge_layer_count = 0
        if kept_old_edges:
            new_edge_layer_count = max(edge.ref.layer for edge in kept_old_edges) + 1

        new_edge_layers: list[list[ElectionEdge]] = [
            [] for _ in range(new_edge_layer_count)
        ]
        new_edge_by_ref: dict[EdgeRef, ElectionEdge] = {}
        new_edge_lookup: dict[tuple[VertexRef, VertexRef, EdgeAction, int], EdgeRef] = {}

        for edge in kept_old_edges:
            edge.ref = edge_ref_map[edge.ref]
            edge.src = vertex_ref_map[edge.src]
            edge.dst = vertex_ref_map[edge.dst]

            new_edge_layers[edge.ref.layer].append(edge)
            new_edge_by_ref[edge.ref] = edge
            new_edge_lookup[(edge.src, edge.dst, edge.action, edge.candidate)] = edge.ref

        self.layers = new_layers
        self.edge_layers = new_edge_layers
        self.edge_by_ref = new_edge_by_ref
        self.edge_lookup = new_edge_lookup

        self.layer_index = []
        self._initialize_graph_specific_storage()
        for layer_idx, layer in enumerate(self.layers):
            self.layer_index.append({})
            self._on_new_layer(layer_idx)

            for vertex in layer:
                self.layer_index[layer_idx][vertex.key] = vertex.ref.local_id
                self._record_vertex_index(layer_idx, vertex.key, vertex.ref)

        self.root_ref = vertex_ref_map[self.root_ref]
        self.primary_parent_edge = {}
        for edge_layer in self.edge_layers:
            for edge in edge_layer:
                self.primary_parent_edge.setdefault(edge.dst, edge.ref)

        outgoing_counts: dict[VertexRef, int] = defaultdict(int)
        for edge_layer in self.edge_layers:
            for edge in edge_layer:
                outgoing_counts[edge.src] += 1

        for layer in self.layers:
            for vertex in layer:
                vertex.path_multiplicity = 0
                if vertex.degree == self.m or outgoing_counts[vertex.ref] == 0:
                    vertex.status = ElectionStatus.TERMINAL
                else:
                    vertex.status = ElectionStatus.EXPANDED

        self.vertex(self.root_ref).path_multiplicity = 1
        for edge_layer in self.edge_layers:
            for edge in edge_layer:
                self.vertex(edge.dst).path_multiplicity += self.vertex(
                    edge.src
                ).path_multiplicity

        self.LAM = new_lam
        self.stack.clear()
        self.enqueued = set()
        self.pending_primary_children = defaultdict(int)
        self.runtime_cache = {}
        self.coherence_checked = False
        self.tightest_margins_assigned = False
        self.terminal_vertices_by_winner_set = {}
        self._terminal_winner_set_tripwire = None
        self._pending_incoherent_leaf_error = None

        for layer in self.layers:
            for vertex in layer:
                if vertex.status == ElectionStatus.TERMINAL:
                    vertex.next_margin = None
                    continue

                cache = self._materialize_cache_from_state(vertex.ref)
                if not self.memory_lite:
                    self.runtime_cache[vertex.ref] = cache
                incoming_edge = self._primary_incoming_edge(vertex.ref)
                context = self._expansion_context(
                    vertex.ref,
                    vertex,
                    cache,
                    incoming_edge,
                )
                vertex.tallies = self._compute_tallies(vertex, cache, context)
                vertex.next_margin = self._next_margin_for_vertex(vertex, context)
                self._maybe_drop_cache(vertex.ref)

        self.assign_tightest_margins()

        return self

    # -----------------------------------------------------------------
    # Low-level graph storage
    # -----------------------------------------------------------------

    def _get_or_create_vertex(
        self,
        layer: int,
        key: Any,
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
            status=self._initial_status_for_degree(degree),
            tightest_margin=None,
        )

        self.layers[layer].append(v)
        self.layer_index[layer][key] = ref.local_id
        self._record_vertex_index(layer, key, ref)

        return ref, True

    def _initial_status_for_degree(self, degree: int) -> ElectionStatus:
        if degree == self.m:
            return ElectionStatus.TERMINAL
        return ElectionStatus.UNEXPANDED

    def _create_vertex_no_dedupe(
        self,
        layer: int,
        key: Any,
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
        self._record_vertex_index(layer, key, ref)

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
        fpv_vec: NDArray[np.integer] | None = None,
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
            fpv_vec=fpv_vec,
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
        if self._deferred_enqueue_refs is not None:
            if ref in self.enqueued:
                return

            if ref in self._deferred_enqueue_refs:
                return

            self._deferred_enqueue_refs.append(ref)
            return

        self._enqueue_now(ref)

    def _enqueue_now(self, ref: VertexRef) -> None:
        if ref in self.enqueued:
            return

        self.stack.append(ref)
        self.enqueued.add(ref)

    def _ensure_layer(self, layer: int) -> None:
        while len(self.layers) <= layer:
            self.layers.append([])
            self.layer_index.append({})
            self._on_new_layer(len(self.layers) - 1)

    def _ensure_edge_layer(self, layer: int) -> None:
        while len(self.edge_layers) <= layer:
            self.edge_layers.append([])

    def vertex(self, ref: VertexRef) -> ElectionState:
        return self.layers[ref.layer][ref.local_id]

    def edge(self, ref: EdgeRef) -> ElectionEdge:
        return self.edge_by_ref[ref]

    # -----------------------------------------------------------------
    # Post-construction analysis
    # -----------------------------------------------------------------

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
            n_terminal = sum(
                len(refs) for refs in self.terminal_vertices_by_winner_set.values()
            )

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
        print(
            f"  distinct terminal winner sets: {len(self.terminal_vertices_by_winner_set)}"
        )
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
        if v.quota is not None:
            print(f"  quota: {v.quota}")

        print("  elected candidates:")
        for item in self._elected_summary_for_vertex(v):
            print(f"    {item}")

        hopefuls = sorted(v.key.hopefuls)
        print("  hopefuls:")
        for cand_idx in hopefuls:
            print(f"    {cand_idx}: {self.candidate_names[cand_idx]}")

        if v.tallies is not None:
            print("  tallies:")
            for i, tally in enumerate(v.tallies):
                print(f"    {i}: {self.candidate_names[i]} -> {tally}")

        if v.keep_factors is not None:
            print(f"  keep_factors: {v.keep_factors}")

        return v

    def lookup_edges_from(self, label: str) -> list[ElectionEdge]:
        """
        Print and return outgoing edges from the vertex with a canonical label.
        """
        ref = self.parse_vertex_label(label)
        edges = self.outgoing_edges(ref)

        print(f"Outgoing edges from {label.upper()}: {len(edges)}")
        for edge in edges:
            self._print_edge_summary(edge)

        return edges

    def lookup_edges_to(self, label: str) -> list[ElectionEdge]:
        """
        Print and return incoming edges to the vertex with a canonical label.
        """
        ref = self.parse_vertex_label(label)
        edges = self.incoming_edges(ref)

        print(f"Incoming edges to {label.upper()}: {len(edges)}")
        for edge in edges:
            self._print_edge_summary(edge)

        return edges

    def _print_edge_summary(self, edge: ElectionEdge) -> None:
        action = edge.action.name if hasattr(edge.action, "name") else str(edge.action)
        candidate_name = self.candidate_names[edge.candidate]

        print(
            f"  {edge.ref}: {self.vertex_label(edge.src)} -> "
            f"{self.vertex_label(edge.dst)}; "
            f"{action} {edge.candidate}: {candidate_name}; "
            f"status={edge.status.name}; margin={edge.margin}"
        )

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

    def _action_word(self, action: EdgeAction) -> str:
        if action.is_election:
            return "elected"
        if action == EdgeAction.ELIMINATE:
            return "eliminated"
        return str(action)

    def _action_symbol(self, action: EdgeAction) -> str:
        if action.is_election:
            return "+"
        if action == EdgeAction.ELIMINATE:
            return "x"
        return "?"

    def _elected_summary_for_vertex(self, v: ElectionState) -> list[Any]:
        return sorted(self._winner_set_for_vertex(v))

    def find_minimal_upset_path(
        self,
        recorded_winner_set: set[int] | frozenset[int] | None = None,
        *,
        tol: float = 1e-9,
        verbose: bool = False,
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

        source_refs = self._margin_source_refs()
        if not source_refs:
            raise ValueError("No source vertices available for upset path.")

        path_edges_reversed: list[EdgeRef] = []
        current_ref = best_ref

        while current_ref not in source_refs:
            incoming = self.incoming_edges(current_ref)

            if not incoming:
                raise ValueError(
                    f"Vertex {current_ref} has no incoming edges and is not a source."
                )

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

        source_ref = current_ref
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
        print(f"  source vertex:       {self.vertex_label(source_ref)}")
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

    # -----------------------------------------------------------------
    # Generic utilities
    # -----------------------------------------------------------------

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
        """Inverse of _layer_alpha."""
        n = 0

        for ch in alpha.upper():
            if not ("A" <= ch <= "Z"):
                raise ValueError(f"Invalid layer label character: {ch}")
            n = 26 * n + (ord(ch) - ord("A") + 1)

        return n - 1

    def vertex_label(self, ref: VertexRef) -> str:
        """Canonical visual label for a vertex."""
        return f"{self._layer_alpha(ref.layer)}{ref.local_id}"

    def parse_vertex_label(self, label: str) -> VertexRef:
        """Parse labels like A0, B13, AA4 into VertexRef objects."""
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

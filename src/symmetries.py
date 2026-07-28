from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from itertools import combinations
from typing import Any, Iterable

from .election_graphs.datatypes import EdgeRef, ElectionStatus, VertexRef


@dataclass(frozen=True, slots=True)
class QuotientStateKey:
    seated_at: tuple[EdgeRef | None, ...]
    hopeful_counts: tuple[int, ...]
    seed_id: int = -1

    @property
    def hopefuls(self) -> frozenset[int]:
        return frozenset(
            idx for idx, count in enumerate(self.hopeful_counts) if count > 0
        )


def _candidate_index(graph, candidate: int | str) -> int:
    if isinstance(candidate, str):
        try:
            return list(graph.candidate_names).index(candidate)
        except ValueError as exc:
            raise ValueError(f"Unknown candidate name: {candidate!r}") from exc

    idx = int(candidate)
    n_candidates = getattr(graph, "n_candidates", None)
    if n_candidates is None and hasattr(graph, "candidate_names"):
        n_candidates = len(graph.candidate_names)

    if n_candidates is not None and not 0 <= idx < n_candidates:
        raise ValueError(
            f"Candidate index {idx} is outside the graph candidate range."
        )

    return idx


def _graph_vertex(graph, ref: VertexRef):
    if hasattr(graph, "vertex"):
        return graph.vertex(ref)
    return graph.layers[ref.layer][ref.local_id]


def _graph_edge(graph, ref: EdgeRef):
    if hasattr(graph, "edge"):
        return graph.edge(ref)
    return graph.edge_by_ref[ref]


def _swap_candidate(candidate: Any, a: int, b: int) -> Any:
    if candidate == a:
        return b
    if candidate == b:
        return a
    return candidate


def _swap_candidate_collection(values: Iterable[Any], a: int, b: int) -> frozenset[Any]:
    return frozenset(_swap_candidate(value, a, b) for value in values)


def _key_signature(key: Any, swap_pair: tuple[int, int] | None = None) -> Any:
    """
    Return a hashable state-key signature.

    Candidate swaps are intentionally applied only to the hopeful set. The
    seated_at tuple, including its edge refs, is preserved exactly.
    """
    a = b = None
    if swap_pair is not None:
        a, b = swap_pair

    if is_dataclass(key):
        items = []
        for field in fields(key):
            value = getattr(key, field.name)
            if field.name == "hopefuls" and swap_pair is not None:
                value = _swap_candidate_collection(value, a, b)
            items.append((field.name, value))
        return type(key), tuple(items)

    seated_at = getattr(key, "seated_at", None)
    hopefuls = getattr(key, "hopefuls", None)
    if hopefuls is not None and swap_pair is not None:
        hopefuls = _swap_candidate_collection(hopefuls, a, b)

    extras = []
    for name in ("seed_id",):
        if hasattr(key, name):
            extras.append((name, getattr(key, name)))

    return type(key), seated_at, hopefuls, tuple(extras)


def _swap_graph_signature_builder(graph, swap_pair: tuple[int, int] | None):
    vertex_memo = {}
    edge_memo = {}

    def vertex_signature(ref: VertexRef):
        if ref in vertex_memo:
            return vertex_memo[ref]

        vertex = _graph_vertex(graph, ref)
        key = vertex.key
        if not is_dataclass(key):
            raise TypeError("candidate equivalence expects dataclass state keys.")

        items = []
        for field in fields(key):
            value = getattr(key, field.name)
            if field.name == "hopefuls" and swap_pair is not None:
                value = _swap_candidate_collection(
                    value,
                    swap_pair[0],
                    swap_pair[1],
                )
            elif field.name == "seated_at":
                value = tuple(
                    None if edge_ref is None else edge_signature(edge_ref)
                    for edge_ref in value
                )
            items.append((field.name, value))

        signature = type(key), tuple(items)
        vertex_memo[ref] = signature
        return signature

    def edge_signature(ref: EdgeRef):
        if ref in edge_memo:
            return edge_memo[ref]

        edge = _graph_edge(graph, ref)
        candidate = edge.candidate
        if swap_pair is not None:
            candidate = _swap_candidate(candidate, swap_pair[0], swap_pair[1])

        signature = (
            vertex_signature(edge.src),
            int(edge.action),
            int(candidate),
        )
        edge_memo[ref] = signature
        return signature

    return vertex_signature


def _layer_signature_sets(graph, swap_pair: tuple[int, int] | None = None):
    vertex_signature = _swap_graph_signature_builder(graph, swap_pair)
    return [
        {vertex_signature(vertex.ref) for vertex in layer}
        for layer in graph.layers
    ]


def _pair_is_equivalent(
    graph,
    layer_signature_sets: list[set[Any]],
    pair: tuple[int, int],
) -> bool:
    swapped_layer_signature_sets = _layer_signature_sets(graph, swap_pair=pair)
    for layer_idx, layer in enumerate(graph.layers):
        if swapped_layer_signature_sets[layer_idx] != layer_signature_sets[layer_idx]:
            return False

    return True


def _vertex_label(graph, ref: VertexRef) -> str:
    if hasattr(graph, "vertex_label"):
        return graph.vertex_label(ref)
    return str(ref)


def _unmatched_vertex_labels(
    graph,
    layer_signature_sets: list[set[Any]],
    pair: tuple[int, int],
) -> tuple[str, ...]:
    swapped_vertex_signature = _swap_graph_signature_builder(graph, pair)
    labels = []
    for layer_idx, layer in enumerate(graph.layers):
        layer_signatures = layer_signature_sets[layer_idx]
        for vertex in layer:
            swapped_signature = swapped_vertex_signature(vertex.ref)
            if swapped_signature not in layer_signatures:
                labels.append(_vertex_label(graph, vertex.ref))
    return tuple(labels)


def _print_pair_failure_report(
    graph,
    candidates: tuple[Any, Any],
    pair: tuple[int, int],
    layer_signature_sets: list[set[Any]],
) -> None:
    unmatched = _unmatched_vertex_labels(graph, layer_signature_sets, pair)
    if not unmatched:
        return

    left, right = candidates
    print(f"Candidate equivalence failed for {left!r} and {right!r}.")
    print("  unmatched vertices:")
    for label in unmatched:
        print(f"    {label}")


def _candidate_layer_counts(
    graph,
    candidate_idx: int,
) -> tuple[tuple[int, int], ...]:
    counts = []
    for layer in graph.layers:
        hopeful_count = 0
        eliminated_count = 0
        for vertex in layer:
            hopefuls = getattr(vertex.key, "hopefuls", frozenset())
            if candidate_idx in hopefuls:
                hopeful_count += 1
            else:
                eliminated_count += 1
        counts.append((hopeful_count, eliminated_count))
    return tuple(counts)


def check_candidate_equivalence(
    graph,
    cand_list=None,
) -> tuple[tuple[Any, ...], ...]:
    """
    Partition candidates into equivalence classes across graph layers.

    For each pair of candidates (x, y), and for each vertex in a layer, this
    checks that the same layer contains a vertex whose state key is identical
    except that x and y have been swapped in the hopeful set. Because seated_at
    is preserved exactly, this tests the elimination-status symmetry described
    by vertices with matching seating histories.

    If cand_list is omitted, all graph candidates are partitioned. If cand_list
    is provided, the full graph candidate set is still used internally, and the
    returned partition contains the full equivalence classes that intersect the
    requested candidates.
    """
    if cand_list is None:
        requested_candidates = None
    else:
        requested_candidates = tuple(cand_list)

    if hasattr(graph, "candidate_names"):
        candidates = tuple(graph.candidate_names)
    else:
        candidates = tuple(range(graph.n_candidates))

    candidate_indices = tuple(_candidate_index(graph, candidate) for candidate in candidates)
    if len(set(candidate_indices)) != len(candidate_indices):
        raise ValueError("cand_list contains duplicate candidates.")

    if requested_candidates is None:
        requested_indices = set(candidate_indices)
    else:
        requested_candidate_indices = tuple(
            _candidate_index(graph, candidate)
            for candidate in requested_candidates
        )
        if len(set(requested_candidate_indices)) != len(requested_candidate_indices):
            raise ValueError("cand_list contains duplicate candidates.")
        requested_indices = set(requested_candidate_indices)

    layer_counts = {
        candidate_idx: _candidate_layer_counts(graph, candidate_idx)
        for candidate_idx in candidate_indices
    }
    layer_signature_sets = _layer_signature_sets(graph)

    parent = list(range(len(candidates)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    report_pair = (
        None
        if requested_candidates is None or len(requested_candidates) != 2
        else tuple(requested_candidate_indices)
    )

    for left, right in combinations(range(len(candidates)), 2):
        pair = (candidate_indices[left], candidate_indices[right])
        if layer_counts[pair[0]] != layer_counts[pair[1]]:
            if report_pair is not None and set(pair) == set(report_pair):
                _print_pair_failure_report(
                    graph,
                    requested_candidates,
                    pair,
                    layer_signature_sets,
                )
            continue
        equivalent = _pair_is_equivalent(graph, layer_signature_sets, pair)
        if equivalent:
            union(left, right)
        elif report_pair is not None and set(pair) == set(report_pair):
            _print_pair_failure_report(
                graph,
                requested_candidates,
                pair,
                layer_signature_sets,
            )

    classes = {}
    for idx, candidate in enumerate(candidates):
        classes.setdefault(find(idx), []).append(candidate)

    return tuple(
        tuple(values)
        for values in classes.values()
        if any(_candidate_index(graph, candidate) in requested_indices for candidate in values)
    )


def _class_label(class_idx: int) -> str:
    n = class_idx + 1
    letters = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(ord("A") + rem))
    return "".join(reversed(letters))


def _quotient_classes(
    graph,
    classes: tuple[tuple[Any, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    nontrivial = []
    assigned = set()

    for equivalence_class in classes:
        class_indices = tuple(
            sorted(_candidate_index(graph, candidate) for candidate in equivalence_class)
        )
        if len(class_indices) <= 1:
            continue
        nontrivial.append(class_indices)
        assigned.update(class_indices)

    quotient_classes = []
    used_nontrivial = {idx: cls for cls in nontrivial for idx in cls}
    seen_nontrivial = set()

    for candidate_idx in range(graph.n_candidates):
        cls = used_nontrivial.get(candidate_idx)
        if cls is None:
            quotient_classes.append((candidate_idx,))
            continue
        if cls in seen_nontrivial:
            continue
        seen_nontrivial.add(cls)
        quotient_classes.append(cls)

    return tuple(quotient_classes)


def _quotient_candidate_map_and_names(
    graph,
    quotient_classes: tuple[tuple[int, ...], ...],
) -> tuple[dict[int, int], list[str]]:
    class_by_candidate = {}
    candidate_names = []
    type_idx = 0

    for class_idx, class_indices in enumerate(quotient_classes):
        for candidate_idx in class_indices:
            class_by_candidate[candidate_idx] = class_idx

        if len(class_indices) > 1:
            candidate_names.append(f"TYPE {_class_label(type_idx)}")
            type_idx += 1
        else:
            candidate_names.append(str(graph.candidate_names[class_indices[0]]))

    return class_by_candidate, candidate_names


def _print_equivalence_class_summary(
    graph,
    quotient_classes: tuple[tuple[int, ...], ...],
) -> None:
    type_idx = 0
    for class_indices in quotient_classes:
        if len(class_indices) <= 1:
            continue

        type_name = f"TYPE {_class_label(type_idx)}"
        type_idx += 1
        members = ", ".join(str(graph.candidate_names[idx]) for idx in class_indices)
        print(f"{type_name}: [{members}]")


def _hopeful_counts(
    values: Iterable[int],
    class_by_candidate: dict[int, int],
    n_classes: int,
) -> tuple[int, ...]:
    counts = [0] * n_classes
    for candidate in values:
        counts[class_by_candidate[int(candidate)]] += 1
    return tuple(counts)


def _quotient_key(
    key: Any,
    class_by_candidate: dict[int, int],
    n_classes: int,
) -> QuotientStateKey:
    return QuotientStateKey(
        seated_at=tuple(getattr(key, "seated_at", ())),
        hopeful_counts=_hopeful_counts(
            getattr(key, "hopefuls", frozenset()),
            class_by_candidate,
            n_classes,
        ),
        seed_id=getattr(key, "seed_id", -1),
    )


def _remap_quotient_key_refs(
    key: QuotientStateKey,
    edge_ref_map: dict[EdgeRef, EdgeRef],
) -> QuotientStateKey:
    return QuotientStateKey(
        seated_at=tuple(
            None if edge_ref is None else edge_ref_map[edge_ref]
            for edge_ref in key.seated_at
        ),
        hopeful_counts=key.hopeful_counts,
        seed_id=key.seed_id,
    )


def _quotient_signature_builder(graph, class_by_candidate: dict[int, int], n_classes: int):
    vertex_memo = {}
    edge_memo = {}

    def vertex_signature(ref: VertexRef):
        if ref in vertex_memo:
            return vertex_memo[ref]

        vertex = _graph_vertex(graph, ref)
        key = vertex.key
        if not is_dataclass(key):
            raise TypeError("quotient_by currently expects dataclass state keys.")

        signature = (
            _hopeful_counts(
                getattr(key, "hopefuls", frozenset()),
                class_by_candidate,
                n_classes,
            ),
            tuple(
                None if edge_ref is None else edge_signature(edge_ref)
                for edge_ref in getattr(key, "seated_at", ())
            ),
            getattr(key, "seed_id", -1),
        )
        vertex_memo[ref] = signature
        return signature

    def edge_signature(ref: EdgeRef):
        if ref in edge_memo:
            return edge_memo[ref]

        edge = _graph_edge(graph, ref)
        signature = (
            vertex_signature(edge.src),
            int(edge.action),
            class_by_candidate[int(edge.candidate)],
        )
        edge_memo[ref] = signature
        return signature

    return vertex_signature


def _merge_edge(existing, incoming) -> None:
    existing.status = max(existing.status, incoming.status)

    if existing.margin is None:
        existing.margin = incoming.margin
    elif incoming.margin is not None:
        existing.margin = min(existing.margin, incoming.margin)

    if existing.transfer_value is None:
        existing.transfer_value = incoming.transfer_value
    if existing.wt_vec is None:
        existing.wt_vec = incoming.wt_vec
    if existing.fpv_vec is None:
        existing.fpv_vec = incoming.fpv_vec


def _project_candidate_array(
    values,
    class_by_candidate: dict[int, int],
    n_classes: int,
    *,
    reducer: str = "sum",
):
    if values is None:
        return None

    try:
        import numpy as np
    except ImportError:
        return values

    arr = np.asarray(values)
    if arr.ndim == 0 or arr.shape[-1] != len(class_by_candidate):
        return values

    projected = np.zeros(arr.shape[:-1] + (n_classes,), dtype=arr.dtype)
    counts = np.zeros(n_classes, dtype=float)
    for old_idx, class_idx in class_by_candidate.items():
        projected[..., class_idx] += arr[..., old_idx]
        counts[class_idx] += 1.0

    if reducer == "mean":
        for class_idx, count in enumerate(counts):
            if count:
                projected[..., class_idx] /= count

    return projected


def _project_fpv_vec(values, class_by_candidate: dict[int, int]):
    if values is None:
        return None

    try:
        import numpy as np
    except ImportError:
        return values

    arr = np.asarray(values).copy()
    for old_idx, class_idx in class_by_candidate.items():
        arr[arr == old_idx] = class_idx
    return arr


def _compact_layers_and_edges(
    new_layers,
    new_edge_layers,
    root_ref: VertexRef | None,
):
    used_refs = set()
    if root_ref is not None:
        used_refs.add(root_ref)

    for edge_layer in new_edge_layers:
        for edge in edge_layer:
            used_refs.add(edge.src)
            used_refs.add(edge.dst)

    vertex_ref_map = {}
    compact_layers = []
    for old_layer_idx, layer in enumerate(new_layers):
        compact_layer = []
        for vertex in layer:
            if vertex.ref not in used_refs:
                continue
            new_ref = VertexRef(old_layer_idx, len(compact_layer))
            vertex_ref_map[vertex.ref] = new_ref
            vertex.ref = new_ref
            compact_layer.append(vertex)
        compact_layers.append(compact_layer)

    compact_edge_layers = []
    edge_ref_map = {}
    for old_edge_layer in new_edge_layers:
        for edge in old_edge_layer:
            if edge.src not in vertex_ref_map or edge.dst not in vertex_ref_map:
                continue

            new_src = vertex_ref_map[edge.src]
            while len(compact_edge_layers) <= new_src.layer:
                compact_edge_layers.append([])

            old_ref = edge.ref
            new_ref = EdgeRef(new_src.layer, len(compact_edge_layers[new_src.layer]))
            edge.ref = new_ref
            edge.src = new_src
            edge.dst = vertex_ref_map[edge.dst]
            compact_edge_layers[new_src.layer].append(edge)
            edge_ref_map[old_ref] = new_ref

    compact_root_ref = None if root_ref is None else vertex_ref_map.get(root_ref)
    return compact_layers, compact_edge_layers, vertex_ref_map, edge_ref_map, compact_root_ref


def quotient_by(graph, cand_list=None):
    """
    Return a WIGM graph quotient by candidate equivalence classes.

    Equivalent candidates are discovered with check_candidate_equivalence. The
    returned graph is a deep copy whose vertices and edges are compacted after
    replacing original candidates by candidate-type classes. Candidate names in
    non-trivial classes are set to TYPE A, TYPE B, ...
    """
    classes = check_candidate_equivalence(graph, cand_list)
    quotient_classes = _quotient_classes(graph, classes)
    _print_equivalence_class_summary(graph, quotient_classes)
    class_by_candidate, candidate_names = _quotient_candidate_map_and_names(
        graph,
        quotient_classes,
    )
    n_classes = len(quotient_classes)
    vertex_signature = _quotient_signature_builder(
        graph,
        class_by_candidate,
        n_classes,
    )

    quotient = deepcopy(graph)
    quotient.candidate_names = candidate_names
    quotient.n_candidates = n_classes
    if hasattr(quotient, "profile") and hasattr(quotient.profile, "candidates"):
        try:
            quotient.profile.candidates = tuple(candidate_names)
        except Exception:
            pass

    vertex_ref_map: dict[VertexRef, VertexRef] = {}
    signature_to_new_ref: dict[tuple[int, Any], VertexRef] = {}
    new_layers = []

    for layer_idx, layer in enumerate(graph.layers):
        new_layer = []
        for old_vertex in layer:
            signature = (layer_idx, vertex_signature(old_vertex.ref))
            new_ref = signature_to_new_ref.get(signature)
            if new_ref is None:
                new_ref = VertexRef(layer_idx, len(new_layer))
                signature_to_new_ref[signature] = new_ref

                new_vertex = deepcopy(old_vertex)
                new_vertex.ref = new_ref
                new_vertex.key = _quotient_key(
                    new_vertex.key,
                    class_by_candidate,
                    n_classes,
                )
                new_vertex.tallies = _project_candidate_array(
                    getattr(new_vertex, "tallies", None),
                    class_by_candidate,
                    n_classes,
                )
                new_vertex.keep_factors = _project_candidate_array(
                    getattr(new_vertex, "keep_factors", None),
                    class_by_candidate,
                    n_classes,
                    reducer="mean",
                )
                new_layer.append(new_vertex)

            vertex_ref_map[old_vertex.ref] = new_ref
        new_layers.append(new_layer)

    edge_ref_map: dict[EdgeRef, EdgeRef] = {}
    new_edge_layers: list[list[Any]] = []
    transition_lookup = {}

    for old_edge_layer in graph.edge_layers:
        for old_edge in old_edge_layer:
            new_src = vertex_ref_map[old_edge.src]
            new_dst = vertex_ref_map[old_edge.dst]
            new_candidate = class_by_candidate[int(old_edge.candidate)]
            transition_key = (new_src, new_dst, old_edge.action, new_candidate)
            existing_ref = transition_lookup.get(transition_key)

            if existing_ref is not None:
                existing_edge = new_edge_layers[existing_ref.layer][
                    existing_ref.local_id
                ]
                _merge_edge(existing_edge, old_edge)
                edge_ref_map[old_edge.ref] = existing_ref
                continue

            while len(new_edge_layers) <= new_src.layer:
                new_edge_layers.append([])

            new_ref = EdgeRef(new_src.layer, len(new_edge_layers[new_src.layer]))
            new_edge = deepcopy(old_edge)
            new_edge.ref = new_ref
            new_edge.src = new_src
            new_edge.dst = new_dst
            new_edge.candidate = new_candidate
            new_edge.fpv_vec = _project_fpv_vec(
                getattr(new_edge, "fpv_vec", None),
                class_by_candidate,
            )

            new_edge_layers[new_src.layer].append(new_edge)
            transition_lookup[transition_key] = new_ref
            edge_ref_map[old_edge.ref] = new_ref

    root_ref = None if graph.root_ref is None else vertex_ref_map[graph.root_ref]

    new_layers, new_edge_layers, compact_vertex_ref_map, compact_edge_ref_map, root_ref = (
        _compact_layers_and_edges(new_layers, new_edge_layers, root_ref)
    )
    vertex_ref_map = {
        old_ref: compact_vertex_ref_map[new_ref]
        for old_ref, new_ref in vertex_ref_map.items()
        if new_ref in compact_vertex_ref_map
    }
    edge_ref_map = {
        old_ref: compact_edge_ref_map[new_ref]
        for old_ref, new_ref in edge_ref_map.items()
        if new_ref in compact_edge_ref_map
    }

    for layer in new_layers:
        for vertex in layer:
            vertex.key = _remap_quotient_key_refs(vertex.key, edge_ref_map)

    quotient.layers = new_layers
    quotient.edge_layers = new_edge_layers
    quotient.edge_by_ref = {
        edge.ref: edge for edge_layer in new_edge_layers for edge in edge_layer
    }
    quotient.edge_lookup = {
        (edge.src, edge.dst, edge.action, edge.candidate): edge.ref
        for edge_layer in new_edge_layers
        for edge in edge_layer
    }
    quotient.root_ref = root_ref

    quotient.layer_index = []
    if hasattr(quotient, "same_seated_index"):
        quotient.same_seated_index = []

    for layer_idx, layer in enumerate(quotient.layers):
        quotient.layer_index.append({})
        if hasattr(quotient, "same_seated_index"):
            quotient.same_seated_index.append(defaultdict(set))

        for vertex in layer:
            quotient.layer_index[layer_idx][vertex.key] = vertex.ref.local_id
            if hasattr(quotient, "same_seated_index"):
                quotient.same_seated_index[layer_idx][
                    quotient._same_seated_index_key(vertex.key)
                ].add(vertex.ref)

    quotient.primary_parent_edge = {}
    for edge_layer in quotient.edge_layers:
        for edge in edge_layer:
            quotient.primary_parent_edge.setdefault(edge.dst, edge.ref)

    outgoing_counts = defaultdict(int)
    for edge_layer in quotient.edge_layers:
        for edge in edge_layer:
            outgoing_counts[edge.src] += 1

    for layer in quotient.layers:
        for vertex in layer:
            vertex.path_multiplicity = 0
            if vertex.degree == quotient.m:
                vertex.status = ElectionStatus.TERMINAL
            elif outgoing_counts[vertex.ref] == 0:
                vertex.status = ElectionStatus.TERMINAL
            else:
                vertex.status = ElectionStatus.EXPANDED

    if quotient.root_ref is not None:
        quotient.vertex(quotient.root_ref).path_multiplicity = 1
    for edge_layer in quotient.edge_layers:
        for edge in edge_layer:
            quotient.vertex(edge.dst).path_multiplicity += quotient.vertex(
                edge.src
            ).path_multiplicity

    quotient.stack = deque()
    quotient.enqueued = set()
    quotient.pending_primary_children = defaultdict(int)
    quotient.runtime_cache = {}
    quotient.tightest_margins_assigned = False
    quotient.coherence_checked = False
    quotient.terminal_vertices_by_winner_set = {}

    if hasattr(quotient, "vertex_post_seed_tallies"):
        quotient.vertex_post_seed_tallies = {
            vertex_ref_map[old_ref]: value
            for old_ref, value in graph.vertex_post_seed_tallies.items()
            if old_ref in vertex_ref_map
        }
    if hasattr(quotient, "edge_weight_scenarios"):
        quotient.edge_weight_scenarios = {
            edge_ref_map[old_ref]: value
            for old_ref, value in graph.edge_weight_scenarios.items()
            if old_ref in edge_ref_map
        }
    if hasattr(quotient, "black_box_edge_ref"):
        old_ref = getattr(graph, "black_box_edge_ref", None)
        quotient.black_box_edge_ref = None if old_ref is None else edge_ref_map.get(old_ref)
    if hasattr(quotient, "_seed_connector_ref"):
        old_ref = getattr(graph, "_seed_connector_ref", None)
        quotient._seed_connector_ref = (
            None if old_ref is None else vertex_ref_map.get(old_ref)
        )
    if hasattr(quotient, "_seed_refs"):
        quotient._seed_refs = list(
            dict.fromkeys(
                vertex_ref_map[ref]
                for ref in getattr(graph, "_seed_refs", [])
                if ref in vertex_ref_map
            )
        )

    return quotient

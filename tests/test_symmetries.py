from types import SimpleNamespace

from src.election_graphs.datatypes import ElectionEdge, ElectionState
from src.symmetries import check_candidate_equivalence, quotient_by
from src.wigm_graphs.datatypes import EdgeAction, EdgeRef, StateKey, VertexRef


def _vertex(layer: int, local_id: int, *, seated_at, hopefuls):
    return SimpleNamespace(
        ref=VertexRef(layer, local_id),
        key=StateKey(
            seated_at=seated_at,
            hopefuls=frozenset(hopefuls),
        ),
    )


class _Graph:
    candidate_names = ("A", "B", "C")
    n_candidates = 3

    def __init__(self, layers, edge_by_ref=None):
        self.layers = layers
        self.edge_by_ref = {} if edge_by_ref is None else edge_by_ref

    def vertex_label(self, ref):
        return f"{ref.layer}:{ref.local_id}"


class _WIGMGraph(_Graph):
    candidate_names = ["A", "B", "C"]
    n_candidates = 3
    m = 1
    root_ref = VertexRef(0, 0)
    layer_index = []
    edge_lookup = {}
    primary_parent_edge = {}
    tightest_margins_assigned = True
    coherence_checked = True
    terminal_vertices_by_winner_set = {}

    def __init__(self):
        self.layers = [
            [
                ElectionState(
                    ref=VertexRef(0, 0),
                    key=StateKey(
                        seated_at=(None,),
                        hopefuls=frozenset({0, 1, 2}),
                    ),
                    degree=0,
                )
            ],
            [
                ElectionState(
                    ref=VertexRef(1, 0),
                    key=StateKey(
                        seated_at=(None,),
                        hopefuls=frozenset({1, 2}),
                    ),
                    degree=0,
                ),
                ElectionState(
                    ref=VertexRef(1, 1),
                    key=StateKey(
                        seated_at=(None,),
                        hopefuls=frozenset({0, 2}),
                    ),
                    degree=0,
                ),
                ElectionState(
                    ref=VertexRef(1, 2),
                    key=StateKey(
                        seated_at=(None,),
                        hopefuls=frozenset({0, 1}),
                    ),
                    degree=0,
                ),
            ],
        ]
        self.edge_layers = [
            [
                ElectionEdge(
                    ref=EdgeRef(0, 0),
                    src=VertexRef(0, 0),
                    dst=VertexRef(1, 0),
                    action=EdgeAction.ELIMINATE,
                    candidate=0,
                    margin=0.0,
                ),
                ElectionEdge(
                    ref=EdgeRef(0, 1),
                    src=VertexRef(0, 0),
                    dst=VertexRef(1, 1),
                    action=EdgeAction.ELIMINATE,
                    candidate=1,
                    margin=2.0,
                ),
                ElectionEdge(
                    ref=EdgeRef(0, 2),
                    src=VertexRef(0, 0),
                    dst=VertexRef(1, 2),
                    action=EdgeAction.ELIMINATE,
                    candidate=2,
                    margin=1.0,
                ),
            ]
        ]
        self.edge_by_ref = {
            edge.ref: edge for edge_layer in self.edge_layers for edge in edge_layer
        }

    def vertex(self, ref):
        return self.layers[ref.layer][ref.local_id]

    def edge(self, ref):
        return self.edge_by_ref[ref]


class _TwoTypeOverlapGraph(_WIGMGraph):
    candidate_names = ["A0", "A1", "B0", "B1"]
    n_candidates = 4

    def __init__(self):
        self.layers = [
            [
                ElectionState(
                    ref=VertexRef(0, 0),
                    key=StateKey(
                        seated_at=(None,),
                        hopefuls=frozenset({0, 1, 2, 3}),
                    ),
                    degree=0,
                )
            ],
            [
                ElectionState(
                    ref=VertexRef(1, 0),
                    key=StateKey(
                        seated_at=(None,),
                        hopefuls=frozenset({1, 2, 3}),
                    ),
                    degree=0,
                ),
                ElectionState(
                    ref=VertexRef(1, 1),
                    key=StateKey(
                        seated_at=(None,),
                        hopefuls=frozenset({0, 2, 3}),
                    ),
                    degree=0,
                ),
                ElectionState(
                    ref=VertexRef(1, 2),
                    key=StateKey(
                        seated_at=(None,),
                        hopefuls=frozenset({0, 1, 3}),
                    ),
                    degree=0,
                ),
                ElectionState(
                    ref=VertexRef(1, 3),
                    key=StateKey(
                        seated_at=(None,),
                        hopefuls=frozenset({0, 1, 2}),
                    ),
                    degree=0,
                ),
            ],
        ]
        self.edge_layers = [
            [
                ElectionEdge(
                    ref=EdgeRef(0, idx),
                    src=VertexRef(0, 0),
                    dst=VertexRef(1, idx),
                    action=EdgeAction.ELIMINATE,
                    candidate=idx,
                    margin=float(idx),
                )
                for idx in range(4)
            ]
        ]
        self.edge_by_ref = {
            edge.ref: edge for edge_layer in self.edge_layers for edge in edge_layer
        }


class _RelaxedSeatedAtGraph(_WIGMGraph):
    candidate_names = ["A", "B", "C"]
    n_candidates = 3

    def __init__(self):
        edge_a = EdgeRef(0, 0)
        edge_b = EdgeRef(0, 1)
        self.layers = [
            [
                ElectionState(
                    ref=VertexRef(0, 0),
                    key=StateKey(
                        seated_at=(None,),
                        hopefuls=frozenset({0, 1, 2}),
                    ),
                    degree=0,
                )
            ],
            [
                ElectionState(
                    ref=VertexRef(1, 0),
                    key=StateKey(
                        seated_at=(edge_a,),
                        hopefuls=frozenset({1, 2}),
                    ),
                    degree=1,
                ),
                ElectionState(
                    ref=VertexRef(1, 1),
                    key=StateKey(
                        seated_at=(edge_b,),
                        hopefuls=frozenset({0, 2}),
                    ),
                    degree=1,
                ),
            ],
        ]
        self.edge_layers = [
            [
                ElectionEdge(
                    ref=edge_a,
                    src=VertexRef(0, 0),
                    dst=VertexRef(1, 0),
                    action=EdgeAction.ELECT,
                    candidate=0,
                    margin=0.0,
                ),
                ElectionEdge(
                    ref=edge_b,
                    src=VertexRef(0, 0),
                    dst=VertexRef(1, 1),
                    action=EdgeAction.ELECT,
                    candidate=1,
                    margin=0.0,
                ),
            ]
        ]
        self.edge_by_ref = {
            edge.ref: edge for edge_layer in self.edge_layers for edge in edge_layer
        }


def test_check_candidate_equivalence_accepts_pairwise_symmetric_layers():
    seated_edge = EdgeRef(0, 0)
    graph = _Graph(
        [
            [
                _vertex(0, 0, seated_at=(None,), hopefuls={0, 1, 2}),
            ],
            [
                _vertex(1, 0, seated_at=(None,), hopefuls={0, 2}),
                _vertex(1, 1, seated_at=(None,), hopefuls={1, 2}),
                _vertex(1, 2, seated_at=(None,), hopefuls={2}),
                _vertex(1, 3, seated_at=(seated_edge,), hopefuls={0, 2}),
                _vertex(1, 4, seated_at=(seated_edge,), hopefuls={1, 2}),
            ],
        ],
        edge_by_ref={
            seated_edge: SimpleNamespace(
                ref=seated_edge,
                src=VertexRef(0, 0),
                action=EdgeAction.ELECT,
                candidate=2,
            )
        },
    )

    classes = check_candidate_equivalence(graph)

    assert classes == (("A", "B"), ("C",))


def test_check_candidate_equivalence_reports_missing_swapped_vertex(capsys):
    graph = _Graph(
        [
            [
                _vertex(0, 0, seated_at=(None,), hopefuls={0, 1, 2}),
            ],
            [
                _vertex(1, 0, seated_at=(None,), hopefuls={0, 2}),
            ],
        ]
    )

    classes = check_candidate_equivalence(graph, [0, 1])

    assert classes == (("A", "C"), ("B",))
    output = capsys.readouterr().out
    assert "Candidate equivalence failed for 0 and 1." in output
    assert "1:0" in output


def test_check_candidate_equivalence_early_outs_on_layer_counts(monkeypatch):
    graph = _Graph(
        [
            [
                _vertex(0, 0, seated_at=(None,), hopefuls={0, 1, 2}),
            ],
            [
                _vertex(1, 0, seated_at=(None,), hopefuls={0, 2}),
                _vertex(1, 1, seated_at=(None,), hopefuls={0}),
            ],
        ]
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("full pair equivalence check should have been skipped")

    monkeypatch.setattr("src.symmetries._pair_is_equivalent", fail_if_called)

    assert check_candidate_equivalence(graph, [0, 1]) == (("A",), ("B",))


def test_check_candidate_equivalence_relaxes_seated_at_edge_refs():
    graph = _RelaxedSeatedAtGraph()

    assert check_candidate_equivalence(graph, [0, 1]) == (("A", "B"),)


def test_quotient_by_deduplicates_equivalent_vertices_and_edges(capsys):
    graph = _WIGMGraph()

    quotient = quotient_by(graph)

    assert quotient is not graph
    assert [len(layer) for layer in graph.layers] == [1, 3]
    assert [len(layer) for layer in quotient.layers] == [1, 1]
    assert [len(layer) for layer in quotient.edge_layers] == [1]
    assert quotient.n_candidates == 1
    assert quotient.candidate_names == ["TYPE A"]

    edge = quotient.edge_layers[0][0]
    assert edge.candidate == 0
    assert edge.margin == 0.0
    assert quotient.layers[0][0].key.hopeful_counts == (3,)
    assert quotient.layers[1][0].key.hopeful_counts == (2,)
    assert capsys.readouterr().out == "TYPE A: [A, B, C]\n"
    parented = {edge.dst for edge_layer in quotient.edge_layers for edge in edge_layer}
    for layer in quotient.layers[1:]:
        for vertex in layer:
            assert vertex.ref in parented


def test_quotient_by_deduplicates_edges_with_different_quotient_candidates(monkeypatch):
    graph = _TwoTypeOverlapGraph()
    monkeypatch.setattr(
        "src.symmetries.check_candidate_equivalence",
        lambda graph, cand_list=None: ((0, 1), (2, 3)),
    )

    quotient = quotient_by(graph)

    assert [len(layer) for layer in quotient.layers] == [1, 2]
    assert [len(layer) for layer in quotient.edge_layers] == [2]
    assert quotient.n_candidates == 2
    assert quotient.candidate_names == ["TYPE A", "TYPE B"]
    labels = [
        quotient.candidate_names[edge.candidate]
        for edge in quotient.edge_layers[0]
    ]
    assert labels == ["TYPE A", "TYPE B"]
    assert {edge.dst for edge in quotient.edge_layers[0]} == {
        VertexRef(1, 0),
        VertexRef(1, 1),
    }
    assert {
        vertex.key.hopeful_counts for vertex in quotient.layers[1]
    } == {(1, 2), (2, 1)}
    parented = {edge.dst for edge_layer in quotient.edge_layers for edge in edge_layer}
    for layer in quotient.layers[1:]:
        for vertex in layer:
            assert vertex.ref in parented

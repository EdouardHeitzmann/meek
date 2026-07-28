import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict

from .wigm_graphs.datatypes import EdgeAction

def _compute_layer_positions(
    graph,
    horizontal_spacing=1.0,
    vertical_spacing=1.5,
    plot_horizontal=False,
    contract_empty_layers=False,
):
    """
    Compute layered positions.

    Default, plot_horizontal=False:
      - layers go top to bottom
      - vertices in each layer spread left to right

    If plot_horizontal=True:
      - layers go left to right
      - vertices in each layer spread top to bottom
    """
    positions = {}

    max_layer_width = max((len(layer) for layer in graph.layers), default=1)

    if max_layer_width <= 1:
        layer_span = 0.0
    else:
        layer_span = horizontal_spacing * (max_layer_width - 1)

    low = -layer_span / 2
    high = layer_span / 2

    display_layer_idx = 0

    for layer_idx, layer in enumerate(graph.layers):
        n = len(layer)
        if n == 0:
            continue

        layout_layer_idx = display_layer_idx if contract_empty_layers else layer_idx

        if n == 1:
            spread_coords = np.array([0.0])
        else:
            spread_coords = np.linspace(low, high, n)

        for spread_coord, vertex in zip(spread_coords, layer):
            if plot_horizontal:
                # Layers move left to right.
                x = layout_layer_idx * vertical_spacing
                y = -float(spread_coord)
            else:
                # Layers move top to bottom.
                x = float(spread_coord)
                y = -layout_layer_idx * vertical_spacing

            positions[vertex.ref] = (float(x), float(y))

        display_layer_idx += 1

    return positions


def _edge_label(graph, edge, mode="literal") -> str:
    if mode is None:
        return ""

    if mode == "literal":
        cand = graph.candidate_names[edge.candidate]
    elif mode == "minimal":
        cand = str(edge.candidate)
    else:
        raise ValueError("label_edges must be None, 'literal', or 'minimal'.")

    if edge.action.is_election:
        return f"+ {cand}"
    elif edge.action == EdgeAction.ELIMINATE:
        return f"x {cand}"
    else:
        return str(cand)
    

def _draw_edges(
    ax,
    graph,
    positions,
    visible_refs=None,
    node_size=100,
    arrowsize=10,
    hidden_edge_refs=None,
    highlight_natural_edges=False,
    highlighted_edge_refs=None,
):
    """
    Draw arrows with endpoints shrunk away from node centers.
    If visible_refs is provided, only draw edges whose endpoints are both visible.
    """
    node_radius_pts = 0.5 * np.sqrt(node_size)
    hidden_edge_refs = set() if hidden_edge_refs is None else set(hidden_edge_refs)
    highlighted_edge_refs = (
        None if highlighted_edge_refs is None else set(highlighted_edge_refs)
    )

    for edge_layer in graph.edge_layers:
        for edge in edge_layer:
            if edge.ref in hidden_edge_refs:
                continue
            if visible_refs is not None:
                if edge.src not in visible_refs or edge.dst not in visible_refs:
                    continue

            x1, y1 = positions[edge.src]
            x2, y2 = positions[edge.dst]
            color = (
                "red"
                if (
                    edge.ref in highlighted_edge_refs
                    if highlighted_edge_refs is not None
                    else highlight_natural_edges and _is_natural_edge(edge)
                )
                else "gray"
            )

            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=0.8,
                    shrinkA=node_radius_pts + 1.5,
                    shrinkB=node_radius_pts + 1.5,
                    mutation_scale=arrowsize,
                ),
                zorder=1,
            )


def _is_natural_edge(edge) -> bool:
    margin = getattr(edge, "margin", None)
    return margin is not None and bool(np.isclose(float(margin), 0.0))


def _natural_path_edge_refs(graph) -> set:
    current_ref = getattr(graph, "root_ref", None)
    if current_ref is None:
        for layer in graph.layers:
            if layer:
                current_ref = layer[0].ref
                break

    if current_ref is None:
        return set()

    path = set()
    seen_vertices = set()
    while current_ref is not None and current_ref not in seen_vertices:
        seen_vertices.add(current_ref)
        natural_edges = [
            edge
            for edge in graph.outgoing_edges(current_ref)
            if _is_natural_edge(edge)
        ]
        if not natural_edges:
            break
        edge = min(
            natural_edges,
            key=lambda e: (e.dst.layer, e.dst.local_id, e.ref.layer, e.ref.local_id),
        )
        path.add(edge.ref)
        current_ref = edge.dst

    return path


def _wrap_edge_label_text(label: str, width: int = 7) -> str:
    """
    Wrap long edge labels, preserving the '+ ' or 'x ' prefix on the first line.
    """
    if len(label) <= width:
        return label

    if len(label) >= 2 and label[1] == " ":
        prefix = label[:2]          # '+ ' or 'x '
        rest = label[2:]
        wrapped = textwrap.fill(
            rest,
            width=max(4, width - 2),
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines = wrapped.split("\n")
        if not lines:
            return label
        return prefix + lines[0] + "".join("\n  " + line for line in lines[1:])

    return textwrap.fill(
        label,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )

def _visible_refs(graph, lam_restriction=None) -> set:
    """
    Return the vertex refs that should be plotted.

    If lam_restriction is None, every vertex is visible.
    Otherwise, only vertices with tightest_margin < lam_restriction are visible.
    """
    visible = set()

    for layer in graph.layers:
        for v in layer:
            if lam_restriction is None:
                visible.add(v.ref)
            elif v.tightest_margin is not None and v.tightest_margin < lam_restriction:
                visible.add(v.ref)

    return visible

def _draw_edge_labels(
    ax,
    graph,
    positions,
    visible_refs=None,
    font_size=6,
    wrap_width=10,
    offset=0.0,
    label_mode="literal",
    parallel_label_spacing=0.45,
    hidden_edge_refs=None,
):
    """
    Draw horizontal edge labels.

    If visible_refs is provided, only label edges whose endpoints are both visible.

    Placement rule:
      - right-pointing edge: label in the upper/early third of the edge
      - left-pointing edge: label in the lower/late third of the edge
      - vertical edge: label at the midpoint

    Here "third" means position along the edge segment, not perpendicular offset.
    """
    label_edges = []
    parallel_groups = defaultdict(list)
    hidden_edge_refs = set() if hidden_edge_refs is None else set(hidden_edge_refs)

    for edge_layer in graph.edge_layers:
        for edge in edge_layer:
            if edge.ref in hidden_edge_refs:
                continue
            if visible_refs is not None:
                if edge.src not in visible_refs or edge.dst not in visible_refs:
                    continue

            label_edges.append(edge)

            if edge.action.is_election:
                parallel_groups[(edge.src, edge.dst)].append(edge.ref)

    parallel_offsets = {}
    for refs in parallel_groups.values():
        if len(refs) <= 1:
            continue

        midpoint = (len(refs) - 1) / 2.0
        for idx, ref in enumerate(refs):
            parallel_offsets[ref] = (idx - midpoint) * parallel_label_spacing

    for edge in label_edges:
        x1, y1 = positions[edge.src]
        x2, y2 = positions[edge.dst]

        dx = x2 - x1
        dy = y2 - y1

        # Choose position along the edge.
        if dx > 0:
            t = 1.0 / 3.0
        elif dx < 0:
            t = 2.0 / 3.0
        else:
            t = 0.5

        label_x = x1 + t * dx
        label_y = y1 + t * dy

        label_offset = offset + parallel_offsets.get(edge.ref, 0.0)
        if label_offset != 0.0:
            length = np.hypot(dx, dy)
            if length != 0:
                nx, ny = -dy / length, dx / length
                label_x += label_offset * nx
                label_y += label_offset * ny

        label = _edge_label(graph, edge, mode=label_mode)
        label = _wrap_edge_label_text(label, width=wrap_width)

        ax.text(
            label_x,
            label_y,
            label,
            fontsize=font_size,
            ha="center",
            va="center",
            multialignment="center",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=0.15,
            ),
            zorder=4,
        )


def _draw_seed_connector(
    ax,
    positions,
    connector_ref,
    seed_refs,
    visible_refs,
    *,
    plot_horizontal=False,
    xpad=0.75,
    ypad=0.75,
    bar_thickness=0.24,
    label=None,
    label_font_size=6,
):
    visible_seed_refs = [
        ref
        for ref in seed_refs
        if ref in positions and (visible_refs is None or ref in visible_refs)
    ]

    if (
        connector_ref is None
        or connector_ref not in positions
        or (visible_refs is not None and connector_ref not in visible_refs)
        or not visible_seed_refs
    ):
        return None

    connector_x, connector_y = positions[connector_ref]
    plot_refs = [
        ref
        for ref in positions
        if visible_refs is None or ref in visible_refs
    ]

    if plot_horizontal:
        seed_xs = [positions[ref][0] for ref in visible_seed_refs]
        plot_ys = [positions[ref][1] for ref in plot_refs]
        box_x = (connector_x + min(seed_xs)) / 2.0
        ymin = min(plot_ys) - ypad
        ymax = max(plot_ys) + ypad
        box_height = max(ymax - ymin, 0.4)
        box_width = bar_thickness
        box_center = (box_x, (ymin + ymax) / 2.0)
        rect = Rectangle(
            (box_x - box_width / 2.0, box_center[1] - box_height / 2.0),
            box_width,
            box_height,
            facecolor="black",
            edgecolor="black",
            zorder=2,
        )
        edge_start = (box_x - box_width / 2.0, box_center[1])
        edge_end = (box_x + box_width / 2.0, box_center[1])
    else:
        seed_ys = [positions[ref][1] for ref in visible_seed_refs]
        plot_xs = [positions[ref][0] for ref in plot_refs]
        box_y = (connector_y + max(seed_ys)) / 2.0
        xmin = min(plot_xs) - xpad
        xmax = max(plot_xs) + xpad
        box_width = max(xmax - xmin, 0.4)
        box_height = bar_thickness
        box_center = ((xmin + xmax) / 2.0, box_y)
        rect = Rectangle(
            (box_center[0] - box_width / 2.0, box_y - box_height / 2.0),
            box_width,
            box_height,
            facecolor="black",
            edgecolor="black",
            zorder=2,
        )
        edge_start = (box_center[0], box_y + box_height / 2.0)
        edge_end = (box_center[0], box_y - box_height / 2.0)

    ax.add_patch(rect)

    if label:
        ax.text(
            box_center[0],
            box_center[1],
            label,
            fontsize=label_font_size,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            zorder=4,
        )

    ax.annotate(
        "",
        xy=edge_start,
        xytext=(connector_x, connector_y),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0, mutation_scale=10),
        zorder=2,
    )

    for seed_ref in visible_seed_refs:
        seed_x, seed_y = positions[seed_ref]
        ax.annotate(
            "",
            xy=(seed_x, seed_y),
            xytext=edge_end,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8, mutation_scale=10),
            zorder=2,
        )

    if plot_horizontal:
        return [
            box_center[0] - box_width / 2.0,
            box_center[0] + box_width / 2.0,
        ], [
            box_center[1] - box_height / 2.0,
            box_center[1] + box_height / 2.0,
        ]

    return [
        box_center[0] - box_width / 2.0,
        box_center[0] + box_width / 2.0,
    ], [
        box_y - box_height / 2.0,
        box_y + box_height / 2.0,
    ]


def plot_wigm_graph(
    graph,
    figsize=(15, 10),
    node_size=100,
    font_size=7,
    label_vertices=False,
    label_edges="literal",
    title=None,
    horizontal_spacing=1.0,
    vertical_spacing=1.5,
    xpad=0.75,
    ypad=0.75,
    lam_restriction=None,
    plot_horizontal=False,
    seeded_build=False,
    parallel_label_spacing=None,
    highlight_natural_edges=False,
    highlight_natural_path=False,
):
    """
    Plot a built WIGMGraphConstructor without materializing a NetworkX graph.

    Layout:
      - each layer has a fixed y-value
      - vertices in each layer are equally spaced horizontally
      - layers are centered at x=0
    """
    if not any(len(layer) > 0 for layer in graph.layers):
        print("No nodes to plot")
        return

    try:
        from .wigm_graphs.black_box import BlackBoxWIGMGraphConstructor
    except ImportError:
        BlackBoxWIGMGraphConstructor = ()

    is_black_box_graph = isinstance(graph, BlackBoxWIGMGraphConstructor)
    if is_black_box_graph:
        seeded_build = True

    positions = _compute_layer_positions(
        graph,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        plot_horizontal=plot_horizontal,
        contract_empty_layers=seeded_build,
    )

    visible_refs = _visible_refs(graph, lam_restriction=lam_restriction)
    hidden_edge_refs = set()
    seed_connector_ref = getattr(graph, "_seed_connector_ref", None)
    seed_connector_label = None
    seed_bar_thickness = 0.24

    if is_black_box_graph:
        black_box_edge_ref = getattr(graph, "black_box_edge_ref", None)
        if black_box_edge_ref is not None:
            black_box_edge = graph.edge(black_box_edge_ref)
            hidden_edge_refs.add(black_box_edge_ref)
            visible_refs.discard(black_box_edge.dst)
            seed_connector_ref = black_box_edge.src
            seed_connector_label = _edge_label(
                graph,
                black_box_edge,
                mode="literal" if label_edges is None else label_edges,
            )
            seed_bar_thickness = max(0.55, 0.08 * len(seed_connector_label))

    if not visible_refs:
        print("No vertices satisfy the lam_restriction")
        return

    fig, ax = plt.subplots(figsize=figsize)
    highlighted_edge_refs = (
        _natural_path_edge_refs(graph) if highlight_natural_path else None
    )

    # Edges first, nodes on top.
    _draw_edges(
        ax,
        graph,
        positions,
        visible_refs=visible_refs,
        node_size=node_size,
        arrowsize=10,
        hidden_edge_refs=hidden_edge_refs,
        highlight_natural_edges=highlight_natural_edges,
        highlighted_edge_refs=highlighted_edge_refs,
    )

    seed_connector_bounds = None
    if seeded_build:
        seed_connector_bounds = _draw_seed_connector(
            ax,
            positions,
            seed_connector_ref,
            getattr(graph, "_seed_refs", []),
            visible_refs,
            plot_horizontal=plot_horizontal,
            xpad=xpad,
            ypad=ypad,
            bar_thickness=seed_bar_thickness,
            label=seed_connector_label,
            label_font_size=max(5, font_size - 1),
        )

    refs = []
    xs = []
    ys = []
    color_values = []

    for layer in graph.layers:
        for v in layer:
            if v.ref not in visible_refs:
                continue

            refs.append(v.ref)
            x, y = positions[v.ref]
            xs.append(x)
            ys.append(y)
            color_values.append(0 if v.color is None else int(v.color))

    cmap = plt.get_cmap("viridis")
    max_color = max(color_values, default=0)

    if max_color == 0:
        node_colors = [cmap(0.3) for _ in color_values]
    else:
        node_colors = [cmap(0.2 + 0.8 * (c / max_color)) for c in color_values]

    ax.scatter(
        xs,
        ys,
        s=node_size,
        c=node_colors,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )

    if label_vertices:
        for ref, x, y in zip(refs, xs, ys):
            label = graph.vertex_label(ref) if hasattr(graph, "vertex_label") else str(ref.local_id)

            ax.text(
                x,
                y,
                label,
                fontsize=font_size,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=5,
            )

    if label_edges is not None:
        if parallel_label_spacing is None:
            parallel_label_spacing = max(0.45, 0.045 * font_size)

        _draw_edge_labels(
            ax,
            graph,
            positions,
            visible_refs=visible_refs,
            font_size=max(5, font_size - 1),
            label_mode=label_edges,
            parallel_label_spacing=parallel_label_spacing,
            hidden_edge_refs=hidden_edge_refs,
        )

    if title is None:
        title = (
            f"WIGM Decision Graph\n"
            f"Election: {graph.m} winners from {graph.n_candidates} candidates"
        )

    ax.set_title(title)

    # Tight plotting window: do not preserve equal aspect ratio.
    all_xs = [x for x, y in positions.values()]
    all_ys = [y for x, y in positions.values()]

    if seed_connector_bounds is not None:
        box_xs, box_ys = seed_connector_bounds
        all_xs.extend(box_xs)
        all_ys.extend(box_ys)

    xmin, xmax = min(all_xs), max(all_xs)
    ymin, ymax = min(all_ys), max(all_ys)

    if seed_connector_bounds is not None and not plot_horizontal:
        xmin, xmax = seed_connector_bounds[0]
    elif xmin == xmax:
        xmin -= xpad
        xmax += xpad
    else:
        xmin -= xpad
        xmax += xpad

    if seed_connector_bounds is not None and plot_horizontal:
        ymin, ymax = seed_connector_bounds[1]
    elif ymin == ymax:
        ymin -= ypad
        ymax += ypad
    else:
        ymin -= ypad
        ymax += ypad

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.axis("off")
    plt.tight_layout()
    plt.show()


def _candidate_index(graph, candidate) -> int:
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


def _candidate_plot_label(graph, candidate_idx: int) -> str:
    if hasattr(graph, "candidate_names"):
        return str(graph.candidate_names[candidate_idx])
    return str(candidate_idx)


def _vertex_winner_set(graph, vertex) -> frozenset[int]:
    if hasattr(graph, "_winner_set_for_vertex"):
        return graph._winner_set_for_vertex(vertex)

    key = getattr(vertex, "key", None)
    seated_at = getattr(key, "seated_at", ())
    winners = set()

    for edge_ref in seated_at:
        if edge_ref is None:
            continue
        winners.add(int(graph.edge(edge_ref).candidate))

    return frozenset(winners)


def _candidate_is_eliminated_at_vertex(graph, vertex, candidate_idx: int) -> bool:
    key = getattr(vertex, "key", None)
    hopefuls = getattr(key, "hopefuls", frozenset())

    if candidate_idx in hopefuls:
        return False

    return candidate_idx not in _vertex_winner_set(graph, vertex)


def plot_vertex_densities(
    graph,
    cand_list,
    figsize=(12, 5),
    alpha=0.35,
    width=0.85,
    title=None,
    plot_numerical_heights=True,
    show_complement_of_eliminated=False,
):
    """
    Plot per-layer elimination densities for selected candidates.

    By default, for a candidate c and graph layer L, the bar height is:

        (# vertices in L where c is eliminated) / (# vertices in L)

    If show_complement_of_eliminated is True, the numerator is replaced by the
    number of vertices where c is not eliminated.

    A candidate is treated as eliminated at a vertex when they are neither in the
    vertex's hopeful set nor in the winners already seated at that vertex.
    """
    candidate_indices = [_candidate_index(graph, candidate) for candidate in cand_list]
    layer_indices = np.arange(len(graph.layers), dtype=int)

    fig, ax = plt.subplots(figsize=figsize)

    for candidate_idx in candidate_indices:
        densities = []
        count_labels = []

        for layer in graph.layers:
            if not layer:
                densities.append(0.0)
                count_labels.append("0/0")
                continue

            eliminated = sum(
                _candidate_is_eliminated_at_vertex(graph, vertex, candidate_idx)
                for vertex in layer
            )
            numerator = len(layer) - eliminated if show_complement_of_eliminated else eliminated
            densities.append(numerator / len(layer))
            count_labels.append(f"{numerator}/{len(layer)}")

        bars = ax.bar(
            layer_indices,
            densities,
            width=width,
            alpha=alpha,
            label=_candidate_plot_label(graph, candidate_idx),
        )

        if plot_numerical_heights:
            for bar, label in zip(bars, count_labels):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    min(height + 0.02, 1.03),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    alpha=0.8,
                )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Eliminated Vertex Frequency")
    ax.set_ylim(0.0, 1.08 if plot_numerical_heights else 1.0)
    ax.set_xticks(layer_indices)
    if title is None:
        title = "Candidate Elimination Density by Layer"
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.show()
    return ax

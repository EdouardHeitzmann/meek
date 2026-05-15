import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict

from .wigm_graphs.datatypes import EdgeAction

def _compute_layer_positions(
    graph,
    horizontal_spacing=1.0,
    vertical_spacing=1.5,
    plot_horizontal=False,
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

    for layer_idx, layer in enumerate(graph.layers):
        n = len(layer)
        if n == 0:
            continue

        if n == 1:
            spread_coords = np.array([0.0])
        else:
            spread_coords = np.linspace(low, high, n)

        for spread_coord, vertex in zip(spread_coords, layer):
            if plot_horizontal:
                # Layers move left to right.
                x = layer_idx * vertical_spacing
                y = -float(spread_coord)
            else:
                # Layers move top to bottom.
                x = float(spread_coord)
                y = -layer_idx * vertical_spacing

            positions[vertex.ref] = (float(x), float(y))

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

    if edge.action in (EdgeAction.ELECT, EdgeAction.FORCE_ELECT):
        return f"+ {cand}"
    elif edge.action == EdgeAction.ELIMINATE:
        return f"x {cand}"
    else:
        return str(cand)
    

def _draw_edges(ax, graph, positions, visible_refs=None, node_size=100, arrowsize=10):
    """
    Draw arrows with endpoints shrunk away from node centers.
    If visible_refs is provided, only draw edges whose endpoints are both visible.
    """
    node_radius_pts = 0.5 * np.sqrt(node_size)

    for edge_layer in graph.edge_layers:
        for edge in edge_layer:
            if visible_refs is not None:
                if edge.src not in visible_refs or edge.dst not in visible_refs:
                    continue

            x1, y1 = positions[edge.src]
            x2, y2 = positions[edge.dst]

            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    color="gray",
                    lw=0.8,
                    shrinkA=node_radius_pts + 1.5,
                    shrinkB=node_radius_pts + 1.5,
                    mutation_scale=arrowsize,
                ),
                zorder=1,
            )

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
    for edge_layer in graph.edge_layers:
        for edge in edge_layer:
            if visible_refs is not None:
                if edge.src not in visible_refs or edge.dst not in visible_refs:
                    continue

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

            # Optional perpendicular nudge, left at 0 by default.
            if offset != 0.0:
                length = np.hypot(dx, dy)
                if length != 0:
                    nx, ny = -dy / length, dx / length
                    label_x += offset * nx
                    label_y += offset * ny

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

    positions = _compute_layer_positions(
        graph,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        plot_horizontal=plot_horizontal,
    )

    visible_refs = _visible_refs(graph, lam_restriction=lam_restriction)

    if not visible_refs:
        print("No vertices satisfy the lam_restriction")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Edges first, nodes on top.
    _draw_edges(
        ax,
        graph,
        positions,
        visible_refs=visible_refs,
        node_size=node_size,
        arrowsize=10,
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

    cmap = plt.cm.get_cmap("viridis")
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
        _draw_edge_labels(
            ax,
            graph,
            positions,
            visible_refs=visible_refs,
            font_size=max(5, font_size - 1),
            label_mode=label_edges,
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

    xmin, xmax = min(all_xs), max(all_xs)
    ymin, ymax = min(all_ys), max(all_ys)

    if xmin == xmax:
        xmin -= xpad
        xmax += xpad
    else:
        xmin -= xpad
        xmax += xpad

    if ymin == ymax:
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
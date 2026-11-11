"""Візуалізація результатів LLM-контролера (лабораторна робота 5)."""

import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from sdn_env import parse_address, node_address


def render_state_ascii(graph, path, source, destination, head=None):
    """ASCII-зображення сітки з виділеним шляхом і головою потоку."""
    side = graph.graph["side"]
    src, dst = parse_address(source), parse_address(destination)
    path_edges = {frozenset(e) for e in zip(path, path[1:])}

    def symbol(n):
        if n == head:
            return "@"
        if n == src:
            return "S"
        if n == dst:
            return "D"
        if n in path:
            return "#"
        return "."

    lines = []
    for row in range(side):
        line = ""
        for col in range(side):
            line += symbol((row, col))
            if col + 1 < side:
                u, v = (row, col), (row, col + 1)
                if graph.has_edge(u, v):
                    edge = frozenset((u, v))
                    ch = "=" if edge in path_edges else "-"
                    line += f"{ch}{graph[u][v]['weight']:>2}{ch}"
                else:
                    line += "    "
        lines.append(line)
        if row + 1 < side:
            line = ""
            for col in range(side):
                u, v = (row, col), (row + 1, col)
                if graph.has_edge(u, v):
                    edge = frozenset((u, v))
                    ch = "!" if edge in path_edges else "|"
                    line += f"{ch}{graph[u][v]['weight']:<2}  "
                else:
                    line += "     "
            lines.append(line)
    return "\n".join(lines)


def draw_state(graph, path, source, destination, ax, title, head=None):
    """Малює сітку, виділяє шлях і голову потоку, підписує затримки (мс)."""
    pos = {n: (d["col"], -d["row"]) for n, d in graph.nodes(data=True)}
    src, dst = parse_address(source), parse_address(destination)
    path_edges = {frozenset(e) for e in zip(path, path[1:])}

    def color(n):
        if n == head:
            return "orange"
        if n == src:
            return "lightgreen"
        if n == dst:
            return "lightcoral"
        if n in path:
            return "khaki"
        return "white"

    on_path = [(u, v) for u, v in graph.edges if frozenset((u, v)) in path_edges]
    off_path = [(u, v) for u, v in graph.edges if frozenset((u, v)) not in path_edges]

    nx.draw_networkx_edges(graph, pos, edgelist=off_path, ax=ax, width=1.5,
                           edge_color="lightgray")
    nx.draw_networkx_edges(graph, pos, edgelist=on_path, ax=ax, width=4,
                           edge_color="forestgreen")
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_size=1500, node_color=[color(n) for n in graph.nodes],
        edgecolors=["blue" if n in (src, dst) else "black" for n in graph.nodes],
        linewidths=[3 if n in (src, dst) else 1 for n in graph.nodes],
    )
    labels = {n: node_address(n) + ("\nS" if n == src else "\nD" if n == dst else "")
              for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7)
    nx.draw_networkx_edge_labels(
        graph, pos, ax=ax, font_size=9, font_color="crimson",
        edge_labels={(u, v): graph[u][v]["weight"] for u, v in graph.edges},
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
    )
    ax.set_title(title)
    ax.set_axis_off()


def plot_paths(graph, results, source, destination, filename=None):
    """Два графіки: шлях квантованої та неквантованої моделі."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    q = results["quantized"]
    f = results["fp16"]
    q_title = (f"Квантована модель: шлях {q['delay']} мс ({q['nodes']} вузлів)"
               if q.get("finished") else "Квантована модель: не завершено")
    f_title = (f"Неквантована модель (FP16): шлях {f['delay']} мс ({f['nodes']} вузлів)"
               if f.get("finished") else "Неквантована модель (FP16): не завершено")
    draw_state(graph, q["path"], source, destination, axes[0], q_title)
    draw_state(graph, f["path"], source, destination, axes[1], f_title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.show()
    return fig


def action_label(action):
    """Короткий підпис дії агента, напр. install_flow(S) або get_link_stats(10.0.1.0)."""
    name = action["tool"]
    args = action.get("arguments", {})
    if "direction" in args:
        return f"{name}({args['direction']})"
    if "node" in args:
        return f"{name}({args['node']})"
    return name


def plot_step_sequence(graph, snapshots, source, destination, title=None,
                       filename=None, cols=6):
    """Сітка графіків: по одному на кожен крок (стан мережі після кроку)."""
    n = len(snapshots)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.2))
    axes = np.asarray(axes).ravel()
    for i, snap in enumerate(snapshots):
        head = parse_address(snap["head"]) if snap.get("head") else None
        path = [tuple(p) for p in snap["path"]]
        actions = ", ".join(action_label(a) for a in snap.get("actions", [])) or "—"
        label = (f"Крок {snap['step']}: {actions}\n"
                 f"голова {snap['head']} | правил {snap.get('installed', 0)}, "
                 f"відкликано {snap.get('revoked', 0)}")
        draw_state(graph, path, source, destination, axes[i], label, head=head)
    for j in range(n, len(axes)):
        axes[j].set_axis_off()
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if filename:
        plt.savefig(filename, dpi=110, bbox_inches="tight")
    plt.show()
    return fig

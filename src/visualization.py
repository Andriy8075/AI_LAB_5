"""Візуалізація результатів LLM-контролера (лабораторна робота 5)."""

import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from sdn_env import parse_address, node_address


def render_state_ascii(graph, path, source, destination, head=None):
    """ASCII-зображення сітки з виділеним шляхом і головою потоку."""
    side = graph.graph["side"]
    src, dst = parse_address(source), parse_address(destination)
    path_nodes = [
        parse_address(p) if isinstance(p, str) else tuple(p) for p in path
    ]
    path_edges = {frozenset(e) for e in zip(path_nodes, path_nodes[1:])}
    if head is not None and isinstance(head, str):
        head = parse_address(head)

    def symbol(n):
        if n == head:
            return "@"
        if n == src:
            return "S"
        if n == dst:
            return "D"
        if n in path_nodes:
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
    path_nodes = [
        parse_address(p) if isinstance(p, str) else tuple(p) for p in path
    ]
    path_edges = {frozenset(e) for e in zip(path_nodes, path_nodes[1:])}
    if head is not None and isinstance(head, str):
        head = parse_address(head)

    def color(n):
        if n == head:
            return "orange"
        if n == src:
            return "lightgreen"
        if n == dst:
            return "lightcoral"
        if n in path_nodes:
            return "khaki"
        return "white"

    on_path = [(u, v) for u, v in graph.edges if frozenset((u, v)) in path_edges]
    off_path = [(u, v) for u, v in graph.edges if frozenset((u, v)) not in path_edges]

    nx.draw_networkx_edges(graph, pos, edgelist=off_path, ax=ax, width=1.5,
                           edge_color="lightgray")
    nx.draw_networkx_edges(graph, pos, edgelist=on_path, ax=ax, width=4,
                           edge_color="forestgreen")
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_size=1200, node_color=[color(n) for n in graph.nodes],
        edgecolors=["blue" if n in (src, dst) else "black" for n in graph.nodes],
        linewidths=[3 if n in (src, dst) else 1 for n in graph.nodes],
    )
    labels = {
        n: node_address(n) + ("\nS" if n == src else "\nD" if n == dst else "")
        for n in graph.nodes
    }
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=6)
    nx.draw_networkx_edge_labels(
        graph, pos, ax=ax, font_size=8, font_color="crimson",
        edge_labels={(u, v): graph[u][v]["weight"] for u, v in graph.edges},
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
    )
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()


def plot_paths(graph, results, source, destination, filename=None):
    """Два графіки: шляхи двох прогонів (напр. Q4 vs Q8)."""
    keys = list(results.keys())[:2]
    fig, axes = plt.subplots(1, max(2, len(keys)), figsize=(18, 9))
    axes = np.asarray(axes).ravel()
    for i, key in enumerate(keys):
        r = results[key]
        path = r.get("path") or []
        delay = r.get("delay")
        if delay is None and r.get("last_path_result"):
            delay = r["last_path_result"].get("delay_ms")
        title = f"{key}: шлях {delay} мс" if delay is not None else f"{key}"
        draw_state(graph, path, source, destination, axes[i], title)
    for j in range(len(keys), len(axes)):
        axes[j].set_axis_off()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.show()
    return fig


def plot_single_path(graph, path, source, destination, title=None, filename=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    draw_state(graph, path, source, destination, ax, title or "Шлях")
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return filename


def action_label(action):
    name = action["tool"]
    args = action.get("arguments", {})
    if "interface" in args:
        return f"{name}({args['interface']})"
    if "direction" in args:
        return f"{name}({args['direction']})"
    if "node" in args:
        return f"{name}({args['node']})"
    if "src" in args and "dst" in args:
        return f"{name}({args['src']}→{args['dst']})"
    if "a" in args and "b" in args:
        return f"{name}({args['a']}—{args['b']})"
    return name


def plot_step_sequence(graph, snapshots, source, destination, title=None,
                       filename=None, cols=6):
    """Залишено для сумісності: сітка кроків (краще використовувати save_step_figures)."""
    n = len(snapshots)
    if n == 0:
        return None
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.2))
    axes = np.asarray(axes).ravel()
    for i, snap in enumerate(snapshots):
        head = snap.get("head")
        path = snap.get("path") or []
        actions = ", ".join(action_label(a) for a in snap.get("actions", [])) or "—"
        label = (
            f"Крок {snap['step']}: {actions}\n"
            f"голова {snap['head']} | правил {snap.get('installed', 0)}, "
            f"відкликано {snap.get('revoked', 0)}"
        )
        draw_state(graph, path, source, destination, axes[i], label, head=head)
    for j in range(n, len(axes)):
        axes[j].set_axis_off()
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if filename:
        plt.savefig(filename, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return fig


def save_step_figures(graph, snapshots, source, destination, out_dir,
                      prefix="step"):
    """Зберігає ОКРЕМЕ зображення мережі на кожен крок."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for snap in snapshots:
        step = snap.get("step", len(paths) + 1)
        head = snap.get("head")
        path = snap.get("path") or []
        actions = ", ".join(action_label(a) for a in snap.get("actions", [])) or "—"
        label = (
            f"Крок {step}: {actions}\n"
            f"голова {head} | правил {snap.get('installed', 0)}, "
            f"відкликано {snap.get('revoked', 0)}"
        )
        fname = os.path.join(out_dir, f"{prefix}_{step:02d}.png")
        fig, ax = plt.subplots(1, 1, figsize=(10, 9))
        draw_state(graph, path, source, destination, ax, label, head=head)
        plt.tight_layout()
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(fname)
    return paths

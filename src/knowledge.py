"""База знань і knowledge-based контролер для побудови повного шляху.

LLM не обчислює маршрут сама: install_path викликає цей модуль.
Пропускні здатності беруться з атрибутів ребер (як «розпізнані») без ONNX.
"""

import networkx as nx

from sdn_env import (
    DIRECTIONS,
    FLOW_SIZE_BYTES,
    WEIGHT_RANGE,
    direction_between,
    digit_to_bandwidth,
    get_interfaces,
    node_address,
    parse_address,
    transmission_time_ms,
)


class LinkStateDatabase:
    """База знань контролера: відомі вузли, канали, пропускні здатності."""

    def __init__(self):
        self.links = {}       # frozenset({u,v}) -> delay_ms
        self.bandwidths = {}  # frozenset({u,v}) -> MBps
        self.complete = set()
        self.nodes = set()
        self.failed = set()   # frozenset edges that are down

    def seed_from_graph(self, graph, env=None):
        """Заповнює базу повним знанням топології (еталон для install_path)."""
        weight_fn = env._weight if env is not None else None
        bw_fn = env._bandwidth if env is not None else None
        for node in graph.nodes:
            self.nodes.add(node)
            self.complete.add(node)
        for u, v, data in graph.edges(data=True):
            edge = frozenset((u, v))
            if weight_fn is not None:
                w = weight_fn(u, v)
                if w is None:
                    self.failed.add(edge)
                    continue
            else:
                w = data["weight"]
            self.links[edge] = w
            if bw_fn is not None:
                bw = bw_fn(u, v)
            else:
                bw = digit_to_bandwidth(data["bw_digit"])
            if bw is not None:
                self.bandwidths[edge] = bw

    def remember_link(self, u, v, delay_ms, bandwidth_MBps=None):
        edge = frozenset((u, v))
        self.nodes.add(u)
        self.nodes.add(v)
        self.links[edge] = delay_ms
        self.failed.discard(edge)
        if bandwidth_MBps is not None:
            self.bandwidths[edge] = bandwidth_MBps

    def mark_failed(self, u, v):
        edge = frozenset((u, v))
        self.nodes.add(u)
        self.nodes.add(v)
        self.failed.add(edge)
        self.links.pop(edge, None)

    def absorb_link_stats(self, stats):
        """Оновлює базу з відповіді get_link_stats."""
        node = parse_address(stats["node"])
        self.nodes.add(node)
        self.complete.add(node)
        for direction, info in stats.get("links", {}).items():
            nbr = parse_address(info["neighbor"])
            self.remember_link(
                node, nbr, info["delay_ms"], info.get("bandwidth_MBps")
            )

    def as_graph(self, avoid=None):
        known = nx.Graph()
        known.add_nodes_from(self.nodes)
        blocked = set()
        if avoid:
            for a in avoid:
                blocked.add(parse_address(a) if isinstance(a, str) else a)
        for edge, weight in self.links.items():
            if edge in self.failed:
                continue
            u, v = tuple(edge)
            if u in blocked or v in blocked:
                continue
            known.add_edge(u, v, weight=weight)
        return known

    def dump(self, compact=True):
        channels = []
        for edge, delay in sorted(
            self.links.items(),
            key=lambda kv: (sorted(kv[0])[0], sorted(kv[0])[1]),
        ):
            u, v = sorted(edge)
            bw = self.bandwidths.get(edge)
            if compact:
                channels.append(
                    f"{node_address(u)}-{node_address(v)}:"
                    f"{delay}ms/{bw}MBps"
                )
            else:
                channels.append({
                    "a": node_address(u),
                    "b": node_address(v),
                    "delay_ms": delay,
                    "bandwidth_MBps": bw,
                })
        return {
            "num_nodes": len(self.nodes),
            "num_channels": len(self.links),
            "nodes": sorted(node_address(n) for n in self.nodes),
            "channels": channels,
            "failed": [
                f"{node_address(sorted(e)[0])} — {node_address(sorted(e)[1])}"
                for e in self.failed
            ],
        }


class KnowledgeController:
    """Детермінований контролер: будує повний шлях через базу знань."""

    def __init__(self, env, db=None, min_weight=WEIGHT_RANGE[0]):
        self.env = env
        self.db = db if db is not None else LinkStateDatabase()
        self.min_weight = min_weight
        self.journal = []

    def ensure_seeded(self):
        if not self.db.links:
            self.db.seed_from_graph(self.env._graph, self.env)

    def install_full_path(self, src, dst, avoid=None, flow_bytes=FLOW_SIZE_BYTES):
        """Будує й встановлює повний шлях; повертає шлях, delay, час передачі."""
        self.ensure_seeded()
        avoid = list(avoid or [])
        src_n = parse_address(src)
        dst_n = parse_address(dst)

        if src_n not in self.env._graph or dst_n not in self.env._graph:
            raise ValueError(f"Вузол поза топологією: {src} або {dst}")
        for a in avoid:
            an = parse_address(a)
            if an not in self.env._graph:
                raise ValueError(f"Вузол у avoid не існує: {a}")

        # Скидаємо попередній потік і ставимо голову на src
        self.env.clear_flow()
        self.env.source = src_n
        self.env.destination = dst_n
        self.env.head = src_n
        self.env._recompute_costs()
        self.journal = []

        known = self.db.as_graph(avoid=avoid)
        # також прибираємо реально упалі канали з live graph
        live = self.env.live_graph()
        # перетин: лише ребра, відомі й живі, без avoid
        g = nx.Graph()
        blocked = {parse_address(a) for a in avoid}
        blocked -= {src_n, dst_n}
        for n in live.nodes:
            if n not in blocked:
                g.add_node(n)
        for u, v, data in live.edges(data=True):
            if u in blocked or v in blocked:
                continue
            if known.has_edge(u, v) or frozenset((u, v)) in self.db.links:
                g.add_edge(u, v, weight=data["weight"])

        if src_n not in g or dst_n not in g or not nx.has_path(g, src_n, dst_n):
            raise ValueError(
                f"Неможливо прокласти шлях {src} → {dst}"
                + (f" в обхід {avoid}" if avoid else "")
            )

        path = nx.shortest_path(g, src_n, dst_n, weight="weight")
        snapshots = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            direction = direction_between(u, v)
            delay = self.env._weight(u, v)
            bw = self.env._bandwidth(u, v)
            self.db.remember_link(u, v, delay, bw)
            self.env.install_rule(direction)
            entry = {
                "step": i + 1,
                "action": "install",
                "interface": direction,
                "from": node_address(u),
                "to": node_address(v),
                "delay_ms": delay,
                "bandwidth_MBps": bw,
                "head": node_address(self.env.head),
                "path": [node_address(n) for n in self.env.path()],
                "installed": self.env.installed,
                "revoked": self.env.revoked,
            }
            self.journal.append(entry)
            snapshots.append({
                "step": i + 1,
                "head": node_address(self.env.head),
                "path": [node_address(n) for n in self.env.path()],
                "installed": self.env.installed,
                "revoked": self.env.revoked,
                "actions": [{
                    "tool": "install_rule",
                    "arguments": {"interface": direction},
                    "status": "ok",
                }],
            })

        delay_total = self.env.path_delay()
        transfer = self.env.transfer_stats(flow_bytes)
        opt_delay = None
        opt_path = None
        try:
            live_full = self.env.live_graph()
            if nx.has_path(live_full, src_n, dst_n):
                opt_path = nx.shortest_path(live_full, src_n, dst_n, weight="weight")
                opt_delay = nx.shortest_path_length(
                    live_full, src_n, dst_n, weight="weight"
                )
        except Exception:
            pass

        return {
            "path": [node_address(n) for n in path],
            "delay_ms": delay_total,
            "transmission_ms": transfer["transmission_ms"],
            "total_ms": transfer["total_ms"],
            "bottleneck": transfer["bottleneck"],
            "min_bandwidth_MBps": transfer["min_bandwidth_MBps"],
            "links": transfer["links"],
            "avoid": avoid,
            "installed_rules": self.env.installed,
            "revoked_rules": self.env.revoked,
            "journal": list(self.journal),
            "snapshots": snapshots,
            "optimal_delay_ms": opt_delay,
            "optimal_path": (
                [node_address(n) for n in opt_path] if opt_path else None
            ),
            "flow_state": self.env.flow_state(),
        }

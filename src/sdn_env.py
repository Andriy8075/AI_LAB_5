"""Середовище (площина даних) для лабораторної роботи 5.

Топологія-сітка 5×5 з затримками та пропускною здатністю каналів
(як у лабораторних роботах 2–4). Керування — через інструменти LLM
(див. tools.py, knowledge.py, llm_controller.py).
"""

import math
import random

import networkx as nx

DIRECTIONS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
DELTA_TO_DIR = {delta: name for name, delta in DIRECTIONS.items()}

NUM_NODES = 25
EDGES_TO_REMOVE = 8
WEIGHT_RANGE = (1, 10)
SEED = 42

SOURCE = "10.0.0.0"
DESTINATION = "10.0.4.4"
PREFIX = "10.0"

FLOW_SIZE_BYTES = 10**9
DIGIT_MIN, DIGIT_MAX = 2, 9
BW_MULTIPLIER = 100


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def digit_to_bandwidth(digit):
    return digit * BW_MULTIPLIER


def transmission_time_ms(flow_bytes, bandwidth_MBps):
    return flow_bytes / (bandwidth_MBps * 1e3)


def generate_grid_graph(num_nodes=NUM_NODES, edges_to_remove=EDGES_TO_REMOVE,
                        weight_range=WEIGHT_RANGE, seed=SEED):
    """Генерує зв'язну прямокутну сітку з затримками та bw_digit."""
    side = math.isqrt(num_nodes)
    if side * side != num_nodes:
        raise ValueError(f"num_nodes має бути повним квадратом, отримано {num_nodes}")

    low, high = weight_range
    if not (0 < low <= high):
        raise ValueError(f"Некоректний діапазон ваг: {weight_range}")

    rng = random.Random(seed)

    graph = nx.Graph(side=side)
    for row in range(side):
        for col in range(side):
            graph.add_node((row, col), row=row, col=col)

    edges = []
    for row in range(side):
        for col in range(side):
            if col + 1 < side:
                edges.append(((row, col), (row, col + 1)))
            if row + 1 < side:
                edges.append(((row, col), (row + 1, col)))

    max_removable = len(edges) - (num_nodes - 1)
    if not (0 <= edges_to_remove <= max_removable):
        raise ValueError(
            f"edges_to_remove має бути в межах 0..{max_removable}, отримано {edges_to_remove}"
        )

    shuffled = edges[:]
    rng.shuffle(shuffled)
    parent = {node: node for node in graph.nodes}
    extra_edges = []
    for u, v in shuffled:
        root_u, root_v = _find(parent, u), _find(parent, v)
        if root_u != root_v:
            parent[root_u] = root_v
            graph.add_edge(u, v)
        else:
            extra_edges.append((u, v))

    kept_extra = rng.sample(extra_edges, len(extra_edges) - edges_to_remove)
    graph.add_edges_from(kept_extra)

    for u, v in graph.edges:
        graph[u][v]["weight"] = rng.randint(low, high)
        graph[u][v]["bw_digit"] = rng.randint(DIGIT_MIN, DIGIT_MAX)

    return graph


def node_address(node, prefix=PREFIX):
    row, col = node
    return f"{prefix}.{row}.{col}"


def parse_address(address):
    parts = str(address).split(".")
    if len(parts) != 4:
        raise ValueError(f"Некоректна адреса вузла: {address!r}")
    return int(parts[2]), int(parts[3])


def get_interfaces(graph, node, weight_fn=None):
    """Сусідні вузли за напрямками N/E/S/W (з урахуванням упалих каналів)."""
    row, col = node
    directions = {
        "N": (row - 1, col),
        "E": (row, col + 1),
        "S": (row + 1, col),
        "W": (row, col - 1),
    }
    result = {}
    for d, nbr in directions.items():
        if not graph.has_edge(node, nbr):
            continue
        if weight_fn is not None and weight_fn(node, nbr) is None:
            continue
        result[d] = nbr
    return result


def direction_between(src, dst):
    return DELTA_TO_DIR[(dst[0] - src[0], dst[1] - src[1])]


def heuristic(node, destination, min_weight):
    return min_weight * (abs(node[0] - destination[0]) + abs(node[1] - destination[1]))


def manhattan(node, destination):
    return abs(node[0] - destination[0]) + abs(node[1] - destination[1])


def shortest_path_delay(graph, source, destination, avoid=None):
    """Оптимальна затримка та шлях (Дейкстра), з опційним обходом вузлів."""
    s, d = parse_address(source), parse_address(destination)
    g = graph
    if avoid:
        blocked = {parse_address(a) if isinstance(a, str) else a for a in avoid}
        blocked -= {s, d}
        g = graph.copy()
        g.remove_nodes_from(blocked)
    if s not in g or d not in g or not nx.has_path(g, s, d):
        raise ValueError("Шлях між вузлами не існує (з урахуванням avoid)")
    path = nx.shortest_path(g, s, d, weight="weight")
    delay = nx.shortest_path_length(g, s, d, weight="weight")
    return delay, path


class NetworkEnvironment:
    """Площина даних: голова потоку, правила, збої каналів."""

    def __init__(self, graph, source, destination):
        self._graph = graph
        self.source = parse_address(source)
        self.destination = parse_address(destination)
        self.head = self.source
        self.rules = []
        self.installed = 0
        self.revoked = 0
        self._overrides = {}  # frozenset({u,v}) -> weight або None (down)
        self._recompute_costs()

    def _recompute_costs(self):
        """Найкоротші затримки до отримувача з урахуванням overrides."""
        live = self.live_graph()
        if self.destination in live:
            self._cost_to_dst = nx.single_source_dijkstra_path_length(
                live, self.destination, weight="weight"
            )
        else:
            self._cost_to_dst = {}

    def live_graph(self):
        """Граф без упалих каналів, з оновленими вагами."""
        g = nx.Graph()
        g.add_nodes_from(self._graph.nodes(data=True))
        for u, v, data in self._graph.edges(data=True):
            w = self._weight(u, v)
            if w is None:
                continue
            attrs = dict(data)
            attrs["weight"] = w
            g.add_edge(u, v, **attrs)
        g.graph.update(self._graph.graph)
        return g

    def _weight(self, u, v):
        edge = frozenset((u, v))
        if edge in self._overrides:
            return self._overrides[edge]
        if not self._graph.has_edge(u, v):
            return None
        return self._graph[u][v]["weight"]

    def _bandwidth(self, u, v):
        if self._weight(u, v) is None:
            return None
        return digit_to_bandwidth(self._graph[u][v]["bw_digit"])

    def _links_of(self, node):
        return {
            direction: self._weight(node, nbr)
            for direction, nbr in get_interfaces(
                self._graph, node, weight_fn=self._weight
            ).items()
        }

    def link_stats(self, node):
        """Телеметрія вузла: delay + bandwidth сусідніх каналів."""
        stats = {}
        for direction, nbr in get_interfaces(
            self._graph, node, weight_fn=self._weight
        ).items():
            stats[direction] = {
                "neighbor": node_address(nbr),
                "delay_ms": self._weight(node, nbr),
                "bandwidth_MBps": self._bandwidth(node, nbr),
                "cost_to_destination": self._cost_to_dst.get(nbr, 10**9),
            }
        return {
            "node": node_address(node),
            "cost_to_destination": self._cost_to_dst.get(node, 10**9),
            "links": stats,
        }

    def telemetry(self):
        neighbor_lsas = {}
        for _, nbr in get_interfaces(self._graph, self.head, weight_fn=self._weight).items():
            neighbor_lsas[node_address(nbr)] = self._links_of(nbr)
        return {
            "node": node_address(self.head),
            "links": self._links_of(self.head),
            "neighbor_lsas": neighbor_lsas,
        }

    def install_rule(self, direction):
        direction = str(direction).upper()
        interfaces = get_interfaces(self._graph, self.head, weight_fn=self._weight)
        if direction not in interfaces:
            raise ValueError(
                f"Немає каналу {direction} на вузлі {node_address(self.head)}"
            )
        nxt = interfaces[direction]
        self.rules.append((self.head, direction, nxt))
        self.head = nxt
        self.installed += 1

    def revoke_last_rule(self):
        if not self.rules:
            raise RuntimeError("Немає правил для відкликання")
        node, _, _ = self.rules.pop()
        self.head = node
        self.revoked += 1

    def clear_flow(self):
        """Скидає встановлений потік (голова на відправнику)."""
        self.head = self.source
        self.rules = []
        self.installed = 0
        self.revoked = 0

    def degrade_link(self, u, v, factor=None, down=False):
        """Збільшити затримку каналу або вивести його з ладу."""
        u = parse_address(u) if isinstance(u, str) else u
        v = parse_address(v) if isinstance(v, str) else v
        old = self._weight(u, v)
        if old is None and not self._graph.has_edge(u, v):
            raise ValueError(
                f"Канал {node_address(u)} — {node_address(v)} не існує"
            )
        if down:
            self._overrides[frozenset((u, v))] = None
            self._recompute_costs()
            return old, None
        if factor is None:
            raise ValueError("Потрібен factor або down=True")
        new = old * factor
        self._overrides[frozenset((u, v))] = new
        self._recompute_costs()
        return old, new

    def path(self):
        return [self.source] + [nxt for _, _, nxt in self.rules]

    def path_delay(self):
        path = self.path()
        return sum(self._weight(u, v) for u, v in zip(path, path[1:]))

    def path_bandwidths(self):
        path = self.path()
        result = []
        for u, v in zip(path, path[1:]):
            result.append({
                "a": node_address(u),
                "b": node_address(v),
                "delay_ms": self._weight(u, v),
                "bandwidth_MBps": self._bandwidth(u, v),
            })
        return result

    def transfer_stats(self, flow_bytes=FLOW_SIZE_BYTES):
        """Час передачі та вузьке місце останнього потоку."""
        links = self.path_bandwidths()
        if not links:
            return {
                "path": [node_address(n) for n in self.path()],
                "delay_ms": 0,
                "bottleneck": None,
                "min_bandwidth_MBps": None,
                "transmission_ms": None,
                "total_ms": None,
                "flow_bytes": flow_bytes,
                "links": [],
            }
        bottleneck = min(links, key=lambda x: x["bandwidth_MBps"])
        min_bw = bottleneck["bandwidth_MBps"]
        tx_ms = transmission_time_ms(flow_bytes, min_bw)
        delay = self.path_delay()
        return {
            "path": [node_address(n) for n in self.path()],
            "delay_ms": delay,
            "bottleneck": f"{bottleneck['a']} — {bottleneck['b']}",
            "min_bandwidth_MBps": min_bw,
            "transmission_ms": round(tx_ms, 2),
            "total_ms": round(delay + tx_ms, 2),
            "flow_bytes": flow_bytes,
            "links": links,
        }

    def flow_state(self):
        return {
            "head": node_address(self.head),
            "source": node_address(self.source),
            "destination": node_address(self.destination),
            "cost_to_destination": self._cost_to_dst.get(self.head, 10**9),
            "path": [node_address(n) for n in self.path()],
            "rules": [
                {
                    "node": node_address(node),
                    "interface": direction,
                    "next": node_address(nxt),
                }
                for node, direction, nxt in self.rules
            ],
            "installed_rules": self.installed,
            "revoked_rules": self.revoked,
            "at_destination": self.head == self.destination,
        }


def greedy_route(graph, source, destination):
    """Детермінований жадібний маршрут (еталон)."""
    env = NetworkEnvironment(graph, source, destination)
    while env.head != env.destination:
        head = env.head
        best = None
        for direction, nbr in get_interfaces(
            graph, head, weight_fn=env._weight
        ).items():
            cost = env._cost_to_dst.get(nbr, 10**9)
            if best is None or cost < best[1]:
                best = (direction, cost)
        if best is None:
            return env.path(), env.path_delay(), False
        env.install_rule(best[0])
    return env.path(), env.path_delay(), True

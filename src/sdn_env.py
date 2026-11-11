"""Середовище (площина даних) для лабораторної роботи 5.

Генерація топології-сітки та мережеве середовище з правилами пересилання.
Код топології повторює лабораторні роботи 2-4, але без розпізнавання
пропускної здатності: тут контролером керує велика мовна модель через
виклики інструментів (див. tools.py та llm_controller.py).
"""

import math
import random

import networkx as nx

# Напрямки руху головою потоку та зворотне відображення.
DIRECTIONS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
DELTA_TO_DIR = {delta: name for name, delta in DIRECTIONS.items()}

NUM_NODES = 16
EDGES_TO_REMOVE = 3
WEIGHT_RANGE = (1, 10)
SEED = 42

SOURCE = "10.0.0.0"
DESTINATION = "10.0.3.3"
PREFIX = "10.0"


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def generate_grid_graph(num_nodes=NUM_NODES, edges_to_remove=EDGES_TO_REMOVE,
                        weight_range=WEIGHT_RANGE, seed=SEED):
    """Генерує зв'язну прямокутну сітку з випадковими вагами (затримками)."""
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

    return graph

def node_address(node, prefix=PREFIX):
    """(рядок, стовпець) -> '10.0.<рядок>.<стовпець>'."""
    row, col = node
    return f"{prefix}.{row}.{col}"


def parse_address(address):
    """'10.0.<рядок>.<стовпець>' -> (рядок, стовпець)."""
    parts = address.split(".")
    return int(parts[2]), int(parts[3])


def get_interfaces(graph, node):
    """Сусідні вузли вузла за напрямками N/E/S/W."""
    row, col = node
    directions = {
        "N": (row - 1, col),
        "E": (row, col + 1),
        "S": (row + 1, col),
        "W": (row, col - 1),
    }
    return {d: nbr for d, nbr in directions.items() if graph.has_edge(node, nbr)}


def direction_between(src, dst):
    return DELTA_TO_DIR[(dst[0] - src[0], dst[1] - src[1])]


def heuristic(node, destination, min_weight):
    """Нижня межа залишкової затримки."""
    return min_weight * (abs(node[0] - destination[0]) + abs(node[1] - destination[1]))


def manhattan(node, destination):
    """Манхеттенська відстань між вузлами (за координатами сітки)."""
    return abs(node[0] - destination[0]) + abs(node[1] - destination[1])


def shortest_path_delay(graph, source, destination):
    """Оптимальна затримка та шлях (Дейкстра) на повному графі."""
    s, d = parse_address(source), parse_address(destination)
    path = nx.shortest_path(graph, s, d, weight="weight")
    delay = nx.shortest_path_length(graph, s, d, weight="weight")
    return delay, path


class NetworkEnvironment:
    """Площина даних: зберігає голову потоку та встановлені правила."""

    def __init__(self, graph, source, destination):
        self._graph = graph
        self.source = parse_address(source)
        self.destination = parse_address(destination)
        self.head = self.source
        self.rules = []
        self.installed = 0
        self.revoked = 0
        # Найкоротша затримка від кожного вузла до отримувача (Дейкстра).
        # Граф неорієнтований, тому відстані від отримувача до всіх вузлів
        # дорівнюють відстаням від вузлів до отримувача.
        self._cost_to_dst = nx.single_source_dijkstra_path_length(
            graph, self.destination, weight="weight"
        )

    # --- низькорівневі операції над графом ---------------------------------
    def _weight(self, u, v):
        return self._graph[u][v]["weight"]

    def _links_of(self, node):
        return {
            direction: self._weight(node, nbr)
            for direction, nbr in get_interfaces(self._graph, node).items()
        }

    def link_stats(self, node):
        """Звіт про сусідні канали вузла: напрямок -> {сусід, затримка, вартість}."""
        stats = {}
        for direction, nbr in get_interfaces(self._graph, node).items():
            stats[direction] = {
                "neighbor": node_address(nbr),
                "delay_ms": self._weight(node, nbr),
                "cost_to_destination": self._cost_to_dst.get(nbr, 10**9),
            }
        return {
            "node": node_address(node),
            "cost_to_destination": self._cost_to_dst.get(node, 10**9),
            "links": stats,
        }

    # --- дії контролера ----------------------------------------------------
    def install_rule(self, direction):
        interfaces = get_interfaces(self._graph, self.head)
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

    # --- стан потоку -------------------------------------------------------
    def path(self):
        return [self.source] + [nxt for _, _, nxt in self.rules]

    def path_delay(self):
        path = self.path()
        return sum(self._weight(u, v) for u, v in zip(path, path[1:]))

    def flow_state(self):
        """Стан потоку для інструмента get_flow_state."""
        return {
            "head": node_address(self.head),
            "source": node_address(self.source),
            "destination": node_address(self.destination),
            "cost_to_destination": self._cost_to_dst.get(self.head, 10**9),
            "path": [node_address(n) for n in self.path()],
            "installed_rules": self.installed,
            "revoked_rules": self.revoked,
            "at_destination": self.head == self.destination,
        }



def greedy_route(graph, source, destination):
    """Детермінований жадібний маршрут (еталон для порівняння).

    На кожному кроці обирає сусіда з найменшою залишковою найкоротшою
    затримкою до отримувача (cost_to_destination). Такий маршрут збігається
    з оптимальним шляхом Дейкстри.
    """
    env = NetworkEnvironment(graph, source, destination)
    while env.head != env.destination:
        head = env.head
        best = None
        for direction, nbr in get_interfaces(graph, head).items():
            cost = env._cost_to_dst.get(nbr, 10**9)
            if best is None or cost < best[1]:
                best = (direction, cost)
        if best is None:
            return env.path(), env.path_delay(), False  # глухий кут
        env.install_rule(best[0])
    return env.path(), env.path_delay(), True

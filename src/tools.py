"""Інструменти SDN-контролера для LLM (tool calling) — частина 2 PDF.

Мінімальний набір:
  get_flow_state, get_link_stats, install_rule, install_path,
  get_topology, simulate_link_failure.
"""

import json

from knowledge import KnowledgeController, LinkStateDatabase
from sdn_env import parse_address


class ToolError(Exception):
    """Помилка виконання інструмента (передається моделі як текст)."""


class SdnToolbox:
    """Зв'язує інструменти з середовищем і базою знань."""

    def __init__(self, env, db=None, controller=None):
        self.env = env
        self.db = db if db is not None else LinkStateDatabase()
        self.controller = controller or KnowledgeController(env, self.db)
        self.call_log = []
        self.last_path_result = None
        # Повне знання топології для knowledge-based install_path
        self.controller.ensure_seeded()

    def get_flow_state(self):
        state = self.env.flow_state()
        transfer = self.env.transfer_stats()
        return {
            "head": state["head"],
            "destination": state["destination"],
            "source": state["source"],
            "path": state["path"],
            "rules": state["rules"],
            "installed_rules": state["installed_rules"],
            "revoked_rules": state["revoked_rules"],
            "at_destination": state["at_destination"],
            "cost_to_destination": state["cost_to_destination"],
            "delay_ms": transfer.get("delay_ms"),
            "transmission_ms": transfer.get("transmission_ms"),
            "total_ms": transfer.get("total_ms"),
            "bottleneck": transfer.get("bottleneck"),
            "min_bandwidth_MBps": transfer.get("min_bandwidth_MBps"),
        }

    def get_link_stats(self, node: str):
        try:
            parsed = parse_address(node)
        except (ValueError, IndexError, TypeError):
            raise ToolError(f"Некоректна адреса вузла: {node!r}")
        if parsed not in self.env._graph:
            raise ToolError(f"Вузол не існує в топології: {node}")
        stats = self.env.link_stats(parsed)
        self.db.absorb_link_stats(stats)
        return stats

    def install_rule(self, interface: str):
        """Встановлює правило N/E/S/W або відкликає останнє (interface=revoke)."""
        iface = str(interface).strip().upper()
        if iface in ("REVOKE", "UNDO", "BACK"):
            try:
                self.env.revoke_last_rule()
            except RuntimeError as exc:
                raise ToolError(str(exc))
            return {"action": "revoke", **self.env.flow_state()}
        if iface not in ("N", "E", "S", "W"):
            raise ToolError(
                f"Невідомий інтерфейс {interface!r}. Очікується N/E/S/W або revoke."
            )
        try:
            self.env.install_rule(iface)
        except ValueError as exc:
            raise ToolError(str(exc))
        # запам'ятати канал у KB
        if self.env.rules:
            node, direction, nxt = self.env.rules[-1]
            self.db.remember_link(
                node, nxt,
                self.env._weight(node, nxt),
                self.env._bandwidth(node, nxt),
            )
        return {"action": "install", "interface": iface, **self.env.flow_state()}

    def install_path(self, src: str, dst: str, avoid=None):
        avoid_list = []
        if avoid is not None:
            if isinstance(avoid, str):
                avoid = [a.strip() for a in avoid.split(",") if a.strip()]
            avoid_list = list(avoid)
        try:
            result = self.controller.install_full_path(src, dst, avoid=avoid_list)
        except (ValueError, KeyError) as exc:
            raise ToolError(str(exc))
        self.last_path_result = result
        # стисла відповідь для LLM (повний journal лишається в last_path_result)
        return {
            "path": result["path"],
            "delay_ms": result["delay_ms"],
            "transmission_ms": result["transmission_ms"],
            "total_ms": result["total_ms"],
            "bottleneck": result["bottleneck"],
            "min_bandwidth_MBps": result["min_bandwidth_MBps"],
            "links": [
                f"{x['a']}-{x['b']}:{x['delay_ms']}ms/{x['bandwidth_MBps']}MBps"
                for x in result["links"]
            ],
            "avoid": result["avoid"],
            "installed_rules": result["installed_rules"],
            "optimal_delay_ms": result["optimal_delay_ms"],
            "optimal_path": result["optimal_path"],
        }

    def get_topology(self):
        self.controller.ensure_seeded()
        return self.db.dump()

    def simulate_link_failure(self, a: str, b: str):
        try:
            ua, ub = parse_address(a), parse_address(b)
        except (ValueError, IndexError, TypeError) as exc:
            raise ToolError(f"Некоректні адреси каналу: {a!r}, {b!r} ({exc})")
        try:
            old, new = self.env.degrade_link(ua, ub, down=True)
        except ValueError as exc:
            raise ToolError(str(exc))
        self.db.mark_failed(ua, ub)
        return {
            "failed_link": f"{a} — {b}",
            "previous_delay_ms": old,
            "status": "down",
            "failed_channels": self.db.dump().get("failed", []),
            "num_channels_known": len(self.db.links),
        }

    def dispatch(self, name, arguments):
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_"):
            raise ToolError(f"Невідомий інструмент: {name!r}")
        args = dict(arguments or {})
        self.call_log.append((name, args))
        return fn(**args)

    def tool_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_flow_state",
                    "description": (
                        "Повертає стан потоку: голову потоку, отримувача, "
                        "шлях і перелік уже встановлених правил."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_link_stats",
                    "description": (
                        "Телеметрія вузла: сусідні канали з затримками (мс) "
                        "та пропускною здатністю (МБ/с)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node": {
                                "type": "string",
                                "description": "Адреса вузла, напр. 10.0.2.3",
                            },
                        },
                        "required": ["node"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "install_rule",
                    "description": (
                        "Одна дія контролера: встановити правило через інтерфейс "
                        "N/E/S/W поточного вузла або відкликати останнє правило "
                        "(interface=\"revoke\")."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "interface": {
                                "type": "string",
                                "description": "N, E, S, W або revoke",
                            },
                        },
                        "required": ["interface"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "install_path",
                    "description": (
                        "Запускає knowledge-based контролер для побудови повного "
                        "шляху між src і dst з опційним обходом вузлів avoid. "
                        "Повертає шлях, затримку та час передачі потоку."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "src": {
                                "type": "string",
                                "description": "Адреса відправника",
                            },
                            "dst": {
                                "type": "string",
                                "description": "Адреса отримувача",
                            },
                            "avoid": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Список вузлів, які потрібно оминути "
                                    "(може бути порожнім)."
                                ),
                            },
                        },
                        "required": ["src", "dst"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_topology",
                    "description": (
                        "Повертає вміст бази знань контролера: відомі вузли "
                        "та канали з затримками й пропускною здатністю."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "simulate_link_failure",
                    "description": (
                        "Емулює вихід з ладу каналу між вузлами a та b і "
                        "повертає оновлений стан мережі."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string", "description": "Перший вузол"},
                            "b": {"type": "string", "description": "Другий вузол"},
                        },
                        "required": ["a", "b"],
                    },
                },
            },
        ]

    def summary(self):
        return {
            "tool_calls": len(self.call_log),
            "by_name": _count_by_name(self.call_log),
        }


def _count_by_name(call_log):
    counts = {}
    for name, _ in call_log:
        counts[name] = counts.get(name, 0) + 1
    return counts


def tool_result_to_text(result):
    return json.dumps(result, ensure_ascii=False)

"""Інструменти (функції), які LLM-контролер викликає під час маршрутизації.

Кожен інструмент відповідає операції SDN-контролера над площиною даних:
  - get_flow_state()  — поточний стан потоку (голова, шлях, отримувач);
  - get_link_stats()  — сусідні канали вузла та їх затримки;
  - install_flow()    — встановити правило й перемістити голову;
  - revoke_flow()     — відкликати останнє правило;
  - finish_flow()     — завершити маршрутизацію в отримувачі.

Повертаємо результати у вигляді JSON-подібних словників, які сериалізуються
в текст і повертаються моделі як результат виклику інструмента.
"""

import json

from sdn_env import parse_address


class ToolError(Exception):
    """Помилка виконання інструмента (передається моделі як текст)."""


class SdnToolbox:
    """Зв'язує інструменти з конкретним екземпляром середовища."""

    def __init__(self, env):
        self.env = env
        self.call_log = []

    # --- реалізація інструментів -----------------------------------------
    def get_flow_state(self):
        """Повертає поточний стан потоку."""
        return self.env.flow_state()

    def get_link_stats(self, node: str):
        """Повертає сусідні канали вузла та їх затримки (мс)."""
        try:
            parsed = parse_address(node)
        except (ValueError, IndexError):
            raise ToolError(f"Некоректна адреса вузла: {node!r}")
        return self.env.link_stats(parsed)

    def install_flow(self, direction: str):
        """Встановлює правило пересилання в напрямку N/E/S/W."""
        direction = str(direction).upper()
        try:
            self.env.install_rule(direction)
        except ValueError as exc:
            raise ToolError(str(exc))
        return self.env.flow_state()

    def revoke_flow(self):
        """Відкликає останнє встановлене правило."""
        try:
            self.env.revoke_last_rule()
        except RuntimeError as exc:
            raise ToolError(str(exc))
        return self.env.flow_state()

    def finish_flow(self):
        """Завершує маршрутизацію. Дозволено лише в отримувачі."""
        if self.env.head != self.env.destination:
            raise ToolError(
                f"Не можна завершити: голова потоку ще не в отримувачі "
                f"(поточний вузол {self.env.flow_state()['head']})."
            )
        return {"finished": True, **self.env.flow_state()}

    # --- виконання за іменем (для циклу агента) ---------------------------
    def dispatch(self, name, arguments):
        fn = getattr(self, name, None)
        if fn is None:
            raise ToolError(f"Невідомий інструмент: {name!r}")
        self.call_log.append((name, dict(arguments)))
        return fn(**arguments)


    # --- схеми для передачі в Ollama -------------------------------------
    def tool_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_flow_state",
                    "description": (
                        "Повертає поточний стан потоку: вузол, де зараз голова "
                        "потоку (head), адреси відправника й отримувача, "
                        "cost_to_destination (мінімальна залишкова затримка до "
                        "отримувача), вже прокладений шлях (path) та кількість "
                        "правил."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_link_stats",
                    "description": (
                        "Повертає сусідні канали заданого вузла: для кожного "
                        "напрямку N/E/S/W — адресу сусіда (neighbor), затримку "
                        "каналу в мс (delay_ms) та cost_to_destination сусіда "
                        "(мінімальна залишкова затримка від сусіда до "
                        "отримувача). Також повертає cost_to_destination самого "
                        "вузла."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node": {
                                "type": "string",
                                "description": (
                                    "Адреса вузла, напр. 10.0.2.3 "
                                    "(10.0.<рядок>.<стовпець>)."
                                ),
                            },
                        },
                        "required": ["node"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "install_flow",
                    "description": (
                        "Встановлює правило пересилання і переміщує голову "
                        "потоку на сусідній вузол у заданому напрямку."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["N", "E", "S", "W"],
                                "description": "Напрямок переходу: N/E/S/W.",
                            },
                        },
                        "required": ["direction"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "revoke_flow",
                    "description": (
                        "Відкликає останнє встановлене правило і повертає "
                        "голову потоку на попередній вузол."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish_flow",
                    "description": (
                        "Завершує маршрутизацію. Викликати лише тоді, коли "
                        "голова потоку вже перебуває в отримувачі."
                    ),
                    "parameters": {"type": "object", "properties": {}},
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
    """Серіалізує результат інструмента в рядок для повідомлення ролі tool."""
    return json.dumps(result, ensure_ascii=False)

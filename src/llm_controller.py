"""LLM-контролер: агент на основі великої мовної моделі (Ollama).

Агент отримує системний промпт із завданням маршрутизації та набір
інструментів (tools) і в циклі:
  1) надсилає повідомлення моделі;
  2) якщо модель повернула tool_calls — виконує їх через SdnToolbox
     і додає результати в історію;
  3) повторює, поки не викликано finish_flow або не вичерпано ліміт кроків.

Збирає метрики: кількість викликів моделі, викликів інструментів, час.
"""

import json
import time

import ollama

SYSTEM_PROMPT = (
    "Ти — SDN-контролер програмно-конфігурованої мережі. Проклади потік від "
    "відправника до отримувача в мережі-сітці. Ти керуєш «головою потоку»: на "
    "кожному кроці можеш перемістити її на один крок у напрямку N (вгору), "
    "E (праворуч), S (вниз) або W (ліворуч). Мета — довести голову потоку до "
    "отримувача з мінімальною сумарною затримкою каналів (мс).\n"
    "Алгоритм дій на кожному кроці:\n"
    "1) Виклич get_flow_state(), щоб дізнатися поточний вузол і його "
    "cost_to_destination (мінімальна залишкова затримка до отримувача).\n"
    "2) Виклич get_link_stats(node) для поточного вузла. Обери напрямок, у "
    "якого сусід має НАЙМЕНШИЙ cost_to_destination.\n"
    "3) Перемістись у цей бік через install_flow(direction).\n"
    "4) Якщо поточний cost_to_destination == 0 (голова вже в отримувачі) — "
    "виклич finish_flow().\n"
    "5) Якщо помилився напрямком — виклич revoke_flow(), щоб повернутися.\n"
    "ВАЖЛИВО: у КОЖНІЙ відповіді викликай РІВНО ОДИН інструмент і не додавай "
    "жодного тексту чи пояснень."
)


class LlmController:
    """Агент, що керує мережею через виклики інструментів моделлю."""

    def __init__(self, model, toolbox, system_prompt=SYSTEM_PROMPT,
                 temperature=0.0, num_ctx=4096, max_steps=80,
                 num_gpu=0, verbose=True):
        self.model = model
        self.toolbox = toolbox
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_steps = max_steps
        self.num_gpu = num_gpu
        self.verbose = verbose


    def run(self):
        tools = self.toolbox.tool_schemas()
        messages = [{"role": "system", "content": self.system_prompt}]
        log = []
        llm_calls = 0
        tool_calls_total = 0
        llm_time_s = 0.0
        eval_tokens = 0
        prompt_tokens = 0
        finished = False
        no_tool_streak = 0

        for step in range(1, self.max_steps + 1):
            t0 = time.perf_counter()
            try:
                resp = ollama.chat(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    options={
                        "temperature": self.temperature,
                        "num_ctx": self.num_ctx,
                        "num_gpu": self.num_gpu,
                    },
                )
            except Exception as exc:
                log.append({
                    "step": step, "tool": "(помилка LLM)",
                    "arguments": {}, "status": "error",
                    "result": json.dumps({"error": str(exc)}, ensure_ascii=False),
                    "llm_time_s": time.perf_counter() - t0,
                })
                break
            elapsed = time.perf_counter() - t0
            llm_calls += 1
            llm_time_s += elapsed

            msg = resp.get("message", {})
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []

            eval_tokens += resp.get("eval_count", 0)
            prompt_tokens += resp.get("prompt_eval_count", 0)

            if self.verbose:
                names = [c.function.name for c in tool_calls] or ["(без викликів)"]
                print(
                    f"  [крок {step:>2}] {elapsed:5.2f} с | токени "
                    f"{resp.get('prompt_eval_count', 0)}->{resp.get('eval_count', 0)} "
                    f"| виклики: {', '.join(names)}",
                    flush=True,
                )

            if tool_calls:
                # зберігаємо відповідь асистента ТІЛЬКИ з викликами (без тексту),
                # щоб модель не «заговорювалася» замість викликів інструментів
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                })
                no_tool_streak = 0

            for call in tool_calls:
                fn = call.function
                name = fn.name
                arguments = fn.arguments or {}
                tool_calls_total += 1
                try:
                    result = self.toolbox.dispatch(name, arguments)
                    result_text = json.dumps(result, ensure_ascii=False)
                    status = "ok"
                except Exception as exc:  # ToolError та інші помилки
                    result_text = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    status = "error"

                messages.append({
                    "role": "tool",
                    "content": result_text,
                    "tool_name": name,
                })
                log.append({
                    "step": step, "tool": name, "arguments": arguments,
                    "status": status, "result": result_text,
                    "llm_time_s": elapsed,
                })
                if name == "finish_flow" and status == "ok":
                    finished = True

            if finished:
                break

            if not tool_calls:
                no_tool_streak += 1
                state = self.toolbox.env.flow_state()
                if state["at_destination"]:
                    log.append({
                        "step": step, "tool": "(жодного виклику)",
                        "arguments": {}, "status": "finish-implicit",
                        "result": json.dumps(state, ensure_ascii=False),
                        "llm_time_s": elapsed,
                    })
                    finished = True
                    break
                if no_tool_streak >= 3:
                    log.append({
                        "step": step, "tool": "(жодного виклику)",
                        "arguments": {}, "status": "stuck",
                        "result": json.dumps(
                            {"note": "модель 3 рази поспіль не викликала інструмент"},
                            ensure_ascii=False,
                        ),
                        "llm_time_s": elapsed,
                    })
                    break
                messages.append({
                    "role": "user",
                    "content": (
                        "Виклич ОДИН інструмент (get_flow_state, get_link_stats, "
                        "install_flow, revoke_flow або finish_flow). Без тексту."
                    ),
                })


        return {
            "model": self.model,
            "finished": finished,
            "steps": llm_calls,
            "tool_calls": tool_calls_total,
            "llm_time_s": llm_time_s,
            "eval_tokens": eval_tokens,
            "prompt_tokens": prompt_tokens,
            "log": log,
            "flow_state": self.toolbox.env.flow_state(),
            "path": self.toolbox.env.path(),
            "delay": self.toolbox.env.path_delay(),
            "nodes": len(self.toolbox.env.path()),
            "installed": self.toolbox.env.installed,
            "revoked": self.toolbox.env.revoked,
            "tool_summary": self.toolbox.summary(),
        }

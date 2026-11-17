import json
import time

import ollama

SYSTEM_PROMPT = (
    "Ти — NL-інтерфейс SDN-контролера програмно-конфігурованої мережі. "
    "Оператор формулює намір природною мовою. Ти НЕ обчислюєш шлях і НЕ "
    "встановлюєш правила текстом: усі дії лише через інструменти. Стан "
    "контролера — єдине джерело істини.\n"
    "Доступні інструменти:\n"
    "- get_flow_state() — голова потоку, отримувач, встановлені правила;\n"
    "- get_link_stats(node) — затримки та пропускна здатність каналів вузла;\n"
    "- install_rule(interface) — N/E/S/W або revoke;\n"
    "- install_path(src, dst, avoid) — повний шлях knowledge-based контролером "
    "(prefer цей інструмент для прокладання шляху);\n"
    "- get_topology() — база знань (вузли й канали);\n"
    "- simulate_link_failure(a, b) — вихід каналу з ладу.\n"
    "Правила:\n"
    "1) Для прокладання шляху викликай install_path з потрібними src/dst/avoid.\n"
    "2) Якщо запит неможливий (неіснуючий вузол тощо) — спробуй tool, "
    "отримай помилку й чесно поясни оператору.\n"
    "3) Після виконання інструментів дай коротку відповідь мовою запиту "
    "оператора (українською або англійською) з фактами зі стану контролера "
    "(шлях, затримка, час передачі, вузьке місце).\n"
    "4) Не вигадуй результати tools.\n"
    "5) Відповідь оператору — коротка (кілька речень), без повного дампу топології.\n"
    "6) Після simulate_link_failure одразу виклич install_path для перебудови."
)


class LlmController:
    def __init__(self, model, toolbox, system_prompt=SYSTEM_PROMPT,
                 temperature=0.0, num_ctx=8192, max_steps=40,
                 num_gpu=0, verbose=True, num_predict=8192):
        self.model = model
        self.toolbox = toolbox
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_steps = max_steps
        self.num_gpu = num_gpu
        self.verbose = verbose
        self.num_predict = num_predict

    def run(self, user_query: str):
        tools = self.toolbox.tool_schemas()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        log = []
        llm_calls = 0
        tool_calls_total = 0
        llm_time_s = 0.0
        eval_tokens = 0
        prompt_tokens = 0
        final_answer = ""
        snapshots = []
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
                        "num_predict": self.num_predict,
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

            eval_tokens += resp.get("eval_count", 0) or 0
            prompt_tokens += resp.get("prompt_eval_count", 0) or 0

            if self.verbose:
                names = [c.function.name for c in tool_calls] or ["(відповідь)"]
                print(
                    f"  [крок {step:>2}] {elapsed:5.2f} с | токени "
                    f"{resp.get('prompt_eval_count', 0)}->{resp.get('eval_count', 0)} "
                    f"| {', '.join(names)}",
                    flush=True,
                )

            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                })
                no_tool_streak = 0
            else:
                messages.append({"role": "assistant", "content": content})
                final_answer = content.strip()
                no_tool_streak += 1
                log.append({
                    "step": step, "tool": "(відповідь)",
                    "arguments": {}, "status": "answer",
                    "result": content,
                    "llm_time_s": elapsed,
                })
                # якщо вже були tools або модель відповіла текстом — завершуємо
                if tool_calls_total > 0 or no_tool_streak >= 1:
                    break
                messages.append({
                    "role": "user",
                    "content": (
                        "Спочатку виклич потрібні інструменти контролера "
                        "(для шляху — install_path; не викликай get_topology "
                        "без потреби), потім коротку відповідь."
                    ),
                })
                continue

            step_actions = []
            for call in tool_calls:
                fn = call.function
                name = fn.name
                arguments = fn.arguments or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                tool_calls_total += 1
                try:
                    result = self.toolbox.dispatch(name, arguments)
                    result_text = json.dumps(result, ensure_ascii=False)
                    status = "ok"
                except Exception as exc:
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
                step_actions.append({
                    "tool": name, "arguments": dict(arguments), "status": status,
                })

            state = self.toolbox.env.flow_state()
            snapshots.append({
                "step": step,
                "head": state["head"],
                "path": list(state["path"]),
                "installed": self.toolbox.env.installed,
                "revoked": self.toolbox.env.revoked,
                "actions": step_actions,
            })

            # якщо install_path повернув внутрішні snapshots — зберегти
            if self.toolbox.last_path_result and self.toolbox.last_path_result.get(
                "snapshots"
            ):
                # не дублюємо на кожному кроці; оновлюємо посилання
                pass

        path_snaps = []
        if self.toolbox.last_path_result:
            path_snaps = self.toolbox.last_path_result.get("snapshots") or []

        return {
            "model": self.model,
            "query": user_query,
            "answer": final_answer,
            "steps": llm_calls,
            "tool_calls": tool_calls_total,
            "llm_time_s": llm_time_s,
            "eval_tokens": eval_tokens,
            "prompt_tokens": prompt_tokens,
            "log": log,
            "snapshots": snapshots,
            "path_snapshots": path_snaps,
            "flow_state": self.toolbox.env.flow_state(),
            "path": self.toolbox.env.path(),
            "delay": self.toolbox.env.path_delay(),
            "nodes": len(self.toolbox.env.path()),
            "installed": self.toolbox.env.installed,
            "revoked": self.toolbox.env.revoked,
            "tool_summary": self.toolbox.summary(),
            "last_path_result": self.toolbox.last_path_result,
            "transfer": self.toolbox.env.transfer_stats(),
        }

    def explain_route(self, journal_text: str, model=None):
        """Частина 3: пояснення маршруту без tools (окремий чат)."""
        model = model or self.model
        prompt = (
            "Ти пояснюєш оператору SDN, чому контролер обрав саме такий шлях. "
            "Нижче — журнал роботи контролера. Поясни українською: (1) чому "
            "обрано цей шлях; (2) де втрачено час або затримку порівняно з "
            "оптимальним шляхом. Спирайся лише на наведені факти.\n\n"
            f"{journal_text}"
        )
        t0 = time.perf_counter()
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_predict": self.num_predict,
            },
        )
        elapsed = time.perf_counter() - t0
        text = (resp.get("message") or {}).get("content") or ""
        return {
            "explanation": text.strip(),
            "llm_time_s": elapsed,
            "eval_tokens": resp.get("eval_count", 0) or 0,
            "prompt_tokens": resp.get("prompt_eval_count", 0) or 0,
            "model": model,
        }

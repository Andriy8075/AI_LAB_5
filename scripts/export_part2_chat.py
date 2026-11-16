"""Експорт діалогів частини 2 у зручний для читання чат (HTML + Markdown).

Не запускає модель — читає outputs/part2_scenarios.json.

Запуск:
  python scripts/export_part2_chat.py
"""

from __future__ import annotations

import html
import json
import os
import sys
import textwrap

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
PART2_PATH = os.path.join(OUT_DIR, "part2_scenarios.json")
ANSWERS_DIR = os.path.join(OUT_DIR, "part2_answers")
HTML_PATH = os.path.join(OUT_DIR, "part2_chat.html")
MD_PATH = os.path.join(OUT_DIR, "part2_chat.md")


def pretty_result(raw: str) -> str:
    """Форматує JSON-рядок результату tool у багаторядковий текст."""
    if not raw:
        return ""
    try:
        obj = json.loads(raw)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, json.JSONDecodeError):
        return str(raw)


def wrap(text: str, width: int = 88) -> str:
    lines = []
    for paragraph in (text or "").splitlines() or [""]:
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width) or [paragraph])
    return "\n".join(lines)


def scenario_turns(sc: dict) -> list[dict]:
    """Будує послідовність «бульбашок» чату для одного сценарію."""
    turns = [
        {
            "role": "user",
            "title": "Оператор",
            "body": sc.get("query") or "",
        }
    ]
    for entry in sc.get("log") or []:
        tool = entry.get("tool") or ""
        status = entry.get("status") or ""
        if status == "answer" or tool == "(відповідь)":
            turns.append({
                "role": "assistant",
                "title": "Qwen (відповідь)",
                "body": entry.get("result") or sc.get("answer") or "",
            })
            continue
        if tool.startswith("("):
            turns.append({
                "role": "system",
                "title": tool,
                "body": entry.get("result") or "",
            })
            continue
        args = entry.get("arguments") or {}
        args_txt = json.dumps(args, ensure_ascii=False, indent=2)
        result_txt = pretty_result(entry.get("result") or "")
        turns.append({
            "role": "tool_call",
            "title": f"Qwen → tool: {tool}",
            "body": f"Аргументи:\n{args_txt}",
        })
        turns.append({
            "role": "tool_result",
            "title": f"Результат {tool} [{status}]",
            "body": result_txt,
        })

    # якщо в log не було answer, але поле answer є
    if sc.get("answer") and not any(t["role"] == "assistant" for t in turns):
        turns.append({
            "role": "assistant",
            "title": "Qwen (відповідь)",
            "body": sc["answer"],
        })
    return turns


def write_answers_txt(data: dict) -> list[str]:
    os.makedirs(ANSWERS_DIR, exist_ok=True)
    paths = []
    for sc in data.get("scenarios") or []:
        sid = sc.get("id") or "scenario"
        answer = sc.get("answer") or ""
        # повний текст з log, якщо довший
        for entry in sc.get("log") or []:
            if entry.get("status") == "answer" and entry.get("result"):
                if len(entry["result"]) > len(answer):
                    answer = entry["result"]
        path = os.path.join(ANSWERS_DIR, f"{sid}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Сценарій: {sid}\n")
            f.write(f"Модель: {data.get('model', '')}\n")
            f.write("=" * 72 + "\n")
            f.write("ЗАПИТ ОПЕРАТОРА\n")
            f.write("-" * 72 + "\n")
            f.write(wrap(sc.get("query") or "") + "\n\n")
            f.write("ВІДПОВІДЬ QWEN\n")
            f.write("-" * 72 + "\n")
            f.write(wrap(answer) + "\n")
        paths.append(path)
    return paths


def write_markdown(data: dict) -> str:
    lines = [
        "# Частина 2 — діалоги з Qwen",
        "",
        f"Модель: `{data.get('model', '')}`",
        "",
        "Цей файл згенеровано з `part2_scenarios.json` для зручного читання.",
        "",
    ]
    for sc in data.get("scenarios") or []:
        lines.append(f"## {sc.get('id')}")
        lines.append("")
        for turn in scenario_turns(sc):
            lines.append(f"### {turn['title']}")
            lines.append("")
            lines.append("```")
            lines.append(wrap(turn["body"]))
            lines.append("```")
            lines.append("")
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return MD_PATH


def write_html(data: dict) -> str:
    css = """
    :root {
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --user: #1f4b3f;
      --assistant: #2c3e50;
      --tool: #5c4a2e;
      --tool-bg: #f0e6d4;
      --result-bg: #e8eef2;
      --border: #d7d0c4;
      --muted: #6b6560;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background: linear-gradient(180deg, #efe8db 0%, #f7f4ee 40%, #f4f1ea 100%);
      color: #1d1a16;
      line-height: 1.45;
    }
    header {
      position: sticky; top: 0; z-index: 2;
      backdrop-filter: blur(8px);
      background: rgba(255,253,248,0.92);
      border-bottom: 1px solid var(--border);
      padding: 1rem 1.25rem;
    }
    header h1 { margin: 0 0 0.25rem; font-size: 1.35rem; }
    header p { margin: 0; color: var(--muted); font-size: 0.95rem; }
    nav {
      display: flex; flex-wrap: wrap; gap: 0.5rem;
      padding: 0.75rem 1.25rem 0;
    }
    nav a {
      text-decoration: none; color: var(--user);
      background: #e4efe9; border: 1px solid #c5d9cf;
      border-radius: 999px; padding: 0.35rem 0.75rem; font-size: 0.85rem;
    }
    main { max-width: 920px; margin: 0 auto; padding: 1rem 1.25rem 3rem; }
    section {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1rem 1rem 0.5rem;
      margin: 1.25rem 0;
      box-shadow: 0 8px 24px rgba(40, 30, 10, 0.05);
    }
    section h2 {
      margin: 0 0 0.75rem;
      font-size: 1.1rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.5rem;
    }
    .bubble {
      max-width: 92%;
      margin: 0.65rem 0 1rem;
      padding: 0.75rem 0.9rem;
      border-radius: 14px;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      border: 1px solid var(--border);
    }
    .bubble .meta {
      font-size: 0.78rem;
      font-weight: 600;
      margin-bottom: 0.4rem;
      opacity: 0.85;
    }
    .bubble.user { background: #e7f2ec; border-color: #bfd5c9; margin-right: auto; }
    .bubble.assistant { background: #eef3f8; border-color: #c8d5e2; margin-left: auto; }
    .bubble.tool_call { background: var(--tool-bg); border-color: #d8c7a8; }
    .bubble.tool_result { background: var(--result-bg); border-color: #c5d0da; font-family: Consolas, "Courier New", monospace; font-size: 0.86rem; }
    .bubble.system { background: #f3f3f3; color: var(--muted); }
    .answer-highlight {
      outline: 2px solid #2f6f5e;
      outline-offset: 2px;
    }
    """

    parts = [
        "<!DOCTYPE html>",
        '<html lang="uk">',
        "<head>",
        '<meta charset="utf-8"/>',
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>',
        "<title>Частина 2 — чат з Qwen</title>",
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        "<header>",
        "<h1>Частина 2 — діалоги з Qwen (чат)</h1>",
        f"<p>Модель: {html.escape(str(data.get('model', '')))}. "
        "Згенеровано з part2_scenarios.json. Фінальні відповіді виділені рамкою.</p>",
        "<nav>",
    ]
    for sc in data.get("scenarios") or []:
        sid = html.escape(sc.get("id") or "")
        parts.append(f'<a href="#{sid}">{sid}</a>')
    parts.append("</nav></header><main>")

    for sc in data.get("scenarios") or []:
        sid = sc.get("id") or "scenario"
        parts.append(f'<section id="{html.escape(sid)}">')
        parts.append(f"<h2>{html.escape(sid)}</h2>")
        for turn in scenario_turns(sc):
            role = turn["role"]
            cls = role
            if role == "assistant":
                cls += " answer-highlight"
            body = html.escape(turn["body"])
            title = html.escape(turn["title"])
            parts.append(f'<div class="bubble {cls}"><div class="meta">{title}</div>{body}</div>')
        parts.append("</section>")

    parts.extend(["</main>", "</body>", "</html>"])
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return HTML_PATH


def export_from_part2(data: dict | None = None) -> dict:
    if data is None:
        if not os.path.exists(PART2_PATH):
            raise FileNotFoundError(
                f"Немає {PART2_PATH}. Спочатку запустіть scripts/run_part2.py"
            )
        with open(PART2_PATH, encoding="utf-8") as f:
            data = json.load(f)

    answer_paths = write_answers_txt(data)
    md_path = write_markdown(data)
    html_path = write_html(data)
    return {
        "html": html_path,
        "markdown": md_path,
        "answers": answer_paths,
    }


def main():
    paths = export_from_part2()
    print("HTML-чат:", paths["html"])
    print("Markdown:", paths["markdown"])
    print("Тексти відповідей:")
    for p in paths["answers"]:
        print(" ", p)


if __name__ == "__main__":
    main()

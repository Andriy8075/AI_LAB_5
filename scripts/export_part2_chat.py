"""Експорт діалогів частини 2 у HTML-чат з tools + графами + відповіддю Qwen.

Не запускає модель — читає outputs/part2_scenarios.json і figures/part2_*/.

Запуск:
  python scripts/export_part2_chat.py
"""

from __future__ import annotations

import base64
import html
import json
import os
import sys
import textwrap

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(ROOT, "figures")
PART2_PATH = os.path.join(OUT_DIR, "part2_scenarios.json")
ANSWERS_DIR = os.path.join(OUT_DIR, "part2_answers")
HTML_PATH = os.path.join(OUT_DIR, "part2_chat.html")
MD_PATH = os.path.join(OUT_DIR, "part2_chat.md")

# Інструменти, після яких показуємо зміну графа
GRAPH_TOOLS = {
    "install_path",
    "install_rule",
    "install_flow",
    "revoke_flow",
    "simulate_link_failure",
}


def pretty_result(raw: str) -> str:
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


def img_to_data_uri(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def ensure_step_figures(sc: dict) -> list[str]:
    """Повертає шляхи PNG кроків; за потреби перегенеровує з path_snapshots."""
    sid = sc.get("id") or "scenario"
    step_dir = os.path.join(FIG_DIR, f"part2_{sid}", "steps")
    final_path = os.path.join(FIG_DIR, f"part2_{sid}", "final_path.png")
    snaps = sc.get("path_snapshots") or []

    existing = []
    if os.path.isdir(step_dir):
        existing = sorted(
            os.path.join(step_dir, n)
            for n in os.listdir(step_dir)
            if n.lower().endswith(".png")
        )

    if snaps and len(existing) < len(snaps):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import sdn_env
            from visualization import save_step_figures, plot_single_path

            graph = sdn_env.generate_grid_graph()
            existing = save_step_figures(
                graph, snaps, sdn_env.SOURCE, sdn_env.DESTINATION,
                out_dir=step_dir, prefix="step",
            )
            path = (sc.get("flow_state") or {}).get("path") or []
            if path:
                plot_single_path(
                    graph, path, sdn_env.SOURCE, sdn_env.DESTINATION,
                    title=f"{sid}",
                    filename=final_path,
                )
        except Exception as exc:
            print(f"  (не вдалося згенерувати графи для {sid}: {exc})", flush=True)

    return existing


def graph_images_for_tool(sc: dict, tool: str, status: str) -> list[dict]:
    """Список зображень графа після tool (підпис + data URI або шлях)."""
    if status != "ok" or tool not in GRAPH_TOOLS:
        return []

    sid = sc.get("id") or "scenario"
    images = []

    if tool == "install_path":
        steps = ensure_step_figures(sc)
        for i, path in enumerate(steps, start=1):
            uri = img_to_data_uri(path)
            if uri:
                images.append({
                    "caption": f"Граф після кроку {i} встановлення шляху ({sid})",
                    "data_uri": uri,
                    "file": path,
                })
        final = os.path.join(FIG_DIR, f"part2_{sid}", "final_path.png")
        uri = img_to_data_uri(final)
        if uri and (not steps or os.path.abspath(final) not in {
            os.path.abspath(p) for p in steps
        }):
            images.append({
                "caption": f"Фінальний шлях після install_path ({sid})",
                "data_uri": uri,
                "file": final,
            })
        return images

    # install_rule / revoke / failure — показуємо актуальний final_path, якщо є
    final = os.path.join(FIG_DIR, f"part2_{sid}", "final_path.png")
    # для failure без нового шляху — topology
    candidates = [final, os.path.join(FIG_DIR, "topology.png")]
    if tool == "simulate_link_failure":
        # після failure шлях ще може бути старим; покажемо topology як контекст
        candidates = [
            os.path.join(FIG_DIR, f"part2_{sid}", "final_path.png"),
            os.path.join(FIG_DIR, "topology.png"),
        ]
    for path in candidates:
        uri = img_to_data_uri(path)
        if uri:
            images.append({
                "caption": f"Стан мережі після {tool} ({sid})",
                "data_uri": uri,
                "file": path,
            })
            break
    return images


def scenario_turns(sc: dict) -> list[dict]:
    """Бульбашки: оператор → tool → результат → граф(и) → … → відповідь Qwen."""
    turns = [
        {
            "role": "user",
            "title": "Оператор",
            "body": sc.get("query") or "",
            "images": [],
        }
    ]

    # список tools на початку сценарію
    tool_names = [
        e.get("tool") for e in (sc.get("log") or [])
        if e.get("tool") and not str(e.get("tool")).startswith("(")
        and e.get("status") != "answer"
    ]
    if tool_names:
        unique = list(dict.fromkeys(tool_names))
        turns.append({
            "role": "system",
            "title": "Інструменти, які викликав Qwen у цьому сценарії",
            "body": "\n".join(f"{i+1}. {name}" for i, name in enumerate(unique)),
            "images": [],
        })

    for entry in sc.get("log") or []:
        tool = entry.get("tool") or ""
        status = entry.get("status") or ""
        if status == "answer" or tool == "(відповідь)":
            turns.append({
                "role": "assistant",
                "title": "Qwen (текстова відповідь)",
                "body": entry.get("result") or sc.get("answer") or "",
                "images": [],
            })
            continue
        if tool.startswith("("):
            turns.append({
                "role": "system",
                "title": tool,
                "body": entry.get("result") or "",
                "images": [],
            })
            continue

        args = entry.get("arguments") or {}
        args_txt = json.dumps(args, ensure_ascii=False, indent=2)
        result_txt = pretty_result(entry.get("result") or "")

        turns.append({
            "role": "tool_call",
            "title": f"Виклик інструмента: {tool}",
            "body": f"Аргументи:\n{args_txt}",
            "images": [],
        })
        turns.append({
            "role": "tool_result",
            "title": f"Результат інструмента {tool} [{status}]",
            "body": result_txt,
            "images": [],
        })

        for img in graph_images_for_tool(sc, tool, status):
            turns.append({
                "role": "graph",
                "title": img["caption"],
                "body": "",
                "images": [img],
            })

    if sc.get("answer") and not any(t["role"] == "assistant" for t in turns):
        turns.append({
            "role": "assistant",
            "title": "Qwen (текстова відповідь)",
            "body": sc["answer"],
            "images": [],
        })
    return turns


def write_answers_txt(data: dict) -> list[str]:
    os.makedirs(ANSWERS_DIR, exist_ok=True)
    paths = []
    for sc in data.get("scenarios") or []:
        sid = sc.get("id") or "scenario"
        answer = sc.get("answer") or ""
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
            tools = [
                e.get("tool") for e in (sc.get("log") or [])
                if e.get("tool") and not str(e.get("tool")).startswith("(")
                and e.get("status") != "answer"
            ]
            f.write("ВИКЛИКАНІ ІНСТРУМЕНТИ\n")
            f.write("-" * 72 + "\n")
            f.write("\n".join(tools) or "(немає)")
            f.write("\n\n")
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
        "Послідовність: запит → виклик tool → результат → граф(и) → відповідь.",
        "",
    ]
    for sc in data.get("scenarios") or []:
        lines.append(f"## {sc.get('id')}")
        lines.append("")
        for turn in scenario_turns(sc):
            lines.append(f"### {turn['title']}")
            lines.append("")
            if turn.get("body"):
                lines.append("```")
                lines.append(wrap(turn["body"]))
                lines.append("```")
                lines.append("")
            for img in turn.get("images") or []:
                rel = os.path.relpath(img["file"], ROOT).replace("\\", "/")
                lines.append(f"![{turn['title']}]({rel})")
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
      --border: #d7d0c4;
      --muted: #6b6560;
      --tool-bg: #f0e6d4;
      --result-bg: #e8eef2;
      --graph-bg: #f7faf7;
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
    main { max-width: 980px; margin: 0 auto; padding: 1rem 1.25rem 3rem; }
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
      max-width: 95%;
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
      opacity: 0.9;
    }
    .bubble.user { background: #e7f2ec; border-color: #bfd5c9; }
    .bubble.assistant { background: #eef3f8; border-color: #c8d5e2; margin-left: auto; }
    .bubble.tool_call { background: var(--tool-bg); border-color: #d8c7a8; }
    .bubble.tool_result {
      background: var(--result-bg); border-color: #c5d0da;
      font-family: Consolas, "Courier New", monospace; font-size: 0.86rem;
    }
    .bubble.system { background: #f3f3f3; color: var(--muted); }
    .bubble.graph {
      background: var(--graph-bg);
      border-color: #b7cbb8;
      max-width: 100%;
      white-space: normal;
    }
    .bubble.graph img {
      display: block;
      width: 100%;
      max-width: 720px;
      height: auto;
      margin: 0.4rem auto 0;
      border-radius: 10px;
      border: 1px solid #c5d5c6;
      background: #fff;
    }
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
        "<h1>Частина 2 — діалоги з Qwen (чат + графи)</h1>",
        f"<p>Модель: {html.escape(str(data.get('model', '')))}. "
        "Для кожного сценарію: запит → виклик tool → результат → граф після зміни "
        "мережі → текстова відповідь Qwen.</p>",
        "<nav>",
    ]
    for sc in data.get("scenarios") or []:
        sid = html.escape(sc.get("id") or "")
        parts.append(f'<a href="#{sid}">{sid}</a>')
    parts.append("</nav></header><main>")

    for sc in data.get("scenarios") or []:
        sid = sc.get("id") or "scenario"
        # прогрів фігур до рендеру
        if any(
            (e.get("tool") in GRAPH_TOOLS and e.get("status") == "ok")
            for e in (sc.get("log") or [])
        ):
            ensure_step_figures(sc)

        parts.append(f'<section id="{html.escape(sid)}">')
        parts.append(f"<h2>{html.escape(sid)}</h2>")
        for turn in scenario_turns(sc):
            role = turn["role"]
            cls = role
            if role == "assistant":
                cls += " answer-highlight"
            title = html.escape(turn["title"])
            body = html.escape(turn.get("body") or "")
            parts.append(f'<div class="bubble {cls}"><div class="meta">{title}</div>')
            if body:
                parts.append(body)
            for img in turn.get("images") or []:
                uri = img.get("data_uri")
                if uri:
                    parts.append(
                        f'<img src="{uri}" alt="{html.escape(img.get("caption") or title)}"/>'
                    )
            parts.append("</div>")
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

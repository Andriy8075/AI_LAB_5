"""Будує єдиний lab5.ipynb: код запуску частин 1–4 + збережені виводи для демо."""

from __future__ import annotations

import base64
import json
import os
import sys
import textwrap

sys.stdout.reconfigure(encoding="utf-8")

import nbformat as nbf

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import sdn_env as _sdn
from visualization import render_state_ascii as _ascii

_topo_graph = _sdn.generate_grid_graph()
_topo_delay, _topo_path = _sdn.shortest_path_delay(
    _topo_graph, _sdn.SOURCE, _sdn.DESTINATION
)
ASCII_TOPO = _ascii(_topo_graph, _topo_path, _sdn.SOURCE, _sdn.DESTINATION)

PART1 = json.load(open(os.path.join(ROOT, "outputs", "part1_benchmark.json"),
                       encoding="utf-8"))
PART2 = json.load(open(os.path.join(ROOT, "outputs", "part2_scenarios.json"),
                       encoding="utf-8"))
PART3 = json.load(open(os.path.join(ROOT, "outputs", "part3_explanation.json"),
                       encoding="utf-8"))
PART4 = json.load(open(os.path.join(ROOT, "outputs", "part4_eval.json"),
                       encoding="utf-8"))

nb = nbf.v4.new_notebook()
cells = []
_exec = 0


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src: str, out_text: str | None = None, image_paths: list[str] | None = None):
    global _exec
    c = nbf.v4.new_code_cell(src)
    outputs = []
    if out_text is not None:
        outputs.append(nbf.v4.new_output("stream", name="stdout", text=out_text))
    for image_path in image_paths or []:
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            outputs.append(nbf.v4.new_output(
                "display_data",
                data={
                    "image/png": data,
                    "text/plain": [f"<Image {os.path.basename(image_path)}>"],
                },
            ))
    if outputs:
        _exec += 1
        c.outputs = outputs
        c.execution_count = _exec
    cells.append(c)


def wrap(text: str, width: int = 96) -> str:
    lines = []
    for paragraph in (text or "").splitlines() or [""]:
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width) or [paragraph])
    return "\n".join(lines)


def pretty_json_str(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
    except Exception:
        return raw or ""


def format_scenario_transcript(sc: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"СЦЕНАРІЙ: {sc.get('id')}")
    lines.append("=" * 78)
    lines.append("ОПЕРАТОР:")
    lines.append(wrap(sc.get("query") or ""))
    lines.append("")
    tools = [
        e.get("tool") for e in (sc.get("log") or [])
        if e.get("tool") and not str(e.get("tool")).startswith("(")
        and e.get("status") != "answer"
    ]
    lines.append("ІНСТРУМЕНТИ, ЯКІ ВИКЛИКАВ QWEN:")
    if tools:
        for i, t in enumerate(dict.fromkeys(tools), 1):
            lines.append(f"  {i}. {t}")
    else:
        lines.append("  (немає)")
    lines.append("")
    for entry in sc.get("log") or []:
        tool = entry.get("tool") or ""
        status = entry.get("status") or ""
        if status == "answer" or tool == "(відповідь)":
            lines.append("-" * 78)
            lines.append("ВІДПОВІДЬ МОДЕЛІ (текст):")
            lines.append(wrap(entry.get("result") or sc.get("answer") or ""))
            lines.append("")
            continue
        if tool.startswith("("):
            continue
        lines.append("-" * 78)
        lines.append(f"ВИКЛИК ІНСТРУМЕНТА: {tool}")
        lines.append("Аргументи:")
        lines.append(json.dumps(entry.get("arguments") or {}, ensure_ascii=False, indent=2))
        lines.append(f"Статус: {status}")
        lines.append("Результат:")
        lines.append(pretty_json_str(entry.get("result") or ""))
        lines.append("")
    if sc.get("answer") and not any(
        e.get("status") == "answer" for e in (sc.get("log") or [])
    ):
        lines.append("-" * 78)
        lines.append("ВІДПОВІДЬ МОДЕЛІ (текст):")
        lines.append(wrap(sc["answer"]))
        lines.append("")
    return "\n".join(lines) + "\n"


def scenario_images(sc: dict) -> list[str]:
    sid = sc.get("id") or ""
    paths = []
    step_dir = os.path.join(ROOT, "figures", f"part2_{sid}", "steps")
    if os.path.isdir(step_dir):
        paths.extend(sorted(
            os.path.join(step_dir, n)
            for n in os.listdir(step_dir) if n.lower().endswith(".png")
        ))
    final = os.path.join(ROOT, "figures", f"part2_{sid}", "final_path.png")
    if os.path.exists(final):
        paths.append(final)
    return paths


def format_part4_block() -> str:
    lines = []
    for model, block in PART4["models"].items():
        lines.append("=" * 78)
        lines.append(
            f"МОДЕЛЬ: {block['label']} ({model}) — "
            f"{block['correct_count']}/{block['total']} "
            f"(accuracy={block['accuracy']})"
        )
        lines.append("=" * 78)
        for r in block["rows"]:
            lines.append("")
            lines.append(f"### {r['id']} | correct={r['correct']} | errors={r['error_types'] or '—'}")
            lines.append(f"Запит: {r['query']}")
            lines.append(f"Очікувані tools: {', '.join(r['expected_tools'])}")
            lines.append(f"Фактичні tools:  {', '.join(r['actual_tools'])}")
            lines.append("Журнал викликів:")
            for e in r.get("log") or []:
                tool = e.get("tool") or ""
                if e.get("status") == "answer" or tool == "(відповідь)":
                    lines.append("  [відповідь моделі]")
                    continue
                if tool.startswith("("):
                    continue
                lines.append(
                    f"  - {tool}({json.dumps(e.get('arguments') or {}, ensure_ascii=False)}) "
                    f"→ {e.get('status')}"
                )
            lines.append("Текст відповіді Qwen:")
            lines.append(wrap(r.get("answer") or "(порожньо)"))
            lines.append("-" * 60)
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Notebook content
# ---------------------------------------------------------------------------

md(
    "# Лабораторна робота 5 — Великі мовні моделі\n\n"
    "Єдиний демонстраційний ноутбук: **код запуску** моделей і **усі результати** "
    "частин 1–4 (tool calls, текст Qwen, графи).\n\n"
    "За замовчуванням клітинки показують уже збережені результати з `outputs/`. "
    "Щоб знову викликати Ollama, поставте `LIVE = True` у відповідних клітинках "
    "(на CPU це довго)."
)

md("## 1. Імпорти та параметри")
code(
    "import json, os, sys, time, textwrap\n"
    "from IPython.display import display, Image, Markdown\n"
    "import pandas as pd\n"
    "\n"
    "ROOT = os.getcwd()\n"
    "sys.path.insert(0, os.path.join(ROOT, 'src'))\n"
    "\n"
    "import sdn_env\n"
    "from knowledge import KnowledgeController, LinkStateDatabase\n"
    "from tools import SdnToolbox\n"
    "from llm_controller import LlmController, SYSTEM_PROMPT\n"
    "from visualization import (\n"
    "    plot_single_path, save_step_figures, render_state_ascii,\n"
    ")\n"
    "\n"
    "QUANTIZED_MODEL = 'qwen2.5:3b'                 # Q4_K_M\n"
    "Q8_MODEL = 'qwen2.5:3b-instruct-q8_0'          # Q8_0\n"
    "SOURCE, DESTINATION = sdn_env.SOURCE, sdn_env.DESTINATION\n"
    "print('OK', SOURCE, '->', DESTINATION)\n"
    "print('SYSTEM_PROMPT (скорочено):')\n"
    "print(SYSTEM_PROMPT[:280], '...')",
    "OK 10.0.0.0 -> 10.0.4.4\n"
    "SYSTEM_PROMPT (скорочено):\n"
    + SYSTEM_PROMPT[:280] + " ...\n"
    if False else
    "OK 10.0.0.0 -> 10.0.4.4\nSYSTEM_PROMPT (скорочено):\n"
    + __import__("llm_controller", fromlist=["SYSTEM_PROMPT"]).SYSTEM_PROMPT[:280]
    + " ...\n",
)

# fix the awkward ternary - rebuild that cell properly
cells.pop()
from llm_controller import SYSTEM_PROMPT as _SP
code(
    "import json, os, sys, time, textwrap\n"
    "from IPython.display import display, Image, Markdown\n"
    "import pandas as pd\n"
    "\n"
    "ROOT = os.getcwd()\n"
    "sys.path.insert(0, os.path.join(ROOT, 'src'))\n"
    "\n"
    "import sdn_env\n"
    "from knowledge import KnowledgeController, LinkStateDatabase\n"
    "from tools import SdnToolbox\n"
    "from llm_controller import LlmController, SYSTEM_PROMPT\n"
    "from visualization import (\n"
    "    plot_single_path, save_step_figures, render_state_ascii,\n"
    ")\n"
    "\n"
    "QUANTIZED_MODEL = 'qwen2.5:3b'                 # Q4_K_M\n"
    "Q8_MODEL = 'qwen2.5:3b-instruct-q8_0'          # Q8_0\n"
    "SOURCE, DESTINATION = sdn_env.SOURCE, sdn_env.DESTINATION\n"
    "print('OK', SOURCE, '->', DESTINATION)\n"
    "print('SYSTEM_PROMPT (скорочено):')\n"
    "print(SYSTEM_PROMPT[:280], '...')",
    "OK 10.0.0.0 -> 10.0.4.4\nSYSTEM_PROMPT (скорочено):\n"
    + _SP[:280] + " ...\n",
)

md(
    "## 2. Характеристики обладнання\n\n"
    "- CPU: 12th Gen Intel Core i5-12500H\n"
    "- RAM: 16 ГБ\n"
    "- GPU: RTX 3060 Laptop (але Ollama з `num_gpu=0` — інференс на CPU)\n"
)

md("## 3. Частина 1. Бенчмарк локальних моделей")
code(
    "# Повний бенчмарк: python benchmark_part1.py\n"
    "# Тут показуємо збережені середні значення.\n"
    "part1 = json.load(open('outputs/part1_benchmark.json', encoding='utf-8'))\n"
    "rows = []\n"
    "for key, label, qn in [\n"
    "    ('qwen2.5:3b-instruct-fp16', 'Qwen2.5-3B FP16', 'FP16'),\n"
    "    ('qwen2.5:3b-instruct-q8_0', 'Qwen2.5-3B Q8_0', 'Q8_0'),\n"
    "    ('qwen2.5:3b', 'Qwen2.5-3B Q4_K_M', 'Q4_K_M'),\n"
    "]:\n"
    "    d = part1[key]\n"
    "    rows.append({\n"
    "        'Модель': label, 'Квантування': qn,\n"
    "        'Розмір, ГБ': d['size_gb'], 'RAM, ГБ': d['ram_used_gb'],\n"
    "        'Обробка, ток/с': d['prompt_processing_tok_s'],\n"
    "        'Генерація, ток/с': d['generation_tok_s'],\n"
    "        'prompt_tokens': d['prompt_tokens'],\n"
    "        'eval_tokens': d['eval_tokens'],\n"
    "    })\n"
    "display(pd.DataFrame(rows))",
    "\n".join(
        f"{label}: size={PART1[key]['size_gb']} RAM={PART1[key]['ram_used_gb']} "
        f"prompt={PART1[key]['prompt_processing_tok_s']} gen={PART1[key]['generation_tok_s']}"
        for key, label, _ in [
            ("qwen2.5:3b-instruct-fp16", "FP16", None),
            ("qwen2.5:3b-instruct-q8_0", "Q8_0", None),
            ("qwen2.5:3b", "Q4_K_M", None),
        ]
    ) + "\n",
)

# ---- Part 2 ----
md(
    "## 4. Частина 2. NL-агент і сценарії\n\n"
    "Топологія 5×5, `10.0.0.0 → 10.0.4.4`. Модель частини 2: **qwen2.5:3b (Q4_K_M)**. "
    "Інструменти: `get_flow_state`, `get_link_stats`, `install_rule`, `install_path`, "
    "`get_topology`, `simulate_link_failure`."
)

code(
    "graph = sdn_env.generate_grid_graph()\n"
    "opt_delay, opt_path = sdn_env.shortest_path_delay(graph, SOURCE, DESTINATION)\n"
    "print(f'Вузлів={graph.number_of_nodes()}, ребер={graph.number_of_edges()}, "
    "оптимум={opt_delay} мс')\n"
    "print(render_state_ascii(graph, opt_path, SOURCE, DESTINATION))\n"
    "display(Image(filename='figures/topology.png'))",
    f"Вузлів={_topo_graph.number_of_nodes()}, ребер={_topo_graph.number_of_edges()}, "
    f"оптимум={_topo_delay} мс\n{ASCII_TOPO}\n",
    [os.path.join(ROOT, "figures", "topology.png")],
)

md(
    "### 4.1. Код запуску частини 2 (Ollama)\n\n"
    "Нижче — той самий цикл, що в `scripts/run_part2.py`. "
    "`LIVE = False` лише демонструє код; поставте `True`, щоб реально викликати модель."
)

_part2_launch = r'''
LIVE = False  # True = реальний прогін через Ollama (довго на CPU)

SCENARIOS = [
    {"id": "s1_basic_path",
     "query": "Проклади шлях для потоку від вузла 10.0.0.0 до вузла 10.0.4.4"},
    {"id": "s2_avoid",
     "query": ("Проклади шлях до 10.0.4.4 в обхід вузлів 10.0.2.2 та 10.0.2.3, "
               "які виводяться на обслуговування. Відправник — 10.0.0.0.")},
    {"id": "s3_head_and_delays",
     "query": ("На якому вузлі зараз зупинилась побудова шляху і які канали з "
               "нього мають найменшу затримку?")},
    {"id": "s4_transfer",
     "query": ("Скільки триватиме передача останнього потоку і який канал є "
               "вузьким місцем?")},
    {"id": "s5_failure",
     "query": ("Канал між 10.0.1.1 та 10.0.1.2 вийшов з ладу – перебудуй шлях "
               "потоку від 10.0.0.0 до 10.0.4.4.")},
]

def run_part2_live(model=QUANTIZED_MODEL):
    import ollama
    graph = sdn_env.generate_grid_graph()
    results = {"model": model, "scenarios": []}
    shared_env = None
    for sc in SCENARIOS:
        reuse = shared_env if sc["id"] in ("s3_head_and_delays", "s4_transfer") else None
        env = reuse or sdn_env.NetworkEnvironment(graph, SOURCE, DESTINATION)
        tb = SdnToolbox(env)
        agent = LlmController(model, tb, num_gpu=0, verbose=True)
        print("\n===", sc["id"], "===")
        res = agent.run(sc["query"])
        shared_env = env
        # збереження фігур
        fig_dir = os.path.join("figures", f"part2_{sc['id']}")
        os.makedirs(os.path.join(fig_dir, "steps"), exist_ok=True)
        path = res.get("flow_state", {}).get("path") or []
        if path:
            plot_single_path(graph, path, SOURCE, DESTINATION,
                             title=sc["id"], filename=os.path.join(fig_dir, "final_path.png"))
        snaps = res.get("path_snapshots") or []
        if snaps:
            save_step_figures(graph, snaps, SOURCE, DESTINATION,
                              out_dir=os.path.join(fig_dir, "steps"), prefix="step")
        results["scenarios"].append({
            "id": sc["id"], "query": sc["query"], "answer": res.get("answer"),
            "log": res.get("log"), "flow_state": res.get("flow_state"),
            "last_path_result": res.get("last_path_result"),
            "path_snapshots": snaps, "transfer": res.get("transfer"),
        })
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/part2_scenarios.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Збережено outputs/part2_scenarios.json")
    return results

if LIVE:
    part2 = run_part2_live()
else:
    part2 = json.load(open("outputs/part2_scenarios.json", encoding="utf-8"))
    print("LIVE=False — завантажено збережені результати частини 2")
    print("Сценарії:", [s["id"] for s in part2["scenarios"]])
'''

code(
    _part2_launch,
    "LIVE=False — завантажено збережені результати частини 2\n"
    "Сценарії: " + str([s["id"] for s in PART2["scenarios"]]) + "\n",
)

md(
    "### 4.2. Результати сценаріїв (tools + графи + текст Qwen)\n\n"
    "Нижче — повний розбір кожного сценарію для демонстрації."
)

code(
    "import glob\n"
    "\n"
    "def show_scenario(sc):\n"
    "    print('=' * 78)\n"
    "    print('СЦЕНАРІЙ:', sc.get('id'))\n"
    "    print('=' * 78)\n"
    "    print('ОПЕРАТОР:')\n"
    "    print(sc.get('query') or '')\n"
    "    tools = [e.get('tool') for e in (sc.get('log') or [])\n"
    "             if e.get('tool') and not str(e.get('tool')).startswith('(')\n"
    "             and e.get('status') != 'answer']\n"
    "    print('\\nІНСТРУМЕНТИ QWEN:')\n"
    "    for i, t in enumerate(dict.fromkeys(tools), 1):\n"
    "        print(f'  {i}. {t}')\n"
    "    for entry in sc.get('log') or []:\n"
    "        tool = entry.get('tool') or ''\n"
    "        status = entry.get('status') or ''\n"
    "        if status == 'answer' or tool == '(відповідь)':\n"
    "            print('-' * 78)\n"
    "            print('ВІДПОВІДЬ МОДЕЛІ (текст):')\n"
    "            print(entry.get('result') or sc.get('answer') or '')\n"
    "            continue\n"
    "        if tool.startswith('('):\n"
    "            continue\n"
    "        print('-' * 78)\n"
    "        print('ВИКЛИК ІНСТРУМЕНТА:', tool)\n"
    "        print('Аргументи:', json.dumps(entry.get('arguments') or {}, ensure_ascii=False))\n"
    "        print('Статус:', status)\n"
    "        raw = entry.get('result') or ''\n"
    "        try:\n"
    "            print('Результат:', json.dumps(json.loads(raw), ensure_ascii=False, indent=2))\n"
    "        except Exception:\n"
    "            print('Результат:', raw)\n"
    "    sid = sc.get('id')\n"
    "    for p in sorted(glob.glob(f'figures/part2_{sid}/steps/step_*.png')):\n"
    "        print('ГРАФ:', p)\n"
    "        display(Image(filename=p))\n"
    "    fp = f'figures/part2_{sid}/final_path.png'\n"
    "    if os.path.exists(fp):\n"
    "        print('ФІНАЛЬНИЙ ШЛЯХ:', fp)\n"
    "        display(Image(filename=fp))\n"
    "\n"
    "print('Функція show_scenario(sc) готова')",
    "Функція show_scenario(sc) готова\n",
)

for sc in PART2["scenarios"]:
    md(f"#### Сценарій `{sc['id']}`")
    imgs = scenario_images(sc)
    out_txt = format_scenario_transcript(sc)
    out_txt += "".join(
        f"ГРАФ: figures/part2_{sc['id']}/steps/{os.path.basename(p)}\n"
        for p in imgs if "steps" in p.replace("\\", "/")
    )
    if os.path.exists(os.path.join(ROOT, "figures", f"part2_{sc['id']}", "final_path.png")):
        out_txt += f"ФІНАЛЬНИЙ ШЛЯХ: figures/part2_{sc['id']}/final_path.png\n"
    code(
        f"show_scenario(next(s for s in part2['scenarios'] if s['id'] == {sc['id']!r}))",
        out_txt,
        imgs,
    )

code(
    "display(Image(filename='figures/paths.png'))\n"
    "print('Порівняння шляхів: базовий / обхід / після збою')",
    "Порівняння шляхів: базовий / обхід / після збою\n",
    [os.path.join(ROOT, "figures", "paths.png")],
)

# ---- Part 3 ----
md(
    "## 5. Частина 3. Пояснення маршруту\n\n"
    "Моделі передається журнал контролера; вона пояснює вибір шляху. "
    "Потім пояснення перевіряється за фактами."
)

_part3_launch = r'''
LIVE_PART3 = False  # True = знову питати модель

def run_part3_live(model=QUANTIZED_MODEL):
    part2_data = json.load(open("outputs/part2_scenarios.json", encoding="utf-8"))
    sc = next(s for s in part2_data["scenarios"] if s.get("last_path_result"))
    # журнал як у scripts/run_part3.py
    from scripts import run_part3 as rp3  # може не імпортуватись; дублюємо коротко
    raise SystemExit("Використайте блок нижче або scripts/run_part3.py")

# Простий live-запуск без окремого модуля:
def explain_from_part2(model=QUANTIZED_MODEL):
    part2_data = json.load(open("outputs/part2_scenarios.json", encoding="utf-8"))
    topo = part2_data["topology"]
    sc = next(
        (s for s in part2_data["scenarios"] if s["id"] == "s1_basic_path" and s.get("last_path_result")),
        None,
    ) or next(s for s in part2_data["scenarios"] if s.get("last_path_result"))
    pr = sc["last_path_result"]
    journal_lines = [
        "ЖУРНАЛ РОБОТИ КОНТРОЛЕРА",
        f"Відправник: {topo['source']}, отримувач: {topo['destination']}",
        f"Встановлений шлях: {' -> '.join(pr.get('path') or [])}",
        f"Затримка шляху: {pr.get('delay_ms')} мс",
        f"Оптимальна затримка: {pr.get('optimal_delay_ms')} мс",
        f"Час передачі: {pr.get('transmission_ms')} мс",
        f"Вузьке місце: {pr.get('bottleneck')}",
    ]
    for link in pr.get("links") or []:
        journal_lines.append(
            f"  {link['a']} — {link['b']}: delay={link['delay_ms']} мс, "
            f"BW={link['bandwidth_MBps']} МБ/с"
        )
    journal = "\n".join(journal_lines)
    env = sdn_env.NetworkEnvironment(sdn_env.generate_grid_graph(), SOURCE, DESTINATION)
    agent = LlmController(model, SdnToolbox(env), num_gpu=0, verbose=True)
    expl = agent.explain_route(journal)
    out = {
        "scenario_id": sc["id"], "journal": journal,
        "explanation": expl["explanation"], "model": model,
        "inaccuracies": [],  # повна перевірка — у scripts/run_part3.py
        "facts": {
            "path": pr.get("path"), "delay_ms": pr.get("delay_ms"),
            "optimal_delay_ms": pr.get("optimal_delay_ms"),
            "bottleneck": pr.get("bottleneck"),
            "transmission_ms": pr.get("transmission_ms"),
        },
    }
    with open("outputs/part3_explanation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out

if LIVE_PART3:
    part3 = explain_from_part2()
else:
    part3 = json.load(open("outputs/part3_explanation.json", encoding="utf-8"))
    print("LIVE_PART3=False — завантажено outputs/part3_explanation.json")
'''

code(
    _part3_launch,
    "LIVE_PART3=False — завантажено outputs/part3_explanation.json\n",
)

_p3_out = []
_p3_out.append("ЖУРНАЛ, ПЕРЕДАНИЙ МОДЕЛІ:\n")
_p3_out.append(wrap(PART3.get("journal") or ""))
_p3_out.append("\n\nПОЯСНЕННЯ QWEN:\n")
_p3_out.append(wrap(PART3.get("explanation") or ""))
_p3_out.append("\n\nНЕТОЧНОСТІ (перевірка за фактами):\n")
if PART3.get("inaccuracies"):
    for i in PART3["inaccuracies"]:
        _p3_out.append(f"- [{i['type']}] {i['detail']}")
else:
    _p3_out.append("- не виявлено")
_p3_out.append("\n\nФАКТИ:\n")
_p3_out.append(json.dumps(PART3.get("facts") or {}, ensure_ascii=False, indent=2))

code(
    "print('ЖУРНАЛ, ПЕРЕДАНИЙ МОДЕЛІ:')\n"
    "print(part3['journal'])\n"
    "print('\\nПОЯСНЕННЯ QWEN:')\n"
    "print(part3['explanation'])\n"
    "print('\\nНЕТОЧНОСТІ:')\n"
    "for i in part3.get('inaccuracies') or []:\n"
    "    print('-', i['type'] + ':', i['detail'])\n"
    "print('\\nФАКТИ:')\n"
    "print(json.dumps(part3.get('facts') or {}, ensure_ascii=False, indent=2))",
    "\n".join(_p3_out) + "\n",
)

# ---- Part 4 ----
md(
    "## 6. Частина 4. Критична оцінка (≥10 запитів × 2 квантування)\n\n"
    "Однакові запити на **Q4_K_M** і **Q8_0**. У таблиці — expected/actual tools, "
    "правильність, тип помилки; нижче — повний текст відповідей."
)

_part4_launch = r'''
LIVE_PART4 = False  # True = прогін 11×2 через Ollama (дуже довго)

PART4_QUERIES = [
    {"id": "q01", "query": "Проклади шлях від 10.0.0.0 до 10.0.4.4",
     "expected_tools": ["install_path"], "kind": "valid"},
    {"id": "q02", "query": "Проклади шлях від 10.0.0.0 до 10.0.4.4 в обхід 10.0.1.1 та 10.0.3.2",
     "expected_tools": ["install_path"], "kind": "valid_avoid"},
    {"id": "q03", "query": "Проклади шлях до вузла 10.0.9.9",
     "expected_tools": ["install_path"], "kind": "impossible_dst"},
    {"id": "q04", "query": "Який зараз стан потоку?",
     "expected_tools": ["get_flow_state"], "kind": "valid_state"},
    {"id": "q05", "query": "Покажи телеметрію вузла 10.0.2.2: затримки і пропускну здатність",
     "expected_tools": ["get_link_stats"], "kind": "valid_stats"},
    {"id": "q06", "query": "Покажи топологію мережі з бази знань",
     "expected_tools": ["get_topology"], "kind": "valid_topo"},
    {"id": "q07", "query": "Канал між 10.0.0.1 і 10.0.0.2 вийшов з ладу",
     "expected_tools": ["simulate_link_failure"], "kind": "valid_failure"},
    {"id": "q08", "query": "Проклади якийсь хороший шлях якомога швидше",
     "expected_tools": ["install_path"], "kind": "ambiguous"},
    {"id": "q09", "query": (
        "Проклади шлях від 10.0.0.0 до 10.0.4.4 в обхід усіх вузлів "
        "10.0.1.0, 10.0.1.1, 10.0.1.2, 10.0.1.3, 10.0.1.4, "
        "10.0.2.0, 10.0.2.1, 10.0.2.2, 10.0.2.3, 10.0.2.4, "
        "10.0.3.0, 10.0.3.1, 10.0.3.2, 10.0.3.3, 10.0.3.4"),
     "expected_tools": ["install_path"], "kind": "impossible_avoid"},
    {"id": "q10", "query": "Встанови правило на схід і одразу скасуй його",
     "expected_tools": ["install_rule"], "kind": "valid_rule"},
    {"id": "q11", "query": "Скільки триватиме передача потоку і де вузьке місце?",
     "expected_tools": ["get_flow_state", "install_path"], "kind": "needs_path_or_state"},
]

def actual_tools(log):
    return [e["tool"] for e in log if e.get("tool") and not e["tool"].startswith("(")]

if LIVE_PART4:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_part4", os.path.join("scripts", "run_part4.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()  # пише outputs/part4_eval.json
    part4 = json.load(open("outputs/part4_eval.json", encoding="utf-8"))
else:
    part4 = json.load(open("outputs/part4_eval.json", encoding="utf-8"))
    print("LIVE_PART4=False — завантажено outputs/part4_eval.json")
    for model, block in part4["models"].items():
        print(f"  {block['label']}: {block['correct_count']}/{block['total']} = {block['accuracy']}")
'''

code(
    _part4_launch,
    "LIVE_PART4=False — завантажено outputs/part4_eval.json\n"
    + "\n".join(
        f"  {b['label']}: {b['correct_count']}/{b['total']} = {b['accuracy']}"
        for b in PART4["models"].values()
    )
    + "\n",
)

# summary tables
_p4_table_lines = []
for model, block in PART4["models"].items():
    _p4_table_lines.append(f"\n=== {block['label']} ({model}) ===")
    for r in block["rows"]:
        _p4_table_lines.append(
            f"{r['id']}: OK={r['correct']} | exp={','.join(r['expected_tools'])} | "
            f"act={','.join(r['actual_tools'])} | err={','.join(r['error_types']) or '—'}"
        )

code(
    "for model, block in part4['models'].items():\n"
    "    print(f\"\\n=== {block['label']} ({model}) "
    "accuracy={block['accuracy']} ===\")\n"
    "    rows = [{\n"
    "        'id': r['id'],\n"
    "        'запит': r['query'][:50],\n"
    "        'очікувані': ','.join(r['expected_tools']),\n"
    "        'фактичні': ','.join(r['actual_tools']),\n"
    "        'OK': r['correct'],\n"
    "        'помилки': ','.join(r['error_types']) or '—',\n"
    "    } for r in block['rows']]\n"
    "    display(pd.DataFrame(rows))",
    "\n".join(_p4_table_lines) + "\n",
)

md("### 6.1. Повні відповіді моделі (tools + текст) для обох квантувань")
code(
    "def show_part4_details(part4):\n"
    "    for model, block in part4['models'].items():\n"
    "        print('=' * 78)\n"
    "        print(f\"МОДЕЛЬ: {block['label']} ({model}) "
    "{block['correct_count']}/{block['total']}\")\n"
    "        print('=' * 78)\n"
    "        for r in block['rows']:\n"
    "            print(f\"\\n### {r['id']} | correct={r['correct']} | "
    "errors={r['error_types'] or '—'}\")\n"
    "            print('Запит:', r['query'])\n"
    "            print('Очікувані:', ', '.join(r['expected_tools']))\n"
    "            print('Фактичні:', ', '.join(r['actual_tools']))\n"
    "            print('Журнал:')\n"
    "            for e in r.get('log') or []:\n"
    "                tool = e.get('tool') or ''\n"
    "                if e.get('status') == 'answer' or tool == '(відповідь)':\n"
    "                    continue\n"
    "                if tool.startswith('('):\n"
    "                    continue\n"
    "                print(f\"  - {tool}({json.dumps(e.get('arguments') or {}, ensure_ascii=False)}) → {e.get('status')}\")\n"
    "            print('Текст відповіді Qwen:')\n"
    "            print(textwrap.fill(r.get('answer') or '(порожньо)', width=96))\n"
    "            print('-' * 60)\n"
    "\n"
    "show_part4_details(part4)",
    format_part4_block(),
)

md(
    "## 7. Висновок\n\n"
    "- **Частина 1:** квантування зменшує розмір/RAM і прискорює генерацію на CPU.\n"
    "- **Частина 2:** LLM — NL-інтерфейс; шлях ставить `install_path` / knowledge-controller.\n"
    "- **Частина 3:** текстове пояснення треба верифікувати (виявили плутанину мс/МБ/с).\n"
    "- **Частина 4:** Q4_K_M "
    f"**{PART4['models']['qwen2.5:3b']['accuracy']}**, Q8_0 "
    f"**{PART4['models']['qwen2.5:3b-instruct-q8_0']['accuracy']}** на 11 запитах.\n\n"
    "LLM доцільно доручати розбір наміру й вибір tools; обчислення шляху та стан "
    "мережі — детермінованому контролеру."
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

out = os.path.join(ROOT, "lab5.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Ноутбук збережено:", out)
print("Клітинок:", len(cells))

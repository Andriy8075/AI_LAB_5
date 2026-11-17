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


def _format_part4_rows(models_key: str) -> list:
    lines = []
    for model, block in PART4.get(models_key, {}).items():
        lines.append("=" * 78)
        lines.append(
            f"МОДЕЛЬ: {block['label']} ({model}) — "
            f"{block['correct_count']}/{block['total']} "
            f"(accuracy={block['accuracy']})"
        )
        lines.append("=" * 78)
        for r in block["rows"]:
            lang = r.get("lang", "uk")
            lines.append("")
            lines.append(
                f"### {r['id']} | lang={lang} | correct={r['correct']} | "
                f"errors={r['error_types'] or '—'}"
            )
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
    return lines


def format_part4_block() -> str:
    return "\n".join(_format_part4_rows("models")) + "\n"


def format_part4_en_block() -> str:
    if not PART4.get("models_en"):
        return "(немає EN-результатів — поставте LIVE_PART4=True у клітинці частини 4)\n"
    return "\n".join(_format_part4_rows("models_en")) + "\n"


def format_lang_compare_block() -> str:
    lc = PART4.get("language_compare") or {}
    lines = []
    summary = lc.get("summary") or {}
    if summary:
        lines.append("Підсумок UA ↔ EN (однакові сценарії):")
        for model, s in summary.items():
            lines.append(
                f"  {s['label']} ({model}): "
                f"uk={s['uk_correct']}/{s['pairs']} ({s['uk_accuracy']}), "
                f"en={s['en_correct']}/{s['pairs']} ({s['en_accuracy']}), "
                f"згода={s['same_outcome']}/{s['pairs']} ({s['agreement']})"
            )
        lines.append("")
    for p in lc.get("pairs") or []:
        lines.append(f"### pair {p['pair_id']} | kind={p['kind']}")
        lines.append(f"  UK: {p['uk_query']}")
        lines.append(f"  EN: {p['en_query']}")
        for model, m in (p.get("by_model") or {}).items():
            lines.append(
                f"  [{m['label']}] uk_OK={m['uk_correct']} tools={m['uk_tools']} | "
                f"en_OK={m['en_correct']} tools={m['en_tools']} | "
                f"same={m['same_outcome']}"
            )
        lines.append("")
    return ("\n".join(lines) + "\n") if lines else "(немає language_compare)\n"


from llm_controller import SYSTEM_PROMPT as _SP

_MODULE_FILES = [
    "sdn_env.py",
    "knowledge.py",
    "tools.py",
    "llm_controller.py",
    "visualization.py",
]

# ---------------------------------------------------------------------------
# Notebook content
# ---------------------------------------------------------------------------

md(
    "# Лабораторна робота 5 — Великі мовні моделі\n\n"
    "Єдиний файл для захисту: **код модулів**, **запуски моделей** частин 1–4 "
    "і збережені результати.\n\n"
    "За замовчуванням клітинки показують `outputs/`. "
    "Щоб знову викликати Ollama, поставте відповідний `LIVE_* = True` "
    "(на CPU це довго)."
)

md("## 1. Імпорти та параметри")
code(
    "import math\n"
    "import random\n"
    "import json\n"
    "import os\n"
    "import sys\n"
    "import time\n"
    "import textwrap\n"
    "import statistics\n"
    "import re\n"
    "\n"
    "from IPython.display import display, Image, Markdown\n"
    "\n"
    "import networkx as nx\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import pandas as pd\n"
    "import ollama\n"
    "import psutil\n"
    "\n"
    "ROOT = os.getcwd()\n"
    "sys.path.insert(0, os.path.join(ROOT, 'src'))\n"
    "os.makedirs('outputs', exist_ok=True)\n"
    "os.makedirs('figures', exist_ok=True)\n"
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
    "FP16_MODEL = 'qwen2.5:3b-instruct-fp16'\n"
    "SOURCE, DESTINATION = sdn_env.SOURCE, sdn_env.DESTINATION\n"
    "\n"
    "print('OK', SOURCE, '->', DESTINATION)\n"
    "print('Моделі:', QUANTIZED_MODEL, '|', Q8_MODEL, '|', FP16_MODEL)\n"
    "print('SYSTEM_PROMPT (скорочено):')\n"
    "print(SYSTEM_PROMPT[:280], '...')\n",
    "OK 10.0.0.0 -> 10.0.4.4\n"
    "Моделі: qwen2.5:3b | qwen2.5:3b-instruct-q8_0 | qwen2.5:3b-instruct-fp16\n"
    "SYSTEM_PROMPT (скорочено):\n"
    + _SP[:280] + " ...\n",
)

md(
    "### 1.1. Код модулів\n\n"
    "Звичайні code-клітинки (не рядок у змінній і не `exec`). "
    "Робочі імпорти — у клітинці вище (`sys.path` → `src/`)."
)

for _fname in _MODULE_FILES:
    _path = os.path.join(ROOT, "src", _fname)
    _body = open(_path, encoding="utf-8").read().rstrip() + "\n"
    md(f"#### `{_fname}`")
    code(_body, f"# модуль {_fname}\n")

md(
    "## 2. Характеристики обладнання\n\n"
    "- CPU: 12th Gen Intel Core i5-12500H\n"
    "- RAM: 16 ГБ\n"
    "- GPU: RTX 3060 Laptop (але Ollama з `num_gpu=0` — інференс на CPU)\n"
)

_part1_out = "\n".join(
    f"{label}: size={PART1[key]['size_gb']} RAM={PART1[key]['ram_used_gb']} "
    f"prompt={PART1[key]['prompt_processing_tok_s']} gen={PART1[key]['generation_tok_s']}"
    for key, label in [
        ("qwen2.5:3b-instruct-fp16", "FP16"),
        ("qwen2.5:3b-instruct-q8_0", "Q8_0"),
        ("qwen2.5:3b", "Q4_K_M"),
    ]
) + "\n"

md(
    "## 3. Частина 1. Бенчмарк локальних моделей\n\n"
    "Вимірювання FP16 / Q8_0 / Q4_K_M на однаковому запиті (≥200 токенів), "
    "`num_predict=128`, 3 повторення. Поставте `LIVE_PART1 = True`, щоб "
    "перезапустити бенчмарк через Ollama."
)
code(
    r'''
LIVE_PART1 = False  # True = повний бенчмарк через Ollama (довго на CPU)

PART1_MODELS = [FP16_MODEL, Q8_MODEL, QUANTIZED_MODEL]
PART1_REPS = 3
PART1_NUM_PREDICT = 128  # лише частина 1 порівнює швидкість
PART1_NUM_GPU = 0
PART1_PROMPT = (
    "Поясни детально, як працює протокол динамічної маршрутизації OSPF у "
    "комп'ютерних мережах. Опиши його основні компоненти: повідомлення Hello, "
    "оголошення про стан каналів (LSA), базу даних станів каналів (LSDB) та "
    "алгоритм Дейкстри. Розкажи, як маршрутизатори обмінюються інформацією, "
    "як будується карта мережі та як обчислюються найкоротші шляхи до кожної "
    "підмережі. Порівняй OSPF з протоколом RIP за критеріями швидкості "
    "збіжності, масштабованості, типу метрики та споживання ресурсів. Наведи "
    "конкретний приклад мережі з чотирма маршрутизаторами, з'єднаними різними "
    "каналами з різною вартістю, і поясни покроково, який маршрут обере OSPF "
    "між двома віддаленими вузлами. Додатково поясни поняття областей (areas), "
    "роль магістральної області та призначення призначених маршрутизаторів. "
    "На завершення перелічи типові проблеми, що виникають під час налаштування "
    "OSPF у реальних мережах, і запропонуй способи їх діагностики та усунення."
)

def part1_unload_all():
    for m in PART1_MODELS:
        try:
            ollama.chat(
                model=m, messages=[{"role": "user", "content": " "}],
                options={"num_predict": 1, "num_gpu": 0}, keep_alive=0,
            )
        except Exception:
            pass
    time.sleep(2)

def part1_model_size_gb(model):
    for m in ollama.list()["models"]:
        if m["model"] == model:
            return m["size"] / 1e9
    return 0.0

def part1_ram_used_gb():
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / 1e9

def part1_ollama_ps_size_gb(model):
    for m in ollama.ps().get("models", []):
        if m["model"] == model:
            return m["size"] / 1e9
    return 0.0

def part1_run_once(model):
    resp = ollama.generate(
        model=model, prompt=PART1_PROMPT,
        options={
            "num_predict": PART1_NUM_PREDICT, "temperature": 0.0,
            "num_gpu": PART1_NUM_GPU, "num_ctx": 4096,
        },
    )
    pe_count = resp.get("prompt_eval_count", 0)
    pe_dur = resp.get("prompt_eval_duration", 0) or 0
    ev_count = resp.get("eval_count", 0)
    ev_dur = resp.get("eval_duration", 0) or 0
    return {
        "prompt_tokens": pe_count,
        "prompt_processing_s": pe_dur / 1e9,
        "prompt_tok_s": pe_count / (pe_dur / 1e9) if pe_dur else 0.0,
        "eval_tokens": ev_count,
        "generation_s": ev_dur / 1e9,
        "gen_tok_s": ev_count / (ev_dur / 1e9) if ev_dur else 0.0,
        "total_s": resp.get("total_duration", 0) / 1e9,
    }

def run_part1_benchmark():
    results = {}
    for model in PART1_MODELS:
        print(f"\n=== {model} ===", flush=True)
        part1_unload_all()
        base_ram = part1_ram_used_gb()
        warm = ollama.generate(
            model=model, prompt="Привіт",
            options={"num_predict": 4, "num_gpu": PART1_NUM_GPU},
        )
        print(f"  прогрів: {warm.get('total_duration', 0)/1e9:.1f} с", flush=True)
        loaded_ram = part1_ram_used_gb()
        ps_size = part1_ollama_ps_size_gb(model)
        runs = []
        for rep in range(1, PART1_REPS + 1):
            part1_unload_all()
            r = part1_run_once(model)
            runs.append(r)
            print(
                f"  реп {rep}: prompt={r['prompt_tok_s']:.2f} ток/с, "
                f"gen={r['gen_tok_s']:.2f} ток/с", flush=True,
            )
        results[model] = {
            "size_gb": round(part1_model_size_gb(model), 2),
            "ram_used_gb": round(loaded_ram - base_ram, 2),
            "ollama_ps_size_gb": round(ps_size, 2),
            "prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in runs)),
            "prompt_processing_tok_s": round(
                statistics.mean(r["prompt_tok_s"] for r in runs), 2),
            "eval_tokens": round(statistics.mean(r["eval_tokens"] for r in runs)),
            "generation_tok_s": round(
                statistics.mean(r["gen_tok_s"] for r in runs), 2),
            "runs": runs,
        }
        part1_unload_all()
    with open("outputs/part1_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Збережено outputs/part1_benchmark.json")
    return results

if LIVE_PART1:
    part1 = run_part1_benchmark()
else:
    part1 = json.load(open("outputs/part1_benchmark.json", encoding="utf-8"))
    print("LIVE_PART1=False — завантажено outputs/part1_benchmark.json")

rows = []
for key, label, qn in [
    (FP16_MODEL, "Qwen2.5-3B FP16", "FP16"),
    (Q8_MODEL, "Qwen2.5-3B Q8_0", "Q8_0"),
    (QUANTIZED_MODEL, "Qwen2.5-3B Q4_K_M", "Q4_K_M"),
]:
    d = part1[key]
    rows.append({
        "Модель": label, "Квантування": qn,
        "Розмір, ГБ": d["size_gb"], "RAM, ГБ": d["ram_used_gb"],
        "Обробка, ток/с": d["prompt_processing_tok_s"],
        "Генерація, ток/с": d["generation_tok_s"],
        "prompt_tokens": d["prompt_tokens"],
        "eval_tokens": d["eval_tokens"],
    })
display(pd.DataFrame(rows))
''',
    "LIVE_PART1=False — завантажено outputs/part1_benchmark.json\n" + _part1_out,
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
    "Повний цикл NL → tool calling → відповідь моделі. "
    "`LIVE = False` показує збережені результати; `True` — реальний прогін."
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
    graph = sdn_env.generate_grid_graph()
    opt_delay, opt_path = sdn_env.shortest_path_delay(graph, SOURCE, DESTINATION)
    results = {
        "topology": {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "source": SOURCE,
            "destination": DESTINATION,
            "opt_delay": opt_delay,
            "opt_path": [sdn_env.node_address(n) for n in opt_path],
        },
        "model": model,
        "scenarios": [],
    }
    shared_env = None
    for sc in SCENARIOS:
        reuse = shared_env if sc["id"] in ("s3_head_and_delays", "s4_transfer") else None
        env = reuse or sdn_env.NetworkEnvironment(graph, SOURCE, DESTINATION)
        tb = SdnToolbox(env)
        agent = LlmController(model, tb, num_gpu=0, verbose=True)
        print("\n===", sc["id"], "===")
        res = agent.run(sc["query"])
        shared_env = env
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
        lpr = res.get("last_path_result") or {}
        results["scenarios"].append({
            "id": sc["id"], "query": sc["query"], "answer": res.get("answer"),
            "log": res.get("log"), "flow_state": res.get("flow_state"),
            "last_path_result": {k: v for k, v in lpr.items() if k != "snapshots"} or None,
            "path_snapshots": snaps, "transfer": res.get("transfer"),
            "steps": res.get("steps"), "tool_calls": res.get("tool_calls"),
            "llm_time_s": res.get("llm_time_s"),
        })
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
    "Потім пояснення перевіряється за фактами (у цій же клітинці)."
)

_part3_launch = r'''
LIVE_PART3 = False  # True = знову питати модель

def build_part3_journal(sc, topo):
    path_res = sc.get("last_path_result") or {}
    lines = [
        "ЖУРНАЛ РОБОТИ КОНТРОЛЕРА",
        f"Відправник: {topo['source']}, отримувач: {topo['destination']}",
        f"Встановлений шлях: {' -> '.join(path_res.get('path') or [])}",
        f"Затримка шляху: {path_res.get('delay_ms')} мс",
        f"Оптимальний шлях (Дейкстра без avoid): "
        f"{' -> '.join(path_res.get('optimal_path') or topo.get('opt_path') or [])}",
        f"Оптимальна затримка: {path_res.get('optimal_delay_ms', topo.get('opt_delay'))} мс",
        f"Час передачі: {path_res.get('transmission_ms')} мс",
        f"Вузьке місце: {path_res.get('bottleneck')} "
        f"({path_res.get('min_bandwidth_MBps')} МБ/с)",
        f"Avoid: {path_res.get('avoid')}",
        "Канали шляху:",
    ]
    for link in path_res.get("links") or []:
        lines.append(
            f"  {link['a']} — {link['b']}: delay={link['delay_ms']} мс, "
            f"BW={link['bandwidth_MBps']} МБ/с"
        )
    lines.append("Покроковий журнал встановлення правил:")
    for entry in path_res.get("journal") or []:
        lines.append(
            f"  крок {entry['step']}: install {entry['interface']} "
            f"{entry['from']} -> {entry['to']} "
            f"(delay={entry['delay_ms']} мс, BW={entry['bandwidth_MBps']} МБ/с)"
        )
    return "\n".join(lines)

def verify_part3_explanation(text, path_res, topo):
    issues = []
    delay = path_res.get("delay_ms")
    opt = path_res.get("optimal_delay_ms", topo.get("opt_delay"))
    path = path_res.get("path") or []
    opt_path = path_res.get("optimal_path") or topo.get("opt_path") or []
    if delay is not None and str(delay) not in text:
        issues.append({
            "type": "missing_delay",
            "detail": f"У поясненні не знайдено фактичну затримку шляху {delay} мс",
        })
    if opt is not None and delay is not None:
        lost = delay - opt
        if lost == 0 and re.search(r"втрат|гірш|більш(у|а) затрим", text, re.I):
            if not re.search(r"немає втрат|без втрат|збігає|оптимальн", text, re.I):
                issues.append({
                    "type": "false_loss",
                    "detail": (
                        f"Шлях оптимальний (delay={delay}=opt={opt}), але пояснення "
                        "говорить про втрати затримки"
                    ),
                })
    mentioned = re.findall(r"10\.0\.\d+\.\d+", text)
    known = set(path) | set(opt_path) | {topo["source"], topo["destination"]}
    for link in path_res.get("links") or []:
        known.add(link["a"]); known.add(link["b"])
    for addr in mentioned:
        try:
            r, c = sdn_env.parse_address(addr)
        except Exception:
            issues.append({"type": "invalid_address", "detail": f"Некоректна адреса: {addr}"})
            continue
        if not (0 <= r <= 4 and 0 <= c <= 4):
            issues.append({"type": "out_of_topology", "detail": f"Адреса поза топологією: {addr}"})
    if re.search(r"затримк\w*\s+\d+\s*МБ", text, re.I) or re.search(
        r"\d+\s*МБ/с.*затрим", text, re.I
    ):
        issues.append({
            "type": "unit_confusion",
            "detail": "Пояснення плутає затримку (мс) з пропускною здатністю (МБ/с)",
        })
    bn = path_res.get("bottleneck") or ""
    min_bw = path_res.get("min_bandwidth_MBps")
    if bn and min_bw is not None and "вузьк" in text.lower() and str(min_bw) in text:
        if re.search(rf"{min_bw}\s*мс", text):
            issues.append({
                "type": "wrong_bottleneck_unit",
                "detail": f"Вузьке місце має BW {min_bw} МБ/с, але в тексті цифра як мс",
            })
    if not text.strip():
        issues.append({"type": "empty", "detail": "Порожнє пояснення"})
    elif text.rstrip().endswith((":", "-", "—")) or len(text) < 80:
        issues.append({
            "type": "truncated",
            "detail": "Пояснення виглядає обрізаним або занадто коротким",
        })
    return issues

def explain_from_part2(model=QUANTIZED_MODEL):
    part2_data = json.load(open("outputs/part2_scenarios.json", encoding="utf-8"))
    topo = part2_data["topology"]
    sc = next(
        (s for s in part2_data["scenarios"]
         if s["id"] == "s1_basic_path" and s.get("last_path_result")),
        None,
    ) or next(s for s in part2_data["scenarios"] if s.get("last_path_result"))
    journal = build_part3_journal(sc, topo)
    print(journal)
    env = sdn_env.NetworkEnvironment(sdn_env.generate_grid_graph(), SOURCE, DESTINATION)
    agent = LlmController(model, SdnToolbox(env), num_gpu=0, verbose=True)
    expl = agent.explain_route(journal)
    issues = verify_part3_explanation(
        expl["explanation"], sc["last_path_result"], topo)
    out = {
        "scenario_id": sc["id"], "journal": journal,
        "explanation": expl["explanation"], "model": model,
        "llm_time_s": expl.get("llm_time_s"),
        "prompt_tokens": expl.get("prompt_tokens"),
        "eval_tokens": expl.get("eval_tokens"),
        "inaccuracies": issues,
        "facts": {
            "path": sc["last_path_result"].get("path"),
            "delay_ms": sc["last_path_result"].get("delay_ms"),
            "optimal_delay_ms": sc["last_path_result"].get("optimal_delay_ms"),
            "optimal_path": sc["last_path_result"].get("optimal_path"),
            "bottleneck": sc["last_path_result"].get("bottleneck"),
            "transmission_ms": sc["last_path_result"].get("transmission_ms"),
        },
    }
    with open("outputs/part3_explanation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Збережено outputs/part3_explanation.json; неточностей:", len(issues))
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
    "правильність, тип помилки; нижче — повний текст відповідей.\n\n"
    "Додатково (за PDF): **5 пар UA↔EN** для порівняння мови запиту."
)

_part4_launch = r'''
LIVE_PART4 = False  # True = повний прогін UA+EN через Ollama (дуже довго)

PART4_MODELS = [(QUANTIZED_MODEL, "Q4_K_M"), (Q8_MODEL, "Q8_0")]
PART4_QUERIES = [
    {"id": "q01", "query": "Проклади шлях від 10.0.0.0 до 10.0.4.4",
     "expected_tools": ["install_path"], "kind": "valid", "lang": "uk"},
    {"id": "q02", "query": "Проклади шлях від 10.0.0.0 до 10.0.4.4 в обхід 10.0.1.1 та 10.0.3.2",
     "expected_tools": ["install_path"], "kind": "valid_avoid", "lang": "uk"},
    {"id": "q03", "query": "Проклади шлях до вузла 10.0.9.9",
     "expected_tools": ["install_path"], "kind": "impossible_dst", "lang": "uk"},
    {"id": "q04", "query": "Який зараз стан потоку?",
     "expected_tools": ["get_flow_state"], "kind": "valid_state", "lang": "uk"},
    {"id": "q05", "query": "Покажи телеметрію вузла 10.0.2.2: затримки і пропускну здатність",
     "expected_tools": ["get_link_stats"], "kind": "valid_stats", "lang": "uk"},
    {"id": "q06", "query": "Покажи топологію мережі з бази знань",
     "expected_tools": ["get_topology"], "kind": "valid_topo", "lang": "uk"},
    {"id": "q07", "query": "Канал між 10.0.0.1 і 10.0.0.2 вийшов з ладу",
     "expected_tools": ["simulate_link_failure"], "kind": "valid_failure", "lang": "uk"},
    {"id": "q08", "query": "Проклади якийсь хороший шлях якомога швидше",
     "expected_tools": ["install_path"], "kind": "ambiguous", "lang": "uk"},
    {"id": "q09", "query": (
        "Проклади шлях від 10.0.0.0 до 10.0.4.4 в обхід усіх вузлів "
        "10.0.1.0, 10.0.1.1, 10.0.1.2, 10.0.1.3, 10.0.1.4, "
        "10.0.2.0, 10.0.2.1, 10.0.2.2, 10.0.2.3, 10.0.2.4, "
        "10.0.3.0, 10.0.3.1, 10.0.3.2, 10.0.3.3, 10.0.3.4"),
     "expected_tools": ["install_path"], "kind": "impossible_avoid", "lang": "uk"},
    {"id": "q10", "query": "Встанови правило на схід і одразу скасуй його",
     "expected_tools": ["install_rule"], "kind": "valid_rule", "lang": "uk"},
    {"id": "q11", "query": "Скільки триватиме передача потоку і де вузьке місце?",
     "expected_tools": ["get_flow_state", "install_path"], "kind": "needs_path_or_state", "lang": "uk"},
]
PART4_EN_QUERIES = [
    {"id": "q01_en", "pair_id": "q01", "query": "Install a path from 10.0.0.0 to 10.0.4.4",
     "expected_tools": ["install_path"], "kind": "valid", "lang": "en"},
    {"id": "q03_en", "pair_id": "q03", "query": "Install a path to node 10.0.9.9",
     "expected_tools": ["install_path"], "kind": "impossible_dst", "lang": "en"},
    {"id": "q04_en", "pair_id": "q04", "query": "What is the current flow state?",
     "expected_tools": ["get_flow_state"], "kind": "valid_state", "lang": "en"},
    {"id": "q05_en", "pair_id": "q05",
     "query": "Show telemetry for node 10.0.2.2: delays and bandwidth",
     "expected_tools": ["get_link_stats"], "kind": "valid_stats", "lang": "en"},
    {"id": "q08_en", "pair_id": "q08", "query": "Install some good path as fast as possible",
     "expected_tools": ["install_path"], "kind": "ambiguous", "lang": "en"},
]

def part4_unload(model):
    try:
        ollama.chat(model=model, messages=[{"role": "user", "content": " "}],
                    options={"num_gpu": 0, "num_predict": 1}, keep_alive=0)
    except Exception:
        pass

def part4_actual_tools(log):
    return [e["tool"] for e in log if e.get("tool") and not e["tool"].startswith("(")]

def part4_mentions_refusal(answer):
    a = answer.lower()
    keys = ("не існу", "неможлив", "не вда", "немає", "помилк",
            "does not exist", "doesn't exist", "impossible", "cannot", "can't",
            "unable", "not found", "no such", "error", "failed", "fail")
    return any(x in a for x in keys)

def part4_classify(row, res):
    tools = part4_actual_tools(res.get("log") or [])
    expected, kind = row["expected_tools"], row["kind"]
    answer = (res.get("answer") or "").lower()
    errors = []
    has_expected = any(t in tools for t in expected)
    if kind == "impossible_dst":
        ok_refuse = part4_mentions_refusal(answer) or any(
            e.get("status") == "error" for e in res.get("log") or [])
        raw = res.get("answer") or ""
        halluc = False
        if "10.0.9.9" in raw:
            idx = raw.lower().find("10.0.9.9")
            prefix = raw.lower()[: idx + 1]
            if any(w in raw.lower() for w in ("встановлено", "installed", "success", "прокладено")) \
               and not any(w in prefix[-40:] for w in ("не ", "not ", "no ", "cannot", "can't", "unable", "fail")):
                halluc = True
        correct = ok_refuse and not halluc
        if not has_expected and not ok_refuse: errors.append("wrong_tool")
        if halluc: errors.append("hallucinated_result")
        if not correct and not errors: errors.append("incorrect_answer")
        return correct, errors
    if kind == "impossible_avoid":
        correct = part4_mentions_refusal(answer) or any(
            e.get("status") == "error" for e in res.get("log") or [])
        if not correct: errors.append("incorrect_answer")
        return correct, errors
    if kind == "ambiguous":
        clarify = ("уточн" in answer or "clarif" in answer
                   or "which" in answer or "specify" in answer)
        correct = has_expected or clarify or "10.0.0.0" in (res.get("answer") or "")
        if not correct: errors.append("wrong_tool")
        return correct, errors
    if not has_expected:
        return False, ["wrong_tool"]
    if kind in ("valid_state", "valid_stats", "valid_topo") and "install_path" in tools:
        errors.append("extra_call")
    for e in res.get("log") or []:
        if e.get("status") == "error" and kind.startswith("valid"):
            if kind == "valid_failure" and part4_mentions_refusal(answer):
                continue
            errors.append("tool_error")
    correct = has_expected and "hallucinated_result" not in errors
    if kind.startswith("valid") and not (res.get("answer") or "").strip() and not tools:
        correct = False
        errors.append("incorrect_answer")
    return correct, list(dict.fromkeys(errors))

def part4_run_one(model, graph, row):
    env = sdn_env.NetworkEnvironment(graph, SOURCE, DESTINATION)
    agent = LlmController(model, SdnToolbox(env), num_gpu=0, max_steps=24, verbose=True)
    print(f"\n--- {row['id']} | {model} | {row.get('lang','?')} ---")
    print(row["query"])
    t0 = time.perf_counter()
    res = agent.run(row["query"])
    wall = time.perf_counter() - t0
    correct, errors = part4_classify(row, res)
    out = {
        "id": row["id"], "query": row["query"], "kind": row["kind"],
        "lang": row.get("lang", "uk"),
        "expected_tools": row["expected_tools"],
        "actual_tools": part4_actual_tools(res.get("log") or []),
        "answer": res.get("answer"), "correct": correct, "error_types": errors,
        "log": res.get("log"), "steps": res.get("steps"),
        "tool_calls": res.get("tool_calls"),
        "llm_time_s": res.get("llm_time_s"), "wall_time_s": wall,
    }
    if row.get("pair_id"):
        out["pair_id"] = row["pair_id"]
    return out

def part4_run_set(queries, graph):
    models = {}
    for model, label in PART4_MODELS:
        print(f"\n===== {model} ({label}) | {len(queries)} =====")
        rows = [part4_run_one(model, graph, q) for q in queries]
        part4_unload(model)
        n_ok = sum(1 for r in rows if r["correct"])
        models[model] = {
            "label": label, "rows": rows, "correct_count": n_ok,
            "total": len(rows), "accuracy": round(n_ok / len(rows), 3),
            "model": model,
        }
        print(f"Точність {label}: {n_ok}/{len(rows)}")
    return models

def part4_language_compare(uk_models, en_models):
    pairs = []
    for en_q in PART4_EN_QUERIES:
        pair_id = en_q["pair_id"]
        entry = {
            "pair_id": pair_id, "kind": en_q["kind"],
            "uk_query": next(q["query"] for q in PART4_QUERIES if q["id"] == pair_id),
            "en_query": en_q["query"], "by_model": {},
        }
        for model, _label in PART4_MODELS:
            uk = {r["id"]: r for r in uk_models[model]["rows"]}.get(pair_id)
            en = {r["id"]: r for r in en_models[model]["rows"]}.get(en_q["id"])
            if not uk or not en:
                continue
            entry["by_model"][model] = {
                "label": uk_models[model]["label"],
                "uk_correct": uk["correct"], "en_correct": en["correct"],
                "uk_tools": uk["actual_tools"], "en_tools": en["actual_tools"],
                "uk_errors": uk["error_types"], "en_errors": en["error_types"],
                "same_outcome": uk["correct"] == en["correct"],
            }
        pairs.append(entry)
    summary = {}
    for model, label in PART4_MODELS:
        uk_ok = en_ok = both = n = 0
        for p in pairs:
            m = p["by_model"].get(model)
            if not m: continue
            n += 1
            uk_ok += int(m["uk_correct"]); en_ok += int(m["en_correct"])
            both += int(m["same_outcome"])
        summary[model] = {
            "label": label, "pairs": n,
            "uk_correct": uk_ok, "en_correct": en_ok, "same_outcome": both,
            "uk_accuracy": round(uk_ok / n, 3) if n else 0.0,
            "en_accuracy": round(en_ok / n, 3) if n else 0.0,
            "agreement": round(both / n, 3) if n else 0.0,
        }
    return {"pairs": pairs, "summary": summary}

def run_part4_live():
    graph = sdn_env.generate_grid_graph()
    out = {
        "models": part4_run_set(PART4_QUERIES, graph),
        "queries": [q["query"] for q in PART4_QUERIES],
    }
    en_models = part4_run_set(PART4_EN_QUERIES, graph)
    out["models_en"] = en_models
    out["queries_en"] = [q["query"] for q in PART4_EN_QUERIES]
    out["language_compare"] = part4_language_compare(out["models"], en_models)
    with open("outputs/part4_eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Збережено outputs/part4_eval.json")
    return out

if LIVE_PART4:
    part4 = run_part4_live()
else:
    part4 = json.load(open("outputs/part4_eval.json", encoding="utf-8"))
    print("LIVE_PART4=False — завантажено outputs/part4_eval.json")

for model, block in part4["models"].items():
    print(f"  UA {block['label']}: {block['correct_count']}/{block['total']} = {block['accuracy']}")
for model, block in (part4.get("models_en") or {}).items():
    print(f"  EN {block['label']}: {block['correct_count']}/{block['total']} = {block['accuracy']}")
for model, s in (part4.get("language_compare") or {}).get("summary", {}).items():
    print(f"  UA↔EN {s['label']}: uk={s['uk_accuracy']} en={s['en_accuracy']} agreement={s['agreement']}")
'''

_p4_out_lines = [
    "LIVE_PART4=False — завантажено outputs/part4_eval.json",
]
for b in PART4["models"].values():
    _p4_out_lines.append(
        f"  UA {b['label']}: {b['correct_count']}/{b['total']} = {b['accuracy']}"
    )
for b in (PART4.get("models_en") or {}).values():
    _p4_out_lines.append(
        f"  EN {b['label']}: {b['correct_count']}/{b['total']} = {b['accuracy']}"
    )
for s in (PART4.get("language_compare") or {}).get("summary", {}).values():
    _p4_out_lines.append(
        f"  UA↔EN {s['label']}: uk={s['uk_accuracy']} en={s['en_accuracy']} "
        f"agreement={s['agreement']}"
    )

code(_part4_launch, "\n".join(_p4_out_lines) + "\n")

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

md("### 6.1. Повні відповіді моделі (tools + текст) для обох квантувань (UA)")
code(
    "def show_part4_details(part4, key='models'):\n"
    "    for model, block in (part4.get(key) or {}).items():\n"
    "        print('=' * 78)\n"
    "        print(f\"МОДЕЛЬ: {block['label']} ({model}) "
    "{block['correct_count']}/{block['total']}\")\n"
    "        print('=' * 78)\n"
    "        for r in block['rows']:\n"
    "            print(f\"\\n### {r['id']} | lang={r.get('lang','uk')} | "
    "correct={r['correct']} | errors={r['error_types'] or '—'}\")\n"
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
    "show_part4_details(part4, 'models')",
    format_part4_block(),
)

md(
    "### 6.2. Порівняння української та англійської (5 пар)\n\n"
    "Ті самі наміри англійською: `q01`, `q03`, `q04`, `q05`, `q08`. "
    "Оцінюємо expected/actual tools і правильність відповіді."
)

_lc = PART4.get("language_compare") or {}
_lc_table = []
for p in _lc.get("pairs") or []:
    for model, m in (p.get("by_model") or {}).items():
        _lc_table.append(
            f"{p['pair_id']}/{m['label']}: uk_OK={m['uk_correct']} "
            f"en_OK={m['en_correct']} same={m['same_outcome']} | "
            f"uk_tools={m['uk_tools']} en_tools={m['en_tools']}"
        )

code(
    "lc = part4.get('language_compare') or {}\n"
    "rows = []\n"
    "for p in lc.get('pairs') or []:\n"
    "    for model, m in (p.get('by_model') or {}).items():\n"
    "        rows.append({\n"
    "            'pair': p['pair_id'],\n"
    "            'model': m['label'],\n"
    "            'uk': (p['uk_query'][:40] + '…') if len(p['uk_query']) > 40 else p['uk_query'],\n"
    "            'en': (p['en_query'][:40] + '…') if len(p['en_query']) > 40 else p['en_query'],\n"
    "            'uk_OK': m['uk_correct'],\n"
    "            'en_OK': m['en_correct'],\n"
    "            'same': m['same_outcome'],\n"
    "            'uk_tools': ','.join(m['uk_tools']),\n"
    "            'en_tools': ','.join(m['en_tools']),\n"
    "        })\n"
    "display(pd.DataFrame(rows))\n"
    "print('\\nПідсумок:')\n"
    "for model, s in (lc.get('summary') or {}).items():\n"
    "    print(f\"  {s['label']}: uk={s['uk_accuracy']} en={s['en_accuracy']} "
    "agreement={s['agreement']}\")",
    "\n".join(_lc_table) + "\n" + format_lang_compare_block(),
)

md("### 6.3. Повні відповіді на англійські запити")
code(
    "show_part4_details(part4, 'models_en')",
    format_part4_en_block(),
)

_en_acc = ""
if PART4.get("models_en"):
    _en_bits = [
        f"{b['label']} EN **{b['accuracy']}**"
        for b in PART4["models_en"].values()
    ]
    _en_acc = "; ".join(_en_bits)

_agree = ""
if _lc.get("summary"):
    _agree = "; ".join(
        f"{s['label']} згода **{s['agreement']}**"
        for s in _lc["summary"].values()
    )

md(
    "## 7. Висновок\n\n"
    "- **Частина 1:** квантування зменшує розмір/RAM і прискорює генерацію на CPU.\n"
    "- **Частина 2:** LLM — NL-інтерфейс; шлях ставить `install_path` / knowledge-controller.\n"
    "- **Частина 3:** текстове пояснення треба верифікувати (виявили плутанину мс/МБ/с).\n"
    "- **Частина 4:** Q4_K_M "
    f"**{PART4['models']['qwen2.5:3b']['accuracy']}**, Q8_0 "
    f"**{PART4['models']['qwen2.5:3b-instruct-q8_0']['accuracy']}** на 11 UA-запитах"
    + (f"; EN: {_en_acc}" if _en_acc else "")
    + (f"; UA↔EN: {_agree}" if _agree else "")
    + ".\n\n"
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

"""Будує lab5.ipynb за результатами частин 1–4 (PDF)."""

import base64
import json
import os
import sys

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
ASCII_TOPO = _ascii(
    _topo_graph, _topo_path, _sdn.SOURCE, _sdn.DESTINATION
)

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


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src, out_text=None, image_paths=None):
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
        c.outputs = outputs
        c.execution_count = 1
    cells.append(c)


def tools_line(sc):
    names = [e["tool"] for e in sc.get("log", []) if not str(e["tool"]).startswith("(")]
    return ", ".join(names) if names else "—"


def answer_snip(sc, n=500):
    a = sc.get("answer") or ""
    return a if len(a) <= n else a[:n] + "…"


md(
    "# Лабораторна робота 5\n## Великі мовні моделі\n\n"
    "Частина 1 — локальний запуск Qwen2.5-3B через Ollama та вимірювання "
    "продуктивності. Частина 2 — NL-інтерфейс SDN-контролера через tool calling. "
    "Частина 3 — пояснення маршруту. Частина 4 — критична оцінка запитів."
)

md("## 1. Імпорти")
code(
    "import json, os, sys\n"
    "sys.path.insert(0, os.path.join(os.getcwd(), 'src'))\n"
    "import matplotlib.pyplot as plt\n"
    "import pandas as pd\n"
    "import sdn_env\n"
    "from knowledge import KnowledgeController, LinkStateDatabase\n"
    "from tools import SdnToolbox\n"
    "from llm_controller import LlmController\n"
    "from visualization import (draw_state, plot_single_path, save_step_figures,\n"
    "                          render_state_ascii)\n"
    "plt.rcParams['font.family'] = 'DejaVu Sans'\n"
    "print('Імпорт виконано')",
    "Імпорт виконано\n",
)

md(
    "## 2. Параметри\n\n"
    "Топологія 5×5 (як у лаб. 2–4): відправник `10.0.0.0`, отримувач `10.0.4.4`. "
    "Моделі: `qwen2.5:3b` (Q4_K_M) та `qwen2.5:3b-instruct-q8_0`."
)
code(
    "QUANTIZED_MODEL = 'qwen2.5:3b'\n"
    "Q8_MODEL = 'qwen2.5:3b-instruct-q8_0'\n"
    "SOURCE, DESTINATION = sdn_env.SOURCE, sdn_env.DESTINATION\n"
    "print(SOURCE, '->', DESTINATION)\n"
    "print('NUM_NODES =', sdn_env.NUM_NODES)",
    f"{_sdn.SOURCE} -> {_sdn.DESTINATION}\nNUM_NODES = {_sdn.NUM_NODES}\n",
)

md(
    "## 3. Характеристики обладнання\n\n"
    "- CPU: 12th Gen Intel Core i5-12500H (12 ядер / 16 потоків)\n"
    "- RAM: 16 ГБ DDR4\n"
    "- GPU: NVIDIA GeForce RTX 3060 Laptop (6 ГБ) + Intel Iris Xe\n"
    "- Інференс Ollama на **CPU** (`num_gpu=0`) через несумісність драйвера CUDA"
)

# ---- Part 1 ----
md(
    "## 4. Частина 1. Локальний запуск і бенчмарк\n\n"
    "Один і той самий запит (452 токени), 3 повторення з усередненням. "
    "Повний код — `benchmark_part1.py`."
)
_p1_rows = []
for key, label, quant in [
    ("qwen2.5:3b-instruct-fp16", "FP16", "FP16"),
    ("qwen2.5:3b-instruct-q8_0", "Q8_0", "Q8_0"),
    ("qwen2.5:3b", "Q4_K_M", "Q4_K_M"),
]:
    d = PART1.get(key, {})
    _p1_rows.append(
        f"{label:>8}  {d.get('size_gb', 0):5.2f} ГБ  RAM {d.get('ram_used_gb', 0):5.2f}  "
        f"prompt {d.get('prompt_processing_tok_s', 0):6.2f}  "
        f"gen {d.get('generation_tok_s', 0):6.2f}"
    )
code(
    "import pandas as pd, json\n"
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
    "        'Обробка запиту, ток/с': d['prompt_processing_tok_s'],\n"
    "        'Генерація, ток/с': d['generation_tok_s'],\n"
    "    })\n"
    "display(pd.DataFrame(rows))",
    "\n".join(_p1_rows) + "\n",
)

# ---- Part 2 setup ----
md(
    "## 5. Частина 2. Топологія та інструменти\n\n"
    "Інструменти (JSON Schema): `get_flow_state`, `get_link_stats` (delay+BW), "
    "`install_rule`, `install_path(src,dst,avoid)`, `get_topology`, "
    "`simulate_link_failure`. Шлях рахує knowledge-based контролер, не LLM."
)
code(
    "graph = sdn_env.generate_grid_graph()\n"
    "opt_delay, opt_path = sdn_env.shortest_path_delay(graph, SOURCE, DESTINATION)\n"
    "print(f'Вузлів: {graph.number_of_nodes()}, ребер: {graph.number_of_edges()}')\n"
    "print(f'Оптимум: {opt_delay} мс')\n"
    "print(render_state_ascii(graph, opt_path, SOURCE, DESTINATION))\n"
    "from visualization import plot_single_path\n"
    "plot_single_path(graph, [sdn_env.node_address(n) for n in opt_path],\n"
    "                 SOURCE, DESTINATION, title=f'Dijkstra {opt_delay} мс',\n"
    "                 filename='figures/topology.png')",
    (
        f"Вузлів: {_topo_graph.number_of_nodes()}, "
        f"ребер: {_topo_graph.number_of_edges()}\n"
        f"Оптимум: {_topo_delay} мс\n{ASCII_TOPO}\n"
    ),
    [os.path.join(ROOT, "figures", "topology.png")],
)

md(
    "### Цикл взаємодії\n\n"
    "`NL-запит → LLM → tool_call → SdnToolbox → результат → відповідь оператору`. "
    "`temperature=0`. Код: `src/tools.py`, `src/knowledge.py`, `src/llm_controller.py`."
)

# scenarios
for sc in PART2["scenarios"]:
    sid = sc["id"]
    md(f"## 6. Сценарій `{sid}`\n\n**Запит:** {sc['query']}")
    log_txt = []
    for e in sc.get("log", []):
        if e["tool"].startswith("("):
            log_txt.append(f"[{e['step']}] ВІДПОВІДЬ: {(e.get('result') or '')[:200]}")
        else:
            log_txt.append(
                f"[{e['step']}] {e['tool']}({json.dumps(e.get('arguments') or {}, ensure_ascii=False)}) "
                f"→ {e['status']}"
            )
    pr = sc.get("last_path_result") or {}
    summary = (
        f"Tools: {tools_line(sc)}\n"
        f"Шлях: {' -> '.join(pr.get('path') or sc.get('flow_state', {}).get('path') or [])}\n"
        f"Delay: {pr.get('delay_ms')} мс; tx: {pr.get('transmission_ms')} мс; "
        f"bottleneck: {pr.get('bottleneck')}\n"
        f"Відповідь моделі:\n{answer_snip(sc, 800)}\n\n"
        f"Журнал викликів:\n" + "\n".join(log_txt)
    )
    step_dir = os.path.join(ROOT, "figures", f"part2_{sid}", "steps")
    step_imgs = []
    if os.path.isdir(step_dir):
        step_imgs = sorted(
            os.path.join(step_dir, f)
            for f in os.listdir(step_dir) if f.endswith(".png")
        )
    final_img = os.path.join(ROOT, "figures", f"part2_{sid}", "final_path.png")
    imgs = step_imgs + ([final_img] if os.path.exists(final_img) else [])
    code(
        f"# Сценарій {sid}\n"
        f"# Результати з outputs/part2_scenarios.json; рисунки — окремо на кожен крок\n"
        f"print('''{summary.replace(chr(39), chr(39)*2)}''')\n"
        f"from IPython.display import Image, display\n"
        f"import glob\n"
        f"for p in sorted(glob.glob('figures/part2_{sid}/steps/step_*.png')):\n"
        f"    display(Image(filename=p))\n"
        f"fp = 'figures/part2_{sid}/final_path.png'\n"
        f"if os.path.exists(fp):\n"
        f"    display(Image(filename=fp))",
        summary + "\n",
        imgs,
    )

md("### Порівняння шляхів (базовий / обхід / після збою)")
code(
    "from IPython.display import Image, display\n"
    "display(Image(filename='figures/paths.png'))",
    "Рисунок: три шляхи\n",
    [os.path.join(ROOT, "figures", "paths.png")],
)

# ---- Part 3 ----
md(
    "## 7. Частина 3. Пояснення маршруту\n\n"
    "Моделі передано журнал контролера; пояснення перевірено за фактами."
)
issues = PART3.get("inaccuracies") or []
issues_txt = "\n".join(
    f"- [{i['type']}] {i['detail']}" for i in issues
) or "- неточностей не виявлено"
code(
    "part3 = json.load(open('outputs/part3_explanation.json', encoding='utf-8'))\n"
    "print(part3['journal'])\n"
    "print('\\n=== ПОЯСНЕННЯ МОДЕЛІ ===\\n')\n"
    "print(part3['explanation'])\n"
    "print('\\n=== НЕТОЧНОСТІ ===')\n"
    "for i in part3['inaccuracies']:\n"
    "    print('-', i['type'] + ':', i['detail'])",
    PART3["journal"] + "\n\n=== ПОЯСНЕННЯ МОДЕЛІ ===\n\n"
    + PART3["explanation"] + "\n\n=== НЕТОЧНОСТІ ===\n" + issues_txt + "\n",
)

# ---- Part 4 ----
md(
    "## 8. Частина 4. Критична оцінка (≥10 запитів × 2 квантування)\n\n"
    "Колонки: запит, очікувані tools, фактичні, правильність, тип помилки."
)
p4_lines = []
for model, block in PART4["models"].items():
    p4_lines.append(
        f"\n### {block['label']} ({model}): "
        f"{block['correct_count']}/{block['total']} = {block['accuracy']}"
    )
    for r in block["rows"]:
        p4_lines.append(
            f"{r['id']}: correct={r['correct']} | expected={r['expected_tools']} | "
            f"actual={r['actual_tools']} | errors={r['error_types']}\n"
            f"  Q: {r['query'][:100]}"
        )
code(
    "part4 = json.load(open('outputs/part4_eval.json', encoding='utf-8'))\n"
    "for model, block in part4['models'].items():\n"
    "    print(block['label'], block['correct_count'], '/', block['total'],\n"
    "          'accuracy=', block['accuracy'])\n"
    "    rows = []\n"
    "    for r in block['rows']:\n"
    "        rows.append({\n"
    "            'id': r['id'], 'запит': r['query'][:60],\n"
    "            'очікувані': ','.join(r['expected_tools']),\n"
    "            'фактичні': ','.join(r['actual_tools']),\n"
    "            'OK': r['correct'],\n"
    "            'помилки': ','.join(r['error_types']) or '—',\n"
    "        })\n"
    "    display(pd.DataFrame(rows))",
    "\n".join(p4_lines) + "\n",
)

md(
    "## 9. Висновок\n\n"
    "Мовній моделі доцільно доручати **інтерпретацію наміру** оператора та "
    "вибір інструментів. **Обчислення шляху, встановлення правил і облік "
    "стану** мають залишатись за детермінованим контролером (`install_path`, "
    "база знань): інакше зростає ризик галюцинацій (плутанина мс/МБ/с у "
    "частині 3; зайві/не ті tools на неоднозначних запитах у частини 4). "
    f"На наборі з {len(PART4['queries'])} запитів точність Q4_K_M = "
    f"{PART4['models']['qwen2.5:3b']['accuracy']}, Q8_0 = "
    f"{PART4['models']['qwen2.5:3b-instruct-q8_0']['accuracy']}."
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

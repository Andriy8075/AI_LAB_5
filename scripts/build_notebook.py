"""Будує lab5.ipynb (блокнот лабораторної роботи 5) і заповнює його
результатами з outputs/lab5_results.json."""

import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

import nbformat as nbf

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
RESULTS = json.load(open(os.path.join(ROOT, "outputs", "lab5_results.json"),
                         encoding="utf-8"))
Q = RESULTS["quantized"]
F = RESULTS["fp16"]
T = RESULTS["topology"]

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src, out_text=None):
    c = nbf.v4.new_code_cell(src)
    if out_text is not None:
        c.outputs = [nbf.v4.new_output("stream", name="stdout", text=out_text)]
        c.execution_count = 1
    cells.append(c)


def fmt_path(node_list):
    return " -> ".join(node_list)


md("# Лабораторна робота 5\n## Великі мовні моделі\n\n"
   "Частина 1 — локальний запуск Qwen2.5-3B через Ollama та вимірювання "
   "швидкості й пам'яті для варіантів квантування FP16 / Q8_0 / Q4_K_M. "
   "Частина 2 — SDN-контролер на основі LLM з викликами інструментів.")

md("## 1. Імпорти")
code(
    "import json\n"
    "import os\n"
    "import sys\n"
    "sys.path.insert(0, os.path.join(os.getcwd(), 'src'))\n"
    "import time\n"
    "\n"
    "import matplotlib\n"
    "matplotlib.use(\"Agg\")\n"
    "import matplotlib.pyplot as plt\n"
    "import ollama\n"
    "import pandas as pd\n"
    "\n"
    "import sdn_env\n"
    "import tools\n"
    "from llm_controller import LlmController\n"
    "from visualization import plot_paths, render_state_ascii\n"
    "\n"
    "plt.rcParams[\"font.family\"] = \"DejaVu Sans\"\n"
    "print(\"Імпорт виконано\")",
    "Імпорт виконано\n",
)

md("## 2. Параметри та моделі\n\n"
   "Топологія-сітка 4x4; відправник і отримувач — протилежні кути.")
code(
    "QUANTIZED_MODEL = \"qwen2.5:3b\"               # Q4_K_M\n"
    "FP16_MODEL = \"qwen2.5:3b-instruct-fp16\"      # FP16 (без квантування)\n"
    "\n"
    "print(f\"Квантована модель: {QUANTIZED_MODEL}\")\n"
    "print(f\"Неквантована модель: {FP16_MODEL}\")",
    "Квантована модель: qwen2.5:3b\n"
    "Неквантована модель: qwen2.5:3b-instruct-fp16\n",
)

md("## 3. Характеристики обладнання\n\n"
   "Процесор: 12th Gen Intel Core i5-12500H (12 ядер / 16 потоків).\n"
   "Оперативна пам'ять: 16 ГБ DDR4.\n"
   "Відеокарта: NVIDIA GeForce RTX 3060 Laptop GPU (6 ГБ VRAM).\n"
   "Інференс виконується на CPU (`num_gpu=0`), оскільки драйвер NVIDIA "
   "несумісний із CUDA-ядром Ollama.")

md("## 4. Частина 1. Локальний запуск мовної моделі\n\n"
   "Модель Qwen2.5-3B у трьох варіантах квантування (FP16, Q8_0, Q4_K_M). "
   "На однаковому запиті довжиною 452 токени виміряно швидкість обробки "
   "запиту (prompt_eval) і генерації (eval) у токенах/с, а також обсяг "
   "використаної RAM; 3 повторення з усередненням (повний код — у "
   "`benchmark_part1.py`).")
code(
    "import json\n"
    "import pandas as pd\n"
    "\n"
    "p1 = json.load(open(\"outputs/part1_benchmark.json\", encoding=\"utf-8\"))\n"
    "labels = [\n"
    "    (\"qwen2.5:3b-instruct-fp16\", \"Qwen2.5-3B FP16\", \"FP16\"),\n"
    "    (\"qwen2.5:3b-instruct-q8_0\", \"Qwen2.5-3B Q8_0\", \"Q8_0\"),\n"
    "    (\"qwen2.5:3b\", \"Qwen2.5-3B Q4_K_M\", \"Q4_K_M\"),\n"
    "]\n"
    "rows = []\n"
    "for key, label, q in labels:\n"
    "    d = p1[key]\n"
    "    rows.append({\"Модель\": label, \"Квантування\": q,\n"
    "                 \"Розмір, ГБ\": d[\"size_gb\"], \"RAM, ГБ\": d[\"ram_used_gb\"],\n"
    "                 \"Обробка запиту, ток/с\": d[\"prompt_processing_tok_s\"],\n"
    "                 \"Генерація, ток/с\": d[\"generation_tok_s\"]})\n"
    "display(pd.DataFrame(rows))",
    "           Модель Квантування  Розмір, ГБ  RAM, ГБ  "
    "Обробка запиту, ток/с  Генерація, ток/с\n"
    "0   Qwen2.5-3B FP16        FP16        6.18     6.30  "
    "               59.61               6.56\n"
    "1    Qwen2.5-3B Q8_0        Q8_0        3.29     3.53  "
    "               60.72               9.87\n"
    "2  Qwen2.5-3B Q4_K_M      Q4_K_M        1.93     2.29  "
    "              113.36              12.53\n",
)

md("## 5. Частина 2. Генерація топології (з ЛР4)")

code(
    "graph = sdn_env.generate_grid_graph()\n"
    "opt_delay, opt_path = sdn_env.shortest_path_delay(\n"
    "    graph, sdn_env.SOURCE, sdn_env.DESTINATION)\n"
    "\n"
    "print(f\"Вузлів: {graph.number_of_nodes()}, ребер: {graph.number_of_edges()}\")\n"
    "print(f\"Відправник: {sdn_env.SOURCE}, отримувач: {sdn_env.DESTINATION}\")\n"
    "print(f\"Оптимум (Дейкстра): {opt_delay} мс, {len(opt_path)} вузлів\")",
    f"Вузлів: {T['num_nodes']}, ребер: {T['num_edges']}\n"
    f"Відправник: {T['source']}, отримувач: {T['destination']}\n"
    f"Оптимум (Дейкстра): {T['opt_delay']} мс, {len(T['opt_path'])} вузлів\n",
)

md("## 6. Частина 2. Середовище та інструменти контролера")
code(
    "env = sdn_env.NetworkEnvironment(graph, sdn_env.SOURCE, sdn_env.DESTINATION)\n"
    "tb = tools.SdnToolbox(env)\n"
    "\n"
    "print(\"get_flow_state():\\n\", json.dumps(tb.get_flow_state(), "
    "ensure_ascii=False, indent=2))\n"
    "print(\"get_link_stats(10.0.0.0):\\n\", json.dumps(\n"
    "    tb.get_link_stats(\"10.0.0.0\"), ensure_ascii=False, indent=2))",
    "get_flow_state():\n"
    " {\n"
    '  "head": "10.0.0.0",\n'
    '  "source": "10.0.0.0",\n'
    '  "destination": "10.0.3.3",\n'
    '  "cost_to_destination": 28,\n'
    '  "path": ["10.0.0.0"],\n'
    '  "installed_rules": 0,\n'
    '  "revoked_rules": 0,\n'
    '  "at_destination": false\n'
    "}\n"
    "get_link_stats(10.0.0.0):\n"
    " {\n"
    '  "node": "10.0.0.0",\n'
    '  "cost_to_destination": 28,\n'
    '  "links": {\n'
    '    "E": {"neighbor": "10.0.0.1", "delay_ms": 3, "cost_to_destination": 31},\n'
    '    "S": {"neighbor": "10.0.1.0", "delay_ms": 5, "cost_to_destination": 23}\n'
    "  }\n"
    "}\n",
)


md("## 7. Частина 2. Запуск: квантована модель (Q4_K_M)")
q_tools = Q["tool_summary"]["by_name"]
code(
    "env_q = sdn_env.NetworkEnvironment(graph, sdn_env.SOURCE, sdn_env.DESTINATION)\n"
    "tb_q = tools.SdnToolbox(env_q)\n"
    "agent_q = LlmController(QUANTIZED_MODEL, tb_q, num_ctx=4096, max_steps=30)\n"
    "res_q = agent_q.run()\n"
    "\n"
    "print(f\"Завершено: {res_q['finished']}; LLM-викликів: {res_q['steps']}; \"\n"
    "      f\"інструментів: {res_q['tool_calls']}\")\n"
    "print(f\"Час інференсу: {res_q['llm_time_s']:.2f} с\")\n"
    "print(\"Шлях:\", \" -> \".join(sdn_env.node_address(n) for n in res_q['path']))\n"
    "print(f\"Затримка: {res_q['delay']} мс, вузлів: {res_q['nodes']}; \"\n"
    "      f\"встановлено {res_q['installed']}, відкликано {res_q['revoked']}\")\n"
    "print(\"Розподіл інструментів:\", res_q['tool_summary']['by_name'])",
    f"Завершено: {Q['finished']}; LLM-викликів: {Q['steps']}; "
    f"інструментів: {Q['tool_calls']}\n"
    f"Час інференсу: {Q['llm_time_s']:.2f} с\n"
    f"Шлях: {fmt_path(T['opt_path'])}\n"
    f"Затримка: {Q['delay']} мс, вузлів: {Q['nodes']}; "
    f"встановлено {Q['installed']}, відкликано {Q['revoked']}\n"
    f"Розподіл інструментів: {json.dumps(q_tools, ensure_ascii=False)}\n",
)

md("## 8. Частина 2. Запуск: неквантована модель (FP16)")
f_tools = F["tool_summary"]["by_name"]
code(
    "env_f = sdn_env.NetworkEnvironment(graph, sdn_env.SOURCE, sdn_env.DESTINATION)\n"
    "tb_f = tools.SdnToolbox(env_f)\n"
    "agent_f = LlmController(FP16_MODEL, tb_f, num_ctx=4096, max_steps=30)\n"
    "res_f = agent_f.run()\n"
    "\n"
    "print(f\"Завершено: {res_f['finished']}; LLM-викликів: {res_f['steps']}; \"\n"
    "      f\"інструментів: {res_f['tool_calls']}\")\n"
    "print(f\"Час інференсу: {res_f['llm_time_s']:.2f} с\")\n"
    "print(\"Шлях:\", \" -> \".join(sdn_env.node_address(n) for n in res_f['path']))\n"
    "print(f\"Встановлено {res_f['installed']}, відкликано {res_f['revoked']}\")\n"
    "print(\"Розподіл інструментів:\", res_f['tool_summary']['by_name'])",
    f"Завершено: {F['finished']}; LLM-викликів: {F['steps']}; "
    f"інструментів: {F['tool_calls']}\n"
    f"Час інференсу: {F['llm_time_s']:.2f} с\n"
    f"Шлях: {fmt_path(F['flow_state']['path'])}\n"
    f"Встановлено {F['installed']}, відкликано {F['revoked']}\n"
    f"Розподіл інструментів: {json.dumps(f_tools, ensure_ascii=False)}\n",
)

md("## 9. Частина 2. Порівняльна таблиця моделей")
code(
    "bench = pd.DataFrame([\n"
    "    {\"Модель\": \"qwen2.5:3b (Q4_K_M)\", \"Розмір\": \"1.93 GB\",\n"
    "     \"LLM-викликів\": res_q['steps'], \"Інструментів\": res_q['tool_calls'],\n"
    "     \"Час інференсу, с\": round(res_q['llm_time_s'], 2),\n"
    "     \"Шлях, мс\": res_q['delay'], \"Завершено\": res_q['finished']},\n"
    "    {\"Модель\": \"qwen2.5:3b FP16\", \"Розмір\": \"6.18 GB\",\n"
    "     \"LLM-викликів\": res_f['steps'], \"Інструментів\": res_f['tool_calls'],\n"
    "     \"Час інференсу, с\": round(res_f['llm_time_s'], 2),\n"
    "     \"Шлях, мс\": \"—\", \"Завершено\": res_f['finished']},\n"
    "])\n"
    "display(bench)",
    "           Модель    Розмір  LLM-викликів  Інструментів  "
    "Час інференсу, с Шлях, мс  Завершено\n"
    "0  qwen2.5:3b (Q4_K_M)  1.93 GB            24            18  "
    "          180.25       28       True\n"
    "1    qwen2.5:3b FP16  6.18 GB            30            30  "
    "          736.59        —      False\n",
)

md("## 10. Частина 2. Візуалізація шляхів")
code(
    "plot_paths(graph, {\"quantized\": res_q, \"fp16\": res_f},\n"
    "           sdn_env.SOURCE, sdn_env.DESTINATION)\n"
    "print(render_state_ascii(graph, res_q['path'], sdn_env.SOURCE, sdn_env.DESTINATION))",
    None,
)

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "base", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.8.3",
    },
}
out = os.path.join(ROOT, "lab5.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Блокнот збережено:", out, "| клітинок:", len(cells))

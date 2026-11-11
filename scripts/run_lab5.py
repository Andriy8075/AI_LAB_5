"""Драйвер лабораторної роботи 5: запускає обидві моделі та зберігає результати.

Запуск:  python run_lab5.py
Результати: outputs/lab5_results.json, figures/*.png
"""

import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use("Agg")  # без графічного вікна (для фонового запуску)

import ollama
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import sdn_env
import tools
from llm_controller import LlmController
from visualization import plot_paths, render_state_ascii

QUANTIZED_MODEL = "qwen2.5:3b"
FP16_MODEL = "qwen2.5:3b-instruct-fp16"

OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(ROOT, "figures")


def model_size_gb(model):
    """Розмір моделі у ГБ (з ollama.list)."""
    try:
        for m in ollama.list()["models"]:
            if m["model"] == model:
                return f"{m['size'] / 1e9:.2f} GB"
    except Exception as exc:
        print(f"  (не вдалося отримати розмір: {exc})", flush=True)
    return "n/a"


def unload_model(model):
    """Вивантажує модель з пам'яті Ollama (keep_alive=0)."""
    try:
        ollama.chat(
            model=model,
            messages=[{"role": "user", "content": " "}],
            options={"num_gpu": 0, "num_predict": 1},
            keep_alive=0,
        )
    except Exception as exc:
        print(f"  (попередження: не вдалося вивантажити {model}: {exc})", flush=True)


def run_one(model, graph):
    env = sdn_env.NetworkEnvironment(graph, sdn_env.SOURCE, sdn_env.DESTINATION)
    tb = tools.SdnToolbox(env)
    agent = LlmController(model, tb, num_ctx=4096, max_steps=30)
    print(f"\n=== Запуск моделі {model} ===", flush=True)
    t0 = time.perf_counter()
    res = agent.run()
    res["wall_time_s"] = time.perf_counter() - t0
    res["size"] = model_size_gb(model)
    unload_model(model)
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    graph = sdn_env.generate_grid_graph()
    opt_delay, opt_path = sdn_env.shortest_path_delay(
        graph, sdn_env.SOURCE, sdn_env.DESTINATION
    )
    print(f"Топологія: {graph.number_of_nodes()} вузлів, "
          f"{graph.number_of_edges()} ребер", flush=True)
    print(f"Оптимум (Дейкстра): {opt_delay} мс, "
          f"{len(opt_path)} вузлів", flush=True)

    results = {
        "topology": {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "source": sdn_env.SOURCE,
            "destination": sdn_env.DESTINATION,
            "opt_delay": opt_delay,
            "opt_path": [sdn_env.node_address(n) for n in opt_path],
        },
        "quantized": run_one(QUANTIZED_MODEL, graph),
        "fp16": run_one(FP16_MODEL, graph),
    }

    # Зручне зведення в JSON
    with open(os.path.join(OUT_DIR, "lab5_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ASCII-шляхи
    with open(os.path.join(OUT_DIR, "paths_ascii.txt"), "w", encoding="utf-8") as f:
        f.write("Квантована модель:\n")
        f.write(render_state_ascii(graph, results["quantized"]["path"],
                                   sdn_env.SOURCE, sdn_env.DESTINATION))
        f.write("\n\nНеквантована модель (FP16):\n")
        f.write(render_state_ascii(graph, results["fp16"]["path"],
                                   sdn_env.SOURCE, sdn_env.DESTINATION))
        f.write("\n")

    # Рисунок зі шляхами
    fig = plot_paths(graph, results, sdn_env.SOURCE, sdn_env.DESTINATION,
                     filename=os.path.join(FIG_DIR, "paths.png"))

    # Підсумок
    print("\n" + "=" * 70, flush=True)
    for key, label in [("quantized", "Квантована"), ("fp16", "Неквантована (FP16)")]:
        r = results[key]
        print(f"{label}: {r['model']} ({r['size']})", flush=True)
        print(f"  завершено={r['finished']}, LLM-викликів={r['steps']}, "
              f"інструментів={r['tool_calls']}", flush=True)
        print(f"  час LLM={r['llm_time_s']:.2f} с, wall={r['wall_time_s']:.2f} с", flush=True)
        print(f"  шлях={r['delay']} мс, вузлів={r['nodes']}, "
              f"встановлено={r['installed']}, відкликано={r['revoked']}", flush=True)
        print(f"  токени: eval={r['eval_tokens']}, prompt={r['prompt_tokens']}", flush=True)
    print("=" * 70, flush=True)
    print("Готово. Результати збережено в", OUT_DIR, "та", FIG_DIR, flush=True)


if __name__ == "__main__":
    main()

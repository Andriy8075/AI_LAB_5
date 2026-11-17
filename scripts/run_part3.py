"""Частина 3: пояснення маршруту моделлю (без автоперевірки тексту).

Запуск: python scripts/run_part3.py
Потрібен outputs/part2_scenarios.json (після run_part2.py).
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import ollama

import sdn_env
from llm_controller import LlmController
from tools import SdnToolbox

MODEL = "qwen2.5:3b"
OUT_DIR = os.path.join(ROOT, "outputs")
PART2_PATH = os.path.join(OUT_DIR, "part2_scenarios.json")


def unload_model(model):
    try:
        ollama.chat(
            model=model,
            messages=[{"role": "user", "content": " "}],
            options={"num_gpu": 0, "num_predict": 1},
            keep_alive=0,
        )
    except Exception:
        pass


def build_journal(sc, topo):
    path_res = sc.get("last_path_result") or {}
    lines = []
    lines.append("ЖУРНАЛ РОБОТИ КОНТРОЛЕРА")
    lines.append(f"Відправник: {topo['source']}, отримувач: {topo['destination']}")
    lines.append(f"Встановлений шлях: {' -> '.join(path_res.get('path') or [])}")
    lines.append(f"Затримка шляху: {path_res.get('delay_ms')} мс")
    lines.append(
        f"Оптимальний шлях (Дейкстра без avoid): "
        f"{' -> '.join(path_res.get('optimal_path') or topo.get('opt_path') or [])}"
    )
    lines.append(
        f"Оптимальна затримка: {path_res.get('optimal_delay_ms', topo.get('opt_delay'))} мс"
    )
    lines.append(f"Час передачі: {path_res.get('transmission_ms')} мс")
    lines.append(
        f"Вузьке місце: {path_res.get('bottleneck')} "
        f"({path_res.get('min_bandwidth_MBps')} МБ/с)"
    )
    lines.append(f"Avoid: {path_res.get('avoid')}")
    lines.append("Канали шляху:")
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(PART2_PATH):
        print("Спочатку запустіть scripts/run_part2.py", flush=True)
        sys.exit(1)

    with open(PART2_PATH, encoding="utf-8") as f:
        part2 = json.load(f)

    sc = None
    for item in part2["scenarios"]:
        if item["id"] == "s1_basic_path" and item.get("last_path_result"):
            sc = item
            break
    if sc is None:
        for item in part2["scenarios"]:
            if item.get("last_path_result"):
                sc = item
                break
    if sc is None:
        print("Немає сценарію з встановленим шляхом", flush=True)
        sys.exit(1)

    topo = part2["topology"]
    journal = build_journal(sc, topo)
    print(journal, flush=True)

    graph = sdn_env.generate_grid_graph()
    tb = SdnToolbox(sdn_env.NetworkEnvironment(
        graph, sdn_env.SOURCE, sdn_env.DESTINATION
    ))
    agent = LlmController(MODEL, tb, num_gpu=0, verbose=True)
    print("\n=== Запит пояснення ===", flush=True)
    expl = agent.explain_route(journal)
    print(expl["explanation"], flush=True)
    unload_model(MODEL)

    out = {
        "scenario_id": sc["id"],
        "journal": journal,
        "explanation": expl["explanation"],
        "model": expl["model"],
        "llm_time_s": expl["llm_time_s"],
        "prompt_tokens": expl["prompt_tokens"],
        "eval_tokens": expl["eval_tokens"],
        "facts": {
            "path": sc["last_path_result"].get("path"),
            "delay_ms": sc["last_path_result"].get("delay_ms"),
            "optimal_delay_ms": sc["last_path_result"].get("optimal_delay_ms"),
            "optimal_path": sc["last_path_result"].get("optimal_path"),
            "transmission_ms": sc["last_path_result"].get("transmission_ms"),
            "bottleneck": sc["last_path_result"].get("bottleneck"),
        },
    }
    out_path = os.path.join(OUT_DIR, "part3_explanation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Збережено:", out_path, flush=True)


if __name__ == "__main__":
    main()

"""Частина 2: NL-сценарії з tool calling.

Запуск: python scripts/run_part2.py
Результати: outputs/part2_scenarios.json, figures/part2_*
"""

import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import ollama

import sdn_env
from llm_controller import LlmController
from tools import SdnToolbox
from visualization import plot_single_path, save_step_figures, render_state_ascii

MODEL = "qwen2.5:3b"
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(ROOT, "figures")

SCENARIOS = [
    {
        "id": "s1_basic_path",
        "query": "Проклади шлях для потоку від вузла 10.0.0.0 до вузла 10.0.4.4",
    },
    {
        "id": "s2_avoid",
        "query": (
            "Проклади шлях до 10.0.4.4 в обхід вузлів 10.0.2.2 та 10.0.2.3, "
            "які виводяться на обслуговування. Відправник — 10.0.0.0."
        ),
    },
    {
        "id": "s3_head_and_delays",
        "query": (
            "На якому вузлі зараз зупинилась побудова шляху і які канали з "
            "нього мають найменшу затримку?"
        ),
    },
    {
        "id": "s4_transfer",
        "query": (
            "Скільки триватиме передача останнього потоку і який канал є "
            "вузьким місцем?"
        ),
    },
    {
        "id": "s5_failure",
        "query": (
            "Канал між 10.0.1.1 та 10.0.1.2 вийшов з ладу – перебудуй шлях "
            "потоку від 10.0.0.0 до 10.0.4.4."
        ),
    },
]


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


def make_toolbox(graph, reuse_env=None):
    if reuse_env is None:
        env = sdn_env.NetworkEnvironment(
            graph, sdn_env.SOURCE, sdn_env.DESTINATION
        )
    else:
        env = reuse_env
    return SdnToolbox(env)


def run_scenario(graph, scenario, env=None):
    tb = make_toolbox(graph, reuse_env=env)
    agent = LlmController(MODEL, tb, num_gpu=0, verbose=True)
    print(f"\n=== {scenario['id']} ===", flush=True)
    print(f"Запит: {scenario['query']}", flush=True)
    t0 = time.perf_counter()
    res = agent.run(scenario["query"])
    res["wall_time_s"] = time.perf_counter() - t0
    res["scenario_id"] = scenario["id"]
    # серіалізація path (кортежі -> списки)
    res["path"] = [
        list(p) if not isinstance(p, str) else p for p in res.get("path", [])
    ]
    if res.get("last_path_result"):
        # journal вже з адресами-рядками
        pass
    return res, tb.env


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    graph = sdn_env.generate_grid_graph()
    opt_delay, opt_path = sdn_env.shortest_path_delay(
        graph, sdn_env.SOURCE, sdn_env.DESTINATION
    )
    print(
        f"Топологія: {graph.number_of_nodes()} вузлів, "
        f"{graph.number_of_edges()} ребер; оптимум {opt_delay} мс",
        flush=True,
    )

    results = {
        "topology": {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "source": sdn_env.SOURCE,
            "destination": sdn_env.DESTINATION,
            "opt_delay": opt_delay,
            "opt_path": [sdn_env.node_address(n) for n in opt_path],
        },
        "model": MODEL,
        "scenarios": [],
    }

    shared_env = None
    for sc in SCENARIOS:
        # s3/s4 працюють на стані після попередніх сценаріїв
        if sc["id"] in ("s3_head_and_delays", "s4_transfer") and shared_env is not None:
            res, shared_env = run_scenario(graph, sc, env=shared_env)
        else:
            res, shared_env = run_scenario(graph, sc, env=None)

        # фігури шляху / кроків
        sc_fig_dir = os.path.join(FIG_DIR, f"part2_{sc['id']}")
        os.makedirs(sc_fig_dir, exist_ok=True)
        path_addrs = res.get("flow_state", {}).get("path") or []
        if path_addrs:
            plot_single_path(
                graph, path_addrs, sdn_env.SOURCE, sdn_env.DESTINATION,
                title=f"{sc['id']}: {res.get('delay')} мс",
                filename=os.path.join(sc_fig_dir, "final_path.png"),
            )
            ascii_txt = render_state_ascii(
                graph, path_addrs, sdn_env.SOURCE, sdn_env.DESTINATION,
                head=res.get("flow_state", {}).get("head"),
            )
            with open(
                os.path.join(OUT_DIR, f"part2_{sc['id']}_ascii.txt"),
                "w", encoding="utf-8",
            ) as f:
                f.write(ascii_txt)

        snaps = res.get("path_snapshots") or []
        if snaps:
            save_step_figures(
                graph, snaps, sdn_env.SOURCE, sdn_env.DESTINATION,
                out_dir=os.path.join(sc_fig_dir, "steps"),
                prefix="step",
            )

        # компактний запис для JSON
        results["scenarios"].append({
            "id": sc["id"],
            "query": sc["query"],
            "answer": res.get("answer"),
            "steps": res.get("steps"),
            "tool_calls": res.get("tool_calls"),
            "llm_time_s": res.get("llm_time_s"),
            "wall_time_s": res.get("wall_time_s"),
            "log": res.get("log"),
            "flow_state": res.get("flow_state"),
            "transfer": res.get("transfer"),
            "last_path_result": {
                k: v for k, v in (res.get("last_path_result") or {}).items()
                if k != "snapshots"
            } if res.get("last_path_result") else None,
            "path_snapshots": snaps,
            "tool_summary": res.get("tool_summary"),
        })

    unload_model(MODEL)

    out_path = os.path.join(OUT_DIR, "part2_scenarios.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Збережено:", out_path, flush=True)

    # Зручний для читання чат (HTML + Markdown + окремі .txt відповідей)
    try:
        import importlib.util
        _chat_path = os.path.join(os.path.dirname(__file__), "export_part2_chat.py")
        _spec = importlib.util.spec_from_file_location("export_part2_chat", _chat_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        chat_paths = _mod.export_from_part2(results)
        print("Чат HTML:", chat_paths["html"], flush=True)
        print("Чат Markdown:", chat_paths["markdown"], flush=True)
        for p in chat_paths["answers"]:
            print("Відповідь:", p, flush=True)
    except Exception as exc:
        print(f"(не вдалося експортувати чат: {exc})", flush=True)


if __name__ == "__main__":
    main()

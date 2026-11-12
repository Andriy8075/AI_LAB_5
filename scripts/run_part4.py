"""Частина 4: критична оцінка ≥10 NL-запитів на двох квантуваннях.

Запуск: python scripts/run_part4.py
Результати: outputs/part4_eval.json
"""

import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import ollama

import sdn_env
from llm_controller import LlmController
from tools import SdnToolbox

MODELS = [
    ("qwen2.5:3b", "Q4_K_M"),
    ("qwen2.5:3b-instruct-q8_0", "Q8_0"),
]
OUT_DIR = os.path.join(ROOT, "outputs")

# expected — орієнтовна послідовність tools (не суворий порядок)
QUERIES = [
    {
        "id": "q01",
        "query": "Проклади шлях від 10.0.0.0 до 10.0.4.4",
        "expected_tools": ["install_path"],
        "kind": "valid",
    },
    {
        "id": "q02",
        "query": (
            "Проклади шлях від 10.0.0.0 до 10.0.4.4 в обхід 10.0.1.1 та 10.0.3.2"
        ),
        "expected_tools": ["install_path"],
        "kind": "valid_avoid",
    },
    {
        "id": "q03",
        "query": "Проклади шлях до вузла 10.0.9.9",
        "expected_tools": ["install_path"],
        "kind": "impossible_dst",
    },
    {
        "id": "q04",
        "query": "Який зараз стан потоку?",
        "expected_tools": ["get_flow_state"],
        "kind": "valid_state",
    },
    {
        "id": "q05",
        "query": "Покажи телеметрію вузла 10.0.2.2: затримки і пропускну здатність",
        "expected_tools": ["get_link_stats"],
        "kind": "valid_stats",
    },
    {
        "id": "q06",
        "query": "Покажи топологію мережі з бази знань",
        "expected_tools": ["get_topology"],
        "kind": "valid_topo",
    },
    {
        "id": "q07",
        "query": "Канал між 10.0.0.1 і 10.0.0.2 вийшов з ладу",
        "expected_tools": ["simulate_link_failure"],
        "kind": "valid_failure",
    },
    {
        "id": "q08",
        "query": "Проклади якийсь хороший шлях якомога швидше",
        "expected_tools": ["install_path"],
        "kind": "ambiguous",
    },
    {
        "id": "q09",
        "query": (
            "Проклади шлях від 10.0.0.0 до 10.0.4.4 в обхід усіх вузлів "
            "10.0.1.0, 10.0.1.1, 10.0.1.2, 10.0.1.3, 10.0.1.4, "
            "10.0.2.0, 10.0.2.1, 10.0.2.2, 10.0.2.3, 10.0.2.4, "
            "10.0.3.0, 10.0.3.1, 10.0.3.2, 10.0.3.3, 10.0.3.4"
        ),
        "expected_tools": ["install_path"],
        "kind": "impossible_avoid",
    },
    {
        "id": "q10",
        "query": "Встанови правило на схід і одразу скасуй його",
        "expected_tools": ["install_rule"],
        "kind": "valid_rule",
    },
    {
        "id": "q11",
        "query": "Скільки триватиме передача потоку і де вузьке місце?",
        "expected_tools": ["get_flow_state", "install_path"],
        "kind": "needs_path_or_state",
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


def actual_tools(log):
    return [e["tool"] for e in log if e.get("tool") and not e["tool"].startswith("(")]


def classify(row, res):
    """Оцінка правильності відповіді / викликів."""
    tools = actual_tools(res.get("log") or [])
    expected = row["expected_tools"]
    kind = row["kind"]
    answer = (res.get("answer") or "").lower()
    errors = []

    has_expected = any(t in tools for t in expected)

    if kind == "impossible_dst":
        # має спробувати tool або відмовити; не стверджувати успішний шлях до 10.0.9.9
        ok_refuse = (
            "не існу" in answer or "неможлив" in answer or "помилк" in answer
            or "error" in answer or any(
                e.get("status") == "error" for e in res.get("log") or []
            )
        )
        halluc_ok_path = (
            "встановлено" in answer and "10.0.9.9" in answer
            and "не" not in answer[: answer.find("10.0.9.9") + 1]
        )
        correct = ok_refuse and not halluc_ok_path
        if not has_expected and not ok_refuse:
            errors.append("wrong_tool")
        if halluc_ok_path:
            errors.append("hallucinated_result")
        if not correct and not errors:
            errors.append("incorrect_answer")
        return correct, errors

    if kind == "impossible_avoid":
        ok_refuse = (
            "неможлив" in answer or "не вда" in answer or "не існує" in answer
            or "error" in answer or any(
                e.get("status") == "error" for e in res.get("log") or []
            )
        )
        correct = ok_refuse
        if not correct:
            errors.append("incorrect_answer")
        return correct, errors

    if kind == "ambiguous":
        # прийнятно: викликати install_path з дефолтними src/dst або уточнити
        correct = has_expected or "уточн" in answer or "10.0.0.0" in (res.get("answer") or "")
        if not correct:
            errors.append("wrong_tool")
        return correct, errors

    if not has_expected:
        errors.append("wrong_tool")
        return False, errors

    # зайві «шкідливі» виклики для простих запитів
    if kind in ("valid_state", "valid_stats", "valid_topo") and "install_path" in tools:
        # не завжди помилка, але зайвий виклик
        errors.append("extra_call")

    # перевірка аргументів install_path для q03 вже вище
    for e in res.get("log") or []:
        if e.get("status") == "error" and kind.startswith("valid"):
            # для valid_failure канал може не існувати — тоді ok якщо пояснили
            if kind == "valid_failure":
                if "не існу" in answer or "немає" in answer or "помилк" in answer:
                    continue
            errors.append("tool_error")

    correct = has_expected and "hallucinated_result" not in errors
    if kind.startswith("valid") and not (res.get("answer") or "").strip():
        # порожня відповідь після tools — частково ок, але слабко
        if not tools:
            correct = False
            errors.append("incorrect_answer")

    return correct, list(dict.fromkeys(errors))


def run_one(model, graph, row):
    env = sdn_env.NetworkEnvironment(graph, sdn_env.SOURCE, sdn_env.DESTINATION)
    tb = SdnToolbox(env)
    agent = LlmController(model, tb, num_gpu=0, max_steps=24, verbose=True)
    print(f"\n--- {row['id']} | {model} ---", flush=True)
    print(row["query"], flush=True)
    t0 = time.perf_counter()
    res = agent.run(row["query"])
    wall = time.perf_counter() - t0
    correct, errors = classify(row, res)
    return {
        "id": row["id"],
        "query": row["query"],
        "kind": row["kind"],
        "expected_tools": row["expected_tools"],
        "actual_tools": actual_tools(res.get("log") or []),
        "answer": res.get("answer"),
        "correct": correct,
        "error_types": errors,
        "log": res.get("log"),
        "steps": res.get("steps"),
        "tool_calls": res.get("tool_calls"),
        "llm_time_s": res.get("llm_time_s"),
        "wall_time_s": wall,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    graph = sdn_env.generate_grid_graph()
    out = {"models": {}, "queries": [q["query"] for q in QUERIES]}

    for model, label in MODELS:
        print(f"\n===== МОДЕЛЬ {model} ({label}) =====", flush=True)
        rows = []
        for q in QUERIES:
            rows.append(run_one(model, graph, q))
        unload_model(model)
        n_ok = sum(1 for r in rows if r["correct"])
        out["models"][model] = {
            "label": label,
            "rows": rows,
            "correct_count": n_ok,
            "total": len(rows),
            "accuracy": round(n_ok / len(rows), 3),
        }
        print(f"Точність {label}: {n_ok}/{len(rows)}", flush=True)

    out_path = os.path.join(OUT_DIR, "part4_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Збережено:", out_path, flush=True)


if __name__ == "__main__":
    main()

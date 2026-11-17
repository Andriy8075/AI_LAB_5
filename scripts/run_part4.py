"""Частина 4: критична оцінка ≥10 NL-запитів на двох квантуваннях.

За бажанням PDF — порівняння частини запитів українською та англійською.

Запуск:
  python scripts/run_part4.py              # усі UA + EN-підмножина
  python scripts/run_part4.py --lang-only  # лише EN (дописати до part4_eval.json)

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
        "lang": "uk",
    },
    {
        "id": "q02",
        "query": (
            "Проклади шлях від 10.0.0.0 до 10.0.4.4 в обхід 10.0.1.1 та 10.0.3.2"
        ),
        "expected_tools": ["install_path"],
        "kind": "valid_avoid",
        "lang": "uk",
    },
    {
        "id": "q03",
        "query": "Проклади шлях до вузла 10.0.9.9",
        "expected_tools": ["install_path"],
        "kind": "impossible_dst",
        "lang": "uk",
    },
    {
        "id": "q04",
        "query": "Який зараз стан потоку?",
        "expected_tools": ["get_flow_state"],
        "kind": "valid_state",
        "lang": "uk",
    },
    {
        "id": "q05",
        "query": "Покажи телеметрію вузла 10.0.2.2: затримки і пропускну здатність",
        "expected_tools": ["get_link_stats"],
        "kind": "valid_stats",
        "lang": "uk",
    },
    {
        "id": "q06",
        "query": "Покажи топологію мережі з бази знань",
        "expected_tools": ["get_topology"],
        "kind": "valid_topo",
        "lang": "uk",
    },
    {
        "id": "q07",
        "query": "Канал між 10.0.0.1 і 10.0.0.2 вийшов з ладу",
        "expected_tools": ["simulate_link_failure"],
        "kind": "valid_failure",
        "lang": "uk",
    },
    {
        "id": "q08",
        "query": "Проклади якийсь хороший шлях якомога швидше",
        "expected_tools": ["install_path"],
        "kind": "ambiguous",
        "lang": "uk",
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
        "lang": "uk",
    },
    {
        "id": "q10",
        "query": "Встанови правило на схід і одразу скасуй його",
        "expected_tools": ["install_rule"],
        "kind": "valid_rule",
        "lang": "uk",
    },
    {
        "id": "q11",
        "query": "Скільки триватиме передача потоку і де вузьке місце?",
        "expected_tools": ["get_flow_state", "install_path"],
        "kind": "needs_path_or_state",
        "lang": "uk",
    },
]

# Підмножина для порівняння UA ↔ EN (ті самі kind / expected_tools)
EN_QUERIES = [
    {
        "id": "q01_en",
        "pair_id": "q01",
        "query": "Install a path from 10.0.0.0 to 10.0.4.4",
        "expected_tools": ["install_path"],
        "kind": "valid",
        "lang": "en",
    },
    {
        "id": "q03_en",
        "pair_id": "q03",
        "query": "Install a path to node 10.0.9.9",
        "expected_tools": ["install_path"],
        "kind": "impossible_dst",
        "lang": "en",
    },
    {
        "id": "q04_en",
        "pair_id": "q04",
        "query": "What is the current flow state?",
        "expected_tools": ["get_flow_state"],
        "kind": "valid_state",
        "lang": "en",
    },
    {
        "id": "q05_en",
        "pair_id": "q05",
        "query": (
            "Show telemetry for node 10.0.2.2: delays and bandwidth"
        ),
        "expected_tools": ["get_link_stats"],
        "kind": "valid_stats",
        "lang": "en",
    },
    {
        "id": "q08_en",
        "pair_id": "q08",
        "query": "Install some good path as fast as possible",
        "expected_tools": ["install_path"],
        "kind": "ambiguous",
        "lang": "en",
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


def _mentions_refusal(answer: str) -> bool:
    a = answer.lower()
    uk = ("не існу", "неможлив", "не вда", "немає", "помилк")
    en = (
        "does not exist", "doesn't exist", "impossible", "cannot", "can't",
        "unable", "not found", "no such", "error", "failed", "fail",
    )
    return any(x in a for x in uk) or any(x in a for x in en)


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
        ok_refuse = _mentions_refusal(answer) or any(
            e.get("status") == "error" for e in res.get("log") or []
        )
        raw = res.get("answer") or ""
        halluc_ok_path = False
        if "10.0.9.9" in raw:
            idx = raw.lower().find("10.0.9.9")
            prefix = raw.lower()[: idx + 1]
            success_words = ("встановлено", "installed", "success", "прокладено")
            refuse_near = ("не ", "not ", "no ", "cannot", "can't", "unable", "fail")
            if any(w in raw.lower() for w in success_words) and not any(
                w in prefix[-40:] for w in refuse_near
            ):
                halluc_ok_path = True
        correct = ok_refuse and not halluc_ok_path
        if not has_expected and not ok_refuse:
            errors.append("wrong_tool")
        if halluc_ok_path:
            errors.append("hallucinated_result")
        if not correct and not errors:
            errors.append("incorrect_answer")
        return correct, errors

    if kind == "impossible_avoid":
        ok_refuse = _mentions_refusal(answer) or any(
            e.get("status") == "error" for e in res.get("log") or []
        )
        correct = ok_refuse
        if not correct:
            errors.append("incorrect_answer")
        return correct, errors

    if kind == "ambiguous":
        # прийнятно: викликати install_path з дефолтними src/dst або уточнити
        clarify = ("уточн" in answer or "clarif" in answer
                   or "which" in answer or "specify" in answer)
        correct = has_expected or clarify or "10.0.0.0" in (res.get("answer") or "")
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
                if _mentions_refusal(answer):
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
    print(f"\n--- {row['id']} | {model} | {row.get('lang', '?')} ---", flush=True)
    print(row["query"], flush=True)
    t0 = time.perf_counter()
    res = agent.run(row["query"])
    wall = time.perf_counter() - t0
    correct, errors = classify(row, res)
    out = {
        "id": row["id"],
        "query": row["query"],
        "kind": row["kind"],
        "lang": row.get("lang", "uk"),
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
    if row.get("pair_id"):
        out["pair_id"] = row["pair_id"]
    return out


def _summarize_block(rows, label, model):
    n_ok = sum(1 for r in rows if r["correct"])
    return {
        "label": label,
        "rows": rows,
        "correct_count": n_ok,
        "total": len(rows),
        "accuracy": round(n_ok / len(rows), 3) if rows else 0.0,
        "model": model,
    }


def build_language_compare(uk_models, en_models):
    """Паралельне порівняння однакових сценаріїв UA vs EN."""
    pairs = []
    for en_q in EN_QUERIES:
        pair_id = en_q["pair_id"]
        entry = {
            "pair_id": pair_id,
            "kind": en_q["kind"],
            "uk_query": next(q["query"] for q in QUERIES if q["id"] == pair_id),
            "en_query": en_q["query"],
            "by_model": {},
        }
        for model, _label in MODELS:
            uk_rows = {r["id"]: r for r in uk_models[model]["rows"]}
            en_rows = {r["id"]: r for r in en_models[model]["rows"]}
            uk = uk_rows.get(pair_id)
            en = en_rows.get(en_q["id"])
            if not uk or not en:
                continue
            entry["by_model"][model] = {
                "label": uk_models[model]["label"],
                "uk_correct": uk["correct"],
                "en_correct": en["correct"],
                "uk_tools": uk["actual_tools"],
                "en_tools": en["actual_tools"],
                "uk_errors": uk["error_types"],
                "en_errors": en["error_types"],
                "same_outcome": uk["correct"] == en["correct"],
            }
        pairs.append(entry)

    per_model = {}
    for model, label in MODELS:
        uk_ok = en_ok = both = n = 0
        for p in pairs:
            m = p["by_model"].get(model)
            if not m:
                continue
            n += 1
            uk_ok += int(m["uk_correct"])
            en_ok += int(m["en_correct"])
            both += int(m["same_outcome"])
        per_model[model] = {
            "label": label,
            "pairs": n,
            "uk_correct": uk_ok,
            "en_correct": en_ok,
            "same_outcome": both,
            "uk_accuracy": round(uk_ok / n, 3) if n else 0.0,
            "en_accuracy": round(en_ok / n, 3) if n else 0.0,
            "agreement": round(both / n, 3) if n else 0.0,
        }
    return {"pairs": pairs, "summary": per_model}


def run_query_set(queries, graph):
    models = {}
    for model, label in MODELS:
        print(f"\n===== МОДЕЛЬ {model} ({label}) | {len(queries)} запитів =====",
              flush=True)
        rows = [run_one(model, graph, q) for q in queries]
        unload_model(model)
        block = _summarize_block(rows, label, model)
        models[model] = block
        print(f"Точність {label}: {block['correct_count']}/{block['total']}",
              flush=True)
    return models


def main(lang_only=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "part4_eval.json")
    graph = sdn_env.generate_grid_graph()

    if lang_only and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            out = json.load(f)
        print("Завантажено наявні UA-результати; прогін лише EN…", flush=True)
    else:
        out = {
            "models": run_query_set(QUERIES, graph),
            "queries": [q["query"] for q in QUERIES],
        }

    en_models = run_query_set(EN_QUERIES, graph)
    out["models_en"] = en_models
    out["queries_en"] = [q["query"] for q in EN_QUERIES]
    out["language_compare"] = build_language_compare(out["models"], en_models)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Збережено:", out_path, flush=True)
    for model, s in out["language_compare"]["summary"].items():
        print(
            f"UA↔EN {s['label']}: uk={s['uk_accuracy']} en={s['en_accuracy']} "
            f"agreement={s['agreement']} ({s['same_outcome']}/{s['pairs']})",
            flush=True,
        )


if __name__ == "__main__":
    main(lang_only="--lang-only" in sys.argv)

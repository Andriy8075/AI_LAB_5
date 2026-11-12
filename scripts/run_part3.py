"""Частина 3: пояснення маршруту моделлю + перевірка неточностей.

Запуск: python scripts/run_part3.py
Потрібен outputs/part2_scenarios.json (після run_part2.py).
"""

import json
import os
import re
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
    lines.append(f"Оптимальний шлях (Дейкстра без avoid): {' -> '.join(path_res.get('optimal_path') or topo.get('opt_path') or [])}")
    lines.append(f"Оптимальна затримка: {path_res.get('optimal_delay_ms', topo.get('opt_delay'))} мс")
    lines.append(f"Час передачі: {path_res.get('transmission_ms')} мс")
    lines.append(f"Вузьке місце: {path_res.get('bottleneck')} ({path_res.get('min_bandwidth_MBps')} МБ/с)")
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


def verify_explanation(text, path_res, topo):
    """Евристична перевірка пояснення за фактами."""
    issues = []
    delay = path_res.get("delay_ms")
    opt = path_res.get("optimal_delay_ms", topo.get("opt_delay"))
    path = path_res.get("path") or []
    opt_path = path_res.get("optimal_path") or topo.get("opt_path") or []

    # чи згадує фактичну затримку
    if delay is not None and str(delay) not in text and f"{delay}" not in text:
        # іноді модель пише словами — м'яке попередження
        issues.append({
            "type": "missing_delay",
            "detail": f"У поясненні не знайдено фактичну затримку шляху {delay} мс",
        })

    if opt is not None and delay is not None:
        lost = delay - opt
        # якщо шлях оптимальний, а модель каже про великі втрати
        if lost == 0 and re.search(r"втрат|гірш|більш(у|а) затрим", text, re.I):
            if not re.search(r"немає втрат|без втрат|збігає|оптимальн", text, re.I):
                issues.append({
                    "type": "false_loss",
                    "detail": (
                        f"Шлях оптимальний (delay={delay}=opt={opt}), але пояснення "
                        "говорить про втрати затримки"
                    ),
                })

    # вигадані адреси 10.0.x.y поза шляхом
    mentioned = re.findall(r"10\.0\.\d+\.\d+", text)
    known = set(path) | set(opt_path) | {
        topo["source"], topo["destination"]
    }
    for link in path_res.get("links") or []:
        known.add(link["a"])
        known.add(link["b"])
    for addr in mentioned:
        try:
            sdn_env.parse_address(addr)
        except Exception:
            issues.append({
                "type": "invalid_address",
                "detail": f"Некоректна адреса в поясненні: {addr}",
            })
            continue
        # адреси поза сіткою 0..4
        r, c = sdn_env.parse_address(addr)
        if not (0 <= r <= 4 and 0 <= c <= 4):
            issues.append({
                "type": "out_of_topology",
                "detail": f"Адреса поза топологією: {addr}",
            })

    # плутанина одиниць: BW як «затримка»
    if re.search(r"затримк\w*\s+\d+\s*МБ", text, re.I) or re.search(
        r"\d+\s*МБ/с.*затрим", text, re.I
    ):
        issues.append({
            "type": "unit_confusion",
            "detail": (
                "Пояснення плутає затримку (мс) з пропускною здатністю (МБ/с)"
            ),
        })

    bn = path_res.get("bottleneck") or ""
    min_bw = path_res.get("min_bandwidth_MBps")
    if bn and min_bw is not None:
        if "вузьк" in text.lower() and str(min_bw) in text:
            if re.search(rf"{min_bw}\s*мс", text):
                issues.append({
                    "type": "wrong_bottleneck_unit",
                    "detail": (
                        f"Вузьке місце має BW {min_bw} МБ/с, але в тексті "
                        "цифра подана як мс"
                    ),
                })

    if not text.strip():
        issues.append({"type": "empty", "detail": "Порожнє пояснення"})
    elif text.rstrip().endswith((":", "-", "—")) or len(text) < 80:
        issues.append({
            "type": "truncated",
            "detail": "Пояснення виглядає обрізаним або занадто коротким",
        })

    return issues


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(PART2_PATH):
        print("Спочатку запустіть scripts/run_part2.py", flush=True)
        sys.exit(1)

    with open(PART2_PATH, encoding="utf-8") as f:
        part2 = json.load(f)

    # беремо s1, якщо немає — перший з last_path_result
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

    issues = verify_explanation(
        expl["explanation"], sc["last_path_result"], topo
    )
    out = {
        "scenario_id": sc["id"],
        "journal": journal,
        "explanation": expl["explanation"],
        "model": expl["model"],
        "llm_time_s": expl["llm_time_s"],
        "prompt_tokens": expl["prompt_tokens"],
        "eval_tokens": expl["eval_tokens"],
        "inaccuracies": issues,
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
    print(f"Неточностей: {len(issues)}", flush=True)


if __name__ == "__main__":
    main()

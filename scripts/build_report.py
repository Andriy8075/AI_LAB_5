# -*- coding: utf-8 -*-
"""Генератор звіту Lab5_report.docx (частини 1–4 за PDF)."""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Cm

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

FONT_BODY = "Times New Roman"
FONT_CODE = "Consolas"


def set_font(run, name=FONT_BODY, size=14, bold=False, italic=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def add_heading(doc, text, level=1):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_font(run, size=16 if level == 1 else 14, bold=True)
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(6 if level == 1 else 4)
    return p


def add_para(doc, text, size=14, bold=False, italic=False, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_font(run, size=size, bold=bold, italic=italic)
    if align is not None:
        p.alignment = align
    p.paragraph_format.line_spacing = 1.5
    return p


def add_code(doc, code, max_chars=12000):
    if len(code) > max_chars:
        code = code[:max_chars] + "\n# … (скорочено у звіті; повний код у src/)"
    for line in code.split("\n"):
        p = doc.add_paragraph()
        run = p.add_run(line if line else " ")
        run.font.name = FONT_CODE
        run.font.size = Pt(8)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(h))
        set_font(run, size=11, bold=True)
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = table.rows[r_i + 1].cells[c_i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            set_font(run, size=10)
    doc.add_paragraph()


def add_picture(doc, path, width_cm=15.5, caption=None):
    if not os.path.exists(path):
        add_para(doc, f"[немає файлу: {path}]", size=11, italic=True)
        return
    doc.add_picture(path, width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        add_para(doc, caption, size=12, italic=True,
                 align=WD_ALIGN_PARAGRAPH.CENTER)


def read_file(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def add_title_header(doc):
    for text in [
        "НАЦІОНАЛЬНИЙ ТЕХНІЧНИЙ УНІВЕРСИТЕТ УКРАЇНИ",
        "«КИЇВСЬКИЙ ПОЛІТЕХНІЧНИЙ ІНСТИТУТ імені Ігоря Сікорського»",
    ]:
        add_para(doc, text, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "", size=14)
    add_para(doc, "Факультет інформатики та обчислювальної техніки",
             align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "Кафедра інформаційних систем та технологій",
             align=WD_ALIGN_PARAGRAPH.CENTER)
    for _ in range(3):
        add_para(doc, "")
    add_para(doc, "Звіт до лабораторної роботи №5", bold=True,
             align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "З дисципліни «Технології штучного інтелекту»",
             align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "")
    add_para(doc, "2026 р.", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)


def add_signature_table(doc):
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    left, right = table.rows[0].cells
    for line in ["Прийняв:", "Викладач", "Кочура Ю. П."]:
        p = left.paragraphs[0] if line.startswith("Прий") else left.add_paragraph()
        if line.startswith("Прий"):
            left.paragraphs[0].clear()
            p = left.paragraphs[0]
        set_font(p.add_run(line), size=14)
    for line in ["Виконав:", "Студент", "________________"]:
        p = right.paragraphs[0] if line.startswith("Вик") else right.add_paragraph()
        if line.startswith("Вик"):
            right.paragraphs[0].clear()
            p = right.paragraphs[0]
        set_font(p.add_run(line), size=14)
    doc.add_paragraph()


def main():
    part1 = json.load(open(os.path.join(ROOT, "outputs", "part1_benchmark.json"),
                           encoding="utf-8"))
    part2 = json.load(open(os.path.join(ROOT, "outputs", "part2_scenarios.json"),
                           encoding="utf-8"))
    part3 = json.load(open(os.path.join(ROOT, "outputs", "part3_explanation.json"),
                           encoding="utf-8"))
    part4 = json.load(open(os.path.join(ROOT, "outputs", "part4_eval.json"),
                           encoding="utf-8"))

    code_env = read_file(os.path.join("src", "sdn_env.py"))
    code_kb = read_file(os.path.join("src", "knowledge.py"))
    code_tools = read_file(os.path.join("src", "tools.py"))
    code_llm = read_file(os.path.join("src", "llm_controller.py"))

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = FONT_BODY
    style.font.size = Pt(14)

    add_title_header(doc)
    add_signature_table(doc)

    add_heading(doc, "Мета роботи", 1)
    add_para(
        doc,
        "Мета роботи: додати до інтелектуального агента — мережевого контролера "
        "інтерфейс природної мови на основі LLM з tool calling; оцінити "
        "продуктивність локальних квантованих моделей і критично оцінити "
        "їхні відповіді.",
    )

    add_heading(doc, "Характеристики обладнання", 1)
    add_para(
        doc,
        "Вимірювання виконано локально: CPU 12th Gen Intel Core i5-12500H "
        "(12/16); RAM 16 ГБ DDR4; GPU NVIDIA GeForce RTX 3060 Laptop (6 ГБ) "
        "та Intel Iris Xe. Через несумісність драйвера NVIDIA з CUDA-ядром "
        "Ollama інференс — на CPU (num_gpu=0); вимірювалась оперативна пам'ять.",
    )

    # ---- Part 1 ----
    add_heading(doc, "Частина 1. Локальний запуск мовної моделі", 1)
    add_para(
        doc,
        "Встановлено Ollama та завантажено Qwen2.5-3B у варіантах FP16, Q8_0 "
        "і Q4_K_M. На однаковому запиті довжиною 452 токени (3 повторення з "
        "усередненням) виміряно розмір файлу, RAM, швидкість обробки запиту "
        "та генерації.",
    )
    rows = []
    for key, label, qn in [
        ("qwen2.5:3b-instruct-fp16", "Qwen2.5-3B FP16", "FP16"),
        ("qwen2.5:3b-instruct-q8_0", "Qwen2.5-3B Q8_0", "Q8_0"),
        ("qwen2.5:3b", "Qwen2.5-3B Q4_K_M", "Q4_K_M"),
    ]:
        d = part1[key]
        rows.append([
            label, qn, f"{d['size_gb']:.2f}", f"{d['ram_used_gb']:.2f}",
            f"{d['prompt_processing_tok_s']:.2f}", f"{d['generation_tok_s']:.2f}",
        ])
    add_para(doc, "Таблиця 1 – Порівняння варіантів квантування (CPU)", bold=True)
    add_table(
        doc,
        ["Модель", "Квант.", "Розмір, ГБ", "RAM, ГБ",
         "Обробка, ток/с", "Генерація, ток/с"],
        rows,
    )

    # ---- Part 2 tools ----
    add_heading(doc, "Частина 2. Інструменти агента та NL-сценарії", 1)
    add_para(
        doc,
        "Функції контролера оформлено як інструменти з JSON Schema. "
        "Мінімальний набір: get_flow_state(); get_link_stats(node) — затримки "
        "та пропускна здатність; install_rule(interface) — N/E/S/W або revoke; "
        "install_path(src, dst, avoid) — knowledge-based побудова повного шляху; "
        "get_topology(); simulate_link_failure(a, b). LLM не обчислює шлях сама: "
        "стан контролера — джерело істини. temperature=0.",
    )
    add_para(
        doc,
        "Цикл: запит користувача → модель → виклик інструмента → виконання → "
        "результат → остаточна відповідь. Топологія 5×5, SOURCE=10.0.0.0, "
        f"DESTINATION=10.0.4.4; оптимальна затримка "
        f"{part2['topology']['opt_delay']} мс.",
    )
    add_picture(
        doc, os.path.join(ROOT, "figures", "topology.png"),
        caption="Рисунок 1 – Топологія та оптимальний шлях Дейкстри",
    )

    add_heading(doc, "Програмний код інтеграції", 1)
    add_heading(doc, "sdn_env.py", 2)
    add_code(doc, code_env)
    add_heading(doc, "knowledge.py", 2)
    add_code(doc, code_kb)
    add_heading(doc, "tools.py", 2)
    add_code(doc, code_tools)
    add_heading(doc, "llm_controller.py", 2)
    add_code(doc, code_llm)

    add_heading(doc, "Журнали сценаріїв частини 2", 1)
    for sc in part2["scenarios"]:
        add_heading(doc, f"Сценарій {sc['id']}", 2)
        add_para(doc, f"Запит: {sc['query']}", italic=True)
        for e in sc.get("log", []):
            if str(e["tool"]).startswith("("):
                add_para(
                    doc,
                    f"Крок {e['step']}: відповідь моделі — "
                    f"{(e.get('result') or '')[:600]}",
                    size=11,
                )
            else:
                add_para(
                    doc,
                    f"Крок {e['step']}: {e['tool']} "
                    f"{json.dumps(e.get('arguments') or {}, ensure_ascii=False)} "
                    f"[{e['status']}]",
                    size=11,
                )
        pr = sc.get("last_path_result") or {}
        if pr:
            add_para(
                doc,
                f"Шлях: {' → '.join(pr.get('path') or [])}; "
                f"затримка {pr.get('delay_ms')} мс; "
                f"час передачі {pr.get('transmission_ms')} мс; "
                f"вузьке місце {pr.get('bottleneck')}.",
            )
        # окремі фото кроків (перші кілька + фінал)
        step_dir = os.path.join(ROOT, "figures", f"part2_{sc['id']}", "steps")
        if os.path.isdir(step_dir):
            steps = sorted(f for f in os.listdir(step_dir) if f.endswith(".png"))
            add_para(
                doc,
                f"Покрокові рисунки ({len(steps)} окремих зображень мережі):",
                size=12,
            )
            for name in steps:
                add_picture(
                    doc, os.path.join(step_dir, name), width_cm=12.0,
                    caption=f"{sc['id']} / {name}",
                )
        final_p = os.path.join(ROOT, "figures", f"part2_{sc['id']}", "final_path.png")
        if os.path.exists(final_p) and not os.path.isdir(step_dir):
            add_picture(doc, final_p, width_cm=12.0,
                        caption=f"Фінальний шлях: {sc['id']}")

    add_picture(
        doc, os.path.join(ROOT, "figures", "paths.png"),
        caption="Рисунок – Порівняння шляхів: базовий, з обходом, після збою каналу",
    )

    # ---- Part 3 ----
    add_heading(doc, "Частина 3. Пояснення маршруту", 1)
    add_para(doc, "Журнал контролера, переданий моделі:", bold=True)
    add_code(doc, part3["journal"], max_chars=8000)
    add_para(doc, "Пояснення моделі:", bold=True)
    add_para(doc, part3["explanation"], size=12)
    add_para(doc, "Виявлені неточності:", bold=True)
    if part3.get("inaccuracies"):
        for i in part3["inaccuracies"]:
            add_para(doc, f"— [{i['type']}] {i['detail']}")
    else:
        add_para(doc, "Неточностей не виявлено.")
    add_para(
        doc,
        "Факти: шлях оптимальний (31 мс = Dijkstra); вузьке місце "
        f"{part3['facts'].get('bottleneck')} має пропускну здатність "
        "400 МБ/с (не затримку). Модель помилково назвала BW «затримкою».",
    )

    # ---- Part 4 ----
    add_heading(doc, "Частина 4. Критична оцінка", 1)
    add_para(
        doc,
        "Виконано 11 NL-запитів (коректні, неоднозначні, неможливі) на двох "
        "квантуваннях Q4_K_M та Q8_0.",
    )
    for model, block in part4["models"].items():
        add_para(
            doc,
            f"Таблиця – {block['label']} ({model}): "
            f"{block['correct_count']}/{block['total']} "
            f"(accuracy={block['accuracy']})",
            bold=True,
        )
        rows = []
        for r in block["rows"]:
            rows.append([
                r["id"],
                (r["query"][:48] + "…") if len(r["query"]) > 48 else r["query"],
                ",".join(r["expected_tools"]),
                ",".join(r["actual_tools"]),
                "так" if r["correct"] else "ні",
                ",".join(r["error_types"]) or "—",
            ])
        add_table(
            doc,
            ["ID", "Запит", "Очікувані", "Фактичні", "OK?", "Помилка"],
            rows,
        )

    q4 = part4["models"]["qwen2.5:3b"]
    q8 = part4["models"]["qwen2.5:3b-instruct-q8_0"]
    add_para(
        doc,
        f"Порівняння квантувань (UA): Q4_K_M — {q4['accuracy']*100:.1f}% "
        f"правильних відповідей; Q8_0 — {q8['accuracy']*100:.1f}%.",
    )

    lc = part4.get("language_compare") or {}
    if lc.get("pairs"):
        add_heading(doc, "Порівняння української та англійської", 2)
        add_para(
            doc,
            "Додатково (за PDF) порівняно 5 пар однакових намірів українською "
            "та англійською: q01 (шлях), q03 (неіснуючий вузол), q04 (стан "
            "потоку), q05 (телеметрія), q08 (неоднозначний запит).",
        )
        lang_rows = []
        for p in lc["pairs"]:
            for model, m in (p.get("by_model") or {}).items():
                lang_rows.append([
                    p["pair_id"],
                    m["label"],
                    (p["uk_query"][:36] + "…") if len(p["uk_query"]) > 36
                    else p["uk_query"],
                    (p["en_query"][:36] + "…") if len(p["en_query"]) > 36
                    else p["en_query"],
                    "так" if m["uk_correct"] else "ні",
                    "так" if m["en_correct"] else "ні",
                    "так" if m["same_outcome"] else "ні",
                ])
        add_table(
            doc,
            ["Пара", "Модель", "UA запит", "EN запит", "UA OK", "EN OK", "Згода"],
            lang_rows,
        )
        for model, s in (lc.get("summary") or {}).items():
            add_para(
                doc,
                f"{s['label']}: UA {s['uk_correct']}/{s['pairs']} "
                f"({s['uk_accuracy']*100:.1f}%), EN {s['en_correct']}/{s['pairs']} "
                f"({s['en_accuracy']*100:.1f}%), згода результатів "
                f"{s['same_outcome']}/{s['pairs']} ({s['agreement']*100:.1f}%).",
            )

    add_heading(doc, "Висновок", 1)
    add_para(
        doc,
        "У частині 1 показано, що квантування зменшує розмір і RAM та "
        "прискорює генерацію на CPU. У частині 2 реалізовано intent-based "
        "NL-інтерфейс: модель викликає інструменти, а шлях встановлює "
        "детермінований knowledge-based контролер (у т.ч. з avoid і після "
        "simulate_link_failure).",
    )
    add_para(
        doc,
        "Частина 3 виявила типову галюцинацію/плутанину одиниць у текстовому "
        "поясненні навіть при коректному журналі. Частина 4 показала високу "
        "частку правильних tool-послідовностей, але й чутливість до "
        "неоднозначних формулювань.",
    )
    add_para(
        doc,
        "Висновок щодо ролей: LLM доцільно доручати розбір наміру оператора "
        "та оркестрацію API; побудову маршруту, облік топології, затримок і "
        "пропускної здатності, а також перевірку результату слід залишати "
        "детермінованим алгоритмам контролера — це зменшує ризик небезпечних "
        "галюцинацій у мережевому керуванні.",
    )

    out_path = os.path.join(ROOT, "Lab5_report.docx")
    doc.save(out_path)
    print("Звіт збережено:", out_path)


if __name__ == "__main__":
    main()

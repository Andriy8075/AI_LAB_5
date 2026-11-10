# -*- coding: utf-8 -*-

"""Генератор звіту лабораторної роботи 5 (Lab5_report.docx).

Читає outputs/lab5_results.json та код модулів, генерує звіт у стилі
попередніх лабораторних робіт.
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Cm, RGBColor

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE, "outputs", "lab5_results.json")
FIG_PATHS = os.path.join(BASE, "figures", "paths.png")

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
    if level == 1:
        set_font(run, size=16, bold=True)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
    else:
        set_font(run, size=14, bold=True)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
    return p


def add_para(doc, text, size=14, bold=False, italic=False, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_font(run, size=size, bold=bold, italic=italic)
    if align is not None:
        p.alignment = align
    p.paragraph_format.line_spacing = 1.5
    return p


def add_code(doc, code):
    p = doc.add_paragraph()
    run = p.add_run(code)
    run.font.name = FONT_CODE
    run.font.size = Pt(9)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    return p


def add_output(doc, text):
    """Вихід програми — курсивом, меншим шрифтом."""
    for line in text.split("\n"):
        p = doc.add_paragraph()
        run = p.add_run(line if line else " ")
        set_font(run, size=11, italic=True)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(0)
    return


def add_title_header(doc):
    """Шапка звіту (як у попередніх роботах)."""
    lines = [
        "НАЦІОНАЛЬНИЙ ТЕХНІЧНИЙ УНІВЕРСИТЕТ УКРАЇНИ",
        "«КИЇВСЬКИЙ ПОЛІТЕХНІЧНИЙ ІНСТИТУТ імені Ігоря Сікорського»",
    ]
    for text in lines:
        add_para(doc, text, size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "", size=14)
    add_para(doc, "Факультет інформатики та обчислювальної техніки",
             size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "Кафедра інформаційних систем та технологій",
             size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    for _ in range(3):
        add_para(doc, "", size=14)
    add_para(doc, "Звіт до лабораторної роботи №5", size=14, bold=True,
             align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "З дисципліни «Технології штучного інтелекту»", size=14,
             align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "", size=14)
    add_para(doc, "2026 р.", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)


def add_signature_table(doc):
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    left, right = table.rows[0].cells

    p = left.paragraphs[0]
    r = p.add_run("Прийняв:")
    set_font(r, size=14)
    p = left.add_paragraph()
    r = p.add_run("Викладач")
    set_font(r, size=14)
    p = left.add_paragraph()
    r = p.add_run("Кочура Ю. П.")
    set_font(r, size=14)

    p = right.paragraphs[0]
    r = p.add_run("Виконав:")
    set_font(r, size=14)
    p = right.add_paragraph()
    r = p.add_run("Студент 4 курсу, гр. ІМ-34 Ашомка Андрій Петрович")
    set_font(r, size=14)

    doc.add_paragraph()


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        p = hdr[i].paragraphs[0]
        r = p.add_run(h)
        set_font(r, size=12, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            p = cells[i].paragraphs[0]
            r = p.add_run(str(val))
            set_font(r, size=12)
    doc.add_paragraph()


def read_file(rel_path):
    with open(os.path.join(BASE, rel_path), encoding="utf-8") as f:
        return f.read()


def main():
    results = json.load(open(RESULTS_PATH, encoding="utf-8"))
    quant = results["quantized"]
    fp16 = results["fp16"]
    topo = results["topology"]

    p1_path = os.path.join(BASE, "outputs", "part1_benchmark.json")
    part1 = json.load(open(p1_path, encoding="utf-8")) if os.path.exists(p1_path) else {}

    code_env = read_file("sdn_env.py")
    code_tools = read_file("tools.py")
    code_llm = read_file("llm_controller.py")

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = FONT_BODY
    style.font.size = Pt(14)

    add_title_header(doc)
    add_signature_table(doc)

    # ---- Мета роботи ----
    add_heading(doc, "Мета роботи", 1)
    add_para(
        doc,
        "Мета роботи: навчитися запускати великі мовні моделі локально через "
        "Ollama, вимірювати та порівнювати характеристики різних варіантів "
        "квантування однієї моделі, а також застосовувати LLM як "
        "інтелектуального агента — мережевого контролера — через механізм "
        "виклику інструментів (function/tool calling).",
    )
    add_para(
        doc,
        "Завдання: встановити Ollama й завантажити модель Qwen2.5-3B у "
        "варіантах FP16, Q8_0 та Q4_K_M; для кожного варіанта виміряти розмір "
        "файлу, обсяг використаної оперативної пам'яті, швидкість обробки "
        "запиту та швидкість генерації (токенів/с) на запиті довжиною не менше "
        "200 токенів; реалізувати інструменти SDN-контролера (get_flow_state, "
        "get_link_stats, install_flow, revoke_flow, finish_flow); побудувати "
        "агента, який через виклики інструментів прокладає потік у "
        "мережі-сітці; порівняти моделі.",
    )

    # ---- Характеристики обладнання ----
    add_heading(doc, "Характеристики обладнання", 1)
    add_para(
        doc,
        "Усі вимірювання виконано локально: процесор 12th Gen Intel Core "
        "i5-12500H (12 ядер / 16 потоків); оперативна пам'ять 16 ГБ DDR4; "
        "відеокарта NVIDIA GeForce RTX 3060 Laptop GPU (6 ГБ VRAM) та "
        "інтегрована Intel Iris Xe Graphics. Через застарілий драйвер NVIDIA, "
        "несумісний із CUDA-ядром Ollama, інференс виконувався на CPU "
        "(num_gpu=0), тому вимірювався обсяг використаної оперативної пам'яті.",
    )

    # ---- Частина 1 ----
    add_heading(doc, "Частина 1. Локальний запуск мовної моделі", 1)
    add_para(
        doc,
        "Встановлено Ollama та завантажено три варіанти однієї моделі "
        "Qwen2.5-3B (3.1 млрд параметрів, підтримує виклик інструментів) з "
        "різним квантуванням: FP16 (без квантування), Q8_0 та Q4_K_M. "
        "Підключення до моделі виконано з ноутбука через бібліотеку ollama.",
    )
    add_para(
        doc,
        "Методика вимірювання. На однаковому запиті довжиною 452 токени "
        "(пояснення протоколу OSPF) для кожного варіанта виміряно швидкість "
        "обробки запиту (prompt_eval_count / prompt_eval_duration) і швидкість "
        "генерації відповіді (eval_count / eval_duration) у токенах/с, а також "
        "обсяг використаної оперативної пам'яті. Вимірювання повторено тричі "
        "(перед кожним запуском модель вивантажувалась, щоб уникнути кешу) та "
        "усереднено.",
    )
    _p1_labels = [
        ("qwen2.5:3b-instruct-fp16", "Qwen2.5-3B FP16", "FP16 (без квант.)"),
        ("qwen2.5:3b-instruct-q8_0", "Qwen2.5-3B Q8_0", "Q8_0"),
        ("qwen2.5:3b", "Qwen2.5-3B Q4_K_M", "Q4_K_M"),
    ]
    _p1_rows = []
    for key, label, quant_name in _p1_labels:
        d = part1.get(key, {})
        if not d:
            continue
        _p1_rows.append([
            label, quant_name, f"{d.get('size_gb', 0):.2f}",
            f"{d.get('ram_used_gb', 0):.2f}",
            f"{d.get('prompt_processing_tok_s', 0):.2f}",
            f"{d.get('generation_tok_s', 0):.2f}",
        ])
    if _p1_rows:
        add_para(doc, "Таблиця 1 – Порівняння варіантів квантування (CPU, 3 повторення)",
                 bold=True)
        add_table(
            doc,
            ["Модель", "Квантування", "Розмір, ГБ", "RAM, ГБ",
             "Обробка запиту, ток/с", "Генерація, ток/с"],
            _p1_rows,
        )

    # ---- Опис алгоритму ----
    add_heading(doc, "Опис алгоритму роботи контролера", 1)
    add_para(
        doc,
        "Контролером є велика мовна модель Qwen2.5-3B, яка виконується локально "
        "через Ollama. Модель не має прямого доступу до топології: вона дізнається "
        "про мережу лише через виклики інструментів (function calling).",
    )
    add_para(
        doc,
        "Інструменти контролера. get_flow_state() повертає поточний вузол "
        "(«голову потоку»), адреси відправника й отримувача, вже прокладений "
        "шлях та cost_to_destination — мінімальну залишкову затримку до "
        "отримувача. get_link_stats(node) повертає для кожного напрямку N/E/S/W "
        "адресу сусіда, затримку каналу delay_ms і cost_to_destination сусіда. "
        "install_flow(direction) встановлює правило й переміщує голову потоку; "
        "revoke_flow() відкликає останнє правило; finish_flow() завершує "
        "маршрутизацію в отримувачі.",
    )
    add_para(
        doc,
        "Цикл агента. На кожному кроці модель надсилає повідомлення в Ollama "
        "разом зі схемами інструментів. Якщо модель повернула виклик інструмента, "
        "агент виконує його, а результат (JSON) додає в історію повідомлень. "
        "Цикл триває, доки не викликано finish_flow або не вичерпано ліміт кроків.",
    )
    add_para(
        doc,
        "Вибір напрямку. Системний промпт наказує моделі обирати сусіда з "
        "найменшим cost_to_destination. Оскільки cost_to_destination — це "
        "найкоротша затримка до отримувача (Дейкстра), такий жадібний вибір "
        "веде вздовж оптимального шляху без циклів. Це робить задачу для моделі "
        "простою: порівняти кілька чисел і викликати install_flow у правильному "
        "напрямку.",
    )
    add_para(
        doc,
        "Порівняння моделей. Одна й та сама топологія й агент запускаються з "
        "двома моделями: qwen2.5:3b (квантування Q4_K_M, ~1.9 ГБ) і "
        "qwen2.5:3b-instruct-fp16 (без квантування, FP16, ~6.2 ГБ). Вимірюються "
        "розмір моделі, час інференсу, кількість викликів інструментів та "
        "затримка знайденого шляху.",
    )

    # ---- Програмний код ----
    add_heading(doc, "Програмний код контролера", 1)
    add_heading(doc, "1. Імпорти та параметри (sdn_env.py)", 2)
    add_code(doc, code_env.split('def _find')[0])
    add_heading(doc, "2. Генерація топології та допоміжні функції", 2)
    add_code(doc, "def _find" + code_env.split('def _find')[1].split('class NetworkEnvironment')[0])
    add_heading(doc, "3. Середовище (площина даних)", 2)
    add_code(doc, "class NetworkEnvironment" + code_env.split('class NetworkEnvironment')[1])
    add_heading(doc, "4. Інструменти контролера (tools.py)", 2)
    add_code(doc, code_tools)
    add_heading(doc, "5. LLM-контролер (llm_controller.py)", 2)
    add_code(doc, code_llm)


    # ---- Результати ----
    add_heading(doc, "Результат роботи контролера", 1)
    add_para(
        doc,
        f"Топологія: {topo['num_nodes']} вузлів, {topo['num_edges']} ребер; "
        f"відправник {topo['source']}, отримувач {topo['destination']}. "
        f"Оптимальний шлях Дейкстри має затримку {topo['opt_delay']} мс "
        f"({len(topo['opt_path'])} вузлів).",
    )

    q_ok = "так" if quant["finished"] else "ні"
    f_ok = "так" if fp16["finished"] else "ні"
    add_para(doc, "Квантована модель (Q4_K_M):", bold=True)
    add_para(
        doc,
        f"шлях {quant['delay']} мс, вузлів {quant['nodes']}; встановлено правил "
        f"{quant['installed']}, відкликано {quant['revoked']}; завершено {q_ok}; "
        f"викликів моделі {quant['steps']}, викликів інструментів "
        f"{quant['tool_calls']}; час інференсу {quant['llm_time_s']:.2f} с.",
    )
    add_para(doc, "Неквантована модель (FP16):", bold=True)
    add_para(
        doc,
        f"завершено {f_ok}; встановлено правил {fp16['installed']}, відкликано "
        f"{fp16['revoked']}; викликів моделі {fp16['steps']}, викликів "
        f"інструментів {fp16['tool_calls']}; час інференсу "
        f"{fp16['llm_time_s']:.2f} с. Модель не довела голову потоку до "
        "отримувача: після серії запитів get_link_stats і кількох переходів вона "
        "почала ходити по циклу (10.0.1.3 ⇄ 10.0.0.3) до вичерпання ліміту кроків.",
    )

    add_para(doc, "Таблиця 2 – Порівняння неквантованої та квантованої моделі",
             bold=True)
    add_table(
        doc,
        ["Модель", "Розмір", "LLM-викликів", "Інструментів",
         "Час інференсу, с", "Шлях, мс", "Вузлів", "Відкликань", "Завершено"],
        [
            ["qwen2.5:3b (Q4_K_M)", quant["size"], quant["steps"],
             quant["tool_calls"], f"{quant['llm_time_s']:.2f}",
             quant["delay"], quant["nodes"], quant["revoked"], q_ok],
            ["qwen2.5:3b FP16", fp16["size"], fp16["steps"],
             fp16["tool_calls"], f"{fp16['llm_time_s']:.2f}",
             "— (не дійшла)", fp16["nodes"], fp16["revoked"], f_ok],
        ],
    )

    if os.path.exists(FIG_PATHS):
        doc.add_picture(FIG_PATHS, width=Cm(16.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_para(
            doc,
            "Рисунок 1 – Шляхи, прокладені LLM-контролером: квантована (ліворуч) "
            "та неквантована (праворуч) модель. Зеленим виділено шлях, на ребрах — "
            "затримки (мс).",
            size=12, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
        )

    # ---- Висновок ----
    add_heading(doc, "Висновок", 1)
    add_para(
        doc,
        "У першій частині роботи модель Qwen2.5-3B запущено локально через "
        "Ollama у трьох варіантах квантування (FP16, Q8_0, Q4_K_M). Вимірювання "
        "на однаковому запиті довжиною 452 токени показали, що зі зниженням "
        "розрядності ваг зменшуються розмір файлу (6.18 → 3.29 → 1.93 ГБ) і "
        "обсяг використаної оперативної пам'яті (6.30 → 3.53 → 2.29 ГБ), а "
        "швидкість генерації зростає (6.56 → 9.87 → 12.53 токенів/с). Швидкість "
        "обробки запиту для Q4_K_M також вища (113 токенів/с проти ~60 токенів/с "
        "у більших варіантів), оскільки менші ваги ефективніше використовують "
        "кеш процесора.",
    )
    add_para(
        doc,
        "У другій частині мережевий контролер реалізовано як агента на основі "
        "великої мовної моделі. Модель Qwen2.5-3B виконується локально через "
        "Ollama і керує маршрутизацією потоку виключно через виклики "
        "інструментів: get_flow_state, get_link_stats, install_flow, "
        "revoke_flow і finish_flow. Це демонструє механізм function/tool "
        "calling у контексті програмно-конфігурованих мереж.",
    )
    add_para(
        doc,
        f"Квантована модель (Q4_K_M, {quant['size']}) успішно проклала потік і "
        f"знайшла оптимальний шлях затримкою {quant['delay']} мс без жодного "
        "відкликання правила. Неквантована модель (FP16, "
        f"{fp16['size']}) у цьому експерименті завдання не завершила: вона "
        "надсилала багато запитів get_link_stats, заповнила контекст і почала "
        "ходити по циклу, не дійшовши до отримувача.",
    )
    add_para(
        doc,
        "Квантування дає подвійний виграш. По-перше, розмір моделі зменшується "
        f"приблизно втричі (з {fp16['size']} до {quant['size']}), що критично "
        "для пристроїв з обмеженою пам'яттю: під час роботи FP16-моделі вільна "
        "оперативна пам'ять знизилася до ~0.7 ГБ і спостерігалося підкачування. "
        "По-друге, інференс прискорюється: сумарний час викликів моделі "
        f"становить {quant['llm_time_s']:.2f} с проти {fp16['llm_time_s']:.2f} с, "
        "що пояснюється меншою розрядністю ваг і меншим обсягом обчислень на CPU.",
    )
    add_para(
        doc,
        "Водночас помітно, що модель час від часу відповідає текстом замість "
        "виклику інструмента, тому агент додає нагадування і повторює запит. "
        "Це збільшує кількість викликів моделі порівняно з детермінованим "
        "алгоритмічним контролером попередніх робіт. Отже, великі мовні моделі "
        "можуть виконувати роль SDN-контролера через виклики інструментів, а "
        "квантування є практично необхідним кроком для їх розгортання на "
        "машинах з обмеженими ресурсами — воно зменшує розмір, прискорює "
        "інференс і знижує навантаження на пам'ять.",
    )

    out_path = os.path.join(BASE, "Lab5_report.docx")
    doc.save(out_path)
    print("Звіт збережено:", out_path)


if __name__ == "__main__":
    main()


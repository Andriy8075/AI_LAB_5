# Лабораторна робота 5 — Великі мовні моделі

NL-інтерфейс SDN-контролера через Ollama tool calling (частини 1–4 за PDF).

## Середовище

- Ollama + моделі `qwen2.5:3b` (Q4_K_M), `qwen2.5:3b-instruct-q8_0`, опційно FP16
- Python: `ollama`, `psutil`, `pandas`, `networkx`, `matplotlib`, `nbformat`, `python-docx`

Інференс на цьому ПК — **CPU** (`num_gpu=0`).

## Демонстрація / захист

**Єдиний файл: `lab5.ipynb`** — повний код модулів + усі запуски моделей частин 1–4
і збережені виводи (tools, текст Qwen, графи).

У клітинках: `LIVE_PART1` / `LIVE` / `LIVE_PART3` / `LIVE_PART4`
(`False` = показати `outputs/`, `True` = знову викликати Ollama).

Опційно (перегенерація артефактів поза ноутбуком):
```powershell
cd c:\Users\Admin\projects\ai_labs\lab5
python scripts/build_notebook.py   # пересобрати lab5.ipynb з outputs/
python scripts/build_report.py     # Lab5_report.docx
```

Читабельні відповіді частини 2 (додатково):
- `outputs/part2_chat.html` — відкрити в браузері
- `outputs/part2_chat.md` — Markdown

## Структура

```
lab5/
├── lab5.ipynb
├── Lab5_report.docx
├── Lab5 (2).pdf
├── benchmark_part1.py
├── src/
│   ├── sdn_env.py          # 5×5, delay+BW, failure
│   ├── knowledge.py        # KB + install_full_path(avoid)
│   ├── tools.py            # PDF tool API
│   ├── llm_controller.py   # NL-агент
│   └── visualization.py    # окремі PNG на крок
├── scripts/
│   ├── run_part2.py
│   ├── run_part3.py
│   ├── run_part4.py
│   ├── generate_steps.py
│   ├── build_notebook.py
│   └── build_report.py
├── outputs/                # JSON результатів
└── figures/                # topology, paths, part2_*/steps/step_XX.png
```

## Інструменти (частина 2)

`get_flow_state`, `get_link_stats`, `install_rule`, `install_path`, `get_topology`, `simulate_link_failure`.

"""Бенчмарк Частини 1: локальний запуск LLM і вимірювання швидкості.

Для кожного варіанту квантування (FP16, Q8_0, Q4_K_M) вимірює:
  - розмір файлу моделі (ГБ);
  - обсяг використаної оперативної пам'яті під час роботи (ГБ);
  - швидкість обробки запиту (prompt eval, токенів/с);
  - швидкість генерації відповіді (eval, токенів/с);
на однаковому запиті довжиною >= 200 токенів, 3 повторення з усередненням.

Запуск: python benchmark_part1.py
"""

import json
import os
import statistics
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

import ollama
import psutil

MODELS = [
    "qwen2.5:3b-instruct-fp16",
    "qwen2.5:3b-instruct-q8_0",
    "qwen2.5:3b",
]
REPS = 3
NUM_PREDICT = 128
NUM_GPU = 0  # CPU (драйвер NVIDIA застарий для CUDA-ядра Ollama)

# Запит довжиною >= 200 токенів (однаковий для всіх моделей).
PROMPT = (
    "Поясни простими словами, як працює домашній Wi-Fi і що відбувається, "
    "коли відкриваєш сайт у браузері. Розкажи, навіщо потрібні роутер, модем "
    "і IP-адреса, чим відрізняються локальна мережа і інтернет, і що таке DNS. "
    "Опиши шлях запиту від телефону до сервера і назад: Wi-Fi, роутер, "
    "провайдер, сервери в інтернеті. Порівняй дротове підключення і Wi-Fi за "
    "швидкістю, стабільністю та зручністю. Наведи короткий приклад: двоє людей "
    "у квартирі дивляться відео й одночасно завантажують файли — що може "
    "сповільнити мережу і як це покращити. На завершення дай 5 практичних "
    "порад, як зробити домашню мережу безпечнішою."
)


def unload_all():
    for m in MODELS:
        try:
            ollama.chat(model=m, messages=[{"role": "user", "content": " "}],
                        options={"num_predict": 1, "num_gpu": 0}, keep_alive=0)
        except Exception:
            pass
    time.sleep(2)


def model_size_gb(model):
    for m in ollama.list()["models"]:
        if m["model"] == model:
            return m["size"] / 1e9
    return 0.0


def ram_used_gb():
    """Повертає обсяг використаної RAM (total - available) у ГБ."""
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / 1e9


def ollama_ps_size_gb(model):
    for m in ollama.ps().get("models", []):
        if m["model"] == model:
            return m["size"] / 1e9
    return 0.0


def run_once(model):
    resp = ollama.generate(
        model=model,
        prompt=PROMPT,
        options={
            "num_predict": NUM_PREDICT,
            "temperature": 0.0,
            "num_gpu": NUM_GPU,
            "num_ctx": 4096,
        },
    )
    pe_count = resp.get("prompt_eval_count", 0)
    pe_dur = resp.get("prompt_eval_duration", 0) or 0
    ev_count = resp.get("eval_count", 0)
    ev_dur = resp.get("eval_duration", 0) or 0
    return {
        "prompt_tokens": pe_count,
        "prompt_processing_s": pe_dur / 1e9,
        "prompt_tok_s": pe_count / (pe_dur / 1e9) if pe_dur else 0.0,
        "eval_tokens": ev_count,
        "generation_s": ev_dur / 1e9,
        "gen_tok_s": ev_count / (ev_dur / 1e9) if ev_dur else 0.0,
        "total_s": resp.get("total_duration", 0) / 1e9,
    }


def main():
    os.makedirs("outputs", exist_ok=True)
    results = {}

    for model in MODELS:
        print(f"\n=== {model} ===", flush=True)
        unload_all()
        base_ram = ram_used_gb()

        # прогрів (завантаження моделі)
        warm = ollama.generate(model=model, prompt="Привіт",
                               options={"num_predict": 4, "num_gpu": NUM_GPU})
        print(f"  прогрів: {warm.get('total_duration', 0)/1e9:.1f} с, "
              f"prompt_tokens={warm.get('prompt_eval_count')}", flush=True)

        loaded_ram = ram_used_gb()
        ps_size = ollama_ps_size_gb(model)

        runs = []
        for rep in range(1, REPS + 1):
            # Перед кожним вимірюванням вивантажуємо модель, щоб скинути
            # KV-кеш Ollama і виміряти «холодну» обробку запиту (без кешу).
            unload_all()
            r = run_once(model)
            runs.append(r)
            print(f"  реп {rep}: prompt={r['prompt_tok_s']:.2f} ток/с "
                  f"({r['prompt_tokens']} ток), gen={r['gen_tok_s']:.2f} ток/с "
                  f"({r['eval_tokens']} ток)", flush=True)

        avg_prompt = statistics.mean(r["prompt_tok_s"] for r in runs)
        avg_gen = statistics.mean(r["gen_tok_s"] for r in runs)
        avg_prompt_tokens = statistics.mean(r["prompt_tokens"] for r in runs)
        avg_eval_tokens = statistics.mean(r["eval_tokens"] for r in runs)

        results[model] = {
            "size_gb": round(model_size_gb(model), 2),
            "ram_used_gb": round(loaded_ram - base_ram, 2),
            "ollama_ps_size_gb": round(ps_size, 2),
            "prompt_tokens": round(avg_prompt_tokens),
            "prompt_processing_tok_s": round(avg_prompt, 2),
            "eval_tokens": round(avg_eval_tokens),
            "generation_tok_s": round(avg_gen, 2),
            "runs": runs,
        }
        print(f"  ПІДСУМОК: RAM+{loaded_ram-base_ram:.2f} ГБ, "
              f"prompt={avg_prompt:.2f} ток/с, gen={avg_gen:.2f} ток/с",
              flush=True)
        unload_all()

    out = os.path.join("outputs", "part1_benchmark.json")
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nЗбережено:", out)


if __name__ == "__main__":
    main()

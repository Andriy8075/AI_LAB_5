"""Точка входу: послідовно частини 2–4 і збірка артефактів.

Запуск: python scripts/run_lab5.py
Частину 1 (бенчмарк) не перезапускає — див. benchmark_part1.py.
"""

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")


def run(name):
    print(f"\n######## {name} ########", flush=True)
    runpy.run_path(os.path.join(SCRIPTS, name), run_name="__main__")


def main():
    run("run_part2.py")
    run("run_part3.py")
    run("run_part4.py")
    run("generate_steps.py")
    run("build_notebook.py")
    run("build_report.py")


if __name__ == "__main__":
    main()

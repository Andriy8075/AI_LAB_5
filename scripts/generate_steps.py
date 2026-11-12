"""Генерує ОККРЕМІ PNG на кожен крок шляху з part2_scenarios.json."""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import sdn_env
from visualization import save_step_figures, plot_single_path

PART2 = os.path.join(ROOT, "outputs", "part2_scenarios.json")
FIG = os.path.join(ROOT, "figures")


def main():
    if not os.path.exists(PART2):
        print("Немає part2_scenarios.json — спочатку run_part2.py")
        return
    with open(PART2, encoding="utf-8") as f:
        data = json.load(f)
    graph = sdn_env.generate_grid_graph()
    src, dst = sdn_env.SOURCE, sdn_env.DESTINATION

    for sc in data.get("scenarios", []):
        snaps = sc.get("path_snapshots") or []
        if not snaps:
            continue
        out_dir = os.path.join(FIG, f"part2_{sc['id']}", "steps")
        paths = save_step_figures(graph, snaps, src, dst, out_dir, prefix="step")
        print(f"{sc['id']}: {len(paths)} окремих фото → {out_dir}")
        path = (sc.get("flow_state") or {}).get("path") or []
        if path:
            plot_single_path(
                graph, path, src, dst,
                title=f"{sc['id']}",
                filename=os.path.join(FIG, f"part2_{sc['id']}", "final_path.png"),
            )


if __name__ == "__main__":
    main()

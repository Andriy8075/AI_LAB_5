"""Реконструює покрокові знімки стану з lab5_results.json і генерує
графіки покрокової маршрутизації та ASCII-журнали для обох моделей.

Запуск:  python generate_steps.py
Результати: figures/steps_q.png, figures/steps_f.png, outputs/*_ascii.txt
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use("Agg")

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import sdn_env
from visualization import render_state_ascii, plot_step_sequence


def _action_label(action):
    args = action.get("arguments", {})
    if "direction" in args:
        return f"{action['tool']}({args['direction']})"
    if "node" in args:
        return f"{action['tool']}({args['node']})"
    return action["tool"]


def reconstruct(log, source):
    """Відновлює стан мережі (голова, шлях, правила) після кожного кроку."""
    by_step = {}
    for e in log:
        by_step.setdefault(e["step"], []).append(e)
    max_step = max(e["step"] for e in log)
    head = source
    path = [sdn_env.parse_address(source)]
    installed = 0
    revoked = 0
    snapshots = []
    for step in range(1, max_step + 1):
        actions = []
        for e in by_step.get(step, []):
            actions.append({
                "tool": e["tool"], "arguments": e.get("arguments", {}),
                "status": e["status"],
            })
            try:
                result = json.loads(e["result"])
            except Exception:
                result = {}
            if "head" in result:  # це flow_state (install/revoke/get_flow_state)
                head = result["head"]
                path = [tuple(sdn_env.parse_address(a)) for a in result["path"]]
                installed = result.get("installed_rules", installed)
                revoked = result.get("revoked_rules", revoked)
        snapshots.append({
            "step": step, "head": head, "path": path,
            "installed": installed, "revoked": revoked, "actions": actions,
        })
    return snapshots


def ascii_journal(graph, snapshots, source, destination):
    lines = []
    for s in snapshots:
        actions = ", ".join(_action_label(a) for a in s["actions"]) or "—"
        lines.append(
            f"--- Крок {s['step']:>2}: {actions} | голова {s['head']} | "
            f"правил {s['installed']}, відкликано {s['revoked']} ---"
        )
        head = sdn_env.parse_address(s["head"]) if s.get("head") else None
        lines.append(render_state_ascii(graph, s["path"], source, destination, head=head))
        lines.append("")
    return "\n".join(lines)


def main():
    results = json.load(open(os.path.join(ROOT, "outputs", "lab5_results.json"),
                             encoding="utf-8"))
    graph = sdn_env.generate_grid_graph()
    src, dst = sdn_env.SOURCE, sdn_env.DESTINATION

    for key, name in [("quantized", "steps_q"), ("fp16", "steps_f")]:
        snap = reconstruct(results[key]["log"], src)
        journal = ascii_journal(graph, snap, src, dst)
        with open(os.path.join(ROOT, "outputs", f"{name}_ascii.txt"), "w",
                  encoding="utf-8") as f:
            f.write(journal)
        with open(os.path.join(ROOT, "outputs", f"{name}_snapshots.json"), "w",
                  encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
        title = (f"Покрокова маршрутизація — {results[key]['model']} "
                 f"({results[key].get('size', '')})")
        plot_step_sequence(graph, snap, src, dst, title=title,
                           filename=os.path.join(ROOT, "figures", f"{name}.png"))
        print(f"{name}: кроків {len(snap)}, збережено", flush=True)


if __name__ == "__main__":
    main()

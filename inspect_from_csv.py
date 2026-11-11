"""
Look at one map using the results you ALREADY have. No LLM calls, no tokens.

Reads <results>/gpt_standard.csv, gpt_cot.csv, gpt_repe.csv and, for the chosen map,
prints per start/goal pair: operation ratio, waypoint count and validity per prompt.
Draws the map with the A* path (recomputed locally, A* needs no LLM) and a text box
with the stored numbers per prompt. The LLM paths themselves are NOT in the CSV.

Usage (PowerShell, from the llm-astar folder):
    python inspect_from_csv.py --env 10 --results results_gemini
    python inspect_from_csv.py --worst 10 --results results_gemini   # 10 hardest maps
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
import copy
import csv
import json
import os
from collections import defaultdict

import numpy as np
from llmastar.pather import AStar
from llmastar.env.search import env as env_search, plotting as plotting_search

_last = {"path": None, "visited": None}


def _capture_animation(self, path, visited, show, name, filepath):
    _last["path"] = [tuple(p) for p in path] if path else None
    _last["visited"] = visited


plotting_search.Plotting.animation = _capture_animation


def load_rows(folder, llm, prompts):
    """{(env_id, start, goal): {prompt: row}} for finished (status ok) cases."""
    data = defaultdict(dict)
    for pr in prompts:
        path = os.path.join(folder, f"{llm}_{pr}.csv")
        if not os.path.exists(path):
            print(f"(missing {path})")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    data[(int(r["env_id"]), r["start"], r["goal"])][pr] = r
    return data


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def draw(env, start, goal, rows, prompts, out):
    e = env_search.Env(env["range_x"][1], env["range_y"][1],
                       env["horizontal_barriers"], env["vertical_barriers"])
    obstacles = {tuple(c) for c in e.obs_map()}
    query = {"start": start, "goal": goal, "size": [env["range_x"][1], env["range_y"][1]],
             "horizontal_barriers": env["horizontal_barriers"],
             "vertical_barriers": env["vertical_barriers"],
             "range_x": env["range_x"], "range_y": env["range_y"]}
    _last["path"], _last["visited"] = None, None
    AStar().searching(query=copy.deepcopy(query))

    rx, ry = env["range_x"][1], env["range_y"][1]
    grid = np.zeros((ry + 1, rx + 1))
    for x, y in obstacles:
        if 0 <= x <= rx and 0 <= y <= ry:
            grid[y, x] = 1
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.imshow(grid, origin="lower", cmap="Greys", vmin=0, vmax=1.4,
              extent=(-0.5, rx + 0.5, -0.5, ry + 0.5), interpolation="nearest")
    try:
        vis = [(v[0], v[1]) for v in (_last["visited"] or [])
               if isinstance(v, (tuple, list)) and len(v) == 2]
        if vis:
            vx, vy = zip(*vis)
            ax.scatter(vx, vy, s=7, marker="s", color="#9ecae1", alpha=0.6, linewidths=0)
    except TypeError:
        pass
    if _last["path"]:
        px, py = zip(*_last["path"])
        ax.plot(px, py, color="red", lw=1.8, label="A* path")
    ax.scatter(*start, color="green", s=60, zorder=6, label="start")
    ax.scatter(*goal, marker="*", color="blue", s=130, zorder=6, label="goal")
    ax.set_xlim(-0.5, rx + 0.5)
    ax.set_ylim(-0.5, ry + 0.5)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)

    lines = []
    for pr in prompts:
        r = rows.get(pr)
        if not r:
            lines.append(f"{pr:9} (not finished)")
            continue
        orr = to_float(r["op_ratio"])
        lines.append(f"{pr:9} OR={orr * 100:5.0f}%  wpts={r['llm_waypoints']}  valid={r['is_valid']}"
                     if orr is not None else f"{pr:9} n/a")
    ax.set_title(f"Map {env['id']}: {start} -> {goal}   (A* ops = "
                 f"{next(iter(rows.values()))['astar_ops'] if rows else '?'})", fontsize=9)
    fig.text(0.5, 0.01, "\n".join(lines), ha="center", va="bottom", family="monospace", fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset/environment_50_30.json")
    p.add_argument("--results", default="results_gemini")
    p.add_argument("--llm", default="gpt")
    p.add_argument("--prompts", default="standard,cot,repe")
    p.add_argument("--env", type=int, help="map id to inspect")
    p.add_argument("--worst", type=int, help="list the N maps with the highest mean OR")
    p.add_argument("--outdir", default="map_inspect")
    args = p.parse_args()
    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]
    data = load_rows(args.results, args.llm, prompts)

    if args.worst:
        per_env = defaultdict(list)
        for (eid, _, _), by_pr in data.items():
            for r in by_pr.values():
                v = to_float(r["op_ratio"])
                if v is not None:
                    per_env[eid].append(v * 100)
        ranked = sorted(per_env.items(), key=lambda kv: -np.mean(kv[1]))[: args.worst]
        print("Hardest maps (highest mean OR over all prompts and pairs, higher = worse):")
        for eid, v in ranked:
            print(f"  map {eid:4}  mean OR = {np.mean(v):6.1f}%   (n={len(v)})")
        if args.env is None:
            return

    if args.env is None:
        raise SystemExit("Give --env <id> or --worst <N>")

    with open(args.dataset, "r") as f:
        environments = json.load(f)
    env = next((e for e in environments if e.get("id") == args.env), None)
    if env is None:
        raise SystemExit(f"No map with id {args.env}")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\nMap {args.env} from {args.results}")
    print(f"{'pair':>4} {'start':>10} {'goal':>10}  " +
          "  ".join(f"{pr + ' OR/wpts/valid':>24}" for pr in prompts))
    for n, sg in enumerate(env["start_goal"]):
        start, goal = list(sg[0]), list(sg[1])
        key = next((k for k in data if k[0] == args.env and json.loads(k[1]) == start
                    and json.loads(k[2]) == goal), None)
        rows = data.get(key, {}) if key else {}
        cells = []
        for pr in prompts:
            r = rows.get(pr)
            orr = to_float(r["op_ratio"]) if r else None
            cells.append(f"{orr * 100:5.0f}% / {r['llm_waypoints']:>2} / {r['is_valid'][:1]}".rjust(24)
                         if orr is not None else "n/a".rjust(24))
        print(f"{n:>4} {str(start):>10} {str(goal):>10}  " + "  ".join(cells))
        draw(env, start, goal, rows, prompts, os.path.join(args.outdir, f"csv_env{args.env}_pair{n}.png"))
    print(f"\nImages saved in {args.outdir}\\ (csv_env{args.env}_pair<N>.png)")


if __name__ == "__main__":
    main()
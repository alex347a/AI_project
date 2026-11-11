"""
Inspect one map: draw A* and LLM-A* (standard / cot / repe) side by side.

For every start/goal pair of the chosen map it runs A* plus one LLM-A* call per
prompt style, prints a small table (operations, ratio vs A*, waypoints, validity)
and saves a picture per pair:  <outdir>/env<ID>_pair<N>.png

Each picture has one panel per method:
  black      = obstacles          light blue = cells the search visited
  red line   = final path         orange diamonds = the LLM's waypoints
  green dot  = start              blue star = goal

Usage (PowerShell, from the llm-astar folder, API key set as usual):
    python inspect_map.py --env 10
    python inspect_map.py --env 10 --pairs 3,7          # only pair 3 and 7 (0-based)
    python inspect_map.py --env 10 --prompts standard,repe --max-tokens 8000

Each LLM call uses tokens: 1 map with 10 pairs and 3 prompts = 30 calls.
"""
import matplotlib
matplotlib.use("Agg")                      # never open a window
import matplotlib.pyplot as plt

import argparse
import copy
import json
import os

import numpy as np
from llmastar.pather import AStar, LLMAStar
from llmastar.env.search import env as env_search, plotting as plotting_search
from llmastar import model as llm_model

# --------------------------------------------------------------------------
# Capture the path (and visited cells) instead of letting the package draw
# --------------------------------------------------------------------------
_last = {"path": None, "visited": None, "empty": False}


def _capture_animation(self, path, visited, show, name, filepath):
    _last["path"] = [tuple(p) for p in path] if path else None
    _last["visited"] = visited


plotting_search.Plotting.animation = _capture_animation


def install_ask_hooks(max_tokens):
    """Raise the token budget (the package hardcodes 1000) and note empty replies."""
    for name in ("ChatGPT", "Llama3"):
        cls = getattr(llm_model, name, None)
        if cls is None or not hasattr(cls, "ask"):
            continue
        orig = cls.ask

        def make(orig):
            def ask(self, *a, **k):
                if max_tokens and "max_tokens" in k:
                    k["max_tokens"] = max_tokens
                r = orig(self, *a, **k)
                if r is None or not str(r).strip():
                    _last["empty"] = True
                return r
            return ask

        cls.ask = make(orig)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_obstacles(env):
    """Obstacle cells from the package's own Env (same map the solvers use)."""
    e = env_search.Env(env["range_x"][1], env["range_y"][1],
                       env["horizontal_barriers"], env["vertical_barriers"])
    return {tuple(c) for c in e.obs_map()}


def path_is_valid(path, start, goal, obstacles, bounds):
    if not path or len(path) < 2:
        return False
    path = [tuple(p) for p in path]
    if path[0] != tuple(start) or path[-1] != tuple(goal):
        return False
    (x0, x1), (y0, y1) = bounds
    for (x, y), (nx, ny) in zip(path, path[1:]):
        if max(abs(nx - x), abs(ny - y)) != 1:
            return False
    for x, y in path:
        if (x, y) in obstacles or not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
    return True


def visited_cells(visited):
    """Best-effort conversion of whatever the solver recorded into (x, y) pairs."""
    out = []
    try:
        for v in visited:
            if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(c, (int, float)) for c in v):
                out.append((v[0], v[1]))
    except TypeError:
        pass
    return out


def run_one(make_solver, query):
    """Run one solver. Returns a dict with result fields; never raises."""
    _last["path"], _last["visited"], _last["empty"] = None, None, False
    r = {"ok": False, "error": "", "ops": None, "storage": None, "length": None,
         "path": None, "visited": [], "waypoints": [], "empty": False}
    try:
        res = make_solver().searching(query=copy.deepcopy(query))
        r["ops"], r["storage"], r["length"] = res.get("operation"), res.get("storage"), res.get("length")
        targets = res.get("llm_output")
        if targets:
            r["waypoints"] = [tuple(t) for t in targets[1:-1]]
        r["path"] = _last["path"]
        r["visited"] = visited_cells(_last["visited"] or [])
        r["empty"] = _last["empty"]
        r["ok"] = r["path"] is not None
        if not r["ok"]:
            r["error"] = "no path captured"
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
    return r


def draw_panel(ax, title, env, obstacles, query, r, valid):
    rx, ry = env["range_x"][1], env["range_y"][1]
    grid = np.zeros((ry + 1, rx + 1))
    for x, y in obstacles:
        if 0 <= x <= rx and 0 <= y <= ry:
            grid[y, x] = 1
    ax.imshow(grid, origin="lower", cmap="Greys", vmin=0, vmax=1.4,
              extent=(-0.5, rx + 0.5, -0.5, ry + 0.5), interpolation="nearest")
    if r["visited"]:
        vx, vy = zip(*r["visited"])
        ax.scatter(vx, vy, s=7, marker="s", color="#9ecae1", alpha=0.6, linewidths=0)
    if r["path"]:
        px, py = zip(*r["path"])
        ax.plot(px, py, color="red", lw=1.8)
    for i, (wx, wy) in enumerate(r["waypoints"], 1):
        ax.scatter([wx], [wy], marker="D", s=45, color="orange", edgecolor="black", zorder=5)
        ax.annotate(str(i), (wx, wy), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.scatter(*query["start"], color="green", s=50, zorder=6)
    ax.scatter(*query["goal"], marker="*", color="blue", s=110, zorder=6)
    ax.set_xlim(-0.5, rx + 0.5)
    ax.set_ylim(-0.5, ry + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def main():
    p = argparse.ArgumentParser(description="Run and draw one map")
    p.add_argument("--dataset", default="dataset/environment_50_30.json")
    p.add_argument("--env", type=int, required=True, help="map id (the 'id' field), e.g. 10")
    p.add_argument("--pairs", default="all", help='"all" or 0-based list, e.g. "0,3,7"')
    p.add_argument("--llm", default="gpt")
    p.add_argument("--prompts", default="standard,cot,repe")
    p.add_argument("--max-tokens", type=int, default=4000,
                   help="token budget per LLM call (package default 1000 can give empty replies)")
    p.add_argument("--outdir", default="map_inspect")
    args = p.parse_args()

    with open(args.dataset, "r") as f:
        environments = json.load(f)
    env = next((e for e in environments if e.get("id") == args.env), None)
    if env is None:
        raise SystemExit(f"No map with id {args.env} in {args.dataset}")

    install_ask_hooks(args.max_tokens)
    os.makedirs(args.outdir, exist_ok=True)
    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]
    obstacles = get_obstacles(env)
    bounds = (tuple(env["range_x"]), tuple(env["range_y"]))

    pairs = list(range(len(env["start_goal"])))
    if args.pairs != "all":
        pairs = [int(i) for i in args.pairs.split(",")]

    for n in pairs:
        sg = env["start_goal"][n]
        start, goal = list(sg[0]), list(sg[1])
        query = {
            "start": start, "goal": goal,
            "size": [env["range_x"][1], env["range_y"][1]],
            "horizontal_barriers": env["horizontal_barriers"],
            "vertical_barriers": env["vertical_barriers"],
            "range_x": env["range_x"], "range_y": env["range_y"],
        }
        print(f"\n=== Map {args.env}, pair {n}: {start} -> {goal} ===")

        results = [("A*", run_one(lambda: AStar(), query))]
        for pr in prompts:
            results.append((f"LLM-A* {pr}", run_one(lambda pr=pr: LLMAStar(llm=args.llm, prompt=pr), query)))

        a_ops = results[0][1]["ops"]
        fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 3.6))
        axes = np.atleast_1d(axes)
        print(f"{'method':18} {'ops':>6} {'ratio':>7} {'length':>8} {'wpts':>5}  valid  note")
        for ax, (name, r) in zip(axes, results):
            valid = r["ok"] and path_is_valid(r["path"], start, goal, obstacles, bounds)
            ratio = f"{r['ops'] / a_ops * 100:.0f}%" if (r["ops"] and a_ops) else "n/a"
            note = r["error"] or ("EMPTY REPLY -> fell back to [start, goal]" if r["empty"] else "")
            ln = f"{r['length']:.1f}" if r["length"] else "n/a"
            print(f"{name:18} {str(r['ops']):>6} {ratio:>7} {ln:>8} {len(r['waypoints']):>5}  "
                  f"{str(valid):5}  {note}")
            if name != "A*":
                print(f"{'':18} waypoints: {r['waypoints']}")
            title = f"{name}\nops={r['ops']} ({ratio})  wpts={len(r['waypoints'])}  valid={valid}"
            if note:
                title += f"\n{note[:60]}"
            draw_panel(ax, title, env, obstacles, query, r, valid)

        fig.suptitle(f"Map {args.env}, pair {n}: {start} -> {goal}", fontsize=10)
        fig.tight_layout()
        out = os.path.join(args.outdir, f"env{args.env}_pair{n}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
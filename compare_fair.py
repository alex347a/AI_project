"""
compare_fair.py - apples-to-apples comparison. Run beside main.py:   python compare_fair.py 20

Every algorithm gets the SAME map, the SAME cached Gemini plan, and runs on the SAME flat Grid:

  A*        plain A* (octile h)
  LLM*      original LLM-A* re-implemented on the Grid (suboptimal by design)
  Bi0       optimal bidirectional A*, no LLM at all
  BiG       Bi + LLM guide only
  BiI       Bi + LLM incumbent bound only
  BiF       full Bi LLM-A* (incumbent + guide)
  orig      the real shapely LLMAStar, as published (parity check for LLM*, not for speed)

Fairness rules
* "work" counts every expansion: main search + incumbent hop A* + guide BFS cells.
  "search" is the main search alone. LLM-A*/A* have work == search.
* Time: best of REPEATS runs, no tracemalloc, no GC, search only (map/Grid build and the Gemini
  call are outside the clock for every algorithm; incumbent/guide building is INSIDE the clock).
* Memory: one separate run under tracemalloc; extra peak Python memory (MiB).
"""
import contextlib
import copy
import gc
import io
import json
import random
import statistics as st
import sys
import time
import tracemalloc

from dotenv import load_dotenv
load_dotenv()  # must run before the llmastar imports

from llmastar.env.search import plotting
from llmastar.pather import AStar, LLMAStar
from llmastar.pather.llm_a_star.fair_core import (Grid, grid_astar, grid_llm_astar,
                                                  solve_counted)

plotting.Plotting.animation = lambda *a, **k: None

N_MAPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
W, H = 51, 31
CACHE_FILE = "llm_cache.json"
SEED = 5
REPEATS = 7
MIB = 1024 * 1024

try:
    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)
except (OSError, json.JSONDecodeError):
    cache = {}


def attach_cache(algo):
    real_ask = algo.model.ask

    def ask(prompt, *args, **kwargs):
        if prompt in cache:
            return cache[prompt]
        answer = real_ask(prompt, *args, **kwargs)
        if answer and answer.strip():
            cache[prompt] = answer
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        return answer

    algo.model.ask = ask


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def random_query(rng):
    hb, vb = [], []
    for _ in range(rng.randint(3, 10)):
        if rng.random() < 0.5:
            a = rng.randint(0, W - 10)
            hb.append([rng.randint(3, H - 4), a, min(W, a + rng.randint(8, 28))])
        else:
            a = rng.randint(0, H - 8)
            vb.append([rng.randint(3, W - 4), a, min(H, a + rng.randint(6, 18))])

    def on_wall(x, y):
        return any(y == h[0] and h[1] <= x <= h[2] for h in hb) or \
               any(x == v[0] and v[1] <= y <= v[2] for v in vb)

    while True:
        s = (rng.randint(2, W - 3), rng.randint(2, H - 3))
        g = (rng.randint(2, W - 3), rng.randint(2, H - 3))
        if not on_wall(*s) and not on_wall(*g) and abs(s[0] - g[0]) + abs(s[1] - g[1]) >= 20:
            break
    return {"start": list(s), "goal": list(g), "size": [W, H], "horizontal_barriers": hb,
            "vertical_barriers": vb, "range_x": [0, W], "range_y": [0, H]}


def get_plan(query):
    """Run only the LLM part of LLMAStar (cached). Returns the planner holding the target list."""
    planner = LLMAStar(llm="gpt", prompt="standard")
    attach_cache(planner)
    with quiet():
        data = planner._parse_query(copy.deepcopy(query))
        planner._initialize_parameters(data)
        planner._initialize_llm_paths()
    return planner


def measure(fn):
    """Return (result, best_ms, extra_peak_mib)."""
    result = fn()                                  # warm-up and the result we report
    gc.collect()
    gc.disable()
    try:
        times = []
        for _ in range(REPEATS):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
    finally:
        gc.enable()
    tracemalloc.start()
    tracemalloc.reset_peak()
    base, _ = tracemalloc.get_traced_memory()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, min(times) * 1000, max(0, peak - base) / MIB


def make_algos(grid, start, goal, target_list):
    wps = grid.waypoint_indices(target_list[1:-1])
    tl = [grid.index(*p) for p in target_list]

    return {
        "A*": lambda: grid_astar(grid, start, goal),
        "LLM*": lambda: grid_llm_astar(grid, start, goal, tl),
        "Bi0": lambda: solve_counted(grid, start, goal, (), use_incumbent=False, use_guide=False),
        "BiG": lambda: solve_counted(grid, start, goal, wps, use_incumbent=False, use_guide=True),
        "BiI": lambda: solve_counted(grid, start, goal, wps, use_incumbent=True, use_guide=False),
        "BiF": lambda: solve_counted(grid, start, goal, wps, use_incumbent=True, use_guide=True),
    }


OPTIMAL = {"A*", "Bi0", "BiG", "BiI", "BiF"}   # these must equal the reference length
NAMES = ["A*", "LLM*", "Bi0", "BiG", "BiI", "BiF"]

rng = random.Random(SEED)
res = {n: [] for n in NAMES + ["orig"]}
parity_len = parity_sq = parity_n = 0
n_no_plan = 0
rows = 0
attempts = 0

print("per-map: path length / best (1.000 = shortest).   orig = shapely original as published")
print(f"{'map':>3} {'best':>8} | " + " ".join(f"{n:>6}" for n in NAMES + ["orig"]) + " | waypoints")

while rows < N_MAPS and attempts < N_MAPS * 5:
    attempts += 1
    query = random_query(rng)
    try:
        with quiet():
            best = AStar().searching(query=copy.deepcopy(query), filepath="ref.png")["length"]
    except Exception:
        continue

    try:
        planner = get_plan(query)
    except Exception as exc:
        print(f"    plan failed ({type(exc).__name__}: {exc}); map skipped")
        continue

    grid = Grid(planner.range_x, planner.range_y, planner.horizontal_barriers,
                planner.vertical_barriers, planner.u_set)
    start, goal = grid.index(*planner.s_start), grid.index(*planner.s_goal)
    algos = make_algos(grid, start, goal, planner.target_list)

    line = {}
    for name in NAMES:
        r, ms, mib = measure(algos[name])
        if r is None:
            line[name] = None
            continue
        line[name] = r["cost"] / best
        res[name].append({"ratio": r["cost"] / best, "work": r["work"], "search": r["expanded"],
                          "ms": ms, "mib": mib})
        if name in OPTIMAL and abs(r["cost"] - best) > 1e-6:
            print(f"    !! {name} length {r['cost']:.6f} != reference {best:.6f}")

    # parity check: does the Grid re-implementation of LLM-A* match the real shapely one?
    orig_ratio = None
    try:
        orig = LLMAStar(llm="gpt", prompt="standard")
        attach_cache(orig)
        t = time.perf_counter()
        with quiet():
            o = orig.searching(query=copy.deepcopy(query), filepath="orig.png")
        o_ms = (time.perf_counter() - t) * 1000
        o_sq = len(set(orig.CLOSED))
        orig_ratio = o["length"] / best
        res["orig"].append({"ratio": orig_ratio, "work": o_sq, "search": o_sq, "ms": o_ms, "mib": None})
        parity_n += 1
        if line["LLM*"] is not None:
            parity_len += abs(o["length"] - line["LLM*"] * best) < 1e-6
            parity_sq += o_sq == res["LLM*"][-1]["search"]
    except Exception as exc:
        print(f"    orig failed: {type(exc).__name__}: {exc}")

    rows += 1
    n_wp = max(0, len(planner.target_list) - 2)
    n_no_plan += n_wp == 0
    cells = [f"{line[n]:6.3f}" if line.get(n) is not None else f"{'-':>6}" for n in NAMES]
    cells.append(f"{orig_ratio:6.3f}" if orig_ratio is not None else f"{'-':>6}")
    print(f"{rows:3d} {best:8.3f} | " + " ".join(cells) + f" | {n_wp}")

print(f"\n=== Summary over {rows} maps ({n_no_plan} with no usable AI waypoints) ===")
print(f"{'algo':<6} {'shortest':>9} {'extra len':>10} {'work':>8} {'search':>8} {'ms':>8} {'MiB':>7}")
for name in NAMES + ["orig"]:
    rs = res[name]
    if not rs:
        continue
    short = sum(1 for r in rs if r["ratio"] < 1 + 1e-9)
    mib = f"{st.mean(r['mib'] for r in rs):7.2f}" if rs[0]["mib"] is not None else f"{'-':>7}"
    print(f"{name:<6} {short:>4}/{len(rs):<4} {100 * (st.mean(r['ratio'] for r in rs) - 1):9.1f}% "
          f"{st.mean(r['work'] for r in rs):8.0f} {st.mean(r['search'] for r in rs):8.0f} "
          f"{st.mean(r['ms'] for r in rs):8.2f} {mib}")

print(f"\nParity (Grid LLM-A* vs shapely original): same length on {parity_len}/{parity_n} maps, "
      f"same unique-squares on {parity_sq}/{parity_n}.")
print("work = search + incumbent hop expansions + guide-BFS cells. Time excludes the Gemini call;\n"
      "Grid/map building is outside the clock for all grid algorithms. Memory = extra peak Python memory.")
print("What the LLM adds to optimal search = compare Bi0 against BiG / BiI / BiF.")
print("What optimality costs vs the original heuristic = compare LLM* against Bi0 / BiF.")

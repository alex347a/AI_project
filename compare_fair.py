"""
compare_fair.py - apples-to-apples comparison. Run beside main.py:   python compare_fair.py 20

Options: --width --height --barrier-scale --seed --radius --repeats --skip-orig --dataset --pairs --skip   (see --help)
Dataset example:  python compare_fair.py 30 --dataset path\\to\\maps.json --skip-orig
Big-map example:  python compare_fair.py 20 --width 201 --height 121 --barrier-scale 4 --skip-orig
New map sizes make new Gemini prompts, so the first run of each configuration calls the API.

Every algorithm gets the SAME map, the SAME cached Gemini plan, and runs on the SAME flat Grid:

  A*        plain A* (octile h)
  LLM*      original LLM-A* re-implemented on the Grid (suboptimal by design)
  U0        optimal unidirectional A*, no LLM at all (A* run through the same pipeline)
  UG        U + LLM guide only
  UI        U + LLM incumbent bound only
  UF        full unidirectional LLM-A* (incumbent + guide)
  orig      the real shapely LLMAStar, as published (parity check for LLM*, not for speed)

Fairness rules
* "work" counts every expansion: main search + incumbent hop A* + guide BFS cells.
  "search" is the main search alone. LLM-A*/A* have work == search.
* Time: best of REPEATS runs, no tracemalloc, no GC, search only (map/Grid build and the Gemini
  call are outside the clock for every algorithm; incumbent/guide building is INSIDE the clock).
* Memory: one separate run under tracemalloc; extra peak Python memory (MiB).
"""
import argparse
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

ap = argparse.ArgumentParser(description="Fair comparison of A*, LLM-A* and unidirectional LLM-A* variants.")
ap.add_argument("n_maps", nargs="?", type=int, default=10, help="number of maps (default 10)")
ap.add_argument("--width", type=int, default=51, help="map width (default 51)")
ap.add_argument("--height", type=int, default=31, help="map height (default 31)")
ap.add_argument("--barrier-scale", type=float, default=1.0,
                help="multiplies the NUMBER of barriers (lengths already scale with map size)")
ap.add_argument("--seed", type=int, default=5, help="map RNG seed (default 5)")
ap.add_argument("--radius", type=int, default=2, help="guide radius for UG/UF (default 2)")
ap.add_argument("--repeats", type=int, default=7, help="timing repeats per algorithm (default 7)")
ap.add_argument("--skip-orig", action="store_true",
                help="skip the slow shapely original (do this on big maps)")
ap.add_argument("--dataset", metavar="JSON", help="use maps from a JSON file instead of random ones "
                "(list of query dicts: start, goal, horizontal_barriers, vertical_barriers, range_x, range_y)")
ap.add_argument("--pairs", type=int, default=1,
                help="with --dataset: start/goal pairs to use per map (default 1; the dataset has 10)")
ap.add_argument("--skip", type=int, default=0, help="with --dataset: skip the first K maps")
ARGS = ap.parse_args()

N_MAPS = ARGS.n_maps
W, H = ARGS.width, ARGS.height
BARRIER_SCALE = ARGS.barrier_scale
SEED = ARGS.seed
RADIUS = ARGS.radius
REPEATS = ARGS.repeats
SKIP_ORIG = ARGS.skip_orig
CACHE_FILE = "llm_cache.json"
DATASET = ARGS.dataset
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
    # barrier lengths and the minimum start-goal distance scale with map size; at the default
    # 51x31 / scale 1.0 this makes exactly the same RNG calls (and maps) as before
    sx, sy = W / 51, H / 31
    hb, vb = [], []
    n_barriers = max(1, round(rng.randint(3, 10) * BARRIER_SCALE))
    for _ in range(n_barriers):
        if rng.random() < 0.5:
            a = rng.randint(0, W - 10)
            hb.append([rng.randint(3, H - 4), a,
                       min(W, a + rng.randint(round(8 * sx), round(28 * sx)))])
        else:
            a = rng.randint(0, H - 8)
            vb.append([rng.randint(3, W - 4), a,
                       min(H, a + rng.randint(round(6 * sy), round(18 * sy)))])

    def on_wall(x, y):
        return any(y == h[0] and h[1] <= x <= h[2] for h in hb) or \
               any(x == v[0] and v[1] <= y <= v[2] for v in vb)

    min_dist = round(20 * (W + H) / 82)
    while True:
        s = (rng.randint(2, W - 3), rng.randint(2, H - 3))
        g = (rng.randint(2, W - 3), rng.randint(2, H - 3))
        if not on_wall(*s) and not on_wall(*g) and abs(s[0] - g[0]) + abs(s[1] - g[1]) >= min_dist:
            break
    return {"start": list(s), "goal": list(g), "size": [W, H], "horizontal_barriers": hb,
            "vertical_barriers": vb, "range_x": [0, W], "range_y": [0, H]}


def load_dataset(path, pairs=1):
    """
    Return a list of query dicts from a JSON file. Understands the LLM-A* dataset layout
    (one entry per map with "start_goal": [[start, goal, text], ...]; the first `pairs` pairs of
    each map are used) and plain query dicts (start, goal, barriers, range_x/range_y or size).
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        lists = [v for v in raw.values() if isinstance(v, list)]
        raw = lists[0] if len(lists) == 1 else list(raw.values())
    out, bad = [], 0
    for item in raw:
        try:
            base = {
                "horizontal_barriers": [list(b) for b in item.get("horizontal_barriers", [])],
                "vertical_barriers": [list(b) for b in item.get("vertical_barriers", [])],
            }
            if "range_x" in item and "range_y" in item:
                base["range_x"], base["range_y"] = list(item["range_x"]), list(item["range_y"])
            else:
                base["range_x"], base["range_y"] = [0, item["size"][0]], [0, item["size"][1]]
            base["size"] = [base["range_x"][1], base["range_y"][1]]
            if "start_goal" in item:
                combos = [(sg[0], sg[1]) for sg in item["start_goal"][:pairs]]
            else:
                combos = [(item["start"], item["goal"])]
            for s, g in combos:
                q = copy.deepcopy(base)
                q["start"], q["goal"] = list(s), list(g)
                q["map_id"] = item.get("id")
                out.append(q)
        except (KeyError, TypeError, IndexError):
            bad += 1
    if bad:
        print(f"warning: skipped {bad} dataset entries with missing/odd fields")
    if not out:
        raise SystemExit("no usable maps found in " + path)
    return out


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
        "U0": lambda: solve_counted(grid, start, goal, (), use_incumbent=False, use_guide=False),
        "UG": lambda: solve_counted(grid, start, goal, wps, use_incumbent=False, use_guide=True,
                                    guide_radius=RADIUS),
        "UI": lambda: solve_counted(grid, start, goal, wps, use_incumbent=True, use_guide=False),
        "UF": lambda: solve_counted(grid, start, goal, wps, use_incumbent=True, use_guide=True,
                                    guide_radius=RADIUS),
    }


OPTIMAL = {"A*", "U0", "UG", "UI", "UF"}   # these must equal the reference length
NAMES = ["A*", "LLM*", "U0", "UG", "UI", "UF"]

rng = random.Random(SEED)
res = {n: [] for n in NAMES + ["orig"]}
parity_len = parity_sq = parity_n = 0
n_no_plan = 0
rows = 0
attempts = 0

_maps = f"dataset {DATASET} ({ARGS.pairs} pair(s) per map)" if DATASET else f"{W}x{H} random maps, barrier scale {BARRIER_SCALE}, seed {SEED}"
print(f"config: {_maps}, guide radius {RADIUS}, "
      f"{REPEATS} timing repeats, orig {'skipped' if SKIP_ORIG else 'included'}")
print("per-map: path length / best (1.000 = shortest).   orig = shapely original as published")
print(f"{'map':>3} {'best':>8} | " + " ".join(f"{n:>6}" for n in NAMES + ["orig"]) + " | waypoints")

dataset_maps = load_dataset(DATASET, ARGS.pairs)[ARGS.skip:] if DATASET else None
dataset_iter = iter(dataset_maps) if dataset_maps is not None else None

while rows < N_MAPS and attempts < N_MAPS * 5:
    attempts += 1
    if dataset_iter is not None:
        query = next(dataset_iter, None)
        if query is None:
            print("dataset exhausted")
            break
    else:
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
                          "storage": r["storage"], "peak": r["peak_open"], "ms": ms, "mib": mib})
        if name in OPTIMAL and abs(r["cost"] - best) > 1e-6:
            print(f"    !! {name} length {r['cost']:.6f} != reference {best:.6f}")

    # parity check: does the Grid re-implementation of LLM-A* match the real shapely one?
    orig_ratio = None
    try:
        if SKIP_ORIG:
            raise RuntimeError("skipped")
        orig = LLMAStar(llm="gpt", prompt="standard")
        attach_cache(orig)
        t = time.perf_counter()
        with quiet():
            o = orig.searching(query=copy.deepcopy(query), filepath="orig.png")
        o_ms = (time.perf_counter() - t) * 1000
        o_sq = len(set(orig.CLOSED))
        orig_ratio = o["length"] / best
        res["orig"].append({"ratio": orig_ratio, "work": o_sq, "search": o_sq, "storage": o["storage"],
                            "peak": None, "ms": o_ms, "mib": None})
        parity_n += 1
        if line["LLM*"] is not None:
            parity_len += abs(o["length"] - line["LLM*"] * best) < 1e-6
            parity_sq += o_sq == res["LLM*"][-1]["search"]
    except Exception as exc:
        if not SKIP_ORIG:
            print(f"    orig failed: {type(exc).__name__}: {exc}")

    rows += 1
    n_wp = max(0, len(planner.target_list) - 2)
    n_no_plan += n_wp == 0
    cells = [f"{line[n]:6.3f}" if line.get(n) is not None else f"{'-':>6}" for n in NAMES]
    cells.append(f"{orig_ratio:6.3f}" if orig_ratio is not None else f"{'-':>6}")
    print(f"{rows:3d} {best:8.3f} | " + " ".join(cells) + f" | {n_wp}")

print(f"\n=== Summary over {rows} maps ({n_no_plan} with no usable AI waypoints) ===")
print(f"{'algo':<6} {'shortest':>9} {'extra len':>10} {'work':>8} {'search':>8} {'storage':>8} "
      f"{'open':>7} {'ms':>8} {'MiB':>7}")
for name in NAMES + ["orig"]:
    rs = res[name]
    if not rs:
        continue
    short = sum(1 for r in rs if r["ratio"] < 1 + 1e-9)
    mib = f"{st.mean(r['mib'] for r in rs):7.2f}" if rs[0]["mib"] is not None else f"{'-':>7}"
    opn = f"{st.mean(r['peak'] for r in rs):7.0f}" if rs[0]["peak"] is not None else f"{'-':>7}"
    label = name + ("*" if name == "orig" else "")
    print(f"{label:<6} {short:>4}/{len(rs):<4} {100 * (st.mean(r['ratio'] for r in rs) - 1):9.1f}% "
          f"{st.mean(r['work'] for r in rs):8.0f} {st.mean(r['search'] for r in rs):8.0f} "
          f"{st.mean(r['storage'] for r in rs):8.0f} {opn} "
          f"{st.mean(r['ms'] for r in rs):8.2f} {mib}")

print(f"\nParity (Grid LLM-A* vs shapely original): same length on {parity_len}/{parity_n} maps, "
      f"same unique-squares on {parity_sq}/{parity_n}.")
print("work = search + incumbent hop expansions + guide-BFS cells. Time excludes the Gemini call;\n"
      "Grid/map building is outside the clock for all grid algorithms. Memory = extra peak Python memory.")
print("storage = g-table entries alive at once (finite g only; U counts the g table + incumbent path +\n"
      "guide cells). open = peak open-list size. * orig's storage is the original's len(g), which also\n"
      "counts blocked neighbours, so compare it to the grid rows only loosely.")
print("What the LLM adds to optimal search = compare U0 against UG / UI / UF.")
print("What optimality costs vs the original heuristic = compare LLM* against U0 / UF.")

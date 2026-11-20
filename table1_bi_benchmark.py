"""
table1_bi_benchmark.py -- Table 1 style comparison (OR / SR / RL / VPR, optional WOR) for the
optimal LLM-A* variants, unidirectional and bidirectional. Every method runs on the SAME flat Grid
and the SAME LLM plan per case, and every ratio divides by a baseline from the same implementation.

Methods (per case):
  astar  plain A* on the Grid (= unidirectional with no LLM)
  big    bi0 + LLM guide only
  bii    bi0 + LLM incumbent bound only
  bif    full Bi LLM-A* (incumbent + guide)
  ug     unidirectional + LLM guide only
  ui     unidirectional + LLM incumbent bound only
  uf     full Uni LLM-A* (incumbent + guide)

Ratios (geometric mean over cases, in %, < 100 = cheaper than the baseline):
  OR  = operations (main search + incumbent hop expansions)
  WOR = ops + w * guide-BFS cells (only with --guide-weight w; w = cost of a BFS cell relative
        to one expansion, measure it with w_calculator.py)
  SR  = storage (touched nodes; search + incumbent hop nodes, an upper bound)
  RL  = relative path length (valid paths only; every method here is optimal, so 100.00)

Read the table against the matching baseline:
  Default baseline is plain A* (--baseline astar).

Usage (PowerShell, from the llm-astar folder, key set as for table1_benchmark.py):
    python table1_bi_benchmark.py --configs gpt:standard --max-cases 5     # quick test
    python table1_bi_benchmark.py                                          # all six configs
    python table1_bi_benchmark.py --baseline astar --table-only            # rebuild vs plain A*
"""
import argparse
import copy
import csv
import json
import os
import time

import table1_benchmark as tb
from table1_benchmark import _last, call_with_retry, geo_mean, is_rate_limit, parse_configs, path_is_valid
from llmastar.pather import LLMAStar
from llmastar.pather.llm_a_star.fair_core import Grid, grid_astar, solve_counted_bi, solve_counted_uni

METHODS = ["astar", "big", "bii", "bif", "ug", "ui", "uf"]
LABEL = {"astar": "A* (grid)", "big": "Bi + guide",
         "bii": "Bi + incumbent", "bif": "Bi LLM-A* (full)",
         "ug": "Uni + guide", "ui": "Uni + incumbent", "uf": "Uni LLM-A* (full)"}
FIELDS = (["env_id", "start", "goal", "llm_waypoints", "status", "error"]
          + [f"{m}_{k}" for m in METHODS for k in ("ops", "work", "touched", "cost", "valid")])


# --------------------------------------------------------------------------
# LLM plan (cached per model + prompt text, so reruns do not spend quota)
# --------------------------------------------------------------------------
def get_plan(llm, prompt, query, cache, cache_file):
    planner = LLMAStar(llm=llm, prompt=prompt)
    real_ask = planner.model.ask

    def ask(*a, **k):
        p = k["prompt"] if "prompt" in k else (a[0] if a else "")
        key = f"{llm}|{prompt}|{p}"
        if key in cache:
            return cache[key]
        ans = real_ask(*a, **k)
        if ans and str(ans).strip():
            cache[key] = ans
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        return ans

    planner.model.ask = ask
    data = planner._parse_query(copy.deepcopy(query))
    planner._initialize_parameters(data)
    planner._initialize_llm_paths()
    return planner


def run_case(llm, prompt, env, start, goal, cache, cache_file, args):
    query = {
        "start": copy.deepcopy(start), "goal": copy.deepcopy(goal),
        "size": [env["range_x"][1], env["range_y"][1]],
        "horizontal_barriers": copy.deepcopy(env["horizontal_barriers"]),
        "vertical_barriers": copy.deepcopy(env["vertical_barriers"]),
        "range_x": copy.deepcopy(env["range_x"]), "range_y": copy.deepcopy(env["range_y"]),
    }
    _last["empty"] = False
    planner = call_with_retry(lambda: get_plan(llm, prompt, query, cache, cache_file),
                              retries=args.retries)
    if _last["empty"]:
        raise ValueError("LLM returned an empty reply: rate limit or token quota probably "
                         "used up; rerun later to continue")

    grid = Grid(planner.range_x, planner.range_y, planner.horizontal_barriers,
                planner.vertical_barriers, planner.u_set)
    s, g = grid.index(*planner.s_start), grid.index(*planner.s_goal)
    wps = grid.waypoint_indices(planner.target_list[1:-1])
    kw = dict(guide_radius=args.guide_radius, hop_cap_factor=args.hop_cap)
    results = {
        "astar": grid_astar(grid, s, g),
        "big": solve_counted_bi(grid, s, g, wps, use_incumbent=False, use_guide=True, **kw),
        "bii": solve_counted_bi(grid, s, g, wps, use_incumbent=True, use_guide=False, **kw),
        "bif": solve_counted_bi(grid, s, g, wps, use_incumbent=True, use_guide=True, **kw),
        "ug": solve_counted_uni(grid, s, g, wps, use_incumbent=False, use_guide=True, **kw),
        "ui": solve_counted_uni(grid, s, g, wps, use_incumbent=True, use_guide=False, **kw),
        "uf": solve_counted_uni(grid, s, g, wps, use_incumbent=True, use_guide=True, **kw),
    }

    row = {"llm_waypoints": max(len(planner.target_list) - 2, 0), "status": "ok", "error": ""}
    for m, r in results.items():
        if r is None:
            raise ValueError(f"{m}: no path found")
        xy = [grid.xy(i) for i in r["path"]]
        row[f"{m}_ops"] = r.get("ops", r["expanded"])
        row[f"{m}_work"] = r["work"]
        row[f"{m}_touched"] = r["storage"]      # search + incumbent hop nodes
        row[f"{m}_cost"] = r["cost"]
        row[f"{m}_valid"] = path_is_valid(xy, grid.xy(s), grid.xy(g))
    return row


# --------------------------------------------------------------------------
# Resume-safe CSV handling and the run loop
# --------------------------------------------------------------------------
def load_rows(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["env_id"], r["start"], r["goal"]): r
                for r in csv.DictReader(f) if r.get("status") == "ok"}


def run_config(envs, path, llm, prompt, limit, args, cache, cache_file):
    """Returns True if it stopped because of a rate limit / empty reply."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ok_rows = load_rows(path)
    with open(path, "w", newline="", encoding="utf-8") as f:   # keep only finished rows
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in ok_rows.values():
            w.writerow({k: r.get(k, "") for k in FIELDS})
    done = set(ok_rows)

    processed = 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        for env in envs:
            env_id = str(env.get("id"))
            for sg in env["start_goal"]:
                if limit is not None and processed >= limit:
                    return False
                start, goal = list(sg[0]), list(sg[1])
                key = (env_id, json.dumps(start), json.dumps(goal))
                if key in done:
                    continue
                processed += 1
                row = {"env_id": env_id, "start": key[1], "goal": key[2]}
                try:
                    row.update(run_case(llm, prompt, env, start, goal, cache, cache_file, args))
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    if is_rate_limit(msg) or "empty reply" in msg:
                        return True        # this case is NOT logged; retried next time
                    row.update(status="error", error=msg)
                w.writerow(row)
                f.flush()
                print(f"[{llm}:{prompt}] Env {env_id} | {start} -> {goal} | "
                      f"{row['status']} {row['error']}")
                if args.sleep:
                    time.sleep(args.sleep)
    return False


def run_all(args, configs, envs, cache, cache_file):
    stopped, rnd = set(), 0
    while len(stopped) < len(configs):
        rnd += 1
        progressed = False
        for llm, prompt in configs:
            name = f"{llm}_{prompt}"
            if name in stopped:
                continue
            path = os.path.join(args.outdir, name + ".csv")
            have = len(load_rows(path))
            batch = args.batch
            if args.max_cases is not None:
                batch = min(batch, args.max_cases - have)
                if batch <= 0:
                    stopped.add(name)
                    continue
            print(f"\n=== Round {rnd}: {llm}:{prompt} ({have} cases done) ===")
            hit = run_config(envs, path, llm, prompt, batch, args, cache, cache_file)
            now = len(load_rows(path))
            progressed |= now > have
            if hit:
                stopped.add(name)
                print(f"*** {llm}:{prompt} stopped: rate limit / quota reached ({now} cases done). "
                      f"Run again later to continue.")
            elif now == have:
                stopped.add(name)
                print(f"*** {llm}:{prompt}: no new finished cases this round, stopping it.")
        if not progressed:
            break


# --------------------------------------------------------------------------
# Table over the cases ALL configurations have finished
# --------------------------------------------------------------------------
def ratios(rows, m, base, key):
    out = []
    for r in rows:
        d = float(r[f"{base}_{key}"])
        if d > 0:
            out.append(float(r[f"{m}_{key}"]) / d)
    return out


def wratios(rows, m, base, w):
    """Ratio of ops + w * guide cells (guide cells = work - ops)."""
    out = []
    for r in rows:
        num = float(r[f"{m}_ops"]) + w * (float(r[f"{m}_work"]) - float(r[f"{m}_ops"]))
        den = float(r[f"{base}_ops"]) + w * (float(r[f"{base}_work"]) - float(r[f"{base}_ops"]))
        if den > 0:
            out.append(num / den)
    return out


def table_row(label, llm, prompt, rows, m, base, w=None):
    valid = [r for r in rows if r[f"{m}_valid"] == "True"]
    rl = ratios(valid, m, base, "cost")
    vpr = 100 * len(valid) / len(rows)
    wp = 100 * sum(1 for r in rows if r["llm_waypoints"] not in ("", "0")) / len(rows)
    pct = lambda v: f"{geo_mean(v) * 100:.2f}"
    cols = [pct(ratios(rows, m, base, "ops"))]
    if w is not None:
        cols.append(pct(wratios(rows, m, base, w)))
    return [label, llm, prompt] + cols + [pct(ratios(rows, m, base, "touched")), pct(rl),
                                          f"{vpr:.2f}", f"{wp:.1f}"]


def build_table(args, configs):
    data = {}
    for llm, prompt in configs:
        rows = load_rows(os.path.join(args.outdir, f"{llm}_{prompt}.csv"))
        if rows:
            data[(llm, prompt)] = rows
    if not data:
        print("\nNo finished cases yet, so there is no table to build.")
        return
    common = sorted(set.intersection(*[set(r) for r in data.values()]))
    n = len(common)
    base = args.baseline
    print("\n" + "=" * 86)
    print(f"OPTIMAL LLM-A* TABLE (values in %, geometric means, baseline = {LABEL[base]} = 100)")
    print("Cases finished per config: " + ", ".join(f"{l}:{p}={len(r)}" for (l, p), r in data.items()))
    print(f"Table uses the {n} cases that ALL listed configs finished.")
    if n < 30:
        print("WARNING: very small sample, so treat these numbers as rough.")
    print("=" * 86)

    w = args.guide_weight
    head = ["Method", "Model", "Prompt", "OR(%)"] + (["WOR(%)"] if w is not None else []) + \
           ["SR(%)", "RL(%)", "VPR(%)", "wpts(%)"]
    table = [[LABEL[base], "-", "-", "100"] + (["100"] if w is not None else []) + ["100", "100", "100", "-"]]
    bad = 0
    for (llm, prompt), rows in data.items():
        rs = [rows[k] for k in common]
        if not rs:
            continue
        for m in ("big", "bii", "bif", "ug", "ui", "uf"):
            table.append(table_row(LABEL[m], llm, prompt, rs, m, base, w))
        for r in rs:   # every method here is optimal, so all lengths must match the baseline
            bad += sum(abs(float(r[f"{m}_cost"]) - float(r[f"{base}_cost"])) > 1e-6 for m in METHODS)

    widths = [max(len(str(row[i])) for row in [head] + table) for i in range(len(head))]
    for row in [head] + table:
        print("  ".join(str(c).ljust(wd) for c, wd in zip(row, widths)))
    print("\nOR = operations (search + incumbent hop expansions), SR = storage (touched nodes incl. hops, "
          "upper bound),\nRL = relative path length, VPR = valid path ratio, wpts = % of cases with >= 1 "
          "LLM waypoint.")
    if w is not None:
        print(f"WOR = ops + {w} * guide-BFS cells.")
    if bad:
        print(f"!! {bad} method/case pairs had a path length different from the baseline: investigate.")
    else:
        print("Length check passed: all methods returned the optimal length on every case.")

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "table_bidirectional.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(head + ["cases_in_table"])
        for row in table:
            wr.writerow(row + [n])
    print(f"Saved -> {out}")


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Table 1 style comparison for optimal (uni/bi) LLM-A*")
    p.add_argument("--dataset", default="dataset/environment_50_30.json")
    p.add_argument("--outdir", default="results_bi")
    p.add_argument("--configs", default=tb.ALL_CONFIGS)
    p.add_argument("--baseline", choices=["astar"], default="astar",
                   help="what the ratios divide by (plain A* on the same grid)")
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--guide-radius", type=int, default=2)
    p.add_argument("--hop-cap", type=int, default=20, help="hop_cap_factor for the incumbent hops")
    p.add_argument("--guide-weight", type=float, default=None,
                   help="cost of one guide-BFS cell relative to one expansion (measure it); "
                        "adds a WOR column = ops + w * guide cells")
    p.add_argument("--table-only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configs = parse_configs(args.configs)
    if not args.table_only:
        tb.install_ask_hooks(args.max_tokens)
        with open(args.dataset, "r") as f:
            envs = json.load(f)
        os.makedirs(args.outdir, exist_ok=True)
        cache_file = os.path.join(args.outdir, "llm_cache.json")
        try:
            with open(cache_file, encoding="utf-8") as f:
                cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            cache = {}
        run_all(args, configs, envs, cache, cache_file)
    build_table(args, configs)
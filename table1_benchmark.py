"""
Replicate Table 1 of "LLM-A*" (arXiv 2407.02511) with the llmastar package.

What this single file does
--------------------------
1. Runs A* and LLM-A* for every (model, prompt) configuration, in ROUNDS of a few
   cases each, so that when your API quota runs out all configurations have
   roughly the same number of cases.
2. Stops a configuration cleanly when it hits a rate limit / empty LLM reply, and
   carries on with the others. Running the same command again later continues
   where it stopped (it is always resumable).
3. Prints a Table 1 style summary computed over the cases that every
   configuration has finished, and saves it to <outdir>/table1_replication.csv.

Usage (PowerShell, from the llm-astar folder; set your key first):
    $env:GROQ_API_KEY = "gsk_..."            # same variable your package reads

    python table1_benchmark.py                                  # all six configs, until quota runs out
    python table1_benchmark.py --configs gpt:standard --max-cases 5   # quick test
    python table1_benchmark.py --table-only                     # just rebuild the table

How the path is obtained
------------------------
llmastar's searching() returns only a stats dict (operation, storage, length,
llm_output). The real path is computed inside the solver and handed to
Plotting.animation(path, visited, show=True, ...), which draws it and opens a
window. We replace Plotting.animation with a function that RECORDS the path and
draws nothing: no blocking plot windows, faster runs, and the real path to validate.
"""

# --- no plot windows, ever (must come before anything imports pyplot) ---------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

import argparse
import copy
import csv
import json
import os
import time

import numpy as np
from llmastar.pather import AStar, LLMAStar
from llmastar.env.search import env as env_search, plotting as plotting_search
from llmastar import model as llm_model

# Never write your API key in this file; set it in the terminal (see usage above).

FIELDS = [
    "env_id", "start", "goal",
    "astar_ops", "astar_storage", "astar_length",
    "llm_ops", "llm_storage", "llm_length",
    "op_ratio", "storage_ratio", "length_ratio",
    "llm_waypoints", "is_valid", "status", "error",
]
ALL_CONFIGS = "gpt:standard,gpt:cot,gpt:repe,llama:standard,llama:cot,llama:repe"

# --------------------------------------------------------------------------
# Path capture: replace the package's drawing routine with a recorder.
# --------------------------------------------------------------------------
_last = {"path": None, "empty": False}


def _capture_animation(self, path, visited, show, name, filepath):
    _last["path"] = [tuple(p) for p in path] if path else None


plotting_search.Plotting.animation = _capture_animation


def install_ask_hooks(max_tokens):
    """Wrap the model classes' ask():
      * raise the token budget the package passes (it hardcodes 1000, which a
        reasoning model such as gpt-oss can use up on thinking, returning an
        EMPTY reply; the package then silently falls back to [start, goal]);
      * remember when a reply comes back empty so the case is treated as a
        quota/limit problem instead of being recorded as a normal result."""
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
def parse_result(res):
    """Return (path, ops, storage, length) from a dict or tuple/list result.
    Missing values come back as None, never as 0, so failures can't pose as data."""
    if isinstance(res, dict):
        d, path = res, res.get("path")
    elif isinstance(res, (tuple, list)):
        path = res[0] if len(res) > 0 else None
        d = next((x for x in res[1:] if isinstance(x, dict)), {})
    else:
        return None, None, None, None

    ops = d.get("operation", d.get("operations"))
    storage = d.get("storage", d.get("max_storage"))
    length = d.get("length")
    return path, ops, storage, length


def build_obstacles(env):
    """Fallback obstacle set from barrier definitions:
    horizontal [y, x_start, x_end], vertical [x, y_start, y_end]."""
    cells = set()
    for y, x1, x2 in env.get("horizontal_barriers", []):
        for x in range(int(x1), int(x2) + 1):
            cells.add((x, int(y)))
    for x, y1, y2 in env.get("vertical_barriers", []):
        for y in range(int(y1), int(y2) + 1):
            cells.add((int(x), y))
    return cells


def get_obstacles(env):
    """Prefer the package's own obstacle map so validation matches the solvers."""
    try:
        e = env_search.Env(env["range_x"][1], env["range_y"][1],
                           env["horizontal_barriers"], env["vertical_barriers"])
        return set(map(tuple, e.obs_map()))
    except Exception:
        return build_obstacles(env)


def path_is_valid(path, start, goal, obstacles=None, bounds=None):
    """Valid = connects start and goal (either direction), every step moves to an
    adjacent cell (8-connected), and (optionally) avoids obstacles / stays in bounds.
    Points are converted to tuples first because (5, 5) != [5, 5] in Python."""
    if not path or not isinstance(path, (list, tuple)):
        return False
    pts = [tuple(p) for p in path]
    s, g = tuple(start), tuple(goal)
    if pts[0] == g and pts[-1] == s:
        pts = pts[::-1]
    if pts[0] != s or pts[-1] != g:
        return False
    for a, b in zip(pts, pts[1:]):
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) != 1:
            return False
    if obstacles is not None and any(p in obstacles for p in pts):
        return False
    if bounds is not None:
        (x0, x1), (y0, y1) = bounds
        if any(not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1) for p in pts):
            return False
    return True


def is_rate_limit(msg):
    m = str(msg).lower()
    return "429" in m or "rate limit" in m or "rate_limit" in m or "tokens per day" in m


def call_with_retry(fn, retries=3, base_delay=2.0):
    """Run fn(); on exception wait with exponential backoff and retry.
    Rate-limit errors wait at least 30 s (per-minute limits clear after that)."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == retries:
                raise
            wait = base_delay * (2 ** attempt)
            if is_rate_limit(e):
                wait = max(wait, 30)
            time.sleep(wait)


def case_key(env_id, start, goal):
    return (str(env_id), json.dumps(list(start)), json.dumps(list(goal)))


def prepare_resume(csv_path):
    """Keep only finished rows (status == ok) so failed or rate-limited cases are
    retried on resume. Returns the set of case keys that are already done."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
            done.add((r["env_id"], r["start"], r["goal"]))
    return done


def load_ok_rows(csv_path):
    """{(env_id, start, goal): row} for finished cases in one results file."""
    rows = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    rows[(r["env_id"], r["start"], r["goal"])] = r
    return rows


def fmt(x):
    return "" if x is None else x


def geo_mean(vals):
    """The paper (Sec. 4.3) aggregates OR / SR / RL with the GEOMETRIC mean."""
    pos = [v for v in vals if v > 0]
    return float(np.exp(np.mean(np.log(pos)))) if pos else float("nan")


# --------------------------------------------------------------------------
# One configuration: run up to `limit` new cases and append them to its CSV
# --------------------------------------------------------------------------
def run_config(dataset_path, output_csv, llm, prompt, limit, retries, sleep_between,
               check_obstacles):
    """Returns True if it stopped because of a rate limit / empty reply."""
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found.")
        return False

    with open(dataset_path, "r") as f:
        environments = json.load(f)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    done = prepare_resume(output_csv)
    new_file = not os.path.exists(output_csv)

    processed = 0
    hit_limit = False
    with open(output_csv, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if new_file:
            writer.writerow(FIELDS)
            csv_file.flush()

        for env in environments:
            env_id = env.get("id")
            obstacles = get_obstacles(env) if check_obstacles else None
            bounds = (tuple(env["range_x"]), tuple(env["range_y"]))

            for sg in env["start_goal"]:
                if limit is not None and processed >= limit:
                    break

                start, goal = list(sg[0]), list(sg[1])
                if case_key(env_id, start, goal) in done:
                    continue
                processed += 1

                query = {
                    "start": copy.deepcopy(start),
                    "goal": copy.deepcopy(goal),
                    "size": [env["range_x"][1], env["range_y"][1]],
                    "horizontal_barriers": copy.deepcopy(env["horizontal_barriers"]),
                    "vertical_barriers": copy.deepcopy(env["vertical_barriers"]),
                    "range_x": copy.deepcopy(env["range_x"]),
                    "range_y": copy.deepcopy(env["range_y"]),
                }

                a_ops = a_storage = a_len = None
                l_ops = l_storage = l_len = None
                n_waypoints = None
                op_ratio = storage_ratio = length_ratio = None
                is_valid = False
                status, error = "ok", ""

                # 1. Standard A* baseline
                try:
                    a_res = AStar().searching(query=copy.deepcopy(query))
                    _, a_ops, a_storage, a_len = parse_result(a_res)
                    if a_ops is None:
                        raise ValueError(f"Could not parse A* result: {type(a_res)}")
                except Exception as e:
                    status, error = "astar_error", f"{type(e).__name__}: {e}"

                # 2. LLM-A*
                if status == "ok":
                    try:
                        _last["path"] = None
                        _last["empty"] = False
                        # A fresh solver per case avoids any state leaking between queries.
                        l_res = call_with_retry(
                            lambda: LLMAStar(llm=llm, prompt=prompt)
                            .searching(query=copy.deepcopy(query)),
                            retries=retries,
                        )
                        _, l_ops, l_storage, l_len = parse_result(l_res)
                        if l_ops is None:
                            raise ValueError(f"Could not parse LLM-A* result: {type(l_res)}")

                        if _last["empty"]:
                            raise ValueError("LLM returned an empty reply: rate limit or token "
                                             "quota probably used up; rerun later to continue")
                        targets = l_res.get("llm_output") if isinstance(l_res, dict) else None
                        if targets is not None:
                            n_waypoints = max(len(targets) - 2, 0)  # excludes start and goal
                        l_path = _last["path"]  # recorded by the animation hook
                        if l_path is None:
                            raise ValueError("search finished but no path was captured")

                        is_valid = path_is_valid(l_path, start, goal, obstacles, bounds)

                        if a_ops:
                            op_ratio = l_ops / a_ops
                        if a_storage and l_storage is not None:
                            storage_ratio = l_storage / a_storage
                        if a_len and l_len is not None:
                            length_ratio = l_len / a_len
                    except Exception as e:
                        status, error = "llm_error", f"{type(e).__name__}: {e}"
                        if is_rate_limit(error):
                            hit_limit = True

                if hit_limit:
                    break  # this case is NOT logged; it is simply retried next time

                writer.writerow([
                    env_id, json.dumps(start), json.dumps(goal),
                    fmt(a_ops), fmt(a_storage), fmt(a_len),
                    fmt(l_ops), fmt(l_storage), fmt(l_len),
                    "" if op_ratio is None else f"{op_ratio:.4f}",
                    "" if storage_ratio is None else f"{storage_ratio:.4f}",
                    "" if length_ratio is None else f"{length_ratio:.4f}",
                    fmt(n_waypoints), is_valid, status, error,
                ])
                csv_file.flush()  # keep progress if the run dies

                shown = "n/a" if op_ratio is None else f"{op_ratio:.4f}"
                print(f"[{llm}:{prompt}] Env {env_id} | {start} -> {goal} | "
                      f"valid={is_valid} | op_ratio={shown} | {status}")

                if sleep_between:
                    time.sleep(sleep_between)

            if hit_limit or (limit is not None and processed >= limit):
                break

    return hit_limit


# --------------------------------------------------------------------------
# Run every configuration in rounds until each one is done or out of quota
# --------------------------------------------------------------------------
def run_all(args, configs):
    stopped = set()
    rnd = 0
    while len(stopped) < len(configs):
        rnd += 1
        progressed = False
        for llm, prompt in configs:
            key = f"{llm}_{prompt}"
            if key in stopped:
                continue
            path = os.path.join(args.outdir, key + ".csv")
            have = len(load_ok_rows(path))

            batch = args.batch
            if args.max_cases is not None:
                batch = min(batch, args.max_cases - have)
                if batch <= 0:
                    stopped.add(key)
                    continue

            print(f"\n=== Round {rnd}: {llm}:{prompt} ({have} cases done) ===")
            hit = run_config(args.dataset, path, llm, prompt, batch, args.retries,
                             args.sleep, not args.no_obstacle_check)
            now = len(load_ok_rows(path))
            if now > have:
                progressed = True
            if hit:
                stopped.add(key)
                print(f"*** {llm}:{prompt} stopped: rate limit / quota reached "
                      f"({now} cases done). Run this command again later to continue.")
            elif now == have:
                stopped.add(key)   # nothing new: dataset finished or cases keep failing
                print(f"*** {llm}:{prompt}: no new finished cases this round, stopping it.")
        if not progressed:
            break


# --------------------------------------------------------------------------
# Table 1 style summary over the cases ALL configurations have finished
# --------------------------------------------------------------------------
def build_table(outdir, configs):
    data = {}
    for llm, prompt in configs:
        rows = load_ok_rows(os.path.join(outdir, f"{llm}_{prompt}.csv"))
        if rows:
            data[(llm, prompt)] = rows
    if not data:
        print("\nNo finished cases yet, so there is no table to build.")
        return

    missing = [f"{l}:{p}" for l, p in configs if (l, p) not in data]
    common = set.intersection(*[set(r) for r in data.values()])
    n = len(common)
    print("\n" + "=" * 78)
    print("TABLE 1 REPLICATION (values in %, geometric means as in the paper)")
    print(f"Cases finished per config: " +
          ", ".join(f"{l}:{p}={len(r)}" for (l, p), r in data.items()))
    print(f"Table uses the {n} cases that ALL listed configs finished.")
    if missing:
        print(f"No results yet for: {', '.join(missing)}")
    if n < 30:
        print("WARNING: very small sample, so treat these numbers as rough.")
    print("=" * 78)

    head = ["Method", "Model", "Prompt", "OR(%)", "SR(%)", "RL(%)", "VPR(%)", "wpts(%)"]
    table = [["A*", "-", "-", "100", "100", "100", "100", "-"]]
    for (llm, prompt), rows in data.items():
        rs = [rows[k] for k in sorted(common)]
        ok = lambda col: [float(r[col]) for r in rs if r[col] != ""]
        valid_rows = [r for r in rs if r["is_valid"] == "True"]
        rl = [float(r["length_ratio"]) for r in valid_rows if r["length_ratio"] != ""]
        vpr = len(valid_rows) / len(rs) * 100 if rs else float("nan")
        wp = sum(1 for r in rs if r["llm_waypoints"] not in ("", "0")) / len(rs) * 100 if rs else float("nan")
        table.append(["LLM-A*", llm, prompt,
                      f"{geo_mean(ok('op_ratio')) * 100:.2f}",
                      f"{geo_mean(ok('storage_ratio')) * 100:.2f}",
                      f"{geo_mean(rl) * 100:.2f}",
                      f"{vpr:.2f}", f"{wp:.1f}"])

    widths = [max(len(str(row[i])) for row in [head] + table) for i in range(len(head))]
    for row in [head] + table:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    print("\nOR = operation ratio, SR = storage ratio, RL = relative path length (valid paths "
          "only), VPR = valid path ratio,\nwpts = % of cases where the LLM gave at least one "
          "waypoint besides start/goal (0 would mean the LLM is not being used).")

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "table1_replication.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head + ["cases_in_table"])
        for row in table:
            w.writerow(row + [n])
    print(f"\nSaved -> {out}")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def parse_configs(text):
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        llm, prompt = item.split(":")
        if llm not in ("gpt", "llama") or prompt not in ("standard", "cot", "repe"):
            raise SystemExit(f"Bad config '{item}'. Use gpt|llama : standard|cot|repe, e.g. gpt:cot")
        out.append((llm, prompt))
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Replicate Table 1 of LLM-A*")
    p.add_argument("--dataset", default="dataset/environment_50_30.json")
    p.add_argument("--outdir", default="results", help="folder for the per-config CSV files")
    p.add_argument("--configs", default=ALL_CONFIGS,
                   help="comma list of model:prompt, e.g. gpt:standard,llama:cot")
    p.add_argument("--batch", type=int, default=10,
                   help="new cases per config per round (keeps configs balanced)")
    p.add_argument("--max-cases", type=int, default=None,
                   help="stop each config after this many finished cases (default: no cap)")
    p.add_argument("--retries", type=int, default=3,
                   help="LLM call retries; rate-limit errors wait 30 s between tries")
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="output-token budget per LLM call (package default 1000 is too small "
                        "for reasoning models; 0 keeps the package's value)")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds to pause between cases")
    p.add_argument("--no-obstacle-check", action="store_true",
                   help="validate start/goal/adjacency only (skip barrier check)")
    p.add_argument("--table-only", action="store_true",
                   help="do not run anything, just rebuild the table from existing results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configs = parse_configs(args.configs)
    if not args.table_only:
        install_ask_hooks(args.max_tokens)
        run_all(args, configs)
    build_table(args.outdir, configs)
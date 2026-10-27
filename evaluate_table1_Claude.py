"""
Benchmark LLM-A* against plain A* and save per-case results plus a summary.

How the path is obtained
------------------------
llmastar's searching() returns only a stats dict (operation, storage, length,
llm_output). The real path is computed inside the solver and handed to
Plotting.animation(path, visited, show=True, ...), which draws it and opens a
window. We replace Plotting.animation with a function that RECORDS the path and
draws nothing. That removes the blocking plot windows, makes runs much faster,
and gives us the actual path to validate.

Usage examples:
    python evaluate_table1_Claude.py --limit 5
    python evaluate_table1_Claude.py --llm gpt --prompt standard --resume --sleep 1

Outputs:
    <output>.csv          one row per (env, start, goal) test case
    <output>_summary.csv  overall metrics (separate so pandas can read the data file)
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
import sys

import numpy as np
from llmastar.pather import AStar, LLMAStar
from llmastar.env.search import env as env_search, plotting as plotting_search
from llmastar import model as llm_model

# --------------------------------------------------------------------------
# API keys: set them in your terminal instead of writing them in this file.
#   PowerShell:  $env:OPENAI_API_KEY = "sk-..."
# The stock llmastar package calls OpenAI for llm="gpt". GROQ_API_KEY only
# matters if you modified the package to call Groq.
# --------------------------------------------------------------------------

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
      * remember when a reply comes back empty so the case is flagged as an error
        instead of being recorded as a normal result."""
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

FIELDS = [
    "env_id", "start", "goal",
    "astar_ops", "astar_storage", "astar_length",
    "llm_ops", "llm_storage", "llm_length",
    "op_ratio", "storage_ratio", "length_ratio",
    "llm_waypoints", "is_valid", "status", "error",
]


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
    adjacent cell (8-connected), and (optionally) avoids obstacles / stays in bounds."""
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


def fmt(x):
    return "" if x is None else x


# --------------------------------------------------------------------------
# Summary (computed from the CSV so resumed runs are summarized correctly)
# --------------------------------------------------------------------------
def write_summary(csv_path, summary_path, llm_model, prompt_style):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    valid = sum(1 for r in rows if r["is_valid"] == "True")
    errors = sum(1 for r in rows if r["status"] != "ok")
    ok_rows = total - errors
    with_wp = sum(1 for r in rows if r.get("llm_waypoints", "") not in ("", "0"))
    wp_pct = with_wp / ok_rows * 100 if ok_rows else 0.0

    def mean_of(col, only_valid=False):
        vals = [
            float(r[col]) for r in rows
            if r[col] != "" and (not only_valid or r["is_valid"] == "True")
        ]
        # The paper (Sec. 4.3) aggregates OR / SR / RL with the GEOMETRIC mean.
        pos = [v for v in vals if v > 0]
        geo = float(np.exp(np.mean(np.log(pos)))) if pos else float("nan")
        arith = float(np.mean(vals)) if vals else float("nan")
        return geo, arith, len(vals)

    op_mean, op_arith, op_n = mean_of("op_ratio")
    st_mean, st_arith, st_n = mean_of("storage_ratio")
    ln_mean, ln_arith, ln_n = mean_of("length_ratio", only_valid=True)

    success_rate = valid / total * 100 if total else 0.0

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["model_prompt", f"{llm_model} ({prompt_style})"])
        w.writerow(["total_test_cases", total])
        w.writerow(["error_cases", errors])
        w.writerow(["pct_cases_with_llm_waypoints", f"{wp_pct:.2f}"])
        w.writerow(["valid_path_ratio_pct", f"{success_rate:.2f}"])
        w.writerow(["geo_mean_operation_ratio", f"{op_mean:.4f}"])
        w.writerow(["geo_mean_storage_ratio", f"{st_mean:.4f}"])
        w.writerow(["geo_mean_length_ratio_valid_only", f"{ln_mean:.4f}"])
        w.writerow(["arith_mean_operation_ratio", f"{op_arith:.4f}"])
        w.writerow(["arith_mean_storage_ratio", f"{st_arith:.4f}"])
        w.writerow(["arith_mean_length_ratio_valid_only", f"{ln_arith:.4f}"])

    print("\n================== SUMMARY ==================")
    print(f"Model / prompt        : {llm_model} ({prompt_style})")
    print(f"Total test cases      : {total}  (errors: {errors})")
    print(f"Valid path ratio      : {success_rate:.2f}%")
    print(f"Cases w/ LLM waypoints: {wp_pct:.2f}%  (0% means LLM-A* is not using the LLM)")
    print("Geometric means (paper's aggregation)  [arithmetic in brackets]")
    print(f"Operation ratio       : {op_mean:.4f}  [{op_arith:.4f}]  (n={op_n})")
    print(f"Storage ratio         : {st_mean:.4f}  [{st_arith:.4f}]  (n={st_n})")
    print(f"Length ratio (valid)  : {ln_mean:.4f}  [{ln_arith:.4f}]  (n={ln_n})")
    print(f"Data    -> {csv_path}")
    print(f"Summary -> {summary_path}")
    print("=============================================")


# --------------------------------------------------------------------------
# Main benchmark
# --------------------------------------------------------------------------
def evaluate_and_save_csv(
    dataset_path="dataset/environment_50_30.json",
    output_csv="table1_results.csv",
    llm_model="gpt",
    prompt_style="standard",
    resume=False,
    limit=None,
    retries=3,
    sleep_between=0.0,
    check_obstacles=True,
):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found.")
        return

    with open(dataset_path, "r") as f:
        environments = json.load(f)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.splitext(output_csv)[0] + "_summary.csv"

    done = prepare_resume(output_csv) if resume else set()
    mode = "a" if (resume and os.path.exists(output_csv)) else "w"

    print(f"--- Running benchmark for {llm_model.upper()} ({prompt_style}) ---")
    if resume and done:
        print(f"Resuming: {len(done)} cases already in {output_csv}")

    processed = 0
    hit_limit = False
    with open(output_csv, mode=mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if mode == "w":
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
                key = case_key(env_id, start, goal)
                if key in done:
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
                            lambda: LLMAStar(llm=llm_model, prompt=prompt_style)
                            .searching(query=copy.deepcopy(query)),
                            retries=retries,
                        )
                        _, l_ops, l_storage, l_len = parse_result(l_res)
                        if l_ops is None:
                            raise ValueError(f"Could not parse LLM-A* result: {type(l_res)}")

                        if _last["empty"]:
                            raise ValueError("LLM returned an empty reply: rate limit or token "
                                            "quota probably used up; rerun with --resume later")
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
                    # Don't log this case as a failure: it will simply be retried on --resume.
                    print("\nRate limit reached - stopping. Wait for the limit to reset, then "
                          "rerun the same command with --resume.")
                    break

                writer.writerow([
                    env_id, json.dumps(start), json.dumps(goal),
                    fmt(a_ops), fmt(a_storage), fmt(a_len),
                    fmt(l_ops), fmt(l_storage), fmt(l_len),
                    "" if op_ratio is None else f"{op_ratio:.4f}",
                    "" if storage_ratio is None else f"{storage_ratio:.4f}",
                    "" if length_ratio is None else f"{length_ratio:.4f}",
                    fmt(n_waypoints), is_valid, status, error,
                ])
                csv_file.flush()  # keep progress if the run dies or gets rate-limited

                shown = "n/a" if op_ratio is None else f"{op_ratio:.4f}"
                print(f"Env {env_id} | {start} -> {goal} | valid={is_valid} | "
                      f"op_ratio={shown} | {status}")

                if sleep_between:
                    time.sleep(sleep_between)

            if hit_limit or (limit is not None and processed >= limit):
                break

    write_summary(output_csv, summary_csv, llm_model, prompt_style)
    return hit_limit


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark LLM-A* vs A*")
    p.add_argument("--dataset", default="dataset/environment_50_30.json")
    p.add_argument("--output", default="table1_results.csv")
    p.add_argument("--llm", default="gpt")
    p.add_argument("--prompt", default="standard")
    p.add_argument("--resume", action="store_true",
                   help="append to the output file and skip cases already recorded")
    p.add_argument("--limit", type=int, default=None, help="max new cases to run")
    p.add_argument("--retries", type=int, default=3, help="LLM call retries with backoff")
    p.add_argument("--max-tokens", type=int, default=4000,
                   help="output-token budget per LLM call (package default 1000 is too small "
                        "for reasoning models; 0 keeps the package's value)")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds to pause between cases")
    p.add_argument("--no-obstacle-check", action="store_true",
                   help="validate start/goal/adjacency only (skip barrier check)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    install_ask_hooks(args.max_tokens)
    stopped = evaluate_and_save_csv(
        dataset_path=args.dataset,
        output_csv=args.output,
        llm_model=args.llm,
        prompt_style=args.prompt,
        resume=args.resume,
        limit=args.limit,
        retries=args.retries,
        sleep_between=args.sleep,
        check_obstacles=not args.no_obstacle_check,
    )
    sys.exit(2 if stopped else 0)   # 2 = stopped by a rate limit / empty reply
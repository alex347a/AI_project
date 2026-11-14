"""Ablation without any LLM: LLM-A* search with an EMPTY waypoint list (no API key needed).

In llmastar/pather/llm_a_star/llm_a_star.py the priority is f(s) = g(s) + h(s) with
    h(s) = dist(s, goal) + dist(s, current target).
Without waypoint the current target is the goal, so h(s) = 2 * dist(s, goal): a weighted
(inadmissible) A*. This script measures how much of the OR / SR / RL of Table 1 is already
obtained by that search rule alone, i.e. what the LLM waypoints must be compared with.

The A* reference (operations, storage, length) is read from the saved CSV of the main run,
so A* is not recomputed. Usage:
    python stats_tools/ablation_no_waypoints.py --ref results_gemini_1000 --out ablation_out
"""
import argparse
import contextlib
import copy
import csv
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmastar.env.search import plotting  # noqa: E402
from llmastar.pather import LLMAStar  # noqa: E402

lla = sys.modules[LLMAStar.__module__]   # module that defines LLMAStar


class NoLLM:
    """Stand-in for the model wrapper: returns text without any waypoint."""

    def __init__(self, *args, **kwargs):
        pass

    def ask(self, *args, **kwargs):
        return "no waypoint"


EPS = 1e-9   # ratios closer to 1 than this are ties (floating-point noise differs between platforms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="results_gemini_1000", help="dir with gpt_standard.csv (A* reference)")
    ap.add_argument("--dataset", default="dataset/environment_50_30.json")
    ap.add_argument("--out", default="ablation_out")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    lla.ChatGPT = NoLLM                      # no network, no key
    plotting.Plotting.animation = lambda *args, **kwargs: None   # no figures

    ref = {}
    with open(os.path.join(a.ref, "gpt_standard.csv"), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ref[(int(r["env_id"]), r["start"], r["goal"])] = r

    with open(a.dataset) as f:
        envs = json.load(f)

    rows, t0 = [], time.time()
    for env in envs:
        for sg in env["start_goal"]:
            start, goal = list(sg[0]), list(sg[1])
            key = (int(env["id"]), json.dumps(start), json.dumps(goal))
            if key not in ref:
                key = (int(env["id"]), str(start), str(goal))
            if key not in ref:
                continue
            if a.limit is not None and len(rows) >= a.limit:
                break
            query = {"start": copy.deepcopy(start), "goal": copy.deepcopy(goal),
                     "size": [env["range_x"][1], env["range_y"][1]],
                     "horizontal_barriers": copy.deepcopy(env["horizontal_barriers"]),
                     "vertical_barriers": copy.deepcopy(env["vertical_barriers"]),
                     "range_x": copy.deepcopy(env["range_x"]), "range_y": copy.deepcopy(env["range_y"])}
            with contextlib.redirect_stdout(io.StringIO()):
                res = lla.LLMAStar(llm="gpt", prompt="standard").searching(query=query)
            r = ref[key]
            rows.append([int(env["id"]), r["start"], r["goal"],
                         float(r["astar_ops"]), float(r["astar_storage"]), float(r["astar_length"]),
                         res["operation"], res["storage"], res["length"], len(res["llm_output"]) - 2])
        if a.limit is not None and len(rows) >= a.limit:
            break
        if int(env["id"]) % 10 == 9:
            print(f"  env {env['id']} done, {len(rows)} cases, {time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(a.out, "no_waypoints.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_id", "start", "goal", "astar_ops", "astar_storage", "astar_length",
                    "nowp_ops", "nowp_storage", "nowp_length", "waypoints"])
        w.writerows(rows)

    env_ids = np.array([r[0] for r in rows])
    ln = {"OR": np.log(np.array([r[6] / r[3] for r in rows])),
          "SR": np.log(np.array([r[7] / r[4] for r in rows])),
          "RL": np.log(np.array([r[8] / r[5] for r in rows]))}
    envs_u, k = np.unique(env_ids, return_inverse=True)
    cnt = np.bincount(k).astype(float)
    draws = np.random.default_rng(a.seed).integers(0, len(envs_u), size=(a.boot, len(envs_u)))
    print(f"\nLLM-A* search with an empty waypoint list ({len(rows)} cases, {len(envs_u)} environments, "
          f"waypoints = {sorted({r[9] for r in rows})})")
    out = []
    for m in ("OR", "SR", "RL"):
        s = np.bincount(k, weights=ln[m], minlength=len(envs_u))
        b = np.exp(s[draws].sum(1) / cnt[draws].sum(1)) * 100
        est, lo, hi = np.exp(ln[m].mean()) * 100, np.percentile(b, 2.5), np.percentile(b, 97.5)
        out.append([m, est, lo, hi, 100 * np.mean(ln[m] > EPS)])
        print(f"  {m}: {est:6.2f}% [{lo:.1f}; {hi:.1f}]   share of cases with ratio > 1: {100 * np.mean(ln[m] > EPS):.1f}%")
    with open(os.path.join(a.out, "no_waypoints_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "gm_pct", "ci_lo", "ci_hi", "pct_cases_gt_1"])
        w.writerows([[x[0]] + [round(v, 3) for v in x[1:]] for x in out])
    print(f"Written in {a.out}/")


if __name__ == "__main__":
    main()

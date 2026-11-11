"""
Weighted-A* baseline on every case, with NO LLM and NO tokens.

The package has no WA* solver. But LLM-A* with only [start, goal] as targets uses
h = dist(s, goal) + dist(s, goal) = 2 * dist(s, goal), i.e. weighted A* with a FIXED
weight w=2. This script forces exactly that: every model call is replaced by an empty
reply, so the package falls back to [start, goal].

NOTE: this is FIXED w=2, NOT the paper's *Dynamic* WA*

Needs table1_benchmark.py in the same folder (it reuses its helpers).

Usage (PowerShell, from the llm-astar folder):
    python wa_baseline.py
    python wa_baseline.py --max-cases 100        # quick test first
Output: results_wa/wa_w2.csv (same columns as the other result files) + a table.
"""
import argparse
import copy
import csv
import json
import os

# Dummy keys so the model classes can be constructed. No request is ever sent,
# because ask() is replaced below.
os.environ.setdefault("GROQ_API_KEY", "unused")
os.environ.setdefault("GEMINI_API_KEY", "unused")

import table1_benchmark as tb
from llmastar.pather import AStar, LLMAStar
from llmastar import model as llm_model

for _name in ("ChatGPT", "Llama3"):
    _cls = getattr(llm_model, _name, None)
    if _cls is not None:
        _cls.ask = lambda self, *a, **k: ""          # empty reply -> [start, goal]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset/environment_50_30.json")
    p.add_argument("--outdir", default="results_wa")
    p.add_argument("--max-cases", type=int, default=None)
    args = p.parse_args()

    with open(args.dataset, "r") as f:
        environments = json.load(f)
    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "wa_w2.csv")

    n = errors = unexpected_wpts = 0
    ors, srs, rls, valids = [], [], [], 0
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(tb.FIELDS)
        for env in environments:
            obstacles = tb.get_obstacles(env)
            bounds = (tuple(env["range_x"]), tuple(env["range_y"]))
            for sg in env["start_goal"]:
                if args.max_cases is not None and n >= args.max_cases:
                    break
                start, goal = list(sg[0]), list(sg[1])
                query = {"start": start, "goal": goal,
                         "size": [env["range_x"][1], env["range_y"][1]],
                         "horizontal_barriers": env["horizontal_barriers"],
                         "vertical_barriers": env["vertical_barriers"],
                         "range_x": env["range_x"], "range_y": env["range_y"]}
                n += 1
                row_status, err = "ok", ""
                a_ops = a_st = a_len = l_ops = l_st = l_len = None
                nw = None
                valid = False
                try:
                    _, a_ops, a_st, a_len = tb.parse_result(AStar().searching(query=copy.deepcopy(query)))
                    tb._last["path"] = None
                    res = LLMAStar(llm="gpt", prompt="standard").searching(query=copy.deepcopy(query))
                    _, l_ops, l_st, l_len = tb.parse_result(res)
                    targets = res.get("llm_output") or []
                    nw = max(len(targets) - 2, 0)
                    if nw:
                        unexpected_wpts += 1
                    valid = tb.path_is_valid(tb._last["path"], start, goal, obstacles, bounds)
                except Exception as e:
                    row_status, err = "error", f"{type(e).__name__}: {e}"
                    errors += 1

                orr = l_ops / a_ops if (row_status == "ok" and a_ops) else None
                sr = l_st / a_st if (row_status == "ok" and a_st) else None
                rl = l_len / a_len if (row_status == "ok" and a_len and l_len is not None) else None
                if orr is not None:
                    ors.append(orr)
                    srs.append(sr)
                    if valid and rl is not None:
                        rls.append(rl)
                    valids += valid
                w.writerow([env.get("id"), json.dumps(start), json.dumps(goal),
                            tb.fmt(a_ops), tb.fmt(a_st), tb.fmt(a_len),
                            tb.fmt(l_ops), tb.fmt(l_st), tb.fmt(l_len),
                            "" if orr is None else f"{orr:.4f}",
                            "" if sr is None else f"{sr:.4f}",
                            "" if rl is None else f"{rl:.4f}",
                            tb.fmt(nw), valid, row_status, err])
                if n % 100 == 0:
                    print(f"  {n} cases done")
            if args.max_cases is not None and n >= args.max_cases:
                break

    print(f"\nCases run: {n}   errors: {errors}")
    if unexpected_wpts:
        print(f"WARNING: {unexpected_wpts} cases got waypoints, so the fallback did not apply to them.")
    if ors:
        print("\nWA* (w=2), geometric means (%):")
        print(f"  OR  = {tb.geo_mean(ors) * 100:.2f}")
        print(f"  SR  = {tb.geo_mean(srs) * 100:.2f}")
        print(f"  RL  = {tb.geo_mean(rls) * 100:.2f}")
        print(f"  VPR = {valids / len(ors) * 100:.2f}")
        print("\nPaper, Dynamic WA* (w=2): OR 60.91, SR 78.53, RL 100.24, VPR 100")
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()
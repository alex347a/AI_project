"""
Split the Table 1 numbers by whether the LLM actually produced waypoints.

With 0 waypoints, LLM-A* searches with only [start, goal], which behaves like
weighted A* (w=2): no LLM guidance at all. So the 0-waypoint group shows what the
weighting alone gives, and the 1+-waypoint group shows cases where the LLM steered.

Reads <results>/gpt_standard.csv, gpt_cot.csv, gpt_repe.csv. No LLM calls, no tokens.

Usage (PowerShell, from the llm-astar folder):
    python split_by_waypoints.py --results results_gemini
"""
import argparse
import csv
import os

import numpy as np


def load(folder, llm, prompt):
    rows = {}
    path = os.path.join(folder, f"{llm}_{prompt}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}")
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok" and r.get("op_ratio") not in ("", None):
                rows[(r["env_id"], r["start"], r["goal"])] = r
    return rows


def gm(vals):
    v = [x for x in vals if x > 0]
    return float(np.exp(np.mean(np.log(v)))) * 100 if v else float("nan")


def summarize(rows):
    valid = [r for r in rows if r["is_valid"] == "True" and r.get("length_ratio")]
    return (len(rows),
            gm([float(r["op_ratio"]) for r in rows]),
            gm([float(r["storage_ratio"]) for r in rows]),
            gm([float(r["length_ratio"]) for r in valid]))


def line(label, rows):
    if not rows:
        return f"  {label:34} n=   0"
    n, o, s, l = summarize(rows)
    return f"  {label:34} n={n:4}  OR={o:6.2f}  SR={s:6.2f}  RL={l:6.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results_gemini")
    p.add_argument("--llm", default="gpt")
    p.add_argument("--prompts", default="standard,cot,repe")
    args = p.parse_args()
    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]
    data = {pr: load(args.results, args.llm, pr) for pr in prompts}

    print("\n=== Per prompt: all cases, and split by waypoints (OR/SR/RL are geometric means, %) ===")
    for pr in prompts:
        rows = list(data[pr].values())
        none = [r for r in rows if r["llm_waypoints"] in ("", "0")]
        some = [r for r in rows if r["llm_waypoints"] not in ("", "0")]
        print(f"\n{pr}")
        print(line("all cases", rows))
        print(line("0 waypoints (= weighted A*, w=2)", none))
        print(line("1+ waypoints (LLM steered)", some))

    common = set.intersection(*[set(d) for d in data.values()])
    print(f"\n=== Same cases for all prompts ({len(common)} common cases) ===")
    zero = {k for k in common if all(data[pr][k]["llm_waypoints"] in ("", "0") for pr in prompts)}
    some = {k for k in common if all(data[pr][k]["llm_waypoints"] not in ("", "0") for pr in prompts)}
    print(f"  all prompts gave 0 waypoints: {len(zero)} cases;  all gave 1+: {len(some)} cases;  "
          f"mixed: {len(common) - len(zero) - len(some)}")
    for pr in prompts:
        print(f"\n{pr}")
        print(line("all prompts 0 waypoints", [data[pr][k] for k in zero]))
        print(line("all prompts 1+ waypoints", [data[pr][k] for k in some]))

    print("\nReading it: the 0-waypoint rows are the weighted-A* effect alone (paper's Dynamic WA* "
          "w=2: OR 60.91, SR 78.53, RL 100.24).")
    print("The 1+-waypoint rows show what the LLM's waypoints give on top. Caution: the two groups "
          "are different cases (the LLM may skip waypoints on easy or hard maps), so treat the "
          "difference as indicative, not proof.")


if __name__ == "__main__":
    main()
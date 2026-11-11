"""
Table 1 variant: LLM-A* where cases with 0 waypoints use dynamic WA* instead of the
package's fixed-w=2 fallback. No LLM calls, no new searches: it merges existing result files.

  - cases where Gemini gave waypoints  -> keep the Gemini row
  - cases where Gemini gave 0 waypoints -> take that case's row from the dynamic WA* file

Usage (PowerShell, from the llm-astar folder):
    python merge_dwa_fallback.py --results results_gemini --dwa results_dwa/dwa_refresh_w2_d0.99.csv
Writes merged_table.csv.
"""
import argparse
import csv

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--results", default="results_gemini")
p.add_argument("--dwa", default="results_dwa/dwa_refresh_w2_d0.99.csv")
p.add_argument("--prompts", default="standard,cot,repe")
p.add_argument("--out", default="merged_table.csv")
args = p.parse_args()


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["env_id"], r["start"], r["goal"]): r for r in csv.DictReader(f) if r["status"] == "ok"}


def gm(vals):
    v = [x for x in vals if x > 0]
    return float(np.exp(np.mean(np.log(v)))) * 100 if v else float("nan")


def metrics(rows):
    valid = [r for r in rows if r["is_valid"] == "True" and r["length_ratio"] != ""]
    return (gm([float(r["op_ratio"]) for r in rows]),
            gm([float(r["storage_ratio"]) for r in rows]),
            gm([float(r["length_ratio"]) for r in valid]),
            len(valid) / len(rows) * 100)


dwa = load(args.dwa)
header = ["Prompt", "Version", "OR(%)", "SR(%)", "RL(%)", "VPR(%)", "n", "replaced"]
table = []
for pr in args.prompts.split(","):
    rows = load(f"{args.results}/gpt_{pr}.csv")
    merged, replaced, mismatch = [], 0, 0
    for k, r in rows.items():
        if r["llm_waypoints"] in ("", "0"):
            if k not in dwa:
                raise SystemExit(f"Case {k} missing from {args.dwa}")
            d = dwa[k]
            if d["astar_ops"] != r["astar_ops"]:
                mismatch += 1
            merged.append(d)
            replaced += 1
        else:
            merged.append(r)
    if mismatch:
        print(f"WARNING {pr}: {mismatch} replaced cases have a different A* baseline in the two files")
    for label, rs, rep in [("as run (fixed w=2 fallback)", list(rows.values()), 0),
                           ("dynamic WA* fallback", merged, replaced)]:
        o, s, l, v = metrics(rs)
        table.append([pr, label, f"{o:.2f}", f"{s:.2f}", f"{l:.2f}", f"{v:.2f}", len(rs), rep])

widths = [max(len(str(r[i])) for r in [header] + table) for i in range(len(header))]
print()
for r in [header] + table:
    print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
with open(args.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(table)
print(f"\nSaved -> {args.out}")
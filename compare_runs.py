"""
Compare two benchmark runs (for example Groq gpt-oss vs Gemini) on the SAME cases.

Reads the per-config CSV files that table1_benchmark.py wrote (one folder per run),
keeps only the cases that EVERY config in BOTH folders has finished, and prints the
Table 1 metrics side by side. It makes no API calls, so it uses no tokens.

Usage (PowerShell, from the llm-astar folder):
    python compare_runs.py --a results --b results_gemini --names "gpt-oss (Groq),Gemini"
"""
import argparse
import csv
import os

import numpy as np


def load_ok_rows(path):
    rows = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    rows[(r["env_id"], r["start"], r["goal"])] = r
    return rows


def geo_mean(vals):
    """The paper aggregates OR / SR / RL with the geometric mean."""
    pos = [v for v in vals if v > 0]
    return float(np.exp(np.mean(np.log(pos)))) if pos else float("nan")


def metrics(rows):
    num = lambda col: [float(r[col]) for r in rows if r.get(col, "") != ""]
    valid = [r for r in rows if r["is_valid"] == "True"]
    rl = [float(r["length_ratio"]) for r in valid if r.get("length_ratio", "") != ""]
    n = len(rows)
    wp = sum(1 for r in rows if r.get("llm_waypoints", "") not in ("", "0"))
    return {
        "OR": geo_mean(num("op_ratio")) * 100,
        "SR": geo_mean(num("storage_ratio")) * 100,
        "RL": geo_mean(rl) * 100,
        "VPR": len(valid) / n * 100,
        "wpts": wp / n * 100,
    }


def main():
    p = argparse.ArgumentParser(description="Compare two runs on their common cases")
    p.add_argument("--a", default="results", help="folder of the first run")
    p.add_argument("--b", default="results_gemini", help="folder of the second run")
    p.add_argument("--names", default="run A,run B", help='two labels, e.g. "gpt-oss (Groq),Gemini"')
    p.add_argument("--configs", default="gpt:standard,gpt:cot,gpt:repe")
    p.add_argument("--out", default="comparison.csv")
    args = p.parse_args()

    names = [s.strip() for s in args.names.split(",")]
    folders = [args.a, args.b]
    configs = [tuple(c.strip().split(":")) for c in args.configs.split(",") if c.strip()]

    data = {}
    for i, folder in enumerate(folders):
        for llm, prompt in configs:
            data[(i, llm, prompt)] = load_ok_rows(os.path.join(folder, f"{llm}_{prompt}.csv"))

    empty = [f"{names[i]} {l}:{p_}" for (i, l, p_), rows in data.items() if not rows]
    if empty:
        raise SystemExit("No finished cases found for: " + ", ".join(empty))

    common = set.intersection(*[set(r) for r in data.values()])
    print(f"\nCases finished per run/config:")
    for (i, l, p_), rows in data.items():
        print(f"  {names[i]:22} {l}:{p_:9} {len(rows)}")
    print(f"\nComparison uses the {len(common)} cases that ALL of these finished.")
    if len(common) < 30:
        print("WARNING: small sample, so differences of a few points are within noise.")
    if not common:
        raise SystemExit("No common cases.")

    order = sorted(common)
    head = ["Config", "Run", "OR(%)", "SR(%)", "RL(%)", "VPR(%)", "wpts(%)"]
    table, by_run = [], {i: {} for i in range(2)}
    for llm, prompt in configs:
        for i in range(2):
            m = metrics([data[(i, llm, prompt)][k] for k in order])
            by_run[i][f"{llm}:{prompt}"] = m
            table.append([f"{llm}:{prompt}", names[i], f"{m['OR']:.2f}", f"{m['SR']:.2f}",
                          f"{m['RL']:.2f}", f"{m['VPR']:.2f}", f"{m['wpts']:.1f}"])

    widths = [max(len(str(r[j])) for r in [head] + table) for j in range(len(head))]
    print()
    for r in [head] + table:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))

    print("\nRanking of the prompts by operation ratio (lower = better):")
    for i in range(2):
        rank = sorted(by_run[i], key=lambda k: by_run[i][k]["OR"])
        print(f"  {names[i]:22} " + "  <  ".join(f"{k} ({by_run[i][k]['OR']:.1f})" for k in rank))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head + ["cases_compared"])
        for r in table:
            w.writerow(r + [len(common)])
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
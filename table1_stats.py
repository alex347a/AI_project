"""Offline Table 1 statistics for LLM-A* (no API key, no GPU needed).

Reads the per-case CSVs written by table1_benchmark.py and recomputes:
  * OR / SR / RL as geometric means of per-case ratios (in %), as in the paper
  * VPR (share of valid paths)
  * SD of ln(ratio) over cases, and SE / 95% CI from a cluster bootstrap over
    environments (cases that share a map are correlated)

Usage:
    python table1_stats.py --dir results_gemini_1000
    python table1_stats.py --dir results_gemini_1000 --out table1_with_stats.csv
"""
import argparse
import csv
import os

import numpy as np

PROMPTS = ["standard", "cot", "repe"]


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("status", "ok") != "ok":
                continue
            try:
                a_ops, l_ops = float(r["astar_ops"]), float(r["llm_ops"])
                a_sto, l_sto = float(r["astar_storage"]), float(r["llm_storage"])
                a_len, l_len = float(r["astar_length"]), float(r["llm_length"])
            except ValueError:
                continue
            if min(a_ops, l_ops, a_sto, l_sto, a_len, l_len) <= 0:
                continue
            rows.append((int(r["env_id"]), l_ops / a_ops, l_sto / a_sto,
                         l_len / a_len, r["is_valid"] == "True"))
    return rows


def cluster_bootstrap_gm(env_ids, logs, n_boot, rng):
    """Geometric mean (in %) with cluster bootstrap over environments."""
    envs = np.unique(env_ids)
    idx = {e: i for i, e in enumerate(envs)}
    k = np.array([idx[e] for e in env_ids])
    sums = np.bincount(k, weights=logs, minlength=len(envs))
    cnts = np.bincount(k, minlength=len(envs)).astype(float)
    draws = rng.integers(0, len(envs), size=(n_boot, len(envs)))
    boot = np.exp(sums[draws].sum(1) / cnts[draws].sum(1)) * 100
    return boot


def summarize(rows, n_boot, seed):
    rng = np.random.default_rng(seed)
    env = np.array([r[0] for r in rows])
    out = {"n": len(rows), "n_envs": len(np.unique(env)),
           "VPR": 100.0 * np.mean([r[4] for r in rows])}
    for j, name in ((1, "OR"), (2, "SR"), (3, "RL")):
        logs = np.log(np.array([r[j] for r in rows]))
        boot = cluster_bootstrap_gm(env, logs, n_boot, rng)
        out[name] = float(np.exp(logs.mean()) * 100)
        out[name + "_lo"], out[name + "_hi"] = np.percentile(boot, [2.5, 97.5])
        out[name + "_sd_ln"] = float(logs.std(ddof=1))      # SD of ln(ratio) per case
        out[name + "_se_boot"] = float(np.std(np.log(boot / 100), ddof=1))  # SE of ln(GM)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results_gemini_1000")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional CSV to write")
    a = p.parse_args()

    table = []
    for pr in PROMPTS:
        path = os.path.join(a.dir, f"gpt_{pr}.csv")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        table.append((pr, summarize(load(path), a.boot, a.seed)))
    if not table:
        raise SystemExit(f"No gpt_*.csv found in '{a.dir}'")

    print(f"\nTable 1 (LLM-A*, geometric means, 95% cluster-bootstrap CI over environments)\n")
    print(f"{'prompt':9s}{'n':>6s}{'OR % [CI]':>26s}{'SR % [CI]':>26s}{'RL %':>9s}{'VPR %':>8s}")
    for pr, s in table:
        print(f"{pr:9s}{s['n']:6d}"
              f"{s['OR']:11.2f} [{s['OR_lo']:.1f};{s['OR_hi']:.1f}]"
              f"{s['SR']:11.2f} [{s['SR_lo']:.1f};{s['SR_hi']:.1f}]"
              f"{s['RL']:9.2f}{s['VPR']:8.2f}")
    print(f"\nSpread of ln(ratio) per case (SD) and bootstrap SE of ln(geometric mean)\n")
    print(f"{'prompt':9s}{'SD ln OR':>10s}{'SD ln SR':>10s}{'SD ln RL':>10s}{'SE ln OR':>10s}{'SE ln SR':>10s}{'SE ln RL':>10s}")
    for pr, s in table:
        print(f"{pr:9s}{s['OR_sd_ln']:10.3f}{s['SR_sd_ln']:10.3f}{s['RL_sd_ln']:10.3f}"
              f"{s['OR_se_boot']:10.3f}{s['SR_se_boot']:10.3f}{s['RL_se_boot']:10.3f}")

    if a.out:
        cols = ["prompt", "n", "n_envs", "OR", "OR_lo", "OR_hi", "SR", "SR_lo", "SR_hi",
                "RL", "RL_lo", "RL_hi", "VPR", "OR_sd_ln", "SR_sd_ln", "RL_sd_ln",
                "OR_se_boot", "SR_se_boot", "RL_se_boot"]
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for pr, s in table:
                w.writerow([pr] + [round(s[c], 4) if isinstance(s[c], float) else s[c] for c in cols[1:]])
        print(f"\nWritten: {a.out}")


if __name__ == "__main__":
    main()

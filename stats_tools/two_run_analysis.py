"""Two-run analysis of the LLM-A* Table 1 reproduction (numpy only, no API key).

Inputs : two directories with gpt_standard.csv / gpt_cot.csv / gpt_repe.csv, written by
         table1_benchmark.py (same 1,000 cases, run twice with the same model and settings).
Outputs: CSV files in --out and a printed summary. Everything is computed from the
         saved per-case results.

Conventions
  * ratios are LLM-A* / A* per case; OR = operations, SR = storage, RL = path length
  * aggregation = geometric mean (mean of ln ratio, back-transformed), as in the paper
  * uncertainty = cluster bootstrap over the 100 environments (10 pairs share a map)
  * "pooled" = per-case mean of ln ratio over the two runs
Usage:
    python stats_tools/two_run_analysis.py --run-a results_gemini_1000 --run-b results_gemini --out two_run_out
"""
import argparse
import csv
import os

import numpy as np

PROMPTS = ["standard", "cot", "repe"]
METRICS = ["OR", "SR", "RL"]
EPS = 1e-9   # ratios closer to 1 than this are ties (floating-point noise differs between platforms)

# Table 1 of the original paper (LLM-A* rows), copied from docs/static/pdfs/LLMAstar.pdf
PAPER = {
    "GPT-3.5": {"standard": (57.39, 74.96, 102.44), "cot": (69.50, 83.65, 102.54), "repe": (85.47, 96.53, 102.41)},
    "LLaMA3": {"standard": (44.59, 64.02, 102.47), "cot": (47.60, 66.27, 102.46), "repe": (64.08, 80.19, 102.54)},
}


def load(directory, prompt):
    rows = {}
    with open(os.path.join(directory, f"gpt_{prompt}.csv"), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("status", "ok") != "ok":
                continue
            rows[(int(r["env_id"]), r["start"], r["goal"])] = r
    return rows


def build(dir_a, dir_b):
    data = {}
    for p in PROMPTS:
        a, b = load(dir_a, p), load(dir_b, p)
        keys = sorted(set(a) & set(b))
        data[p] = {"keys": keys}
        for tag, d in (("A", a), ("B", b)):
            g = lambda col: np.array([float(d[k][col]) for k in keys])
            data[p][tag] = {
                "ln": {"OR": np.log(g("llm_ops") / g("astar_ops")),
                       "SR": np.log(g("llm_storage") / g("astar_storage")),
                       "RL": np.log(g("llm_length") / g("astar_length"))},
                "ops": g("llm_ops"), "astar_ops": g("astar_ops"),
                "wp": np.array([int(d[k]["llm_waypoints"]) for k in keys]),
                "valid": np.array([d[k]["is_valid"] == "True" for k in keys]),
            }
        data[p]["env"] = np.array([k[0] for k in keys])
    envs = [data[p]["env"] for p in PROMPTS]
    assert all((envs[0] == e).all() for e in envs), "case sets differ between prompts"
    return data


class Boot:
    """Cluster bootstrap over environments, shared resampling for every statistic."""

    def __init__(self, env, n_boot, seed):
        self.envs, k = np.unique(env, return_inverse=True)
        self.k = k
        self.cnt = np.bincount(k).astype(float)
        rng = np.random.default_rng(seed)
        self.draws = rng.integers(0, len(self.envs), size=(n_boot, len(self.envs)))
        self.den = self.cnt[self.draws].sum(1)

    def mean(self, x):
        s = np.bincount(self.k, weights=x, minlength=len(self.envs))
        return s[self.draws].sum(1) / self.den

    def stat(self, x):
        """estimate, 95% CI, SE (all on the ln scale) of mean(x)."""
        b = self.mean(x)
        return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), b.std(ddof=1), b


def pval(b):
    """two-sided bootstrap p-value for 'mean = 0' (floor at 1/n_boot)."""
    n = len(b)
    return max(2 * min((b <= 0).mean(), (b >= 0).mean()), 1.0 / n)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([round(v, 4) if isinstance(v, float) else v for v in r])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", default="results_gemini_1000")
    ap.add_argument("--run-b", default="results_gemini")
    ap.add_argument("--out", default="two_run_out")
    ap.add_argument("--ablation", default=None,
                    help="no_waypoints.csv written by ablation_no_waypoints.py (optional, adds section 7)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    D = build(a.run_a, a.run_b)
    n = len(D["standard"]["keys"])
    bt = Boot(D["standard"]["env"], a.boot, a.seed)
    print(f"{n} common cases, {len(bt.envs)} environments, {a.boot} bootstrap resamples\n")

    pooled = {p: {m: (D[p]["A"]["ln"][m] + D[p]["B"]["ln"][m]) / 2 for m in METRICS} for p in PROMPTS}
    series = lambda p, tag, m: pooled[p][m] if tag == "pooled" else D[p][tag]["ln"][m]

    # 1. Table 1 per run and pooled ------------------------------------------------------
    rows = []
    print("1. Table 1 (geometric means in %, 95% cluster-bootstrap CI)")
    print(f"{'run':7s}{'prompt':10s}{'OR':>22s}{'SR':>22s}{'RL':>10s}{'VPR':>7s}{'wpts':>6s}")
    for tag in ("A", "B", "pooled"):
        for p in PROMPTS:
            r = {}
            for m in METRICS:
                est, lo, hi, se, _ = bt.stat(series(p, tag, m))
                r[m] = (np.exp(est) * 100, np.exp(lo) * 100, np.exp(hi) * 100, se, series(p, tag, m).std(ddof=1))
            if tag == "pooled":
                vpr = 100 * np.mean(np.concatenate([D[p]["A"]["valid"], D[p]["B"]["valid"]]))
                wp = np.mean(np.concatenate([D[p]["A"]["wp"] > 0, D[p]["B"]["wp"] > 0])) * 100
            else:
                vpr = 100 * D[p][tag]["valid"].mean()
                wp = 100 * np.mean(D[p][tag]["wp"] > 0)
            print(f"{tag:7s}{p:10s}{r['OR'][0]:8.2f} [{r['OR'][1]:.1f};{r['OR'][2]:.1f}]"
                  f"{r['SR'][0]:9.2f} [{r['SR'][1]:.1f};{r['SR'][2]:.1f}]{r['RL'][0]:10.2f}{vpr:7.1f}{wp:6.1f}")
            rows.append([tag, p] + [v for m in METRICS for v in r[m]] + [vpr, wp])
    hdr = ["run", "prompt"] + [f"{m}_{s}" for m in METRICS for s in ("gm", "lo", "hi", "se_ln", "sd_ln")] + ["VPR", "pct_cases_with_waypoint"]
    write(os.path.join(a.out, "1_table1_runs_and_pooled.csv"), hdr, rows)

    # 2. Run-to-run noise ------------------------------------------------------------------
    print("\n2. Run-to-run variability (run B / run A, same cases)")
    rows = []
    allp = {m: np.mean([D[p]["B"]["ln"][m] - D[p]["A"]["ln"][m] for p in PROMPTS], axis=0) for m in ("OR", "SR")}
    for p in PROMPTS + ["all prompts"]:
        for m in ("OR", "SR"):
            diff = allp[m] if p == "all prompts" else D[p]["B"]["ln"][m] - D[p]["A"]["ln"][m]
            est, lo, hi, _, b = bt.stat(diff)
            if p == "all prompts":
                same, corr = "", ""
            else:
                la, lb = D[p]["A"]["ln"][m], D[p]["B"]["ln"][m]
                corr = float(np.corrcoef(la, lb)[0, 1])
                same = float(np.mean(D[p]["A"]["ops"] == D[p]["B"]["ops"])) * 100 if m == "OR" else ""
            rows.append([p, m, np.exp(est), np.exp(lo), np.exp(hi), pval(b), corr, same])
            print(f"  {p:12s}{m}: B/A = {np.exp(est):.3f} [{np.exp(lo):.3f}; {np.exp(hi):.3f}] p={pval(b):.3f}"
                  + (f"  corr(ln ratio)={corr:.2f}" if corr != "" else "")
                  + (f"  identical llm_ops={same:.0f}%" if same != "" else ""))
    write(os.path.join(a.out, "2_run_to_run.csv"),
          ["prompt", "metric", "ratio_B_over_A", "lo", "hi", "p_boot", "corr_ln_ratio", "pct_identical_llm_ops"], rows)

    # 3. Prompt contrasts (paired, cluster bootstrap) -------------------------------------
    print("\n3. Prompt contrasts (ratio of geometric means, paired on cases)")
    rows = []
    pairs = [("cot", "standard"), ("repe", "standard"), ("repe", "cot")]
    for tag in ("A", "B", "pooled"):
        for m in ("OR", "SR"):
            for x, y in pairs:
                est, lo, hi, _, b = bt.stat(series(x, tag, m) - series(y, tag, m))
                rows.append([tag, m, f"{x}/{y}", np.exp(est), np.exp(lo), np.exp(hi), pval(b)])
                if tag == "pooled" or m == "OR":
                    print(f"  {tag:7s}{m} {x}/{y}: {np.exp(est):.3f} [{np.exp(lo):.3f}; {np.exp(hi):.3f}] p={pval(b):.3f}")
    write(os.path.join(a.out, "3_prompt_contrasts.csv"), ["run", "metric", "contrast", "ratio", "lo", "hi", "p_boot"], rows)

    # 4. Is the prompt a bigger source of per-case variation than the run? ------------------
    print("\n4. Correlation of per-case ln(OR): same prompt in the other run vs. other prompt in the same run")
    cols = {f"{p}_{t}": D[p][t]["ln"]["OR"] for p in PROMPTS for t in ("A", "B")}
    names = list(cols)
    C = np.corrcoef(np.array([cols[k] for k in names]))
    write(os.path.join(a.out, "4_corr_lnOR.csv"), [""] + names, [[names[i]] + list(C[i]) for i in range(len(names))])
    same_prompt = [C[names.index(f"{p}_A"), names.index(f"{p}_B")] for p in PROMPTS]
    diff_prompt = [C[names.index(f"{x}_{t}"), names.index(f"{y}_{t}")] for t in ("A", "B") for x, y in pairs]
    print(f"  same prompt, A vs B   : {np.round(same_prompt, 2)}  (mean {np.mean(same_prompt):.2f})")
    print(f"  other prompt, same run: {np.round(diff_prompt, 2)}  (mean {np.mean(diff_prompt):.2f})")

    # 5. Where do the savings come from? (pooled over runs and prompts; case level) --------
    print("\n5. Breakdown (all 6 prompt x run series, geometric mean of OR in %)")
    ln_all = np.concatenate([D[p][t]["ln"]["OR"] for p in PROMPTS for t in ("A", "B")])
    wp_all = np.concatenate([D[p][t]["wp"] for p in PROMPTS for t in ("A", "B")])
    aops = np.concatenate([D[p][t]["astar_ops"] for p in PROMPTS for t in ("A", "B")])
    pr_all = np.concatenate([[p] * n for p in PROMPTS for t in ("A", "B")])
    rows = []
    for lab, mask in (("0 waypoint", wp_all == 0), ("1 waypoint", wp_all == 1),
                      ("2 waypoints", wp_all == 2), ("3+ waypoints", wp_all >= 3)):
        s = {p: (mask & (pr_all == p)) for p in PROMPTS}
        line = [lab, int(mask.sum())] + [np.exp(ln_all[mask].mean()) * 100] + \
               [np.exp(ln_all[s[p]].mean()) * 100 if s[p].sum() >= 30 else float("nan") for p in PROMPTS] + \
               [int(s[p].sum()) for p in PROMPTS]
        rows.append(line)
        print(f"  {lab:13s}n={line[1]:5d}  OR={line[2]:6.1f}   by prompt (std/cot/repe): "
              + "/".join("  n/a" if np.isnan(v) else f"{v:5.1f}" for v in line[3:6]) + f"   n={line[6:]}")
    write(os.path.join(a.out, "5a_by_waypoints.csv"),
          ["group", "n", "OR_all", "OR_standard", "OR_cot", "OR_repe", "n_standard", "n_cot", "n_repe"], rows)
    q = np.quantile(aops, [1 / 3, 2 / 3])
    rows = []
    for lab, mask in ((f"A* ops <= {q[0]:.0f} (easy)", aops <= q[0]),
                      (f"{q[0]:.0f} < A* ops <= {q[1]:.0f}", (aops > q[0]) & (aops <= q[1])),
                      (f"A* ops > {q[1]:.0f} (hard)", aops > q[1])):
        rows.append([lab, int(mask.sum()), np.exp(ln_all[mask].mean()) * 100, 100 * np.mean(ln_all[mask] > EPS)])
        print(f"  {lab:24s}n={rows[-1][1]:5d}  OR={rows[-1][2]:6.1f}  share OR>1={rows[-1][3]:4.1f}%")
    write(os.path.join(a.out, "5b_by_difficulty.csv"), ["group", "n", "OR_gm", "pct_OR_gt_1"], rows)
    zero = wp_all == 0
    print(f"  cases without waypoint: OR exactly 1 in {100 * np.mean(np.abs(ln_all[zero]) < EPS):.0f}% of them")
    for p in PROMPTS:
        mean_wp = np.mean(np.concatenate([D[p]["A"]["wp"], D[p]["B"]["wp"]]))
        print(f"  mean number of waypoints, {p}: {mean_wp:.2f}")

    # 6. Comparison with the paper (pooled and each run) ------------------------------------
    print("\n6. Ours / paper (ratio of geometric means); Holm-adjusted within the 12 comparisons of each series")
    rows, ps = [], []
    for tag in ("A", "B", "pooled"):
        for model, vals in PAPER.items():
            for p in PROMPTS:
                for j, m in enumerate(("OR", "SR")):
                    x = series(p, tag, m)
                    est, lo, hi, _, b = bt.stat(x - np.log(vals[p][j] / 100))
                    rows.append([tag, model, p, m, np.exp(est * 1) * vals[p][j] / vals[p][j], np.exp(lo), np.exp(hi),
                                 np.exp(x.mean()) * 100, vals[p][j]])
                    ps.append(pval(b))
    # Holm correction within each family = the 12 comparisons of one run/pooled series
    # (2 paper models x 3 prompts x 2 metrics)
    adj = np.empty(len(ps))
    for tag in ("A", "B", "pooled"):
        idx = [i for i, r in enumerate(rows) if r[0] == tag]
        adj[idx] = holm([ps[i] for i in idx])
    for r, pr, pa in zip(rows, ps, adj):
        r += [pr, pa]
    write(os.path.join(a.out, "6_vs_paper.csv"),
          ["run", "paper_model", "prompt", "metric", "ratio_ours_over_paper", "lo", "hi", "ours_gm", "paper", "p_boot", "p_holm"], rows)
    for r in rows:
        if r[0] == "pooled":
            print(f"  {r[1]:8s}{r[2]:9s}{r[3]}: ours {r[7]:6.2f} vs paper {r[8]:6.2f}  ratio {r[4]:.3f} [{r[5]:.3f}; {r[6]:.3f}]  p_holm={r[10]:.4f}")

    # 7. LLM waypoints vs. the same search with an EMPTY waypoint list (no LLM) ---------------
    if a.ablation:
        ab = {}
        with open(a.ablation, newline="") as f:
            for r in csv.DictReader(f):
                ab[(int(r["env_id"]), r["start"], r["goal"])] = r
        keys = D["standard"]["keys"]
        miss = [k for k in keys if k not in ab]
        if miss:
            print(f"\n7. skipped: {len(miss)} cases missing from {a.ablation}")
        else:
            nw = {"OR": np.log(np.array([float(ab[k]["nowp_ops"]) / float(ab[k]["astar_ops"]) for k in keys])),
                  "SR": np.log(np.array([float(ab[k]["nowp_storage"]) / float(ab[k]["astar_storage"]) for k in keys])),
                  "RL": np.log(np.array([float(ab[k]["nowp_length"]) / float(ab[k]["astar_length"]) for k in keys]))}
            nw_ops = np.array([float(ab[k]["nowp_ops"]) for k in keys])
            print("\n7. LLM-A* with Gemini waypoints vs. the same search with an empty waypoint list")
            est0 = {m: bt.stat(nw[m]) for m in METRICS}
            print("   no-waypoint search: " + ", ".join(
                f"{m} {np.exp(est0[m][0]) * 100:.2f} [{np.exp(est0[m][1]) * 100:.1f}; {np.exp(est0[m][2]) * 100:.1f}]" for m in METRICS))
            # sanity check: when the LLM gave no waypoint, the search must be identical
            same = []
            for p in PROMPTS:
                for t in ("A", "B"):
                    z = D[p][t]["wp"] == 0
                    same.append(np.mean(D[p][t]["ops"][z] == nw_ops[z]))
            print(f"   sanity check: cases without LLM waypoint reproduce the empty-list search exactly in {100 * np.mean(same):.1f}% of them")
            rows = []
            for p in PROMPTS:
                for m in METRICS:
                    d = pooled[p][m] - nw[m]
                    est, lo, hi, _, b = bt.stat(d)
                    better = 100 * np.mean(d < -EPS)
                    rows.append([p, m, np.exp(est), np.exp(lo), np.exp(hi), pval(b), better])
                    print(f"   {p:9s}{m}: LLM / no-waypoint = {np.exp(est):.3f} [{np.exp(lo):.3f}; {np.exp(hi):.3f}] p={pval(b):.4f}"
                          f"   waypoints better in {better:.0f}% of cases")
            write(os.path.join(a.out, "7_vs_no_waypoint.csv"),
                  ["prompt", "metric", "ratio_llm_over_nowp", "lo", "hi", "p_boot", "pct_cases_llm_better"], rows)
            wp_p = {p: (D[p]["A"]["wp"] + D[p]["B"]["wp"]) / 2 for p in PROMPTS}
            allwp = np.concatenate([D[p][t]["wp"] for p in PROMPTS for t in ("A", "B")])
            lnd = np.concatenate([D[p][t]["ln"]["OR"] - nw["OR"] for p in PROMPTS for t in ("A", "B")])
            print("   excess OR over the no-waypoint search, by number of waypoints (all series):")
            for lab, mask in (("1 waypoint", allwp == 1), ("2 waypoints", allwp == 2), ("3+ waypoints", allwp >= 3)):
                print(f"     {lab:13s}n={int(mask.sum()):5d}  LLM/no-waypoint OR = {np.exp(lnd[mask].mean()):.3f}")

    print(f"\nFiles written in {a.out}/")


if __name__ == "__main__":
    main()

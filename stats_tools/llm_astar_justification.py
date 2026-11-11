#!/usr/bin/env python3
"""
llm_astar_justification.py - statistical justification of the gap between our Table 1 and the paper's.

Usage:
  python llm_astar_justification.py --run1 results_gemini_1000 --run2 results_gemini --out just_out

run1 = complete 1000-case Gemini run (git commit 548b661), run2 = later partial rerun (625 cases common to 3 prompts).
All inference is on ln(ratio) (geometric mean), resampling unit = environment (cluster bootstrap on env_id).
"""
import argparse, os, math
import numpy as np, pandas as pd
from scipy import stats

CFG = ["standard", "cot", "repe"]
MET = {"OR": "op_ratio", "SR": "storage_ratio", "RL": "length_ratio"}
PAPER = {  # Table 1, LLM-A*: (OR, SR, RL) in %
    "GPT-3.5": {"standard": (57.39, 74.96, 102.44), "cot": (69.50, 83.65, 102.54), "repe": (85.47, 96.53, 102.41)},
    "LLaMA3": {"standard": (44.59, 64.02, 102.47), "cot": (47.60, 66.27, 102.46), "repe": (64.08, 80.19, 102.54)},
}
MI = {"OR": 0, "SR": 1, "RL": 2}


def load(d, c):
    df = pd.read_csv(os.path.join(d, f"gpt_{c}.csv"))
    df = df[df.get("status", "ok") == "ok"].copy()
    df["key"] = df.env_id.astype(str) + "|" + df.start + "|" + df.goal
    return df.set_index("key")


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj


def env_sums(env, y, envs):
    s = pd.Series(y).groupby(env).agg(["sum", "count"]).reindex(envs).fillna(0)
    return s["sum"].values, s["count"].values


def boot_means(parts, B, rng, envs):
    """parts: list of (env, y). One shared resample of environments -> list of bootstrap means."""
    k = len(envs); idx = rng.integers(0, k, size=(B, k))
    out = []
    for env, y in parts:
        s, c = env_sums(env, y, envs)
        out.append(s[idx].sum(1) / c[idx].sum(1))
    return out


def lg(df, col):
    d = df if col != "length_ratio" else df[df.is_valid]
    return d.env_id.values, np.log(d[col].values), d.index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", required=True); ap.add_argument("--run2", required=True)
    ap.add_argument("--out", default="just_out"); ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(1)
    r1 = {c: load(a.run1, c) for c in CFG}; r2 = {c: load(a.run2, c) for c in CFG}
    S = set.intersection(*[set(d.index) for d in r2.values()])
    sets = {"run1_1000": r1, "run2_625": {c: d.loc[sorted(S)] for c, d in r2.items()}}
    envs_all = np.arange(100)
    pd.set_option("display.width", 250)

    # ---------- 1. one-sample tests vs paper ----------
    rows = []
    est = {}
    for sname, data in sets.items():
        for c in CFG:
            for m, col in MET.items():
                env, y, _ = lg(data[c], col)
                (bm,) = boot_means([(env, y)], a.B, rng, envs_all if sname == "run1_1000" else np.unique(env))
                mean, n, k = y.mean(), len(y), len(np.unique(env))
                se_naive = y.std(ddof=1) / math.sqrt(n)
                se_cl = bm.std(ddof=1)
                est[(sname, c, m)] = dict(mean=mean, se=se_cl, sd=y.std(ddof=1), n=n, k=k)
                for pm, tab in PAPER.items():
                    lp = math.log(tab[c][MI[m]] / 100)
                    d = mean - lp
                    z = d / se_cl; p = 2 * (1 - stats.norm.cdf(abs(z)))
                    lo, hi = np.percentile(bm, [2.5, 97.5]) - lp
                    rows.append(dict(dataset=sname, config=c, metric=m, paper_model=pm, n=n, envs=k,
                                     ours=100 * math.exp(mean), paper=tab[c][MI[m]], sd_log=y.std(ddof=1),
                                     se_naive=se_naive, se_cluster=se_cl,
                                     diff_pts=100 * (math.exp(mean) - tab[c][MI[m]] / 100),
                                     ratio_ours_over_paper=math.exp(d), ci_lo=math.exp(lo), ci_hi=math.exp(hi),
                                     z=z, p=p, d_z=d / y.std(ddof=1)))
    t1 = pd.DataFrame(rows)
    # conservative variant: paper SE assumed equal to our 1000-run cluster SE (same sd/ICC/n); sensitivity only
    t1["se_paper_assumed"] = [est[("run1_1000", r.config, r.metric)]["se"] for r in t1.itertuples()]
    t1["z_conservative"] = np.log(t1.ratio_ours_over_paper) / np.sqrt(t1.se_cluster ** 2 + t1.se_paper_assumed ** 2)
    t1["p_conservative"] = 2 * (1 - stats.norm.cdf(t1.z_conservative.abs()))
    t1["p_holm"] = np.nan; t1["p_holm_conservative"] = np.nan
    for (ds, pm), g in t1.groupby(["dataset", "paper_model"]):
        t1.loc[g.index, "p_holm"] = holm(g.p.values)
        t1.loc[g.index, "p_holm_conservative"] = holm(g.p_conservative.values)
    t1.to_csv(f"{a.out}/1_tests_vs_paper.csv", index=False)
    show = t1[(t1.paper_model == "GPT-3.5") & (t1.metric != "RL")]
    print("=== 1. Ours vs paper GPT-3.5 (cluster-bootstrap SE; paper treated as a known constant) ===")
    print(show[["dataset", "config", "metric", "ours", "paper", "diff_pts", "ratio_ours_over_paper", "ci_lo", "ci_hi", "se_cluster", "z", "p_holm", "p_holm_conservative", "d_z"]].round(3).to_string(index=False))
    print("\n=== 1b. vs LLaMA3 ===")
    show = t1[(t1.paper_model == "LLaMA3") & (t1.metric != "RL")]
    print(show[["dataset", "config", "metric", "ours", "paper", "diff_pts", "ci_lo", "ci_hi", "z", "p_holm", "p_holm_conservative"]].round(3).to_string(index=False))
    print("\n=== 1c. RL ===")
    show = t1[(t1.metric == "RL")]
    print(show[["dataset", "config", "paper_model", "ours", "paper", "diff_pts", "ci_lo", "ci_hi", "p_holm"]].round(3).to_string(index=False))

    # ---------- 2. prompt-effect contrasts (difference of differences) ----------
    rows = []
    for sname, data in sets.items():
        for m, col in MET.items():
            if m == "RL": continue
            for x, y_ in (("cot", "standard"), ("repe", "standard"), ("repe", "cot")):
                A, Bd = data[x], data[y_]
                common = A.index.intersection(Bd.index)
                A, Bd = A.loc[common], Bd.loc[common]
                d = np.log(A[col].values) - np.log(Bd[col].values)
                env = A.env_id.values
                (bm,) = boot_means([(env, d)], a.B, rng, np.unique(env))
                se = bm.std(ddof=1); mean = d.mean()
                for pm, tab in PAPER.items():
                    pc = math.log(tab[x][MI[m]] / tab[y_][MI[m]])
                    z = (mean - pc) / se
                    rows.append(dict(dataset=sname, metric=m, contrast=f"{x}/{y_}", ours_ratio=math.exp(mean),
                                     ours_lo=math.exp(np.percentile(bm, 2.5)), ours_hi=math.exp(np.percentile(bm, 97.5)),
                                     paper_model=pm, paper_ratio=math.exp(pc), z=z, p=2 * (1 - stats.norm.cdf(abs(z)))))
    t2 = pd.DataFrame(rows); t2.to_csv(f"{a.out}/2_prompt_contrasts.csv", index=False)
    print("\n=== 2. Prompt effect (ratio of OR/SR between prompts) : ours vs paper ===")
    print(t2.round(3).to_string(index=False))

    # ---------- 3. run-to-run noise and decomposition 1000-run vs 625-run ----------
    rows = []
    for c in CFG:
        A1 = r1[c]; A2 = r2[c].loc[sorted(S)]
        A1S = A1.loc[A2.index]
        for m, col in MET.items():
            if m == "RL": continue
            e1A, y1A, _ = lg(A1, col); e1S, y1S, _ = lg(A1S, col); e2S, y2S, _ = lg(A2, col)
            bm2S, bm1S, bm1A = boot_means([(e2S, y2S), (e1S, y1S), (e1A, y1A)], a.B, rng, envs_all)
            def ci(v): return np.percentile(v, [2.5, 97.5])
            noise, cov, tot = bm2S - bm1S, bm1S - bm1A, bm2S - bm1A
            pt = dict(noise=y2S.mean() - y1S.mean(), cov=y1S.mean() - y1A.mean(), tot=y2S.mean() - y1A.mean())
            for nm, v, key in (("run noise (run2 vs run1, same 625 cases)", noise, "noise"),
                               ("coverage (run1: 625 subset vs all 1000)", cov, "cov"),
                               ("total (run2 625 vs run1 1000)", tot, "tot")):
                lo, hi = ci(v)
                rows.append(dict(config=c, metric=m, component=nm, ratio=math.exp(pt[key]), lo=math.exp(lo), hi=math.exp(hi),
                                 p=2 * (1 - stats.norm.cdf(abs(pt[key] / v.std(ddof=1))))))
            # per-case reproducibility
            if m == "OR":
                rr = np.corrcoef(y1S, y2S)[0, 1]
                same = (A1S.llm_ops.values == A2.llm_ops.values).mean()
                rows.append(dict(config=c, metric=m, component=f"per-case corr(ln OR run1, ln OR run2)={rr:.2f}; identical llm_ops={same:.0%}; noise share of variance~{1 - rr:.0%}",
                                 ratio=np.nan, lo=np.nan, hi=np.nan, p=np.nan))
    t3 = pd.DataFrame(rows); t3.to_csv(f"{a.out}/3_run_noise_decomposition.csv", index=False)
    print("\n=== 3. Why do the 625-run and the 1000-run differ? (ratios; 1 = no effect) ===")
    print(t3.round(3).to_string(index=False))

    # ---------- 4. normality / robustness ----------
    rows = []
    for sname, data in sets.items():
        for c in CFG:
            env, y, _ = lg(data[c], "op_ratio")
            gm = 100 * math.exp(y.mean()); am = 100 * np.exp(y).mean()
            rows.append(dict(dataset=sname, config=c, n=len(y), GM_OR=gm, arith_OR=am, median_OR=100 * math.exp(np.median(y)),
                             skew=stats.skew(y), exkurt=stats.kurtosis(y), shapiro_p=stats.shapiro(y).pvalue,
                             share_OR_gt1=(y > 0).mean()))
    t4 = pd.DataFrame(rows); t4.to_csv(f"{a.out}/4_distribution_checks.csv", index=False)
    print("\n=== 4. Distribution of ln(OR) ===")
    print(t4.round(3).to_string(index=False))

    # ---------- 5. minimal detectable difference ----------
    rows = []
    for (sname, c, m), e in est.items():
        if m == "RL": continue
        mde_log = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * e["se"]
        base = 100 * math.exp(e["mean"])
        rows.append(dict(dataset=sname, config=c, metric=m, GM=base, se_log=e["se"], MDE_pts_80pct_power=base * (math.exp(mde_log) - 1)))
    t5 = pd.DataFrame(rows); t5.to_csv(f"{a.out}/5_min_detectable_diff.csv", index=False)
    print("\n=== 5. Minimal detectable difference vs a constant reference (alpha=.05, power=.8) ===")
    print(t5.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

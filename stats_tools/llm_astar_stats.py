#!/usr/bin/env python3
"""
llm_astar_stats.py - sample-size / precision analysis for the LLM-A* Table 1 replication.

Usage:
    python llm_astar_stats.py --dir results_gemini            # expects gpt_standard.csv, gpt_cot.csv, gpt_repe.csv
    python llm_astar_stats.py --dir results_gemini --B 10000 --out stats_out

Columns required: env_id,start,goal,op_ratio,storage_ratio,length_ratio,is_valid
Everything is computed on ln(ratio); aggregation = geometric mean (as in the paper).
Resampling unit = environment (cluster bootstrap on env_id).
"""
import argparse, os, math
import numpy as np
import pandas as pd
from scipy import stats

CONFIGS = ["standard", "cot", "repe"]
METRICS = {"OR": "op_ratio", "SR": "storage_ratio", "RL": "length_ratio"}
# Paper Table 1, LLM-A*, GPT-3.5 (OR, SR, RL) and LLaMA3 (paper uses "Few-Shot" for our "standard")
PAPER = {
    "GPT-3.5": {"standard": (57.39, 74.96, 102.44), "cot": (69.50, 83.65, 102.54), "repe": (85.47, 96.53, 102.41)},
    "LLaMA3":  {"standard": (44.59, 64.02, 102.47), "cot": (47.60, 66.27, 102.46), "repe": (64.08, 80.19, 102.54)},
}
Z = 1.959964


def load(d, cfg):
    df = pd.read_csv(os.path.join(d, f"gpt_{cfg}.csv"))
    df = df[df.get("status", "ok") == "ok"].copy()
    df["key"] = df.env_id.astype(str) + "|" + df.start + "|" + df.goal
    return df


def logs(df, metric):
    d = df if metric != "length_ratio" else df[df.is_valid]
    d = d[(d[metric] > 0) & np.isfinite(d[metric])]
    return d.env_id.values, np.log(d[metric].values)


def cluster_stats(env, y):
    """per-env sums and counts"""
    s = pd.DataFrame({"e": env, "y": y}).groupby("e")["y"].agg(["sum", "count"])
    return s["sum"].values, s["count"].values.astype(float)


def icc_anova(env, y):
    g = pd.DataFrame({"e": env, "y": y}).groupby("e")["y"]
    ni = g.count().values.astype(float); mi = g.mean().values
    k, N = len(ni), ni.sum()
    gm = y.mean()
    ssb = (ni * (mi - gm) ** 2).sum()
    dfm = pd.DataFrame({"e": env, "y": y})
    ssw = ((dfm.y - dfm.groupby("e").y.transform("mean")) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (ni ** 2).sum() / N) / (k - 1)
    return max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))


def boot(env, y, B, rng):
    sy, cnt = cluster_stats(env, y)
    k = len(sy)
    idx = rng.integers(0, k, size=(B, k))
    return sy[idx].sum(1) / cnt[idx].sum(1)


def analyse(df, B, rng):
    rows = []
    for name, col in METRICS.items():
        env, y = logs(df, col)
        n, k = len(y), len(np.unique(env))
        sd = y.std(ddof=1)
        mean = y.mean()
        se = sd / math.sqrt(n)
        h_naive = Z * se
        bm = boot(env, y, B, rng)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        icc = icc_anova(env, y)
        m = n / k
        deff = 1 + (m - 1) * icc
        # empirical design effect from bootstrap
        deff_boot = (bm.std(ddof=1) / se) ** 2
        rows.append(dict(metric=name, n=n, envs=k, GM=100 * math.exp(mean), sd_log=sd,
                         naive_lo=100 * math.exp(mean - h_naive), naive_hi=100 * math.exp(mean + h_naive),
                         boot_lo=100 * math.exp(lo), boot_hi=100 * math.exp(hi),
                         rel_up=100 * (math.exp(hi - mean) - 1), rel_dn=100 * (1 - math.exp(lo - mean)),
                         ICC=icc, m=m, deff_icc=deff, deff_boot=deff_boot,
                         skew=stats.skew(y), exkurt=stats.kurtosis(y)))
    return pd.DataFrame(rows)


def n_required(sd, prec, deff=1.0, z=Z):
    eps = math.log(1 + prec)
    return math.ceil((z * sd / eps) ** 2 * deff)


def halfwidth_rel(sd, n_eff, z=Z):
    return 100 * (math.exp(z * sd / math.sqrt(n_eff)) - 1)


def budget_row(sd, icc, n, E=100, N=1000, z=Z):
    """Relative upper half-width for n cases drawn STRATIFIED over E environments (n/E per env, equal allocation).
    gen: target = unseen environments (no FPC): Var = sd^2/n * (1+(m-1)icc), m=n/E (m<=1 -> n distinct envs, deff=1)
    fin: target = these 1000 cases (FPC): only within-env variance remains for stratified sampling:
         Var = sd^2 (1-icc)/n * (1-n/N)"""
    m = max(n / E, 1.0)
    deff = 1 + (m - 1) * icc
    gen = halfwidth_rel(sd, n / deff)
    fin_var = sd ** 2 * (1 - icc) / n * (1 - n / N)
    fin = 100 * (math.exp(z * math.sqrt(fin_var)) - 1) if fin_var > 0 else 0.0
    # naive "first n cases" = complete envs: m=10, E=n/10
    deff_first = 1 + 9 * icc
    first = halfwidth_rel(sd, n / deff_first)
    return gen, fin, first


def paired(a, b, B, rng, metric="op_ratio"):
    m = a[["key", "env_id", metric]].merge(b[["key", metric]], on="key", suffixes=("_a", "_b"))
    d = np.log(m[metric + "_a"].values) - np.log(m[metric + "_b"].values)
    ya, yb = np.log(m[metric + "_a"].values), np.log(m[metric + "_b"].values)
    rho = np.corrcoef(ya, yb)[0, 1]
    env = m.env_id.values
    bm = boot(env, d, B, rng)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    se = d.std(ddof=1) / math.sqrt(len(d))
    z = d.mean() / se
    zc = d.mean() / bm.std(ddof=1)
    return dict(n=len(d), rho=rho, sd_diff=d.std(ddof=1), ratio=math.exp(d.mean()),
                lo=math.exp(lo), hi=math.exp(hi), z_naive=z, z_cluster=zc,
                p_cluster=2 * (1 - stats.norm.cdf(abs(zc))), icc_diff=icc_anova(env, d),
                deff_diff=(bm.std(ddof=1) / se) ** 2)


def n_for_diff(sd_diff, delta_log, deff=1.0, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return math.ceil((za + zb) ** 2 * sd_diff ** 2 * deff / delta_log ** 2)


def clopper_pearson_lower(k, n, alpha=0.05):
    return 1.0 if k == n and n == 0 else (0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1))


def stratified_nested_plan(env_ids, pairs_per_env=10, seed=42):
    """Nested stratified plan for the 1000 cases: shuffle the 10 pairs inside each env once, then
    level L (1..10) = the first L pairs of every env. n=100 -> L=1, 200 -> L=2, ... 1000 -> L=10.
    For n<100 take n random envs with their level-1 pair."""
    rng = np.random.default_rng(seed)
    envs = sorted(set(env_ids))
    order = {e: rng.permutation(pairs_per_env) for e in envs}
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_gemini")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--out", default="stats_out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    data = {c: load(a.dir, c) for c in CONFIGS}
    common = set.intersection(*[set(d.key) for d in data.values()])
    print(f"cases: " + ", ".join(f"{c}={len(d)}" for c, d in data.items()) + f"; common={len(common)}")
    for c, d in data.items():
        per = d.groupby("env_id").size()
        print(f"  {c}: envs {per.index.min()}-{per.index.max()} ({len(per)} envs), pairs/env min={per.min()} max={per.max()}, "
              f"incomplete envs={(per < 10).sum()}, valid={d.is_valid.mean():.3f}")

    # ---- A. all available cases per config ----
    allres = []
    for c, d in data.items():
        r = analyse(d, a.B, rng); r.insert(0, "config", c); r.insert(1, "set", "all_available"); allres.append(r)
    # ---- B. common cases (the ones used in table1_replication.csv) ----
    for c, d in data.items():
        r = analyse(d[d.key.isin(common)], a.B, rng); r.insert(0, "config", c); r.insert(1, "set", "common"); allres.append(r)
    res = pd.concat(allres, ignore_index=True)
    res.to_csv(f"{a.out}/summary_by_config.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== Summary (cluster bootstrap on env_id) ===")
    print(res.round(3).to_string(index=False))

    # ---- C. comparison with the paper (common cases) ----
    rows = []
    com = res[res.set == "common"]
    for _, r in com.iterrows():
        for model, tab in PAPER.items():
            idx = {"OR": 0, "SR": 1, "RL": 2}[r.metric]
            pv = tab[r.config][idx]
            rows.append(dict(config=r.config, metric=r.metric, ours=r.GM, ci_lo=r.boot_lo, ci_hi=r.boot_hi,
                             paper_model=model, paper=pv, diff=r.GM - pv, paper_inside_CI=r.boot_lo <= pv <= r.boot_hi))
    cmp_ = pd.DataFrame(rows); cmp_.to_csv(f"{a.out}/comparison_with_paper.csv", index=False)
    print("\n=== Ours vs paper Table 1 (LLM-A*) ===")
    print(cmp_.round(2).to_string(index=False))

    # ---- D. required n (clustered, using measured sd and ICC, all-available stats) ----
    rows = []
    for _, r in res[res.set == "all_available"].iterrows():
        for p in (0.10, 0.05, 0.03):
            n_iid = n_required(r.sd_log, p)
            n_cl = n_required(r.sd_log, p, r.deff_icc)
            floor = Z * r.sd_log * math.sqrt(r.ICC / 100)  # E=100 envs, infinite m
            rows.append(dict(config=r.config, metric=r.metric, precision=p, n_iid=n_iid, n_clustered_m10=n_cl,
                             floor_halfwidth_pct_E100=100 * (math.exp(floor) - 1)))
    req = pd.DataFrame(rows); req.to_csv(f"{a.out}/required_n.csv", index=False)
    print("\n=== Required n ===")
    print(req.round(2).to_string(index=False))

    # ---- E. budget table (worst config on OR / SR) ----
    ref = res[(res.set == "all_available") & (res.metric == "OR")]
    sd_or, icc_or = ref.sd_log.max(), ref.ICC.max()
    refs = res[(res.set == "all_available") & (res.metric == "SR")]
    sd_sr, icc_sr = refs.sd_log.max(), refs.ICC.max()
    rows = []
    for n in (50, 100, 200, 300, 400, 500, 625, 1000):
        g, f, fi = budget_row(sd_or, icc_or, n)
        gs, fs, fis = budget_row(sd_sr, icc_sr, n)
        rows.append(dict(n=n, OR_new_envs=g, OR_these1000_FPC=f, OR_first_n_cases=fi, SR_new_envs=gs, SR_these1000_FPC=fs))
    bud = pd.DataFrame(rows); bud.to_csv(f"{a.out}/budget_plan.csv", index=False)
    print(f"\n=== Budget plan (stratified, upper rel. half-width %, sd_OR={sd_or:.3f}, ICC_OR={icc_or:.3f}, sd_SR={sd_sr:.3f}, ICC_SR={icc_sr:.3f}) ===")
    print(bud.round(1).to_string(index=False))

    # ---- F. paired comparisons ----
    rows = []
    for x, y in (("standard", "cot"), ("standard", "repe"), ("cot", "repe")):
        for name, col in (("OR", "op_ratio"), ("SR", "storage_ratio")):
            r = paired(data[x][data[x].key.isin(common)], data[y][data[y].key.isin(common)], a.B, rng, col)
            r.update(pair=f"{x} vs {y}", metric=name)
            for X in (3, 5, 10):
                if name == "OR":
                    r[f"n_for_{X}pts"] = n_for_diff(r["sd_diff"], math.log((50 + X) / 50), r["deff_diff"])
            rows.append(r)
    pr = pd.DataFrame(rows); pr.to_csv(f"{a.out}/paired.csv", index=False)
    print("\n=== Paired comparisons (common cases; n_for_Xpts assumes reference OR=50%) ===")
    print(pr.round(3).to_string(index=False))

    # ---- G. VPR ----
    for c, d in data.items():
        k, n = int(d.is_valid.sum()), len(d)
        print(f"VPR {c}: {k}/{n}, Clopper-Pearson 95% lower bound = {100 * clopper_pearson_lower(k, n):.2f}%")
    for c, d in data.items():
        pass
    print("n for VPR>=99% with 0 failures (one-sided 95%): ", math.ceil(math.log(0.05) / math.log(0.99)))


if __name__ == "__main__":
    main()

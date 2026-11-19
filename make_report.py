#!/usr/bin/env python3
"""Build Table 2, Table 3 and Figures 2-6 from the raw experiment output.

Usage:  python make_report.py --in results
"""
import argparse, os, itertools
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hsmp.config import CONFIGS

TABLE2_COLS = [("latency_ms", "Latency (ms)", 2), ("energy_j", "Energy (J)", 4),
               ("available", "Availability Rate", 4), ("stall_s", "Stall (s)", 4),
               ("switch", "Switch Rate", 4), ("cache_hit", "Cache Hit Rate", 4)]
TEST_METRICS = [("latency_ms", "Latency (ms)"), ("energy_j", "Energy (J)"),
                ("switch", "Quality Switch Rate"), ("cache_hit", "Cache Hit Rate")]   # as in Table 3
SUPPLEMENTARY = [("bitrate_mbps", "Mean delivered bitrate (Mbps)"), ("stall_ratio", "Stall ratio")]


def ci_half(x, conf=0.95):
    x = np.asarray(x, float)
    n = len(x)
    return stats.t.ppf(0.5 + conf / 2, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0


def table2(df):
    rows = []
    for s, spec in CONFIGS.items():
        d = df[df.strategy == s]
        r = {"Configuration": spec["label"], "Code": s}
        for col, name, dp in TABLE2_COLS:
            r[name] = f"{d[col].mean():.{dp}f} ± {ci_half(d[col]):.{dp}f}"
        rows.append(r)
    return pd.DataFrame(rows)


def table3(df, alpha=0.05):
    rows = []
    for a, b in itertools.combinations(CONFIGS.keys(), 2):            # HH-HQ, HH-MH, HH-MQ, HQ-MH, HQ-MQ, MH-MQ
        A = df[df.strategy == a].sort_values("seed")
        B = df[df.strategy == b].sort_values("seed")
        for col, name in TEST_METRICS:
            x, y = A[col].values, B[col].values
            d = x - y
            if np.allclose(d, d[0]) and d.std() == 0:                  # zero variance -> t undefined
                t, p, dz = np.nan, np.nan, np.nan
            else:
                t, p = stats.ttest_rel(x, y)
                dz = d.mean() / d.std(ddof=1)                          # paired Cohen's d_z = t/sqrt(n)
            rows.append(dict(Comparison=f"{a} vs {b}", Metric=name, t_value=t, p_value=p,
                             mean_diff=d.mean(), cohen_d=dz))
    t3 = pd.DataFrame(rows)
    t3["Significant (p<0.05)"] = np.where(t3.p_value < alpha, "Yes", "No")
    # Supplementary: Holm-Bonferroni across all 24 tests (NOT in the manuscript)
    p = t3.p_value.values.copy()
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    m = int(np.isfinite(p).sum()); adj = np.full(len(p), np.nan); run = 0.0
    for rank, idx in enumerate(order[:m]):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(1.0, run)
    t3["p_holm"] = adj
    t3["Significant after Holm"] = np.where(t3.p_holm < alpha, "Yes", "No")
    return t3


def supplementary(df):
    rows = []
    for s, spec in CONFIGS.items():
        d = df[df.strategy == s]
        r = {"Code": s}
        for col, name in SUPPLEMENTARY:
            r[name] = f"{d[col].mean():.4f} ± {ci_half(d[col]):.4f}"
        rows.append(r)
    return pd.DataFrame(rows)


def fig_timeseries(ts, cols, strat, seed_idx, path, mean=False):
    labels = [("latency_ms", "Latency (ms)"), ("energy_j", "Energy (J)"), ("available", "Availability"),
              ("stall_s", "Stalls (s)"), ("switch", "Quality Switches"), ("cache_hit", "Cache Hit Rate")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, title) in zip(axes.ravel(), labels):
        j = list(cols).index(k)
        y = ts[strat][:, :, j].mean(0) if mean else ts[strat][seed_idx, :, j]
        ax.plot(np.arange(len(y)), y, lw=2)
        ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel(title); ax.grid(ls="--", alpha=.6)
    tag = "mean over seeds" if mean else f"seed {seed_idx}"
    fig.suptitle(f"Performance Metrics — {CONFIGS[strat]['label']} ({tag})", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(path, dpi=150); plt.close(fig)


def fig_significance(t3, path):
    cnt = t3[t3.p_value < 0.05].groupby("Metric").size().reindex([m for _, m in TEST_METRICS]).fillna(0)
    cnt_h = t3[t3.p_holm < 0.05].groupby("Metric").size().reindex(cnt.index).fillna(0)
    x = np.arange(len(cnt)); w = .38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w / 2, cnt.values, w, label="p < 0.05 (as in manuscript)")
    b2 = ax.bar(x + w / 2, cnt_h.values, w, label="after Holm correction (supplementary)", alpha=.7)
    ax.bar_label(b1); ax.bar_label(b2)
    ax.set_xticks(x); ax.set_xticklabels(cnt.index); ax.set_ylim(0, 6.8)
    ax.set_ylabel("# significant pairwise comparisons (of 6)")
    ax.set_title("Statistically significant differences per metric"); ax.legend(); ax.grid(axis="y", ls="--", alpha=.5)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_means(df, path):
    spec = [("latency_ms", "Latency (ms) ↓"), ("energy_j", "Energy (J) ↓"),
            ("cache_hit", "Cache Hit Rate ↑"), ("switch", "Quality Switch Rate ↓")]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, (k, title) in zip(axes, spec):
        m = [df[df.strategy == s][k].mean() for s in CONFIGS]
        e = [ci_half(df[df.strategy == s][k]) for s in CONFIGS]
        ax.bar(list(CONFIGS), m, yerr=e, color=colors, capsize=4)
        ax.set_title(title, fontweight="bold"); ax.grid(axis="y", ls="--", alpha=.5)
    fig.suptitle("Mean performance of the four strategies (error bars: 95% CI over seeds)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results")
    ap.add_argument("--ts-seed", type=int, default=0, help="index of the seed shown in Figs 2-5")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.inp, "per_seed_summary.csv"))
    z = np.load(os.path.join(a.inp, "timeseries.npz"))
    cols = list(z["columns"]); ts = {s: z[s] for s in CONFIGS}
    out = a.inp
    t2 = table2(df); t3 = table3(df); sup = supplementary(df)
    t2.to_csv(os.path.join(out, "table2_performance.csv"), index=False)
    t3.to_csv(os.path.join(out, "table3_paired_ttests.csv"), index=False)
    sup.to_csv(os.path.join(out, "table_supplementary.csv"), index=False)
    for i, s in enumerate(CONFIGS):
        fig_timeseries(ts, cols, s, a.ts_seed, os.path.join(out, f"fig{i+2}_timeseries_{s}.png"))
    fig_significance(t3, os.path.join(out, "fig6_significance_counts.png"))
    fig_means(df, os.path.join(out, "fig_means_bar.png"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(f"\n=== Table 2 (mean ± 95% CI half-width, n={df.seed.nunique()} seeds) ===")
    print(t2.drop(columns="Code").to_string(index=False))
    print("\n=== Table 3 (paired t-tests) ===")
    print(t3.round({"t_value": 3, "p_value": 4, "mean_diff": 5, "cohen_d": 3, "p_holm": 4}).to_string(index=False))
    print("\n=== Supplementary (not in manuscript) ===")
    print(sup.to_string(index=False))
    print(f"\nFiles written to {out}/")


if __name__ == "__main__":
    main()

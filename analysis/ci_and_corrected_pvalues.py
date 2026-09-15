#!/usr/bin/env python3
"""Confidence intervals and multiple-comparison-corrected p-values for Tables 5-8.

Reads the per-patient, per-region metrics of the main study
(results/main_study/raw_predictions.csv: 12 activations x 73 validation patients)
and writes two files next to it:

  ci_tables.csv        mean, SD and 95 % CI for every activation x metric x region.
                       Two CIs are given: a percentile bootstrap over patients
                       (5 000 resamples, seed 42) and the normal approximation
                       mean +/- 1.96 * SD / sqrt(n).  The bootstrap CI is the one
                       printed in Tables 5-8.
  corrected_pvalues.csv  Wilcoxon signed-rank p-values of every activation against
                       the ReLU baseline, paired by patient, together with Holm and
                       Benjamini-Hochberg adjusted values.  Adjustment is applied
                       within each metric family (Dice, HD95, Sens, Prec) across
                       the 11 activations x 5 regions = 55 comparisons of that family.

Significance marks: '*' p < 0.05, '**' p < 0.01 (the marks used in the paper).

Usage:
    python analysis/ci_and_corrected_pvalues.py [--raw results/main_study/raw_predictions.csv]
                                                 [--outdir results/main_study]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["Dice", "HD95", "Sens", "Prec"]
REGIONS = ["ET", "TC", "WT", "NCR", "ED"]
BASELINE = "relu"
N_BOOT = 5000
SEED = 42


def sig_mark(p):
    if p != p:  # NaN
        return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def holm(p):
    """Holm step-down adjustment (equivalent to statsmodels 'holm')."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def benjamini_hochberg(p):
    """Benjamini-Hochberg step-up adjustment (equivalent to statsmodels 'fdr_bh')."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    # enforce monotonicity from the largest p downwards
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    return adj


def confidence_intervals(df, rng):
    rows = []
    # Activations are visited in order of first appearance in the CSV, drawing
    # from one generator seeded once; this is what reproduces the archived
    # ci_tables.csv bit-for-bit.  Changing the order changes the bootstrap draws.
    for act in df["Activation"].unique():
        sub = df[df["Activation"] == act]
        for metric in METRICS:
            for region in REGIONS:
                x = sub[f"{metric}_{region}"].dropna().to_numpy()
                n = len(x)
                boot = rng.choice(x, size=(N_BOOT, n), replace=True).mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                half = 1.96 * x.std(ddof=1) / np.sqrt(n)
                rows.append({
                    "Activation": act, "Metric": metric, "Region": region, "n": n,
                    "mean": round(x.mean(), 4), "sd": round(x.std(ddof=1), 4),
                    "ci_boot_low": round(lo, 4), "ci_boot_high": round(hi, 4),
                    "ci_norm_low": round(x.mean() - half, 4),
                    "ci_norm_high": round(x.mean() + half, 4),
                })
    return pd.DataFrame(rows)


def corrected_pvalues(df):
    base = df[df["Activation"] == BASELINE]
    out = []
    for metric in METRICS:
        fam = []
        for region in REGIONS:
            col = f"{metric}_{region}"
            for act in sorted(a for a in df["Activation"].unique() if a != BASELINE):
                cur = df[df["Activation"] == act][["Patient", col]]
                m = pd.merge(cur, base[["Patient", col]], on="Patient", suffixes=("", "_base"))
                p = np.nan
                if len(m) and not (m[col] == m[f"{col}_base"]).all():
                    _, p = wilcoxon(m[col], m[f"{col}_base"])
                fam.append({"Metric": metric, "Region": region, "Activation": act, "p_raw": p})
        fam = pd.DataFrame(fam)
        fam["p_holm"] = holm(fam["p_raw"])
        fam["p_bh"] = benjamini_hochberg(fam["p_raw"])
        for c in ["raw", "holm", "bh"]:
            fam[f"sig_{c}"] = fam[f"p_{c}"].map(sig_mark)
        out.append(fam)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    default_raw = os.path.join(here, "..", "results", "main_study", "raw_predictions.csv")
    ap.add_argument("--raw", default=default_raw)
    ap.add_argument("--outdir", default=os.path.dirname(default_raw))
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    print(f"{len(df)} rows, {df['Activation'].nunique()} activations, "
          f"{df['Patient'].nunique()} patients")

    ci = confidence_intervals(df, np.random.default_rng(SEED))
    ci.to_csv(os.path.join(args.outdir, "ci_tables.csv"), index=False)

    pv = corrected_pvalues(df)
    pv.to_csv(os.path.join(args.outdir, "corrected_pvalues.csv"), index=False)

    key = pv[(pv.Metric == "Dice") & (pv.Region == "NCR") & (pv.Activation == "swish")].iloc[0]
    print(f"Swish vs ReLU, NCR Dice: p_raw={key.p_raw:.4g}  Holm={key.p_holm:.4g}  BH={key.p_bh:.4g}")
    print("written:", os.path.join(args.outdir, "ci_tables.csv"),
          os.path.join(args.outdir, "corrected_pvalues.csv"))


if __name__ == "__main__":
    main()

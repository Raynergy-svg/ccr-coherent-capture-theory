#!/usr/bin/env python3
"""
T15 — Multi-Element Simultaneous Coherence Analysis
=====================================================
Certan (2026) | Coherent Capture Theory | APOGEE DR17 OCCAM

Tests whether the 41 C/O-coherent APOGEE clusters are SIMULTANEOUSLY
coherent across multiple abundance ratios (C/O, Mg/Fe, Si/Fe, Al/Fe, Fe/H).
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

STAR_FILE    = "tapogee_matched_stars.csv"
CLUSTER_FILE = "tapogee_cluster_stats_with_age.csv"
PLOT_FILE    = "t15_multielement_plot.png"
RESULTS_FILE = "t15_results.txt"
MATRIX_FILE  = "t15_coherence_matrix.csv"

CO_THRESH   = 0.05
MGFE_THRESH = 0.03
SIFE_THRESH = 0.03
ALFE_THRESH = 0.05
FEH_THRESH  = 0.03
MIN_MEMBERS = 3
SENTINEL    = -9000

def cluster_scatter(stars_df, col):
    results = {}
    for clust, grp in stars_df.groupby("CLUSTER"):
        vals = grp[col].dropna()
        vals = vals[vals > SENTINEL]
        if len(vals) >= MIN_MEMBERS:
            results[clust] = vals.std(ddof=1)
    return results

def main():
    out = []
    def info(msg):
        line = "[INFO] " + str(msg)
        print(line, flush=True)
        out.append(line)

    info("=" * 72)
    info("T15  Multi-Element Simultaneous Coherence Analysis")
    info("Certan (2026) | CCT | APOGEE DR17 OCCAM")
    info("=" * 72)

    stars  = pd.read_csv(STAR_FILE)
    cstats = pd.read_csv(CLUSTER_FILE)
    info("Loaded " + str(len(stars)) + " stars, " + str(len(cstats)) + " clusters")

    coherent_names = set(cstats.loc[cstats["C_O_std"] < CO_THRESH, "cluster"])
    info("C/O-coherent clusters (std < " + str(CO_THRESH) + "): " + str(len(coherent_names)))

    # Compute per-cluster scatter for each element
    info("-" * 60)
    info("Computing per-cluster scatter for each abundance ratio...")
    co_sc   = cluster_scatter(stars, "C_O")
    mgfe_sc = cluster_scatter(stars, "MG_FE")
    sife_sc = cluster_scatter(stars, "SI_FE")
    alfe_sc = cluster_scatter(stars, "AL_FE")
    feh_sc  = cluster_scatter(stars, "FE_H")

    info("  C/O:   " + str(len(co_sc)) + " clusters")
    info("  Mg/Fe: " + str(len(mgfe_sc)) + " clusters")
    info("  Si/Fe: " + str(len(sife_sc)) + " clusters")
    info("  Al/Fe: " + str(len(alfe_sc)) + " clusters")
    info("  Fe/H:  " + str(len(feh_sc)) + " clusters")

    all_names = sorted(set(co_sc) | set(mgfe_sc) | set(sife_sc) | set(alfe_sc) | set(feh_sc))
    sdf = pd.DataFrame({
        "cluster":  all_names,
        "CO_std":   [co_sc.get(n, np.nan) for n in all_names],
        "MgFe_std": [mgfe_sc.get(n, np.nan) for n in all_names],
        "SiFe_std": [sife_sc.get(n, np.nan) for n in all_names],
        "AlFe_std": [alfe_sc.get(n, np.nan) for n in all_names],
        "FeH_std":  [feh_sc.get(n, np.nan) for n in all_names],
    })

    age_map = dict(zip(cstats["cluster"], cstats["age_gyr"]))
    sdf["age_gyr"] = sdf["cluster"].map(age_map)
    sdf["CO_coherent"] = sdf["cluster"].isin(coherent_names)

    # Spearman correlation matrix
    info("-" * 60)
    info("Spearman rank correlations among element scatter values:")
    ratio_cols   = ["CO_std", "MgFe_std", "SiFe_std", "AlFe_std", "FeH_std"]
    ratio_labels = ["C/O", "Mg/Fe", "Si/Fe", "Al/Fe", "Fe/H"]

    complete = sdf.dropna(subset=ratio_cols)
    info("  Clusters with all 5 scatter values: " + str(len(complete)))

    corr = np.full((5, 5), np.nan)
    pval = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            if i == j:
                corr[i, j], pval[i, j] = 1.0, 0.0
            else:
                corr[i, j], pval[i, j] = stats.spearmanr(complete[ratio_cols[i]], complete[ratio_cols[j]])

    info("")
    header = "          " + "".join(f"{l:>10}" for l in ratio_labels)
    info(header)
    for i, lab in enumerate(ratio_labels):
        row_str = f"{lab:>10}" + "".join(f"{corr[i,j]:>10.3f}" for j in range(5))
        info(row_str)

    info("")
    for i in range(5):
        for j in range(i+1, 5):
            sig = "***" if pval[i,j] < 0.001 else "**" if pval[i,j] < 0.01 else "*" if pval[i,j] < 0.05 else "n.s."
            info(f"  {ratio_labels[i]:>5} vs {ratio_labels[j]:<5}: rho={corr[i,j]:+.3f}  p={pval[i,j]:.2e}  {sig}")

    # Simultaneous coherence
    info("-" * 60)
    info("Simultaneous coherence analysis:")

    sdf["MgFe_coh"] = sdf["MgFe_std"] < MGFE_THRESH
    sdf["SiFe_coh"] = sdf["SiFe_std"] < SIFE_THRESH
    sdf["AlFe_coh"] = sdf["AlFe_std"] < ALFE_THRESH
    sdf["FeH_coh"]  = sdf["FeH_std"]  < FEH_THRESH

    n_valid = len(complete)
    co_set   = set(sdf.loc[sdf["CO_coherent"], "cluster"])
    mgfe_set = set(sdf.loc[sdf["MgFe_coh"].fillna(False), "cluster"])
    sife_set = set(sdf.loc[sdf["SiFe_coh"].fillna(False), "cluster"])
    alfe_set = set(sdf.loc[sdf["AlFe_coh"].fillna(False), "cluster"])
    feh_set  = set(sdf.loc[sdf["FeH_coh"].fillna(False), "cluster"])
    valid_names = set(complete["cluster"])

    n_co   = len(co_set & valid_names)
    n_mgfe = len(mgfe_set & valid_names)
    n_sife = len(sife_set & valid_names)
    n_alfe = len(alfe_set & valid_names)
    n_feh  = len(feh_set & valid_names)

    info(f"  C/O  coherent: {n_co}/{n_valid} = {n_co/max(n_valid,1):.1%}")
    info(f"  Mg/Fe coherent: {n_mgfe}/{n_valid} = {n_mgfe/max(n_valid,1):.1%}")
    info(f"  Si/Fe coherent: {n_sife}/{n_valid} = {n_sife/max(n_valid,1):.1%}")
    info(f"  Al/Fe coherent: {n_alfe}/{n_valid} = {n_alfe/max(n_valid,1):.1%}")
    info(f"  Fe/H  coherent: {n_feh}/{n_valid} = {n_feh/max(n_valid,1):.1%}")

    co_valid = co_set & valid_names
    n_co_v = len(co_valid)

    co_mgfe = len(co_valid & mgfe_set)
    co_sife = len(co_valid & sife_set)
    co_alfe = len(co_valid & alfe_set)
    co_feh  = len(co_valid & feh_set)
    co_all3 = len(co_valid & mgfe_set & sife_set & alfe_set)

    info(f"\n  Among {n_co_v} C/O-coherent clusters:")
    info(f"    Also Mg/Fe coherent: {co_mgfe} ({co_mgfe/max(n_co_v,1):.0%})")
    info(f"    Also Si/Fe coherent: {co_sife} ({co_sife/max(n_co_v,1):.0%})")
    info(f"    Also Al/Fe coherent: {co_alfe} ({co_alfe/max(n_co_v,1):.0%})")
    info(f"    Also Fe/H coherent:  {co_feh} ({co_feh/max(n_co_v,1):.0%})")
    info(f"    All 3 (Mg+Si+Al):    {co_all3} ({co_all3/max(n_co_v,1):.0%})")

    # Expected if independent
    f_co   = n_co / max(n_valid, 1)
    f_mgfe = n_mgfe / max(n_valid, 1)
    f_sife = n_sife / max(n_valid, 1)
    f_alfe = n_alfe / max(n_valid, 1)
    exp_co_mgfe = f_co * f_mgfe * n_valid
    exp_all     = f_co * f_mgfe * f_sife * f_alfe * n_valid

    info(f"\n  Expected if independent:")
    info(f"    C/O & Mg/Fe: {exp_co_mgfe:.1f}  (observed: {co_mgfe})")
    info(f"    C/O & all-3: {exp_all:.1f}  (observed: {co_all3})")
    if exp_co_mgfe > 0:
        info(f"    Enrichment C/O&Mg/Fe: {co_mgfe/exp_co_mgfe:.1f}x over random")
    if exp_all > 0:
        info(f"    Enrichment all-4:     {co_all3/exp_all:.1f}x over random")

    # Fisher's exact tests
    info("-" * 60)
    info("Fisher's exact test: C/O coherence vs Mg/Fe coherence")
    a = len((co_set & mgfe_set) & valid_names)
    b = len((co_set - mgfe_set) & valid_names)
    c = len((mgfe_set - co_set) & valid_names)
    d = len(valid_names - co_set - mgfe_set)
    info(f"                    Mg/Fe coh   Mg/Fe NOT")
    info(f"    C/O coherent      {a:>4}        {b:>4}")
    info(f"    C/O NOT           {c:>4}        {d:>4}")
    odds_r, fisher_p = stats.fisher_exact([[a, b], [c, d]])
    info(f"  Odds ratio = {odds_r:.2f},  p = {fisher_p:.4e}")
    info(f"  {'=> SIGNIFICANT' if fisher_p < 0.05 else '=> Not significant'}")

    info("")
    info("Fisher's exact test: C/O coherence vs Si/Fe coherence")
    a2 = len((co_set & sife_set) & valid_names)
    b2 = len((co_set - sife_set) & valid_names)
    c2 = len((sife_set - co_set) & valid_names)
    d2 = len(valid_names - co_set - sife_set)
    odds2, fp2 = stats.fisher_exact([[a2, b2], [c2, d2]])
    info(f"  Odds ratio = {odds2:.2f},  p = {fp2:.4e}")

    # Multi-element coherence score
    info("-" * 60)
    info("Multi-element coherence score (lower = more coherent):")
    score_df = complete.copy()
    for col in ratio_cols:
        vmin, vmax = score_df[col].min(), score_df[col].max()
        score_df[col + "_norm"] = (score_df[col] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    norm_cols = [c + "_norm" for c in ratio_cols]
    score_df["coherence_score"] = score_df[norm_cols].mean(axis=1)
    score_df = score_df.sort_values("coherence_score")

    info(f"  Top 15 most multi-element-coherent clusters:")
    info(f"  {'Rank':>4}  {'Cluster':<22} {'Score':>6} {'CO':>7} {'MgFe':>7} {'SiFe':>7} {'AlFe':>7} {'FeH':>7}  {'CCT':>4}")
    for rank, (_, row) in enumerate(score_df.head(15).iterrows(), 1):
        flag = "YES" if row["CO_coherent"] else "no"
        info(f"  {rank:>4}  {row['cluster']:<22} {row['coherence_score']:.4f} "
             f"{row['CO_std']:.4f}  {row['MgFe_std']:.4f}  {row['SiFe_std']:.4f}  "
             f"{row['AlFe_std']:.4f}  {row['FeH_std']:.4f}   {flag:>4}")

    top10_co = score_df.head(10)["CO_coherent"].sum()
    info(f"\n  Of top 10 multi-element-coherent: {top10_co}/10 are C/O-coherent")

    # Mann-Whitney U tests
    info("-" * 60)
    info("Mann-Whitney U: C/O-coherent vs C/O-incoherent scatter")
    for col, label in [("MgFe_std","Mg/Fe"), ("SiFe_std","Si/Fe"), ("AlFe_std","Al/Fe"), ("FeH_std","Fe/H")]:
        coh_v = sdf.loc[sdf["CO_coherent"] & sdf[col].notna(), col]
        inc_v = sdf.loc[~sdf["CO_coherent"] & sdf[col].notna(), col]
        if len(coh_v) >= 3 and len(inc_v) >= 3:
            u, p = stats.mannwhitneyu(coh_v, inc_v, alternative="less")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            info(f"  {label:>5}: coh median={coh_v.median():.4f}(n={len(coh_v)})  "
                 f"inc median={inc_v.median():.4f}(n={len(inc_v)})  U={u:.0f} p={p:.4e} {sig}")

    # Summary
    info("")
    info("=" * 72)
    info("T15 SUMMARY")
    info("=" * 72)
    upper = [corr[i,j] for i in range(5) for j in range(i+1,5)]
    mean_rho = np.mean(upper)
    info(f"Mean Spearman rho across all scatter pairs: {mean_rho:.3f}")
    info(f"C/O-coherent also Mg/Fe coherent: {co_mgfe}/{n_co_v} ({co_mgfe/max(n_co_v,1):.0%})")
    info(f"C/O-coherent also Si/Fe coherent: {co_sife}/{n_co_v} ({co_sife/max(n_co_v,1):.0%})")
    info(f"C/O-coherent also Al/Fe coherent: {co_alfe}/{n_co_v} ({co_alfe/max(n_co_v,1):.0%})")
    info(f"Simultaneously coherent in all 4: {co_all3}/{n_co_v}")
    info(f"Fisher (C/O vs Mg/Fe): OR={odds_r:.2f}, p={fisher_p:.4e}")
    info(f"Top cluster: {score_df.iloc[0]['cluster']} (score={score_df.iloc[0]['coherence_score']:.4f})")

    if mean_rho > 0.3:
        info("=> STRONG multi-element coherence: clusters tight in one channel")
        info("   are tight in ALL channels. Multi-dimensional chemical fingerprint CONFIRMED.")
    elif mean_rho > 0:
        info("=> Positive scatter correlations: partial multi-element coherence detected.")
    else:
        info("=> No clear multi-element correlation.")

    # Save CSV
    score_map = dict(zip(score_df["cluster"], score_df["coherence_score"]))
    sdf["coherence_score"] = sdf["cluster"].map(score_map)
    save_cols = ["cluster", "CO_std", "MgFe_std", "SiFe_std", "AlFe_std", "FeH_std",
                 "CO_coherent", "MgFe_coh", "SiFe_coh", "AlFe_coh", "FeH_coh", "age_gyr", "coherence_score"]
    sdf[[c for c in save_cols if c in sdf.columns]].sort_values("CO_std").to_csv(MATRIX_FILE, index=False)
    info(f"Saved: {MATRIX_FILE}")

    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(out))
    info(f"Saved: {RESULTS_FILE}")

    # PLOT
    info("Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("T15 Multi-Element Simultaneous Coherence | Certan (2026) | APOGEE DR17 OCCAM",
                 fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: Correlation heatmap
    ax1 = axes[0, 0]
    im = ax1.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax1.set_xticks(range(5)); ax1.set_yticks(range(5))
    ax1.set_xticklabels(ratio_labels, fontsize=10, rotation=45, ha="right")
    ax1.set_yticklabels(ratio_labels, fontsize=10)
    for i in range(5):
        for j in range(5):
            sig_s = ""
            if i != j:
                sig_s = "***" if pval[i,j]<0.001 else "**" if pval[i,j]<0.01 else "*" if pval[i,j]<0.05 else ""
            c = "white" if abs(corr[i,j]) > 0.5 else "black"
            ax1.text(j, i, f"{corr[i,j]:.2f}\n{sig_s}", ha="center", va="center", fontsize=9, color=c, fontweight="bold")
    plt.colorbar(im, ax=ax1, shrink=0.8, label="Spearman rho")
    ax1.set_title("Element Scatter Correlation Matrix", fontsize=11, fontweight="bold")

    # Panel 2: C/O_std vs Mg/Fe_std
    ax2 = axes[0, 1]
    pdf = sdf.dropna(subset=["CO_std", "MgFe_std"]).copy()
    coh_m = pdf["CO_coherent"]; inc_m = ~pdf["CO_coherent"]
    has_age = pdf["age_gyr"].notna()
    ax2.scatter(pdf.loc[inc_m, "CO_std"], pdf.loc[inc_m, "MgFe_std"],
                c="gray", alpha=0.5, s=40, edgecolors="none", label="Incoherent")
    if (coh_m & has_age).any():
        sc = ax2.scatter(pdf.loc[coh_m & has_age, "CO_std"], pdf.loc[coh_m & has_age, "MgFe_std"],
                         c=pdf.loc[coh_m & has_age, "age_gyr"], cmap="viridis", alpha=0.9,
                         s=90, edgecolors="red", linewidths=1.5, marker="D", vmin=0, vmax=8,
                         label="C/O-coherent")
        plt.colorbar(sc, ax=ax2, shrink=0.8, label="Age (Gyr)")
    if (coh_m & ~has_age).any():
        ax2.scatter(pdf.loc[coh_m & ~has_age, "CO_std"], pdf.loc[coh_m & ~has_age, "MgFe_std"],
                    c="gold", alpha=0.9, s=90, edgecolors="red", linewidths=1.5, marker="D")
    ax2.axvline(CO_THRESH, color="red", ls="--", alpha=0.5, lw=1)
    ax2.axhline(MGFE_THRESH, color="blue", ls="--", alpha=0.5, lw=1)
    ax2.set_xlabel("C/O scatter (std)", fontsize=10)
    ax2.set_ylabel("Mg/Fe scatter (std)", fontsize=10)
    ax2.set_title("C/O vs Mg/Fe Scatter", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)

    # Panel 3: Co-coherence bar chart
    ax3 = axes[1, 0]
    labels3 = ["Mg/Fe", "Si/Fe", "Al/Fe", "Fe/H"]
    obs_f = [co_mgfe/max(n_co_v,1), co_sife/max(n_co_v,1), co_alfe/max(n_co_v,1), co_feh/max(n_co_v,1)]
    rnd_f = [n_mgfe/max(n_valid,1), n_sife/max(n_valid,1), n_alfe/max(n_valid,1), n_feh/max(n_valid,1)]
    x3 = np.arange(len(labels3)); bw = 0.35
    ax3.bar(x3 - bw/2, obs_f, bw, color="steelblue", edgecolor="navy", label="Observed (C/O-coherent)")
    ax3.bar(x3 + bw/2, rnd_f, bw, color="lightcoral", edgecolor="darkred", alpha=0.7, label="Random expectation")
    for i, (o, r) in enumerate(zip(obs_f, rnd_f)):
        ax3.text(i - bw/2, o + 0.01, f"{o:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax3.text(i + bw/2, r + 0.01, f"{r:.0%}", ha="center", va="bottom", fontsize=8, color="darkred")
    ax3.set_xticks(x3); ax3.set_xticklabels(labels3)
    ax3.set_ylabel("Fraction coherent"); ax3.set_title("Co-Coherence with C/O-Coherent Clusters", fontweight="bold")
    ax3.legend(fontsize=8); ax3.set_ylim(0, max(max(obs_f), max(rnd_f)) * 1.3 + 0.05)

    # Panel 4: Top 10 grouped bar chart
    ax4 = axes[1, 1]
    top10 = score_df.head(10)
    n_top = len(top10); x4 = np.arange(n_top)
    tw = 0.75; bw4 = tw / 5
    colors4 = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for k, (nc, lab, clr) in enumerate(zip(norm_cols, ratio_labels, colors4)):
        offset = (k - 2.5 + 0.5) * bw4
        ax4.bar(x4 + offset, top10[nc].values, bw4, color=clr, edgecolor="white", lw=0.5, label=lab)
    ax4.set_xticks(x4); ax4.set_xticklabels(top10["cluster"].values, fontsize=7, rotation=45, ha="right")
    ax4.set_ylabel("Normalized scatter (0=tightest)")
    ax4.set_title("Top 10 Multi-Element-Coherent Clusters", fontweight="bold")
    ax4.legend(fontsize=7, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.close()
    info(f"Saved: {PLOT_FILE}")
    info("T15 complete.")

if __name__ == "__main__":
    main()

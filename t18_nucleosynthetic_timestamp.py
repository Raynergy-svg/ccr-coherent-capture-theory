#!/usr/bin/env python3
"""
T18 — The Nucleosynthetic Timestamp Test
==========================================
Certan (2025) | Coherent Capture Theory | GALAH DR4

Within open clusters, tests whether alpha-element coherence (Mg/Fe, Si/Fe —
core-collapse SNe, ~10 Myr delay) is tighter than s-process coherence
(Ba/Fe, Ce/Fe — AGB stars, ~1-3 Gyr delay).

Computes the alpha/s-process coherence ratio per cluster as a function of
age. The prediction:

  - Young clusters (< 1 Gyr): alpha tight, s-process loose (AGB enrichment
    hadn't fully homogenized yet)
  - Old clusters (> 3 Gyr): ratio inverts — s-process had time to equilibrate

If this age-dependent ratio exists, we have measured the chemical maturation
timeline of molecular clouds — the sequence in which different nucleosynthetic
products homogenize before star formation.
"""

import numpy as np
import pandas as pd
from astropy.table import Table
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
T9_STARS      = "t9_matched_stars.csv"
T9_CLUSTERS   = "t9_cluster_stats_with_age.csv"
GALAH_FITS    = "galah_dr4_allstar_240705.fits"

RESULTS_FILE  = "t18_results.txt"
PLOT_FILE     = "t18_nucleosynthetic_timestamp_plot.png"
CSV_FILE      = "t18_cluster_nucleosynthetic.csv"

MIN_MEMBERS   = 5      # require more members for reliable scatter
MATCH_TOL     = 0.5 / 3600.0  # 0.5 arcsec in degrees
SNR_MIN       = 30

out_lines = []
def info(msg):
    line = "[INFO] " + str(msg)
    print(line, flush=True)
    out_lines.append(line)

# ---------------------------------------------------------------------------
# 1. Load T9 cluster stars and GALAH s-process abundances
# ---------------------------------------------------------------------------
info("=" * 72)
info("T18  The Nucleosynthetic Timestamp Test")
info("Certan (2025) | CCT | GALAH DR4")
info("=" * 72)

t9_stars = pd.read_csv(T9_STARS)
t9_stats = pd.read_csv(T9_CLUSTERS)
info(f"T9 matched stars: {len(t9_stars)} in {t9_stars['cluster_name'].nunique()} clusters")
info(f"T9 cluster stats: {len(t9_stats)}")

# Load GALAH — need s-process columns
info("\nLoading GALAH DR4 for s-process abundances...")
galah_cols = ["ra", "dec", "sobject_id", "snr_px_ccd3", "flag_sp",
              "mg_fe", "e_mg_fe", "flag_mg_fe",
              "si_fe", "e_si_fe", "flag_si_fe",
              "ba_fe", "e_ba_fe", "flag_ba_fe",
              "ce_fe", "e_ce_fe", "flag_ce_fe",
              "la_fe", "e_la_fe", "flag_la_fe",
              "eu_fe", "e_eu_fe", "flag_eu_fe",
              "y_fe",  "e_y_fe",  "flag_y_fe",
              "fe_h", "e_fe_h", "flag_fe_h"]

galah_table = Table.read(GALAH_FITS, memmap=True)
galah_keep = [c for c in galah_cols if c in galah_table.colnames]
galah = galah_table[galah_keep].to_pandas()
info(f"GALAH loaded: {len(galah)} stars, {len(galah_keep)} columns")

# ---------------------------------------------------------------------------
# 2. Cross-match T9 stars with GALAH by ra/dec
# ---------------------------------------------------------------------------
info("\nCross-matching T9 cluster stars with GALAH by position...")

# Build KD-tree for fast matching
from scipy.spatial import cKDTree

galah_coords = np.deg2rad(np.column_stack([galah["ra"].values, galah["dec"].values]))
galah_xyz = np.column_stack([
    np.cos(galah_coords[:, 1]) * np.cos(galah_coords[:, 0]),
    np.cos(galah_coords[:, 1]) * np.sin(galah_coords[:, 0]),
    np.sin(galah_coords[:, 1])
])
tree = cKDTree(galah_xyz)

t9_coords = np.deg2rad(np.column_stack([t9_stars["ra"].values, t9_stars["dec"].values]))
t9_xyz = np.column_stack([
    np.cos(t9_coords[:, 1]) * np.cos(t9_coords[:, 0]),
    np.cos(t9_coords[:, 1]) * np.sin(t9_coords[:, 0]),
    np.sin(t9_coords[:, 1])
])

# Query within tolerance (convert arcsec to Cartesian distance)
tol_rad = np.deg2rad(MATCH_TOL)
tol_cart = 2 * np.sin(tol_rad / 2)  # chord distance

dists, indices = tree.query(t9_xyz, k=1)
matched_mask = dists < tol_cart
info(f"Matched: {matched_mask.sum()}/{len(t9_stars)} stars within {MATCH_TOL*3600:.1f} arcsec")

# Build merged dataframe
matched_t9 = t9_stars[matched_mask].copy().reset_index(drop=True)
matched_galah_idx = indices[matched_mask]

# Add GALAH s-process columns
s_process_cols = ["ba_fe", "e_ba_fe", "flag_ba_fe",
                  "ce_fe", "e_ce_fe", "flag_ce_fe",
                  "la_fe", "e_la_fe", "flag_la_fe",
                  "eu_fe", "e_eu_fe", "flag_eu_fe",
                  "y_fe",  "e_y_fe",  "flag_y_fe"]

for col in s_process_cols:
    if col in galah.columns:
        matched_t9[col] = galah.iloc[matched_galah_idx][col].values

# Also refresh alpha columns from GALAH (for consistency)
for col in ["mg_fe", "si_fe", "fe_h", "flag_mg_fe", "flag_si_fe", "flag_fe_h",
            "snr_px_ccd3", "flag_sp"]:
    if col in galah.columns:
        matched_t9["galah_" + col] = galah.iloc[matched_galah_idx][col].values

info(f"Stars with Ba/Fe: {matched_t9['ba_fe'].notna().sum()}")
info(f"Stars with Ce/Fe: {matched_t9['ce_fe'].notna().sum()}")
info(f"Stars with Y/Fe:  {matched_t9['y_fe'].notna().sum()}" if "y_fe" in matched_t9.columns else "")
info(f"Stars with Eu/Fe: {matched_t9['eu_fe'].notna().sum()}" if "eu_fe" in matched_t9.columns else "")

# ---------------------------------------------------------------------------
# 3. Quality cuts
# ---------------------------------------------------------------------------
info("\nApplying quality cuts...")
n0 = len(matched_t9)

# SNR
if "galah_snr_px_ccd3" in matched_t9.columns:
    matched_t9 = matched_t9[matched_t9["galah_snr_px_ccd3"] > SNR_MIN].copy()
    info(f"  After SNR > {SNR_MIN}: {len(matched_t9)}")

# flag_sp
if "galah_flag_sp" in matched_t9.columns:
    matched_t9 = matched_t9[matched_t9["galah_flag_sp"] == 0].copy()
    info(f"  After flag_sp == 0: {len(matched_t9)}")

info(f"  Total removed: {n0 - len(matched_t9)}")

# ---------------------------------------------------------------------------
# 4. Compute per-cluster scatter for alpha and s-process elements
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Computing per-cluster element scatter...")

# Alpha elements (CCSN, ~10 Myr delay)
alpha_cols = ["mg_fe", "si_fe"]
# S-process elements (AGB, ~1-3 Gyr delay)
sproc_cols = ["ba_fe", "ce_fe"]
# R-process (neutron star mergers, ~100 Myr delay) — bonus
rproc_cols = ["eu_fe"]

def cluster_element_scatter(df, col, flag_col=None, min_n=MIN_MEMBERS):
    """Compute per-cluster scatter (std) for an element."""
    results = {}
    for cname, grp in df.groupby("cluster_name"):
        vals = grp[col].dropna()
        if flag_col and flag_col in grp.columns:
            flags = grp.loc[vals.index, flag_col]
            vals = vals[flags == 0]
        vals = vals[(vals > -5) & (vals < 5)]  # sanity
        if len(vals) >= min_n:
            results[cname] = {"std": vals.std(ddof=1), "mean": vals.mean(),
                              "n": len(vals), "median": vals.median()}
    return results

# Compute for each element
info("  Alpha elements:")
mg_scatter = cluster_element_scatter(matched_t9, "mg_fe", "flag_mg_fe" if "flag_mg_fe" in matched_t9.columns else None)
si_scatter = cluster_element_scatter(matched_t9, "si_fe", "flag_si_fe" if "flag_si_fe" in matched_t9.columns else None)
info(f"    Mg/Fe: {len(mg_scatter)} clusters")
info(f"    Si/Fe: {len(si_scatter)} clusters")

info("  S-process elements:")
ba_scatter = cluster_element_scatter(matched_t9, "ba_fe", "flag_ba_fe")
ce_scatter = cluster_element_scatter(matched_t9, "ce_fe", "flag_ce_fe")
info(f"    Ba/Fe: {len(ba_scatter)} clusters")
info(f"    Ce/Fe: {len(ce_scatter)} clusters")

if "eu_fe" in matched_t9.columns:
    info("  R-process elements:")
    eu_scatter = cluster_element_scatter(matched_t9, "eu_fe", "flag_eu_fe")
    info(f"    Eu/Fe: {len(eu_scatter)} clusters")
else:
    eu_scatter = {}

# ---------------------------------------------------------------------------
# 5. Build cluster-level dataframe with alpha vs s-process coherence
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Building nucleosynthetic coherence table...")

# Get age map from T9
age_map = dict(zip(t9_stats["cluster"], t9_stats["age_gyr"]))

# All clusters with at least one alpha AND one s-process measurement
all_clusters = sorted(set(mg_scatter.keys()) | set(si_scatter.keys()) |
                      set(ba_scatter.keys()) | set(ce_scatter.keys()))

rows = []
for cname in all_clusters:
    row = {"cluster": cname}

    # Alpha scatter (mean of Mg and Si std)
    alpha_stds = []
    if cname in mg_scatter:
        row["mg_fe_std"] = mg_scatter[cname]["std"]
        row["mg_fe_mean"] = mg_scatter[cname]["mean"]
        row["mg_fe_n"] = mg_scatter[cname]["n"]
        alpha_stds.append(mg_scatter[cname]["std"])
    if cname in si_scatter:
        row["si_fe_std"] = si_scatter[cname]["std"]
        row["si_fe_mean"] = si_scatter[cname]["mean"]
        row["si_fe_n"] = si_scatter[cname]["n"]
        alpha_stds.append(si_scatter[cname]["std"])

    # S-process scatter (mean of Ba and Ce std)
    sproc_stds = []
    if cname in ba_scatter:
        row["ba_fe_std"] = ba_scatter[cname]["std"]
        row["ba_fe_mean"] = ba_scatter[cname]["mean"]
        row["ba_fe_n"] = ba_scatter[cname]["n"]
        sproc_stds.append(ba_scatter[cname]["std"])
    if cname in ce_scatter:
        row["ce_fe_std"] = ce_scatter[cname]["std"]
        row["ce_fe_mean"] = ce_scatter[cname]["mean"]
        row["ce_fe_n"] = ce_scatter[cname]["n"]
        sproc_stds.append(ce_scatter[cname]["std"])

    # R-process
    if cname in eu_scatter:
        row["eu_fe_std"] = eu_scatter[cname]["std"]
        row["eu_fe_mean"] = eu_scatter[cname]["mean"]
        row["eu_fe_n"] = eu_scatter[cname]["n"]

    if alpha_stds:
        row["alpha_scatter"] = np.mean(alpha_stds)
    if sproc_stds:
        row["sproc_scatter"] = np.mean(sproc_stds)

    # Alpha/s-process ratio
    if alpha_stds and sproc_stds:
        row["alpha_sproc_ratio"] = np.mean(alpha_stds) / np.mean(sproc_stds)

    # Age
    # T9 uses different name format — try both
    cname_norm = cname.replace("_", " ")
    if cname in age_map:
        row["age_gyr"] = age_map[cname]
    elif cname_norm in age_map:
        row["age_gyr"] = age_map[cname_norm]

    rows.append(row)

nuc_df = pd.DataFrame(rows)
has_both = nuc_df["alpha_scatter"].notna() & nuc_df["sproc_scatter"].notna()
has_age = nuc_df["age_gyr"].notna()
info(f"Total clusters: {len(nuc_df)}")
info(f"With alpha + s-process: {has_both.sum()}")
info(f"With alpha + s-process + age: {(has_both & has_age).sum()}")

# ---------------------------------------------------------------------------
# 6. Core analysis: alpha vs s-process coherence
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("CORE ANALYSIS: Alpha vs S-process Coherence")

complete = nuc_df[has_both].copy()
info(f"\nUsing {len(complete)} clusters with both alpha and s-process scatter")

info(f"\nOverall statistics:")
info(f"  Alpha scatter  (Mg+Si mean std): median={complete['alpha_scatter'].median():.4f}, "
     f"mean={complete['alpha_scatter'].mean():.4f}")
info(f"  S-proc scatter (Ba+Ce mean std): median={complete['sproc_scatter'].median():.4f}, "
     f"mean={complete['sproc_scatter'].mean():.4f}")

# Paired comparison: is alpha tighter than s-process?
wil_stat, wil_p = stats.wilcoxon(complete["alpha_scatter"], complete["sproc_scatter"],
                                  alternative="less")
info(f"\nWilcoxon signed-rank (alpha < s-process): W={wil_stat:.0f}, p={wil_p:.4e}")
if wil_p < 0.05:
    info("  => SIGNIFICANT: alpha elements are tighter than s-process across clusters")
else:
    info("  => Not significant at p < 0.05")

# Fraction where alpha < s-process
frac_alpha_tighter = (complete["alpha_scatter"] < complete["sproc_scatter"]).mean()
info(f"  Fraction where alpha < s-process: {frac_alpha_tighter:.1%}")

# ---------------------------------------------------------------------------
# 7. Age-dependent analysis: does the ratio change with age?
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("AGE-DEPENDENT ANALYSIS: Alpha/S-process ratio vs cluster age")

aged = nuc_df[has_both & has_age].copy()
info(f"Clusters with ratio + age: {len(aged)}")

if len(aged) >= 10:
    rho_ratio, p_ratio = stats.spearmanr(aged["age_gyr"], aged["alpha_sproc_ratio"])
    info(f"\nSpearman (age vs alpha/s-process ratio): ρ = {rho_ratio:.4f}, p = {p_ratio:.4e}")
    if p_ratio < 0.05 and rho_ratio > 0:
        info("  => SIGNIFICANT: ratio INCREASES with age (alpha becomes relatively looser)")
        info("     Consistent with s-process homogenizing over time")
    elif p_ratio < 0.05 and rho_ratio < 0:
        info("  => SIGNIFICANT: ratio DECREASES with age (s-process becomes relatively looser)")
    else:
        info("  => Not significant at p < 0.05")

    # Individual element trends
    info("\nIndividual element scatter vs age:")
    for col, label in [("alpha_scatter", "Alpha (Mg+Si)"),
                        ("sproc_scatter", "S-process (Ba+Ce)"),
                        ("mg_fe_std", "Mg/Fe"), ("si_fe_std", "Si/Fe"),
                        ("ba_fe_std", "Ba/Fe"), ("ce_fe_std", "Ce/Fe"),
                        ("eu_fe_std", "Eu/Fe")]:
        valid = aged[[col, "age_gyr"]].dropna()
        if len(valid) >= 10:
            rho_e, p_e = stats.spearmanr(valid["age_gyr"], valid[col])
            sig = "***" if p_e < 0.001 else "**" if p_e < 0.01 else "*" if p_e < 0.05 else "n.s."
            info(f"  {label:<18}: ρ={rho_e:+.4f}  p={p_e:.4e}  {sig}")

    # Binned analysis
    info("\nBinned alpha/s-process ratio:")
    bins = [(0, 0.3), (0.3, 0.7), (0.7, 1.5), (1.5, 3.0), (3.0, 10.0)]
    info(f"  {'Age bin':>12}  {'N':>4}  {'Alpha std':>10}  {'S-proc std':>11}  {'Ratio':>8}  {'α<s frac':>9}")
    bin_data = []
    for lo, hi in bins:
        mask = (aged["age_gyr"] >= lo) & (aged["age_gyr"] < hi)
        b = aged[mask]
        if len(b) >= 3:
            a_med = b["alpha_scatter"].median()
            s_med = b["sproc_scatter"].median()
            r_med = b["alpha_sproc_ratio"].median()
            frac = (b["alpha_scatter"] < b["sproc_scatter"]).mean()
            info(f"  {lo:.1f}-{hi:.1f} Gyr  {len(b):>4}  {a_med:>10.4f}  {s_med:>11.4f}  {r_med:>8.3f}  {frac:>9.1%}")
            bin_data.append({"age_lo": lo, "age_hi": hi, "age_mid": (lo+hi)/2,
                            "N": len(b), "alpha_med": a_med, "sproc_med": s_med,
                            "ratio_med": r_med, "frac_alpha_tighter": frac})
        else:
            info(f"  {lo:.1f}-{hi:.1f} Gyr  {len(b):>4}       --           --        --         --")

    # Mann-Whitney: young vs old ratio
    young = aged[aged["age_gyr"] < 1.0]["alpha_sproc_ratio"].dropna()
    old = aged[aged["age_gyr"] >= 1.0]["alpha_sproc_ratio"].dropna()
    if len(young) >= 5 and len(old) >= 5:
        mw_u, mw_p = stats.mannwhitneyu(young, old, alternative="two-sided")
        info(f"\nMann-Whitney (young<1Gyr vs old>=1Gyr ratio): U={mw_u:.0f}, p={mw_p:.4e}")
        info(f"  Young median ratio: {young.median():.3f} (N={len(young)})")
        info(f"  Old median ratio:   {old.median():.3f} (N={len(old)})")

# ---------------------------------------------------------------------------
# 8. Cross-channel correlation: alpha coherence vs s-process coherence
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Cross-channel correlation: alpha scatter vs s-process scatter")

rho_cross, p_cross = stats.spearmanr(complete["alpha_scatter"], complete["sproc_scatter"])
info(f"Spearman (alpha vs s-process scatter): ρ = {rho_cross:.4f}, p = {p_cross:.4e}")
if p_cross < 0.05 and rho_cross > 0:
    info("  => SIGNIFICANT positive: clusters tight in alpha are also tight in s-process")
    info("     Birth cloud chemical homogeneity spans nucleosynthetic channels")
elif p_cross < 0.05 and rho_cross < 0:
    info("  => SIGNIFICANT negative: alpha-tight clusters have loose s-process (or vice versa)")
else:
    info("  => No significant cross-channel correlation")

# Individual cross-correlations
info("\nDetailed cross-channel Spearman correlations:")
pairs = [("mg_fe_std", "ba_fe_std", "Mg/Fe vs Ba/Fe"),
         ("mg_fe_std", "ce_fe_std", "Mg/Fe vs Ce/Fe"),
         ("si_fe_std", "ba_fe_std", "Si/Fe vs Ba/Fe"),
         ("si_fe_std", "ce_fe_std", "Si/Fe vs Ce/Fe")]
if eu_scatter:
    pairs += [("mg_fe_std", "eu_fe_std", "Mg/Fe vs Eu/Fe"),
              ("ba_fe_std", "eu_fe_std", "Ba/Fe vs Eu/Fe")]

for col1, col2, label in pairs:
    valid = complete[[col1, col2]].dropna()
    if len(valid) >= 10:
        rho_p, p_p = stats.spearmanr(valid[col1], valid[col2])
        sig = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "n.s."
        info(f"  {label:<20}: ρ={rho_p:+.4f}  p={p_p:.4e}  {sig}")

# ---------------------------------------------------------------------------
# 9. Save outputs
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Saving outputs...")

nuc_df.to_csv(CSV_FILE, index=False)
info(f"Saved: {CSV_FILE}")

with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(out_lines))
info(f"Saved: {RESULTS_FILE}")

# ---------------------------------------------------------------------------
# 10. Generate 4-panel plot
# ---------------------------------------------------------------------------
info("Generating plot...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("T18 Nucleosynthetic Timestamp Test | Certan (2025) | GALAH DR4\n"
             "Alpha-element (CCSNe, ~10 Myr) vs S-process (AGB, ~1-3 Gyr) coherence in open clusters",
             fontsize=12, fontweight="bold", y=0.99)

# Panel 1: Alpha scatter vs S-process scatter
ax1 = axes[0, 0]
if len(complete) > 0:
    has_age_plot = complete["age_gyr"].notna()
    if has_age_plot.any():
        sc = ax1.scatter(complete.loc[has_age_plot, "alpha_scatter"],
                         complete.loc[has_age_plot, "sproc_scatter"],
                         c=complete.loc[has_age_plot, "age_gyr"], cmap="viridis",
                         s=60, alpha=0.8, edgecolors="white", linewidths=0.5,
                         vmin=0, vmax=5, zorder=5)
        plt.colorbar(sc, ax=ax1, label="Age (Gyr)", pad=0.02)
    if (~has_age_plot).any():
        ax1.scatter(complete.loc[~has_age_plot, "alpha_scatter"],
                    complete.loc[~has_age_plot, "sproc_scatter"],
                    c="gray", s=40, alpha=0.5, marker="x", label="No age")

    # 1:1 line
    lim = max(complete["alpha_scatter"].max(), complete["sproc_scatter"].max()) * 1.1
    ax1.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.4, label="1:1")
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)

ax1.set_xlabel("Alpha scatter (mean of Mg/Fe, Si/Fe std)", fontsize=10)
ax1.set_ylabel("S-process scatter (mean of Ba/Fe, Ce/Fe std)", fontsize=10)
ax1.set_title(f"Alpha vs S-process Coherence (ρ={rho_cross:.3f}, p={p_cross:.2e})",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.15)

# Panel 2: Alpha/S-process ratio vs age
ax2 = axes[0, 1]
if len(aged) > 0:
    ax2.scatter(aged["age_gyr"], aged["alpha_sproc_ratio"],
                s=50, c="steelblue", alpha=0.7, edgecolors="white", linewidths=0.5)

    # Trend line
    if len(aged) >= 5:
        z = np.polyfit(aged["age_gyr"].values, aged["alpha_sproc_ratio"].values, 1)
        xfit = np.linspace(aged["age_gyr"].min(), aged["age_gyr"].max(), 100)
        ax2.plot(xfit, np.polyval(z, xfit), "r--", lw=2, alpha=0.7,
                 label=f"Linear: slope={z[0]:+.3f}")

    ax2.axhline(1.0, color="black", ls=":", lw=1.5, alpha=0.5, label="Ratio = 1 (equal)")

    ax2.set_xlabel("Cluster Age (Gyr)", fontsize=11)
    ax2.set_ylabel("Alpha / S-process scatter ratio", fontsize=10)
    ax2.set_title(f"Nucleosynthetic Ratio vs Age (ρ={rho_ratio:.3f}, p={p_ratio:.2e})",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.15)

# Panel 3: Binned comparison — alpha vs s-process by age bin
ax3 = axes[1, 0]
if bin_data:
    bdf = pd.DataFrame(bin_data)
    x3 = np.arange(len(bdf))
    bw = 0.35
    bin_labels3 = [f"{r['age_lo']:.1f}–{r['age_hi']:.1f}" for _, r in bdf.iterrows()]

    ax3.bar(x3 - bw/2, bdf["alpha_med"], bw, color="#2ca02c", edgecolor="darkgreen",
            alpha=0.8, label="Alpha (Mg+Si)")
    ax3.bar(x3 + bw/2, bdf["sproc_med"], bw, color="#d62728", edgecolor="darkred",
            alpha=0.8, label="S-process (Ba+Ce)")

    for i, (_, r) in enumerate(bdf.iterrows()):
        ax3.text(i - bw/2, r["alpha_med"] + 0.002, f"{r['alpha_med']:.3f}",
                 ha="center", va="bottom", fontsize=7, color="darkgreen")
        ax3.text(i + bw/2, r["sproc_med"] + 0.002, f"{r['sproc_med']:.3f}",
                 ha="center", va="bottom", fontsize=7, color="darkred")
        ax3.text(i, -0.008, f"N={r['N']:.0f}", ha="center", fontsize=8, color="gray")

    ax3.set_xticks(x3)
    ax3.set_xticklabels(bin_labels3, fontsize=10)
    ax3.set_xlabel("Age Bin (Gyr)", fontsize=11)
    ax3.set_ylabel("Median Intra-cluster Scatter (std)", fontsize=10)
    ax3.set_title("Alpha vs S-process Scatter by Age", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.15, axis="y")

# Panel 4: Individual element scatter vs age
ax4 = axes[1, 1]
element_data = [
    ("mg_fe_std", "Mg/Fe (CCSNe)", "#2ca02c", "o"),
    ("si_fe_std", "Si/Fe (CCSNe)", "#98df8a", "s"),
    ("ba_fe_std", "Ba/Fe (AGB)",   "#d62728", "D"),
    ("ce_fe_std", "Ce/Fe (AGB)",   "#ff9896", "^"),
]
if eu_scatter:
    element_data.append(("eu_fe_std", "Eu/Fe (r-proc)", "#9467bd", "v"))

for col, label, color, marker in element_data:
    valid = aged[[col, "age_gyr"]].dropna()
    if len(valid) >= 5:
        ax4.scatter(valid["age_gyr"], valid[col], s=30, c=color, marker=marker,
                    alpha=0.6, label=label, edgecolors="white", linewidths=0.3)
        # Trend
        z = np.polyfit(valid["age_gyr"].values, valid[col].values, 1)
        xf = np.linspace(valid["age_gyr"].min(), valid["age_gyr"].max(), 50)
        ax4.plot(xf, np.polyval(z, xf), "--", color=color, alpha=0.5, lw=1.5)

ax4.set_xlabel("Cluster Age (Gyr)", fontsize=11)
ax4.set_ylabel("Intra-cluster Scatter (std)", fontsize=10)
ax4.set_title("Element-by-Element Scatter vs Age", fontsize=11, fontweight="bold")
ax4.legend(fontsize=7, ncol=2, loc="upper right")
ax4.grid(True, alpha=0.15)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(PLOT_FILE, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
info(f"Saved: {PLOT_FILE}")

# ---------------------------------------------------------------------------
# 11. Final Summary
# ---------------------------------------------------------------------------
info("\n" + "=" * 72)
info("T18 SUMMARY")
info("=" * 72)
info(f"Clusters analyzed: {len(complete)} (with alpha + s-process scatter)")
info(f"With ages: {len(aged)}")
info(f"")
info(f"Alpha scatter median:    {complete['alpha_scatter'].median():.4f}")
info(f"S-process scatter median: {complete['sproc_scatter'].median():.4f}")
info(f"Alpha < S-process: {frac_alpha_tighter:.1%} of clusters")
info(f"Wilcoxon (alpha < s-proc): p = {wil_p:.4e}")
info(f"")
info(f"Cross-channel (alpha vs s-proc): ρ = {rho_cross:.4f}, p = {p_cross:.4e}")

if len(aged) >= 10:
    info(f"Ratio vs age: ρ = {rho_ratio:.4f}, p = {p_ratio:.4e}")

info(f"")
# Interpretation
if wil_p < 0.05 and frac_alpha_tighter > 0.5:
    info("=> Alpha elements (CCSNe, ~10 Myr) are TIGHTER than s-process (AGB, ~1-3 Gyr)")
    info("   across open clusters. This is consistent with the nucleosynthetic timeline:")
    info("   CCSNe enrich the birth cloud rapidly and homogeneously before star formation,")
    info("   while AGB enrichment is still inhomogeneous at the time of cluster formation.")
elif wil_p < 0.05:
    info("=> S-process elements are tighter than alpha — unexpected.")
    info("   May indicate that AGB products are better mixed in pre-stellar clouds,")
    info("   or that alpha scatter has additional sources (e.g., SN yield variations).")
else:
    info("=> No significant difference between alpha and s-process coherence.")

if len(aged) >= 10 and p_ratio < 0.05 and rho_ratio > 0:
    info(f"\n=> KEY RESULT: Alpha/s-process ratio INCREASES with age (ρ={rho_ratio:.3f})")
    info("   S-process coherence improves relative to alpha in older clusters.")
    info("   This measures the chemical maturation of molecular clouds —")
    info("   the rate at which AGB products homogenize relative to CCSN products.")
elif len(aged) >= 10 and p_ratio < 0.05:
    info(f"\n=> Alpha/s-process ratio DECREASES with age — opposite to simple prediction.")

if p_cross < 0.05 and rho_cross > 0:
    info(f"\n=> Cross-channel coherence CONFIRMED (ρ={rho_cross:.3f})")
    info("   Clusters that are tight in alpha are also tight in s-process.")
    info("   Birth cloud chemical homogeneity is a multi-channel property,")
    info("   not specific to any single nucleosynthetic pathway.")

info(f"\nOutput files:")
info(f"  {RESULTS_FILE}")
info(f"  {CSV_FILE}")
info(f"  {PLOT_FILE}")
info(f"\nT18 complete.")

# Save final results
with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(out_lines))

#!/usr/bin/env python3
# CCT T11 - Multi-Element Fingerprinting
# Tests whether clusters coherent in C/O are also coherent across
# Ba/Eu/Ce/Al/Mg/Si/Y/La simultaneously — CCT multi-element signature
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/root/ccr_crossmatch"

print("=" * 60)
print("CCT T11 - Multi-Element Fingerprinting")
print("=" * 60)

# Elements to test — (abundance_col, flag_col, label)
ELEMENTS = [
    ("ba_fe", "flag_ba_fe", "Ba/Fe"),
    ("eu_fe", "flag_eu_fe", "Eu/Fe"),
    ("ce_fe", "flag_ce_fe", "Ce/Fe"),
    ("al_fe", "flag_al_fe", "Al/Fe"),
    ("mg_fe", "flag_mg_fe", "Mg/Fe"),
    ("si_fe", "flag_si_fe", "Si/Fe"),
    ("y_fe",  "flag_y_fe",  "Y/Fe"),
    ("la_fe", "flag_la_fe", "La/Fe"),
]
MIN_STARS = 3

# Load cluster membership from T9
print("Loading T9 matched stars...")
stars = pd.read_csv(DATA_DIR + "/t9_matched_stars.csv")
print("Stars: " + str(len(stars)))

# Load GALAH DR4 for abundances — match by ra/dec
print("Loading GALAH DR4 fits...")
hdu = fits.open(DATA_DIR + "/galah_dr4_allstar_240705.fits")
galah = pd.DataFrame(hdu[1].data)
hdu.close()
galah.columns = [c.lower() for c in galah.columns]
print("GALAH rows: " + str(len(galah)))

# T9 matched stars have ra, dec of individual stars
# Join on ra+dec with tolerance, or use sobject_id if available
# Check overlap columns
overlap = [c for c in stars.columns if c in galah.columns]
print("Overlap cols: " + str(overlap[:10]))

# Use ra/dec match if no direct ID
# Round to 4dp for join
stars["ra_r"]  = stars["ra"].round(4)
stars["dec_r"] = stars["dec"].round(4)
galah["ra_r"]  = galah["ra"].round(4)
galah["dec_r"] = galah["dec"].round(4)

needed_galah = ["ra_r", "dec_r"] + [e[0] for e in ELEMENTS] + [e[1] for e in ELEMENTS]
needed_galah = [c for c in needed_galah if c in galah.columns]
galah_sub = galah[needed_galah].copy()

merged = stars.merge(galah_sub, on=["ra_r", "dec_r"], how="inner")
print("After join: " + str(len(merged)) + " stars with abundances")

# Load cluster age info
stats = pd.read_csv(DATA_DIR + "/t9_cluster_stats_with_age.csv")
stats = stats.rename(columns={"cluster": "cluster_name"})

# For each cluster compute per-element std (flag==0 only)
print("Computing per-element cluster dispersions...")
records = []
for cname, grp in merged.groupby("cluster_name"):
    rec = {"cluster_name": cname, "n_stars": len(grp)}
    n_coherent = 0
    for abund_col, flag_col, label in ELEMENTS:
        if abund_col not in grp.columns:
            rec[abund_col + "_std"] = np.nan
            rec[abund_col + "_n"]   = 0
            continue
        if flag_col in grp.columns:
            clean = grp[grp[flag_col] == 0][abund_col].dropna()
        else:
            clean = grp[abund_col].dropna()
        if len(clean) >= MIN_STARS:
            std_val = clean.std()
            rec[abund_col + "_std"] = std_val
            rec[abund_col + "_n"]   = len(clean)
            if std_val < 0.1:
                n_coherent += 1
        else:
            rec[abund_col + "_std"] = np.nan
            rec[abund_col + "_n"]   = 0
    rec["n_elements_coherent"] = n_coherent
    records.append(rec)

clust = pd.DataFrame(records)
clust = clust.merge(stats[["cluster_name","age_gyr","age_bin","C_O_std"]], on="cluster_name", how="left")
print("Clusters with data: " + str(len(clust)))

# Age split
young = clust[clust["age_gyr"] < 1.0].copy()
old   = clust[clust["age_gyr"] >= 1.0].copy()
print("Young (<1 Gyr): " + str(len(young)) + "  Old (>=1 Gyr): " + str(len(old)))

# Test each element: young vs old MWU
print("\nPer-element young vs old dispersion (Mann-Whitney U):")
print("-" * 55)
results = []
for abund_col, flag_col, label in ELEMENTS:
    col = abund_col + "_std"
    if col not in clust.columns:
        continue
    y_vals = young[col].dropna()
    o_vals = old[col].dropna()
    if len(y_vals) < 3 or len(o_vals) < 3:
        continue
    stat, p = mannwhitneyu(y_vals, o_vals, alternative="less")
    sig = "ns"
    if p < 0.001: sig = "***"
    elif p < 0.01: sig = "**"
    elif p < 0.05: sig = "*"
    print(label + ": young_mean=" + str(round(y_vals.mean(),4)) +
          " old_mean=" + str(round(o_vals.mean(),4)) +
          " p=" + str(round(p,4)) + " " + sig)
    results.append({"element": label, "col": col,
                    "young_mean": y_vals.mean(), "old_mean": o_vals.mean(),
                    "p": p, "sig": sig,
                    "y_n": len(y_vals), "o_n": len(o_vals)})

res_df = pd.DataFrame(results)

# Multi-element coherence score vs age
print("\nMulti-element coherence (n elements with std<0.1) vs age:")
has_age = clust["age_gyr"].notna() & clust["n_elements_coherent"].notna()
sub = clust[has_age]
r, p_r = spearmanr(sub["age_gyr"], sub["n_elements_coherent"])
print("Spearman r (age vs n_coherent_elements) = " + str(round(r,4)) + " p=" + str(round(p_r,4)))

# C/O std vs multi-element coherence
co_sub = clust[clust["C_O_std"].notna() & clust["n_elements_coherent"].notna()]
r2, p2 = spearmanr(co_sub["C_O_std"], co_sub["n_elements_coherent"])
print("Spearman r (C/O std vs n_coherent_elements) = " + str(round(r2,4)) + " p=" + str(round(p2,4)))

# Plot
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

# Panel A - per element young vs old bar chart
ax1 = fig.add_subplot(gs[0, :2])
if len(res_df) > 0:
    x = np.arange(len(res_df))
    w = 0.35
    ax1.bar(x - w/2, res_df["young_mean"], w, label="Young <1 Gyr", color="steelblue", alpha=0.8)
    ax1.bar(x + w/2, res_df["old_mean"],   w, label="Old >=1 Gyr",  color="coral",     alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(res_df["element"], fontsize=9)
    ax1.set_ylabel("Mean abundance std (dex)")
    ax1.set_title("T11: Per-Element Dispersion — Young vs Old Clusters")
    ax1.legend(fontsize=9)
    for i, row in res_df.iterrows():
        if row["sig"] != "ns":
            ax1.text(i, max(row["young_mean"], row["old_mean"]) + 0.002,
                     row["sig"], ha="center", fontsize=10, color="darkred")

# Panel B - p-value summary
ax2 = fig.add_subplot(gs[0, 2])
if len(res_df) > 0:
    colors = ["green" if p < 0.05 else "grey" for p in res_df["p"]]
    ax2.barh(res_df["element"], -np.log10(res_df["p"].clip(1e-4)), color=colors, alpha=0.8)
    ax2.axvline(-np.log10(0.05), color="red", ls="--", lw=1.5, label="p=0.05")
    ax2.axvline(-np.log10(0.01), color="orange", ls="--", lw=1, label="p=0.01")
    ax2.set_xlabel("-log10(p)")
    ax2.set_title("Significance\n(young < old dispersion)")
    ax2.legend(fontsize=7)

# Panel C - multi-element coherence vs age scatter
ax3 = fig.add_subplot(gs[1, 0])
sc = ax3.scatter(sub["age_gyr"], sub["n_elements_coherent"],
                 alpha=0.4, s=15, c=sub["age_gyr"], cmap="plasma")
ax3.set_xlabel("Age (Gyr)")
ax3.set_ylabel("N elements coherent (std<0.1)")
ax3.set_title("Multi-element coherence vs Age\nSpearman r=" + str(round(r,3)) + " p=" + str(round(p_r,3)))

# Panel D - C/O std vs multi-element coherence
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(co_sub["C_O_std"], co_sub["n_elements_coherent"],
            alpha=0.4, s=15, color="teal")
ax4.set_xlabel("C/O std (dex)")
ax4.set_ylabel("N elements coherent")
ax4.set_title("C/O coherence vs Multi-element\nr=" + str(round(r2,3)) + " p=" + str(round(p2,3)))

# Panel E - result summary
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
sig_elements = res_df[res_df["sig"] != "ns"]["element"].tolist() if len(res_df) > 0 else []
lines = ["T11 RESULT SUMMARY", "-"*28,
         "Clusters: " + str(len(clust)),
         "Young: " + str(len(young)) + "  Old: " + str(len(old)),
         "",
         "Sig elements (young<old):",
         str(sig_elements) if sig_elements else "none",
         "",
         "Age vs N_coherent:",
         "r=" + str(round(r,4)) + " p=" + str(round(p_r,4)),
         "",
         "C/O vs N_coherent:",
         "r=" + str(round(r2,4)) + " p=" + str(round(p2,4))]
ax5.text(0.05, 0.95, "\n".join(lines), transform=ax5.transAxes,
         fontsize=8, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.suptitle("CCT T11 - Multi-Element Fingerprinting", fontsize=13, fontweight="bold")
plt.savefig(DATA_DIR + "/t11_multielement_plot.png", dpi=150, bbox_inches="tight")
print("Plot: t11_multielement_plot.png")

with open(DATA_DIR + "/t11_results.txt", "w") as f:
    f.write("CCT T11 - Multi-Element Fingerprinting\n" + "="*45 + "\n")
    f.write("Clusters: " + str(len(clust)) + "\n")
    f.write("Young (<1 Gyr): " + str(len(young)) + "\n")
    f.write("Old (>=1 Gyr): " + str(len(old)) + "\n\n")
    f.write("Per-element MWU (young < old dispersion):\n")
    for _, row in res_df.iterrows():
        f.write("  " + row["element"] + ": young=" + str(round(row["young_mean"],4)) +
                " old=" + str(round(row["old_mean"],4)) +
                " p=" + str(round(row["p"],4)) + " " + row["sig"] + "\n")
    f.write("\nAge vs N_coherent_elements: r=" + str(round(r,4)) + " p=" + str(round(p_r,4)) + "\n")
    f.write("C/O std vs N_coherent: r=" + str(round(r2,4)) + " p=" + str(round(p2,4)) + "\n")
    f.write("\nSignificant elements: " + str(sig_elements) + "\n")

print("Results: t11_results.txt")
print("T11 complete.")

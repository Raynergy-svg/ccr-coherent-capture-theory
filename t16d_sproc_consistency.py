#!/usr/bin/env python3
"""
T16d — S-process Consistency Check
=====================================
Certan (2026) | CCT | GALAH DR4

Tests whether field stars that match a cluster template in 4 primary
dimensions (C/O, Mg/Fe, Si/Fe, Fe/H) ALSO align in Ba/Fe — an element
NOT used in the matching. If yes, that's an independent 5th dimension
confirming the chemical fingerprint is real.
"""

import numpy as np
import pandas as pd
from astropy.table import Table
from scipy import stats
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

SOLAR_CO = 0.549
T9_STARS = "t9_matched_stars.csv"
T9_CLUSTERS = "t9_cluster_stats_with_age.csv"
GALAH_FITS = "galah_dr4_allstar_240705.fits"

FIXED_TOL = np.array([0.08, 0.05, 0.05, 0.10])
DIM_COLS = ["C_O", "mg_fe", "si_fe", "fe_h"]
MIN_MEMBERS = 5
CO_STD_THRESH = 0.10
MATCH_ARCSEC = 0.5

out = []
def info(msg):
    line = "[INFO] " + str(msg)
    print(line, flush=True)
    out.append(line)

info("=" * 72)
info("T16d  S-process Consistency Check")
info("Certan (2026) | CCT | GALAH DR4")
info("=" * 72)
info("Question: do 4D-matched field stars ALSO align in Ba/Fe (5th dimension)?")

# 1. Load T9 stars and cross-match with GALAH for Ba/Fe
t9_stars = pd.read_csv(T9_STARS)
t9_stats = pd.read_csv(T9_CLUSTERS)
info(f"T9: {len(t9_stars)} stars")

info("\nCross-matching T9 stars with GALAH for Ba/Fe...")
galah_table = Table.read(GALAH_FITS, memmap=True)
galah_full = galah_table[["ra", "dec", "ba_fe", "flag_ba_fe",
                           "ce_fe", "flag_ce_fe"]].to_pandas()

# KD-tree match
g_coords = np.deg2rad(np.column_stack([galah_full["ra"].values, galah_full["dec"].values]))
g_xyz = np.column_stack([np.cos(g_coords[:,1])*np.cos(g_coords[:,0]),
                          np.cos(g_coords[:,1])*np.sin(g_coords[:,0]), np.sin(g_coords[:,1])])
tree_g = cKDTree(g_xyz)

t9_coords = np.deg2rad(np.column_stack([t9_stars["ra"].values, t9_stars["dec"].values]))
t9_xyz = np.column_stack([np.cos(t9_coords[:,1])*np.cos(t9_coords[:,0]),
                           np.cos(t9_coords[:,1])*np.sin(t9_coords[:,0]), np.sin(t9_coords[:,1])])
tol = 2 * np.sin(np.deg2rad(MATCH_ARCSEC / 3600) / 2)
dists, indices = tree_g.query(t9_xyz, k=1)
matched = dists < tol
info(f"Matched: {matched.sum()}/{len(t9_stars)}")

t9_stars.loc[matched, "ba_fe"] = galah_full.iloc[indices[matched]]["ba_fe"].values
t9_stars.loc[matched, "flag_ba_fe"] = galah_full.iloc[indices[matched]]["flag_ba_fe"].values
t9_stars.loc[matched, "ce_fe"] = galah_full.iloc[indices[matched]]["ce_fe"].values
t9_stars.loc[matched, "flag_ce_fe"] = galah_full.iloc[indices[matched]]["flag_ce_fe"].values

del galah_full, g_xyz, tree_g  # free memory

# Compute cluster Ba/Fe means
info("Computing cluster Ba/Fe centroids...")
template_cl = t9_stats[(t9_stats["N"] >= MIN_MEMBERS) & (t9_stats["C_O_std"] < CO_STD_THRESH)]
templates = {}

for _, row in template_cl.iterrows():
    cname = row["cluster"]
    grp = t9_stars[t9_stars["cluster_name"] == cname].copy()
    grp_4d = grp.dropna(subset=DIM_COLS)
    if len(grp_4d) < MIN_MEMBERS:
        continue

    # Ba/Fe: require unflagged
    ba_vals = grp.loc[grp["ba_fe"].notna() & (grp["flag_ba_fe"] == 0), "ba_fe"]
    if len(ba_vals) < 3:
        continue

    templates[cname] = {
        "centroid": grp_4d[DIM_COLS].mean().values,
        "ba_fe_mean": float(ba_vals.mean()),
        "ba_fe_std": float(ba_vals.std(ddof=1)) if len(ba_vals) > 1 else 0.05,
        "ba_fe_n": len(ba_vals),
        "N": len(grp_4d),
        "age_gyr": row["age_gyr"] if pd.notna(row["age_gyr"]) else np.nan,
    }

info(f"Templates with Ba/Fe: {len(templates)}")

# 2. Load GALAH field stars
info("\nLoading GALAH field stars...")
galah_table2 = Table.read(GALAH_FITS, memmap=True)
cols_need = ["ra", "dec", "snr_px_ccd3", "flag_sp", "c_fe", "o_fe",
             "flag_c_fe", "flag_o_fe", "mg_fe", "flag_mg_fe", "si_fe",
             "flag_si_fe", "fe_h", "flag_fe_h", "ba_fe", "flag_ba_fe"]
galah = galah_table2[[c for c in cols_need if c in galah_table2.colnames]].to_pandas()
del galah_table2

galah = galah[galah["snr_px_ccd3"] > 30].copy()
galah = galah[galah["flag_sp"] == 0].copy()
galah["C_O"] = (10.0 ** (galah["c_fe"] - galah["o_fe"])) * SOLAR_CO

for col, fc in [("C_O", None), ("mg_fe", "flag_mg_fe"),
                ("si_fe", "flag_si_fe"), ("fe_h", "flag_fe_h")]:
    galah = galah[galah[col].notna()].copy()
    if fc and fc in galah.columns:
        galah = galah[galah[fc] == 0].copy()
galah = galah[(galah["C_O"] > 0.05) & (galah["C_O"] < 2.0)].copy()

# Exclude cluster regions
cl_pos = t9_stars.groupby("cluster_name").agg(
    ra_cl=("ra_cl", "first"), dec_cl=("dec_cl", "first")).reset_index()
cl_c = np.deg2rad(np.column_stack([cl_pos["ra_cl"].values, cl_pos["dec_cl"].values]))
cl_xyz2 = np.column_stack([np.cos(cl_c[:,1])*np.cos(cl_c[:,0]),
                            np.cos(cl_c[:,1])*np.sin(cl_c[:,0]), np.sin(cl_c[:,1])])
tree2 = cKDTree(cl_xyz2)
gc2 = np.deg2rad(np.column_stack([galah["ra"].values, galah["dec"].values]))
gxyz2 = np.column_stack([np.cos(gc2[:,1])*np.cos(gc2[:,0]),
                          np.cos(gc2[:,1])*np.sin(gc2[:,0]), np.sin(gc2[:,1])])
d2, _ = tree2.query(gxyz2, k=1)
galah = galah[d2 > 2*np.sin(np.deg2rad(0.5)/2)].copy().reset_index(drop=True)
info(f"Field stars: {len(galah)}")

field_matrix = galah[DIM_COLS].values
field_ba = galah["ba_fe"].values
field_ba_flag = galah["flag_ba_fe"].values if "flag_ba_fe" in galah.columns else np.zeros(len(galah))
# Valid Ba/Fe mask
ba_valid = ~np.isnan(field_ba) & (field_ba_flag == 0)
info(f"Field stars with valid Ba/Fe: {ba_valid.sum()}")
n_field = len(galah)

# 3. For each cluster: compare Ba/Fe of 4D-matched vs random
info("\n" + "-" * 60)
info("S-PROCESS CONSISTENCY TEST")

rng = np.random.default_rng(42)
results = []

info(f"\n{'Cluster':<22} {'N_m':>5} {'N_ba':>5} {'Cl Ba':>7} {'M Ba':>7} {'R Ba':>7} "
     f"{'|Δ_m|':>6} {'|Δ_r|':>6} {'Closer':>7}")

for cname in sorted(templates.keys()):
    t = templates[cname]
    centroid = t["centroid"]
    cl_ba = t["ba_fe_mean"]

    # 4D match
    delta = np.abs(field_matrix - centroid)
    match_idx = np.where((delta < FIXED_TOL).all(axis=1))[0]
    if len(match_idx) < 10:
        continue

    # Ba/Fe of matched stars (valid only)
    match_ba_mask = ba_valid[match_idx]
    match_ba = field_ba[match_idx[match_ba_mask]]
    if len(match_ba) < 10:
        continue

    # Random sample of Ba/Fe (same size, from valid Ba/Fe field stars)
    all_ba_valid_idx = np.where(ba_valid)[0]
    rand_idx = rng.choice(all_ba_valid_idx, size=min(len(match_ba), len(all_ba_valid_idx)), replace=False)
    rand_ba = field_ba[rand_idx]

    # Offset from cluster Ba/Fe
    delta_match = np.abs(match_ba - cl_ba)
    delta_rand = np.abs(rand_ba - cl_ba)

    med_match = float(np.median(match_ba))
    med_rand = float(np.median(rand_ba))
    med_delta_match = float(np.median(delta_match))
    med_delta_rand = float(np.median(delta_rand))
    closer = med_delta_match < med_delta_rand

    # Mann-Whitney: are matched Ba/Fe offsets smaller?
    mw_u, mw_p = stats.mannwhitneyu(delta_match, delta_rand, alternative="less")

    results.append({
        "cluster": cname, "age_gyr": t["age_gyr"],
        "n_match": len(match_idx), "n_ba": len(match_ba),
        "cl_ba_fe": cl_ba,
        "match_ba_median": med_match, "rand_ba_median": med_rand,
        "delta_match": med_delta_match, "delta_rand": med_delta_rand,
        "closer": closer, "mw_p": mw_p,
    })

    sig = "*" if mw_p < 0.05 else ""
    info(f"{cname:<22} {len(match_idx):>5} {len(match_ba):>5} {cl_ba:>+7.3f} "
         f"{med_match:>+7.3f} {med_rand:>+7.3f} "
         f"{med_delta_match:>6.3f} {med_delta_rand:>6.3f} {'YES' if closer else 'no':>7} {sig}")

rdf = pd.DataFrame(results)
info(f"\nClusters tested: {len(rdf)}")

# Aggregate
n_closer = int(rdf["closer"].sum())
n_sig = int((rdf["mw_p"] < 0.05).sum())
info(f"Matched Ba/Fe closer to cluster than random: {n_closer}/{len(rdf)} ({n_closer/len(rdf):.1%})")
info(f"Individually significant (p<0.05): {n_sig}/{len(rdf)}")

# Wilcoxon: is delta_match systematically < delta_rand?
wil_stat, wil_p = stats.wilcoxon(rdf["delta_match"] - rdf["delta_rand"], alternative="less")
info(f"\nWilcoxon (matched offset < random offset): W={wil_stat:.0f}, p={wil_p:.4e}")
if wil_p < 0.05:
    info("=> SIGNIFICANT: 4D-matched field stars are ALSO closer in Ba/Fe")
    info("   Ba/Fe is an INDEPENDENT 5th dimension confirming the chemical fingerprint")
    info("   The match is NOT a coincidence in 4D — it's a real multi-channel identity")
else:
    info("=> Not significant: Ba/Fe does not add confirmation")

# Binomial test: is fraction closer > 50%?
binom_p = stats.binomtest(n_closer, len(rdf), 0.5, alternative="greater").pvalue if len(rdf) > 0 else 1.0
info(f"Binomial (fraction closer > 50%): p={binom_p:.4e}")

# Mean offset comparison
mean_delta_m = rdf["delta_match"].mean()
mean_delta_r = rdf["delta_rand"].mean()
info(f"\nMean |Ba/Fe offset|: matched={mean_delta_m:.4f}, random={mean_delta_r:.4f}")
info(f"Reduction: {(1 - mean_delta_m/mean_delta_r)*100:.1f}%")

# Age stratification
info("\n" + "-" * 60)
info("AGE STRATIFICATION")
aged = rdf[rdf["age_gyr"].notna()].copy()
if len(aged) >= 10:
    rho_a, p_a = stats.spearmanr(aged["age_gyr"], aged["delta_match"] - aged["delta_rand"])
    info(f"Spearman (age vs Ba/Fe advantage): ρ={rho_a:.4f}, p={p_a:.4e}")

    bins = [(0, 0.5), (0.5, 1.5), (1.5, 4.0), (4.0, 10.0)]
    info(f"\n  {'Age bin':>12}  {'N':>4}  {'Frac closer':>12}  {'Mean Δ_m':>9}  {'Mean Δ_r':>9}")
    for lo, hi in bins:
        mask = (aged["age_gyr"] >= lo) & (aged["age_gyr"] < hi)
        ab = aged[mask]
        if len(ab) >= 3:
            fc = ab["closer"].mean()
            info(f"  {lo:.1f}-{hi:.1f} Gyr  {len(ab):>4}  {fc:>12.1%}  "
                 f"{ab['delta_match'].mean():>9.4f}  {ab['delta_rand'].mean():>9.4f}")

# Save
rdf.to_csv("t16d_sproc_data.csv", index=False)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("T16d S-process Consistency Check | Certan (2026) | GALAH DR4\n"
             "Do 4D-matched field stars also align in Ba/Fe (independent 5th dimension)?",
             fontsize=13, fontweight="bold", y=0.99)

# P1: Cluster Ba/Fe vs matched Ba/Fe median
ax = axes[0, 0]
ax.scatter(rdf["cl_ba_fe"], rdf["match_ba_median"], s=30, c="steelblue", alpha=0.6,
           label="Matched median")
ax.scatter(rdf["cl_ba_fe"], rdf["rand_ba_median"], s=15, c="lightcoral", alpha=0.4,
           label="Random median", marker="x")
lim = max(abs(rdf["cl_ba_fe"].min()), abs(rdf["cl_ba_fe"].max()),
          abs(rdf["match_ba_median"].min()), abs(rdf["match_ba_median"].max())) * 1.1
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.4, label="1:1")
ax.set_xlabel("Cluster mean [Ba/Fe]", fontsize=10)
ax.set_ylabel("Field star median [Ba/Fe]", fontsize=10)
ax.set_title("Ba/Fe: Cluster vs Matched Field Stars", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15)

# P2: |Δ Ba/Fe| distribution (matched vs random)
ax = axes[0, 1]
ax.hist(rdf["delta_match"], bins=25, alpha=0.7, color="steelblue", edgecolor="white",
        label=f"Matched (median={rdf['delta_match'].median():.3f})")
ax.hist(rdf["delta_rand"], bins=25, alpha=0.5, color="lightcoral", edgecolor="white",
        label=f"Random (median={rdf['delta_rand'].median():.3f})")
ax.set_xlabel("|Ba/Fe - cluster mean|", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title(f"Ba/Fe Offset Distribution (Wilcoxon p={wil_p:.2e})", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# P3: Scatter of delta_match vs delta_rand per cluster
ax = axes[1, 0]
sig_m = rdf["mw_p"] < 0.05
ax.scatter(rdf.loc[~sig_m, "delta_rand"], rdf.loc[~sig_m, "delta_match"],
           s=25, c="gray", alpha=0.4, label="Not sig")
if sig_m.any():
    ax.scatter(rdf.loc[sig_m, "delta_rand"], rdf.loc[sig_m, "delta_match"],
               s=40, c="steelblue", alpha=0.7, label="Sig (p<0.05)")
lim2 = max(rdf["delta_match"].max(), rdf["delta_rand"].max()) * 1.05
ax.plot([0, lim2], [0, lim2], "k--", lw=1, alpha=0.4, label="1:1 (no advantage)")
ax.set_xlabel("Random: median |ΔBa/Fe|", fontsize=10)
ax.set_ylabel("Matched: median |ΔBa/Fe|", fontsize=10)
ax.set_title(f"Per-Cluster Ba/Fe Consistency ({n_closer}/{len(rdf)} closer)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15)

# P4: Age dependence
ax = axes[1, 1]
if len(aged) > 0:
    advantage = aged["delta_rand"] - aged["delta_match"]
    ax.scatter(aged["age_gyr"], advantage, s=30, c="steelblue", alpha=0.6)
    ax.axhline(0, color="red", ls="--", lw=1.5, label="No advantage")
    if len(aged) >= 5:
        z = np.polyfit(aged["age_gyr"].values, advantage.values, 1)
        xf = np.linspace(0, aged["age_gyr"].max(), 100)
        ax.plot(xf, np.polyval(z, xf), "r--", lw=1.5, alpha=0.5)
    ax.set_xlabel("Cluster Age (Gyr)", fontsize=10)
    ax.set_ylabel("Ba/Fe advantage (Δ_random - Δ_matched)", fontsize=10)
    ax.set_title("Ba/Fe Consistency vs Age", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("t16d_sproc_plot.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
info("Saved: t16d_sproc_plot.png")

# Summary
info("\n" + "=" * 72)
info("T16d SUMMARY")
info("=" * 72)
info(f"Clusters tested: {len(rdf)}")
info(f"Ba/Fe closer in matched stars: {n_closer}/{len(rdf)} ({n_closer/len(rdf):.1%})")
info(f"Wilcoxon (matched < random offset): p = {wil_p:.4e}")
info(f"Significant clusters: {n_sig}/{len(rdf)}")
info(f"Mean Ba/Fe offset reduction: {(1 - mean_delta_m/mean_delta_r)*100:.1f}%")
if wil_p < 0.05:
    info("\n=> Ba/Fe CONFIRMS the 4D chemical match as a real multi-channel fingerprint.")
    info("   The 5th independent dimension validates dissolved cluster recovery.")
info("\nT16d complete.")

with open("t16d_results.txt", "w") as f:
    f.write("\n".join(out))

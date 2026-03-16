#!/usr/bin/env python3
"""
T16e — Kinematic Traceback Test
=================================
Certan (2026) | CCT | GALAH DR4

Tests whether chemically matched field stars share kinematics (RV) with
their parent cluster more than random field stars. Stratifies by age to
test whether kinematic alignment decays faster than chemical alignment.
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

FIXED_TOL = np.array([0.08, 0.05, 0.05, 0.10])  # C/O, Mg/Fe, Si/Fe, Fe/H
DIM_COLS = ["C_O", "mg_fe", "si_fe", "fe_h"]
MIN_MEMBERS = 5
MIN_RV_MEMBERS = 3
CO_STD_THRESH = 0.10
RV_WINDOW = 10.0  # km/s for "close RV" test

out = []
def info(msg):
    line = "[INFO] " + str(msg)
    print(line, flush=True)
    out.append(line)

info("=" * 72)
info("T16e  Kinematic Traceback Test")
info("Certan (2026) | CCT | GALAH DR4")
info("=" * 72)

# 1. Load T9 data
t9_stars = pd.read_csv(T9_STARS)
t9_stats = pd.read_csv(T9_CLUSTERS)
info(f"T9: {len(t9_stars)} stars, {len(t9_stats)} clusters")

# Build templates
template_cl = t9_stats[(t9_stats["N"] >= MIN_MEMBERS) & (t9_stats["C_O_std"] < CO_STD_THRESH)]
templates = {}
for _, row in template_cl.iterrows():
    cname = row["cluster"]
    grp = t9_stars[t9_stars["cluster_name"] == cname].dropna(subset=DIM_COLS)
    if len(grp) < MIN_MEMBERS:
        continue
    rvs = grp["rv_gaia_dr3"].dropna()
    if len(rvs) < MIN_RV_MEMBERS:
        continue
    templates[cname] = {
        "centroid": grp[DIM_COLS].mean().values,
        "rv_mean": float(rvs.mean()),
        "rv_std": float(rvs.std(ddof=1)) if len(rvs) > 1 else 5.0,
        "N": len(grp),
        "age_gyr": row["age_gyr"] if pd.notna(row["age_gyr"]) else np.nan,
    }

info(f"Templates with RV: {len(templates)}")

# 2. Load GALAH field stars
info("\nLoading GALAH field stars...")
galah_table = Table.read(GALAH_FITS, memmap=True)
cols = ["ra", "dec", "snr_px_ccd3", "flag_sp", "c_fe", "o_fe",
        "flag_c_fe", "flag_o_fe", "mg_fe", "flag_mg_fe", "si_fe",
        "flag_si_fe", "fe_h", "flag_fe_h", "rv_comp_1", "e_rv_comp_1"]
galah = galah_table[[c for c in cols if c in galah_table.colnames]].to_pandas()

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
cl_xyz = np.column_stack([np.cos(cl_c[:,1])*np.cos(cl_c[:,0]),
                           np.cos(cl_c[:,1])*np.sin(cl_c[:,0]), np.sin(cl_c[:,1])])
tree = cKDTree(cl_xyz)
gc = np.deg2rad(np.column_stack([galah["ra"].values, galah["dec"].values]))
gxyz = np.column_stack([np.cos(gc[:,1])*np.cos(gc[:,0]),
                         np.cos(gc[:,1])*np.sin(gc[:,0]), np.sin(gc[:,1])])
d, _ = tree.query(gxyz, k=1)
galah = galah[d > 2*np.sin(np.deg2rad(0.5)/2)].copy().reset_index(drop=True)
info(f"Field stars: {len(galah)}")

field_matrix = galah[DIM_COLS].values
field_rv = galah["rv_comp_1"].values
n_field = len(galah)

# 3. Matching + RV comparison
info("\n" + "-" * 60)
info("KINEMATIC TRACEBACK: fixed-threshold chemical match + RV test")

rng = np.random.default_rng(42)
results = []

info(f"\n{'Cluster':<22} {'Age':>6} {'N_m':>6} {'N_rv':>5} {'ΔRV_m':>7} {'ΔRV_r':>7} {'Ratio':>6} {'In10':>5} {'Exp10':>6} {'p':>10}")

for cname in sorted(templates.keys()):
    t = templates[cname]
    centroid = t["centroid"]
    cl_rv = t["rv_mean"]

    # Fixed-threshold match
    delta = np.abs(field_matrix - centroid)
    match_mask = (delta < FIXED_TOL).all(axis=1)
    match_idx = np.where(match_mask)[0]

    if len(match_idx) < 10:
        continue

    # RVs of matched stars
    match_rvs = field_rv[match_idx]
    valid_rv = ~np.isnan(match_rvs)
    match_rvs_clean = match_rvs[valid_rv]
    if len(match_rvs_clean) < 10:
        continue

    # RVs of random field stars (same size sample)
    all_valid_rv = np.where(~np.isnan(field_rv))[0]
    rand_idx = rng.choice(all_valid_rv, size=min(len(match_rvs_clean), len(all_valid_rv)), replace=False)
    rand_rvs = field_rv[rand_idx]

    # ΔRV relative to cluster
    drv_match = np.abs(match_rvs_clean - cl_rv)
    drv_rand = np.abs(rand_rvs - cl_rv)

    med_drv_match = float(np.median(drv_match))
    med_drv_rand = float(np.median(drv_rand))
    ratio = med_drv_match / med_drv_rand if med_drv_rand > 0 else np.inf

    # Fraction within ±10 km/s
    frac_in10_match = float(np.sum(drv_match < RV_WINDOW) / len(drv_match))
    frac_in10_rand = float(np.sum(drv_rand < RV_WINDOW) / len(drv_rand))

    # Mann-Whitney: is matched ΔRV smaller?
    mw_u, mw_p = stats.mannwhitneyu(drv_match, drv_rand, alternative="less")

    age = t["age_gyr"]
    age_str = f"{age:.3f}" if not np.isnan(age) else "   --"

    results.append({
        "cluster": cname, "age_gyr": age,
        "n_match": len(match_idx), "n_rv": len(match_rvs_clean),
        "cl_rv": cl_rv,
        "med_drv_match": med_drv_match, "med_drv_rand": med_drv_rand,
        "ratio": ratio,
        "frac_in10_match": frac_in10_match, "frac_in10_rand": frac_in10_rand,
        "mw_p": mw_p,
    })

    sig = "*" if mw_p < 0.05 else ""
    info(f"{cname:<22} {age_str:>6} {len(match_idx):>6} {len(match_rvs_clean):>5} "
         f"{med_drv_match:>7.1f} {med_drv_rand:>7.1f} {ratio:>6.3f} "
         f"{frac_in10_match:>5.1%} {frac_in10_rand:>6.1%} {mw_p:>10.4e} {sig}")

rdf = pd.DataFrame(results)
info(f"\nClusters tested: {len(rdf)}")
n_sig = int((rdf["mw_p"] < 0.05).sum())
info(f"Significant (p<0.05, matched ΔRV < random): {n_sig}/{len(rdf)}")

# Aggregate
info("\n" + "-" * 60)
info("AGGREGATE RESULTS")
info(f"Median ΔRV ratio (match/random): {rdf['ratio'].median():.4f}")
info(f"Mean ΔRV ratio: {rdf['ratio'].mean():.4f}")

# Wilcoxon: is ratio systematically < 1?
wil_stat, wil_p = stats.wilcoxon(rdf["ratio"] - 1.0, alternative="less")
info(f"Wilcoxon (ratio < 1): W={wil_stat:.0f}, p={wil_p:.4e}")
if wil_p < 0.05:
    info("=> SIGNIFICANT: matched field stars have SMALLER ΔRV than random")
    info("   Residual kinematic coherence detected in chemically matched stars!")
else:
    info("=> Not significant: no kinematic preference detected")

# Fraction within 10 km/s
info(f"\nMedian frac within ±{RV_WINDOW} km/s:")
info(f"  Matched: {rdf['frac_in10_match'].median():.3f}")
info(f"  Random:  {rdf['frac_in10_rand'].median():.3f}")
wil2_stat, wil2_p = stats.wilcoxon(rdf["frac_in10_match"] - rdf["frac_in10_rand"], alternative="greater")
info(f"  Wilcoxon (matched > random): p={wil2_p:.4e}")

# 4. Age stratification
info("\n" + "-" * 60)
info("AGE STRATIFICATION: does kinematic alignment decay?")
aged = rdf[rdf["age_gyr"].notna()].copy()
if len(aged) >= 10:
    rho_age, p_age = stats.spearmanr(aged["age_gyr"], aged["ratio"])
    info(f"Spearman (age vs ΔRV ratio): ρ={rho_age:.4f}, p={p_age:.4e}")
    if p_age < 0.05 and rho_age > 0:
        info("=> Kinematic alignment DECAYS with age (ratio increases)")
        info("   Chemistry outlasts kinematics — confirms T17")
    elif p_age < 0.05 and rho_age < 0:
        info("=> Kinematic alignment IMPROVES with age (unexpected)")

    bins = [(0, 0.5), (0.5, 1.5), (1.5, 4.0), (4.0, 10.0)]
    info(f"\n  {'Age bin':>12}  {'N':>4}  {'Med ratio':>10}  {'Med frac10':>11}  {'N sig':>6}")
    for lo, hi in bins:
        mask = (aged["age_gyr"] >= lo) & (aged["age_gyr"] < hi)
        ab = aged[mask]
        if len(ab) >= 3:
            ns = (ab["mw_p"] < 0.05).sum()
            info(f"  {lo:.1f}-{hi:.1f} Gyr  {len(ab):>4}  {ab['ratio'].median():>10.4f}  "
                 f"{ab['frac_in10_match'].median():>11.3f}  {ns:>6}")

# 5. Save
rdf.to_csv("t16e_kinematic_data.csv", index=False)
with open("t16e_results.txt", "w") as f:
    f.write("\n".join(out))

# 6. Plot
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("T16e Kinematic Traceback | Certan (2026) | GALAH DR4\n"
             "Do chemically matched field stars share kinematics with parent cluster?",
             fontsize=13, fontweight="bold", y=0.99)

# P1: ΔRV ratio distribution
ax = axes[0, 0]
ax.hist(rdf["ratio"], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(1.0, color="red", ls="--", lw=2, label="No preference (1.0)")
ax.axvline(rdf["ratio"].median(), color="navy", ls="-", lw=2,
           label=f"Median = {rdf['ratio'].median():.3f}")
ax.set_xlabel("ΔRV ratio (match/random)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("RV Alignment Ratio Distribution", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# P2: ΔRV ratio vs age
ax = axes[0, 1]
if len(aged) > 0:
    ax.scatter(aged["age_gyr"], aged["ratio"], s=30, c="steelblue", alpha=0.6)
    ax.axhline(1.0, color="red", ls="--", lw=1.5)
    if len(aged) >= 5:
        z = np.polyfit(aged["age_gyr"], aged["ratio"], 1)
        xf = np.linspace(0, aged["age_gyr"].max(), 100)
        ax.plot(xf, np.polyval(z, xf), "r--", lw=1.5, alpha=0.5)
    rho_str = f"ρ={rho_age:.3f}" if len(aged) >= 10 else ""
    ax.set_title(f"ΔRV Ratio vs Age ({rho_str})", fontsize=11, fontweight="bold")
ax.set_xlabel("Cluster Age (Gyr)", fontsize=10)
ax.set_ylabel("ΔRV ratio", fontsize=10)

# P3: Frac within ±10 km/s (matched vs random)
ax = axes[0, 2]
ax.scatter(rdf["frac_in10_rand"], rdf["frac_in10_match"], s=25, c="steelblue", alpha=0.5)
lim = max(rdf["frac_in10_match"].max(), rdf["frac_in10_rand"].max()) * 1.1
ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.4, label="1:1")
ax.set_xlabel(f"Random: frac within ±{RV_WINDOW:.0f} km/s", fontsize=10)
ax.set_ylabel(f"Matched: frac within ±{RV_WINDOW:.0f} km/s", fontsize=10)
ax.set_title("RV Proximity Test", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# P4: p-value distribution
ax = axes[1, 0]
ax.hist(-np.log10(rdf["mw_p"].clip(lower=1e-10)), bins=30, color="coral", edgecolor="white")
ax.axvline(-np.log10(0.05), color="red", ls="--", lw=1.5, label="p=0.05")
ax.set_xlabel("-log10(p)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("Mann-Whitney p-value Distribution", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# P5: Matched vs random median ΔRV per cluster
ax = axes[1, 1]
ax.scatter(rdf["med_drv_rand"], rdf["med_drv_match"], s=25, c="steelblue", alpha=0.5)
lim2 = max(rdf["med_drv_match"].max(), rdf["med_drv_rand"].max()) * 1.05
ax.plot([0, lim2], [0, lim2], "k--", lw=1, alpha=0.4, label="1:1")
ax.set_xlabel("Random: median |ΔRV| (km/s)", fontsize=10)
ax.set_ylabel("Matched: median |ΔRV| (km/s)", fontsize=10)
ax.set_title("Median ΔRV: Matched vs Random", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# P6: Binned age summary
ax = axes[1, 2]
if len(aged) >= 10:
    bin_data = []
    for lo, hi in [(0, 0.5), (0.5, 1.5), (1.5, 4.0), (4.0, 10.0)]:
        m = (aged["age_gyr"] >= lo) & (aged["age_gyr"] < hi)
        if m.sum() >= 3:
            bin_data.append({"mid": (lo+hi)/2, "ratio": aged.loc[m, "ratio"].median(),
                            "label": f"{lo:.1f}-{hi:.1f}", "N": m.sum()})
    if bin_data:
        bdf = pd.DataFrame(bin_data)
        ax.bar(range(len(bdf)), bdf["ratio"], color=plt.cm.viridis(np.linspace(0.2, 0.8, len(bdf))),
               edgecolor="white")
        ax.axhline(1.0, color="red", ls="--", lw=1.5)
        ax.set_xticks(range(len(bdf)))
        ax.set_xticklabels([f"{r['label']}\nN={r['N']}" for _, r in bdf.iterrows()], fontsize=8)
        ax.set_ylabel("Median ΔRV ratio", fontsize=10)
        ax.set_title("Kinematic Alignment by Age", fontsize=11, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("t16e_kinematic_plot.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
info(f"Saved: t16e_kinematic_plot.png")

# Summary
info("\n" + "=" * 72)
info("T16e SUMMARY")
info("=" * 72)
info(f"Clusters tested: {len(rdf)}")
info(f"Median ΔRV ratio: {rdf['ratio'].median():.4f}")
info(f"Wilcoxon (ratio < 1): p = {wil_p:.4e}")
info(f"Significant clusters (p<0.05): {n_sig}/{len(rdf)}")
if wil_p < 0.05:
    info("=> Residual kinematic coherence detected in chemically matched field stars")
else:
    info("=> No aggregate kinematic preference — fully thermalized")
info("T16e complete.")

with open("t16e_results.txt", "w") as f:
    f.write("\n".join(out))

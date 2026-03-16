#!/usr/bin/env python3
"""
T16c — The Permanence Test
============================
Certan (2025) | Coherent Capture Theory | GALAH DR4

The decisive test: does the chemical fingerprint decay or is it permanent?

T16-revised showed flat enrichment across all ages (2× at every age bin).
But the Mahalanobis matching uses cluster-specific covariance — older
surviving clusters are tighter (T17 survivorship), so their matching
ellipsoids shrink, potentially masking real decay.

This test uses a FIXED matching threshold — the same hyperbox in
(C/O, Mg/Fe, Si/Fe, Fe/H) space for every cluster, regardless of
internal scatter. If enrichment is flat with a fixed threshold, the
chemical signal is permanent. If it decays, the dissolved members are
drifting and T16-revised's flatness was an artifact.

The claim being tested: Every star carries its birth chemistry forever.
The Galaxy's archive is complete. The fingerprint is eternal.
"""

import numpy as np
import pandas as pd
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

RESULTS_FILE  = "t16c_results.txt"
PLOT_FILE     = "t16c_permanence_plot.png"
CSV_FILE      = "t16c_permanence_data.csv"

CO_STD_THRESH = 0.10     # template cluster selection
MIN_MEMBERS   = 5
N_MC          = 200      # Monte Carlo iterations
SOLAR_CO      = 0.549

# FIXED matching thresholds (same for ALL clusters)
# These are absolute tolerances in each dimension
# Set at typical GALAH precision level (~2× measurement uncertainty)
FIXED_TOL = {
    "C_O":   0.08,    # C/O ratio (linear space)
    "mg_fe": 0.05,    # [Mg/Fe] in dex
    "si_fe": 0.05,    # [Si/Fe] in dex
    "fe_h":  0.10,    # [Fe/H] in dex (broader — metallicity is coarser)
}

# Also test multiple threshold scales to map the enrichment-vs-scale curve
SCALE_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

DIM_COLS = ["C_O", "mg_fe", "si_fe", "fe_h"]
DIM_LABELS = ["C/O", "[Mg/Fe]", "[Si/Fe]", "[Fe/H]"]
NDIM = len(DIM_COLS)

out_lines = []
def info(msg):
    line = "[INFO] " + str(msg)
    print(line, flush=True)
    out_lines.append(line)

def save_results():
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(out_lines))

# ---------------------------------------------------------------------------
# 1. Load data (reuse T16b's pipeline for field star pool)
# ---------------------------------------------------------------------------
info("=" * 72)
info("T16c  The Permanence Test")
info("Certan (2025) | CCT | GALAH DR4")
info("=" * 72)
info("Question: Is the chemical fingerprint eternal or does it decay?")
info(f"Method: Fixed matching threshold for ALL clusters")
info(f"  C/O:    ±{FIXED_TOL['C_O']}")
info(f"  Mg/Fe:  ±{FIXED_TOL['mg_fe']} dex")
info(f"  Si/Fe:  ±{FIXED_TOL['si_fe']} dex")
info(f"  Fe/H:   ±{FIXED_TOL['fe_h']} dex")

t9_stars = pd.read_csv(T9_STARS)
t9_stats = pd.read_csv(T9_CLUSTERS)
info(f"\nT9 stars: {len(t9_stars)}, clusters: {len(t9_stats)}")

# Build templates (centroids only — no covariance used)
template_clusters = t9_stats[
    (t9_stats["N"] >= MIN_MEMBERS) & (t9_stats["C_O_std"] < CO_STD_THRESH)
]
info(f"Template clusters: {len(template_clusters)}")

templates = {}
for _, row in template_clusters.iterrows():
    cname = row["cluster"]
    grp = t9_stars[t9_stars["cluster_name"] == cname].copy()
    grp_clean = grp.dropna(subset=DIM_COLS)
    if len(grp_clean) < MIN_MEMBERS:
        continue

    centroid = grp_clean[DIM_COLS].mean().values
    templates[cname] = {
        "centroid": centroid,
        "N": len(grp_clean),
        "age_gyr": row["age_gyr"] if pd.notna(row["age_gyr"]) else np.nan,
        "C_O_std": row["C_O_std"],
    }

info(f"Valid templates: {len(templates)}")

# Load GALAH field stars (same pipeline as T16b — load from pre-processed if available)
info("\nLoading GALAH field star pool...")
from astropy.table import Table

galah_cols_needed = [
    "ra", "dec", "snr_px_ccd3", "flag_sp",
    "c_fe", "o_fe", "flag_c_fe", "flag_o_fe",
    "mg_fe", "flag_mg_fe", "si_fe", "flag_si_fe",
    "fe_h", "flag_fe_h",
]

galah_table = Table.read("galah_dr4_allstar_240705.fits", memmap=True)
galah_keep = [c for c in galah_cols_needed if c in galah_table.colnames]
galah = galah_table[galah_keep].to_pandas()

# Quality cuts
galah = galah[galah["snr_px_ccd3"] > 30].copy()
galah = galah[galah["flag_sp"] == 0].copy()
galah["C_O"] = (10.0 ** (galah["c_fe"] - galah["o_fe"])) * SOLAR_CO

for col, flag_col in [("C_O", None), ("mg_fe", "flag_mg_fe"),
                       ("si_fe", "flag_si_fe"), ("fe_h", "flag_fe_h")]:
    galah = galah[galah[col].notna()].copy()
    if flag_col and flag_col in galah.columns:
        galah = galah[galah[flag_col] == 0].copy()

galah = galah[(galah["C_O"] > 0.05) & (galah["C_O"] < 2.0)].copy()

# Exclude cluster members
from scipy.spatial import cKDTree
all_cl_pos = t9_stars.groupby("cluster_name").agg(
    ra_cl=("ra_cl", "first"), dec_cl=("dec_cl", "first")
).reset_index()
cl_coords = np.deg2rad(np.column_stack([all_cl_pos["ra_cl"].values, all_cl_pos["dec_cl"].values]))
cl_xyz = np.column_stack([
    np.cos(cl_coords[:, 1]) * np.cos(cl_coords[:, 0]),
    np.cos(cl_coords[:, 1]) * np.sin(cl_coords[:, 0]),
    np.sin(cl_coords[:, 1])
])
cl_tree = cKDTree(cl_xyz)

g_coords = np.deg2rad(np.column_stack([galah["ra"].values, galah["dec"].values]))
g_xyz = np.column_stack([
    np.cos(g_coords[:, 1]) * np.cos(g_coords[:, 0]),
    np.cos(g_coords[:, 1]) * np.sin(g_coords[:, 0]),
    np.sin(g_coords[:, 1])
])
tol_cart = 2 * np.sin(np.deg2rad(0.5) / 2)
dists_to_cl, _ = cl_tree.query(g_xyz, k=1)
galah_field = galah[dists_to_cl > tol_cart].copy().reset_index(drop=True)
info(f"Field star pool: {len(galah_field)}")

n_field = len(galah_field)
field_matrix = galah_field[DIM_COLS].values

# Fixed tolerance vector
tol_vec = np.array([FIXED_TOL[c] for c in DIM_COLS])

# ---------------------------------------------------------------------------
# 2. Fixed-threshold matching at baseline scale
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("FIXED-THRESHOLD MATCHING (scale = 1.0)")

cluster_counts = {}
for cname in sorted(templates.keys()):
    centroid = templates[cname]["centroid"]
    delta = np.abs(field_matrix - centroid)
    within = (delta < tol_vec).all(axis=1)
    cluster_counts[cname] = int(within.sum())

total_matches = sum(cluster_counts.values())
info(f"Total matches: {total_matches}")
info(f"Mean per cluster: {total_matches / len(templates):.0f}")

# ---------------------------------------------------------------------------
# 3. Monte Carlo: random-center null with FIXED threshold
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info(f"Monte Carlo ({N_MC} iterations, random centers, fixed threshold)...")

rng = np.random.default_rng(42)
mc_counts = {cname: np.zeros(N_MC) for cname in templates}

for mc_i in range(N_MC):
    if mc_i % 50 == 0:
        print(f"  MC {mc_i}/{N_MC}...", flush=True)
    for cname in templates:
        center_idx = rng.integers(0, n_field)
        fake_center = field_matrix[center_idx]
        delta = np.abs(field_matrix - fake_center)
        mc_counts[cname][mc_i] = (delta < tol_vec).all(axis=1).sum()

# ---------------------------------------------------------------------------
# 4. Compute enrichment per cluster
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("PER-CLUSTER ENRICHMENT (fixed threshold)")

enrich_data = []
n_sig = 0

for cname in sorted(templates.keys()):
    obs = cluster_counts[cname]
    mc = mc_counts[cname]
    mc_med = np.median(mc)
    mc_std = mc.std()
    enrich = obs / mc_med if mc_med > 0 else (np.inf if obs > 0 else 1.0)
    p_val = (mc >= obs).mean()
    if p_val == 0:
        p_val = 1.0 / (N_MC + 1)

    sig = p_val < 0.01 and enrich >= 2.0
    if sig:
        n_sig += 1

    age = templates[cname]["age_gyr"]
    enrich_data.append({
        "cluster": cname, "age_gyr": age, "N_template": templates[cname]["N"],
        "C_O_std": templates[cname]["C_O_std"],
        "observed": obs, "mc_median": mc_med, "mc_std": mc_std,
        "enrichment": enrich, "p_value": p_val, "significant": sig,
    })

edf = pd.DataFrame(enrich_data)
info(f"Significantly enriched (≥2× at p<0.01): {n_sig}/{len(templates)}")
info(f"Aggregate enrichment: {edf['observed'].sum() / max(edf['mc_median'].sum(), 1):.2f}×")

# Top enriched
info(f"\nTop enriched clusters:")
info(f"{'Cluster':<22} {'Age':>6} {'Obs':>7} {'MC':>7} {'Enrich':>7} {'p':>10}")
for _, row in edf.nlargest(15, "enrichment").iterrows():
    age_str = f"{row['age_gyr']:.3f}" if pd.notna(row['age_gyr']) else "  --"
    sig_str = "**" if row["significant"] else ""
    info(f"{row['cluster']:<22} {age_str:>6} {row['observed']:>7.0f} "
         f"{row['mc_median']:>7.0f} {row['enrichment']:>6.2f}× {row['p_value']:>10.4e} {sig_str}")

# ---------------------------------------------------------------------------
# 5. THE KEY TEST: enrichment vs age (fixed threshold)
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("*** THE PERMANENCE TEST: Enrichment vs Cluster Age ***")
info("Fixed threshold — no covariance adaptation — pure signal test")

aged = edf[edf["age_gyr"].notna()].copy()
aged["enrich_capped"] = aged["enrichment"].clip(upper=20)
info(f"Clusters with age: {len(aged)}")

rho_perm, p_perm = stats.spearmanr(aged["age_gyr"], aged["enrich_capped"])
info(f"\nSpearman (age vs enrichment): ρ = {rho_perm:.4f}, p = {p_perm:.4e}")

# Binned analysis
info("\nBinned enrichment (fixed threshold):")
age_bins = [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 10.0)]
bin_results = []
info(f"  {'Age bin':>12}  {'N':>4}  {'Med enrich':>11}  {'Mean enrich':>12}  "
     f"{'Med obs':>8}  {'Med MC':>8}  {'N sig':>6}")
for lo, hi in age_bins:
    mask = (aged["age_gyr"] >= lo) & (aged["age_gyr"] < hi)
    ab = aged[mask]
    if len(ab) >= 3:
        n_s = ab["significant"].sum()
        info(f"  {lo:.1f}-{hi:.1f} Gyr  {len(ab):>4}  {ab['enrich_capped'].median():>11.2f}  "
             f"{ab['enrich_capped'].mean():>12.2f}  "
             f"{ab['observed'].median():>8.0f}  {ab['mc_median'].median():>8.0f}  {n_s:>6}")
        bin_results.append({
            "age_lo": lo, "age_hi": hi, "age_mid": (lo + hi) / 2,
            "N": len(ab), "med_enrich": ab["enrich_capped"].median(),
            "mean_enrich": ab["enrich_capped"].mean(),
            "med_obs": ab["observed"].median(),
            "n_sig": int(n_s),
        })
    else:
        info(f"  {lo:.1f}-{hi:.1f} Gyr  {len(ab):>4}       --")

bin_df = pd.DataFrame(bin_results)

# Fit exponential decay
info("\nExponential decay fit: E(t) = E0 * exp(-t/τ)")
try:
    def enrich_decay(t, E0, tau):
        return E0 * np.exp(-t / tau)

    valid = aged[(aged["enrichment"] > 0) & (aged["enrichment"] < 20)].copy()
    if len(valid) >= 10:
        popt, pcov = curve_fit(
            enrich_decay, valid["age_gyr"].values, valid["enrichment"].values,
            p0=[3.0, 2.0], bounds=([0.1, 0.05], [50, 100]),
            maxfev=10000
        )
        E0_fit, tau_fit = popt
        perr = np.sqrt(np.diag(pcov))
        info(f"  E0 = {E0_fit:.3f} ± {perr[0]:.3f}")
        info(f"  τ  = {tau_fit:.3f} ± {perr[1]:.3f} Gyr")
        tau_fit_ok = True

        if tau_fit > 20:
            info(f"  => τ >> 10 Gyr: NO DECAY DETECTED. Fingerprint is PERMANENT.")
        elif tau_fit > 5:
            info(f"  => τ = {tau_fit:.1f} Gyr: Very slow decay. Fingerprint persists for Hubble time.")
        elif 1.0 < tau_fit < 5.0:
            info(f"  => τ = {tau_fit:.1f} Gyr: Moderate decay detected.")
            if 0.8 < tau_fit < 2.0:
                info(f"     MATCHES T14 disk heating timescale (τ=1.29 Gyr)")
        else:
            info(f"  => τ = {tau_fit:.1f} Gyr: Rapid decay.")
except Exception as e:
    info(f"  Decay fit failed: {e}")
    tau_fit = np.nan
    tau_fit_ok = False

# Flat model comparison
info("\nFlat model: E(t) = constant")
flat_val = aged["enrich_capped"].median()
resid_flat = aged["enrich_capped"] - flat_val
ss_flat = (resid_flat**2).sum()
if tau_fit_ok:
    resid_decay = aged["enrichment"].clip(upper=20) - enrich_decay(aged["age_gyr"].values, *popt)
    ss_decay = (resid_decay**2).sum()
    n = len(aged)
    aic_flat = n * np.log(ss_flat / n) + 2 * 1
    aic_decay = n * np.log(ss_decay / n) + 2 * 2
    info(f"  AIC(flat)  = {aic_flat:.1f}")
    info(f"  AIC(decay) = {aic_decay:.1f}")
    if aic_flat < aic_decay:
        info(f"  => FLAT MODEL PREFERRED (ΔAIC = {aic_decay - aic_flat:.1f})")
        info(f"     The data favor NO DECAY. Chemical fingerprint is permanent.")
    else:
        info(f"  => DECAY MODEL PREFERRED (ΔAIC = {aic_flat - aic_decay:.1f})")

# Mann-Whitney: young vs old enrichment
young = aged[aged["age_gyr"] < 1.0]["enrich_capped"]
old = aged[aged["age_gyr"] >= 1.0]["enrich_capped"]
if len(young) >= 10 and len(old) >= 10:
    mw_u, mw_p = stats.mannwhitneyu(young, old, alternative="two-sided")
    info(f"\nMann-Whitney (young<1Gyr vs old≥1Gyr enrichment):")
    info(f"  Young median: {young.median():.3f} (N={len(young)})")
    info(f"  Old median:   {old.median():.3f} (N={len(old)})")
    info(f"  U={mw_u:.0f}, p={mw_p:.4e}")
    if mw_p > 0.05:
        info(f"  => NOT DIFFERENT: young and old clusters have SAME enrichment")
        info(f"     Chemical fingerprint does not fade with time.")

# ---------------------------------------------------------------------------
# 6. Multi-scale enrichment curve
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("MULTI-SCALE ANALYSIS: enrichment vs matching radius")
info("How does enrichment change as we tighten/loosen the threshold?")

scale_results = []
for scale in SCALE_FACTORS:
    tol_scaled = tol_vec * scale
    total_obs_s = 0
    total_mc_s = 0

    for cname in templates:
        centroid = templates[cname]["centroid"]
        delta = np.abs(field_matrix - centroid)
        obs = int((delta < tol_scaled).all(axis=1).sum())
        total_obs_s += obs

        # Quick MC (50 iterations per scale)
        mc_sum = 0
        for _ in range(50):
            ci = rng.integers(0, n_field)
            fc = field_matrix[ci]
            d = np.abs(field_matrix - fc)
            mc_sum += (d < tol_scaled).all(axis=1).sum()
        total_mc_s += mc_sum / 50

    enrich_s = total_obs_s / total_mc_s if total_mc_s > 0 else np.inf
    scale_results.append({
        "scale": scale, "total_obs": total_obs_s,
        "total_mc": total_mc_s, "enrichment": enrich_s,
    })
    info(f"  Scale {scale:.2f}×: obs={total_obs_s:>10}, MC={total_mc_s:>10.0f}, "
         f"enrichment={enrich_s:.3f}×")

scale_df = pd.DataFrame(scale_results)

# ---------------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Saving outputs...")

edf.to_csv(CSV_FILE, index=False)
info(f"Saved: {CSV_FILE}")
save_results()
info(f"Saved: {RESULTS_FILE}")

# ---------------------------------------------------------------------------
# 8. Generate 4-panel plot
# ---------------------------------------------------------------------------
info("Generating plot...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("T16c: The Permanence Test | Certan (2025) | GALAH DR4\n"
             "Fixed-threshold chemical matching — does the fingerprint decay?",
             fontsize=13, fontweight="bold", y=0.99)

# Panel 1: Enrichment vs Age (the key plot)
ax1 = axes[0, 0]
sig_mask = aged["significant"]
ax1.scatter(aged.loc[~sig_mask, "age_gyr"], aged.loc[~sig_mask, "enrich_capped"],
            s=30, c="gray", alpha=0.4, edgecolors="none", rasterized=True)
if sig_mask.any():
    ax1.scatter(aged.loc[sig_mask, "age_gyr"], aged.loc[sig_mask, "enrich_capped"],
                s=70, c="crimson", alpha=0.8, edgecolors="white", linewidths=0.5,
                zorder=5, marker="D", label="Sig (≥2×, p<0.01)")

# Binned medians with error bars
if len(bin_df) > 0:
    ax1.plot(bin_df["age_mid"], bin_df["med_enrich"], "ko-", lw=2.5, markersize=10,
             zorder=10, label="Binned median")

# Flat line
ax1.axhline(flat_val, color="steelblue", ls="-", lw=2, alpha=0.7,
            label=f"Flat model (E={flat_val:.2f})")

# Decay fit
if tau_fit_ok:
    t_plot = np.linspace(0.01, 8, 200)
    ax1.plot(t_plot, enrich_decay(t_plot, *popt), "r--", lw=2, alpha=0.6,
             label=f"Decay: τ={tau_fit:.1f} Gyr")

# T14 reference
ax1.axvline(1.288, color="orange", ls=":", lw=1.5, alpha=0.5,
            label="T14 τ=1.29 Gyr")

ax1.axhline(1.0, color="black", ls=":", lw=1, alpha=0.3)
ax1.set_xlabel("Cluster Age (Gyr)", fontsize=12)
ax1.set_ylabel("Enrichment Factor (fixed threshold)", fontsize=11)
ax1.set_title(f"THE PERMANENCE TEST (ρ={rho_perm:.3f}, p={p_perm:.2e})",
              fontsize=12, fontweight="bold")
ax1.legend(fontsize=7.5, loc="upper right")
ax1.set_ylim(0, min(aged["enrich_capped"].quantile(0.98) * 1.3, 15))
ax1.grid(True, alpha=0.15)

# Panel 2: Binned bar chart
ax2 = axes[0, 1]
if len(bin_df) > 0:
    x2 = np.arange(len(bin_df))
    colors2 = plt.cm.viridis(np.linspace(0.2, 0.8, len(bin_df)))
    bars = ax2.bar(x2, bin_df["med_enrich"], color=colors2, edgecolor="white",
                   linewidth=1, alpha=0.85)
    for i, (_, row) in enumerate(bin_df.iterrows()):
        ax2.text(i, row["med_enrich"] + 0.05,
                 f"{row['med_enrich']:.2f}×\n(N={row['N']:.0f})",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.axhline(1.0, color="red", ls="--", lw=1.5, label="Random (1×)")
    labels2 = [f"{r['age_lo']:.1f}–{r['age_hi']:.1f}" for _, r in bin_df.iterrows()]
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=9)
    ax2.set_xlabel("Age Bin (Gyr)", fontsize=11)
    ax2.set_ylabel("Median Enrichment Factor", fontsize=11)
    ax2.set_title("Fixed-Threshold Enrichment by Age", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.15, axis="y")
    ax2.set_ylim(0, max(bin_df["med_enrich"]) * 1.4)

# Panel 3: Multi-scale enrichment
ax3 = axes[1, 0]
if len(scale_df) > 0:
    ax3.plot(scale_df["scale"], scale_df["enrichment"], "ko-", lw=2.5, markersize=10)
    ax3.axhline(1.0, color="red", ls="--", lw=1.5, alpha=0.6, label="Random (1×)")
    ax3.fill_between([0.4, 2.1], 1.0, 1.0, alpha=0.05, color="red")
    for _, row in scale_df.iterrows():
        ax3.annotate(f"{row['enrichment']:.2f}×",
                     xy=(row["scale"], row["enrichment"]),
                     xytext=(0, 12), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold")
    ax3.set_xlabel("Threshold Scale Factor", fontsize=11)
    ax3.set_ylabel("Aggregate Enrichment", fontsize=11)
    ax3.set_title("Enrichment vs Matching Radius", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.15)
    ax3.set_xlim(0.3, 2.2)

# Panel 4: Enrichment distribution histogram
ax4 = axes[1, 1]
ax4.hist(aged["enrich_capped"], bins=30, color="steelblue", edgecolor="white",
         alpha=0.8, density=True)
ax4.axvline(1.0, color="red", ls="--", lw=1.5, label="Random (1×)")
ax4.axvline(aged["enrich_capped"].median(), color="navy", ls="-", lw=2,
            label=f"Median = {aged['enrich_capped'].median():.2f}×")
ax4.set_xlabel("Enrichment Factor", fontsize=11)
ax4.set_ylabel("Density", fontsize=11)
ax4.set_title("Enrichment Distribution (all aged clusters)", fontsize=11, fontweight="bold")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.15)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(PLOT_FILE, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
info(f"Saved: {PLOT_FILE}")

# ---------------------------------------------------------------------------
# 9. Final Summary
# ---------------------------------------------------------------------------
info("\n" + "=" * 72)
info("T16c PERMANENCE TEST — SUMMARY")
info("=" * 72)
info(f"Templates:        {len(templates)} GALAH clusters")
info(f"Field stars:      {n_field}")
info(f"Matching:         FIXED threshold (same for all clusters)")
info(f"  C/O ±{FIXED_TOL['C_O']}, Mg/Fe ±{FIXED_TOL['mg_fe']}, "
     f"Si/Fe ±{FIXED_TOL['si_fe']}, Fe/H ±{FIXED_TOL['fe_h']}")
info(f"")
info(f"Spearman (age vs enrichment): ρ = {rho_perm:.4f}, p = {p_perm:.4e}")
if tau_fit_ok:
    info(f"Decay fit τ = {tau_fit:.3f} ± {perr[1]:.3f} Gyr")
info(f"")

if len(bin_df) > 0:
    info("Binned medians:")
    for _, row in bin_df.iterrows():
        info(f"  {row['age_lo']:.1f}–{row['age_hi']:.1f} Gyr: {row['med_enrich']:.2f}× (N={row['N']:.0f})")

info("")
# The verdict
if p_perm > 0.05 and (not tau_fit_ok or tau_fit > 10):
    info("╔══════════════════════════════════════════════════════════════╗")
    info("║  VERDICT: THE FINGERPRINT IS PERMANENT                     ║")
    info("║                                                            ║")
    info("║  No significant decay in enrichment with cluster age.      ║")
    info("║  Fixed-threshold matching eliminates survivorship bias.    ║")
    info("║  The chemical identity written at birth does not fade.     ║")
    info("║                                                            ║")
    info("║  Every star in the Galaxy still carries its birth          ║")
    info("║  certificate. The archive is complete.                     ║")
    info("╚══════════════════════════════════════════════════════════════╝")
elif p_perm < 0.05 and rho_perm < 0:
    info("╔══════════════════════════════════════════════════════════════╗")
    info("║  VERDICT: DECAY DETECTED                                   ║")
    info("║                                                            ║")
    if tau_fit_ok:
        info(f"║  τ = {tau_fit:.2f} Gyr — chemical memory fades on this timescale  ║")
    info("║  The fingerprint degrades. The archive is partial.         ║")
    info("╚══════════════════════════════════════════════════════════════╝")
elif p_perm < 0.05 and rho_perm > 0:
    info("╔══════════════════════════════════════════════════════════════╗")
    info("║  VERDICT: ENRICHMENT INCREASES WITH AGE                    ║")
    info("║                                                            ║")
    info("║  Older clusters leave MORE recoverable field stars.        ║")
    info("║  More dissolved members + permanent chemistry = stronger   ║")
    info("║  signal with time. The archive accumulates.                ║")
    info("╚══════════════════════════════════════════════════════════════╝")
else:
    info("VERDICT: Inconclusive at current precision.")

info(f"\nOutput files: {RESULTS_FILE}, {CSV_FILE}, {PLOT_FILE}")
info("T16c complete.")
save_results()

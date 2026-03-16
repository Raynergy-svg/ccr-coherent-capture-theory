#!/usr/bin/env python3
"""
T19 — The Galactic Radius Coherence Gradient
==============================================
Certan (2026) | Coherent Capture Theory | GALAH DR4

Tests whether clusters at different Galactocentric radii have different
coherence properties. Inner disk (R < 7 kpc) has higher SFR, more
turbulence, shorter dynamical timescales. Outer disk (R > 9 kpc) is
quieter.

If inner disk clusters show faster coherence decay or different
coherence fractions, that directly links CCT coherence to the local
dynamical environment.
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
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
T9_STARS     = "t9_matched_stars.csv"
T9_CLUSTERS  = "t9_cluster_stats_with_age.csv"

RESULTS_FILE = "t19_results.txt"
PLOT_FILE    = "t19_galactic_radius_plot.png"
CSV_FILE     = "t19_cluster_rgal.csv"

R_SUN        = 8.2      # kpc, Galactocentric radius of Sun
CO_THRESH    = 0.05     # C/O coherence threshold
MIN_MEMBERS  = 3
MIN_PER_BIN  = 10

# Radial bins
RADIAL_BINS = [
    (0, 7.0,   "Inner disk (R < 7 kpc)"),
    (7.0, 8.0, "Inner solar (7–8 kpc)"),
    (8.0, 9.0, "Outer solar (8–9 kpc)"),
    (9.0, 25,  "Outer disk (R > 9 kpc)"),
]

out_lines = []
def info(msg):
    line = "[INFO] " + str(msg)
    print(line, flush=True)
    out_lines.append(line)

# ---------------------------------------------------------------------------
# 1. Load data and compute Galactocentric radius
# ---------------------------------------------------------------------------
info("=" * 72)
info("T19  The Galactic Radius Coherence Gradient")
info("Certan (2026) | CCT | GALAH DR4")
info("=" * 72)

stars = pd.read_csv(T9_STARS)
clusters = pd.read_csv(T9_CLUSTERS)
info(f"T9 stars: {len(stars)}, clusters: {len(clusters)}")

# Get unique cluster positions
cl_pos = stars.groupby("cluster_name").agg(
    ra_cl=("ra_cl", "first"),
    dec_cl=("dec_cl", "first"),
    dist_cl=("dist_cl", "first")
).reset_index()

info(f"Unique clusters with positions: {len(cl_pos)}")
info(f"Distance range: {cl_pos['dist_cl'].min():.3f} – {cl_pos['dist_cl'].max():.2f} kpc")

# Compute Galactocentric radius using astropy
info("\nComputing Galactocentric coordinates...")

coords = SkyCoord(
    ra=cl_pos["ra_cl"].values * u.deg,
    dec=cl_pos["dec_cl"].values * u.deg,
    distance=cl_pos["dist_cl"].values * u.kpc,
    frame="icrs"
)

# Transform to Galactocentric
galcen = coords.transform_to(Galactocentric())
cl_pos["X_gal"] = galcen.x.to(u.kpc).value
cl_pos["Y_gal"] = galcen.y.to(u.kpc).value
cl_pos["Z_gal"] = galcen.z.to(u.kpc).value
cl_pos["R_gal"] = np.sqrt(cl_pos["X_gal"]**2 + cl_pos["Y_gal"]**2 + cl_pos["Z_gal"]**2)
cl_pos["R_cyl"] = np.sqrt(cl_pos["X_gal"]**2 + cl_pos["Y_gal"]**2)

# Use cylindrical R (in-plane) for disk analysis
cl_pos["R"] = cl_pos["R_cyl"]

info(f"R_gal range: {cl_pos['R'].min():.2f} – {cl_pos['R'].max():.2f} kpc")
info(f"R_gal median: {cl_pos['R'].median():.2f} kpc")
info(f"|Z| range: {np.abs(cl_pos['Z_gal']).min():.3f} – {np.abs(cl_pos['Z_gal']).max():.2f} kpc")

# Merge with cluster stats
cl_pos_map = dict(zip(cl_pos["cluster_name"], cl_pos["R"]))
z_map = dict(zip(cl_pos["cluster_name"], cl_pos["Z_gal"]))
clusters["R_gal"] = clusters["cluster"].map(cl_pos_map)
clusters["Z_gal"] = clusters["cluster"].map(z_map)

# Filter
valid = clusters[clusters["R_gal"].notna() & (clusters["N"] >= MIN_MEMBERS)].copy()
info(f"\nClusters with R_gal + valid stats: {len(valid)}")
info(f"Coherent (C/O std < {CO_THRESH}): {(valid['C_O_std'] < CO_THRESH).sum()}/{len(valid)}")

# ---------------------------------------------------------------------------
# 2. Radial distribution of coherence
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("COHERENCE BY GALACTOCENTRIC RADIUS")

info(f"\n{'Radial bin':<25} {'N':>5} {'Coh':>5} {'Frac':>7} {'Med std':>9} {'Med age':>8}")

bin_results = []
for lo, hi, label in RADIAL_BINS:
    mask = (valid["R_gal"] >= lo) & (valid["R_gal"] < hi)
    b = valid[mask]
    n = len(b)
    if n < MIN_PER_BIN:
        info(f"{label:<25} {n:>5}    --      --        --       --")
        continue

    n_coh = (b["C_O_std"] < CO_THRESH).sum()
    frac = n_coh / n
    med_std = b["C_O_std"].median()
    med_age = b["age_gyr"].median() if b["age_gyr"].notna().sum() > 0 else np.nan

    info(f"{label:<25} {n:>5} {n_coh:>5} {frac:>7.1%} {med_std:>9.4f} {med_age:>8.3f}")

    bin_results.append({
        "R_lo": lo, "R_hi": hi, "label": label,
        "N": n, "n_coherent": n_coh, "frac_coherent": frac,
        "C_O_std_median": med_std, "C_O_std_mean": b["C_O_std"].mean(),
        "age_median": med_age,
        "feh_median": b["feh_mean"].median(),
    })

bin_df = pd.DataFrame(bin_results)

# ---------------------------------------------------------------------------
# 3. Statistical tests: inner vs outer
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("STATISTICAL TESTS")

inner = valid[valid["R_gal"] < 7.5]
solar = valid[(valid["R_gal"] >= 7.5) & (valid["R_gal"] < 8.5)]
outer = valid[valid["R_gal"] >= 8.5]

info(f"\nInner (R < 7.5 kpc): N={len(inner)}")
info(f"Solar (7.5-8.5 kpc): N={len(solar)}")
info(f"Outer (R > 8.5 kpc): N={len(outer)}")

# Mann-Whitney: inner vs outer C/O scatter
if len(inner) >= 10 and len(outer) >= 10:
    u_io, p_io = stats.mannwhitneyu(inner["C_O_std"], outer["C_O_std"],
                                      alternative="two-sided")
    info(f"\nMann-Whitney (inner vs outer C/O scatter):")
    info(f"  Inner median: {inner['C_O_std'].median():.4f}")
    info(f"  Outer median: {outer['C_O_std'].median():.4f}")
    info(f"  U={u_io:.0f}, p={p_io:.4e}")
    if p_io < 0.05:
        if inner["C_O_std"].median() < outer["C_O_std"].median():
            info("  => SIGNIFICANT: inner disk clusters MORE coherent")
        else:
            info("  => SIGNIFICANT: outer disk clusters MORE coherent")
    else:
        info("  => Not significant")

# Kruskal-Wallis across all radial bins
if len(bin_df) >= 3:
    groups = []
    group_labels = []
    for lo, hi, label in RADIAL_BINS:
        mask = (valid["R_gal"] >= lo) & (valid["R_gal"] < hi)
        b = valid.loc[mask, "C_O_std"].values
        if len(b) >= MIN_PER_BIN:
            groups.append(b)
            group_labels.append(label)

    if len(groups) >= 3:
        kw_stat, kw_p = stats.kruskal(*groups)
        info(f"\nKruskal-Wallis across radial bins: H={kw_stat:.3f}, p={kw_p:.4e}")
        if kw_p < 0.05:
            info("  => SIGNIFICANT: C/O scatter differs across Galactocentric radius")
        else:
            info("  => Not significant: no radial dependence of scatter")

# Coherent fraction comparison (Fisher exact)
if len(inner) >= 10 and len(outer) >= 10:
    a = (inner["C_O_std"] < CO_THRESH).sum()
    b_val = len(inner) - a
    c = (outer["C_O_std"] < CO_THRESH).sum()
    d = len(outer) - c
    odds, fisher_p = stats.fisher_exact([[a, b_val], [c, d]])
    info(f"\nFisher exact (coherent fraction inner vs outer):")
    info(f"  Inner: {a}/{len(inner)} = {a/len(inner):.1%}")
    info(f"  Outer: {c}/{len(outer)} = {c/len(outer):.1%}")
    info(f"  OR={odds:.2f}, p={fisher_p:.4e}")

# Spearman: continuous R_gal vs C/O scatter
rho_r, p_r = stats.spearmanr(valid["R_gal"], valid["C_O_std"])
info(f"\nSpearman (R_gal vs C/O scatter): ρ={rho_r:.4f}, p={p_r:.4e}")
if p_r < 0.05 and rho_r > 0:
    info("  => C/O scatter INCREASES with R_gal (inner disk more coherent)")
elif p_r < 0.05 and rho_r < 0:
    info("  => C/O scatter DECREASES with R_gal (outer disk more coherent)")
else:
    info("  => No significant radial trend")

# ---------------------------------------------------------------------------
# 4. Coherence decay rate by radial zone
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("DECAY RATE BY RADIAL ZONE")
info("Testing whether τ varies with Galactocentric radius...")

def exp_decay(t, sigma0, A, tau):
    return sigma0 + A * (1.0 - np.exp(-t / tau))

zones = [
    (0, 8.0, "Inner (R < 8 kpc)"),
    (8.0, 25, "Outer (R ≥ 8 kpc)"),
]

zone_fits = {}
for r_lo, r_hi, zlabel in zones:
    mask = (valid["R_gal"] >= r_lo) & (valid["R_gal"] < r_hi) & valid["age_gyr"].notna()
    z = valid[mask]
    if len(z) < 20:
        info(f"  {zlabel}: insufficient data (N={len(z)})")
        continue

    ages = z["age_gyr"].values
    scatter = z["C_O_std"].values

    rho_z, p_z = stats.spearmanr(ages, scatter)
    info(f"\n  {zlabel} (N={len(z)}):")
    info(f"    Spearman (age vs scatter): ρ={rho_z:.4f}, p={p_z:.4e}")

    try:
        p0 = [0.05, 0.10, 1.5]
        bounds = ([0, 0, 0.05], [0.5, 1.0, 50.0])
        popt, pcov = curve_fit(exp_decay, ages, scatter, p0=p0,
                                bounds=bounds, maxfev=10000)
        sigma0, A, tau = popt
        perr = np.sqrt(np.diag(pcov))
        info(f"    Exp fit: σ0={sigma0:.4f}, A={A:.4f}, τ={tau:.3f}±{perr[2]:.3f} Gyr")
        zone_fits[zlabel] = {"popt": popt, "perr": perr, "tau": tau,
                             "tau_err": perr[2], "N": len(z), "rho": rho_z, "p": p_z}
    except Exception as e:
        info(f"    Exp fit failed: {e}")

if len(zone_fits) >= 2:
    taus = {k: v["tau"] for k, v in zone_fits.items()}
    info(f"\n  Decay timescale comparison:")
    for label, tau in taus.items():
        err = zone_fits[label]["tau_err"]
        info(f"    {label}: τ = {tau:.3f} ± {err:.3f} Gyr")
    tau_vals = list(taus.values())
    if max(tau_vals) > 1.5 * min(tau_vals):
        info("  => DIFFERENT decay rates: Galactic environment modulates coherence lifetime")
    else:
        info("  => Similar decay rates across radial zones")

# ---------------------------------------------------------------------------
# 5. Metallicity gradient and its effect
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("METALLICITY GRADIENT CHECK")
info("Is the coherence gradient driven by metallicity rather than dynamics?")

rho_feh_r, p_feh_r = stats.spearmanr(valid["R_gal"], valid["feh_mean"])
rho_feh_std, p_feh_std = stats.spearmanr(valid["feh_mean"], valid["C_O_std"])

info(f"Spearman (R_gal vs [Fe/H]):    ρ={rho_feh_r:.4f}, p={p_feh_r:.4e}")
info(f"Spearman ([Fe/H] vs C/O std):  ρ={rho_feh_std:.4f}, p={p_feh_std:.4e}")

# Partial correlation: R_gal vs C/O_std controlling for [Fe/H]
from scipy.stats import pearsonr
if len(valid) > 20:
    # Rank-based partial correlation
    r_ranks = stats.rankdata(valid["R_gal"])
    std_ranks = stats.rankdata(valid["C_O_std"])
    feh_ranks = stats.rankdata(valid["feh_mean"])

    # Residualize
    slope_r, _, _, _, _ = stats.linregress(feh_ranks, r_ranks)
    slope_s, _, _, _, _ = stats.linregress(feh_ranks, std_ranks)
    resid_r = r_ranks - slope_r * feh_ranks
    resid_s = std_ranks - slope_s * feh_ranks
    rho_partial, p_partial = pearsonr(resid_r, resid_s)
    info(f"\nPartial correlation (R_gal vs C/O std | [Fe/H]):")
    info(f"  ρ_partial = {rho_partial:.4f}, p = {p_partial:.4e}")
    if p_partial < 0.05:
        info("  => Radial coherence gradient PERSISTS after controlling for metallicity")
        info("     The gradient is not simply a metallicity effect")
    else:
        info("  => Gradient disappears after controlling for metallicity")
        info("     Metallicity may be the primary driver")

# ---------------------------------------------------------------------------
# 6. Height above plane |Z| and coherence
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("DISK HEIGHT CHECK: |Z| vs coherence")

valid["abs_Z"] = np.abs(valid["Z_gal"])
rho_z_coh, p_z_coh = stats.spearmanr(valid["abs_Z"].dropna(),
                                       valid.loc[valid["abs_Z"].notna(), "C_O_std"])
info(f"Spearman (|Z| vs C/O scatter): ρ={rho_z_coh:.4f}, p={p_z_coh:.4e}")

thin_disk = valid[valid["abs_Z"] < 0.3]
thick_disk = valid[valid["abs_Z"] >= 0.3]
if len(thin_disk) >= 10 and len(thick_disk) >= 10:
    u_z, p_z = stats.mannwhitneyu(thin_disk["C_O_std"], thick_disk["C_O_std"],
                                    alternative="two-sided")
    info(f"Thin disk (|Z|<0.3 kpc, N={len(thin_disk)}): median std = {thin_disk['C_O_std'].median():.4f}")
    info(f"Thick disk (|Z|≥0.3 kpc, N={len(thick_disk)}): median std = {thick_disk['C_O_std'].median():.4f}")
    info(f"Mann-Whitney: U={u_z:.0f}, p={p_z:.4e}")

# ---------------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Saving outputs...")

valid.to_csv(CSV_FILE, index=False)
info(f"Saved: {CSV_FILE}")

# ---------------------------------------------------------------------------
# 8. Generate 4-panel plot
# ---------------------------------------------------------------------------
info("Generating plot...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("T19 Galactic Radius Coherence Gradient | Certan (2026) | GALAH DR4\n"
             f"C/O coherence vs Galactocentric radius across {len(valid)} open clusters",
             fontsize=13, fontweight="bold", y=0.99)

# Panel 1: R_gal vs C/O scatter
ax1 = axes[0, 0]
has_age = valid["age_gyr"].notna()
if has_age.any():
    sc = ax1.scatter(valid.loc[has_age, "R_gal"], valid.loc[has_age, "C_O_std"],
                     c=valid.loc[has_age, "age_gyr"], cmap="viridis",
                     s=30, alpha=0.6, edgecolors="none", vmin=0, vmax=5,
                     rasterized=True)
    plt.colorbar(sc, ax=ax1, label="Age (Gyr)", pad=0.02)
if (~has_age).any():
    ax1.scatter(valid.loc[~has_age, "R_gal"], valid.loc[~has_age, "C_O_std"],
                c="gray", s=15, alpha=0.3, edgecolors="none", rasterized=True)

ax1.axhline(CO_THRESH, color="red", ls="--", lw=1.5, alpha=0.6, label=f"Threshold = {CO_THRESH}")
ax1.axvline(R_SUN, color="orange", ls=":", lw=1.5, alpha=0.6, label=f"R☉ = {R_SUN} kpc")

# Running median
r_sorted = valid.sort_values("R_gal")
window = max(30, len(valid) // 15)
r_rolling = r_sorted["R_gal"].rolling(window, center=True).median()
std_rolling = r_sorted["C_O_std"].rolling(window, center=True).median()
ax1.plot(r_rolling, std_rolling, "r-", lw=2.5, alpha=0.8, label=f"Running median (w={window})")

ax1.set_xlabel("Galactocentric Radius R (kpc)", fontsize=11)
ax1.set_ylabel("C/O Scatter (std)", fontsize=11)
ax1.set_title(f"C/O Scatter vs R_gal (ρ={rho_r:.3f}, p={p_r:.2e})", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.15)
ax1.set_xlim(4, 16)

# Panel 2: Coherent fraction by radial bin
ax2 = axes[0, 1]
if len(bin_df) > 0:
    x2 = np.arange(len(bin_df))
    colors2 = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(bin_df)))

    bars = ax2.bar(x2, bin_df["frac_coherent"], color=colors2, edgecolor="white",
                   linewidth=1, alpha=0.85)

    for i, (_, row) in enumerate(bin_df.iterrows()):
        ax2.text(i, row["frac_coherent"] + 0.01,
                 f"{row['frac_coherent']:.1%}\n(N={row['N']:.0f})",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels([r["label"] for _, r in bin_df.iterrows()], fontsize=8, rotation=15)
    ax2.set_ylabel("Fraction C/O-Coherent", fontsize=11)
    ax2.set_title("Coherent Fraction by Galactic Zone", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(bin_df["frac_coherent"]) * 1.3 + 0.05)
    ax2.grid(True, alpha=0.15, axis="y")

# Panel 3: Age vs scatter, split by inner/outer
ax3 = axes[1, 0]
for label, subset, color, marker in [
    ("Inner (R<8)", valid[(valid["R_gal"] < 8) & valid["age_gyr"].notna()], "#d62728", "o"),
    ("Outer (R≥8)", valid[(valid["R_gal"] >= 8) & valid["age_gyr"].notna()], "#1f77b4", "D"),
]:
    if len(subset) > 0:
        ax3.scatter(subset["age_gyr"], subset["C_O_std"], s=25, c=color,
                    marker=marker, alpha=0.5, edgecolors="none", label=label,
                    rasterized=True)
        # Trend line
        if len(subset) >= 10:
            z = np.polyfit(subset["age_gyr"].values, subset["C_O_std"].values, 1)
            xf = np.linspace(subset["age_gyr"].min(), subset["age_gyr"].max(), 50)
            ax3.plot(xf, np.polyval(z, xf), "--", color=color, lw=2, alpha=0.7)

ax3.axhline(CO_THRESH, color="red", ls="--", lw=1, alpha=0.5)
ax3.set_xlabel("Cluster Age (Gyr)", fontsize=11)
ax3.set_ylabel("C/O Scatter (std)", fontsize=11)
ax3.set_title("Age vs Scatter by Radial Zone", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.15)

# Panel 4: XY Galactic plane view
ax4 = axes[1, 1]
# Merge positions with coherence
cl_xy = cl_pos.merge(
    valid[["cluster", "C_O_std", "N"]].rename(columns={"cluster": "cluster_name"}),
    on="cluster_name", how="inner"
)

if len(cl_xy) > 0:
    coherent_xy = cl_xy[cl_xy["C_O_std"] < CO_THRESH]
    incoherent_xy = cl_xy[cl_xy["C_O_std"] >= CO_THRESH]

    ax4.scatter(incoherent_xy["X_gal"], incoherent_xy["Y_gal"],
                s=15, c="gray", alpha=0.3, edgecolors="none", label="Incoherent",
                rasterized=True)
    ax4.scatter(coherent_xy["X_gal"], coherent_xy["Y_gal"],
                s=30, c="crimson", alpha=0.7, edgecolors="white", linewidths=0.3,
                label=f"Coherent (N={len(coherent_xy)})", zorder=5)

    # Sun position
    ax4.scatter([0], [0], marker="*", s=200, c="gold", edgecolors="black",
                linewidths=0.8, zorder=10, label="Sun")
    # Galactic center
    ax4.scatter([R_SUN], [0], marker="+", s=150, c="black", linewidths=2,
                zorder=10, label="GC")

    # Circles
    for r in [6, 8, 10, 12]:
        theta = np.linspace(0, 2*np.pi, 200)
        # In astropy Galactocentric, Sun is at negative X
        ax4.plot((r - R_SUN) * np.cos(theta) + R_SUN,
                 (r - R_SUN) * np.sin(theta), ":", color="gray", alpha=0.2, lw=0.5)

    ax4.set_xlabel("X_gal (kpc)", fontsize=11)
    ax4.set_ylabel("Y_gal (kpc)", fontsize=11)
    ax4.set_title("Galactic Plane View: Coherent vs Incoherent Clusters",
                  fontsize=11, fontweight="bold")
    ax4.legend(fontsize=7, loc="upper left")
    ax4.set_aspect("equal")
    ax4.grid(True, alpha=0.15)
    # Set reasonable limits
    ax4.set_xlim(-5, 5)
    ax4.set_ylim(-5, 5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(PLOT_FILE, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
info(f"Saved: {PLOT_FILE}")

# ---------------------------------------------------------------------------
# 9. Final Summary
# ---------------------------------------------------------------------------
info("\n" + "=" * 72)
info("T19 SUMMARY")
info("=" * 72)
info(f"Clusters analyzed: {len(valid)}")
info(f"R_gal range: {valid['R_gal'].min():.2f} – {valid['R_gal'].max():.2f} kpc")
info(f"")
info(f"Spearman (R_gal vs C/O scatter): ρ={rho_r:.4f}, p={p_r:.4e}")

if len(bin_df) > 0:
    info(f"\nCoherent fractions by zone:")
    for _, row in bin_df.iterrows():
        info(f"  {row['label']:<25}: {row['frac_coherent']:.1%} (N={row['N']:.0f})")

if zone_fits:
    info(f"\nDecay timescales by zone:")
    for label, zf in zone_fits.items():
        info(f"  {label}: τ = {zf['tau']:.3f} ± {zf['tau_err']:.3f} Gyr")

if 'rho_partial' in dir() and p_partial < 0.05:
    info(f"\nPartial correlation (controlling for [Fe/H]): ρ={rho_partial:.4f}, p={p_partial:.4e}")
    info("  Radial gradient persists after metallicity correction")

info(f"\n|Z| vs coherence: ρ={rho_z_coh:.4f}, p={p_z_coh:.4e}")

# Interpretation
if p_r < 0.05 and rho_r > 0:
    info(f"\n=> KEY RESULT: C/O scatter increases with Galactocentric radius (ρ={rho_r:.4f})")
    info("   Inner disk clusters are MORE chemically coherent than outer disk clusters.")
    info("   Higher SFR and denser ISM in the inner disk produce more homogeneous")
    info("   birth clouds, leading to tighter chemical fingerprints.")
elif p_r < 0.05 and rho_r < 0:
    info(f"\n=> KEY RESULT: C/O scatter decreases with Galactocentric radius (ρ={rho_r:.4f})")
    info("   Outer disk clusters are MORE coherent — quieter dynamical environment")
    info("   preserves coherence longer, or lower metallicity reduces abundance scatter.")
else:
    info(f"\n=> No significant radial coherence gradient detected at current precision.")
    info("   Coherence properties are approximately uniform across the disk.")

info(f"\nOutput files:")
info(f"  {RESULTS_FILE}")
info(f"  {CSV_FILE}")
info(f"  {PLOT_FILE}")
info(f"\nT19 complete.")

# Save final results
with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(out_lines))

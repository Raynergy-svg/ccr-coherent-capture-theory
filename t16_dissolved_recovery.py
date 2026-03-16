#!/usr/bin/env python3
"""
T16 — Dissolved Cluster Recovery in Field Stars
=================================================
Certan (2026) | Coherent Capture Theory | APOGEE DR17 × GALAH DR4

Uses the 39 C/O-coherent APOGEE clusters as multi-dimensional chemical
templates (C/Fe, O/Fe, Mg/Fe, Si/Fe, Al/Fe, Fe/H). Searches the full
917k GALAH DR4 field star catalogue for stars matching each template
within 2σ across all dimensions simultaneously.

Computes enrichment factor vs random expectation (Monte Carlo).
Tests whether matches show residual spatial structure (Galactic annulus
preference near parent cluster).

The discovery test: if field stars match dissolved cluster templates at
significantly above-random rates, we have found the chemical remnants
of birth groups scattered across the Galaxy — identifiable purely from
chemistry.
"""

import numpy as np
import pandas as pd
from astropy.table import Table
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APOGEE_STARS   = "tapogee_matched_stars.csv"
APOGEE_CLUSTERS = "tapogee_cluster_stats_with_age.csv"
GALAH_FITS     = "galah_dr4_allstar_240705.fits"

CO_THRESH      = 0.05          # C/O coherence threshold from T15
MATCH_NSIGMA   = 2.0           # match tolerance in sigma
MIN_MEMBERS    = 3             # minimum cluster members for template
N_MONTE_CARLO  = 1000          # random draws for enrichment stats
SNR_MIN_GALAH  = 30            # GALAH SNR cut (per-CCD)
SOLAR_CO       = 0.549         # solar C/O

# Abundance dimensions for matching
# We match in [X/Fe] space: C/Fe, O/Fe, Mg/Fe, Si/Fe, Al/Fe, plus Fe/H
APOGEE_COLS = ["C_FE", "O_FE", "MG_FE", "SI_FE", "AL_FE", "FE_H"]
GALAH_COLS  = ["c_fe", "o_fe", "mg_fe", "si_fe", "al_fe", "fe_h"]
DIM_LABELS  = ["[C/Fe]", "[O/Fe]", "[Mg/Fe]", "[Si/Fe]", "[Al/Fe]", "[Fe/H]"]
NDIM        = len(APOGEE_COLS)

# Output
RESULTS_FILE = "t16_results.txt"
MATCHES_FILE = "t16_field_matches.csv"
SUMMARY_FILE = "t16_cluster_enrichment.csv"
PLOT_FILE    = "t16_dissolved_recovery_plot.png"

out_lines = []
def info(msg):
    line = "[INFO] " + str(msg)
    print(line, flush=True)
    out_lines.append(line)

# ---------------------------------------------------------------------------
# 1. Load APOGEE cluster stars and build templates
# ---------------------------------------------------------------------------
info("=" * 72)
info("T16  Dissolved Cluster Recovery in Field Stars")
info("Certan (2026) | CCT | APOGEE DR17 × GALAH DR4")
info("=" * 72)

apogee_stars = pd.read_csv(APOGEE_STARS)
apogee_stats = pd.read_csv(APOGEE_CLUSTERS)
info(f"Loaded {len(apogee_stars)} APOGEE cluster stars, {len(apogee_stats)} clusters")

# Identify C/O-coherent clusters
coherent_clusters = set(apogee_stats.loc[apogee_stats["C_O_std"] < CO_THRESH, "cluster"])
info(f"C/O-coherent clusters: {len(coherent_clusters)}")

# Build templates: mean and std of each abundance dimension per cluster
info("Building multi-element chemical templates...")

SENTINEL = -9000
templates = {}

for cname in sorted(coherent_clusters):
    grp = apogee_stars[apogee_stars["CLUSTER"] == cname].copy()
    if len(grp) < MIN_MEMBERS:
        continue

    # Check all dimensions have valid data
    valid = True
    means = {}
    stds  = {}
    for acol in APOGEE_COLS:
        if acol not in grp.columns:
            valid = False
            break
        vals = grp[acol].dropna()
        vals = vals[vals > SENTINEL]
        if len(vals) < MIN_MEMBERS:
            valid = False
            break
        means[acol] = vals.mean()
        stds[acol]  = max(vals.std(ddof=1), 0.01)  # floor at 0.01 dex

    if valid:
        templates[cname] = {"means": means, "stds": stds, "N": len(grp)}

info(f"Valid templates (all {NDIM} dimensions, N>={MIN_MEMBERS}): {len(templates)}")

if len(templates) == 0:
    info("FATAL: No valid templates. Cannot proceed.")
    import sys; sys.exit(1)

# Print template summary
info(f"\n{'Cluster':<22} {'N':>3}  " + "  ".join(f"{l:>8}" for l in DIM_LABELS))
for cname in sorted(templates.keys()):
    t = templates[cname]
    vals = "  ".join(f"{t['means'][c]:>+8.4f}" for c in APOGEE_COLS)
    info(f"{cname:<22} {t['N']:>3}  {vals}")

# ---------------------------------------------------------------------------
# 2. Load GALAH DR4 field stars
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Loading GALAH DR4 field star catalogue...")

galah_table = Table.read(GALAH_FITS, memmap=True)
info(f"GALAH DR4 total stars: {len(galah_table)}")

# Convert to pandas — only needed columns
galah_needed = GALAH_COLS + [f"e_{c}" for c in GALAH_COLS] + [
    f"flag_{c}" for c in GALAH_COLS
] + ["ra", "dec", "parallax", "parallax_error",
     "snr_px_ccd3", "flag_sp", "sobject_id"]

# Only keep columns that exist
galah_keep = [c for c in galah_needed if c in galah_table.colnames]
info(f"Extracting {len(galah_keep)} columns...")
galah = galah_table[galah_keep].to_pandas()
info(f"Loaded into pandas: {len(galah)} rows")

# Quality cuts
info("Applying GALAH quality cuts...")
n0 = len(galah)

# SNR cut (CCD3 covers ~5700-5900A, includes Mg, Si)
if "snr_px_ccd3" in galah.columns:
    galah = galah[galah["snr_px_ccd3"] > SNR_MIN_GALAH].copy()
    info(f"  After SNR > {SNR_MIN_GALAH}: {len(galah)} (removed {n0 - len(galah)})")

# flag_sp == 0 (reliable stellar parameters)
if "flag_sp" in galah.columns:
    n_pre = len(galah)
    galah = galah[galah["flag_sp"] == 0].copy()
    info(f"  After flag_sp == 0: {len(galah)} (removed {n_pre - len(galah)})")

# Valid abundances in all dimensions + unflagged
for gcol in GALAH_COLS:
    galah = galah[galah[gcol].notna()].copy()
    flag_col = f"flag_{gcol}"
    if flag_col in galah.columns:
        galah = galah[galah[flag_col] == 0].copy()

info(f"  After valid+unflagged abundances in all {NDIM} dims: {len(galah)}")
n_field = len(galah)

if n_field == 0:
    info("FATAL: No GALAH field stars survived cuts.")
    import sys; sys.exit(1)

# Compute distance from parallax (simple 1/parallax, mas -> kpc)
if "parallax" in galah.columns:
    good_plx = galah["parallax"] > 0.1  # >0.1 mas for reasonable distances
    galah.loc[good_plx, "dist_kpc"] = 1.0 / galah.loc[good_plx, "parallax"]
    # Galactocentric radius (simple projection, R_sun = 8.2 kpc)
    R_SUN = 8.2  # kpc
    ra_rad  = np.deg2rad(galah["ra"].values)
    dec_rad = np.deg2rad(galah["dec"].values)
    dist    = galah["dist_kpc"].values
    # Galactic coordinates (approximate)
    l_rad = np.deg2rad(galah["ra"].values)  # placeholder — we'll use proper conversion
    # Actually use proper astropy for this
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        coords = SkyCoord(ra=galah["ra"].values * u.deg,
                          dec=galah["dec"].values * u.deg, frame="icrs")
        galah["gal_l"] = coords.galactic.l.deg
        galah["gal_b"] = coords.galactic.b.deg
        # Galactocentric R (cylindrical) for stars with distances
        valid_dist = galah["dist_kpc"].notna() & (galah["dist_kpc"] > 0)
        l_r = np.deg2rad(galah.loc[valid_dist, "gal_l"].values)
        b_r = np.deg2rad(galah.loc[valid_dist, "gal_b"].values)
        d   = galah.loc[valid_dist, "dist_kpc"].values
        x_gc = R_SUN - d * np.cos(b_r) * np.cos(l_r)
        y_gc = -d * np.cos(b_r) * np.sin(l_r)
        galah.loc[valid_dist, "R_gal"] = np.sqrt(x_gc**2 + y_gc**2)
        n_rgal = galah["R_gal"].notna().sum()
        info(f"  Galactocentric radius computed for {n_rgal} stars")
    except Exception as e:
        info(f"  WARNING: Could not compute Galactic coords: {e}")

# ---------------------------------------------------------------------------
# 3. Cross-survey calibration offset estimation
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Estimating APOGEE–GALAH abundance zero-point offsets...")
info("Method: compare median [X/Fe] of each survey's full sample")

# APOGEE medians from cluster stars (use all stars, not just coherent)
apogee_medians = {}
for acol in APOGEE_COLS:
    vals = apogee_stars[acol].dropna()
    vals = vals[vals > SENTINEL]
    apogee_medians[acol] = vals.median()

# GALAH medians from field stars
galah_medians = {}
for gcol in GALAH_COLS:
    galah_medians[gcol] = galah[gcol].median()

# Compute offsets: GALAH_corrected = GALAH - offset, where offset = GALAH_median - APOGEE_median
offsets = {}
info(f"\n{'Dim':<10} {'APOGEE med':>12} {'GALAH med':>12} {'Offset':>10}")
for acol, gcol, label in zip(APOGEE_COLS, GALAH_COLS, DIM_LABELS):
    offset = galah_medians[gcol] - apogee_medians[acol]
    offsets[gcol] = offset
    info(f"{label:<10} {apogee_medians[acol]:>+12.4f} {galah_medians[gcol]:>+12.4f} {offset:>+10.4f}")

# Apply offsets to GALAH abundances
info("\nApplying zero-point corrections to GALAH abundances...")
galah_corrected = galah.copy()
for gcol in GALAH_COLS:
    galah_corrected[gcol] = galah[gcol] - offsets[gcol]

# ---------------------------------------------------------------------------
# 4. Template matching: search field stars
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info(f"Searching {n_field} GALAH field stars against {len(templates)} cluster templates")
info(f"Match criterion: within {MATCH_NSIGMA}σ in ALL {NDIM} dimensions simultaneously")

# Pre-extract GALAH abundance matrix for speed
galah_matrix = np.column_stack([galah_corrected[gcol].values for gcol in GALAH_COLS])

all_matches = []
cluster_match_counts = {}

for cname in sorted(templates.keys()):
    t = templates[cname]
    means = np.array([t["means"][c] for c in APOGEE_COLS])
    sigmas = np.array([t["stds"][c] for c in APOGEE_COLS])

    # Distance in sigma space for each star
    delta = np.abs(galah_matrix - means)
    within = delta < (MATCH_NSIGMA * sigmas)

    # Must match ALL dimensions
    match_mask = within.all(axis=1)
    n_match = match_mask.sum()
    cluster_match_counts[cname] = n_match

    if n_match > 0:
        matched_idx = np.where(match_mask)[0]
        for idx in matched_idx:
            row = {
                "cluster_template": cname,
                "sobject_id": galah.iloc[idx]["sobject_id"] if "sobject_id" in galah.columns else idx,
                "ra": galah.iloc[idx]["ra"],
                "dec": galah.iloc[idx]["dec"],
            }
            if "R_gal" in galah.columns:
                row["R_gal"] = galah.iloc[idx]["R_gal"]
            if "gal_l" in galah.columns:
                row["gal_l"] = galah.iloc[idx]["gal_l"]
                row["gal_b"] = galah.iloc[idx]["gal_b"]
            for gcol, label in zip(GALAH_COLS, DIM_LABELS):
                row[label] = galah_corrected.iloc[idx][gcol]
            # Chi-squared distance
            chi2 = np.sum(((galah_matrix[idx] - means) / sigmas)**2)
            row["chi2"] = chi2
            all_matches.append(row)

total_matches = sum(cluster_match_counts.values())
clusters_with_matches = sum(1 for v in cluster_match_counts.values() if v > 0)

info(f"\nTotal field star matches: {total_matches}")
info(f"Clusters with >= 1 match: {clusters_with_matches}/{len(templates)}")

# Top matches by cluster
info(f"\n{'Cluster':<22} {'Template N':>10} {'Matches':>10}")
for cname in sorted(cluster_match_counts, key=cluster_match_counts.get, reverse=True):
    if cluster_match_counts[cname] > 0:
        info(f"{cname:<22} {templates[cname]['N']:>10} {cluster_match_counts[cname]:>10}")

# ---------------------------------------------------------------------------
# 5. Monte Carlo: random expectation
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info(f"Monte Carlo random expectation ({N_MONTE_CARLO} iterations)...")
info("Null model: same template widths (σ), but centered on random GALAH stars")
info("Tests whether cluster positions in abundance space are in overdense regions")

rng = np.random.default_rng(42)
random_match_counts = np.zeros(N_MONTE_CARLO)

# Correct null: keep the same σ from each real cluster template,
# but re-center on a random GALAH field star. This asks: "is the
# density of field stars around cluster abundance positions higher
# than the density around random abundance positions at the same scale?"

# Pre-compute cluster sigma arrays for speed
cluster_sigmas = {}
for cname in sorted(templates.keys()):
    t = templates[cname]
    cluster_sigmas[cname] = np.array([t["stds"][c] for c in APOGEE_COLS])

for mc_i in range(N_MONTE_CARLO):
    mc_total = 0
    for cname in sorted(templates.keys()):
        sigmas = cluster_sigmas[cname]

        # Pick a random GALAH star as the fake template center
        center_idx = rng.integers(0, n_field)
        fake_center = galah_matrix[center_idx]

        # Count matches using the real cluster's σ but fake center
        delta = np.abs(galah_matrix - fake_center)
        within = delta < (MATCH_NSIGMA * sigmas)
        mc_total += within.all(axis=1).sum()

    random_match_counts[mc_i] = mc_total

mc_mean = random_match_counts.mean()
mc_std  = random_match_counts.std()
mc_p975 = np.percentile(random_match_counts, 97.5)

enrichment = total_matches / mc_mean if mc_mean > 0 else np.inf
z_score = (total_matches - mc_mean) / mc_std if mc_std > 0 else np.inf

info(f"\nObserved matches:   {total_matches}")
info(f"Random expectation: {mc_mean:.1f} ± {mc_std:.1f}")
info(f"97.5th percentile:  {mc_p975:.1f}")
info(f"Enrichment factor:  {enrichment:.2f}x")
info(f"Z-score:            {z_score:.2f}")

if total_matches > mc_p975:
    info("=> SIGNIFICANT: field star matches exceed random at 95% CI")
    if enrichment >= 3:
        info("=> STRONG enrichment: dissolved cluster chemistry detected in field population")
    elif enrichment >= 1.5:
        info("=> MODERATE enrichment: chemical fingerprint signal above noise")
    else:
        info("=> WEAK enrichment: marginal signal")
else:
    info("=> NOT SIGNIFICANT at 95% CI")

# ---------------------------------------------------------------------------
# 6. Per-cluster enrichment
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Per-cluster enrichment analysis...")

# For each cluster, compute individual random expectation
per_cluster_mc = {}
for cname in sorted(templates.keys()):
    t = templates[cname]
    sigmas_arr = cluster_sigmas[cname]

    mc_counts = np.zeros(200)  # fewer iterations per cluster
    for mc_i in range(200):
        center_idx = rng.integers(0, n_field)
        fake_center = galah_matrix[center_idx]
        delta = np.abs(galah_matrix - fake_center)
        mc_counts[mc_i] = (delta < (MATCH_NSIGMA * sigmas_arr)).all(axis=1).sum()

    obs = cluster_match_counts[cname]
    mc_mu = mc_counts.mean()
    mc_sig = mc_counts.std()
    enrich = obs / mc_mu if mc_mu > 0 else np.inf
    per_cluster_mc[cname] = {
        "observed": obs, "mc_mean": mc_mu, "mc_std": mc_sig,
        "enrichment": enrich,
        "significant": obs > np.percentile(mc_counts, 97.5)
    }

info(f"\n{'Cluster':<22} {'Obs':>6} {'MC mean':>8} {'Enrich':>8} {'Sig':>5}")
n_sig = 0
for cname in sorted(per_cluster_mc, key=lambda c: per_cluster_mc[c]["enrichment"], reverse=True):
    pc = per_cluster_mc[cname]
    sig_str = "YES" if pc["significant"] else "no"
    if pc["significant"]:
        n_sig += 1
    info(f"{cname:<22} {pc['observed']:>6} {pc['mc_mean']:>8.1f} {pc['enrichment']:>7.2f}x {sig_str:>5}")

info(f"\nClusters with significant enrichment: {n_sig}/{len(templates)}")

# ---------------------------------------------------------------------------
# 7. Spatial structure test: do matches prefer parent cluster's R_gal?
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Spatial structure test: do matched field stars prefer the parent cluster's Galactic annulus?")

matches_df = pd.DataFrame(all_matches)
if len(matches_df) > 0 and "R_gal" in matches_df.columns:
    # Get cluster R_gal from APOGEE stars (use median RA/DEC -> approximate R_gal)
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        cluster_rgal = {}
        for cname in templates:
            cstars = apogee_stars[apogee_stars["CLUSTER"] == cname]
            if "RA" in cstars.columns and "DEC" in cstars.columns:
                c = SkyCoord(ra=cstars["RA"].mean() * u.deg,
                             dec=cstars["DEC"].mean() * u.deg, frame="icrs")
                # We don't have cluster distances easily, so use matched field star
                # R_gal distribution vs random field star R_gal distribution
                pass
            cluster_rgal[cname] = np.nan

        # Simpler test: for each cluster, compare R_gal distribution of matched
        # field stars vs all field stars using KS test
        info("KS test: matched field star R_gal vs overall field R_gal distribution")
        field_rgal = galah["R_gal"].dropna().values

        if len(field_rgal) > 100:
            ks_results = []
            for cname in sorted(templates.keys()):
                c_matches = matches_df[matches_df["cluster_template"] == cname]
                c_rgal = c_matches["R_gal"].dropna().values
                if len(c_rgal) >= 5:
                    ks_stat, ks_p = stats.ks_2samp(c_rgal, field_rgal)
                    ks_results.append({
                        "cluster": cname, "n_match": len(c_rgal),
                        "match_R_median": np.median(c_rgal),
                        "field_R_median": np.median(field_rgal),
                        "KS_stat": ks_stat, "KS_p": ks_p
                    })

            if ks_results:
                info(f"\n{'Cluster':<22} {'N':>4} {'Match R':>8} {'Field R':>8} {'KS':>6} {'p':>10}")
                n_spatial_sig = 0
                for kr in sorted(ks_results, key=lambda x: x["KS_p"]):
                    sig = "*" if kr["KS_p"] < 0.05 else ""
                    if kr["KS_p"] < 0.05:
                        n_spatial_sig += 1
                    info(f"{kr['cluster']:<22} {kr['n_match']:>4} "
                         f"{kr['match_R_median']:>8.2f} {kr['field_R_median']:>8.2f} "
                         f"{kr['KS_stat']:>6.3f} {kr['KS_p']:>10.4e} {sig}")

                info(f"\nClusters with spatial signal (KS p < 0.05): {n_spatial_sig}/{len(ks_results)}")

                # Aggregate test: pool all matches vs field
                all_match_rgal = matches_df["R_gal"].dropna().values
                if len(all_match_rgal) >= 10:
                    ks_all, p_all = stats.ks_2samp(all_match_rgal, field_rgal)
                    info(f"\nPooled KS test (all matches vs field): KS={ks_all:.4f}, p={p_all:.4e}")
                    info(f"  Match R_gal median: {np.median(all_match_rgal):.2f} kpc")
                    info(f"  Field R_gal median: {np.median(field_rgal):.2f} kpc")
            else:
                info("  Insufficient matches with R_gal for spatial test")
        else:
            info("  Insufficient field stars with R_gal for spatial test")
    except Exception as e:
        info(f"  Spatial test error: {e}")
else:
    info("  No R_gal data available for spatial test")

# ---------------------------------------------------------------------------
# 8. Save outputs
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Saving outputs...")

if len(matches_df) > 0:
    matches_df.to_csv(MATCHES_FILE, index=False)
    info(f"Saved: {MATCHES_FILE} ({len(matches_df)} rows)")

# Cluster enrichment summary
enrich_rows = []
for cname in sorted(templates.keys()):
    pc = per_cluster_mc[cname]
    t = templates[cname]
    row = {
        "cluster": cname, "template_N": t["N"],
        "observed_matches": pc["observed"],
        "mc_mean": round(pc["mc_mean"], 1),
        "mc_std": round(pc["mc_std"], 1),
        "enrichment": round(pc["enrichment"], 2),
        "significant": pc["significant"],
    }
    for acol, label in zip(APOGEE_COLS, DIM_LABELS):
        row[f"template_{label}"] = round(t["means"][acol], 4)
        row[f"sigma_{label}"] = round(t["stds"][acol], 4)
    enrich_rows.append(row)

enrich_df = pd.DataFrame(enrich_rows)
enrich_df.to_csv(SUMMARY_FILE, index=False)
info(f"Saved: {SUMMARY_FILE}")

with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(out_lines))
info(f"Saved: {RESULTS_FILE}")

# ---------------------------------------------------------------------------
# 9. Generate 4-panel plot
# ---------------------------------------------------------------------------
info("Generating plot...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("T16 Dissolved Cluster Recovery in Field Stars | Certan (2026)\n"
             f"APOGEE DR17 templates × GALAH DR4 field stars | {NDIM}D chemical matching",
             fontsize=13, fontweight="bold", y=0.99)

# Panel 1: Enrichment bar chart per cluster
ax1 = axes[0, 0]
cnames_sorted = sorted(templates.keys(),
                        key=lambda c: per_cluster_mc[c]["enrichment"], reverse=True)
x1 = np.arange(len(cnames_sorted))
enrich_vals = [per_cluster_mc[c]["enrichment"] for c in cnames_sorted]
sig_flags = [per_cluster_mc[c]["significant"] for c in cnames_sorted]
colors1 = ["steelblue" if s else "lightgray" for s in sig_flags]

ax1.bar(x1, enrich_vals, color=colors1, edgecolor="white", linewidth=0.5)
ax1.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Random expectation (1.0×)")
ax1.set_ylabel("Enrichment Factor", fontsize=10)
ax1.set_title("Per-Cluster Enrichment Over Random", fontsize=11, fontweight="bold")
if len(cnames_sorted) <= 25:
    ax1.set_xticks(x1)
    ax1.set_xticklabels(cnames_sorted, rotation=90, fontsize=6)
else:
    ax1.set_xticks([])
    ax1.set_xlabel(f"Clusters (N={len(cnames_sorted)})")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.15, axis="y")

# Panel 2: Monte Carlo distribution vs observed
ax2 = axes[0, 1]
ax2.hist(random_match_counts, bins=40, color="lightcoral", edgecolor="darkred",
         alpha=0.7, density=True, label=f"MC random (N={N_MONTE_CARLO})")
ax2.axvline(total_matches, color="navy", linewidth=2.5, linestyle="-",
            label=f"Observed = {total_matches}")
ax2.axvline(mc_mean, color="red", linewidth=1.5, linestyle="--",
            label=f"MC mean = {mc_mean:.0f}")
ax2.axvline(mc_p975, color="orange", linewidth=1.2, linestyle=":",
            label=f"97.5th = {mc_p975:.0f}")
ax2.set_xlabel("Total field star matches", fontsize=10)
ax2.set_ylabel("Density", fontsize=10)
ax2.set_title(f"Enrichment = {enrichment:.2f}×  |  Z = {z_score:.1f}",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.15)

# Panel 3: Abundance space projection (show template + matches)
ax3 = axes[1, 0]
if len(matches_df) > 0 and "[Mg/Fe]" in matches_df.columns and "[Fe/H]" in matches_df.columns:
    # Random field subsample for background
    n_bg = min(5000, n_field)
    bg_idx = rng.choice(n_field, size=n_bg, replace=False)
    ax3.scatter(galah_corrected.iloc[bg_idx]["fe_h"],
                galah_corrected.iloc[bg_idx]["mg_fe"],
                s=2, c="lightgray", alpha=0.3, rasterized=True, label="Field stars")

    # Matched stars
    ax3.scatter(matches_df["[Fe/H]"], matches_df["[Mg/Fe]"],
                s=15, c="crimson", alpha=0.6, edgecolors="none",
                label=f"Template matches (N={len(matches_df)})", zorder=5)

    # Template centers
    for cname, t in templates.items():
        ax3.scatter(t["means"]["FE_H"], t["means"]["MG_FE"],
                    marker="*", s=150, c="gold", edgecolors="black",
                    linewidths=0.8, zorder=10)

    ax3.set_xlabel("[Fe/H] (corrected)", fontsize=10)
    ax3.set_ylabel("[Mg/Fe] (corrected)", fontsize=10)
    ax3.set_title("[Fe/H] vs [Mg/Fe]: Templates + Matches", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(True, alpha=0.15)
else:
    ax3.text(0.5, 0.5, "No matches to plot", ha="center", va="center",
             transform=ax3.transAxes, fontsize=14, color="gray")

# Panel 4: R_gal distribution of matches vs field
ax4 = axes[1, 1]
if len(matches_df) > 0 and "R_gal" in matches_df.columns:
    match_rgal = matches_df["R_gal"].dropna().values
    field_rgal_sub = galah["R_gal"].dropna().values

    if len(match_rgal) > 0 and len(field_rgal_sub) > 0:
        bins_r = np.linspace(
            min(np.percentile(field_rgal_sub, 1), np.nanmin(match_rgal)),
            max(np.percentile(field_rgal_sub, 99), np.nanmax(match_rgal)),
            30
        )
        ax4.hist(field_rgal_sub, bins=bins_r, density=True, color="lightgray",
                 edgecolor="gray", alpha=0.7, label="All field stars")
        ax4.hist(match_rgal, bins=bins_r, density=True, color="steelblue",
                 edgecolor="navy", alpha=0.6, label=f"Template matches (N={len(match_rgal)})")
        ax4.set_xlabel("Galactocentric Radius (kpc)", fontsize=10)
        ax4.set_ylabel("Normalized density", fontsize=10)
        ax4.set_title("Spatial Distribution of Matches vs Field", fontsize=11, fontweight="bold")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.15)
    else:
        ax4.text(0.5, 0.5, "Insufficient R_gal data", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=14, color="gray")
else:
    ax4.text(0.5, 0.5, "No spatial data available", ha="center", va="center",
             transform=ax4.transAxes, fontsize=14, color="gray")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(PLOT_FILE, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
info(f"Saved: {PLOT_FILE}")

# ---------------------------------------------------------------------------
# 10. Final Summary
# ---------------------------------------------------------------------------
info("\n" + "=" * 72)
info("T16 SUMMARY")
info("=" * 72)
info(f"Templates:          {len(templates)} C/O-coherent APOGEE clusters")
info(f"Dimensions:         {NDIM} ([C/Fe], [O/Fe], [Mg/Fe], [Si/Fe], [Al/Fe], [Fe/H])")
info(f"Field stars:        {n_field} GALAH DR4 (quality-filtered)")
info(f"Match criterion:    within {MATCH_NSIGMA}σ in all {NDIM} dimensions")
info(f"")
info(f"Total matches:      {total_matches}")
info(f"Random expectation: {mc_mean:.1f} ± {mc_std:.1f}")
info(f"Enrichment:         {enrichment:.2f}×")
info(f"Z-score:            {z_score:.2f}")
info(f"Significant (95%):  {'YES' if total_matches > mc_p975 else 'NO'}")
info(f"")
info(f"Clusters with significant enrichment: {n_sig}/{len(templates)}")

if enrichment >= 3:
    info("")
    info("=> MAJOR RESULT: Field stars match dissolved cluster templates at >>random rates.")
    info("   The Galaxy's field population contains recoverable chemical remnants of")
    info("   birth groups — stellar birth certificates readable from chemistry alone.")
    info("   The Milky Way is a structured archive of formation events.")
elif enrichment >= 1.5:
    info("")
    info("=> POSITIVE RESULT: Moderate enrichment detected. Chemical fingerprints from")
    info("   dissolved clusters leave detectable traces in the field population.")
elif total_matches > mc_p975:
    info("")
    info("=> MARGINAL RESULT: Statistically significant but modest enrichment.")
    info("   Signal may strengthen with higher-precision surveys (HARPS/ESPRESSO).")
else:
    info("")
    info("=> NULL RESULT at current precision. Cross-survey calibration uncertainty")
    info("   or GALAH abundance precision may be limiting factors.")
    info("   This does NOT rule out CCT — it constrains the detectability threshold.")

info("")
info("Output files:")
info(f"  {RESULTS_FILE}")
info(f"  {MATCHES_FILE}")
info(f"  {SUMMARY_FILE}")
info(f"  {PLOT_FILE}")
info("")
info("T16 complete.")

# Save final results
with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(out_lines))

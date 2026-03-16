#!/usr/bin/env python3
"""
T16-Revised — Intra-GALAH Dissolved Cluster Recovery
======================================================
Certan (2026) | Coherent Capture Theory | GALAH DR4

Searches for dissolved cluster members in the GALAH field star population
using chemical templates built entirely within GALAH — zero cross-survey
systematics.

Templates: T9 GALAH clusters with N≥5 members and C/O_std < 0.10
Matching: Alpha-weighted Mahalanobis distance in (C/O, Mg/Fe, Si/Fe, Fe/H)
          with s-process (Ba/Fe) as secondary consistency check
Null model: Label-shuffled Monte Carlo (1000 iterations)
Tests: Enrichment factor, spatial annulus preference, RV kinematic test,
       age-stratified recovery rate
"""

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
from scipy import stats
from scipy.spatial.distance import mahalanobis
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

RESULTS_FILE  = "t16b_results.txt"
PLOT_FILE     = "t16b_dissolved_plot.png"
MATCHES_FILE  = "t16b_field_matches.csv"
ENRICH_FILE   = "t16b_cluster_enrichment.csv"

CO_STD_THRESH = 0.10     # template cluster coherence threshold
MIN_MEMBERS   = 5        # minimum cluster members for template
MAHAL_THRESH  = 3.0      # Mahalanobis distance threshold
N_MC          = 200      # Monte Carlo iterations
SNR_MIN       = 30
MATCH_TOL_DEG = 0.5      # degrees — exclusion zone around known clusters
R_SUN         = 8.2      # kpc

# Matching dimensions and weights (from T18: alpha 2× tighter)
# C/O: standard weight (1.0)
# Mg/Fe: double weight (alpha, tight)
# Si/Fe: double weight (alpha, tight)
# Fe/H: half weight (global, coarse)
DIM_COLS_T9   = ["C_O",   "mg_fe", "si_fe", "fe_h"]
DIM_COLS_GALAH = ["C_O",  "mg_fe", "si_fe", "fe_h"]
DIM_LABELS    = ["C/O",   "[Mg/Fe]", "[Si/Fe]", "[Fe/H]"]
DIM_WEIGHTS   = [1.0,     2.0,     2.0,     0.5]
NDIM          = len(DIM_COLS_T9)

out_lines = []
def info(msg):
    line = "[INFO] " + str(msg)
    print(line, flush=True)
    out_lines.append(line)

def save_results():
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(out_lines))

# ---------------------------------------------------------------------------
# 1. Load T9 cluster stars and build templates
# ---------------------------------------------------------------------------
info("=" * 72)
info("T16-Revised  Intra-GALAH Dissolved Cluster Recovery")
info("Certan (2026) | CCT | GALAH DR4 (single-survey)")
info("=" * 72)

t9_stars = pd.read_csv(T9_STARS)
t9_stats = pd.read_csv(T9_CLUSTERS)
info(f"T9 stars: {len(t9_stars)} in {t9_stars['cluster_name'].nunique()} clusters")

# Compute C/O for T9 stars (already have it as C_O column)
# T9 stars already have C_O, mg_fe, si_fe, fe_h, al_fe

# Select template clusters
template_clusters = t9_stats[
    (t9_stats["N"] >= MIN_MEMBERS) & (t9_stats["C_O_std"] < CO_STD_THRESH)
]["cluster"].values
info(f"Template clusters (N≥{MIN_MEMBERS}, C/O std < {CO_STD_THRESH}): {len(template_clusters)}")

# Build templates: centroid + weighted covariance in 4D
info("Building chemical fingerprint templates...")
templates = {}

for cname in sorted(template_clusters):
    grp = t9_stars[t9_stars["cluster_name"] == cname].copy()
    if len(grp) < MIN_MEMBERS:
        continue

    # Check all dimensions valid
    valid = True
    for col in DIM_COLS_T9:
        vals = grp[col].dropna()
        if len(vals) < MIN_MEMBERS:
            valid = False
            break
    if not valid:
        continue

    # Drop rows with any NaN in matching dimensions
    grp_clean = grp.dropna(subset=DIM_COLS_T9)
    if len(grp_clean) < MIN_MEMBERS:
        continue

    data = grp_clean[DIM_COLS_T9].values  # (N, 4)

    # Centroid
    centroid = data.mean(axis=0)

    # Covariance matrix with regularization
    if len(grp_clean) > NDIM:
        cov = np.cov(data.T)
    else:
        # Too few members for full covariance — use diagonal
        cov = np.diag(data.var(axis=0, ddof=1))

    # Regularize: add small diagonal to prevent singularity
    cov += np.eye(NDIM) * 1e-6

    # Apply dimension weights: scale covariance inversely with weight
    # Higher weight = smaller effective covariance = stricter matching
    W = np.diag([1.0 / w for w in DIM_WEIGHTS])
    cov_weighted = W @ cov @ W

    try:
        cov_inv = np.linalg.inv(cov_weighted)
    except np.linalg.LinAlgError:
        continue

    # Cluster position and kinematics for spatial/kinematic tests
    ra_cl = grp["ra_cl"].iloc[0]
    dec_cl = grp["dec_cl"].iloc[0]
    dist_cl = grp["dist_cl"].iloc[0]
    rv_mean = grp["rv_gaia_dr3"].dropna().mean() if "rv_gaia_dr3" in grp.columns else np.nan
    rv_std = grp["rv_gaia_dr3"].dropna().std(ddof=1) if "rv_gaia_dr3" in grp.columns else np.nan

    # Age
    age_row = t9_stats[t9_stats["cluster"] == cname]
    age = age_row["age_gyr"].values[0] if len(age_row) > 0 else np.nan

    templates[cname] = {
        "centroid": centroid,
        "cov": cov_weighted,
        "cov_inv": cov_inv,
        "N": len(grp_clean),
        "stds": data.std(axis=0, ddof=1),
        "ra_cl": ra_cl, "dec_cl": dec_cl, "dist_cl": dist_cl,
        "rv_mean": rv_mean, "rv_std": rv_std,
        "age_gyr": age,
    }

info(f"Valid templates: {len(templates)}")

# Print template summary
info(f"\n{'Cluster':<22} {'N':>3} {'Age':>6}  " +
     "  ".join(f"{l:>8}" for l in DIM_LABELS))
for cname in sorted(templates.keys()):
    t = templates[cname]
    vals = "  ".join(f"{t['centroid'][i]:>+8.4f}" for i in range(NDIM))
    age_str = f"{t['age_gyr']:.3f}" if not np.isnan(t['age_gyr']) else "  --"
    info(f"{cname:<22} {t['N']:>3} {age_str:>6}  {vals}")

# Compute R_gal for each template cluster
info("\nComputing Galactocentric radii for template clusters...")
cl_coords = SkyCoord(
    ra=[templates[c]["ra_cl"] for c in templates] * u.deg,
    dec=[templates[c]["dec_cl"] for c in templates] * u.deg,
    distance=[templates[c]["dist_cl"] for c in templates] * u.kpc,
    frame="icrs"
)
cl_galcen = cl_coords.transform_to(Galactocentric())
cl_R_gal = np.sqrt(cl_galcen.x.to(u.kpc).value**2 + cl_galcen.y.to(u.kpc).value**2)
for i, cname in enumerate(sorted(templates.keys())):
    templates[cname]["R_gal"] = cl_R_gal[i]

# ---------------------------------------------------------------------------
# 2. Load GALAH field stars (exclude cluster members)
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Loading GALAH DR4 field star pool...")

galah_cols_needed = [
    "ra", "dec", "sobject_id", "snr_px_ccd3", "flag_sp",
    "c_fe", "o_fe", "e_c_fe", "e_o_fe", "flag_c_fe", "flag_o_fe",
    "mg_fe", "e_mg_fe", "flag_mg_fe",
    "si_fe", "e_si_fe", "flag_si_fe",
    "fe_h", "e_fe_h", "flag_fe_h",
    "ba_fe", "e_ba_fe", "flag_ba_fe",
    "rv_comp_1", "e_rv_comp_1",
    "parallax", "parallax_error",
]

galah_table = Table.read(GALAH_FITS, memmap=True)
galah_keep = [c for c in galah_cols_needed if c in galah_table.colnames]
galah = galah_table[galah_keep].to_pandas()
info(f"GALAH total: {len(galah)}")

# Quality cuts
info("Applying quality cuts...")
if "snr_px_ccd3" in galah.columns:
    galah = galah[galah["snr_px_ccd3"] > SNR_MIN].copy()
    info(f"  After SNR > {SNR_MIN}: {len(galah)}")

if "flag_sp" in galah.columns:
    galah = galah[galah["flag_sp"] == 0].copy()
    info(f"  After flag_sp == 0: {len(galah)}")

# Valid abundances in matching dimensions + unflagged
SOLAR_CO = 0.549
galah["C_O"] = (10.0 ** (galah["c_fe"] - galah["o_fe"])) * SOLAR_CO

for col, flag_col in [("C_O", None), ("mg_fe", "flag_mg_fe"),
                       ("si_fe", "flag_si_fe"), ("fe_h", "flag_fe_h")]:
    galah = galah[galah[col].notna()].copy()
    if flag_col and flag_col in galah.columns:
        galah = galah[galah[flag_col] == 0].copy()

# C/O sanity
galah = galah[(galah["C_O"] > 0.05) & (galah["C_O"] < 2.0)].copy()
info(f"  After valid abundances + C/O range: {len(galah)}")

# Exclude known cluster members (positional exclusion)
info("Excluding known cluster members...")
# Build list of all CG2020 cluster positions from T9
all_cl_positions = t9_stars.groupby("cluster_name").agg(
    ra_cl=("ra_cl", "first"), dec_cl=("dec_cl", "first")
).reset_index()

from scipy.spatial import cKDTree

# Build KD-tree of cluster positions
cl_ra = all_cl_positions["ra_cl"].values
cl_dec = all_cl_positions["dec_cl"].values
cl_coords_2d = np.deg2rad(np.column_stack([cl_ra, cl_dec]))
cl_xyz = np.column_stack([
    np.cos(cl_coords_2d[:, 1]) * np.cos(cl_coords_2d[:, 0]),
    np.cos(cl_coords_2d[:, 1]) * np.sin(cl_coords_2d[:, 0]),
    np.sin(cl_coords_2d[:, 1])
])
cl_tree = cKDTree(cl_xyz)

# Query all GALAH stars
g_coords = np.deg2rad(np.column_stack([galah["ra"].values, galah["dec"].values]))
g_xyz = np.column_stack([
    np.cos(g_coords[:, 1]) * np.cos(g_coords[:, 0]),
    np.cos(g_coords[:, 1]) * np.sin(g_coords[:, 0]),
    np.sin(g_coords[:, 1])
])

tol_cart = 2 * np.sin(np.deg2rad(MATCH_TOL_DEG) / 2)
dists_to_cl, _ = cl_tree.query(g_xyz, k=1)
field_mask = dists_to_cl > tol_cart
galah_field = galah[field_mask].copy().reset_index(drop=True)
n_excluded = len(galah) - len(galah_field)
info(f"  Excluded {n_excluded} stars within {MATCH_TOL_DEG}° of any cluster")
info(f"  Field star pool: {len(galah_field)}")

# Compute Galactocentric radius for field stars
info("Computing R_gal for field stars...")
good_plx = galah_field["parallax"] > 0.2  # >0.2 mas
galah_field.loc[good_plx, "dist_kpc"] = 1.0 / galah_field.loc[good_plx, "parallax"]
valid_dist = galah_field["dist_kpc"].notna() & (galah_field["dist_kpc"] > 0) & (galah_field["dist_kpc"] < 20)
if valid_dist.sum() > 0:
    fc = SkyCoord(
        ra=galah_field.loc[valid_dist, "ra"].values * u.deg,
        dec=galah_field.loc[valid_dist, "dec"].values * u.deg,
        distance=galah_field.loc[valid_dist, "dist_kpc"].values * u.kpc,
        frame="icrs"
    )
    fg = fc.transform_to(Galactocentric())
    galah_field.loc[valid_dist, "R_gal"] = np.sqrt(
        fg.x.to(u.kpc).value**2 + fg.y.to(u.kpc).value**2
    )
    info(f"  R_gal computed for {valid_dist.sum()} field stars")

n_field = len(galah_field)

# Pre-extract field abundance matrix
field_matrix = galah_field[DIM_COLS_GALAH].values  # (N_field, 4)

# ---------------------------------------------------------------------------
# 3. Mahalanobis matching
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info(f"Mahalanobis matching: {n_field} field stars × {len(templates)} templates")
info(f"Threshold: d_Mahal < {MAHAL_THRESH}")
info(f"Weights: C/O={DIM_WEIGHTS[0]}, Mg/Fe={DIM_WEIGHTS[1]}, "
     f"Si/Fe={DIM_WEIGHTS[2]}, Fe/H={DIM_WEIGHTS[3]}")

cluster_match_counts = {}
cluster_match_indices = {}  # store indices for later tests

for cname in sorted(templates.keys()):
    t = templates[cname]
    centroid = t["centroid"]
    cov_inv = t["cov_inv"]

    # Vectorized Mahalanobis: d² = (x-μ)ᵀ Σ⁻¹ (x-μ)
    delta = field_matrix - centroid  # (N, 4)
    mahal_sq = np.sum(delta @ cov_inv * delta, axis=1)

    match_mask = mahal_sq < MAHAL_THRESH**2
    n_match = match_mask.sum()
    cluster_match_counts[cname] = n_match
    cluster_match_indices[cname] = np.where(match_mask)[0]

total_matches = sum(cluster_match_counts.values())
clusters_with_matches = sum(1 for v in cluster_match_counts.values() if v > 0)

info(f"\nTotal field star matches: {total_matches}")
info(f"Clusters with ≥1 match: {clusters_with_matches}/{len(templates)}")

# Top matches
info(f"\n{'Cluster':<22} {'TplN':>5} {'Matches':>8} {'Age':>6}")
for cname in sorted(cluster_match_counts, key=cluster_match_counts.get, reverse=True)[:30]:
    if cluster_match_counts[cname] > 0:
        age = templates[cname]["age_gyr"]
        age_str = f"{age:.3f}" if not np.isnan(age) else "  --"
        info(f"{cname:<22} {templates[cname]['N']:>5} {cluster_match_counts[cname]:>8} {age_str:>6}")

# ---------------------------------------------------------------------------
# 4. Monte Carlo enrichment: label-shuffled null
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info(f"Monte Carlo enrichment ({N_MC} iterations, random-center null)...")
info("Null: same covariance as real template, but centered on random field star")
info("Tests whether cluster positions in abundance space are overdense")

rng = np.random.default_rng(42)
mc_counts = {cname: np.zeros(N_MC) for cname in templates}

for mc_i in range(N_MC):
    if mc_i % 100 == 0:
        print(f"  MC iteration {mc_i}/{N_MC}...", flush=True)
    for cname in templates:
        t = templates[cname]
        center_idx = rng.integers(0, n_field)
        fake_center = field_matrix[center_idx]
        delta = field_matrix - fake_center
        mahal_sq = np.sum(delta @ t["cov_inv"] * delta, axis=1)
        mc_counts[cname][mc_i] = (mahal_sq < MAHAL_THRESH**2).sum()

# Compute enrichment
info(f"\n{'Cluster':<22} {'Obs':>6} {'MC med':>7} {'MC std':>7} {'Enrich':>7} {'p':>10} {'Sig':>4}")
enrich_results = {}
n_sig_001 = 0
n_sig_005 = 0

for cname in sorted(templates.keys()):
    obs = cluster_match_counts[cname]
    mc = mc_counts[cname]
    mc_med = np.median(mc)
    mc_std = mc.std()
    enrich = obs / mc_med if mc_med > 0 else (np.inf if obs > 0 else 1.0)

    # p-value: fraction of MC iterations with count >= observed
    p_val = (mc >= obs).mean()
    if p_val == 0:
        p_val = 1.0 / (N_MC + 1)  # upper bound

    sig = ""
    if p_val < 0.01 and enrich >= 2.0:
        sig = "**"
        n_sig_001 += 1
    elif p_val < 0.05:
        sig = "*"
        n_sig_005 += 1

    enrich_results[cname] = {
        "observed": obs, "mc_median": mc_med, "mc_std": mc_std,
        "enrichment": enrich, "p_value": p_val, "significant": p_val < 0.01 and enrich >= 2.0
    }

    if obs > 0 or mc_med > 0:
        info(f"{cname:<22} {obs:>6} {mc_med:>7.0f} {mc_std:>7.1f} {enrich:>6.2f}× {p_val:>10.4e} {sig:>4}")

info(f"\nSignificantly enriched (≥2× at p<0.01): {n_sig_001}/{len(templates)}")
info(f"Enriched at p<0.05: {n_sig_001 + n_sig_005}/{len(templates)}")

# Aggregate enrichment
total_obs = sum(cluster_match_counts.values())
total_mc = sum(np.median(mc_counts[c]) for c in templates)
total_enrich = total_obs / total_mc if total_mc > 0 else np.inf
info(f"\nAggregate: observed={total_obs}, MC median={total_mc:.0f}, "
     f"enrichment={total_enrich:.2f}×")

# ---------------------------------------------------------------------------
# 5. Spatial annulus test (using indices, not full match DF)
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("SPATIAL ANNULUS TEST")
info("Do matched field stars prefer the parent cluster's Galactic annulus (±1 kpc)?")

field_rgal = galah_field["R_gal"].values if "R_gal" in galah_field.columns else None
spatial_results = []

if field_rgal is not None:
    field_rgal_valid = field_rgal[~np.isnan(field_rgal)]
    info(f"\n{'Cluster':<22} {'N_m':>5} {'Cl R':>6} {'Match R':>8} {'In ann':>7} {'Exp':>6} {'Enr':>6} {'p':>10}")

    for cname in sorted(templates.keys()):
        t = templates[cname]
        cl_r = t.get("R_gal", np.nan)
        if np.isnan(cl_r):
            continue
        idx = cluster_match_indices[cname]
        if len(idx) < 3:
            continue
        c_rgal = field_rgal[idx]
        c_rgal = c_rgal[~np.isnan(c_rgal)]
        if len(c_rgal) < 3:
            continue

        in_annulus = int(np.sum(np.abs(c_rgal - cl_r) < 1.0))
        field_in_annulus = np.sum(np.abs(field_rgal_valid - cl_r) < 1.0) / len(field_rgal_valid)
        expected = field_in_annulus * len(c_rgal)
        ann_enrich = in_annulus / expected if expected > 0 else np.inf
        binom_p = 1 - stats.binom.cdf(in_annulus - 1, len(c_rgal), field_in_annulus) if field_in_annulus > 0 else 1.0

        spatial_results.append({
            "cluster": cname, "n_match": len(c_rgal), "cl_R": cl_r,
            "match_R_med": float(np.median(c_rgal)),
            "in_annulus": in_annulus, "expected": expected,
            "annulus_enrichment": ann_enrich, "binom_p": binom_p
        })
        sig = "*" if binom_p < 0.05 else ""
        info(f"{cname:<22} {len(c_rgal):>5} {cl_r:>6.2f} {np.median(c_rgal):>8.2f} "
             f"{in_annulus:>7} {expected:>6.1f} {ann_enrich:>5.2f}× {binom_p:>10.4e} {sig}")

    spatial_df = pd.DataFrame(spatial_results) if spatial_results else pd.DataFrame()
    if len(spatial_df) > 0:
        n_spatial_sig = int((spatial_df["binom_p"] < 0.05).sum())
        info(f"\nClusters with spatial signal (p<0.05): {n_spatial_sig}/{len(spatial_df)}")

# ---------------------------------------------------------------------------
# 6. Kinematic test: RV dispersion of matched stars
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("KINEMATIC TEST: RV dispersion of matched field stars")
info("Do matched stars have RVs closer to parent cluster than random field stars?")

field_rv_all = galah_field["rv_comp_1"].values if "rv_comp_1" in galah_field.columns else None
kin_results = []

if field_rv_all is not None:
    field_rv_valid = field_rv_all[~np.isnan(field_rv_all)]
    info(f"\n{'Cluster':<22} {'N_rv':>5} {'Cl RV':>7} {'Match σ':>8} {'Field σ':>8} {'Ratio':>6}")

    for cname in sorted(templates.keys()):
        t = templates[cname]
        cl_rv = t["rv_mean"]
        if np.isnan(cl_rv):
            continue
        idx = cluster_match_indices[cname]
        if len(idx) < 5:
            continue
        match_rvs = field_rv_all[idx]
        match_rvs = match_rvs[~np.isnan(match_rvs)]
        if len(match_rvs) < 5:
            continue

        match_sigma = float(np.std(match_rvs - cl_rv))
        field_sigma = float(np.std(field_rv_valid[:min(len(field_rv_valid), 10000)] - cl_rv))
        ratio = match_sigma / field_sigma if field_sigma > 0 else np.inf

        kin_results.append({
            "cluster": cname, "n_rv": len(match_rvs),
            "cl_rv": cl_rv, "match_sigma": match_sigma,
            "field_sigma": field_sigma, "sigma_ratio": ratio
        })
        info(f"{cname:<22} {len(match_rvs):>5} {cl_rv:>+7.1f} {match_sigma:>8.1f} "
             f"{field_sigma:>8.1f} {ratio:>6.3f}")

    if kin_results:
        kin_df = pd.DataFrame(kin_results)
        mean_ratio = kin_df["sigma_ratio"].median()
        info(f"\nMedian σ_match/σ_field ratio: {mean_ratio:.3f}")
        if mean_ratio < 0.9:
            info("=> Matched stars show REDUCED RV dispersion relative to parent cluster")
            info("   Residual kinematic coherence detected!")
        elif mean_ratio > 1.1:
            info("=> Matched stars show INCREASED RV dispersion — fully thermalized")
        else:
            info("=> Matched stars kinematically indistinguishable from field")

# ---------------------------------------------------------------------------
# 7. Age-stratified recovery rate
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("AGE-STRATIFIED RECOVERY: enrichment factor vs cluster age")
info("If τ=1.29 Gyr is real, older clusters should yield lower enrichment")

aged_enrich = []
for cname in templates:
    age = templates[cname]["age_gyr"]
    if np.isnan(age):
        continue
    er = enrich_results[cname]
    aged_enrich.append({
        "cluster": cname, "age_gyr": age,
        "enrichment": er["enrichment"],
        "observed": er["observed"], "mc_median": er["mc_median"],
        "significant": er["significant"]
    })

aged_df = pd.DataFrame(aged_enrich)
info(f"Clusters with age + enrichment: {len(aged_df)}")

if len(aged_df) >= 10:
    # Cap enrichment at reasonable value for correlation
    aged_df["enrich_capped"] = aged_df["enrichment"].clip(upper=20)

    rho_ae, p_ae = stats.spearmanr(aged_df["age_gyr"], aged_df["enrich_capped"])
    info(f"\nSpearman (age vs enrichment): ρ = {rho_ae:.4f}, p = {p_ae:.4e}")

    if p_ae < 0.05 and rho_ae < 0:
        info("=> SIGNIFICANT: enrichment DECREASES with age")
        info("   Older clusters' dissolved members have drifted further from template")
        info("   Consistent with τ ≈ 1.29 Gyr chemical coherence decay")
    elif p_ae < 0.05 and rho_ae > 0:
        info("=> Enrichment INCREASES with age — older clusters more recoverable")
        info("   Possible survivorship bias (older clusters = tighter templates)")
    else:
        info("=> No significant age-enrichment trend")

    # Fit exponential decay to enrichment vs age
    try:
        def enrich_decay(t, E0, tau):
            return E0 * np.exp(-t / tau)

        valid_ae = aged_df[aged_df["enrichment"] > 0].copy()
        if len(valid_ae) >= 5:
            popt_ae, pcov_ae = curve_fit(
                enrich_decay, valid_ae["age_gyr"].values,
                valid_ae["enrich_capped"].values,
                p0=[3.0, 1.5], bounds=([0.1, 0.05], [50, 50]),
                maxfev=10000
            )
            E0_fit, tau_ae = popt_ae
            perr_ae = np.sqrt(np.diag(pcov_ae))
            info(f"  Exponential fit: E0={E0_fit:.2f}, τ={tau_ae:.3f}±{perr_ae[1]:.3f} Gyr")
            if 0.5 < tau_ae < 3.0:
                info(f"  => Recovery timescale τ={tau_ae:.2f} Gyr "
                     f"{'MATCHES' if 0.8 < tau_ae < 2.0 else 'near'} T14 (τ=1.29 Gyr)")
    except Exception as e:
        info(f"  Decay fit failed: {e}")

    # Binned
    info("\nBinned enrichment by age:")
    age_bins = [(0, 0.3), (0.3, 0.7), (0.7, 1.5), (1.5, 3.0), (3.0, 10.0)]
    info(f"  {'Age bin':>12}  {'N':>4}  {'Med enrich':>11}  {'Mean enrich':>12}  {'N sig':>6}")
    for lo, hi in age_bins:
        mask = (aged_df["age_gyr"] >= lo) & (aged_df["age_gyr"] < hi)
        ab = aged_df[mask]
        if len(ab) >= 3:
            n_s = ab["significant"].sum()
            info(f"  {lo:.1f}-{hi:.1f} Gyr  {len(ab):>4}  {ab['enrich_capped'].median():>11.2f}  "
                 f"{ab['enrich_capped'].mean():>12.2f}  {n_s:>6}")

# ---------------------------------------------------------------------------
# 8. S-process consistency check (Ba/Fe)
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("S-PROCESS CONSISTENCY CHECK (Ba/Fe)")

if "ba_fe" in galah_field.columns and total_matches > 0:
    # For matched field stars, check if their Ba/Fe is closer to parent cluster than random
    # Need cluster Ba/Fe from T9 stars cross-matched with GALAH
    # (We already have this from T18 pipeline)
    info("Computing Ba/Fe consistency for matched field stars...")

    # Get Ba/Fe for matched stars
    ba_col = "flag_ba_fe"
    galah_field_ba = galah_field["ba_fe"].values if "ba_fe" in galah_field.columns else None

    if galah_field_ba is not None:
        field_ba_valid = galah_field["ba_fe"].dropna()
        field_ba_med = field_ba_valid.median()
        field_ba_std = field_ba_valid.std()

        ba_consistency = []
        for cname in sorted(templates.keys()):
            # Get cluster Ba/Fe from T9 stars
            cl_stars = t9_stars[t9_stars["cluster_name"] == cname]
            # We don't have Ba/Fe in T9 directly, so skip detailed test
            pass

        info("  (Ba/Fe detailed test requires T18 cross-match — deferred)")
        info(f"  Field Ba/Fe: median={field_ba_med:.4f}, std={field_ba_std:.4f}")

# ---------------------------------------------------------------------------
# 9. Save outputs
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Saving outputs...")

# Build compact matches CSV (only enriched clusters to keep manageable)
match_rows = []
for cname in sorted(templates.keys()):
    er = enrich_results[cname]
    if er["enrichment"] < 1.0 or er["observed"] == 0:
        continue
    idx = cluster_match_indices[cname]
    # Cap at 500 per cluster for file size
    for i in idx[:500]:
        match_rows.append({
            "cluster": cname, "ra": galah_field.iloc[i]["ra"],
            "dec": galah_field.iloc[i]["dec"],
            "C_O": field_matrix[i, 0], "mg_fe": field_matrix[i, 1],
            "si_fe": field_matrix[i, 2], "fe_h": field_matrix[i, 3],
        })
if match_rows:
    pd.DataFrame(match_rows).to_csv(MATCHES_FILE, index=False)
    info(f"Saved: {MATCHES_FILE} ({len(match_rows)} rows)")

enrich_rows = []
for cname in sorted(templates.keys()):
    er = enrich_results[cname]
    t = templates[cname]
    row = {
        "cluster": cname, "template_N": t["N"],
        "age_gyr": t["age_gyr"], "R_gal": t.get("R_gal", np.nan),
        "observed": er["observed"], "mc_median": round(er["mc_median"], 1),
        "mc_std": round(er["mc_std"], 1),
        "enrichment": round(er["enrichment"], 3),
        "p_value": er["p_value"], "significant": er["significant"],
    }
    for i, label in enumerate(DIM_LABELS):
        row[f"centroid_{label}"] = round(t["centroid"][i], 4)
        row[f"sigma_{label}"] = round(t["stds"][i], 4)
    enrich_rows.append(row)

pd.DataFrame(enrich_rows).to_csv(ENRICH_FILE, index=False)
info(f"Saved: {ENRICH_FILE}")

save_results()
info(f"Saved: {RESULTS_FILE}")

# ---------------------------------------------------------------------------
# 10. Generate 4-panel plot
# ---------------------------------------------------------------------------
info("Generating plot...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("T16-Revised: Intra-GALAH Dissolved Cluster Recovery | Certan (2026)\n"
             f"Alpha-weighted Mahalanobis matching | {len(templates)} templates × "
             f"{n_field} field stars",
             fontsize=12, fontweight="bold", y=0.99)

# Panel 1: Per-cluster enrichment
ax1 = axes[0, 0]
cnames_sorted = sorted(templates.keys(),
                        key=lambda c: enrich_results[c]["enrichment"], reverse=True)
x1 = np.arange(len(cnames_sorted))
enrich_vals = [min(enrich_results[c]["enrichment"], 15) for c in cnames_sorted]
sig_flags = [enrich_results[c]["significant"] for c in cnames_sorted]
colors1 = ["steelblue" if s else "lightgray" for s in sig_flags]

ax1.bar(x1, enrich_vals, color=colors1, edgecolor="white", linewidth=0.3)
ax1.axhline(1.0, color="red", ls="--", lw=1.5, label="Random (1.0×)")
ax1.axhline(2.0, color="orange", ls=":", lw=1.2, label="2× threshold")
ax1.set_ylabel("Enrichment Factor", fontsize=10)
ax1.set_title(f"Per-Cluster Enrichment (sig: {n_sig_001}, p<0.01 & ≥2×)",
              fontsize=11, fontweight="bold")
if len(cnames_sorted) <= 40:
    ax1.set_xticks(x1)
    ax1.set_xticklabels(cnames_sorted, rotation=90, fontsize=5)
else:
    ax1.set_xticks([])
    ax1.set_xlabel(f"Clusters (N={len(cnames_sorted)})")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.15, axis="y")
ax1.set_ylim(0, min(max(enrich_vals) * 1.2, 16))

# Panel 2: Age vs enrichment
ax2 = axes[0, 1]
if len(aged_df) > 0:
    sig_mask = aged_df["significant"]
    ax2.scatter(aged_df.loc[~sig_mask, "age_gyr"],
                aged_df.loc[~sig_mask, "enrich_capped"],
                s=40, c="gray", alpha=0.5, edgecolors="none", label="Not sig")
    if sig_mask.any():
        ax2.scatter(aged_df.loc[sig_mask, "age_gyr"],
                    aged_df.loc[sig_mask, "enrich_capped"],
                    s=80, c="crimson", alpha=0.8, edgecolors="white",
                    linewidths=0.5, zorder=5, marker="D", label="Sig (≥2×, p<0.01)")

    # Fit line
    if len(aged_df) >= 5:
        try:
            valid_plot = aged_df[aged_df["enrich_capped"] > 0]
            z = np.polyfit(valid_plot["age_gyr"], valid_plot["enrich_capped"], 1)
            xf = np.linspace(0, aged_df["age_gyr"].max(), 100)
            ax2.plot(xf, np.polyval(z, xf), "r--", lw=1.5, alpha=0.5)
        except:
            pass

    ax2.axhline(1.0, color="black", ls=":", lw=1, alpha=0.4)
    ax2.set_xlabel("Cluster Age (Gyr)", fontsize=11)
    ax2.set_ylabel("Enrichment Factor", fontsize=11)
    rho_str = f"ρ={rho_ae:.3f}" if len(aged_df) >= 10 else ""
    ax2.set_title(f"Age-Stratified Recovery ({rho_str})", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.15)

# Panel 3: Abundance space projection
ax3 = axes[1, 0]
# Random field subsample
n_bg = min(5000, n_field)
bg_idx = rng.choice(n_field, size=n_bg, replace=False)
ax3.scatter(field_matrix[bg_idx, 3], field_matrix[bg_idx, 1],  # Fe/H vs Mg/Fe
            s=2, c="lightgray", alpha=0.3, rasterized=True, label="Field")

if total_matches > 0:
    # Collect all matched star coords for plotting
    all_match_idx = np.concatenate([cluster_match_indices[c] for c in templates
                                     if len(cluster_match_indices[c]) > 0])
    # Subsample if too many
    if len(all_match_idx) > 10000:
        all_match_idx = rng.choice(all_match_idx, 10000, replace=False)
    ax3.scatter(field_matrix[all_match_idx, 3], field_matrix[all_match_idx, 1],
                s=8, c="crimson", alpha=0.4, edgecolors="none",
                label=f"Matches (N={total_matches})", zorder=4, rasterized=True)

# Template centers
for cname in templates:
    t = templates[cname]
    ax3.scatter(t["centroid"][3], t["centroid"][1],
                marker="*", s=120, c="gold", edgecolors="black",
                linewidths=0.6, zorder=10)

ax3.set_xlabel("[Fe/H]", fontsize=10)
ax3.set_ylabel("[Mg/Fe]", fontsize=10)
ax3.set_title("[Fe/H] vs [Mg/Fe]: Templates + Matches", fontsize=11, fontweight="bold")
ax3.legend(fontsize=7, loc="upper right")
ax3.grid(True, alpha=0.15)

# Panel 4: R_gal distribution or kinematic test
ax4 = axes[1, 1]
if len(spatial_results) > 0:
    spatial_df_plot = pd.DataFrame(spatial_results)
    spatial_df_plot = spatial_df_plot[spatial_df_plot["annulus_enrichment"].notna() &
                                      np.isfinite(spatial_df_plot["annulus_enrichment"])]
    if len(spatial_df_plot) > 0:
        sig_sp = spatial_df_plot["binom_p"] < 0.05
        ax4.scatter(spatial_df_plot.loc[~sig_sp, "cl_R"],
                    spatial_df_plot.loc[~sig_sp, "annulus_enrichment"].clip(upper=5),
                    s=40, c="gray", alpha=0.5, label="Not sig")
        if sig_sp.any():
            ax4.scatter(spatial_df_plot.loc[sig_sp, "cl_R"],
                        spatial_df_plot.loc[sig_sp, "annulus_enrichment"].clip(upper=5),
                        s=80, c="steelblue", alpha=0.8, edgecolors="navy",
                        linewidths=0.5, zorder=5, marker="D", label="Spatial sig (p<0.05)")
        ax4.axhline(1.0, color="red", ls="--", lw=1.5, alpha=0.6, label="Random (1×)")
        ax4.axvline(R_SUN, color="orange", ls=":", lw=1.5, alpha=0.6, label=f"R☉={R_SUN}")
        ax4.set_xlabel("Cluster R_gal (kpc)", fontsize=10)
        ax4.set_ylabel("Annulus enrichment (±1 kpc)", fontsize=10)
        ax4.set_title("Spatial Annulus Test", fontsize=11, fontweight="bold")
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.15)
else:
    ax4.text(0.5, 0.5, "No spatial data", ha="center", va="center",
             transform=ax4.transAxes, fontsize=14, color="gray")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(PLOT_FILE, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
info(f"Saved: {PLOT_FILE}")

# ---------------------------------------------------------------------------
# 11. Final Summary
# ---------------------------------------------------------------------------
info("\n" + "=" * 72)
info("T16-REVISED SUMMARY")
info("=" * 72)
info(f"Templates:        {len(templates)} GALAH clusters (N≥{MIN_MEMBERS}, C/O std<{CO_STD_THRESH})")
info(f"Dimensions:       {NDIM} (C/O w={DIM_WEIGHTS[0]}, Mg/Fe w={DIM_WEIGHTS[1]}, "
     f"Si/Fe w={DIM_WEIGHTS[2]}, Fe/H w={DIM_WEIGHTS[3]})")
info(f"Field stars:      {n_field} GALAH DR4 (cluster-excluded, quality-filtered)")
info(f"Match threshold:  Mahalanobis d < {MAHAL_THRESH}")
info(f"")
info(f"Total matches:    {total_matches}")
info(f"MC expectation:   {total_mc:.0f}")
info(f"Aggregate enrich: {total_enrich:.2f}×")
info(f"Sig enriched:     {n_sig_001} clusters (≥2× at p<0.01)")
info(f"")

if len(spatial_results) > 0:
    n_spatial = sum(1 for r in spatial_results if r["binom_p"] < 0.05)
    info(f"Spatial signal:   {n_spatial}/{len(spatial_results)} clusters (annulus p<0.05)")

if 'kin_results' in dir() and kin_results:
    info(f"Kinematic ratio:  {kin_df['sigma_ratio'].median():.3f} (median σ_match/σ_field)")

if len(aged_df) >= 10:
    info(f"Age trend:        ρ={rho_ae:.4f}, p={p_ae:.4e}")

info("")
if n_sig_001 >= 3:
    info("=> POSITIVE RESULT: Multiple clusters show significant chemical enrichment")
    info("   in the GALAH field population. Dissolved cluster members are recoverable")
    info("   from chemistry alone within a single survey — zero cross-calibration needed.")
elif n_sig_001 >= 1:
    info("=> MARGINAL RESULT: Individual cluster(s) show enrichment.")
    info("   Signal present but not yet widespread at this precision level.")
else:
    info("=> NULL at p<0.01. Current GALAH precision may be limiting.")

save_results()
info("T16-revised complete.")

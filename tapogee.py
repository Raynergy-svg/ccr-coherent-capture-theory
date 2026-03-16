#!/usr/bin/env python3
"""
T-APOGEE: Open Cluster Chemical Coherence Analysis using APOGEE DR17 + OCCAM VAC
====================================================================================
Adapts the T9 methodology (GALAH-based) to SDSS/APOGEE DR17 data with the
OCCAM (Open Cluster Chemical Abundances and Mapping) VAC member catalog.

Tests whether open clusters show age-dependent chemical coherence in C/O ratios,
as predicted by Coherent Capture Theory (CCT).

Author: Certan (2025)
Data:   APOGEE DR17 allStar + OCCAM member catalog (Donor et al.)
Ages:   Cantat-Gaudin & Anders (2020), VizieR J/A+A/640/A1
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.table import Table
from scipy.stats import kruskal, mannwhitneyu

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOLAR_CO             = 0.549           # Solar C/O ratio
SNR_MIN              = 50             # APOGEE typical SNR is much higher than GALAH
CO_COHERENCE_THRESH  = 0.05           # dex — coherence threshold
MIN_MEMBERS          = 3              # minimum quality members per cluster
OCCAM_PROB_THRESHOLD = 0.5            # membership probability cut
OCCAM_FILE           = "occam_member-DR17.fits"
ALLSTAR_CSV          = "apogee_occam_abundances.csv"
ALLSTAR_FITS         = "allStar-dr17-synspec_rev1.fits"
ALLSTAR_LITE_FITS    = "allStarLite-dr17-synspec_rev1.fits"
STAR_BAD_BIT         = 23             # ASPCAPFLAG bit 23 = STAR_BAD
AGE_SPLIT_GYR        = 0.5            # young/old boundary

# Output files
OUT_CLUSTER_STATS    = "tapogee_cluster_stats.csv"
OUT_MATCHED_STARS    = "tapogee_matched_stars.csv"
OUT_STATS_WITH_AGE   = "tapogee_cluster_stats_with_age.csv"
OUT_PLOT             = "tapogee_plot.png"

# T9 reference values (GALAH-based)
T9_CCR               = 0.68
T9_N_CLUSTERS        = 655
T9_COHERENT          = 34

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg):
    """Print with [INFO] prefix, matching T9 style."""
    print("[INFO] " + str(msg), flush=True)


def strip_col(series):
    """Strip whitespace from string columns (handles FITS byte strings)."""
    def _clean(x):
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace").strip()
        if isinstance(x, str):
            return x.strip()
        return str(x).strip()
    return series.apply(_clean)


def normalize_cluster_name(name):
    """Normalize cluster name for crossmatching.
    Handles: spaces vs underscores, case differences, common prefixes.
    """
    s = str(name).strip()
    # Decode bytes if needed
    if isinstance(name, bytes):
        s = name.decode("utf-8", errors="replace").strip()
    # Normalize whitespace/underscores
    s = s.replace("_", " ")
    # Collapse multiple spaces
    s = " ".join(s.split())
    # Uppercase for matching
    return s.upper()


def check_star_bad(flag_value):
    """Check if ASPCAPFLAG bit 23 (STAR_BAD) is set.
    Returns True if star is BAD (should be excluded).
    """
    try:
        return bool(int(flag_value) & (1 << STAR_BAD_BIT))
    except (ValueError, TypeError):
        return True  # exclude if flag is unreadable


# ---------------------------------------------------------------------------
# 1. Load OCCAM member catalog
# ---------------------------------------------------------------------------
log("=" * 70)
log("T-APOGEE: Open Cluster Chemical Coherence (APOGEE DR17 + OCCAM)")
log("=" * 70)

log("Loading OCCAM member catalog: " + OCCAM_FILE)
occam = Table.read(OCCAM_FILE).to_pandas()
log("Raw OCCAM rows: " + str(len(occam)))
log("Unique clusters in OCCAM: " + str(occam["CLUSTER"].nunique()))

# Strip string columns
for col in ["CLUSTER", "APOGEE_ID"]:
    if col in occam.columns:
        occam[col] = strip_col(occam[col])

# ---------------------------------------------------------------------------
# 2. Apply OCCAM membership probability cuts
#    Standard cut from Donor et al.: at least 2 of 3 probabilities > 0.5
# ---------------------------------------------------------------------------
log("Applying membership probability cuts (>=2 of 3 probs > " +
    str(OCCAM_PROB_THRESHOLD) + ")...")

prob_cols = ["RV_PROB", "FEH_PROB", "PM_PROB"]
# Count how many probabilities exceed threshold for each star
occam["n_probs_pass"] = 0
for pc in prob_cols:
    if pc in occam.columns:
        occam["n_probs_pass"] += (occam[pc] > OCCAM_PROB_THRESHOLD).astype(int)

occam_filtered = occam[occam["n_probs_pass"] >= 2].copy()
log("After membership cuts: " + str(len(occam_filtered)) + " stars in " +
    str(occam_filtered["CLUSTER"].nunique()) + " clusters")

# ---------------------------------------------------------------------------
# 3. Load allStar abundance data (CSV primary, FITS fallback)
# ---------------------------------------------------------------------------
allstar_cols = ["APOGEE_ID", "C_FE", "C_FE_ERR", "O_FE", "O_FE_ERR",
                "FE_H", "FE_H_ERR", "SNR", "TEFF", "LOGG", "ASPCAPFLAG",
                "VHELIO_AVG", "RA", "DEC", "MG_FE", "SI_FE", "AL_FE"]

allstar = None

# Mode 1: CSV (pre-fetched abundances for OCCAM stars)
if os.path.exists(ALLSTAR_CSV):
    log("Loading allStar abundances from CSV: " + ALLSTAR_CSV)
    allstar = pd.read_csv(ALLSTAR_CSV)
    log("CSV rows: " + str(len(allstar)))
    log("CSV columns: " + str(list(allstar.columns)))

    # Normalize column names — CSV may have mixed case
    col_map = {}
    for expected in allstar_cols:
        for actual in allstar.columns:
            if actual.upper() == expected.upper():
                col_map[actual] = expected
    if col_map:
        allstar = allstar.rename(columns=col_map)

    # Strip APOGEE_ID
    if "APOGEE_ID" in allstar.columns:
        allstar["APOGEE_ID"] = strip_col(allstar["APOGEE_ID"])

# Mode 2: FITS fallback (allStarLite preferred, then full allStar)
else:
    fits_file = None
    if os.path.exists(ALLSTAR_LITE_FITS):
        fits_file = ALLSTAR_LITE_FITS
    elif os.path.exists(ALLSTAR_FITS):
        fits_file = ALLSTAR_FITS

    if fits_file:
        log("CSV not found. Loading allStar abundances from FITS: " + fits_file)
        log("(This may take a while for large files...)")
        try:
            allstar_table = Table.read(fits_file)
            # Extract only needed columns to save memory
            keep = [c for c in allstar_cols if c in allstar_table.colnames]
            allstar = allstar_table[keep].to_pandas()
            log("allStar FITS rows: " + str(len(allstar)))

            # Strip APOGEE_ID
            if "APOGEE_ID" in allstar.columns:
                allstar["APOGEE_ID"] = strip_col(allstar["APOGEE_ID"])

            # Filter to only OCCAM stars to save memory
            occam_ids = set(occam_filtered["APOGEE_ID"].values)
            allstar = allstar[allstar["APOGEE_ID"].isin(occam_ids)].copy()
            log("After filtering to OCCAM stars: " + str(len(allstar)))
        except Exception as e:
            log("ERROR loading allStar FITS: " + str(e))
            allstar = None
    else:
        log("ERROR: No abundance data found. Need one of:")
    log("  1. " + ALLSTAR_CSV + " (pre-fetched CSV)")
    log("  2. " + ALLSTAR_LITE_FITS + " (recommended, ~1GB)")
    log("  3. " + ALLSTAR_FITS + " (full allStar, ~4GB)")
    log("")
    log("Required columns: " + ", ".join(allstar_cols))
    sys.exit(1)

if allstar is None or len(allstar) == 0:
    log("FATAL: No allStar abundance data available.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Merge OCCAM membership with allStar abundances
# ---------------------------------------------------------------------------
log("Merging OCCAM members with allStar abundances on APOGEE_ID...")
merged = occam_filtered.merge(allstar, on="APOGEE_ID", how="inner",
                               suffixes=("_occam", "_allstar"))
log("Merged rows: " + str(len(merged)))

if len(merged) == 0:
    log("FATAL: No matches between OCCAM and allStar on APOGEE_ID.")
    log("Check that APOGEE_ID formats match (e.g., '2M19204583+1541351').")
    sys.exit(1)

log("Unique clusters after merge: " + str(merged["CLUSTER"].nunique()))

# Resolve column name conflicts — prefer allStar values for abundances
for col in ["FE_H", "FE_H_ERR", "VHELIO_AVG"]:
    if col + "_allstar" in merged.columns:
        merged[col] = merged[col + "_allstar"]
    elif col + "_occam" in merged.columns:
        merged[col] = merged[col + "_occam"]

# ---------------------------------------------------------------------------
# 5. Quality cuts
# ---------------------------------------------------------------------------
log("Applying quality cuts...")
n_before = len(merged)

# 5a. ASPCAPFLAG bit 23 (STAR_BAD) — do NOT require flag==0
if "ASPCAPFLAG" in merged.columns:
    merged["star_bad"] = merged["ASPCAPFLAG"].apply(check_star_bad)
    n_bad = merged["star_bad"].sum()
    log("  STAR_BAD flagged: " + str(n_bad))
    merged = merged[~merged["star_bad"]].copy()
    log("  After STAR_BAD cut: " + str(len(merged)))
else:
    log("  WARNING: ASPCAPFLAG column not found, skipping STAR_BAD cut")

# 5b. SNR cut
if "SNR" in merged.columns:
    merged = merged[merged["SNR"] > SNR_MIN].copy()
    log("  After SNR > " + str(SNR_MIN) + " cut: " + str(len(merged)))

# 5c. Valid C_FE and O_FE (not NaN, not -9999)
for col in ["C_FE", "O_FE"]:
    if col in merged.columns:
        merged = merged[merged[col].notna() & (merged[col] > -9000)].copy()
log("  After valid C_FE/O_FE cut: " + str(len(merged)))

# 5d. Valid errors
for col in ["C_FE_ERR", "O_FE_ERR"]:
    if col in merged.columns:
        merged = merged[merged[col].notna() & (merged[col] > 0) &
                        (merged[col] < 9000)].copy()
log("  After valid error cut: " + str(len(merged)))

log("Quality cuts removed " + str(n_before - len(merged)) + " stars, " +
    str(len(merged)) + " remain")

if len(merged) == 0:
    log("FATAL: No stars survived quality cuts.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 6. Compute C/O ratio and uncertainty
# ---------------------------------------------------------------------------
log("Computing C/O ratios...")

# C/O = 10^([C/Fe] - [O/Fe]) * Solar_CO
merged["delta_co"] = merged["C_FE"] - merged["O_FE"]
merged["C_O"] = (10.0 ** merged["delta_co"]) * SOLAR_CO

# Error propagation: sigma_CO = C/O * ln(10) * sqrt(sigma_C^2 + sigma_O^2)
merged["sigma_CO"] = merged["C_O"] * np.log(10) * np.sqrt(
    merged["C_FE_ERR"]**2 + merged["O_FE_ERR"]**2)

# Filter unreasonable C/O values
n_pre_co = len(merged)
merged = merged[
    (merged["C_O"] > 0.05) & (merged["C_O"] < 2.0) &
    (merged["sigma_CO"] > 0) & (merged["sigma_CO"].notna())
].copy()
log("After C/O range filter (0.05-2.0): " + str(len(merged)) +
    " (removed " + str(n_pre_co - len(merged)) + ")")

if len(merged) == 0:
    log("FATAL: No stars with valid C/O ratios.")
    sys.exit(1)

log("C/O statistics:")
log("  Mean C/O:  " + str(round(merged["C_O"].mean(), 4)))
log("  Median C/O: " + str(round(merged["C_O"].median(), 4)))
log("  Std C/O:   " + str(round(merged["C_O"].std(), 4)))
log("  Mean sigma_CO: " + str(round(merged["sigma_CO"].mean(), 4)))

# ---------------------------------------------------------------------------
# 7. Fetch cluster ages from Cantat-Gaudin & Anders (2020) via VizieR
# ---------------------------------------------------------------------------
log("Fetching cluster ages from Cantat-Gaudin & Anders (2020) via VizieR...")
log("  Catalog: J/A+A/640/A1")

cg20 = None
try:
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    catalogs = Vizier.get_catalogs("J/A+A/640/A1")
    cg20_raw = catalogs[0].to_pandas()
    log("  CG20 raw clusters: " + str(len(cg20_raw)))
    log("  CG20 columns: " + str(list(cg20_raw.columns)))

    # Identify columns
    name_col = next((c for c in cg20_raw.columns
                     if c.lower() in ["cluster", "name"]), None)
    age_col = next((c for c in cg20_raw.columns
                    if "logage" in c.lower() or "log_age" in c.lower()
                    or c.lower() == "agenn"), None)

    if name_col and age_col:
        cg20 = cg20_raw[[name_col, age_col]].copy()
        cg20.columns = ["cg20_name", "logage"]
        cg20["cg20_name_norm"] = cg20["cg20_name"].apply(normalize_cluster_name)
        cg20["age_gyr"] = 10.0 ** (cg20["logage"].astype(float) - 9.0)
        log("  Age range: " + str(round(cg20["age_gyr"].min(), 3)) +
            " - " + str(round(cg20["age_gyr"].max(), 2)) + " Gyr")
        log("  CG20 clusters with ages: " + str(len(cg20)))
    else:
        log("  WARNING: Could not identify name/age columns in CG20")
        log("  Available columns: " + str(list(cg20_raw.columns)))

except Exception as e:
    log("  WARNING: VizieR query failed: " + str(e))
    log("  Will proceed without cluster ages.")

# ---------------------------------------------------------------------------
# 8. Match cluster names between OCCAM and CG20
# ---------------------------------------------------------------------------
if cg20 is not None:
    log("Crossmatching OCCAM cluster names with CG20...")

    # Normalize OCCAM cluster names
    occam_clusters = merged["CLUSTER"].unique()
    occam_norm = {c: normalize_cluster_name(c) for c in occam_clusters}

    # Build CG20 lookup
    cg20_lookup = {}
    for _, row in cg20.iterrows():
        cg20_lookup[row["cg20_name_norm"]] = {
            "age_gyr": row["age_gyr"],
            "logage": row["logage"],
            "cg20_name": row["cg20_name"]
        }

    # Match
    age_map = {}
    matched_names = []
    unmatched_names = []
    for oname, onorm in occam_norm.items():
        if onorm in cg20_lookup:
            age_map[oname] = cg20_lookup[onorm]
            matched_names.append(oname)
        else:
            unmatched_names.append(oname)

    log("  Matched: " + str(len(matched_names)) + "/" +
        str(len(occam_clusters)) + " clusters")
    if unmatched_names:
        log("  Unmatched: " + str(unmatched_names[:20]))
        if len(unmatched_names) > 20:
            log("  ... and " + str(len(unmatched_names) - 20) + " more")

    # Add age column to merged dataframe
    merged["age_gyr"] = merged["CLUSTER"].map(
        lambda c: age_map[c]["age_gyr"] if c in age_map else np.nan)
    merged["logage"] = merged["CLUSTER"].map(
        lambda c: age_map[c]["logage"] if c in age_map else np.nan)

    n_with_age = merged["age_gyr"].notna().sum()
    log("  Stars with age info: " + str(n_with_age) + "/" + str(len(merged)))
else:
    merged["age_gyr"] = np.nan
    merged["logage"] = np.nan

# ---------------------------------------------------------------------------
# 9. Per-cluster statistics (clusters with >= MIN_MEMBERS quality members)
# ---------------------------------------------------------------------------
log("Computing per-cluster statistics (min " + str(MIN_MEMBERS) + " members)...")

cl_stats = []
cluster_co_arrays = {}  # for Kruskal-Wallis test

for clname in merged["CLUSTER"].unique():
    grp = merged[merged["CLUSTER"] == clname]
    if len(grp) < MIN_MEMBERS:
        continue

    co_vals = grp["C_O"].values
    sig_vals = grp["sigma_CO"].values

    # Weighted statistics
    w = 1.0 / np.maximum(sig_vals**2, 1e-10)
    w_mean = np.sum(co_vals * w) / np.sum(w)
    w_std = np.sqrt(np.sum(w * (co_vals - w_mean)**2) / np.sum(w))

    # Intrinsic scatter: remove measurement noise contribution
    intrinsic = np.sqrt(max(0.0, w_std**2 - np.mean(sig_vals**2)))

    # RV statistics
    rv_vals = grp["VHELIO_AVG"].values if "VHELIO_AVG" in grp.columns else None
    rv_mean = float(np.nanmean(rv_vals)) if rv_vals is not None else np.nan
    rv_std = float(np.nanstd(rv_vals, ddof=1)) if (
        rv_vals is not None and len(rv_vals) > 1) else np.nan

    row = {
        "cluster":           clname,
        "N":                 len(grp),
        "C_O_mean":          round(float(w_mean), 4),
        "C_O_std":           round(float(w_std), 4),
        "intrinsic_scatter": round(float(intrinsic), 4),
        "mean_sigma_CO":     round(float(np.mean(sig_vals)), 4),
        "feh_mean":          round(float(grp["FE_H"].mean()), 4),
        "feh_std":           round(float(grp["FE_H"].std()), 4),
        "rv_mean":           round(rv_mean, 2) if not np.isnan(rv_mean) else np.nan,
        "rv_std":            round(rv_std, 2) if not np.isnan(rv_std) else np.nan,
        "snr_mean":          round(float(grp["SNR"].mean()), 1) if "SNR" in grp.columns else np.nan,
    }

    # Add age if available
    if "age_gyr" in grp.columns and grp["age_gyr"].notna().any():
        row["age_gyr"] = round(float(grp["age_gyr"].iloc[0]), 4)
        row["logage"] = round(float(grp["logage"].iloc[0]), 3)

    cl_stats.append(row)
    cluster_co_arrays[clname] = co_vals

cl_df = pd.DataFrame(cl_stats).sort_values("C_O_std").reset_index(drop=True)
log("Clusters with >= " + str(MIN_MEMBERS) + " quality members: " + str(len(cl_df)))

if len(cl_df) == 0:
    log("FATAL: No clusters with enough quality members.")
    log("Check data quality or lower MIN_MEMBERS threshold.")
    sys.exit(1)

log("Top 10 most coherent clusters:")
log(cl_df.head(10).to_string(index=False))

# Save cluster stats
cl_df.to_csv(OUT_CLUSTER_STATS, index=False)
log("Cluster stats saved: " + OUT_CLUSTER_STATS)

# ---------------------------------------------------------------------------
# 10. CCT Core Tests
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("CCT CORE TESTS")
log("=" * 70)

# 10a. CCR = between-cluster C/O std / within-cluster C/O std mean
within_scatter = cl_df["C_O_std"].mean()
between_scatter = cl_df["C_O_mean"].std()
CCR = between_scatter / within_scatter if within_scatter > 0 else 0.0

log("Within-cluster C/O scatter (mean of stds): " + str(round(within_scatter, 4)))
log("Between-cluster C/O scatter (std of means): " + str(round(between_scatter, 4)))
log("CCR = " + str(round(CCR, 4)) +
    ("  [SIGNAL: between > within]" if CCR > 1 else "  [NO SIGNAL: within >= between]"))

# 10b. KCR = between-cluster RV std / within-cluster RV std mean (kinematics analog)
rv_valid = cl_df[cl_df["rv_std"].notna() & (cl_df["rv_std"] > 0)]
if len(rv_valid) >= 2:
    rv_within = rv_valid["rv_std"].mean()
    rv_between = rv_valid["rv_mean"].std()
    KCR = rv_between / rv_within if rv_within > 0 else 0.0
    log("KCR (RV analog) = " + str(round(KCR, 4)) +
        ("  [SUPPORTS CCT]" if KCR > 1 else "  [MIXED]"))
else:
    KCR = np.nan
    log("KCR: insufficient RV data")

# 10c. Kruskal-Wallis test across cluster C/O distributions
if len(cluster_co_arrays) >= 2:
    groups_co = [arr for arr in cluster_co_arrays.values() if len(arr) >= 2]
    if len(groups_co) >= 2:
        kw_stat, kw_p = kruskal(*groups_co)
        log("Kruskal-Wallis H = " + str(round(kw_stat, 3)) +
            "  p = " + str(kw_p))
        if kw_p < 0.05:
            log("  -> Significant (p < 0.05): clusters have different C/O distributions")
        else:
            log("  -> Not significant: clusters have similar C/O distributions")

# 10d. Coherent clusters
n_coherent = int((cl_df["C_O_std"] < CO_COHERENCE_THRESH).sum())
log("Coherent clusters (C/O std < " + str(CO_COHERENCE_THRESH) + " dex): " +
    str(n_coherent) + "/" + str(len(cl_df)))
if n_coherent > 0:
    log("Coherent cluster details:")
    log(cl_df[cl_df["C_O_std"] < CO_COHERENCE_THRESH].to_string(index=False))

# 10e. Age-dependent analysis
if "age_gyr" in cl_df.columns and cl_df["age_gyr"].notna().sum() >= 2:
    young = cl_df[cl_df["age_gyr"] < AGE_SPLIT_GYR]
    old = cl_df[cl_df["age_gyr"] >= AGE_SPLIT_GYR]

    log("")
    log("--- Age-Dependent Analysis ---")
    log("Young clusters (< " + str(AGE_SPLIT_GYR) + " Gyr): N = " + str(len(young)))
    log("Old clusters (>= " + str(AGE_SPLIT_GYR) + " Gyr): N = " + str(len(old)))

    if len(young) > 0:
        log("  Young mean C/O std: " + str(round(young["C_O_std"].mean(), 4)))
        young_coherent = int((young["C_O_std"] < CO_COHERENCE_THRESH).sum())
        log("  Young coherent: " + str(young_coherent) + "/" + str(len(young)))
    if len(old) > 0:
        log("  Old mean C/O std:   " + str(round(old["C_O_std"].mean(), 4)))
        old_coherent = int((old["C_O_std"] < CO_COHERENCE_THRESH).sum())
        log("  Old coherent:   " + str(old_coherent) + "/" + str(len(old)))

    # Mann-Whitney U test: young vs old scatter
    if len(young) >= 2 and len(old) >= 2:
        mw_stat, mw_p = mannwhitneyu(young["C_O_std"].values,
                                      old["C_O_std"].values,
                                      alternative="less")
        log("  Mann-Whitney U (young < old scatter): U = " +
            str(round(mw_stat, 3)) + "  p = " + str(round(mw_p, 6)))
        if mw_p < 0.05:
            log("  -> Significant: young clusters have LOWER scatter")
            log("  -> CONSISTENT WITH CCT: younger clusters more chemically coherent")
        else:
            log("  -> Not significant at p < 0.05")

    if len(young) > 0 and len(old) > 0:
        if young["C_O_std"].mean() < old["C_O_std"].mean():
            log("  RESULT: Younger clusters more coherent -> CONSISTENT WITH CCT")
        else:
            log("  RESULT: No age-coherence trend in C/O scatter")

# ---------------------------------------------------------------------------
# 11. Save outputs
# ---------------------------------------------------------------------------
log("")
log("Saving outputs...")

# Save matched stars
star_cols = ["CLUSTER", "APOGEE_ID", "C_O", "sigma_CO", "C_FE", "O_FE",
             "FE_H", "FE_H_ERR", "SNR", "TEFF", "LOGG", "VHELIO_AVG",
             "RA", "DEC", "age_gyr", "logage"]
star_cols_present = [c for c in star_cols if c in merged.columns]
# Also include extra abundances if present
for extra in ["MG_FE", "SI_FE", "AL_FE", "RV_PROB", "FEH_PROB", "PM_PROB"]:
    if extra in merged.columns and extra not in star_cols_present:
        star_cols_present.append(extra)

merged[star_cols_present].to_csv(OUT_MATCHED_STARS, index=False)
log("Matched stars saved: " + OUT_MATCHED_STARS + " (" + str(len(merged)) + " rows)")

# Save cluster stats with age
if "age_gyr" in cl_df.columns:
    cl_df.to_csv(OUT_STATS_WITH_AGE, index=False)
    log("Cluster stats with age saved: " + OUT_STATS_WITH_AGE)

# ---------------------------------------------------------------------------
# 12. Generate 3-panel plot
# ---------------------------------------------------------------------------
log("Generating plot: " + OUT_PLOT)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.patch.set_facecolor("white")

# ---------- Panel 1: Bar chart of within-cluster C/O scatter (sorted) ----------
ax1 = axes[0]
bar_colors = ["steelblue" if v < CO_COHERENCE_THRESH else "salmon"
              for v in cl_df["C_O_std"].values]
x_pos = np.arange(len(cl_df))
ax1.bar(x_pos, cl_df["C_O_std"].values,
        color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.3)
ax1.axhline(CO_COHERENCE_THRESH, color="red", linestyle="--",
            linewidth=1.5, label="Coherence threshold (" +
            str(CO_COHERENCE_THRESH) + " dex)")
ax1.axhline(within_scatter, color="navy", linestyle=":",
            linewidth=1.5, label="Mean = " + str(round(within_scatter, 4)))
ax1.set_xlabel("Open Cluster (sorted by C/O scatter)", fontsize=10)
ax1.set_ylabel("Weighted C/O Std Dev", fontsize=10)
ax1.set_title("Within-Cluster C/O Scatter", fontsize=11, fontweight="bold")

# Only show tick labels if fewer clusters
if len(cl_df) <= 40:
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(cl_df["cluster"].values, rotation=90,
                        fontsize=max(4, 8 - len(cl_df) // 8))
else:
    ax1.set_xticks([])
    ax1.set_xlabel("Open Cluster (sorted, N=" + str(len(cl_df)) + ")", fontsize=10)

ax1.legend(fontsize=7, loc="upper left")
ax1.grid(True, alpha=0.15, axis="y")
ax1.set_xlim(-0.5, len(cl_df) - 0.5)

# ---------- Panel 2: [Fe/H] vs C/O colored by cluster ----------
ax2 = axes[1]
cmap = plt.cm.tab20
cl_names = cl_df["cluster"].values
legend_handles = []
for idx, clname in enumerate(cl_names):
    grp = merged[merged["CLUSTER"] == clname]
    color = cmap((idx % 20) / 20.0)
    ax2.scatter(grp["FE_H"], grp["C_O"],
                s=18, alpha=0.7, color=color,
                edgecolors="none", rasterized=True)
    if idx < 10:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=5, label=clname))

ax2.axhline(SOLAR_CO, color="darkorange", linestyle=":", linewidth=1.5,
            label="Solar C/O = " + str(SOLAR_CO))
legend_handles.append(
    Line2D([0], [0], color="darkorange", linestyle=":", linewidth=1.5,
           label="Solar C/O"))
ax2.set_xlabel("[Fe/H]", fontsize=10)
ax2.set_ylabel("C/O", fontsize=10)
ax2.set_title("[Fe/H] vs C/O by Open Cluster", fontsize=11, fontweight="bold")
if len(legend_handles) <= 12:
    ax2.legend(handles=legend_handles, fontsize=6, ncol=2, loc="best")
ax2.grid(True, alpha=0.15)

# ---------- Panel 3: Age vs C/O scatter ----------
ax3 = axes[2]
if "age_gyr" in cl_df.columns and cl_df["age_gyr"].notna().sum() >= 3:
    age_valid = cl_df[cl_df["age_gyr"].notna()].copy()
    sc = ax3.scatter(age_valid["age_gyr"], age_valid["C_O_std"],
                     c=age_valid["feh_mean"], cmap="coolwarm",
                     s=age_valid["N"].values * 5, alpha=0.75,
                     edgecolors="white", linewidth=0.5, zorder=3)
    cbar = plt.colorbar(sc, ax=ax3, label="Mean [Fe/H]", pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    ax3.axhline(CO_COHERENCE_THRESH, color="red", linestyle="--",
                linewidth=1.5, label="Threshold " + str(CO_COHERENCE_THRESH))

    # Trend line
    if len(age_valid) >= 3:
        z = np.polyfit(age_valid["age_gyr"].values,
                       age_valid["C_O_std"].values, 1)
        xfit = np.linspace(age_valid["age_gyr"].min(),
                           age_valid["age_gyr"].max(), 100)
        ax3.plot(xfit, np.polyval(z, xfit), "k--", linewidth=1.2, alpha=0.6,
                 label="slope = " + str(round(z[0], 5)))

    ax3.set_xlabel("Cluster Age (Gyr)", fontsize=10)
    ax3.set_ylabel("Weighted C/O Std Dev", fontsize=10)
    ax3.set_title("Age vs C/O Scatter\n(CCT: younger = more coherent?)",
                  fontsize=11, fontweight="bold")
    ax3.legend(fontsize=7, loc="best")
    ax3.grid(True, alpha=0.15)
else:
    ax3.text(0.5, 0.5, "Insufficient age data\n(need VizieR CG20 match)",
             ha="center", va="center", transform=ax3.transAxes,
             fontsize=12, color="gray", style="italic")
    ax3.set_title("Age vs C/O Scatter", fontsize=11, fontweight="bold")

# Suptitle
plt.suptitle(
    "T-APOGEE Open Cluster Chemical Coherence | Certan (2025) | OCCAM DR17\n"
    "CCR=" + str(round(CCR, 4)) +
    "  |  N_clusters=" + str(len(cl_df)) +
    "  |  Coherent=" + str(n_coherent) + "/" + str(len(cl_df)) +
    "  |  Within scatter=" + str(round(within_scatter, 4)) +
    "  |  SNR>" + str(SNR_MIN),
    fontsize=9.5, style="italic", color="gray", y=1.02
)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log("Plot saved: " + OUT_PLOT)

# ---------------------------------------------------------------------------
# 13. Final Summary
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("T-APOGEE SUMMARY")
log("=" * 70)
log("Survey:          APOGEE DR17 + OCCAM VAC")
log("Stars analyzed:  " + str(len(merged)))
log("Clusters (>=" + str(MIN_MEMBERS) + " members): " + str(len(cl_df)))
log("Total members:   " + str(int(cl_df["N"].sum())))
log("CCR:             " + str(round(CCR, 4)) +
    ("  [SIGNAL]" if CCR > 1 else "  [NO SIGNAL]"))
if not np.isnan(KCR):
    log("KCR:             " + str(round(KCR, 4)) +
        ("  [SUPPORTS CCT]" if KCR > 1 else "  [MIXED]"))
log("Coherent (<" + str(CO_COHERENCE_THRESH) + " dex): " +
    str(n_coherent) + "/" + str(len(cl_df)) +
    " (" + str(round(100.0 * n_coherent / len(cl_df), 1)) + "%)")
log("Within scatter:  " + str(round(within_scatter, 4)))
log("Between scatter: " + str(round(between_scatter, 4)))

if "age_gyr" in cl_df.columns and cl_df["age_gyr"].notna().sum() >= 2:
    n_with_age = int(cl_df["age_gyr"].notna().sum())
    log("Clusters with ages: " + str(n_with_age))

log("")
log("--- Comparison with T9 (GALAH) ---")
log("T9 (GALAH):    CCR=" + str(T9_CCR) +
    "  N_clusters=" + str(T9_N_CLUSTERS) +
    "  Coherent=" + str(T9_COHERENT) + "/" + str(T9_N_CLUSTERS))
log("T-APOGEE:      CCR=" + str(round(CCR, 4)) +
    "  N_clusters=" + str(len(cl_df)) +
    "  Coherent=" + str(n_coherent) + "/" + str(len(cl_df)))

if CCR > 0:
    ratio = CCR / T9_CCR
    log("CCR ratio (T-APOGEE / T9): " + str(round(ratio, 3)))
    if ratio > 0.8 and ratio < 1.2:
        log("  -> CCR values are CONSISTENT between surveys")
    elif ratio >= 1.2:
        log("  -> T-APOGEE shows STRONGER between-cluster signal")
    else:
        log("  -> T-APOGEE shows WEAKER between-cluster signal")

log("")
log("Output files:")
log("  " + OUT_CLUSTER_STATS)
log("  " + OUT_MATCHED_STARS)
log("  " + OUT_STATS_WITH_AGE)
log("  " + OUT_PLOT)
log("")
log("DONE")

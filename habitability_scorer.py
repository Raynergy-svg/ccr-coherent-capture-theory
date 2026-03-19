#!/usr/bin/env python3
"""
Stellar Habitability Chemistry Scorer
=======================================
Certan (2026) | Built on Coherent Capture Theory

Scores GALAH DR4 field stars for rocky planet habitability potential
based on host star chemistry. Uses theoretical constraints from
planetary science anchored by CCT birth environment reconstruction.

Chemistry dimensions scored:
  1. C/O ratio — silicate vs carbide mineralogy (Bond+2010)
  2. Mg/Si ratio — mantle composition, plate tectonics (Unterborn+2016)
  3. [Fe/H] — iron core mass, magnetic dynamo
  4. [Mg/Fe] — mantle-to-core ratio
  5. [Si/Fe] — silicate budget
  6. [Ca/Fe] — calcium for biological processes
  7. [Al/Fe] — radiogenic heating (26Al)

Output: ranked target list with sky coordinates, chemistry scores,
and CCT birth environment classification.
"""

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

SOLAR_CO = 0.549
GALAH_FITS = "galah_dr4_allstar_240705.fits"
T9_STARS = "t9_matched_stars.csv"
T9_CLUSTERS = "t9_cluster_stats_with_age.csv"

# ---------------------------------------------------------------------------
# Habitability chemistry windows (theoretical)
# ---------------------------------------------------------------------------
# Each returns a score 0-1: 1 = optimal, 0 = incompatible

def score_co(co):
    """C/O ratio score. Earth C/O ~ 0.55. Below 0.8 = silicate worlds.
    Bond et al. (2010), Delgado Mena et al. (2010)."""
    # Optimal: 0.3-0.7 (Earth-like silicate mineralogy)
    # Acceptable: 0.1-0.8
    # Above 0.8: carbide planets, no liquid water chemistry
    score = np.ones_like(co, dtype=float)
    # Penalty above 0.8 — hard cutoff
    score[co > 0.8] = 0.0
    # Penalty above 0.7 — transition zone
    mask = (co > 0.7) & (co <= 0.8)
    score[mask] = 1.0 - (co[mask] - 0.7) / 0.1
    # Slight penalty below 0.2 (very oxygen-rich, unusual)
    mask_lo = co < 0.2
    score[mask_lo] = co[mask_lo] / 0.2
    # Peak at 0.4-0.6 (Earth-like)
    optimal = (co >= 0.4) & (co <= 0.6)
    score[optimal] = 1.0
    return np.clip(score, 0, 1)

def score_mgsi(mgsi):
    """Mg/Si ratio score. Earth Mg/Si ~ 1.02.
    Controls mantle mineralogy and plate tectonics.
    Unterborn et al. (2016), Dorn et al. (2015)."""
    # Optimal: 0.8-1.5 (olivine-dominated mantle, plate tectonics likely)
    # Below 0.8: pyroxene-dominated, stagnant lid likely
    # Above 2.0: periclase mantle, no plate tectonics
    score = np.exp(-0.5 * ((mgsi - 1.02) / 0.4)**2)
    return np.clip(score, 0, 1)

def score_feh(feh):
    """[Fe/H] score. Solar [Fe/H] = 0.
    Need sufficient iron for differentiated core + dynamo.
    Lineweaver (2001), Gonzalez (2005)."""
    # Optimal: -0.2 to +0.3 (enough iron, not so much it dominates)
    # Below -0.5: insufficient iron for Earth-mass core
    # Above +0.5: very iron-rich, potentially hostile
    score = np.exp(-0.5 * ((feh - 0.0) / 0.3)**2)
    # Hard penalty below -0.5
    score[feh < -0.5] *= np.exp(-((feh[feh < -0.5] + 0.5) / 0.2)**2)
    return np.clip(score, 0, 1)

def score_mgfe(mgfe):
    """[Mg/Fe] score. Earth host ~ solar.
    Determines mantle-to-core mass ratio."""
    # Optimal: near solar (0.0 ± 0.1)
    # Alpha-enhanced (>0.2): too much mantle relative to core
    score = np.exp(-0.5 * (mgfe / 0.15)**2)
    return np.clip(score, 0, 1)

def score_sife(sife):
    """[Si/Fe] score. Silicate budget for rocky planet formation."""
    # Optimal: near solar
    score = np.exp(-0.5 * (sife / 0.15)**2)
    return np.clip(score, 0, 1)

def score_cafe(cafe):
    """[Ca/Fe] score. Calcium for biological processes, crustal composition."""
    # Moderately important — near solar preferred
    score = np.exp(-0.5 * (cafe / 0.20)**2)
    return np.clip(score, 0, 1)

def score_alfe(alfe):
    """[Al/Fe] score. 26Al radiogenic heating drives planetary differentiation.
    Lichtenberg et al. (2019)."""
    # Need some Al for radiogenic heating, not too much
    # Optimal: near solar to slightly enhanced
    score = np.exp(-0.5 * ((alfe - 0.05) / 0.15)**2)
    return np.clip(score, 0, 1)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=" * 72)
print("Stellar Habitability Chemistry Scorer")
print("Certan (2026) | Built on CCT | GALAH DR4")
print("=" * 72)

# 1. Load GALAH
print("\nLoading GALAH DR4...")
galah_table = Table.read(GALAH_FITS, memmap=True)
cols = ["ra", "dec", "gaiadr3_source_id", "sobject_id",
        "snr_px_ccd3", "flag_sp", "teff", "logg",
        "c_fe", "o_fe", "flag_c_fe", "flag_o_fe",
        "mg_fe", "flag_mg_fe", "si_fe", "flag_si_fe",
        "fe_h", "flag_fe_h", "al_fe", "flag_al_fe",
        "ca_fe", "flag_ca_fe", "ba_fe", "flag_ba_fe",
        "parallax", "parallax_error", "rv_comp_1"]
galah = galah_table[[c for c in cols if c in galah_table.colnames]].to_pandas()
print(f"Total: {len(galah)}")

# Quality cuts
galah = galah[galah["snr_px_ccd3"] > 30].copy()
galah = galah[galah["flag_sp"] == 0].copy()

# Require core abundances
for col, fc in [("c_fe", "flag_c_fe"), ("o_fe", "flag_o_fe"),
                ("mg_fe", "flag_mg_fe"), ("si_fe", "flag_si_fe"),
                ("fe_h", "flag_fe_h")]:
    galah = galah[galah[col].notna()].copy()
    if fc in galah.columns:
        galah = galah[galah[fc] == 0].copy()

# Compute derived ratios
galah["C_O"] = (10.0 ** (galah["c_fe"] - galah["o_fe"])) * SOLAR_CO
galah["Mg_Si"] = 10.0 ** (galah["mg_fe"] - galah["si_fe"])  # relative to solar Mg/Si
galah = galah[(galah["C_O"] > 0.01) & (galah["C_O"] < 3.0)].copy()
galah = galah.reset_index(drop=True)
print(f"After quality cuts: {len(galah)}")

# 2. FGK dwarf filter — habitable zone exists for these stars
print("\nFiltering for FGK dwarfs (potential habitable zone hosts)...")
fgk_mask = (galah["teff"].notna() & galah["logg"].notna() &
            (galah["teff"] > 4000) & (galah["teff"] < 7000) &
            (galah["logg"] > 3.8))
galah_fgk = galah[fgk_mask].copy()
print(f"FGK dwarfs: {len(galah_fgk)}")

# 3. Compute habitability scores
print("\nComputing habitability scores...")

co_vals = galah_fgk["C_O"].values
mgsi_vals = galah_fgk["Mg_Si"].values
feh_vals = galah_fgk["fe_h"].values
mgfe_vals = galah_fgk["mg_fe"].values
sife_vals = galah_fgk["si_fe"].values

s_co = score_co(co_vals)
s_mgsi = score_mgsi(mgsi_vals)
s_feh = score_feh(feh_vals)
s_mgfe = score_mgfe(mgfe_vals)
s_sife = score_sife(sife_vals)

# Optional scores (if available)
if "ca_fe" in galah_fgk.columns:
    cafe_valid = galah_fgk["ca_fe"].notna()
    s_cafe = np.where(cafe_valid, score_cafe(galah_fgk["ca_fe"].fillna(0).values), 0.8)
else:
    s_cafe = np.full(len(galah_fgk), 0.8)

if "al_fe" in galah_fgk.columns:
    alfe_valid = galah_fgk["al_fe"].notna()
    s_alfe = np.where(alfe_valid, score_alfe(galah_fgk["al_fe"].fillna(0).values), 0.8)
else:
    s_alfe = np.full(len(galah_fgk), 0.8)

# Composite score — weighted geometric mean
# C/O and Mg/Si are most critical (weight 2)
# Fe/H is important (weight 1.5)
# Others weight 1
weights = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.5, 0.5])
scores = np.column_stack([s_co, s_mgsi, s_feh, s_mgfe, s_sife, s_cafe, s_alfe])

# Weighted geometric mean
log_scores = np.log(np.maximum(scores, 1e-10))
composite = np.exp(np.average(log_scores, weights=weights, axis=1))

galah_fgk["hab_score"] = composite
galah_fgk["s_CO"] = s_co
galah_fgk["s_MgSi"] = s_mgsi
galah_fgk["s_FeH"] = s_feh
galah_fgk["s_MgFe"] = s_mgfe
galah_fgk["s_SiFe"] = s_sife

# 4. CCT birth environment classification
print("\nClassifying birth environments (CCT cluster matching)...")
t9_stats = pd.read_csv(T9_CLUSTERS)
t9_stars = pd.read_csv(T9_STARS)

# Build cluster templates
templates = {}
for _, row in t9_stats.iterrows():
    cname = row["cluster"]
    grp = t9_stars[t9_stars["cluster_name"] == cname].dropna(subset=["C_O", "mg_fe", "si_fe", "fe_h"])
    if len(grp) < 5:
        continue
    templates[cname] = {
        "centroid": grp[["C_O", "mg_fe", "si_fe", "fe_h"]].mean().values,
        "age": row["age_gyr"] if pd.notna(row["age_gyr"]) else np.nan,
        "N": len(grp),
    }

# For each FGK star, find closest cluster template
print(f"Matching against {len(templates)} cluster templates...")
field_matrix = galah_fgk[["C_O", "mg_fe", "si_fe", "fe_h"]].values
tol = np.array([0.08, 0.05, 0.05, 0.10])

best_cluster = []
best_dist = []
for i in range(len(galah_fgk)):
    star = field_matrix[i]
    min_d = np.inf
    min_c = "field"
    for cname, t in templates.items():
        delta = np.abs(star - t["centroid"]) / tol
        d = np.sqrt(np.sum(delta**2))
        if d < min_d:
            min_d = d
            min_c = cname
    best_cluster.append(min_c if min_d < 4.0 else "field")  # 4σ threshold
    best_dist.append(min_d)

galah_fgk["cct_cluster"] = best_cluster
galah_fgk["cct_distance"] = best_dist
n_matched = (galah_fgk["cct_cluster"] != "field").sum()
print(f"Stars with CCT cluster assignment: {n_matched}/{len(galah_fgk)}")

# 5. Distance and sky coordinates
print("\nComputing distances and galactic coordinates...")
good_plx = galah_fgk["parallax"] > 0.5  # >0.5 mas = within 2 kpc
galah_fgk.loc[good_plx, "dist_pc"] = 1000.0 / galah_fgk.loc[good_plx, "parallax"]

coords = SkyCoord(ra=galah_fgk["ra"].values * u.deg,
                   dec=galah_fgk["dec"].values * u.deg, frame="icrs")
galah_fgk["gal_l"] = coords.galactic.l.deg
galah_fgk["gal_b"] = coords.galactic.b.deg

# 6. Rank and output
print("\nRanking targets...")
ranked = galah_fgk.sort_values("hab_score", ascending=False).reset_index(drop=True)

# Summary statistics
print(f"\n{'='*72}")
print(f"HABITABILITY SCORING SUMMARY")
print(f"{'='*72}")
print(f"Total FGK dwarfs scored: {len(ranked)}")
print(f"Score distribution:")
for threshold, label in [(0.9, "Excellent (>0.9)"), (0.8, "Very Good (0.8-0.9)"),
                           (0.7, "Good (0.7-0.8)"), (0.5, "Moderate (0.5-0.7)"),
                           (0.0, "Poor (<0.5)")]:
    if threshold == 0.0:
        n = (ranked["hab_score"] < 0.5).sum()
    elif threshold == 0.5:
        n = ((ranked["hab_score"] >= 0.5) & (ranked["hab_score"] < 0.7)).sum()
    else:
        lo = threshold - 0.0001 if threshold == 0.9 else threshold
        hi = threshold + 0.1 if threshold < 0.9 else 1.01
        n = ((ranked["hab_score"] >= lo) & (ranked["hab_score"] < hi)).sum()
    print(f"  {label}: {n} ({n/len(ranked):.1%})")

# Top 25 candidates
print(f"\nTop 25 Habitability Candidates:")
print(f"{'Rank':>4} {'Gaia ID':>22} {'RA':>8} {'DEC':>8} {'Score':>6} "
      f"{'C/O':>5} {'Mg/Si':>5} {'[Fe/H]':>7} {'Teff':>5} {'Dist':>6} {'Cluster':>15}")
for i, (_, row) in enumerate(ranked.head(25).iterrows()):
    dist = f"{row['dist_pc']:.0f}" if pd.notna(row.get("dist_pc")) else "--"
    cl = row["cct_cluster"][:15] if row["cct_cluster"] != "field" else "field"
    print(f"{i+1:>4} {int(row['gaiadr3_source_id']):>22} {row['ra']:>8.3f} {row['dec']:>8.3f} "
          f"{row['hab_score']:>6.4f} {row['C_O']:>5.3f} {row['Mg_Si']:>5.2f} "
          f"{row['fe_h']:>+7.3f} {row['teff']:>5.0f} {dist:>6} {cl:>15}")

# 7. Save outputs
print("\nSaving outputs...")
save_cols = ["gaiadr3_source_id", "ra", "dec", "gal_l", "gal_b",
             "hab_score", "s_CO", "s_MgSi", "s_FeH", "s_MgFe", "s_SiFe",
             "C_O", "Mg_Si", "fe_h", "mg_fe", "si_fe", "al_fe", "ca_fe",
             "teff", "logg", "dist_pc", "parallax", "rv_comp_1",
             "cct_cluster", "cct_distance"]
save_cols = [c for c in save_cols if c in ranked.columns]
ranked[save_cols].to_csv("habitability_targets.csv", index=False)
print(f"Saved: habitability_targets.csv ({len(ranked)} stars)")

# Top 1000 for quick access
ranked.head(1000)[save_cols].to_csv("habitability_top1000.csv", index=False)
print(f"Saved: habitability_top1000.csv")

# 8. Generate sky map
print("\nGenerating plots...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Stellar Habitability Chemistry Map | Certan (2026)\n"
             "GALAH DR4 FGK Dwarfs Scored for Rocky Planet Potential",
             fontsize=14, fontweight="bold", y=0.99)

# P1: Aitoff sky projection — hab score
ax = axes[0, 0]
ax = fig.add_subplot(221, projection="aitoff")
l_rad = np.deg2rad(ranked["gal_l"].values - 180)  # center on GC
b_rad = np.deg2rad(ranked["gal_b"].values)
# Subsample for plotting
rng = np.random.default_rng(42)
n_plot = min(20000, len(ranked))
idx = rng.choice(len(ranked), n_plot, replace=False)
sc = ax.scatter(l_rad[idx], b_rad[idx], c=ranked["hab_score"].values[idx],
                cmap="RdYlGn", s=1, alpha=0.5, vmin=0.3, vmax=1.0, rasterized=True)
plt.colorbar(sc, ax=ax, label="Habitability Score", shrink=0.7, pad=0.08)
ax.set_title("Sky Map: Habitability Score", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.3)

# P2: C/O vs Mg/Si colored by score
ax2 = axes[0, 1]
sub = ranked.iloc[idx]
sc2 = ax2.scatter(sub["C_O"], sub["Mg_Si"], c=sub["hab_score"],
                   cmap="RdYlGn", s=3, alpha=0.4, vmin=0.3, vmax=1.0, rasterized=True)
plt.colorbar(sc2, ax=ax2, label="Hab Score", shrink=0.8)
ax2.axvline(0.8, color="red", ls="--", lw=1.5, label="C/O = 0.8 (carbide limit)")
ax2.axhline(1.02, color="blue", ls=":", lw=1, label="Mg/Si = 1.02 (Earth)")
ax2.scatter([SOLAR_CO], [1.0], s=200, c="gold", edgecolors="black",
            marker="*", zorder=10, label="Sun")
ax2.set_xlabel("C/O ratio", fontsize=11)
ax2.set_ylabel("Mg/Si ratio (relative to solar)", fontsize=11)
ax2.set_title("Habitability Chemistry Space", fontsize=11, fontweight="bold")
ax2.set_xlim(0, 1.5)
ax2.set_ylim(0.3, 2.5)
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.15)

# P3: Score distribution
ax3 = axes[1, 0]
ax3.hist(ranked["hab_score"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
ax3.axvline(ranked["hab_score"].median(), color="navy", lw=2,
            label=f"Median = {ranked['hab_score'].median():.3f}")
ax3.axvline(0.9, color="red", ls="--", lw=1.5, label="Excellent threshold")
ax3.set_xlabel("Habitability Score", fontsize=11)
ax3.set_ylabel("Count", fontsize=11)
ax3.set_title("Score Distribution", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9)

# P4: [Fe/H] vs score
ax4 = axes[1, 1]
ax4.scatter(sub["fe_h"], sub["hab_score"], s=3, c="steelblue", alpha=0.3, rasterized=True)
ax4.axhline(0.9, color="red", ls="--", lw=1, alpha=0.5)
ax4.axvline(0.0, color="orange", ls=":", lw=1, alpha=0.5, label="Solar [Fe/H]")
ax4.set_xlabel("[Fe/H]", fontsize=11)
ax4.set_ylabel("Habitability Score", fontsize=11)
ax4.set_title("[Fe/H] vs Habitability", fontsize=11, fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.15)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("habitability_map.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: habitability_map.png")

print(f"\n{'='*72}")
print(f"COMPLETE")
print(f"{'='*72}")
print(f"Scored {len(ranked)} FGK dwarfs")
print(f"Top score: {ranked['hab_score'].max():.4f}")
print(f"Stars with score > 0.9: {(ranked['hab_score'] > 0.9).sum()}")
print(f"Stars with CCT cluster match: {n_matched}")
print(f"\nThe target list is ready for cross-referencing with TESS/Kepler")
print(f"planet search results and future transit survey planning.")

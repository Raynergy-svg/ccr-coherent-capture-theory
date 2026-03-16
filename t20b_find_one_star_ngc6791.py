#!/usr/bin/env python3
"""
T20b — Find One Star: NGC 6791
=================================
Certan (2026) | CCT | GALAH DR4 × Gaia DR3

NGC 6791 is ideal because:
  - [Fe/H] = +0.30 — among the most metal-rich clusters known
  - Only ~0.5% of GALAH stars have [Fe/H] > +0.25
  - 67 APOGEE members define a precise chemical template
  - Age ~8 Gyr — maximally dissolved, testing permanence at extreme age
  - Chemistry so distinctive that false positive rate is near zero

Uses APOGEE template with GALAH zero-point correction from T16.
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

SOLAR_CO = 0.549

# NGC 6791 properties (APOGEE + literature + Gaia DR3)
NGC6791_RA    = 290.22
NGC6791_DEC   = 37.77
NGC6791_PLX   = 0.243   # mas (~4.1 kpc)
NGC6791_PMRA  = -0.42   # mas/yr (Gaia DR3)
NGC6791_PMDEC = -2.28   # mas/yr (Gaia DR3)
NGC6791_RV    = -47.0   # km/s

# APOGEE template (from tapogee_matched_stars.csv)
# Apply APOGEE→GALAH offsets from T16:
# [C/Fe]: +0.0699, [O/Fe]: +0.1048, [Mg/Fe]: +0.0733
# [Si/Fe]: +0.0700, [Fe/H]: -0.0902
APOGEE_OFFSETS = {"c_fe": 0.0699, "o_fe": 0.1048, "mg_fe": 0.0733,
                  "si_fe": 0.0700, "fe_h": -0.0902}

EXCLUSION_RADIUS = 1.0  # degrees around NGC 6791

out = []
def info(msg):
    line = "[INFO] " + str(msg)
    print(line, flush=True)
    out.append(line)

info("=" * 72)
info("T20b  Find One Star: NGC 6791")
info("Certan (2026) | CCT | GALAH DR4 × Gaia DR3")
info("=" * 72)
info("[Fe/H]=+0.30 | Age=8 Gyr | The most metal-rich old cluster")

# 1. Build template from APOGEE
apogee = pd.read_csv("tapogee_matched_stars.csv")
n6791 = apogee[apogee["CLUSTER"] == "NGC 6791"].copy()
info(f"\nAPOGEE members: {len(n6791)}")

# Template in APOGEE space
template_apogee = {
    "c_fe": n6791["C_FE"].mean(),
    "o_fe": n6791["O_FE"].mean(),
    "mg_fe": n6791["MG_FE"].mean(),
    "si_fe": n6791["SI_FE"].mean(),
    "fe_h": n6791["FE_H"].mean(),
}
info("APOGEE template:")
for k, v in template_apogee.items():
    info(f"  {k}: {v:+.4f}")

# Convert to GALAH scale
template_galah = {}
for k in template_apogee:
    template_galah[k] = template_apogee[k] + APOGEE_OFFSETS[k]

info("GALAH-calibrated template:")
for k, v in template_galah.items():
    info(f"  {k}: {v:+.4f}")

# Compute C/O in GALAH scale
template_CO = (10.0 ** (template_galah["c_fe"] - template_galah["o_fe"])) * SOLAR_CO
info(f"  C/O: {template_CO:.4f}")

# Internal scatter (from APOGEE, generous)
scatter = {
    "C_O": max(n6791["C_O"].std(), 0.04),
    "mg_fe": max(n6791["MG_FE"].std(), 0.02),
    "si_fe": max(n6791["SI_FE"].std(), 0.02),
    "fe_h": max(n6791["FE_H"].std(), 0.05),
}
info(f"Template scatter: {scatter}")

# Matching tolerances — TIGHT because chemistry is distinctive
# Use 2× scatter or minimum, whichever is larger
CHEM_TOL = {
    "C_O":   max(2 * scatter["C_O"], 0.08),
    "mg_fe": max(2 * scatter["mg_fe"], 0.05),
    "si_fe": max(2 * scatter["si_fe"], 0.05),
    "fe_h":  max(2 * scatter["fe_h"], 0.08),  # TIGHT — this is the killer filter
}
info(f"Matching tolerances: {CHEM_TOL}")

# 2. Load GALAH
info("\nLoading GALAH DR4...")
galah_table = Table.read("galah_dr4_allstar_240705.fits", memmap=True)
cols = ["ra", "dec", "snr_px_ccd3", "flag_sp",
        "c_fe", "o_fe", "flag_c_fe", "flag_o_fe",
        "mg_fe", "flag_mg_fe", "si_fe", "flag_si_fe",
        "fe_h", "flag_fe_h",
        "rv_comp_1", "e_rv_comp_1",
        "parallax", "parallax_error",
        "gaiadr3_source_id",
        "ba_fe", "flag_ba_fe",
        "teff", "logg"]
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

# Exclude NGC 6791 region
sep = np.sqrt((galah["ra"] - NGC6791_RA)**2 * np.cos(np.deg2rad(NGC6791_DEC))**2 +
              (galah["dec"] - NGC6791_DEC)**2)
galah = galah[sep > EXCLUSION_RADIUS].copy().reset_index(drop=True)
info(f"Field stars: {len(galah)}")

# 3. Chemical matching
info("\n" + "-" * 60)
info("STEP 1: Chemical matching")

# First check: how rare is [Fe/H] > 0.2 in GALAH?
n_metal_rich = (galah["fe_h"] > 0.2).sum()
info(f"Stars with [Fe/H] > 0.2: {n_metal_rich}/{len(galah)} ({n_metal_rich/len(galah):.2%})")

centroid = np.array([template_CO, template_galah["mg_fe"],
                     template_galah["si_fe"], template_galah["fe_h"]])
tol = np.array([CHEM_TOL["C_O"], CHEM_TOL["mg_fe"],
                CHEM_TOL["si_fe"], CHEM_TOL["fe_h"]])

DIM_COLS = ["C_O", "mg_fe", "si_fe", "fe_h"]
field_matrix = galah[DIM_COLS].values
delta = np.abs(field_matrix - centroid)
chem_match = (delta < tol).all(axis=1)
n_chem = chem_match.sum()
info(f"Chemical matches: {n_chem}")

if n_chem == 0:
    info("No chemical matches — template may be too offset. Trying broader tolerance...")
    CHEM_TOL_BROAD = {"C_O": 0.12, "mg_fe": 0.08, "si_fe": 0.08, "fe_h": 0.12}
    tol_broad = np.array([CHEM_TOL_BROAD[k] for k in ["C_O", "mg_fe", "si_fe", "fe_h"]])
    chem_match = (np.abs(field_matrix - centroid) < tol_broad).all(axis=1)
    n_chem = chem_match.sum()
    info(f"Broad chemical matches: {n_chem}")

chem_cands = galah[chem_match].copy()

if n_chem > 0:
    info(f"  [Fe/H] range: {chem_cands['fe_h'].min():.3f} to {chem_cands['fe_h'].max():.3f}")
    info(f"  C/O range: {chem_cands['C_O'].min():.3f} to {chem_cands['C_O'].max():.3f}")

    # 4. Distance filter
    info("\n" + "-" * 60)
    info("STEP 2: Distance filter")
    info(f"NGC 6791 parallax: {NGC6791_PLX:.3f} mas ({1/NGC6791_PLX:.0f} pc)")

    # At 4.1 kpc, Gaia parallax errors are ~0.02-0.05 mas
    # NGC 6791 parallax is 0.243 mas
    # Use generous cut: parallax 0.05 to 0.5 mas (2-20 kpc — very broad)
    # Then tighten to ±0.15 mas of cluster parallax
    plx_valid = chem_cands["parallax"].notna() & (chem_cands["parallax"] > 0)
    chem_plx = chem_cands[plx_valid].copy()
    info(f"With valid parallax: {len(chem_plx)}")

    # Parallax consistent within generous window (3σ or 0.15 mas)
    plx_err = np.maximum(chem_plx["parallax_error"].values, 0.05)
    plx_offset = np.abs(chem_plx["parallax"].values - NGC6791_PLX)
    plx_match = plx_offset < np.maximum(3 * plx_err, 0.15)
    plx_cands = chem_plx[plx_match].copy()
    info(f"Parallax-consistent: {len(plx_cands)}")

    if len(plx_cands) > 0:
        info(f"  Parallax range: {plx_cands['parallax'].min():.4f} - {plx_cands['parallax'].max():.4f}")

    # 5. RV filter
    info("\n" + "-" * 60)
    info("STEP 3: RV filter")
    info(f"NGC 6791 RV: {NGC6791_RV:.1f} km/s")

    if len(plx_cands) > 0:
        rv_valid = plx_cands["rv_comp_1"].notna()
        rv_cands = plx_cands[rv_valid].copy()
        info(f"With RV: {len(rv_cands)}")

        if len(rv_cands) > 0:
            rv_offset = np.abs(rv_cands["rv_comp_1"].values - NGC6791_RV)
            rv_match = rv_offset < 15  # ±15 km/s
            rv_final = rv_cands[rv_match].copy()
            info(f"RV-consistent (±15 km/s): {len(rv_final)}")

    # 6. Gaia PM query
    info("\n" + "-" * 60)
    info("STEP 4: Gaia proper motion query")

    # Query all parallax candidates (or rv candidates if available)
    query_set = plx_cands if len(plx_cands) > 0 else chem_cands
    if len(query_set) > 0 and len(query_set) < 5000:
        source_ids = [int(s) for s in query_set["gaiadr3_source_id"].values
                      if s > 0 and not np.isnan(s)]
        info(f"Querying Gaia for {len(source_ids)} stars...")

        try:
            from astroquery.gaia import Gaia
            id_list = ",".join(str(s) for s in source_ids)
            q = f"""SELECT source_id, ra, dec, pmra, pmdec, parallax,
                           pmra_error, pmdec_error, parallax_error,
                           radial_velocity, phot_g_mean_mag, bp_rp
                    FROM gaiadr3.gaia_source
                    WHERE source_id IN ({id_list})"""
            job = Gaia.launch_job(q)
            gaia = job.get_results().to_pandas()
            info(f"Gaia results: {len(gaia)}")

            if len(gaia) > 0 and "pmra" in gaia.columns:
                gaia["pm_offset"] = np.sqrt((gaia["pmra"] - NGC6791_PMRA)**2 +
                                             (gaia["pmdec"] - NGC6791_PMDEC)**2)
                # PM tolerance: generous for 4 kpc cluster
                pm_tol = 3.0  # mas/yr
                pm_match = gaia["pm_offset"] < pm_tol
                pm_cands = gaia[pm_match].copy()
                info(f"PM-consistent (within {pm_tol} mas/yr): {len(pm_cands)}")

                # Show closest PM matches
                info(f"\nClosest 10 by PM offset:")
                for _, row in gaia.nsmallest(10, "pm_offset").iterrows():
                    info(f"  source_id={int(row['source_id'])}: "
                         f"PM=({row['pmra']:+.3f},{row['pmdec']:+.3f}), "
                         f"offset={row['pm_offset']:.3f}, "
                         f"plx={row['parallax']:.4f}, "
                         f"RV={row.get('radial_velocity', np.nan):.1f}, "
                         f"G={row.get('phot_g_mean_mag', np.nan):.2f}")

                # Merge with GALAH chemistry
                if len(pm_cands) > 0:
                    query_set["gaiadr3_source_id"] = query_set["gaiadr3_source_id"].astype(np.int64)
                    pm_cands["source_id"] = pm_cands["source_id"].astype(np.int64)
                    final = pm_cands.merge(
                        query_set[["gaiadr3_source_id", "C_O", "mg_fe", "si_fe", "fe_h",
                                   "rv_comp_1", "ba_fe", "ra", "dec", "teff", "logg"]],
                        left_on="source_id", right_on="gaiadr3_source_id",
                        how="inner", suffixes=("_gaia", "_galah"))

                    info(f"\n{'='*72}")
                    info(f"CANDIDATE DISSOLVED NGC 6791 MEMBERS: {len(final)}")
                    info(f"{'='*72}")

                    for i, (_, row) in enumerate(final.iterrows()):
                        info(f"\n  --- Candidate {i+1} ---")
                        info(f"  Gaia DR3: {int(row['source_id'])}")
                        ra_v = row.get("ra_gaia", row.get("ra_galah", row.get("ra")))
                        dec_v = row.get("dec_gaia", row.get("dec_galah", row.get("dec")))
                        info(f"  Position: ({ra_v:.5f}, {dec_v:.5f})")
                        sep_deg = np.sqrt((ra_v-NGC6791_RA)**2*np.cos(np.deg2rad(NGC6791_DEC))**2 +
                                         (dec_v-NGC6791_DEC)**2)
                        info(f"  Sep from NGC 6791: {sep_deg:.1f}°")
                        info(f"  Parallax: {row['parallax']:.4f} ± {row.get('parallax_error',0):.4f} mas "
                             f"(NGC 6791: {NGC6791_PLX:.3f})")
                        info(f"  PM: ({row['pmra']:+.3f}, {row['pmdec']:+.3f}) mas/yr "
                             f"(NGC 6791: {NGC6791_PMRA:+.3f}, {NGC6791_PMDEC:+.3f})")
                        info(f"  PM offset: {row['pm_offset']:.3f} mas/yr")
                        if pd.notna(row.get("rv_comp_1")):
                            info(f"  RV (GALAH): {row['rv_comp_1']:+.2f} km/s (NGC 6791: {NGC6791_RV:+.1f})")
                        if pd.notna(row.get("radial_velocity")):
                            info(f"  RV (Gaia): {row['radial_velocity']:+.2f} km/s")
                        info(f"  G mag: {row.get('phot_g_mean_mag', np.nan):.2f}")
                        info(f"  Chemistry: C/O={row['C_O']:.4f}, [Mg/Fe]={row['mg_fe']:+.4f}, "
                             f"[Si/Fe]={row['si_fe']:+.4f}, [Fe/H]={row['fe_h']:+.4f}")
                        if pd.notna(row.get("teff")):
                            info(f"  Teff={row['teff']:.0f} K, logg={row.get('logg', np.nan):.2f}")

                    final.to_csv("t20b_candidates.csv", index=False)
                    info(f"\nSaved: t20b_candidates.csv")

                # False positive estimate
                info("\n" + "-" * 60)
                info("FALSE POSITIVE ESTIMATE")
                f_chem = n_chem / len(galah)
                f_plx = len(plx_cands) / max(n_chem, 1)
                # Background PM density: query random stars at same distance
                # At 4 kpc, PMs are small — most stars have |PM| < 5 mas/yr
                # NGC 6791's PM is (-0.42, -2.28) — not extreme but specific
                f_pm = len(pm_cands) / max(len(gaia), 1)
                combined = f_chem * f_plx * f_pm
                info(f"  P(chem): {f_chem:.6f} ({n_chem}/{len(galah)})")
                info(f"  P(plx|chem): {f_plx:.4f}")
                info(f"  P(PM|plx+chem): {f_pm:.4f}")
                info(f"  Combined: {combined:.2e}")
                info(f"  Expected random in {len(galah)}: {combined*len(galah):.2f}")

        except Exception as e:
            info(f"Gaia query failed: {e}")

# Plot
info("\nGenerating plot...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("T20b: Find One Star — NGC 6791 ([Fe/H]=+0.30, 8 Gyr)\n"
             "GALAH DR4 × Gaia DR3 | Chemistry → Distance → PM",
             fontsize=13, fontweight="bold", y=0.99)

# P1: [Fe/H] distribution
ax = axes[0, 0]
ax.hist(galah["fe_h"], bins=100, color="lightgray", edgecolor="gray", alpha=0.7,
        label="All field", density=True)
ax.axvline(template_galah["fe_h"], color="red", lw=2.5,
           label=f"NGC 6791 ({template_galah['fe_h']:+.3f})")
ax.axvspan(centroid[3]-tol[3], centroid[3]+tol[3], alpha=0.15, color="red")
if n_chem > 0:
    ax.hist(chem_cands["fe_h"], bins=30, color="steelblue", edgecolor="navy",
            alpha=0.6, density=True, label=f"Chem match (N={n_chem})")
ax.set_xlabel("[Fe/H]", fontsize=11)
ax.set_title("Metallicity: NGC 6791 is EXTREME", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# P2: Chemistry space
ax = axes[0, 1]
rng = np.random.default_rng(42)
bg = rng.choice(len(galah), min(5000, len(galah)), replace=False)
ax.scatter(galah.iloc[bg]["fe_h"], galah.iloc[bg]["mg_fe"],
           s=2, c="lightgray", alpha=0.3, rasterized=True)
if n_chem > 0:
    ax.scatter(chem_cands["fe_h"], chem_cands["mg_fe"],
               s=20, c="salmon", alpha=0.5, label=f"Chem match")
ax.scatter(centroid[3], centroid[1], s=200, c="red", marker="+",
           linewidths=3, zorder=10, label="NGC 6791 template")
if 'final' in dir() and len(final) > 0:
    ax.scatter(final["fe_h"], final["mg_fe"], s=150, c="blue",
               edgecolors="navy", linewidths=2, marker="D", zorder=12,
               label=f"CANDIDATES ({len(final)})")
ax.set_xlabel("[Fe/H]", fontsize=11)
ax.set_ylabel("[Mg/Fe]", fontsize=11)
ax.set_title("[Fe/H] vs [Mg/Fe]", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# P3: PM space
ax = axes[1, 0]
if 'gaia' in dir() and len(gaia) > 0:
    ax.scatter(gaia["pmra"], gaia["pmdec"], s=15, c="lightgray", alpha=0.4,
               label="Chem+Plx candidates")
    ax.scatter(NGC6791_PMRA, NGC6791_PMDEC, s=300, c="red", marker="+",
               linewidths=3, zorder=10, label="NGC 6791")
    circle = plt.Circle((NGC6791_PMRA, NGC6791_PMDEC), 3.0,
                        fill=False, color="red", ls="--", lw=2)
    ax.add_patch(circle)
    if 'pm_cands' in dir() and len(pm_cands) > 0:
        ax.scatter(pm_cands["pmra"], pm_cands["pmdec"], s=100, c="blue",
                   edgecolors="navy", zorder=5, label=f"PM match ({len(pm_cands)})")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 5)
ax.set_xlabel("μα* (mas/yr)", fontsize=11)
ax.set_ylabel("μδ (mas/yr)", fontsize=11)
ax.set_title("Proper Motion Space", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# P4: Pipeline funnel
ax = axes[1, 1]
stages = ["GALAH field", "Chem match", "Plx match"]
counts = [len(galah), n_chem, len(plx_cands) if 'plx_cands' in dir() else 0]
if 'pm_cands' in dir():
    stages.append("PM match")
    counts.append(len(pm_cands))
if 'final' in dir():
    stages.append("FINAL")
    counts.append(len(final))

colors_bar = plt.cm.RdYlBu(np.linspace(0.8, 0.2, len(stages)))
ax.barh(range(len(stages)), counts, color=colors_bar, edgecolor="white")
for i, (s, c) in enumerate(zip(stages, counts)):
    ax.text(max(counts)*0.02, i, f"{s}: {c:,}", va="center", fontsize=10, fontweight="bold")
ax.set_yticks(range(len(stages)))
ax.set_yticklabels(stages)
ax.set_xlabel("Number of stars", fontsize=11)
ax.set_title("Selection Funnel", fontsize=11, fontweight="bold")
ax.set_xscale("log")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("t20b_ngc6791_plot.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
info("Saved: t20b_ngc6791_plot.png")

# Summary
info("\n" + "=" * 72)
info("T20b SUMMARY — NGC 6791")
info("=" * 72)
info(f"Chemical matches: {n_chem}")
if 'plx_cands' in dir():
    info(f"Parallax-consistent: {len(plx_cands)}")
if 'pm_cands' in dir():
    info(f"PM-consistent: {len(pm_cands)}")
if 'final' in dir():
    info(f"FINAL CANDIDATES: {len(final)}")
info("\nT20b complete.")

with open("t20b_results.txt", "w") as f:
    f.write("\n".join(out))

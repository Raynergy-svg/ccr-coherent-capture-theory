#!/usr/bin/env python3
"""
T20c — Find One Star: NGC 6253 (6D + Age)
============================================
Certan (2025) | CCT | GALAH DR4 × Gaia DR3

NGC 6253: [Fe/H]=+0.36-0.43, age=3-5 Gyr, distance ~1.5 kpc
Among the most metal-rich open clusters known.
80 GALAH members available — clean the template using the metal-rich core.
Filter field stars in 6 chemical dimensions + Gaia age estimate.
"""

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

SOLAR_CO = 0.549

# NGC 6253 literature properties
CL_RA = 254.778
CL_DEC = -52.712
CL_FEH_LIT = 0.40   # literature [Fe/H]
CL_AGE_LIT = 4.0     # Gyr (literature range 3-5)
CL_RV_LIT = -29.0    # km/s (literature)

EXCLUSION_RADIUS = 2.0  # degrees

out = []
def info(msg):
    line = "[INFO] " + str(msg)
    print(line, flush=True)
    out.append(line)

info("=" * 72)
info("T20c  Find One Star: NGC 6253 (6D + Age)")
info("Certan (2025) | CCT | GALAH DR4 × Gaia DR3")
info("=" * 72)
info("[Fe/H]=+0.36-0.43 | Age=3-5 Gyr | Most metal-rich old OC in GALAH")

# 1. Build cleaned template from GALAH members
t9 = pd.read_csv("t9_matched_stars.csv")
members = t9[t9["cluster_name"] == "NGC_6253"].copy()
info(f"\nAll T9 members: {len(members)}")

# The 80 members have huge scatter because of contamination
# Real NGC 6253 members should have [Fe/H] > +0.15 (GALAH scale)
# Select the metal-rich core
metal_rich = members[members["fe_h"] > 0.10].copy()
info(f"Metal-rich members ([Fe/H]>0.10): {len(metal_rich)}")

if len(metal_rich) < 5:
    # Try looser cut
    metal_rich = members[members["fe_h"] > 0.0].copy()
    info(f"Relaxed cut ([Fe/H]>0.0): {len(metal_rich)}")

dims = ["C_O", "mg_fe", "si_fe", "fe_h", "al_fe"]
clean = metal_rich.dropna(subset=dims)
info(f"Clean members with all 5D: {len(clean)}")

# Template centroid and scatter
centroid = {}
scatter = {}
for d in dims:
    centroid[d] = clean[d].mean()
    scatter[d] = clean[d].std(ddof=1)
    info(f"  {d}: {centroid[d]:+.4f} ± {scatter[d]:.4f}")

# Cluster RV from clean members
rv_clean = clean["rv_gaia_dr3"].dropna()
cl_rv = rv_clean.mean() if len(rv_clean) >= 3 else CL_RV_LIT
cl_rv_std = rv_clean.std(ddof=1) if len(rv_clean) > 1 else 5.0
info(f"  RV: {cl_rv:.2f} ± {cl_rv_std:.2f} km/s (N={len(rv_clean)})")

# Cluster PM and distance
cl_pmra = clean["pmra_cl"].iloc[0]
cl_pmdec = clean["pmdec_cl"].iloc[0]
cl_dist = clean["dist_cl"].iloc[0]
cl_plx = 1.0 / cl_dist if cl_dist > 0 else 0.67  # ~1.5 kpc
info(f"  PM: ({cl_pmra:.3f}, {cl_pmdec:.3f}) mas/yr")
info(f"  Distance: {cl_dist:.3f} kpc (parallax: {cl_plx:.3f} mas)")

# Matching tolerances — tight, using cluster scatter
TOL = {}
for d in dims:
    TOL[d] = max(2.5 * scatter[d], 0.06)  # 2.5σ or minimum 0.06
info(f"\nMatching tolerances:")
for d in dims:
    info(f"  {d}: ±{TOL[d]:.4f}")

# 2. Load GALAH field stars
info("\nLoading GALAH DR4...")
galah_table = Table.read("galah_dr4_allstar_240705.fits", memmap=True)
cols = ["ra", "dec", "snr_px_ccd3", "flag_sp",
        "c_fe", "o_fe", "flag_c_fe", "flag_o_fe",
        "mg_fe", "flag_mg_fe", "si_fe", "flag_si_fe",
        "fe_h", "flag_fe_h", "al_fe", "flag_al_fe",
        "ba_fe", "flag_ba_fe",
        "rv_comp_1", "e_rv_comp_1",
        "parallax", "parallax_error",
        "gaiadr3_source_id", "teff", "logg"]
galah = galah_table[[c for c in cols if c in galah_table.colnames]].to_pandas()

galah = galah[galah["snr_px_ccd3"] > 30].copy()
galah = galah[galah["flag_sp"] == 0].copy()
galah["C_O"] = (10.0 ** (galah["c_fe"] - galah["o_fe"])) * SOLAR_CO

for col, fc in [("C_O", None), ("mg_fe", "flag_mg_fe"), ("si_fe", "flag_si_fe"),
                ("fe_h", "flag_fe_h"), ("al_fe", "flag_al_fe")]:
    galah = galah[galah[col].notna()].copy()
    if fc and fc in galah.columns:
        galah = galah[galah[fc] == 0].copy()
galah = galah[(galah["C_O"] > 0.05) & (galah["C_O"] < 2.0)].copy()

# Exclude NGC 6253 region
sep = np.sqrt((galah["ra"] - CL_RA)**2 * np.cos(np.deg2rad(CL_DEC))**2 +
              (galah["dec"] - CL_DEC)**2)
galah = galah[sep > EXCLUSION_RADIUS].copy().reset_index(drop=True)
info(f"Field stars (5D valid, cluster-excluded): {len(galah)}")

# 3. Chemical matching (5D)
info("\n" + "-" * 60)
info("STEP 1: 5D Chemical matching")

n_high_feh = (galah["fe_h"] > 0.15).sum()
info(f"Stars with [Fe/H] > 0.15: {n_high_feh}/{len(galah)} ({n_high_feh/len(galah):.2%})")

field_matrix = galah[dims].values
cent_arr = np.array([centroid[d] for d in dims])
tol_arr = np.array([TOL[d] for d in dims])

delta = np.abs(field_matrix - cent_arr)
chem_match = (delta < tol_arr).all(axis=1)
n_chem = chem_match.sum()
info(f"5D chemical matches: {n_chem}")

if n_chem == 0:
    info("No matches. Broadening tolerance by 50%...")
    tol_arr *= 1.5
    chem_match = (np.abs(field_matrix - cent_arr) < tol_arr).all(axis=1)
    n_chem = chem_match.sum()
    info(f"Broad 5D matches: {n_chem}")

chem_cands = galah[chem_match].copy()
if n_chem > 0:
    info(f"  [Fe/H] range: {chem_cands['fe_h'].min():.3f} to {chem_cands['fe_h'].max():.3f}")

# 4. Gaia query for PM, parallax, age
info("\n" + "-" * 60)
info("STEP 2: Gaia DR3 kinematics + age")

if n_chem > 0 and n_chem < 10000:
    source_ids = [int(s) for s in chem_cands["gaiadr3_source_id"].values if s > 0 and not np.isnan(s)]
    info(f"Querying Gaia for {len(source_ids)} stars...")

    from astroquery.gaia import Gaia

    # Batch query
    all_gaia = []
    batch_size = 2000
    for i in range(0, len(source_ids), batch_size):
        batch = source_ids[i:i+batch_size]
        id_list = ",".join(str(s) for s in batch)
        q = f"""SELECT g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.parallax,
                       g.pmra_error, g.pmdec_error, g.parallax_error,
                       g.radial_velocity, g.phot_g_mean_mag, g.bp_rp,
                       a.age_flame, a.age_flame_lower, a.age_flame_upper,
                       a.mass_flame, a.teff_gspspec, a.mh_gspspec, a.alphafe_gspspec
                FROM gaiadr3.gaia_source g
                LEFT JOIN gaiadr3.astrophysical_parameters a ON g.source_id = a.source_id
                WHERE g.source_id IN ({id_list})"""
        try:
            job = Gaia.launch_job(q)
            all_gaia.append(job.get_results().to_pandas())
        except Exception as e:
            info(f"  Batch {i} failed: {e}")

    if all_gaia:
        gaia = pd.concat(all_gaia, ignore_index=True)
        info(f"Gaia results: {len(gaia)}")
    else:
        gaia = pd.DataFrame()
        info("No Gaia results")

    if len(gaia) > 0:
        # Parallax filter
        info("\n  PARALLAX FILTER:")
        info(f"  Cluster parallax: {cl_plx:.3f} mas")
        plx_err = np.maximum(gaia["parallax_error"].values, 0.1)
        plx_match = np.abs(gaia["parallax"] - cl_plx) < 3 * plx_err
        info(f"  Parallax-consistent: {plx_match.sum()}")

        # PM filter
        info(f"\n  PM FILTER:")
        info(f"  Cluster PM: ({cl_pmra:.3f}, {cl_pmdec:.3f}) mas/yr")
        pm_tol = 3.0  # mas/yr
        pm_offset = np.sqrt((gaia["pmra"] - cl_pmra)**2 + (gaia["pmdec"] - cl_pmdec)**2)
        pm_match = pm_offset < pm_tol
        info(f"  PM-consistent (within {pm_tol} mas/yr): {pm_match.sum()}")

        # RV filter
        info(f"\n  RV FILTER:")
        info(f"  Cluster RV: {cl_rv:.1f} km/s")
        rv_valid = gaia["radial_velocity"].notna()
        rv_match = rv_valid & (np.abs(gaia["radial_velocity"] - cl_rv) < 15)
        info(f"  RV-consistent (±15 km/s): {rv_match.sum()}")

        # AGE filter
        info(f"\n  AGE FILTER:")
        info(f"  Cluster age: {CL_AGE_LIT:.1f} Gyr (literature: 3-5 Gyr)")
        age_valid = gaia["age_flame"].notna()
        age_match = age_valid & (gaia["age_flame"] > 2.0) & (gaia["age_flame"] < 7.0)
        info(f"  Age-consistent (2-7 Gyr): {age_match.sum()}")

        # Combined filters
        combined = plx_match & pm_match
        info(f"\n  COMBINED: Plx + PM: {combined.sum()}")
        combined_rv = combined & rv_match
        info(f"  + RV: {combined_rv.sum()}")
        combined_age = combined_rv & age_match
        info(f"  + Age: {combined_age.sum()}")
        # Also: combined without strict RV (some stars lack Gaia RV)
        combined_plx_pm_age = plx_match & pm_match & age_match
        info(f"  Plx + PM + Age (no RV req): {combined_plx_pm_age.sum()}")

        # Show ALL candidates at each stage
        for label, mask in [("Plx+PM", combined),
                            ("Plx+PM+RV", combined_rv),
                            ("Plx+PM+Age", combined_plx_pm_age),
                            ("Plx+PM+RV+Age", combined_age)]:
            final = gaia[mask].copy()
            if len(final) > 0:
                # Merge back with GALAH chemistry
                chem_cands["gaiadr3_source_id"] = chem_cands["gaiadr3_source_id"].astype(np.int64)
                final["source_id"] = final["source_id"].astype(np.int64)
                merged = final.merge(
                    chem_cands[["gaiadr3_source_id", "C_O", "mg_fe", "si_fe",
                                "fe_h", "al_fe", "ba_fe", "rv_comp_1", "teff", "logg"]],
                    left_on="source_id", right_on="gaiadr3_source_id", how="inner",
                    suffixes=("_gaia", "_galah"))

                info(f"\n  {'='*60}")
                info(f"  {label} CANDIDATES: {len(merged)}")
                info(f"  {'='*60}")

                for i, (_, row) in enumerate(merged.iterrows()):
                    info(f"\n    --- Candidate ({label}) {i+1} ---")
                    info(f"    Gaia DR3: {int(row['source_id'])}")
                    ra_v = row.get("ra_gaia", row.get("ra", np.nan))
                    dec_v = row.get("dec_gaia", row.get("dec", np.nan))
                    info(f"    Position: ({ra_v:.5f}, {dec_v:.5f})")
                    sep_deg = np.sqrt((ra_v-CL_RA)**2*np.cos(np.deg2rad(CL_DEC))**2 +
                                     (dec_v-CL_DEC)**2)
                    info(f"    Sep from NGC 6253: {sep_deg:.1f}°")
                    info(f"    Parallax: {row['parallax']:.4f} ± {row.get('parallax_error',0):.4f} "
                         f"(cluster: {cl_plx:.3f})")
                    info(f"    PM: ({row['pmra']:+.3f}, {row['pmdec']:+.3f}) "
                         f"(cluster: {cl_pmra:+.3f}, {cl_pmdec:+.3f})")
                    info(f"    PM offset: {np.sqrt((row['pmra']-cl_pmra)**2+(row['pmdec']-cl_pmdec)**2):.3f}")
                    if pd.notna(row.get("radial_velocity")):
                        info(f"    RV (Gaia): {row['radial_velocity']:+.2f} (cluster: {cl_rv:+.1f})")
                    if pd.notna(row.get("rv_comp_1")):
                        info(f"    RV (GALAH): {row['rv_comp_1']:+.2f}")
                    if pd.notna(row.get("age_flame")):
                        lo = row.get("age_flame_lower", np.nan)
                        hi = row.get("age_flame_upper", np.nan)
                        info(f"    Age (Gaia): {row['age_flame']:.2f} Gyr "
                             f"({lo:.1f}-{hi:.1f}) (cluster: {CL_AGE_LIT:.1f})")
                    info(f"    G mag: {row.get('phot_g_mean_mag', np.nan):.2f}")
                    info(f"    Chemistry (GALAH):")
                    info(f"      [Fe/H]  = {row['fe_h']:+.4f}  (template: {centroid['fe_h']:+.4f})")
                    info(f"      [Mg/Fe] = {row['mg_fe']:+.4f}  (template: {centroid['mg_fe']:+.4f})")
                    info(f"      [Si/Fe] = {row['si_fe']:+.4f}  (template: {centroid['si_fe']:+.4f})")
                    info(f"      [Al/Fe] = {row['al_fe']:+.4f}  (template: {centroid['al_fe']:+.4f})")
                    info(f"      C/O     = {row['C_O']:.4f}  (template: {centroid['C_O']:.4f})")
                    if pd.notna(row.get("ba_fe")):
                        info(f"      [Ba/Fe] = {row['ba_fe']:+.4f}")
                    if pd.notna(row.get("mh_gspspec")):
                        info(f"    Gaia GSP-Spec [M/H]: {row['mh_gspspec']:+.3f}")
                    if pd.notna(row.get("alphafe_gspspec")):
                        info(f"    Gaia GSP-Spec [α/Fe]: {row['alphafe_gspspec']:+.3f}")

                if len(merged) > 0:
                    merged.to_csv(f"t20c_candidates_{label.replace('+','_')}.csv", index=False)

        # False positive estimate
        info("\n" + "-" * 60)
        info("FALSE POSITIVE ESTIMATE")
        f_chem = n_chem / len(galah)
        info(f"  P(5D chem): {f_chem:.6f} ({n_chem}/{len(galah)})")
        if plx_match.sum() > 0:
            f_plx = plx_match.sum() / len(gaia)
            info(f"  P(plx|chem): {f_plx:.4f}")
        if pm_match.sum() > 0:
            f_pm = pm_match.sum() / len(gaia)
            info(f"  P(PM|chem): {f_pm:.4f}")
        if age_match.sum() > 0:
            f_age = age_match.sum() / len(gaia)
            info(f"  P(age 2-7 Gyr|chem): {f_age:.4f}")
        if combined_age.sum() > 0:
            combined_p = f_chem * (combined_age.sum() / len(gaia))
            info(f"  Combined P: {combined_p:.2e}")
            info(f"  Expected random in {len(galah)}: {combined_p * len(galah):.2f}")

# Save
info("\nSaving...")
with open("t20c_results.txt", "w") as f:
    f.write("\n".join(out))
info("Saved: t20c_results.txt")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("T20c: NGC 6253 — 6D + Age Dissolved Member Search", fontsize=13, fontweight="bold")

ax = axes[0]
ax.hist(galah["fe_h"], bins=100, color="lightgray", density=True, label="Field")
ax.axvline(centroid["fe_h"], color="red", lw=2.5, label=f"NGC 6253 ({centroid['fe_h']:+.3f})")
ax.axvspan(centroid["fe_h"]-TOL["fe_h"], centroid["fe_h"]+TOL["fe_h"], alpha=0.15, color="red")
ax.set_xlabel("[Fe/H]"); ax.set_title("[Fe/H] Distribution"); ax.legend(fontsize=8)

ax = axes[1]
rng = np.random.default_rng(42)
bg = rng.choice(len(galah), min(5000, len(galah)), replace=False)
ax.scatter(galah.iloc[bg]["fe_h"], galah.iloc[bg]["mg_fe"], s=2, c="lightgray", alpha=0.3)
if n_chem > 0:
    ax.scatter(chem_cands["fe_h"], chem_cands["mg_fe"], s=20, c="salmon", alpha=0.5)
ax.scatter(centroid["fe_h"], centroid["mg_fe"], s=200, c="red", marker="+", linewidths=3, zorder=10)
ax.scatter(clean["fe_h"], clean["mg_fe"], s=50, c="gold", edgecolors="black", marker="*", zorder=9)
ax.set_xlabel("[Fe/H]"); ax.set_ylabel("[Mg/Fe]"); ax.set_title("Chemistry Space")

ax = axes[2]
stages = ["Field", "5D Chem"]
counts = [len(galah), n_chem]
if 'plx_match' in dir(): stages.append("+ Plx"); counts.append(int(plx_match.sum()))
if 'combined' in dir(): stages.append("+ PM"); counts.append(int(combined.sum()))
if 'combined_rv' in dir(): stages.append("+ RV"); counts.append(int(combined_rv.sum()))
if 'combined_age' in dir(): stages.append("+ Age"); counts.append(int(combined_age.sum()))
ax.barh(range(len(stages)), counts, color=plt.cm.RdYlBu(np.linspace(0.8, 0.2, len(stages))))
for i, (s, c) in enumerate(zip(stages, counts)):
    ax.text(max(counts)*0.01, i, f"{s}: {c:,}", va="center", fontsize=9, fontweight="bold")
ax.set_xscale("log"); ax.set_xlabel("N stars"); ax.set_title("Selection Funnel")

plt.tight_layout()
plt.savefig("t20c_ngc6253_plot.png", dpi=200, bbox_inches="tight")
plt.close()
info("Saved: t20c_ngc6253_plot.png")

info("\nT20c complete.")
with open("t20c_results.txt", "w") as f:
    f.write("\n".join(out))

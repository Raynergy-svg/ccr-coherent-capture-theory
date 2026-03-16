#!/usr/bin/env python3
"""
T20 — Find One Star
=====================
Certan (2026) | Coherent Capture Theory | GALAH DR4 × Gaia DR3

Takes GALAH field stars that chemically match the Praesepe (NGC 2632)
template. Cross-matches against Gaia DR3 for proper motions and parallaxes.
Filters for stars kinematically consistent with Praesepe's well-known
signature. Identifies individual candidate dissolved Praesepe members.

Praesepe properties:
  μα* = -35.7 to -36.1 mas/yr
  μδ  = -12.9 mas/yr
  parallax ~ 5.37 mas (186 pc)
  RV ~ 34 km/s
  Age ~ 700 Myr
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

# Praesepe known properties (Gaia DR3 / literature)
PRAE_PMRA  = -36.09   # mas/yr (from T9 data, consistent with literature)
PRAE_PMDEC = -12.92   # mas/yr
PRAE_PLX   = 5.37     # mas (= 186 pc)
PRAE_RV    = 34.0     # km/s (literature)
PRAE_RA    = 130.054  # deg
PRAE_DEC   = 19.621   # deg

# Matching tolerances
# Chemistry: fixed threshold (same as T16c)
CHEM_TOL = np.array([0.08, 0.05, 0.05, 0.10])  # C/O, Mg/Fe, Si/Fe, Fe/H
DIM_COLS = ["C_O", "mg_fe", "si_fe", "fe_h"]

# Kinematics: 3σ windows
PM_SIGMA    = 3.0    # mas/yr tolerance for PM matching
PLX_SIGMA   = 3.0    # sigma for parallax matching
RV_SIGMA    = 15.0   # km/s tolerance for RV

# Praesepe's PM dispersion (internal + measurement)
PRAE_PM_DISP = 1.5   # mas/yr (internal velocity dispersion + Gaia errors at 186 pc)
PRAE_PLX_ERR = 0.3   # mas (typical Gaia parallax error + intrinsic depth)
PRAE_RV_DISP = 3.0   # km/s (internal)

# Exclusion zone around Praesepe center (degrees) — don't want bound members
EXCLUSION_RADIUS = 5.0  # degrees (Praesepe tidal radius ~12 pc = ~3.7° at 186 pc)

out = []
def info(msg):
    line = "[INFO] " + str(msg)
    print(line, flush=True)
    out.append(line)

info("=" * 72)
info("T20  Find One Star — Dissolved Praesepe Member Recovery")
info("Certan (2026) | CCT | GALAH DR4 × Gaia DR3")
info("=" * 72)

# 1. Build Praesepe chemical template
t9_stars = pd.read_csv("t9_matched_stars.csv")
prae_members = t9_stars[t9_stars["cluster_name"] == "NGC_2632"].copy()
info(f"Praesepe members in T9: {len(prae_members)}")

prae_clean = prae_members.dropna(subset=DIM_COLS)
centroid = prae_clean[DIM_COLS].mean().values
scatter = prae_clean[DIM_COLS].std(ddof=1).values
info(f"Chemical centroid: C/O={centroid[0]:.4f}, Mg/Fe={centroid[1]:.4f}, "
     f"Si/Fe={centroid[2]:.4f}, Fe/H={centroid[3]:.4f}")
info(f"Internal scatter:  C/O={scatter[0]:.4f}, Mg/Fe={scatter[1]:.4f}, "
     f"Si/Fe={scatter[2]:.4f}, Fe/H={scatter[3]:.4f}")

# 2. Load GALAH with Gaia info
info("\nLoading GALAH DR4 with Gaia cross-match info...")
galah_table = Table.read("galah_dr4_allstar_240705.fits", memmap=True)
cols = ["ra", "dec", "snr_px_ccd3", "flag_sp",
        "c_fe", "o_fe", "flag_c_fe", "flag_o_fe",
        "mg_fe", "flag_mg_fe", "si_fe", "flag_si_fe",
        "fe_h", "flag_fe_h",
        "rv_comp_1", "e_rv_comp_1",
        "parallax", "parallax_error",
        "gaiadr3_source_id",
        "ba_fe", "flag_ba_fe"]
galah = galah_table[[c for c in cols if c in galah_table.colnames]].to_pandas()
info(f"GALAH total: {len(galah)}")

# Quality cuts
galah = galah[galah["snr_px_ccd3"] > 30].copy()
galah = galah[galah["flag_sp"] == 0].copy()
galah["C_O"] = (10.0 ** (galah["c_fe"] - galah["o_fe"])) * SOLAR_CO

for col, fc in [("C_O", None), ("mg_fe", "flag_mg_fe"),
                ("si_fe", "flag_si_fe"), ("fe_h", "flag_fe_h")]:
    galah = galah[galah[col].notna()].copy()
    if fc and fc in galah.columns:
        galah = galah[galah[fc] == 0].copy()
galah = galah[(galah["C_O"] > 0.05) & (galah["C_O"] < 2.0)].copy()
info(f"After quality cuts: {len(galah)}")

# Exclude stars within EXCLUSION_RADIUS of Praesepe center
sep = np.sqrt((galah["ra"] - PRAE_RA)**2 * np.cos(np.deg2rad(PRAE_DEC))**2 +
              (galah["dec"] - PRAE_DEC)**2)
galah = galah[sep > EXCLUSION_RADIUS].copy().reset_index(drop=True)
info(f"After excluding within {EXCLUSION_RADIUS}° of Praesepe: {len(galah)}")

# 3. Chemical matching
info("\n" + "-" * 60)
info("STEP 1: Chemical matching (fixed threshold)")
field_matrix = galah[DIM_COLS].values
delta = np.abs(field_matrix - centroid)
chem_match = (delta < CHEM_TOL).all(axis=1)
n_chem = chem_match.sum()
info(f"Chemical matches: {n_chem}")

chem_candidates = galah[chem_match].copy()
info(f"C/O range of matches: {chem_candidates['C_O'].min():.3f} - {chem_candidates['C_O'].max():.3f}")

# 4. Parallax filter (distance consistency)
info("\n" + "-" * 60)
info("STEP 2: Parallax filter (distance consistency with Praesepe)")
info(f"Praesepe parallax: {PRAE_PLX:.2f} mas (186 pc)")
info(f"Tolerance: ±{PLX_SIGMA}σ (σ = max(Gaia_error, {PRAE_PLX_ERR} mas))")

plx_valid = chem_candidates["parallax"].notna() & (chem_candidates["parallax"] > 0)
chem_candidates = chem_candidates[plx_valid].copy()
info(f"With valid parallax: {len(chem_candidates)}")

# Effective parallax error: max of reported error and intrinsic dispersion
plx_err = np.maximum(chem_candidates["parallax_error"].values, PRAE_PLX_ERR)
plx_offset = np.abs(chem_candidates["parallax"].values - PRAE_PLX)
plx_match = plx_offset < (PLX_SIGMA * plx_err)

plx_candidates = chem_candidates[plx_match].copy()
info(f"Parallax-consistent: {len(plx_candidates)}")
if len(plx_candidates) > 0:
    info(f"  Parallax range: {plx_candidates['parallax'].min():.3f} - {plx_candidates['parallax'].max():.3f} mas")
    info(f"  Distance range: {1000/plx_candidates['parallax'].max():.0f} - {1000/plx_candidates['parallax'].min():.0f} pc")

# 5. Query Gaia DR3 for proper motions
info("\n" + "-" * 60)
info("STEP 3: Gaia DR3 proper motion query")

if len(plx_candidates) == 0:
    info("No parallax-consistent candidates. Cannot proceed to PM check.")
    info("Trying broader parallax cut...")
    # Broaden to 5σ
    plx_match_broad = plx_offset < (5.0 * plx_err)
    plx_candidates = chem_candidates[plx_match_broad].copy()
    info(f"With 5σ parallax: {len(plx_candidates)}")

if len(plx_candidates) > 0:
    source_ids = plx_candidates["gaiadr3_source_id"].values
    info(f"Querying Gaia DR3 for {len(source_ids)} source IDs...")

    try:
        from astroquery.gaia import Gaia
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

        # Batch query
        batch_size = 2000
        all_gaia_rows = []

        for i in range(0, len(source_ids), batch_size):
            batch = source_ids[i:i+batch_size]
            # Filter valid IDs
            batch = [int(sid) for sid in batch if sid > 0 and not np.isnan(sid)]
            if not batch:
                continue

            id_list = ",".join(str(s) for s in batch)
            query = f"""
            SELECT source_id, ra, dec, pmra, pmdec, parallax, parallax_error,
                   pmra_error, pmdec_error, radial_velocity, radial_velocity_error,
                   phot_g_mean_mag, bp_rp
            FROM gaiadr3.gaia_source
            WHERE source_id IN ({id_list})
            """
            try:
                job = Gaia.launch_job(query)
                result = job.get_results().to_pandas()
                all_gaia_rows.append(result)
                if i % 10000 == 0:
                    info(f"  Queried {i+len(batch)}/{len(source_ids)}...")
            except Exception as e:
                info(f"  Batch query failed: {e}")
                continue

        if all_gaia_rows:
            gaia_df = pd.concat(all_gaia_rows, ignore_index=True)
            info(f"Gaia results: {len(gaia_df)} stars")
        else:
            gaia_df = pd.DataFrame()
            info("No Gaia results returned")

    except ImportError:
        info("astroquery not available — using GALAH parallax + local PM estimation")
        gaia_df = pd.DataFrame()
    except Exception as e:
        info(f"Gaia query failed: {e}")
        gaia_df = pd.DataFrame()

    # If Gaia query worked, filter on proper motions
    if len(gaia_df) > 0 and "pmra" in gaia_df.columns:
        info("\n" + "-" * 60)
        info("STEP 4: Proper motion filter")
        info(f"Praesepe PM: μα*={PRAE_PMRA:.2f}, μδ={PRAE_PMDEC:.2f} mas/yr")
        info(f"Tolerance: ±{PM_SIGMA} × {PRAE_PM_DISP} mas/yr")

        pm_tol = PM_SIGMA * PRAE_PM_DISP  # 3 × 1.5 = 4.5 mas/yr

        gaia_df["dpmra"] = np.abs(gaia_df["pmra"] - PRAE_PMRA)
        gaia_df["dpmdec"] = np.abs(gaia_df["pmdec"] - PRAE_PMDEC)
        gaia_df["pm_offset"] = np.sqrt(gaia_df["dpmra"]**2 + gaia_df["dpmdec"]**2)

        pm_match = gaia_df["pm_offset"] < pm_tol
        pm_candidates = gaia_df[pm_match].copy()
        info(f"PM-consistent candidates: {len(pm_candidates)}")

        if len(pm_candidates) > 0:
            # Merge back with GALAH data
            plx_candidates["gaiadr3_source_id"] = plx_candidates["gaiadr3_source_id"].astype(np.int64)
            pm_candidates["source_id"] = pm_candidates["source_id"].astype(np.int64)

            final = pm_candidates.merge(
                plx_candidates[["gaiadr3_source_id", "C_O", "mg_fe", "si_fe", "fe_h",
                                "rv_comp_1", "ba_fe", "ra", "dec"]],
                left_on="source_id", right_on="gaiadr3_source_id", how="inner",
                suffixes=("_gaia", "_galah")
            )
            info(f"Final candidates (chem + parallax + PM): {len(final)}")

            # RV filter (if available)
            if "rv_comp_1" in final.columns and len(final) > 0:
                rv_valid = final["rv_comp_1"].notna()
                if rv_valid.any():
                    rv_offset = np.abs(final.loc[rv_valid, "rv_comp_1"] - PRAE_RV)
                    rv_match = rv_offset < RV_SIGMA
                    n_rv_match = rv_match.sum()
                    info(f"  Also RV-consistent (within ±{RV_SIGMA} km/s): {n_rv_match}")

            # Print candidates
            if len(final) > 0:
                info("\n" + "=" * 72)
                info("CANDIDATE DISSOLVED PRAESEPE MEMBERS")
                info("=" * 72)

                for idx, row in final.iterrows():
                    info(f"\n  --- Candidate {idx+1} ---")
                    ra_val = row.get("ra_gaia", row.get("ra_galah", row.get("ra", np.nan)))
                    dec_val = row.get("dec_gaia", row.get("dec_galah", row.get("dec", np.nan)))
                    info(f"  Gaia DR3 source_id: {int(row['source_id'])}")
                    info(f"  Position: RA={ra_val:.6f}, DEC={dec_val:.6f}")
                    sep_deg = np.sqrt((ra_val - PRAE_RA)**2 * np.cos(np.deg2rad(PRAE_DEC))**2 +
                                     (dec_val - PRAE_DEC)**2)
                    info(f"  Angular sep from Praesepe: {sep_deg:.1f}°")
                    info(f"  Parallax: {row['parallax']:.3f} ± {row.get('parallax_error', 0):.3f} mas "
                         f"(Praesepe: {PRAE_PLX:.2f})")
                    info(f"  PM: μα*={row['pmra']:.3f}±{row.get('pmra_error', 0):.3f}, "
                         f"μδ={row['pmdec']:.3f}±{row.get('pmdec_error', 0):.3f} mas/yr "
                         f"(Praesepe: {PRAE_PMRA:.1f}, {PRAE_PMDEC:.1f})")
                    info(f"  PM offset: {row['pm_offset']:.3f} mas/yr")
                    if pd.notna(row.get("rv_comp_1")):
                        info(f"  RV (GALAH): {row['rv_comp_1']:.2f} km/s (Praesepe: {PRAE_RV:.1f})")
                    if pd.notna(row.get("radial_velocity")):
                        info(f"  RV (Gaia):  {row['radial_velocity']:.2f} km/s")
                    info(f"  G mag: {row.get('phot_g_mean_mag', np.nan):.2f}")
                    info(f"  Chemistry: C/O={row['C_O']:.4f}, [Mg/Fe]={row['mg_fe']:.4f}, "
                         f"[Si/Fe]={row['si_fe']:.4f}, [Fe/H]={row['fe_h']:.4f}")
                    if pd.notna(row.get("ba_fe")):
                        info(f"  [Ba/Fe]: {row['ba_fe']:.4f}")

                    # Compute combined probability of random coincidence
                    # Chemistry: ~17k matches out of 519k = 3.3%
                    # Parallax: ~few hundred out of 17k
                    # PM: few out of hundreds
                    info(f"  Distance from Praesepe center: {1000/row['parallax']:.0f} pc")

                # Save candidates
                final.to_csv("t20_candidates.csv", index=False)
                info(f"\nSaved: t20_candidates.csv")

            # Random coincidence estimate
            info("\n" + "-" * 60)
            info("RANDOM COINCIDENCE ESTIMATE")

            # Fraction of field stars in each filter
            f_chem = n_chem / len(galah)
            f_plx = len(plx_candidates) / max(n_chem, 1)
            f_pm = len(pm_candidates) / max(len(plx_candidates), 1) if len(plx_candidates) > 0 else 0

            info(f"  Chemical match rate: {f_chem:.4f} ({n_chem}/{len(galah)})")
            info(f"  Parallax match | chem: {f_plx:.4f} ({len(plx_candidates)}/{n_chem})")
            info(f"  PM match | plx+chem: {f_pm:.4f} ({len(pm_candidates)}/{len(plx_candidates)})")
            f_combined = f_chem * f_plx * f_pm
            info(f"  Combined probability: {f_combined:.2e}")
            info(f"  Expected random matches in {len(galah)} stars: {f_combined * len(galah):.2f}")

            if f_combined * len(galah) < 1:
                info("  => Less than 1 star expected by chance")
                info("     Any candidate is INDIVIDUALLY significant")

        else:
            info("No candidates passed PM filter.")

    elif len(gaia_df) == 0:
        # Fallback: use GALAH parallax only (no Gaia PM)
        info("No Gaia PM data — reporting parallax+chemistry candidates only")
        info(f"Parallax+chemistry candidates: {len(plx_candidates)}")
        if len(plx_candidates) > 0:
            info(f"  These need Gaia PM verification")
            for i, (_, row) in enumerate(plx_candidates.head(10).iterrows()):
                info(f"  {i+1}. source_id={int(row['gaiadr3_source_id'])}, "
                     f"plx={row['parallax']:.3f}, "
                     f"C/O={row['C_O']:.4f}, [Fe/H]={row['fe_h']:.4f}")

else:
    info("No candidates at any stage.")

# 6. Plot
info("\nGenerating plot...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("T20: Find One Star — Dissolved Praesepe Member Recovery\n"
             "Certan (2026) | GALAH DR4 × Gaia DR3",
             fontsize=13, fontweight="bold", y=0.99)

# P1: Chemistry space showing Praesepe template and matches
ax = axes[0, 0]
# All field stars (subsample)
rng = np.random.default_rng(42)
bg_idx = rng.choice(len(galah), min(5000, len(galah)), replace=False)
ax.scatter(galah.iloc[bg_idx]["fe_h"], galah.iloc[bg_idx]["mg_fe"],
           s=2, c="lightgray", alpha=0.3, rasterized=True, label="Field")
# Chemical matches
ax.scatter(chem_candidates["fe_h"], chem_candidates["mg_fe"],
           s=5, c="salmon", alpha=0.3, rasterized=True,
           label=f"Chem match (N={n_chem})")
# Praesepe members
ax.scatter(prae_clean["fe_h"], prae_clean["mg_fe"],
           s=100, c="gold", edgecolors="black", linewidths=1, zorder=10,
           marker="*", label="Praesepe members")
# Template center
ax.scatter(centroid[3], centroid[1], s=200, c="red", marker="+",
           linewidths=3, zorder=11, label="Template center")

# If we have final candidates, highlight them
if 'final' in dir() and len(final) > 0:
    ax.scatter(final["fe_h"], final["mg_fe"], s=150, c="blue",
               edgecolors="navy", linewidths=2, marker="D", zorder=12,
               label=f"CANDIDATES (N={len(final)})")

ax.set_xlabel("[Fe/H]", fontsize=11)
ax.set_ylabel("[Mg/Fe]", fontsize=11)
ax.set_title("Chemical Space: Praesepe Template", fontsize=11, fontweight="bold")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.15)

# P2: Parallax distribution
ax = axes[0, 1]
if len(chem_candidates) > 0:
    plx_vals = chem_candidates["parallax"].dropna()
    ax.hist(plx_vals[plx_vals > 0], bins=100, color="lightgray", edgecolor="gray",
            alpha=0.7, label="Chem matches")
    ax.axvline(PRAE_PLX, color="red", ls="-", lw=2.5, label=f"Praesepe ({PRAE_PLX} mas)")
    ax.axvspan(PRAE_PLX - PLX_SIGMA*PRAE_PLX_ERR, PRAE_PLX + PLX_SIGMA*PRAE_PLX_ERR,
               alpha=0.15, color="red", label=f"±{PLX_SIGMA}σ")
    if len(plx_candidates) > 0:
        ax.hist(plx_candidates["parallax"], bins=50, color="steelblue",
                edgecolor="navy", alpha=0.7, label=f"Plx match (N={len(plx_candidates)})")
ax.set_xlabel("Parallax (mas)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Parallax Filter", fontsize=11, fontweight="bold")
ax.set_xlim(0, 15)
ax.legend(fontsize=8)

# P3: Proper motion space (if available)
ax = axes[1, 0]
if 'gaia_df' in dir() and len(gaia_df) > 0 and "pmra" in gaia_df.columns:
    ax.scatter(gaia_df["pmra"], gaia_df["pmdec"], s=5, c="lightgray", alpha=0.3,
               rasterized=True, label="Plx candidates")
    ax.scatter(PRAE_PMRA, PRAE_PMDEC, s=200, c="red", marker="+",
               linewidths=3, zorder=10, label="Praesepe")
    circle = plt.Circle((PRAE_PMRA, PRAE_PMDEC), PM_SIGMA * PRAE_PM_DISP,
                        fill=False, color="red", ls="--", lw=2, label=f"±{PM_SIGMA}σ")
    ax.add_patch(circle)
    if 'pm_candidates' in dir() and len(pm_candidates) > 0:
        ax.scatter(pm_candidates["pmra"], pm_candidates["pmdec"],
                   s=80, c="blue", edgecolors="navy", linewidths=1,
                   zorder=5, label=f"PM match (N={len(pm_candidates)})")
    ax.set_xlim(PRAE_PMRA - 20, PRAE_PMRA + 20)
    ax.set_ylim(PRAE_PMDEC - 20, PRAE_PMDEC + 20)
else:
    ax.text(0.5, 0.5, "No Gaia PM data available", ha="center", va="center",
            transform=ax.transAxes, fontsize=14, color="gray")
ax.set_xlabel("μα* (mas/yr)", fontsize=11)
ax.set_ylabel("μδ (mas/yr)", fontsize=11)
ax.set_title("Proper Motion Space", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15)

# P4: Sky map
ax = axes[1, 1]
if len(chem_candidates) > 0:
    ax.scatter(chem_candidates["ra"], chem_candidates["dec"],
               s=2, c="lightgray", alpha=0.2, rasterized=True, label="Chem matches")
if len(plx_candidates) > 0:
    ax.scatter(plx_candidates["ra"], plx_candidates["dec"],
               s=15, c="salmon", alpha=0.5, label=f"Plx match (N={len(plx_candidates)})")
if 'final' in dir() and len(final) > 0:
    ra_f = final.get("ra_gaia", final.get("ra_galah", final.get("ra")))
    dec_f = final.get("dec_gaia", final.get("dec_galah", final.get("dec")))
    ax.scatter(ra_f, dec_f, s=150, c="blue", edgecolors="navy",
               linewidths=2, marker="D", zorder=10, label=f"CANDIDATES (N={len(final)})")
ax.scatter(PRAE_RA, PRAE_DEC, s=300, c="gold", edgecolors="black",
           linewidths=2, marker="*", zorder=11, label="Praesepe center")
# Exclusion zone
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(PRAE_RA + EXCLUSION_RADIUS*np.cos(theta)/np.cos(np.deg2rad(PRAE_DEC)),
        PRAE_DEC + EXCLUSION_RADIUS*np.sin(theta),
        "r--", lw=1, alpha=0.5, label=f"Exclusion ({EXCLUSION_RADIUS}°)")
ax.set_xlabel("RA (deg)", fontsize=11)
ax.set_ylabel("DEC (deg)", fontsize=11)
ax.set_title("Sky Distribution", fontsize=11, fontweight="bold")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.15)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("t20_find_one_star_plot.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
info("Saved: t20_find_one_star_plot.png")

# Summary
info("\n" + "=" * 72)
info("T20 SUMMARY")
info("=" * 72)
info(f"Praesepe template: {len(prae_clean)} members")
info(f"Chemical matches: {n_chem}")
info(f"Parallax-consistent: {len(plx_candidates)}")
if 'pm_candidates' in dir():
    info(f"PM-consistent: {len(pm_candidates)}")
if 'final' in dir():
    info(f"FINAL CANDIDATES: {len(final)}")
    if len(final) > 0:
        info("")
        info("These stars were identified by CHEMISTRY FIRST, then confirmed")
        info("by independent Gaia kinematics. They are candidate dissolved")
        info("Praesepe members — individual stars carrying their birth")
        info("certificate, recoverable from the field population.")

info("\nT20 complete.")

with open("t20_results.txt", "w") as f:
    f.write("\n".join(out))

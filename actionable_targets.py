#!/usr/bin/env python3
"""
Actionable Target List: From 4,970 to Observable
==================================================
Certan (2026)

Sequential filters on the habitability v2 excellent-chemistry sample:
1. Binary removal (Gaia RUWE > 1.4)
2. Brightness + distance (G < 12, dist < 200 pc)
3. Known planet host cross-match
4. Age tightening (2-8 Gyr, gyrochronology-consistent)
5. Stellar activity (quiet stars preferred)

Produces a ranked, observable target list for RV/transit/direct imaging follow-up.
"""

import numpy as np, pandas as pd
from astropy.table import Table
from scipy.spatial import cKDTree
from astroquery.gaia import Gaia
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 72)
print("ACTIONABLE TARGET LIST")
print("From 4,970 excellent-chemistry stars to observable targets")
print("=" * 72)

# Load the v2 catalog
hab = pd.read_csv("habitability_v2_targets.csv")
excellent = hab[hab["hab_score"] > 0.9].copy()
print(f"\nStarting sample: {len(excellent)} excellent-chemistry FGK dwarfs")

# We need Gaia data for RUWE and G magnitude
# Query in batches by source_id
sids = excellent["gaiadr3_source_id"].astype(np.int64).values
valid_sids = sids[sids > 0]
print(f"Valid Gaia source IDs: {len(valid_sids)}")

print("\nQuerying Gaia DR3 for RUWE, G magnitude, binary flags...")
all_gaia = []
batch_size = 2000
for i in range(0, len(valid_sids), batch_size):
    batch = valid_sids[i:i+batch_size]
    id_list = ",".join(str(int(s)) for s in batch)
    q = f"""SELECT source_id, phot_g_mean_mag, ruwe,
                   non_single_star, phot_bp_rp_excess_factor
            FROM gaiadr3.gaia_source
            WHERE source_id IN ({id_list})"""
    try:
        job = Gaia.launch_job(q)
        all_gaia.append(job.get_results().to_pandas())
        if i % 10000 == 0 and i > 0:
            print(f"  Queried {i}/{len(valid_sids)}...")
    except Exception as e:
        print(f"  Batch {i} failed: {e}")

if all_gaia:
    gaia_df = pd.concat(all_gaia, ignore_index=True)
    print(f"Gaia data retrieved: {len(gaia_df)} stars")
else:
    print("Gaia query failed — using parallax-only filtering")
    gaia_df = pd.DataFrame()

# Merge Gaia data
if len(gaia_df) > 0:
    gaia_df["source_id"] = gaia_df["source_id"].astype(np.int64)
    excellent = excellent.merge(gaia_df, left_on="gaiadr3_source_id",
                                 right_on="source_id", how="left")

# =========================================================================
# FILTER 1: Binary removal (RUWE > 1.4)
# =========================================================================
print(f"\n{'='*60}")
print("FILTER 1: BINARY REMOVAL (RUWE > 1.4)")
n_before = len(excellent)

if "ruwe" in excellent.columns:
    ruwe_valid = excellent["ruwe"].notna()
    n_ruwe = ruwe_valid.sum()
    n_binary = ((excellent["ruwe"] > 1.4) & ruwe_valid).sum()
    excellent = excellent[~((excellent["ruwe"] > 1.4) & ruwe_valid)].copy()
    print(f"  RUWE data: {n_ruwe}/{n_before}")
    print(f"  Binaries removed (RUWE > 1.4): {n_binary}")
    print(f"  Remaining: {len(excellent)}")
else:
    print("  No RUWE data — skipping")

# Also check non_single_star flag
if "non_single_star" in excellent.columns:
    nss = (excellent["non_single_star"] > 0).sum() if excellent["non_single_star"].notna().any() else 0
    excellent = excellent[~(excellent["non_single_star"] > 0)].copy()
    print(f"  NSS flagged removed: {nss}")
    print(f"  After NSS: {len(excellent)}")

# =========================================================================
# FILTER 2: Brightness + Distance
# =========================================================================
print(f"\n{'='*60}")
print("FILTER 2: BRIGHTNESS + DISTANCE")

# Distance from parallax
excellent["dist_pc_calc"] = np.where(
    excellent["parallax"] > 0.5,
    1000.0 / excellent["parallax"],
    np.nan
)

# G magnitude
if "phot_g_mean_mag" in excellent.columns:
    g_col = "phot_g_mean_mag"
else:
    g_col = None

n_before = len(excellent)

# Apply distance cut
dist_mask = excellent["dist_pc_calc"].notna() & (excellent["dist_pc_calc"] < 200)
print(f"  Within 200 pc: {dist_mask.sum()}/{n_before}")

# Apply brightness cut
if g_col:
    bright_mask = excellent[g_col].notna() & (excellent[g_col] < 12)
    print(f"  G < 12: {bright_mask.sum()}/{n_before}")
    combined = dist_mask & bright_mask
else:
    combined = dist_mask

excellent_nearby = excellent[combined].copy()
print(f"  After distance + brightness: {len(excellent_nearby)}")

# =========================================================================
# FILTER 3: Known planet host cross-match
# =========================================================================
print(f"\n{'='*60}")
print("FILTER 3: KNOWN PLANET HOST CROSS-MATCH")

planets = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="pl_name,hostname,ra,dec,pl_rade,pl_orbper,pl_orbsmax,pl_eqt",
).to_pandas()

print(f"Total confirmed planets: {len(planets)}")

# Cross-match by position
g_ra = excellent_nearby["ra"].values
g_dec = excellent_nearby["dec"].values
g_coords = np.deg2rad(np.column_stack([g_ra, g_dec]))
g_xyz = np.column_stack([np.cos(g_coords[:,1])*np.cos(g_coords[:,0]),
                          np.cos(g_coords[:,1])*np.sin(g_coords[:,0]),
                          np.sin(g_coords[:,1])])

p_coords = np.deg2rad(np.column_stack([planets["ra"].values, planets["dec"].values]))
p_xyz = np.column_stack([np.cos(p_coords[:,1])*np.cos(p_coords[:,0]),
                          np.cos(p_coords[:,1])*np.sin(p_coords[:,0]),
                          np.sin(p_coords[:,1])])

if len(excellent_nearby) > 0:
    tree = cKDTree(g_xyz)
    tol = 2 * np.sin(np.deg2rad(5/3600) / 2)
    dists, indices = tree.query(p_xyz, k=1)
    matched = dists < tol

    matched_planets = planets[matched].copy()
    matched_host_sids = excellent_nearby.iloc[indices[matched]]["gaiadr3_source_id"].values

    # Tag excellent stars that host planets
    excellent_nearby["has_planet"] = excellent_nearby["gaiadr3_source_id"].isin(matched_host_sids)
    excellent_nearby["has_rocky"] = False
    excellent_nearby["has_hz"] = False

    # Check for rocky / HZ planets
    for sid in matched_host_sids:
        idx_star = excellent_nearby[excellent_nearby["gaiadr3_source_id"] == sid].index
        if len(idx_star) == 0:
            continue
        star_planets = matched_planets[
            excellent_nearby.iloc[indices[matched]]["gaiadr3_source_id"].values == sid
        ]
        if (star_planets["pl_rade"] < 2.0).any():
            excellent_nearby.loc[idx_star, "has_rocky"] = True
        # HZ: equilibrium temp 200-350 K (rough)
        if ((star_planets["pl_eqt"] > 200) & (star_planets["pl_eqt"] < 350)).any():
            excellent_nearby.loc[idx_star, "has_hz"] = True

    n_hosts = excellent_nearby["has_planet"].sum()
    n_rocky = excellent_nearby["has_rocky"].sum()
    n_hz = excellent_nearby["has_hz"].sum()
    print(f"  Known planet hosts: {n_hosts}")
    print(f"  With rocky planet (R < 2 R_E): {n_rocky}")
    print(f"  With HZ planet: {n_hz}")
else:
    excellent_nearby["has_planet"] = False
    excellent_nearby["has_rocky"] = False
    excellent_nearby["has_hz"] = False

# =========================================================================
# FILTER 4: Age tightening (2-8 Gyr)
# =========================================================================
print(f"\n{'='*60}")
print("FILTER 4: AGE TIGHTENING (2-8 Gyr)")

n_before = len(excellent_nearby)
if "age" in excellent_nearby.columns:
    age_valid = excellent_nearby["age"].notna() & (excellent_nearby["age"] > 0)
    age_good = age_valid & (excellent_nearby["age"] > 2.0) & (excellent_nearby["age"] < 8.0)
    age_bad = age_valid & ((excellent_nearby["age"] < 1.0) | (excellent_nearby["age"] > 10.0))

    # Don't remove stars without ages — just flag
    excellent_nearby["age_quality"] = "unknown"
    excellent_nearby.loc[age_good, "age_quality"] = "optimal"
    excellent_nearby.loc[age_bad, "age_quality"] = "suboptimal"
    excellent_nearby.loc[age_valid & ~age_good & ~age_bad, "age_quality"] = "acceptable"

    print(f"  Optimal (2-8 Gyr): {age_good.sum()}")
    print(f"  Acceptable (1-2 or 8-10 Gyr): {(age_valid & ~age_good & ~age_bad).sum()}")
    print(f"  Suboptimal (<1 or >10 Gyr): {age_bad.sum()}")
    print(f"  Unknown age: {(~age_valid).sum()}")

    # Remove only clearly suboptimal
    excellent_nearby = excellent_nearby[~age_bad].copy()
    print(f"  After removing suboptimal: {len(excellent_nearby)}")

# =========================================================================
# FILTER 5: Stellar activity (quiet stars)
# =========================================================================
print(f"\n{'='*60}")
print("FILTER 5: STELLAR ACTIVITY")

# GALAH doesn't have direct activity indicators in our extract
# Use Teff as a rough proxy: cooler K dwarfs tend to be more active
# but we're already FGK filtered. Flag the coolest as potentially active.
if "teff" in excellent_nearby.columns:
    very_cool = excellent_nearby["teff"] < 4500
    n_cool = very_cool.sum()
    excellent_nearby["activity_flag"] = "normal"
    excellent_nearby.loc[very_cool, "activity_flag"] = "potentially_active"
    print(f"  Potentially active (Teff < 4500 K): {n_cool}")
    print(f"  Normal: {len(excellent_nearby) - n_cool}")
    # Don't remove — just flag
    print(f"  (Flagged, not removed — need activity data for hard cut)")

# =========================================================================
# FINAL RANKING
# =========================================================================
print(f"\n{'='*60}")
print("FINAL RANKING")
print(f"{'='*60}")

# Priority score: hab_score + bonuses
excellent_nearby["priority"] = excellent_nearby["hab_score"].copy()

# Bonuses
if "has_planet" in excellent_nearby.columns:
    excellent_nearby.loc[excellent_nearby["has_rocky"], "priority"] += 0.05
    excellent_nearby.loc[excellent_nearby["has_hz"], "priority"] += 0.10
    excellent_nearby.loc[excellent_nearby["has_planet"] & ~excellent_nearby["has_rocky"], "priority"] += 0.02

if "age_quality" in excellent_nearby.columns:
    excellent_nearby.loc[excellent_nearby["age_quality"] == "optimal", "priority"] += 0.02

# Penalty for potentially active
if "activity_flag" in excellent_nearby.columns:
    excellent_nearby.loc[excellent_nearby["activity_flag"] == "potentially_active", "priority"] -= 0.01

# Distance bonus (closer = better)
if "dist_pc_calc" in excellent_nearby.columns:
    excellent_nearby["dist_bonus"] = np.clip(1.0 - excellent_nearby["dist_pc_calc"] / 200, 0, 0.03)
    excellent_nearby["priority"] += excellent_nearby["dist_bonus"]

ranked = excellent_nearby.sort_values("priority", ascending=False).reset_index(drop=True)

# Summary
print(f"\n  Starting sample:     4,970")
print(f"  After binary cut:    {n_before}")
print(f"  After dist+bright:   {len(excellent_nearby)}")
print(f"  FINAL ACTIONABLE:    {len(ranked)}")

# Print funnel
print(f"\n  FILTER FUNNEL:")
print(f"    Excellent chemistry (>0.9):     4,970")
if "ruwe" in hab.columns or "ruwe" in excellent.columns:
    print(f"    - Binaries (RUWE>1.4):        ~{n_binary}")
print(f"    + Distance < 200 pc:            {dist_mask.sum()}")
print(f"    + G < 12:                       {len(ranked)}")
if age_bad.any():
    print(f"    - Bad age (<1 or >10 Gyr):    ~{age_bad.sum()}")

# Top 30
print(f"\nTOP 30 ACTIONABLE TARGETS:")
print(f"{'Rk':>3} {'Gaia ID':>22} {'RA':>8} {'DEC':>8} {'G':>5} {'Dist':>5} "
      f"{'Score':>6} {'Age':>5} {'CCT':>15} {'Planet':>7} {'Prior':>6}")
for i, (_, r) in enumerate(ranked.head(30).iterrows()):
    g_str = f"{r['phot_g_mean_mag']:.1f}" if pd.notna(r.get("phot_g_mean_mag")) else " --"
    d_str = f"{r['dist_pc_calc']:.0f}" if pd.notna(r.get("dist_pc_calc")) else "--"
    age_str = f"{r['age']:.1f}" if pd.notna(r.get("age")) and r["age"] > 0 else " --"
    cl = r["cct_cluster"][:15] if r.get("cct_cluster", "field") != "field" else "field"
    pl = "ROCKY" if r.get("has_rocky") else ("YES" if r.get("has_planet") else "no")
    print(f"  {i+1:>3} {int(r['gaiadr3_source_id']):>22} {r['ra']:>8.3f} {r['dec']:>8.3f} "
          f"{g_str:>5} {d_str:>5} {r['hab_score']:>6.4f} {age_str:>5} {cl:>15} {pl:>7} {r['priority']:>6.4f}")

# Save
save_cols = ["gaiadr3_source_id", "ra", "dec", "priority", "hab_score",
             "phot_g_mean_mag", "dist_pc_calc", "ruwe",
             "C_O", "fe_h", "mg_fe", "si_fe", "ba_fe", "age",
             "teff", "logg", "cct_cluster", "cct_distance",
             "has_planet", "has_rocky", "has_hz",
             "age_quality", "activity_flag",
             "s_CO", "s_MgSi", "s_FeH", "s_volatile", "s_age"]
save_cols = [c for c in save_cols if c in ranked.columns]
ranked[save_cols].to_csv("actionable_targets.csv", index=False)
print(f"\nSaved: actionable_targets.csv ({len(ranked)} stars)")
ranked.head(100)[save_cols].to_csv("actionable_top100.csv", index=False)
print(f"Saved: actionable_top100.csv")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f"Actionable Target List: {len(ranked)} Observable Stars\n"
             "Certan (2026) | 9D Habitability + Binary/Distance/Age Filters",
             fontsize=13, fontweight="bold")

ax = axes[0, 0]
ax.hist(ranked["dist_pc_calc"].dropna(), bins=30, color="steelblue", edgecolor="white")
ax.set_xlabel("Distance (pc)"); ax.set_ylabel("Count")
ax.set_title(f"Distance Distribution (N={len(ranked)})"); ax.axvline(100, color="red", ls="--")

ax = axes[0, 1]
if "phot_g_mean_mag" in ranked.columns:
    ax.hist(ranked["phot_g_mean_mag"].dropna(), bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("G magnitude"); ax.set_title("Brightness Distribution")

ax = axes[1, 0]
ax.scatter(ranked["dist_pc_calc"], ranked["hab_score"], s=10, c="steelblue", alpha=0.5)
if ranked["has_planet"].any():
    pl_stars = ranked[ranked["has_planet"]]
    ax.scatter(pl_stars["dist_pc_calc"], pl_stars["hab_score"], s=80, c="red",
               marker="*", zorder=10, label=f"Known hosts ({len(pl_stars)})")
    ax.legend(fontsize=8)
ax.set_xlabel("Distance (pc)"); ax.set_ylabel("Hab Score")
ax.set_title("Score vs Distance")

ax = axes[1, 1]
if "age" in ranked.columns:
    age_ok = ranked["age"].notna() & (ranked["age"] > 0)
    ax.hist(ranked.loc[age_ok, "age"], bins=30, color="steelblue", edgecolor="white")
    ax.axvline(4.57, color="gold", lw=2, label="Sun")
    ax.axvspan(2, 8, alpha=0.1, color="green", label="Optimal")
    ax.set_xlabel("Age (Gyr)"); ax.set_title("Age Distribution"); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("actionable_targets.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: actionable_targets.png")
print("\nDone.")

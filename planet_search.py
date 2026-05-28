"""
Find a planet around a chemistry-priority target.

Aggregates target list:
  - 32 actionable nearby dwarfs (8D scorer)
  - 47 reachable backups (200-300 pc relaxation)
  - HD 28888 (#1 subgiant actionable)
  - top 25 chemistry leaders
  -> deduplicated ~125 star pool

Queries:
  1. NASA Exoplanet Archive TOI (TESS Object of Interest) table by TIC ID and
     by position (5 arcsec). Looks for unconfirmed planet candidates.
  2. NASA Exoplanet Archive pscomppars (confirmed planets) by position.
  3. Gaia DR3 nss_two_body_orbit (astrometric/RV companions) by source_id.
  4. SIMBAD for V mag (for ESPRESSO feasibility) and SpType.

Reports any hits, prioritised by chemistry score + observability.
"""
import io, time, requests
import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

GAIA = "https://gea.esac.esa.int/tap-server/tap/sync"
NEA  = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
SIM  = "https://simbad.u-strasbg.fr/simbad/sim-tap/sync"

def tap(url, q, retries=4):
    for a in range(retries):
        try:
            r = requests.post(url, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=180)
            if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
            print(f"[WARN] HTTP {r.status_code}: {r.text[:120]}")
        except Exception as e: print(f"[WARN] {e}")
        time.sleep(2**a)
    return pd.DataFrame()

# ---- build target list ----
print("Building target list...")
dwarfs = pd.read_csv("real_dwarf_targets.csv")
print(f"  32 actionable dwarfs: {len(dwarfs)}")
backups = pd.read_csv("relaxation_backups.csv") if False else pd.DataFrame()
try:
    backups = pd.read_csv("relaxation_backups.csv")
    print(f"  47 reachable backups: {len(backups)}")
except: pass

# rename Gaia id columns to align
def col(df, name): return df[name] if name in df.columns else None
dwarfs_g = dwarfs["gaiadr3_source_id"].astype("int64")
backup_g = backups["gaiadr3_source_id"].astype("int64") if len(backups) else pd.Series([],dtype="int64")
# top-25 chemistry leaders
top25 = pd.read_csv("top25_chemistry_leaders.csv") if False else pd.DataFrame()
try:
    top25 = pd.read_csv("top25_chemistry_leaders.csv")
    print(f"  top 25 chemistry leaders: {len(top25)}")
    top25_g = top25["sid"].astype("int64")
except: top25_g = pd.Series([], dtype="int64")

# HD 28888 itself
HD28888 = 3312653647717790720

all_ids = list(set([HD28888] + dwarfs_g.tolist() + backup_g.tolist() + top25_g.tolist()))
print(f"\nDEDUPLICATED TARGET POOL: {len(all_ids)} stars")
idlist = ",".join(str(i) for i in all_ids)

# ---- 1. Pull Gaia DR3 RA/Dec + TIC + 2MASS for the pool ----
print("\nQuerying Gaia DR3 + TIC for the pool...")
g = tap(GAIA, f"""SELECT g.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.parallax,
                          g.ruwe, g.non_single_star
                   FROM gaiadr3.gaia_source g
                   WHERE g.source_id IN ({idlist})""")
print(f"  Gaia rows: {len(g)}")
g = g.rename(columns={"source_id":"gaia_id"})

# ---- 2. NSS astrometric / orbital companions ----
print("\nQuerying Gaia DR3 NSS (astrometric/orbital companions)...")
nss = tap(GAIA, f"""SELECT source_id, nss_solution_type, period, t_periastron,
                            eccentricity, semi_amplitude_primary
                     FROM gaiadr3.nss_two_body_orbit
                     WHERE source_id IN ({idlist})""")
print(f"  NSS two-body orbit hits: {len(nss)}")

nss_acc = tap(GAIA, f"""SELECT source_id, nss_solution_type, mass_function,
                                 inclination, semimajor_axis_ratio
                          FROM gaiadr3.nss_acceleration_astro
                          WHERE source_id IN ({idlist})""")
print(f"  NSS acceleration hits: {len(nss_acc)}")

# ---- 3. NASA Exoplanet Archive TOI table ----
print("\nQuerying NASA Exoplanet Archive TOI catalog...")
toi = tap(NEA, "SELECT toi, tid, toipfx, ra, dec, pl_orbper, pl_rade, pl_eqt, "
               "tfopwg_disp, toi_created FROM toi")
print(f"  TOI catalog total: {len(toi)}")

# position match: 5 arcsec
if len(g) and len(toi):
    gc = SkyCoord(g.ra.values*u.deg, g.dec.values*u.deg)
    tc = SkyCoord(toi.ra.values*u.deg, toi.dec.values*u.deg)
    idx, sep, _ = gc.match_to_catalog_sky(tc)
    matches = sep.arcsec < 5
    hits_toi = pd.DataFrame({
        "gaia_id": g.gaia_id.values[matches],
        "ra_gaia": g.ra.values[matches], "dec_gaia": g.dec.values[matches],
        "G": g.phot_g_mean_mag.values[matches],
        "toi": toi.toi.values[idx[matches]],
        "tid": toi.tid.values[idx[matches]],
        "pl_orbper": toi.pl_orbper.values[idx[matches]],
        "pl_rade": toi.pl_rade.values[idx[matches]],
        "pl_eqt": toi.pl_eqt.values[idx[matches]],
        "tfopwg_disp": toi.tfopwg_disp.values[idx[matches]],
        "sep_arcsec": sep.arcsec[matches],
    })
    print(f"\n>>> TOI MATCHES IN TARGET POOL: {len(hits_toi)} <<<")
    if len(hits_toi):
        print(hits_toi.to_string(index=False))
else:
    hits_toi = pd.DataFrame()

# ---- 4. Confirmed planets (pscomppars) ----
print("\nQuerying NASA confirmed planets (pscomppars)...")
pps = tap(NEA, "SELECT pl_name, hostname, ra, dec, pl_orbper, pl_rade, pl_eqt, "
               "pl_orbsmax, gaia_dr3_id FROM pscomppars")
print(f"  confirmed planets total: {len(pps)}")
if len(g) and len(pps):
    gc = SkyCoord(g.ra.values*u.deg, g.dec.values*u.deg)
    pc = SkyCoord(pps.ra.values*u.deg, pps.dec.values*u.deg)
    idx2, sep2, _ = gc.match_to_catalog_sky(pc)
    pmatch = sep2.arcsec < 5
    hits_pl = pd.DataFrame({
        "gaia_id": g.gaia_id.values[pmatch],
        "pl_name": pps.pl_name.values[idx2[pmatch]],
        "hostname": pps.hostname.values[idx2[pmatch]],
        "pl_orbper": pps.pl_orbper.values[idx2[pmatch]],
        "pl_rade": pps.pl_rade.values[idx2[pmatch]],
        "pl_eqt": pps.pl_eqt.values[idx2[pmatch]],
        "pl_orbsmax": pps.pl_orbsmax.values[idx2[pmatch]],
        "sep_arcsec": sep2.arcsec[pmatch],
    })
    print(f"\n>>> CONFIRMED PLANET MATCHES IN TARGET POOL: {len(hits_pl)} <<<")
    if len(hits_pl):
        print(hits_pl.to_string(index=False))
else:
    hits_pl = pd.DataFrame()

# ---- save & summarise ----
print("\n" + "="*78)
print("SUMMARY")
print("="*78)
print(f"target pool:           {len(all_ids)} stars")
print(f"Gaia DR3 found:        {len(g)}")
print(f"NSS two-body orbit:    {len(nss)}")
print(f"NSS acceleration:      {len(nss_acc)}")
print(f"TOI candidates:        {len(hits_toi)}")
print(f"Confirmed planets:     {len(hits_pl)}")
if len(nss):
    print(f"\nNSS two-body details:")
    print(nss.to_string(index=False))
if len(nss_acc):
    print(f"\nNSS acceleration details:")
    print(nss_acc.to_string(index=False))
hits_toi.to_csv("planet_search_TOI_hits.csv", index=False)
hits_pl.to_csv("planet_search_confirmed_hits.csv", index=False)
nss.to_csv("planet_search_NSS_orbit.csv", index=False)
nss_acc.to_csv("planet_search_NSS_acc.csv", index=False)
print("\n[saved 4 CSVs]")
print("DONE")

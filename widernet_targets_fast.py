"""Quick build: take widernet_targets_predicted.csv (from tess-point) and
prepare it directly for the BLS pipeline -- skip the slow MAST confirmation
loop. The BLS pipeline handles MAST downloads per target itself.

Output schema must match what widernet_bls_pipeline expects:
  apogee_id, ra, dec, tic_id, Tmag, R_star, M_star, Teff_tic,
  n_lc_sectors_actual

For the missing fields:
  - tic_id, Tmag: leave blank; pipeline can lookup or skip
  - R_star, M_star: estimate from APOGEE Teff + logg via simple
    main-sequence relations (R ~ (Teff/5778)^0.5 * (10^(4.44-logg))^0.5)
    or use sensible defaults (1.0 R_sun, 1.0 M_sun) -- pipeline will
    only use these for if-real R_p/T_eq calculations, not for the BLS
    itself
  - Teff_tic: use APOGEE Teff
  - n_lc_sectors_actual: use n_sectors_predicted as proxy (close enough)

Filter to top-N by predicted sector count, exclude known TOIs/hosts.
"""
import os, io, time, requests
import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
N_TARGETS = 500

def tap(q, retries=3, timeout=300):
    for a in range(retries):
        try:
            r = requests.post(NEA, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=timeout)
            if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
        except Exception as e: print(f"  WARN {e}")
        time.sleep(2**a)
    return pd.DataFrame()

print("="*78)
print("FAST WIDERNET TARGET PREP (skip MAST confirmation)")
print("="*78)

pred = pd.read_csv("widernet_targets_predicted.csv")
print(f"Predicted-sector pool: {len(pred)}")

# Cross-match to TOI/confirmed planets and exclude (refresh, in case earlier excl was stale)
print("\nExcluding NEA TOIs and confirmed planet hosts (10'')...")
toi = tap("SELECT tid, ra, dec FROM toi")
pps = tap("SELECT hostname, ra, dec FROM pscomppars")
print(f"  TOI: {len(toi)}  confirmed: {len(pps)}")
coords = SkyCoord(pred.RA.values*u.deg, pred.DEC.values*u.deg)
known = np.zeros(len(pred), dtype=bool)
if len(toi):
    tc = SkyCoord(toi.ra.values*u.deg, toi.dec.values*u.deg)
    idx, sep, _ = coords.match_to_catalog_sky(tc)
    known |= sep.arcsec < 10
if len(pps):
    pc = SkyCoord(pps.ra.values*u.deg, pps.dec.values*u.deg)
    idx, sep, _ = coords.match_to_catalog_sky(pc)
    known |= sep.arcsec < 10
print(f"  known TOI/host within 10'': {known.sum()}")
pred = pred[~known].reset_index(drop=True)

# Take top N by predicted sectors
pred = pred.sort_values("n_sectors_predicted", ascending=False).reset_index(drop=True)
top = pred.head(N_TARGETS).copy()
print(f"\nTop {N_TARGETS} by predicted sector count: kept {len(top)}")

# Estimate R_star, M_star from APOGEE Teff + logg via simple MS relations
# For main-sequence FGK dwarfs:
#   R ~ (Teff/5778)^0.5 -- rough; better is via Stefan-Boltzmann + log L
# Use logg-based R: g = G*M/R^2, so R^2 = G*M/g
# Or just: R ~ (10^4.44 / 10^logg)^0.5 (ratio of solar gravity to stellar g)
# Combined: R[Rsun] ~ (10^4.44 / 10^logg)^0.5 * (M[Msun])^0.5
# For dwarfs, mass ~ (Teff/5778)^0.6 is a rough estimate
top["Teff_tic"] = top["TEFF"]
top["M_star"] = (top["TEFF"] / 5778) ** 0.6  # rough MS mass
top["R_star"] = np.sqrt(top["M_star"] * (10 ** (4.44 - top["LOGG"])))  # log g balance
top["Tmag"] = np.nan
top["tic_id"] = ""
top["n_lc_sectors_actual"] = top["n_sectors_predicted"].astype(int)
top["apogee_id"] = top["APOGEE_ID"]
top["ra"] = top["RA"]
top["dec"] = top["DEC"]

out_cols = ["apogee_id","ra","dec","tic_id","Tmag","R_star","M_star","Teff_tic",
            "n_lc_sectors_actual","TEFF","LOGG","FE_H","SNR","n_sectors_predicted"]
out = top[out_cols].copy()
out.to_csv("widernet_targets.csv", index=False)
print(f"\nSaved widernet_targets.csv with {len(out)} entries")
print(f"\nSector-count distribution:")
print(out.n_lc_sectors_actual.value_counts().sort_index(ascending=False).head(10).to_string())
print(f"\nMedian Teff: {out.TEFF.median():.0f}  logg: {out.LOGG.median():.2f}  [Fe/H]: {out.FE_H.median():.2f}")
print(f"Median R*: {out.R_star.median():.3f}  M*: {out.M_star.median():.3f}")
print("DONE")

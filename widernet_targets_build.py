"""Build the wider-net target list: bright FGK dwarfs with TESS multi-sector
coverage that aren't already TOI/CTOI hosts.

Strategy (fast path):
  1. Start from APOGEE DR17 FGK dwarfs already in cct_test_field.csv (159k).
  2. Pre-filter by APOGEE SNR > 150 as a brightness proxy.
  3. Use tess_stars2px (TESS team's pure-computational pointing model) to
     compute predicted sector coverage per star -- 90 stars/sec.
  4. Filter to predicted multi-sector (>=3) -- expected ~50% pass.
  5. Cross-match to NEA TOI/confirmed-planet catalogs; exclude existing hosts.
  6. For the top N (by predicted sector count), query MAST to confirm
     actual SPOC LC sector counts and pull TIC stellar params.
  7. Save target list ranked by actual sector count + TIC params.

Output:
  widernet_targets_predicted.csv -- all 76k with predicted sector count
  widernet_targets.csv -- final ranked list ready for BLS
"""
import io, time, requests, warnings, os, re
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from astroquery.mast import Observations, Catalogs
from astropy.coordinates import SkyCoord
import astropy.units as u
from tess_stars2px import tess_stars2px_function_entry as t2p

NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def tap(url, q, retries=3, timeout=300):
    for a in range(retries):
        try:
            r = requests.post(url, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=timeout)
            if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
        except Exception as e: print(f"  WARN {e}")
        time.sleep(2**a)
    return pd.DataFrame()

print("="*78)
print("BUILDING WIDER-NET TARGET LIST (fast tess-point predict + MAST confirm)")
print("="*78)

# 1. Load APOGEE FGK field dwarfs
print("\n[1] Loading APOGEE FGK field pool (159k)...")
field = pd.read_csv("cct_test_field.csv")
print(f"  rows: {len(field)}")

# 2. Pre-filter by SNR as brightness proxy and basic quality
print("\n[2] Pre-filter to bright candidates (APOGEE SNR > 150, FGK params)...")
bright = field[(field.SNR > 150) &
               field.TEFF.between(4500, 7000) &
               (field.LOGG > 3.8) &
               field.FE_H.between(-2, 1) &
               field.RA.between(0, 360) &
               field.DEC.between(-90, 90)].copy().reset_index(drop=True)
print(f"  bright FGK pool: {len(bright)}")

# 3. tess-point predict sector coverage -- CHUNKED with disk checkpointing
print(f"\n[3] tess-point predict sector coverage for {len(bright)} stars (chunked + resumable)...")
t0 = time.time()
CHUNK = 2000
CKPT_DIR = "/tmp/tesspoint_chunks"
os.makedirs(CKPT_DIR, exist_ok=True)

all_sids = []; all_secs = []
for i in range(0, len(bright), CHUNK):
    j = min(i + CHUNK, len(bright))
    ckpt = f"{CKPT_DIR}/chunk_{i:06d}_{j:06d}.npz"
    if os.path.exists(ckpt):
        d = np.load(ckpt)
        out_ids = d["sids"]; out_sectors = d["secs"]
        loaded = " (loaded from checkpoint)"
    else:
        out_ids, _, _, out_sectors, _, _, _, _, _ = t2p(
            np.arange(i, j), bright.RA.values[i:j], bright.DEC.values[i:j])
        np.savez_compressed(ckpt, sids=np.array(out_ids), secs=np.array(out_sectors))
        loaded = ""
    all_sids.extend(np.array(out_ids).tolist())
    all_secs.extend(np.array(out_sectors).tolist())
    dt = time.time() - t0
    rate = j / dt if dt > 0 else 0
    eta = (len(bright) - j) / rate if rate > 0 else 0
    print(f"  chunk {i:>6}-{j:>6}  ({j*100//len(bright):>3}%)  rate={rate:.0f} stars/s  ETA={eta:.0f}s{loaded}")
print(f"  done in {time.time()-t0:.0f}s total")

# Aggregate sectors per star
df_pred = pd.DataFrame(dict(star_idx=all_sids, sector=all_secs))
sec_count = df_pred.groupby("star_idx").sector.nunique().reindex(np.arange(len(bright)), fill_value=0)
sec_list = df_pred.groupby("star_idx").sector.apply(lambda s: sorted(set(s)))
bright["n_sectors_predicted"] = sec_count.values
bright["sectors_predicted_list"] = sec_list.reindex(np.arange(len(bright))).fillna("").values

print(f"  predicted sector distribution:")
print(f"    0 sectors: {(bright.n_sectors_predicted == 0).sum()}")
print(f"    1-2:       {bright.n_sectors_predicted.between(1, 2).sum()}")
print(f"    3-5:       {bright.n_sectors_predicted.between(3, 5).sum()}")
print(f"    6-10:      {bright.n_sectors_predicted.between(6, 10).sum()}")
print(f"    11-20:     {bright.n_sectors_predicted.between(11, 20).sum()}")
print(f"    21+:       {(bright.n_sectors_predicted >= 21).sum()}")

# 4. Filter to predicted multi-sector
print(f"\n[4] Filter to predicted >= 3 sectors...")
ms = bright[bright.n_sectors_predicted >= 3].copy().reset_index(drop=True)
print(f"  candidates: {len(ms)}")

# Save the predicted list
ms.to_csv("widernet_targets_predicted.csv", index=False)

# 5. NEA cross-checks: TOI, confirmed planets
print(f"\n[5] Excluding known NEA TOIs and confirmed planet hosts...")
toi = tap(NEA, "SELECT tid, ra, dec FROM toi")
pps = tap(NEA, "SELECT hostname, ra, dec FROM pscomppars")
print(f"  TOI rows: {len(toi)}, confirmed planet rows: {len(pps)}")
ms_coords = SkyCoord(ms.RA.values*u.deg, ms.DEC.values*u.deg)
known = np.zeros(len(ms), dtype=bool)
if len(toi):
    tc = SkyCoord(toi.ra.values*u.deg, toi.dec.values*u.deg)
    idx, sep, _ = ms_coords.match_to_catalog_sky(tc)
    known |= (sep.arcsec < 10.0)
if len(pps):
    pc = SkyCoord(pps.ra.values*u.deg, pps.dec.values*u.deg)
    idx, sep, _ = ms_coords.match_to_catalog_sky(pc)
    known |= (sep.arcsec < 10.0)
print(f"  known TOI/host matches within 10'' of {len(ms)}: {known.sum()}")
ms = ms[~known].reset_index(drop=True)
print(f"  candidates after exclusion: {len(ms)}")

# 6. Rank by predicted sector count and select top N for actual MAST confirmation + BLS
ms = ms.sort_values("n_sectors_predicted", ascending=False).reset_index(drop=True)
N_TOP = min(len(ms), 500)   # reduced from 1500: keep wall time tractable
top = ms.head(N_TOP).copy()
print(f"\n[6] Top {N_TOP} candidates by predicted sector count -- confirm with MAST...")

# Resume from existing widernet_targets_full.csv if present
RES_CSV = "widernet_targets_full.csv"
done_idx = set()
results = []
if os.path.exists(RES_CSV):
    prev = pd.read_csv(RES_CSV)
    results = prev.to_dict("records")
    done_idx = set(prev["apogee_id"].astype(str).tolist())
    print(f"  resuming -- {len(done_idx)} already done")

# Query MAST for actual SPOC LC sector counts + TIC params
print(f"  (this loop hits MAST 2x per target -- pacing ~3-10 s/star)")
t_start = time.time()
for i, r in top.iterrows():
    if str(r.APOGEE_ID) in done_idx:
        continue
    ra, dec = float(r.RA), float(r.DEC)
    try:
        obs = Observations.query_region(f"{ra} {dec}", radius="20 arcsec")
        ts = obs[(obs["obs_collection"] == "TESS") & (obs["dataproduct_type"] == "timeseries")]
        if len(ts) == 0:
            n_lc_sectors = 0; sectors_actual = []
        else:
            prods = Observations.get_product_list(ts)
            lc_prods = prods[prods["productSubGroupDescription"] == "LC"]
            secs = set()
            for fn in lc_prods["productFilename"]:
                m = re.search(r"-s(\d+)-", str(fn))
                if m: secs.add(int(m.group(1)))
            n_lc_sectors = len(secs)
            sectors_actual = sorted(secs)
    except Exception as e:
        n_lc_sectors = -1; sectors_actual = []

    try:
        tic = Catalogs.query_region(f"{ra} {dec}", radius="5 arcsec", catalog="TIC")
    except Exception:
        tic = None
    if tic is not None and len(tic) > 0:
        tic = tic[np.argsort(tic["dstArcSec"])]
        row = tic[0]
        tic_id = str(row["ID"])
        Tmag = float(row["Tmag"]) if row["Tmag"] else np.nan
        R_star = float(row["rad"]) if row["rad"] else np.nan
        M_star = float(row["mass"]) if row["mass"] else np.nan
        Teff_tic = float(row["Teff"]) if row["Teff"] else np.nan
        sep_arcsec = float(row["dstArcSec"])
    else:
        tic_id = ""; Tmag = np.nan; R_star = np.nan; M_star = np.nan; Teff_tic = np.nan; sep_arcsec = np.nan

    results.append(dict(
        apogee_id=str(r.APOGEE_ID), ra=ra, dec=dec,
        teff_ap=float(r.TEFF), logg_ap=float(r.LOGG), feh_ap=float(r.FE_H),
        snr_ap=float(r.SNR),
        n_sectors_predicted=int(r.n_sectors_predicted),
        n_lc_sectors_actual=int(n_lc_sectors),
        sectors_actual="|".join(str(s) for s in sectors_actual),
        tic_id=tic_id, Tmag=Tmag, R_star=R_star, M_star=M_star, Teff_tic=Teff_tic,
        tic_sep_arcsec=sep_arcsec,
    ))
    # Checkpoint every 25 stars so a restart can resume cheaply
    if (i+1) % 25 == 0 or (i+1) == N_TOP:
        pd.DataFrame(results).to_csv(RES_CSV, index=False)
        dt = time.time() - t_start
        rate = len(results) / dt if dt > 0 else 0
        eta = (N_TOP - (i+1)) / rate if rate > 0 else 0
        n_multi = sum(1 for x in results if x.get("n_lc_sectors_actual", 0) >= 3)
        print(f"  [{i+1:>5}/{N_TOP}]  multi_actual={n_multi}  rate={rate:.2f}/s  ETA={eta/60:.1f}m", flush=True)

R = pd.DataFrame(results)

# Filter and save
final = R[(R.n_lc_sectors_actual >= 3) &
          (R.Tmag.isna() | (R.Tmag <= 11.5))].sort_values(
              "n_lc_sectors_actual", ascending=False).reset_index(drop=True)
final.to_csv("widernet_targets.csv", index=False)

# Summary
print(f"\n{'='*78}\nFINAL WIDER-NET TARGET LIST\n{'='*78}")
print(f"  bright FGK pool:                          {len(bright)}")
print(f"  predicted multi-sector (>=3):             {(bright.n_sectors_predicted >= 3).sum()}")
print(f"  after known-TOI/host exclusion:           {len(ms)}")
print(f"  MAST-confirmed with actual >= 3 sectors:  {len(final)}")
print(f"  long-baseline (>= 10 actual sectors):     {(final.n_lc_sectors_actual >= 10).sum()}")
print(f"  long-baseline (>= 20 actual sectors):     {(final.n_lc_sectors_actual >= 20).sum()}")
print(f"\n  sector-count distribution (top):")
print(final.n_lc_sectors_actual.value_counts().sort_index(ascending=False).head(15).to_string())
print(f"\n  saved widernet_targets.csv  ({len(final)} stars)")
print("\nDONE")

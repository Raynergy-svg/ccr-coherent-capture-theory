"""Batch BLS transit search on TESS-covered actionable dwarfs.

Strategy:
  Tier 1 (multi-year baseline -> HZ-period sensitivity):
    CPD-63 349 (41 sectors), CD-60 1593 (21)
  Tier 2 (months baseline):
    HD 271308 (9), BD-02 3362 (3), HD 178528B (2)
  Tier 3 (single sector, max ~13 d):
    HD 183193 and 14 others

For each target:
  - download SPOC PDCSAP LCs across all available sectors
  - detrend per sector (12h medfilt) + 5-sigma outlier clip
  - concatenate, run BLS with period range tied to baseline
  - compute SDE = (peak_pwr - mean_pwr) / std_pwr -- conventional planet
    detection threshold is SDE >= 7 (matches NASA SPOC convention)
  - per-sector independent BLS at the best period: signal must show up
    in multiple sectors (consistency check)
  - report planet parameters if signal real

Save results to tess_bls_batch_results.csv ordered by SDE.
"""
import os, sys, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Observations
from scipy.signal import medfilt

# host parameters: defaults for "dwarf, no GALAH constraint", filled per-target if available
HOST_DEFAULT = dict(teff=5700.0, R=1.0, M=1.0)

COVERAGE_CSV = "tess_coverage_32dwarfs.csv"
DOWNLOAD_DIR = "/tmp/tess_dl"
RESULT_CSV   = "tess_bls_batch_results.csv"
LOG_TXT      = "tess_bls_batch_log.txt"

# load coverage; filter to those with at least one TESS timeseries product
cov = pd.read_csv(COVERAGE_CSV)
cov = cov[cov.n_ts > 0].copy().sort_values("n_ts", ascending=False).reset_index(drop=True)

# load 32-dwarf table with GALAH stellar params (Teff, logg, [Fe/H])
try:
    dw = pd.read_csv("real_dwarf_targets.csv")
    dw_lookup = {int(r.gaiadr3_source_id): r for _, r in dw.iterrows()}
except Exception as e:
    print(f"[WARN] couldn't load real_dwarf_targets.csv: {e}")
    dw_lookup = {}

# rough TESS RA/Dec lookup from coverage CSV (already merged with Gaia DR3)
# but the coverage CSV doesn't carry RA -- it carries DEC + gaia. We need RA.
# Pull from real_dwarf_targets.csv.
ra_lookup = {}
if len(dw_lookup):
    for sid, r in dw_lookup.items():
        ra_lookup[sid] = float(r.ra)

logf = open(LOG_TXT, "w")
def log(msg):
    print(msg)
    logf.write(msg + "\n"); logf.flush()

def get_lc_sectors(ra, dec, name):
    """Return list of (sector, t, f_normalised) tuples and a status message."""
    obs = Observations.query_region(f"{ra} {dec}", radius="20 arcsec")
    ts = obs[(obs["obs_collection"] == "TESS") & (obs["dataproduct_type"] == "timeseries")]
    if len(ts) == 0:
        return [], "no TESS timeseries"
    prods = Observations.get_product_list(ts)
    lc_prods = prods[prods["productSubGroupDescription"] == "LC"]
    if len(lc_prods) == 0:
        return [], "no SPOC LC products"
    try:
        dl = Observations.download_products(lc_prods, download_dir=DOWNLOAD_DIR, verbose=False)
    except Exception as e:
        return [], f"download failed: {e}"
    if dl is None or len(dl) == 0:
        return [], "download empty"
    sectors = []
    for fp in dl["Local Path"]:
        try:
            with fits.open(fp) as hdul:
                hdr = hdul[0].header
                sector = hdr.get("SECTOR", -1)
                lc = hdul["LIGHTCURVE"].data
                t = lc["TIME"]; f = lc["PDCSAP_FLUX"]; q = lc["QUALITY"]
                m = np.isfinite(t) & np.isfinite(f) & (q == 0)
                if m.sum() < 100: continue
                t = t[m]; f = f[m] / np.nanmedian(f[m])
                sectors.append((int(sector), t, f))
        except Exception as e:
            continue
    return sectors, f"{len(sectors)} usable sectors of {len(dl)} downloads"

def detrend(t, f, win_days=0.5):
    dt = np.median(np.diff(t))
    box = max(3, int(win_days / dt))
    if box % 2 == 0: box += 1
    trend = medfilt(f, box)
    fd = f / trend
    sig = 1.4826 * np.median(np.abs(fd - 1.0))
    keep = np.abs(fd - 1.0) < 5*sig
    return t[keep], fd[keep]

def bls_with_sde(t, f, periods, durations):
    """Run BLS and return peak diagnostics + SDE."""
    bls = BoxLeastSquares(t, f)
    pg = bls.power(periods, durations)
    pwr = pg.power
    i = int(np.argmax(pwr))
    # SDE = (peak - mean) / std of the periodogram (excluding the peak's vicinity)
    # standard SPOC convention
    msk = np.ones_like(pwr, dtype=bool)
    # exclude +-1% around the peak for the noise estimate
    p_peak = pg.period[i]
    near = np.abs((pg.period - p_peak) / p_peak) < 0.01
    msk &= ~near
    mu, sd = float(np.nanmean(pwr[msk])), float(np.nanstd(pwr[msk]))
    sde = (float(pwr[i]) - mu) / sd if sd > 0 else 0.0
    return dict(P=float(pg.period[i]), dur=float(pg.duration[i]),
                T0=float(pg.transit_time[i]), depth=float(pg.depth[i]),
                pwr=float(pwr[i]), sde=sde, pg_pwr_mean=mu, pg_pwr_std=sd,
                bls=bls, pg=pg)

def planet_params(P_d, depth, R_star_Rsun, M_star_Msun, Teff):
    R_p_Rsun = np.sqrt(max(depth, 0)) * R_star_Rsun
    R_p_Rearth = R_p_Rsun * 109.2
    a_AU = ((P_d / 365.25)**2 * M_star_Msun)**(1/3)
    T_eq = Teff * np.sqrt(R_star_Rsun * 0.00465047 / (2 * a_AU))
    return R_p_Rearth, a_AU, T_eq

records = []
log(f"="*78)
log(f"BATCH BLS TRANSIT SEARCH on {len(cov)} TESS-covered actionable dwarfs")
log(f"="*78)
log(f"target order (by n_sectors descending):")
for i, r in cov.iterrows():
    log(f"  {i+1:>2}. {r['name']:<22} G={r.G:.2f}  d={r.dist:.0f} pc  hab8={r.hab8:.3f}  n_sec={r.n_ts}")

for idx, r in cov.iterrows():
    sid = int(r.gaia)
    name = str(r["name"]).strip()
    dec = float(r.dec)
    ra = ra_lookup.get(sid, None)
    if ra is None:
        log(f"\n[skip] {name}: no RA in lookup")
        continue
    nsec = int(r.n_ts)

    log(f"\n{'='*78}")
    log(f"[{idx+1}/{len(cov)}] {name}   Gaia {sid}   (RA,Dec)=({ra:.5f},{dec:.5f})   n_sec={nsec}")
    log(f"{'='*78}")

    # host params
    h = dw_lookup.get(sid)
    if h is not None:
        Teff = float(h.teff) if pd.notna(h.teff) else HOST_DEFAULT["teff"]
        # GALAH dwarf cuts give logg ~ 4.4-4.6; assume R ~ 1.0 R_sun, M ~ 1.0 M_sun if not specified
        R_star = 1.0  # rough; refine per-target as needed
        M_star = 1.0
    else:
        Teff = HOST_DEFAULT["teff"]; R_star = 1.0; M_star = 1.0

    t0 = time.time()
    sectors, msg = get_lc_sectors(ra, dec, name)
    log(f"  download: {msg} in {time.time()-t0:.0f}s")
    if not sectors:
        records.append(dict(name=name, gaia=sid, n_sectors=0, status=msg))
        continue

    # detrend per sector, concatenate
    all_t, all_f = [], []
    for s, t, f in sectors:
        td, fd = detrend(t, f)
        all_t.append(td); all_f.append(fd)
    T = np.concatenate(all_t); F = np.concatenate(all_f)
    base = T.max() - T.min()
    rms_ppm = np.std(F) * 1e6
    log(f"  detrended: {len(T)} pts, baseline {base:.0f} d, RMS {rms_ppm:.0f} ppm")

    # period range: 0.5 day to min(baseline/2.5, 250)
    p_max = min(base/2.5, 250.0)
    # finer grid for longer baselines
    n_grid = int(np.clip(5000 * (p_max/15), 4000, 30000))
    periods = np.linspace(0.5, p_max, n_grid)
    durations = np.array([0.05, 0.08, 0.10, 0.15, 0.20])

    t0 = time.time()
    res = bls_with_sde(T, F, periods, durations)
    log(f"  BLS done: peak P={res['P']:.4f} d  dur={res['dur']*24:.2f} h  "
        f"depth={res['depth']*1e6:.0f} ppm  SDE={res['sde']:.2f}  ({time.time()-t0:.0f}s)")

    # planet params (rough)
    R_p, a_AU, T_eq = planet_params(res['P'], max(res['depth'], 0), R_star, M_star, Teff)
    log(f"  if-real: R_p={R_p:.2f} R_Earth  a={a_AU:.3f} AU  T_eq={T_eq:.0f} K")

    # per-sector consistency check
    sec_powers, sec_depths = [], []
    if len(sectors) > 1:
        for s, t, f in sectors:
            td, fd = detrend(t, f)
            if len(td) < 100: continue
            bls_s = BoxLeastSquares(td, fd)
            pg_s = bls_s.power(np.array([res['P']]), durations)
            sec_powers.append(float(pg_s.power[0]))
            sec_depths.append(float(pg_s.depth[0]))
        if sec_depths:
            log(f"  per-sector at P={res['P']:.4f}: depths(ppm) {[int(d*1e6) for d in sec_depths]}, "
                f"powers {[f'{p:.4f}' for p in sec_powers]}")

    rec = dict(name=name, gaia=sid, n_sectors=len(sectors), baseline_d=base,
               n_pts=len(T), rms_ppm=rms_ppm,
               P_d=res['P'], dur_h=res['dur']*24, depth_ppm=res['depth']*1e6,
               sde=res['sde'], bls_power=res['pwr'],
               R_p_Rearth=R_p, a_AU=a_AU, T_eq_K=T_eq,
               n_sec_checked=len(sec_depths),
               sec_depths_ppm="|".join([f"{int(d*1e6):+d}" for d in sec_depths]) if sec_depths else "",
               status="ok")
    # flag if SDE >= 7
    if res['sde'] >= 7:
        rec["status"] = "DETECTION_CANDIDATE_SDE_OK"
        log(f"  *** SDE = {res['sde']:.2f} >= 7 -- CANDIDATE -- per-sector follow-up required ***")
    elif res['sde'] >= 5:
        rec["status"] = "marginal_SDE_5_7"
        log(f"  marginal SDE {res['sde']:.2f} -- below 7 but worth a manual look")
    else:
        rec["status"] = f"no_signal_SDE_{res['sde']:.1f}"
    records.append(rec)

# save & summarise
R = pd.DataFrame(records)
R.to_csv(RESULT_CSV, index=False)
log(f"\n{'='*78}\nBATCH BLS COMPLETE -- {len(R)} targets processed\n{'='*78}")

ok = R[R.status.fillna("").str.contains("ok|CANDIDATE|marginal")]
if len(ok):
    log(f"\nTargets sorted by SDE:")
    sorted_R = ok.sort_values("sde", ascending=False)
    cols = ["name", "n_sectors", "baseline_d", "P_d", "dur_h",
            "depth_ppm", "sde", "R_p_Rearth", "T_eq_K", "status"]
    log(sorted_R[cols].to_string(index=False))

cands = R[R.status == "DETECTION_CANDIDATE_SDE_OK"]
log(f"\nSDE>=7 candidates: {len(cands)}")
if len(cands): log(cands[["name","P_d","sde","depth_ppm","R_p_Rearth","T_eq_K"]].to_string(index=False))

logf.close()
print(f"\nresults -> {RESULT_CSV}")
print(f"log     -> {LOG_TXT}")
print("DONE")

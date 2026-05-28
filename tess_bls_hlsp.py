"""BLS transit search using HLSP TESS-SPOC FFI lightcurves + K2 lightcurves
for the 12 actionable dwarfs without standard SPOC LC products.

Priorities:
  HD 271200      -- 41 QLP + 32 TESS-SPOC FFI = >70 LCs (CVZ-like!)
  BD-03 3321     -- TESS-SPOC FFI + K2 (EVEREST/K2SC/K2SFF) -- #1 hab8=0.997
  CD-30 7056     -- 3 TESS-SPOC FFI + 5 QLP sectors -- #2 hab8=0.995
  TYC 5232-208-1 -- K2 (EVEREST/K2SC/K2SFF) + TESS-SPOC -- small R=0.843
  HD 122625      -- K2 (EVEREST/K2SC/K2SFF) -- hab8=0.991
  + 7 others with 1-2 HLSP products
"""
import os, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Observations
from scipy.signal import medfilt

cov = pd.read_csv("tess_coverage_32dwarfs.csv")
dw  = pd.read_csv("real_dwarf_targets.csv")
tic = pd.read_csv("tic_params_32dwarfs.csv")
ra_lookup    = {int(r.gaiadr3_source_id): float(r.ra) for _, r in dw.iterrows()}
tic_lookup   = {int(r.gaia): r for _, r in tic.iterrows()}

# the 12 dwarfs with no SPOC LC but with HLSP
no_lc = cov[(cov.n_ts == 0) & (cov.n_ffi > 0)].copy().sort_values("hab8", ascending=False).reset_index(drop=True)
print(f"{len(no_lc)} HLSP/K2-only actionable dwarfs to search\n")

# preferred HLSP order: TESS-SPOC (FFI) is best; QLP is good; K2 EVEREST/K2SFF/K2SC are equivalent
PREFERRED = ["TESS-SPOC", "QLP", "EVEREST", "K2SFF", "K2SC", "TGLC", "GSFC-ELEANOR-LITE"]

def get_hlsp_lcs(ra, dec, name):
    """Return dict of provenance -> list of (sector/campaign, t, f) tuples."""
    obs = Observations.query_region(f"{ra} {dec}", radius="20 arcsec")
    ts = obs[(obs["dataproduct_type"] == "timeseries") & (obs["obs_collection"] == "HLSP")]
    if len(ts) == 0: return {}, "no HLSP timeseries"
    # download products in batches per provenance to avoid huge downloads
    by_prov = {}
    for prov in PREFERRED:
        rows = ts[ts["provenance_name"] == prov]
        if len(rows) == 0: continue
        try:
            prods = Observations.get_product_list(rows)
        except Exception as e:
            print(f"    {prov}: product list error {e}")
            continue
        # filter to LC fits
        names = np.array([str(n) for n in prods["productFilename"]])
        keep = np.array(["_lc.fits" in n or "llc.fits" in n or "lc-tess_" in n or "_lc-" in n for n in names])
        lc_prods = prods[keep]
        if len(lc_prods) == 0:
            # try anything with .fits
            keep2 = np.array([n.endswith(".fits") for n in names])
            lc_prods = prods[keep2]
        if len(lc_prods) == 0:
            print(f"    {prov}: no LC-like products in {len(prods)} files")
            continue
        try:
            dl = Observations.download_products(lc_prods, download_dir="/tmp/tess_dl", verbose=False)
        except Exception as e:
            print(f"    {prov}: dl error {e}")
            continue
        if dl is None or len(dl) == 0: continue
        files = []
        for fp in dl["Local Path"]:
            try:
                with fits.open(fp) as hdul:
                    # try LIGHTCURVE extension first
                    data = None
                    for ext in ["LIGHTCURVE", 1]:
                        try:
                            data = hdul[ext].data
                            break
                        except: pass
                    if data is None: continue
                    # find time and flux columns
                    cols = [c.lower() for c in data.names]
                    tcol = next((c for c in ["time","btjd","tmid"] if c in cols), None)
                    # prefer detrended flux from HLSP
                    fcandidates = ["pdcsap_flux","flux","sap_flux","kspsap_flux","det_flux",
                                   "fcor","fraw"]
                    fcol = next((c for c in fcandidates if c in cols), None)
                    if not tcol or not fcol: continue
                    t = np.array(data[data.names[cols.index(tcol)]])
                    f = np.array(data[data.names[cols.index(fcol)]])
                    q = np.zeros_like(t, dtype=int)
                    if "quality" in cols:
                        q = np.array(data[data.names[cols.index("quality")]])
                    m = np.isfinite(t) & np.isfinite(f) & (q == 0)
                    if m.sum() < 100: continue
                    t, f = t[m], f[m]/np.nanmedian(f[m])
                    # sector/campaign from header
                    h = hdul[0].header
                    seg = h.get("SECTOR", h.get("CAMPAIGN", h.get("CAMPAIGN_ID", "?")))
                    files.append((seg, t, f, fp))
            except Exception as e:
                pass
        if files: by_prov[prov] = files
    return by_prov, f"got {sum(len(v) for v in by_prov.values())} LCs from {len(by_prov)} pipelines"

def detrend(t, f, win_days=0.5):
    dt = np.median(np.diff(t))
    if dt <= 0: dt = 0.02  # K2 long cadence
    box = max(3, int(win_days/dt))
    if box % 2 == 0: box += 1
    trend = medfilt(f, box)
    fd = f / trend
    sig = 1.4826 * np.median(np.abs(fd - 1.0))
    keep = np.abs(fd - 1.0) < 5*sig
    return t[keep], fd[keep]

def bls_with_sde(t, f, periods, durations):
    bls = BoxLeastSquares(t, f)
    pg = bls.power(periods, durations)
    pwr = pg.power
    i = int(np.argmax(pwr))
    p_peak = pg.period[i]
    near = np.abs((pg.period - p_peak)/p_peak) < 0.01
    mu = float(np.nanmean(pwr[~near])); sd = float(np.nanstd(pwr[~near]))
    sde = (float(pwr[i]) - mu) / sd if sd > 0 else 0.0
    return dict(P=float(pg.period[i]), dur=float(pg.duration[i]),
                T0=float(pg.transit_time[i]), depth=float(pg.depth[i]),
                pwr=float(pwr[i]), sde=sde)

records = []
for idx, r in no_lc.iterrows():
    sid = int(r.gaia); name = r["name"].strip(); dec = float(r.dec)
    ra = ra_lookup.get(sid)
    if ra is None: continue
    th = tic_lookup.get(sid)
    R_star = float(th.R_star) if th is not None and pd.notna(th.R_star) else 1.0
    M_star = float(th.M_star) if th is not None and pd.notna(th.M_star) else 1.0
    Teff   = float(th.Teff_tic) if th is not None and pd.notna(th.Teff_tic) else 5700.0
    print(f"\n{'='*78}\n[{idx+1}/{len(no_lc)}] {name}  Gaia {sid}  hab8={r.hab8:.3f}  R*={R_star:.3f} M*={M_star:.3f} Teff={Teff:.0f}\n{'='*78}")
    t0 = time.time()
    lcs_by_prov, msg = get_hlsp_lcs(ra, dec, name)
    print(f"  download: {msg} in {time.time()-t0:.0f}s")
    if not lcs_by_prov:
        records.append(dict(name=name, gaia=sid, hab8=r.hab8, status="no_LCs"))
        continue
    # combine all LCs from preferred pipeline (TESS-SPOC if available, else best avail)
    best_prov = next((p for p in PREFERRED if p in lcs_by_prov), None)
    if best_prov is None: continue
    print(f"  using pipeline: {best_prov} ({len(lcs_by_prov[best_prov])} LCs)")
    files = lcs_by_prov[best_prov]
    all_t, all_f = [], []
    for seg, t, f, fp in files:
        td, fd = detrend(t, f)
        if len(td) < 100: continue
        all_t.append(td); all_f.append(fd)
    if not all_t:
        records.append(dict(name=name, gaia=sid, hab8=r.hab8, status="all_LCs_too_short"))
        continue
    T = np.concatenate(all_t); F = np.concatenate(all_f)
    base = T.max() - T.min(); rms = np.std(F)*1e6
    print(f"  detrended: {len(T)} pts, baseline {base:.0f} d, RMS {rms:.0f} ppm")
    p_max = min(base/2.5, 200.0)
    n_grid = int(np.clip(5000 * (p_max/15), 4000, 25000))
    periods = np.linspace(0.5, p_max, n_grid)
    durations = np.array([0.05, 0.08, 0.10, 0.15, 0.20])
    t0 = time.time()
    res = bls_with_sde(T, F, periods, durations)
    print(f"  BLS: P={res['P']:.4f} d  dur={res['dur']*24:.2f} h  depth={res['depth']*1e6:.0f} ppm  SDE={res['sde']:.2f}  ({time.time()-t0:.0f}s)")
    # planet params via TIC
    R_p_Rsun = np.sqrt(max(res['depth'],0)) * R_star
    R_p_Rearth = R_p_Rsun * 109.2
    a_AU = ((res['P']/365.25)**2 * M_star)**(1/3)
    T_eq = Teff * np.sqrt(R_star*0.00465047/(2*a_AU))
    print(f"  if-real: R_p={R_p_Rearth:.2f} R_Earth  a={a_AU:.3f} AU  T_eq={T_eq:.0f} K")
    status = "ok"
    if res['sde'] >= 7:
        status = "DETECTION_CANDIDATE_SDE_OK"
        print(f"  *** SDE >= 7 -- candidate ***")
    elif res['sde'] >= 5:
        status = "marginal"
    records.append(dict(
        name=name, gaia=sid, hab8=float(r.hab8), pipeline=best_prov,
        n_segments=len(files), n_pts=len(T), baseline_d=float(base), rms_ppm=float(rms),
        P_d=res['P'], dur_h=res['dur']*24, depth_ppm=res['depth']*1e6, sde=res['sde'],
        R_star=R_star, M_star=M_star, Teff=Teff,
        R_p_Rearth=R_p_Rearth, a_AU=a_AU, T_eq_K=T_eq,
        status=status
    ))

R = pd.DataFrame(records)
R.to_csv("tess_bls_hlsp_results.csv", index=False)
print(f"\n{'='*78}\nHLSP BLS COMPLETE -- {len(R)} targets\n{'='*78}")
ok = R[R.status.fillna("").str.contains("ok|CANDIDATE|marginal")]
if len(ok):
    print(ok.sort_values("sde", ascending=False)[
        ["name","hab8","pipeline","n_segments","baseline_d","P_d","depth_ppm","sde","R_p_Rearth","T_eq_K","status"]
    ].to_string(index=False))
cands = R[R.status == "DETECTION_CANDIDATE_SDE_OK"]
print(f"\nSDE>=7 candidates: {len(cands)}")
if len(cands): print(cands[["name","P_d","sde","depth_ppm","R_p_Rearth","T_eq_K"]].to_string(index=False))
print("DONE")

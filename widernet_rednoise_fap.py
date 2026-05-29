"""Red-noise false-alarm-probability (FAP) test for the SURVIVED candidates.

The EB screen rules out eclipsing binaries and blends. It does NOT test the
dominant false-positive mode for long-period, modest-SDE detections:
correlated (red) instrumental/stellar noise that happens to align into a
BLS peak. The clustering of SURVIVED candidates at P ~ 170-250 d is exactly
the signature you'd expect if a chunk of them are red-noise alignments
rather than real transits.

Method -- block bootstrap (preserves red noise, destroys period coherence):
  1. Load the detrended, concatenated lightcurve (same as the pipeline/EB screen)
  2. Bin to 30 min (long-period transits have multi-hour durations; binning
     costs nothing and makes the bootstrap tractable)
  3. Recompute the OBSERVED SDE at the candidate period on the binned data,
     using the same BLS grid the null will use (apples-to-apples)
  4. N times: split the binned series into contiguous time-blocks of
     ~max(1 d, 3*duration), randomly permute the block ORDER, reassign the
     permuted flux onto the original time sampling. This keeps within-block
     correlated noise intact but scrambles any coherence on timescales longer
     than a block -- which is exactly what a real P>>block transit needs.
  5. Run the identical full-grid BLS on each scrambled series; record the
     MAX SDE anywhere in the grid (look-elsewhere corrected)
  6. FAP = fraction of null trials with max-SDE >= observed SDE

Verdict:
  FAP < 0.05         SURVIVES_REDNOISE   (noise rarely reaches this SDE)
  0.05 <= FAP < 0.20 MARGINAL
  FAP >= 0.20        LIKELY_RED_NOISE    (noise reaches this SDE often)

Scope: this test is meaningful only when the period is much longer than the
block size (so block-permutation actually destroys the signal). For
P < ~10*block (the short-period candidates) it is reported as N/A -- their
reality rests on centroid + EB screen, and red noise does not produce
SDE>30 at P~4 d anyway.
"""
import os, warnings, time, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Observations
from scipy.signal import medfilt
import multiprocessing as mp

N_BOOT      = 100
BIN_MINUTES = 30.0
RNG_SEED    = 20260528
N_WORKERS   = 4
GRID_CAP    = 5000          # trimmed from 15000: ample for a max-SDE FAP measure
RESULTS_IN  = "widernet_bls_results.csv"
RESULTS_OUT = "widernet_rednoise_fap.csv"

# module globals populated per-candidate before the worker pool is forked
_G = {}

def _boot_worker(k):
    """One bootstrap iteration: block-permute the binned LC, full-grid BLS,
    return max SDE. Reads the candidate arrays from module globals (inherited
    by fork on Linux)."""
    rng = np.random.default_rng(RNG_SEED + 1 + k)
    _, Fp = block_permute(_G["Tb"], _G["Fb"], _G["block_days"], rng)
    s, _ = bls_max_sde(_G["Tb"], Fp, _G["periods"], _G["durations"])
    return s

import glob as _glob
CACHE_ROOT = "/tmp/tess_dl/mastDownload/TESS"

def load_lcs(ra, dec):
    """Cache-direct loader. The container clock is skewed so the S3 SSL
    handshake in Observations.download_products fails even for cached files
    (it does a HEAD request to stpubdata.s3.amazonaws.com). The MAST query
    endpoint (different host) still works, so we get the product list and
    then read the FITS straight off disk, skipping any that aren't cached."""
    try:
        obs = Observations.query_region(f"{ra} {dec}", radius="20 arcsec")
        ts = obs[(obs["obs_collection"]=="TESS") & (obs["dataproduct_type"]=="timeseries")]
        if len(ts) == 0: return [], "no_ts"
        prods = Observations.get_product_list(ts)
        lc = prods[prods["productSubGroupDescription"]=="LC"]
        if len(lc) == 0: return [], "no_lc_products"
        filenames = [str(x) for x in lc["productFilename"]]
    except Exception as e:
        return [], f"query_fail:{type(e).__name__}"
    out = []; n_cached = 0; n_missing = 0
    for fn in filenames:
        hits = _glob.glob(f"{CACHE_ROOT}/**/{fn}", recursive=True)
        if not hits:
            n_missing += 1; continue
        n_cached += 1
        try:
            with fits.open(hits[0]) as h:
                d = h["LIGHTCURVE"].data
                t = d["TIME"]; f = d["PDCSAP_FLUX"]; q = d["QUALITY"]
                m = np.isfinite(t)&np.isfinite(f)&(q==0)
                if m.sum() < 100: continue
                out.append((t[m], f[m]/np.nanmedian(f[m])))
        except Exception: continue
    return out, f"cached={n_cached}/{len(filenames)} missing={n_missing}"

def detrend_concat(lcs):
    allt, allf = [], []
    for t, f in lcs:
        dt = np.median(np.diff(t));  dt = dt if dt>0 else 0.02
        box = max(3, int(0.5/dt));  box = box+1 if box%2==0 else box
        fd = f/medfilt(f, box)
        s = 1.4826*np.median(np.abs(fd-1)); keep = np.abs(fd-1) < 5*s
        allt.append(t[keep]); allf.append(fd[keep])
    if not allt: return np.array([]), np.array([])
    T = np.concatenate(allt); F = np.concatenate(allf)
    order = np.argsort(T)
    return T[order], F[order]

def bin_lc(T, F, bin_min):
    if len(T)==0: return T, F
    bw = bin_min/60.0/24.0
    edges = np.arange(T.min(), T.max()+bw, bw)
    idx = np.digitize(T, edges)
    tb, fb = [], []
    for i in np.unique(idx):
        m = idx==i
        if m.sum()==0: continue
        tb.append(np.median(T[m])); fb.append(np.median(F[m]))
    return np.array(tb), np.array(fb)

def bls_max_sde(T, F, periods, durations):
    bls = BoxLeastSquares(T, F)
    pg = bls.power(periods, durations)
    pwr = pg.power
    i = int(np.argmax(pwr))
    Ppk = pg.period[i]
    near = np.abs((pg.period-Ppk)/Ppk) < 0.01
    mu = np.nanmean(pwr[~near]); sd = np.nanstd(pwr[~near])
    sde = (pwr[i]-mu)/sd if sd>0 else 0.0
    return float(sde), float(Ppk)

def sde_at_period(T, F, P, durations):
    bls = BoxLeastSquares(T, F)
    pers = np.linspace(P*0.97, P*1.03, 1500)
    pg = bls.power(pers, durations)
    pwr = pg.power; i = int(np.argmax(pwr))
    near = np.abs((pg.period-pg.period[i])/pg.period[i]) < 0.01
    mu = np.nanmean(pwr[~near]); sd = np.nanstd(pwr[~near])
    return (float(pwr[i]-mu)/sd) if sd>0 else 0.0

def block_permute(T, F, block_days, rng):
    """Split into contiguous time-blocks, permute block order, reassign flux
    onto the original time sampling. Preserves within-block red noise,
    destroys >block period coherence."""
    blocks = []
    start = 0
    t0 = T[0]
    for i in range(len(T)):
        if T[i] - t0 >= block_days:
            blocks.append((start, i)); start = i; t0 = T[i]
    blocks.append((start, len(T)))
    order = rng.permutation(len(blocks))
    f_chunks = [F[a:b] for (a,b) in blocks]
    F_perm = np.concatenate([f_chunks[k] for k in order])
    # lengths match -> assign onto original time array
    return T, F_perm

def run_fap(name, ra, dec, P, dur_d, baseline_hint, n_boot=N_BOOT):
    lcs, lc_msg = load_lcs(ra, dec)
    if not lcs:
        return dict(name=name, status=f"no_lc ({lc_msg})")
    T, F = detrend_concat(lcs)
    Tb, Fb = bin_lc(T, F, BIN_MINUTES)
    if len(Tb) < 200:
        return dict(name=name, status="too_few_bins", n_bins=len(Tb))
    baseline = Tb.max()-Tb.min()
    block_days = max(1.0, 3.0*dur_d)
    # scope check: need P >> block to destroy coherence
    scope_ok = P > 10.0*block_days
    pmax = min(baseline/2.5, 300.0)
    n_grid = int(np.clip(6000*(pmax/15), 3000, GRID_CAP))
    periods = np.linspace(0.5, pmax, n_grid)
    durations = np.array([0.05, 0.08, 0.10, 0.15, 0.20, 0.30])
    sde_obs = sde_at_period(Tb, Fb, P, durations)
    t0 = time.time()
    # populate globals for forked workers, then run bootstraps in parallel
    _G["Tb"], _G["Fb"] = Tb, Fb
    _G["periods"], _G["durations"] = periods, durations
    _G["block_days"] = block_days
    with mp.Pool(N_WORKERS) as pool:
        null_max = np.array(pool.map(_boot_worker, range(n_boot)))
    fap = float((null_max >= sde_obs).mean())
    null_med = float(np.median(null_max)); null_95 = float(np.percentile(null_max, 95))
    if not scope_ok:
        verdict = "N/A_short_period"
    elif fap < 0.05:
        verdict = "SURVIVES_REDNOISE"
    elif fap < 0.20:
        verdict = "MARGINAL"
    else:
        verdict = "LIKELY_RED_NOISE"
    return dict(name=name, status="ok", P_d=P, dur_d=dur_d,
                n_bins=len(Tb), baseline_d=float(baseline),
                block_days=float(block_days), scope_ok=bool(scope_ok),
                sde_obs_binned=float(sde_obs),
                null_median=null_med, null_95pct=null_95,
                fap=fap, n_boot=n_boot, verdict=verdict,
                proc_s=float(time.time()-t0))

def main():
    df = pd.read_csv(RESULTS_IN)
    keep = df[df.eb_screen_verdict.fillna("").isin(["SURVIVED","FLAG_REVIEW"])].copy()
    keep = keep.sort_values("eb_screen_verdict")  # SURVIVED first alpha? just iterate
    print(f"Red-noise FAP test on {len(keep)} candidates (N_boot={N_BOOT}, bin={BIN_MINUTES}min)")
    rows = []
    # resume support
    done = set()
    if os.path.exists(RESULTS_OUT):
        prev = pd.read_csv(RESULTS_OUT); rows = prev.to_dict("records")
        done = set(prev["name"].astype(str).str.strip())
        print(f"  resuming -- {len(done)} already done")
    for _, r in keep.iterrows():
        nm = str(r["name"]).strip()
        if nm in done: continue
        dur_d = float(r["dur_h"])/24.0
        print(f"\n  {nm}  P={r['P_d']:.3f}d  SDE_pipeline={r['sde']:.2f}  eb={r['eb_screen_verdict']}", flush=True)
        try:
            res = run_fap(nm, float(r["ra"]), float(r["dec"]), float(r["P_d"]), dur_d, float(r.get("baseline_d", 0)))
        except Exception as e:
            res = dict(name=nm, status=f"error:{type(e).__name__}:{str(e)[:80]}")
        if res.get("status")=="ok":
            print(f"    SDE_obs(binned)={res['sde_obs_binned']:.2f}  null_median={res['null_median']:.2f}  "
                  f"null_95={res['null_95pct']:.2f}  FAP={res['fap']:.3f}  -> {res['verdict']}  ({res['proc_s']:.0f}s)", flush=True)
        else:
            print(f"    status: {res.get('status')}", flush=True)
        # attach pipeline columns for context
        res["sde_pipeline"] = float(r["sde"]); res["R_p_Rearth"]=float(r["R_p_Rearth"])
        res["T_eq_K"]=float(r["T_eq_K"]); res["eb_verdict"]=r["eb_screen_verdict"]
        res["centroid_verdict"]=r.get("centroid_verdict","")
        rows.append(res)
        pd.DataFrame(rows).to_csv(RESULTS_OUT, index=False)
    df_out = pd.DataFrame(rows)
    print(f"\n{'='*78}\nRED-NOISE FAP SUMMARY\n{'='*78}")
    ok = df_out[df_out.status=="ok"]
    if len(ok):
        cols = ["name","P_d","sde_pipeline","sde_obs_binned","null_95pct","fap","verdict","eb_verdict"]
        print(ok[cols].sort_values("fap").to_string(index=False))
    print(f"\nsaved {RESULTS_OUT}")

if __name__ == "__main__":
    main()

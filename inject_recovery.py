"""Injection-recovery: is a borderline candidate noise, or just below TESS's reach?

A FAP in the 0.05-0.35 range is ambiguous. It can mean either:
  (1) the observed peak is a red-noise alignment (no real planet), OR
  (2) a real planet of these parameters is intrinsically at the edge of
      detectability in this lightcurve, so even a genuine transit would only
      produce an SDE down in the noise -- the data simply can't decide.

Injection-recovery separates them. Inject synthetic box transits of the
candidate's exact depth/duration/period at RANDOM phase into the candidate's
own (detrended, binned) lightcurve, run the identical full-grid BLS, and
record the SDE a KNOWN-real planet of these parameters produces.

  - If injected-real SDE sits well ABOVE the red-noise null-95, but the
    OBSERVED SDE sits down in the null  ->  observed is consistent with NOISE
    (a real planet would have shown up louder; this one didn't).
  - If injected-real SDE also sits down near the null-95  ->  DATA_LIMITED:
    a real planet of these params is undetectable above this star's noise
    with TESS, so the candidate can't be dispositioned from archival data
    alone -- it needs ground follow-up, it is NOT "dead".

Reuses the same binning, block/period grid, and SDE definition as
widernet_rednoise_fap.py so the numbers are directly comparable.
"""
import os, glob, warnings, time, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from scipy.signal import medfilt
import multiprocessing as mp

import widernet_rednoise_fap as F   # reuse loaders, binning, BLS, grid logic

N_INJ     = 100
N_WORKERS = 4
RNG_SEED  = 20260530
OUT_CSV   = "inject_recovery_results.csv"

_G = {}

def inject_box(T, F_in, P, t0, dur_d, depth):
    """Add a box transit of fractional depth at ephemeris (P, t0)."""
    phase = ((T - t0) / P) % 1.0
    ph = np.where(phase > 0.5, phase - 1.0, phase)
    in_tr = np.abs(ph) < (dur_d / P / 2.0)
    F_out = F_in.copy()
    F_out[in_tr] *= (1.0 - depth)
    return F_out

def _inj_worker(k):
    rng = np.random.default_rng(RNG_SEED + 1 + k)
    Tb = _G["Tb"]; Fb = _G["Fb"]
    P = _G["P"]; dur_d = _G["dur_d"]; depth = _G["depth"]
    periods = _G["periods"]; durations = _G["durations"]
    # random phase, plus tiny period jitter so we are not locked to a grid node
    t0 = Tb[0] + rng.uniform(0, P)
    Pj = P * (1.0 + rng.uniform(-0.002, 0.002))
    F_inj = inject_box(Tb, Fb, Pj, t0, dur_d, depth)
    # SDE recovered at the injected period (narrow scan) and full-grid max
    sde_at = F.sde_at_period(Tb, F_inj, Pj, durations)
    sde_max, _ = F.bls_max_sde(Tb, F_inj, periods, durations)
    return sde_at, sde_max

def run_injection(name, tic, P, dur_d, depth, n_inj=N_INJ):
    lcs, msg = F.load_lcs_by_tic(tic)
    if not lcs:
        return dict(name=name, status=f"no_lc ({msg})")
    T, Fl = F.detrend_concat(lcs)
    Tb, Fb = F.bin_lc(T, Fl, F.BIN_MINUTES)
    if len(Tb) < 200:
        return dict(name=name, status="too_few_bins", n_bins=len(Tb))
    baseline = Tb.max() - Tb.min()
    pmax = min(baseline / 2.5, 300.0)
    n_grid = int(np.clip(6000 * (pmax / 15), 3000, F.GRID_CAP))
    periods = np.linspace(0.5, pmax, n_grid)
    durations = np.array([0.05, 0.08, 0.10, 0.15, 0.20, 0.30])
    _G.update(Tb=Tb, Fb=Fb, P=P, dur_d=dur_d, depth=depth,
              periods=periods, durations=durations)
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        res = pool.map(_inj_worker, range(n_inj))
    sde_at  = np.array([r[0] for r in res])
    sde_max = np.array([r[1] for r in res])
    return dict(name=name, status="ok", P_d=P, dur_d=dur_d, depth_ppm=depth*1e6,
                n_bins=len(Tb), baseline_d=float(baseline),
                inj_sde_at_median=float(np.median(sde_at)),
                inj_sde_at_5pct=float(np.percentile(sde_at, 5)),
                inj_sde_max_median=float(np.median(sde_max)),
                inj_sde_max_5pct=float(np.percentile(sde_max, 5)),
                proc_s=float(time.time()-t0))

def main(targets):
    bls = pd.read_csv("widernet_bls_results.csv")
    fap = pd.read_csv("widernet_rednoise_fap.csv") if os.path.exists("widernet_rednoise_fap.csv") else pd.DataFrame()
    rows = []
    for nm in targets:
        r = bls[bls.name.astype(str).str.strip() == nm]
        if len(r) == 0:
            print(f"  {nm}: not in BLS results"); continue
        r = r.iloc[0]
        tic = F.tic_for_name(nm)
        if tic is None:
            print(f"  {nm}: no TIC"); continue
        P = float(r.P_d); dur_d = float(r.dur_h)/24.0; depth = float(r.depth_ppm)*1e-6
        print(f"\n  {nm}  TIC={tic}  P={P:.3f}d  depth={r.depth_ppm:.0f}ppm  inject N={N_INJ}", flush=True)
        out = run_injection(nm, tic, P, dur_d, depth)
        # attach observed + null context from the FAP run
        if len(fap):
            fr = fap[fap.name.astype(str).str.strip() == nm]
            if len(fr):
                out["obs_sde_binned"] = float(fr.iloc[0]["sde_obs_binned"])
                out["null_95pct"]     = float(fr.iloc[0]["null_95pct"])
                out["fap"]            = float(fr.iloc[0]["fap"])
        # disposition
        if out.get("status") == "ok":
            obs = out.get("obs_sde_binned", np.nan)
            null95 = out.get("null_95pct", np.nan)
            inj_med = out["inj_sde_at_median"]
            inj_5 = out["inj_sde_at_5pct"]
            if np.isfinite(null95) and inj_5 <= null95:
                disp = "DATA_LIMITED (real planet of these params buried in noise)"
            elif np.isfinite(null95) and np.isfinite(obs) and inj_med > null95 and obs < null95:
                disp = "CONSISTENT_WITH_NOISE (real planet would show louder; observed in noise)"
            elif np.isfinite(obs) and obs >= inj_5:
                disp = "CONSISTENT_WITH_PLANET (observed in injected-real range)"
            else:
                disp = "AMBIGUOUS"
            out["disposition"] = disp
            print(f"    injected-real SDE@P: median={inj_med:.2f} 5pct={inj_5:.2f} | "
                  f"obs={obs:.2f} null95={null95:.2f}  -> {disp}", flush=True)
        else:
            print(f"    status: {out.get('status')}", flush=True)
        rows.append(out)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}")

if __name__ == "__main__":
    import sys
    # default: CPD-63 349 + any borderline 0.05<=FAP<0.35 from the FAP run
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = ["CPD-63   349"]
        if os.path.exists("widernet_rednoise_fap.csv"):
            fp = pd.read_csv("widernet_rednoise_fap.csv")
            bord = fp[(fp.fap >= 0.05) & (fp.fap < 0.35)].name.astype(str).str.strip().tolist()
            for b in bord:
                if b not in targets: targets.append(b)
    print(f"Injection-recovery targets: {targets}")
    main(targets)

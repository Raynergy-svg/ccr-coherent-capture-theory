"""Block-size robustness re-test: testing my own method's blind spot.

The main red-noise FAP test uses ~1-day blocks. That preserves intra-day
correlated noise (the dominant TESS noise: stellar granulation, jitter, the
30-min cadence aliasing) and destroys >1-day coherence. But it does NOT
preserve red noise on multi-day timescales -- rotation modulation, scattered
light "seasons", and sector-to-sector stitching offsets. If a candidate's
"survival" was riding multi-day correlated noise that the 1-day block test
happens to scramble out of existence, the 7-day block test will catch it:
the null SDE distribution will widen, the observed SDE will fall in the
null tail, and the FAP will rise.

Method: rerun the identical FAP machinery with block_days=7 on the 5
SURVIVES_REDNOISE + 1 MARGINAL candidates. Compare null_95 and FAP between
the 1-day and 7-day runs.

Verdict per candidate:
  ROBUST              FAP_7d still < 0.05 (signal is real to both blind spots)
  WEAKENED            FAP_7d in 0.05-0.20 (multi-day noise contributes)
  COLLAPSED           FAP_7d >= 0.20 (was riding multi-day red noise)
"""
import os, time, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import multiprocessing as mp

import widernet_rednoise_fap as F

N_BOOT     = 100
BLOCK_DAYS = 7.0          # one TESS-week: covers rotation, scattered-light seasons
RNG_SEED   = 20260530
N_WORKERS  = 4
OUT_CSV    = "block_robustness_results.csv"

# survivors + the marginal -- the ones that need to hold up
TARGETS = [
    "HD 271308",
    "2M18464473+5923422",
    "2M17004013+6026044",
    "2M16513518+5934383",
    "2M19510829+6545443",
    "2M19515616+7029139",   # MARGINAL at 1d, expect movement
]

_G = {}

def _boot_worker(k):
    rng = np.random.default_rng(RNG_SEED + 1 + k)
    _, Fp = F.block_permute(_G["Tb"], _G["Fb"], _G["block_days"], rng)
    s, _ = F.bls_max_sde(_G["Tb"], Fp, _G["periods"], _G["durations"])
    return s

def run_one(name, tic, P, dur_d):
    lcs, msg = F.load_lcs_by_tic(tic)
    if not lcs:
        return dict(name=name, status=f"no_lc ({msg})")
    T, Fl = F.detrend_concat(lcs)
    Tb, Fb = F.bin_lc(T, Fl, F.BIN_MINUTES)
    if len(Tb) < 200:
        return dict(name=name, status="too_few_bins", n_bins=len(Tb))
    baseline = Tb.max() - Tb.min()
    block_days = BLOCK_DAYS
    scope_ok = P > 10.0 * block_days   # need P >> 7d for blocks to destroy signal
    pmax = min(baseline / 2.5, 300.0)
    n_grid = int(np.clip(6000 * (pmax / 15), 3000, F.GRID_CAP))
    periods = np.linspace(0.5, pmax, n_grid)
    durations = np.array([0.05, 0.08, 0.10, 0.15, 0.20, 0.30])
    sde_obs = F.sde_at_period(Tb, Fb, P, durations)
    t0 = time.time()
    _G["Tb"], _G["Fb"] = Tb, Fb
    _G["periods"], _G["durations"] = periods, durations
    _G["block_days"] = block_days
    with mp.Pool(N_WORKERS) as pool:
        null_max = np.array(pool.map(_boot_worker, range(N_BOOT)))
    fap = float((null_max >= sde_obs).mean())
    null_med = float(np.median(null_max)); null_95 = float(np.percentile(null_max, 95))
    if not scope_ok:
        verdict = "N/A_period_too_short_for_7d_blocks"
    elif fap < 0.05:
        verdict = "ROBUST"
    elif fap < 0.20:
        verdict = "WEAKENED"
    else:
        verdict = "COLLAPSED"
    return dict(name=name, status="ok", P_d=P, dur_d=dur_d,
                block_days=block_days, scope_ok=bool(scope_ok),
                sde_obs_binned=float(sde_obs),
                null_median=null_med, null_95pct=null_95,
                fap=fap, n_boot=N_BOOT, verdict=verdict,
                proc_s=float(time.time()-t0))

def main():
    bls = pd.read_csv("widernet_bls_results.csv")
    fap_1d = pd.read_csv("widernet_rednoise_fap.csv")
    rows = []
    for nm in TARGETS:
        r = bls[bls.name.astype(str).str.strip() == nm.strip()]
        if len(r) == 0:
            print(f"  {nm}: not in BLS results"); continue
        r = r.iloc[0]
        tic = F.tic_for_name(nm)
        if tic is None:
            print(f"  {nm}: no TIC"); continue
        P = float(r.P_d); dur_d = float(r.dur_h)/24.0
        # carry the 1d-block FAP for direct comparison
        f1 = fap_1d[fap_1d.name.astype(str).str.strip() == nm.strip()]
        fap_1d_val = float(f1.iloc[0]["fap"]) if len(f1) else float("nan")
        null95_1d  = float(f1.iloc[0]["null_95pct"]) if len(f1) else float("nan")
        print(f"\n  {nm}  TIC={tic}  P={P:.3f}d  | 1d-block: FAP={fap_1d_val:.2f} null95={null95_1d:.2f}", flush=True)
        out = run_one(nm, tic, P, dur_d)
        out["fap_1d"] = fap_1d_val
        out["null_95pct_1d"] = null95_1d
        if out.get("status") == "ok":
            print(f"    7d-block: SDE_obs={out['sde_obs_binned']:.2f}  null95={out['null_95pct']:.2f}  "
                  f"FAP={out['fap']:.3f}  -> {out['verdict']}  ({out['proc_s']:.0f}s)", flush=True)
        else:
            print(f"    status: {out.get('status')}", flush=True)
        rows.append(out)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}")

if __name__ == "__main__":
    main()

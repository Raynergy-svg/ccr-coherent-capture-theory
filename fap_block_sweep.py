"""Block-size FAP sweep: every SURVIVED candidate gets a robustness curve,
not a point estimate.

The single-block-size FAP (widernet_rednoise_fap.py at 1d, block_robustness.py
at 7d) gives one number, which hides whether that number is stable under
the choice of block size. A real signal should hold low FAP across block
sizes from sub-day to sector-stitch scale. A red-noise alignment will
shift as the block size changes (typically: FAP rises as blocks get longer
because the bootstrap preserves more of the actual correlated noise).

Sweeps block_days in [1, 3, 7, 14] per candidate and reports FAP at each.
Verdict per candidate:
  ROBUST       FAP < 0.05 at every applicable block size (scope-allowed ones)
  STABLE       FAP < 0.10 at every applicable block size
  DEGRADES     FAP < 0.05 at 1d but >0.20 at any longer block
  UNSTABLE     FAP varies > 0.30 across block sizes
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import multiprocessing as mp

import widernet_rednoise_fap as F

N_BOOT     = 100
BLOCK_SWEEP_DAYS = [1.0, 3.0, 7.0, 14.0]
RNG_SEED   = 20260531
N_WORKERS  = 4
OUT_CSV    = "fap_block_sweep_results.csv"

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
    pmax = min(baseline / 2.5, 300.0)
    n_grid = int(np.clip(6000 * (pmax / 15), 3000, F.GRID_CAP))
    periods = np.linspace(0.5, pmax, n_grid)
    durations = np.array([0.05, 0.08, 0.10, 0.15, 0.20, 0.30])
    sde_obs = F.sde_at_period(Tb, Fb, P, durations)

    _G["Tb"], _G["Fb"] = Tb, Fb
    _G["periods"], _G["durations"] = periods, durations

    sweep = {}
    t0 = time.time()
    for bd in BLOCK_SWEEP_DAYS:
        scope_ok = P > 10.0 * bd
        if not scope_ok:
            sweep[bd] = dict(fap=float("nan"), null_95=float("nan"),
                             null_med=float("nan"), scope="N/A")
            continue
        _G["block_days"] = bd
        with mp.Pool(N_WORKERS) as pool:
            null_max = np.array(pool.map(_boot_worker, range(N_BOOT)))
        sweep[bd] = dict(
            fap=float((null_max >= sde_obs).mean()),
            null_95=float(np.percentile(null_max, 95)),
            null_med=float(np.median(null_max)),
            scope="ok",
        )

    # Verdict from the curve
    applicable = [bd for bd, v in sweep.items() if v["scope"] == "ok"]
    if not applicable:
        verdict = "N/A_all_blocks_too_long"
    else:
        faps = [sweep[bd]["fap"] for bd in applicable]
        if all(f < 0.05 for f in faps):
            verdict = "ROBUST_ACROSS_BLOCKS"
        elif all(f < 0.10 for f in faps):
            verdict = "STABLE_ACROSS_BLOCKS"
        elif sweep[applicable[0]]["fap"] < 0.05 and max(faps) > 0.20:
            verdict = "DEGRADES_WITH_BLOCK_SIZE"
        elif max(faps) - min(faps) > 0.30:
            verdict = "UNSTABLE_ACROSS_BLOCKS"
        else:
            verdict = "MIXED"

    out = dict(name=name, status="ok", P_d=P, dur_d=dur_d,
               sde_obs=float(sde_obs), baseline_d=float(baseline),
               verdict=verdict, proc_s=float(time.time() - t0))
    for bd in BLOCK_SWEEP_DAYS:
        out[f"fap_{int(bd):02d}d"]      = sweep[bd]["fap"]
        out[f"null95_{int(bd):02d}d"]   = sweep[bd]["null_95"]
        out[f"nullmed_{int(bd):02d}d"]  = sweep[bd]["null_med"]
        out[f"scope_{int(bd):02d}d"]    = sweep[bd]["scope"]
    return out

def main(target_names=None):
    bls = pd.read_csv("widernet_bls_results.csv")
    if target_names:
        names_norm = [n.strip() for n in target_names]
        survived = bls[bls.name.astype(str).str.strip().isin(names_norm)]
    else:
        survived = bls[bls.eb_screen_verdict.fillna("").str.contains("SURVIVED")]
    print(f"FAP block-size sweep on {len(survived)} candidates "
          f"(blocks={BLOCK_SWEEP_DAYS}, N_boot={N_BOOT})")
    rows = []
    done = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV); rows = prev.to_dict("records")
        done = set(prev["name"].astype(str).str.strip())
        print(f"  resuming -- {len(done)} already done")
    for _, r in survived.iterrows():
        nm = str(r["name"]).strip()
        if nm in done: continue
        tic = F.tic_for_name(nm)
        if tic is None:
            print(f"  {nm}: no TIC"); continue
        P = float(r.P_d); dur_d = float(r.dur_h)/24.0
        print(f"\n  {nm:<22}  TIC={tic}  P={P:.3f}d", flush=True)
        out = run_one(nm, tic, P, dur_d)
        if out.get("status") == "ok":
            fap_row = "  ".join(f"{int(bd):2d}d:{out[f'fap_{int(bd):02d}d']:.3f}" for bd in BLOCK_SWEEP_DAYS)
            print(f"    SDE={out['sde_obs']:.2f}  FAPs: {fap_row}  -> {out['verdict']}  ({out['proc_s']:.0f}s)", flush=True)
        else:
            print(f"    status: {out.get('status')}", flush=True)
        rows.append(out)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1:] if len(sys.argv) > 1 else None)

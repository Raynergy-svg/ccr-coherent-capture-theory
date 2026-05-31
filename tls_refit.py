"""TLS (Transit Least Squares) refit on the SURVIVED candidates.

BLS uses a box-shaped transit model. TLS uses a properly limb-darkened
transit shape (Mandel & Agol), so the recovered transit parameters are
physically meaningful, not the box approximations BLS produces.

For each SURVIVED candidate:
  - Load cached LCs, detrend, bin
  - Run TLS over a narrow grid around the BLS period
  - Record refined P, T0, duration, R_p/R*, impact parameter, ρ*, SDE_TLS

Comparison to BLS gives us:
  - Real planet radius (R_p / R_star from the modeled depth, not box-depth)
  - Goodness of fit (is the shape transit-like, or is BLS misfitting?)
  - Updated ephemeris precision for ground follow-up

Reference: Hippke & Heller 2019, A&A 623, A39
"""
import os, glob, time, warnings, sys
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from astropy.io import fits

from transitleastsquares import transitleastsquares
import widernet_rednoise_fap as F

OUT_CSV = "tls_refit_results.csv"

def run_one(name, tic, P_bls, R_star, M_star):
    lcs, msg = F.load_lcs_by_tic(tic)
    if not lcs:
        return dict(name=name, status=f"no_lc ({msg})")
    T, Fl = F.detrend_concat(lcs)
    if len(T) < 500:
        return dict(name=name, status="too_few_pts", n_pts=len(T))
    t0 = time.time()
    try:
        model = transitleastsquares(T, Fl)
        # Narrow period window around BLS peak: +/-3%. TLS is expensive at
        # full grid; we already know P from BLS to <1%, so a focused refit
        # is the right move.
        pmin = max(0.5, P_bls * 0.97)
        pmax = P_bls * 1.03
        r = model.power(
            R_star=R_star if R_star > 0 else 1.0,
            M_star=M_star if M_star > 0 else 1.0,
            period_min=pmin, period_max=pmax,
            n_transits_min=2,
            show_progress_bar=False,
            use_threads=4,
        )
        # transit_depths_uncertainties may be None if only 1 transit
        rp_over_rs = float(np.sqrt(max(r.depth_mean[0] - r.depth_mean[1], 0.0))) if r.depth_mean else float("nan")
        # planet radius in R_Earth from R_p/R_star * R_star
        R_p_earth = float(rp_over_rs * (R_star if R_star > 0 else 1.0) * 109.2)
        out = dict(
            name=name, status="ok",
            P_TLS=float(r.period),
            T0_TLS=float(r.T0),
            duration_TLS_h=float(r.duration * 24.0),
            depth_TLS=float(r.depth),
            depth_mean=float(r.depth_mean[0]) if r.depth_mean else float("nan"),
            depth_mean_err=float(r.depth_mean[1]) if r.depth_mean else float("nan"),
            SDE_TLS=float(r.SDE),
            SNR_TLS=float(r.snr),
            odd_even_mismatch=float(r.odd_even_mismatch),
            rp_rs=float(r.rp_rs),
            R_p_Earth_TLS=R_p_earth,
            P_bls=float(P_bls),
            P_delta_pct=float((r.period - P_bls) / P_bls * 100.0),
            n_transits=int(len(r.transit_times)),
            proc_s=float(time.time() - t0),
        )
        return out
    except Exception as e:
        return dict(name=name, status=f"error:{type(e).__name__}:{str(e)[:120]}",
                    proc_s=float(time.time()-t0))

def main(targets_arg=None):
    bls = pd.read_csv("widernet_bls_results.csv")
    if targets_arg:
        names = targets_arg
    else:
        # Default: all SURVIVED candidates from the EB screen
        names = bls[bls.eb_screen_verdict.fillna("").str.contains("SURVIVED")].name.astype(str).str.strip().tolist()
    print(f"TLS refit on {len(names)} candidates")
    rows = []
    for nm in names:
        r = bls[bls.name.astype(str).str.strip() == nm.strip()]
        if len(r) == 0:
            print(f"  {nm}: not in BLS"); continue
        r = r.iloc[0]
        tic = F.tic_for_name(nm)
        if tic is None:
            print(f"  {nm}: no TIC"); continue
        Rs = float(r.R_star) if pd.notna(r.R_star) else 1.0
        Ms = float(r.M_star) if pd.notna(r.M_star) else 1.0
        print(f"\n  {nm:<22}  TIC={tic}  P_BLS={r.P_d:.4f}d  R*={Rs:.2f}", flush=True)
        out = run_one(nm, tic, float(r.P_d), Rs, Ms)
        if out.get("status") == "ok":
            print(f"    P_TLS={out['P_TLS']:.4f}d ({out['P_delta_pct']:+.2f}%)  "
                  f"R_p={out['R_p_Earth_TLS']:.2f} R_E  SDE_TLS={out['SDE_TLS']:.2f}  "
                  f"SNR={out['SNR_TLS']:.1f}  n_tr={out['n_transits']}  ({out['proc_s']:.0f}s)", flush=True)
        else:
            print(f"    status: {out.get('status')}", flush=True)
        rows.append(out)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}")

if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)

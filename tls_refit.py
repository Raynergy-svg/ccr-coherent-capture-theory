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
        # Period window around BLS peak: +/-10%. Wide enough that TLS has
        # room to find an actual transit-shaped fit even when BLS' period
        # is several percent off OR when the BLS-period harmonic is the real one.
        pmin = max(0.5, P_bls * 0.90)
        pmax = P_bls * 1.10
        Rs = R_star if R_star > 0 else 1.0
        Ms = M_star if M_star > 0 else 1.0
        r = model.power(
            R_star=Rs, R_star_min=Rs*0.9, R_star_max=Rs*1.1,
            M_star=Ms, M_star_min=Ms*0.9, M_star_max=Ms*1.1,
            period_min=pmin, period_max=pmax,
            n_transits_min=2,
            show_progress_bar=False,
            use_threads=4,
        )
        # r.depth_mean may be a 2-element tuple/array (mean, err) OR a scalar,
        # depending on whether enough transits were found. Be defensive.
        def _pair(v, idx):
            if v is None: return float("nan")
            try:
                if hasattr(v, "__len__") and len(v) >= 2:
                    return float(v[idx])
                return float(v) if idx == 0 else float("nan")
            except Exception:
                return float("nan")
        dm  = _pair(getattr(r, "depth_mean", None), 0)
        dme = _pair(getattr(r, "depth_mean", None), 1)
        # rp/rs from the model: prefer r.rp_rs if present, else sqrt(depth_mean)
        rprs = getattr(r, "rp_rs", None)
        if rprs is None or not np.isfinite(rprs):
            rprs = float(np.sqrt(max(dm if np.isfinite(dm) else 0.0, 0.0)))
        rprs = float(rprs)
        R_p_earth = float(rprs * (R_star if R_star > 0 else 1.0) * 109.2)
        n_transits = 0
        tt = getattr(r, "transit_times", None)
        if tt is not None:
            try: n_transits = int(len(tt))
            except Exception: n_transits = 0
        sde_val = float(getattr(r, "SDE", 0.0) or 0.0)
        no_fit = (sde_val < 1.0) or (n_transits == 0) or (not np.isfinite(dm))
        out = dict(
            name=name,
            status="tls_no_fit_in_window" if no_fit else "ok",
            P_TLS=float(r.period) if not no_fit else float("nan"),
            T0_TLS=float(r.T0) if not no_fit else float("nan"),
            duration_TLS_h=float(r.duration * 24.0) if not no_fit else float("nan"),
            depth_TLS=float(r.depth) if not no_fit else float("nan"),
            depth_mean=dm,
            depth_mean_err=dme,
            SDE_TLS=float(r.SDE),
            SNR_TLS=float(r.snr),
            odd_even_mismatch=float(getattr(r, "odd_even_mismatch", float("nan"))),
            rp_rs=rprs,
            R_p_Earth_TLS=R_p_earth,
            P_bls=float(P_bls),
            P_delta_pct=float((r.period - P_bls) / P_bls * 100.0),
            n_transits=n_transits,
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

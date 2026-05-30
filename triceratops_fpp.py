"""TRICERATOPS FPP wrapper for the SURVIVED candidate list.

Runs Giacalone et al. 2021 (AJ 161, 24) statistical validation on every
candidate that survived BLS + centroid + 7-test EB screen, computing:

  FPP   total false-positive probability (sum over all FP scenarios)
  NFPP  nearby false-positive probability (only nearby-stars-EB scenarios)

Validation thresholds from Giacalone+2021 / Castro-Gonzalez+2023:
  FPP  < 0.015 AND NFPP < 0.001  -> STAT_VALIDATED
  FPP  < 0.5   AND NFPP < 0.001  -> LIKELY_PLANET
  NFPP > 0.1                     -> LIKELY_NEARBY_FP
  else                           -> AMBIGUOUS_NEEDS_IMAGING

Runs without a high-resolution imaging contrast curve, so reported FPPs are
the WITHOUT-imaging baseline. Per Castro-Gonzalez+2023 a sub-Neptune at
P~200d without imaging typically lands FPP ~0.05-0.15 -- above the 0.015
validation threshold. Adding a TFOP-SG3 contrast curve (Zorro/'Alopeke/SOAR
HRCam/NIRC2) typically drops FPP by ~3x.

Inputs:
  widernet_bls_results.csv  (with eb_screen_verdict column)
  cached LC FITS under /tmp/tess_dl/mastDownload/TESS/

Output:
  triceratops_fpp_results.csv

Notes:
  TRICERATOPS' lightkurve internal calls may try to hit MAST. The container
  has clock skew that breaks SSL handshakes to MAST, so we pre-cache by
  loading from disk and feeding (time, flux, flux_err) arrays directly to
  calc_probs() rather than letting TRICERATOPS download them.

  TRICERATOPS also needs a TRILEGAL stellar-population-synthesis catalog
  (background-star density priors). It tries to query stev.oapd.inaf.it
  live, but that endpoint may be unreachable -- in this container TRILEGAL
  returns 503 due to upstream TLS issues. To bypass, pre-stage TRILEGAL
  output files at TRILEGAL_DIR/<TIC>.dat (one per target) and the wrapper
  will pass them via the trilegal_fname parameter. Generate them on any
  machine that can reach https://stev.oapd.inaf.it/cgi-bin/trilegal_1.6 ;
  the file format is the standard TRILEGAL CSV output with the standard
  header (mass, age, [M/H], TESS, J, H, Ks, ...). When neither route works,
  the wrapper records status="TRICERATOPS_BLOCKED:trilegal_unreachable" so
  the candidate stays in the queue for re-processing on a different host.
"""
import os, sys, glob, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from astropy.io import fits

import triceratops.triceratops as tr

RESULTS_IN   = "widernet_bls_results.csv"
RESULTS_OUT  = "triceratops_fpp_results.csv"
CACHE_ROOT   = "/tmp/tess_dl/mastDownload/TESS"
TRILEGAL_DIR = "triceratops_trilegal"     # pre-staged TRILEGAL files, optional
N_MC         = 50_000          # Giacalone+2021 default 1M; 50k gives FPP +/- ~0.005 which is plenty for our thresholds

def load_lcs_by_tic(tic):
    """Cached LC loader (zero MAST calls). Returns (time, flux, flux_err) concatenated."""
    pad = str(tic).split(".")[0].zfill(16)
    files = sorted(glob.glob(f"{CACHE_ROOT}/**/*-{pad}-*_lc.fits", recursive=True))
    out_t, out_f, out_e, sectors = [], [], [], []
    for fp in files:
        try:
            with fits.open(fp) as h:
                hdr = h["PRIMARY"].header
                sec = hdr.get("SECTOR", -1)
                d = h["LIGHTCURVE"].data
                t = d["TIME"]; f = d["PDCSAP_FLUX"]; fe = d["PDCSAP_FLUX_ERR"]; q = d["QUALITY"]
                m = np.isfinite(t)&np.isfinite(f)&np.isfinite(fe)&(q==0)
                if m.sum() < 100: continue
                fm = f[m]; em = fe[m]; med = np.nanmedian(fm)
                out_t.append(t[m]); out_f.append(fm/med); out_e.append(em/med)
                sectors.append(int(sec))
        except Exception:
            continue
    if not out_t: return np.array([]), np.array([]), np.array([]), []
    T = np.concatenate(out_t); F = np.concatenate(out_f); E = np.concatenate(out_e)
    o = np.argsort(T)
    return T[o], F[o], E[o], sorted(set(sectors))

def classify(fpp, nfpp):
    if not (np.isfinite(fpp) and np.isfinite(nfpp)):
        return "TRICERATOPS_ERR"
    if fpp < 0.015 and nfpp < 0.001:  return "STAT_VALIDATED"
    if fpp < 0.5   and nfpp < 0.001:  return "LIKELY_PLANET"
    if nfpp > 0.1:                    return "LIKELY_NEARBY_FP"
    return "AMBIGUOUS_NEEDS_IMAGING"

def run_one(name, tic, ra, dec, P_d, T0, dur_d, depth_frac):
    """Returns dict with FPP, NFPP, classification, and top scenarios."""
    T, F, FE, sectors = load_lcs_by_tic(tic)
    if len(T) < 200:
        return dict(name=name, status="no_lc", n_pts=len(T))

    # Phase-fold to a near-transit window (TRICERATOPS expects the local
    # transit light curve, not the full multi-sector LC; use one transit width
    # on either side so the model fit doesn't drown in baseline.)
    phase = ((T - T0) / P_d) % 1.0
    ph = np.where(phase > 0.5, phase - 1.0, phase)
    half_win = max(2.0 * dur_d / P_d, 0.02)
    keep = np.abs(ph) < half_win
    if keep.sum() < 40:
        return dict(name=name, status="too_few_in_phase_window", n_in=int(keep.sum()))
    t_loc = (ph[keep] * P_d)         # time relative to transit center, in days
    f_loc = F[keep]
    fe_med = float(np.median(FE[keep]))
    if not np.isfinite(fe_med) or fe_med <= 0:
        fe_med = float(np.nanstd(f_loc))

    t0 = time.time()
    try:
        # TRICERATOPS pulls Gaia + TIC priors via lightkurve. Pass ra/dec to
        # bypass name resolution. sectors=array of sector numbers observed.
        trilegal_file = os.path.join(TRILEGAL_DIR, f"{int(tic)}.dat")
        trilegal_arg = trilegal_file if os.path.exists(trilegal_file) else None
        tgt = tr.target(ID=int(tic), sectors=np.array(sectors, dtype=int),
                        search_radius=10, ra=ra, dec=dec, verify_ssl=True,
                        trilegal_fname=trilegal_arg)
        tgt.calc_depths(tdepth=float(depth_frac))
        # FPP run -- N MC scenarios, no contrast curve, default exptime=2min
        tgt.calc_probs(time=t_loc, flux_0=f_loc, flux_err_0=fe_med,
                       P_orb=float(P_d), N=N_MC, parallel=False,
                       contrast_curve_file=None, verbose=0)
        fpp  = float(tgt.FPP)
        nfpp = float(tgt.NFPP)
        # Top scenarios
        try:
            probs = tgt.probs.sort_values("prob", ascending=False).head(3)
            top = "; ".join(f"{r.scenario}:{r['prob']:.3f}" for _, r in probs.iterrows())
        except Exception:
            top = ""
        cls = classify(fpp, nfpp)
        return dict(name=name, status="ok", FPP=fpp, NFPP=nfpp,
                    classification=cls, top_scenarios=top,
                    trilegal_source=("cached" if trilegal_arg else "live"),
                    proc_s=time.time()-t0)
    except Exception as e:
        msg = str(e)
        if "select" in msg or "NoneType" in msg.lower():
            # Most commonly: TRILEGAL query returned an unparsable page.
            status = "TRICERATOPS_BLOCKED:trilegal_unreachable"
        else:
            status = f"error:{type(e).__name__}:{msg[:120]}"
        return dict(name=name, status=status, proc_s=time.time()-t0)

def main():
    bls = pd.read_csv(RESULTS_IN)
    # SURVIVED only (post 7-test EB screen). Drop EB_REJECT / SKIPPED / blanks.
    keep = bls[bls.eb_screen_verdict.fillna("").str.contains("SURVIVED")].copy()
    print(f"Loaded {len(bls)} BLS rows; running TRICERATOPS on {len(keep)} SURVIVED")
    rows = []
    if os.path.exists(RESULTS_OUT):
        prev = pd.read_csv(RESULTS_OUT); rows = prev.to_dict("records")
        done = set(prev.name.astype(str).str.strip())
        print(f"  resuming -- {len(done)} already done")
    else:
        done = set()
    for _, r in keep.iterrows():
        nm = str(r["name"]).strip()
        if nm in done: continue
        tic = r.get("tic_id")
        if pd.isna(tic):
            from widernet_rednoise_fap import tic_for_name
            tic = tic_for_name(nm)
        if tic is None or (isinstance(tic, float) and not np.isfinite(tic)):
            rows.append(dict(name=nm, status="no_tic")); continue
        print(f"\n  {nm:<22}  TIC={int(float(tic))}  P={r.P_d:.3f}d  depth={r.depth_ppm:.0f}ppm", flush=True)
        out = run_one(name=nm, tic=int(float(tic)), ra=float(r.ra), dec=float(r.dec),
                      P_d=float(r.P_d), T0=float(r.T0_BTJD),
                      dur_d=float(r.dur_h)/24.0, depth_frac=float(r.depth_ppm)*1e-6)
        if out.get("status") == "ok":
            print(f"    FPP={out['FPP']:.4f}  NFPP={out['NFPP']:.4f}  -> {out['classification']}  ({out['proc_s']:.0f}s)", flush=True)
            if out.get("top_scenarios"):
                print(f"    top: {out['top_scenarios']}", flush=True)
        else:
            print(f"    status: {out.get('status')}", flush=True)
        rows.append(out)
        pd.DataFrame(rows).to_csv(RESULTS_OUT, index=False)
    print(f"\nsaved {RESULTS_OUT}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        # Quick single-target sanity test: triceratops_fpp.py --one "HD 271308"
        nm = sys.argv[2] if len(sys.argv) > 2 else "HD 271308"
        bls = pd.read_csv(RESULTS_IN)
        r = bls[bls.name.astype(str).str.strip() == nm.strip()].iloc[0]
        from widernet_rednoise_fap import tic_for_name
        tic = tic_for_name(nm)
        print(f"sanity test: {nm}  TIC={tic}")
        out = run_one(name=nm, tic=int(tic), ra=float(r.ra), dec=float(r.dec),
                      P_d=float(r.P_d), T0=float(r.T0_BTJD),
                      dur_d=float(r.dur_h)/24.0, depth_frac=float(r.depth_ppm)*1e-6)
        print(out)
    else:
        main()

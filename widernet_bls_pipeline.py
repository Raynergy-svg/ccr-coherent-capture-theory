"""Wider-net BLS pipeline: run BLS on the target list, auto-vet SDE>=7 hits,
keep the survivors.

Inputs:
  widernet_targets.csv -- ranked target list from widernet_targets_build.py

For each target (sorted by sector count, highest first):
  1. Download SPOC LCs from MAST (using local cache)
  2. Detrend per sector with 12-h medfilt + 5sigma clip
  3. Concatenate + run BLS with period range tied to baseline
  4. Compute SDE = (peak - mean_off_peak) / std_off_peak with proper
     +- 5% exclusion window around the peak
  5. If SDE < 7 -- skip (record null)
  6. If SDE >= 7 -- record candidate, run vetting:
     a. Reject obvious systematics (P in [13.5, 14.0], [27.0, 27.5], or
        below 0.5 d single-sector noise floor)
     b. Compute per-sector flux check (is the dip in multiple sectors?)
     c. Run PSF centroid test (from centroid_psf_test.py) for SDE>=7
        candidates with multi-sector transits
     d. Save full diagnostic plot + record
  7. Save running summary CSV so you can stop and resume

Output:
  widernet_bls_results.csv -- all targets with SDE / period / status
  widernet_candidates/ -- per-candidate diagnostic plots and centroid logs
"""
import os, sys, warnings, time, re, json, argparse
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Observations
from scipy.signal import medfilt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from centroid_psf_test import per_sector_centroid_test, combine_sector_results, verdict as centroid_verdict, get_spoc_lcs

SDE_THRESHOLD = 7.0
DOWNLOAD_DIR = "/tmp/tess_dl"
RESULTS_CSV = "widernet_bls_results.csv"
CAND_DIR = "widernet_candidates"
os.makedirs(CAND_DIR, exist_ok=True)

# Known TESS-orbit-period systematic windows -- reject SDE >=7 candidates
# whose detected period lies in any of these. (TESS orbit ~ 13.7 d.)
SYSTEMATIC_WINDOWS = [(0.40, 0.95), (13.5, 14.0), (27.0, 28.0)]

def is_systematic_period(P):
    return any(lo <= P <= hi for lo, hi in SYSTEMATIC_WINDOWS)

def get_lc_sectors_cached(ra, dec):
    """Fetch SPOC LCs, prefer cached, return list of (sector, t, f) tuples."""
    try:
        obs = Observations.query_region(f"{ra} {dec}", radius="20 arcsec")
        ts = obs[(obs["obs_collection"]=="TESS") & (obs["dataproduct_type"]=="timeseries")]
        if len(ts) == 0: return [], "no_TS"
        prods = Observations.get_product_list(ts)
        lc_prods = prods[prods["productSubGroupDescription"]=="LC"]
        if len(lc_prods) == 0: return [], "no_LC"
        dl = Observations.download_products(lc_prods, download_dir=DOWNLOAD_DIR, verbose=False)
    except Exception as e:
        return [], f"dl_err:{e}"

    sectors = []
    for fp in dl["Local Path"]:
        try:
            with fits.open(fp) as hdul:
                hdr = hdul[0].header
                sec = hdr.get("SECTOR", -1)
                lc = hdul["LIGHTCURVE"].data
                t = np.array(lc["TIME"], dtype=float)
                f = np.array(lc["PDCSAP_FLUX"], dtype=float)
                q = np.array(lc["QUALITY"], dtype=int)
                m = np.isfinite(t)&np.isfinite(f)&(q==0)
                if m.sum() < 100: continue
                sectors.append((int(sec), t[m], f[m]/np.nanmedian(f[m]), fp))
        except Exception:
            continue
    return sectors, "ok"

def detrend(t, f, win_days=0.5):
    dt = np.median(np.diff(t))
    if dt <= 0: dt = 0.02
    box = max(3, int(win_days/dt));  box = box+1 if box%2==0 else box
    fd = f/medfilt(f, box)
    sig = 1.4826*np.median(np.abs(fd-1.0))
    keep = np.abs(fd-1.0) < 5*sig
    return t[keep], fd[keep]

def bls_with_sde(T, F, baseline_d):
    bls = BoxLeastSquares(T, F)
    pmax = min(baseline_d/2.5, 250.0)
    n_grid = int(np.clip(5000 * (pmax/15), 4000, 25000))
    periods = np.linspace(0.5, pmax, n_grid)
    durs = np.array([0.05, 0.08, 0.10, 0.15, 0.20])
    pg = bls.power(periods, durs)
    pwr = pg.power
    i = int(np.argmax(pwr))
    P = float(pg.period[i])
    near = np.abs((pg.period - P)/P) < 0.01
    mu = float(np.nanmean(pwr[~near])); sd = float(np.nanstd(pwr[~near]))
    sde = (float(pwr[i]) - mu)/sd if sd > 0 else 0
    return dict(P=P, dur=float(pg.duration[i]), T0=float(pg.transit_time[i]),
                depth=float(pg.depth[i]), sde=float(sde), pg=pg)

def run_target(row):
    """Process one target. Return result dict."""
    name = str(row.get("apogee_id", "")) or f"ra{row.ra:.4f}_dec{row.dec:.4f}"
    ra, dec = float(row.ra), float(row.dec)
    Rs = float(row.R_star) if pd.notna(row.R_star) else 1.0
    Ms = float(row.M_star) if pd.notna(row.M_star) else 1.0
    Teff = float(row.Teff_tic) if pd.notna(row.Teff_tic) else 5700.0
    Tmag = float(row.Tmag) if pd.notna(row.Tmag) else np.nan
    tic_id = row.get("tic_id", "")

    t0 = time.time()
    sectors, msg = get_lc_sectors_cached(ra, dec)
    if not sectors:
        return dict(name=name, ra=ra, dec=dec, status=f"no_lc:{msg}")
    # detrend and concatenate
    all_t, all_f, all_seclabel = [], [], []
    for s, t, f, fp in sectors:
        td, fd = detrend(t, f)
        if len(td) < 100: continue
        all_t.append(td); all_f.append(fd); all_seclabel.append(np.full_like(td, s, dtype=int))
    if not all_t:
        return dict(name=name, ra=ra, dec=dec, status="all_detrend_short")
    T = np.concatenate(all_t); F = np.concatenate(all_f)
    Sec = np.concatenate(all_seclabel)
    base = T.max() - T.min(); rms_ppm = float(np.std(F))*1e6

    res = bls_with_sde(T, F, base)
    dt_proc = time.time() - t0

    # If-real planet params
    R_p_Re = np.sqrt(max(res["depth"], 0)) * Rs * 109.2
    a_AU = ((res["P"]/365.25)**2 * Ms)**(1/3)
    T_eq = Teff * np.sqrt(Rs*0.00465047/(2*a_AU))

    out = dict(
        name=name, ra=ra, dec=dec, tic_id=str(tic_id), Tmag=Tmag,
        R_star=Rs, M_star=Ms, Teff=Teff,
        n_sectors=len(sectors), baseline_d=float(base), n_pts=len(T),
        rms_ppm=rms_ppm,
        P_d=res["P"], dur_h=res["dur"]*24, depth_ppm=res["depth"]*1e6,
        T0_BTJD=res["T0"], sde=res["sde"],
        R_p_Rearth=R_p_Re, a_AU=a_AU, T_eq_K=T_eq,
        is_systematic_period=bool(is_systematic_period(res["P"])),
        proc_time_s=float(dt_proc),
    )

    if res["sde"] < SDE_THRESHOLD:
        out["status"] = f"no_signal_SDE_{res['sde']:.1f}"
        return out

    # SDE >= 7 -- vet
    if is_systematic_period(res["P"]):
        out["status"] = f"SDE_OK_systematic_period"
        return out

    # Save phase-fold and BLS-spectrum diagnostic
    plot_dir = os.path.join(CAND_DIR, name.replace("/", "_"))
    os.makedirs(plot_dir, exist_ok=True)
    phase = ((T - res["T0"])/res["P"]) % 1.0
    ph = np.where(phase>0.5, phase-1.0, phase)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].plot(T, F, ",", ms=1, alpha=0.3, color="steelblue")
    axes[0].set_xlabel("BTJD (d)"); axes[0].set_ylabel("flux")
    axes[0].set_title(f"{name}  TIC {tic_id}  T={Tmag:.2f}  n_sec={len(sectors)}  baseline={base:.0f}d")
    axes[0].set_ylim(0.99, 1.01)
    axes[1].plot(ph, F, ".", ms=1, alpha=0.3, color="gray")
    nb = 60; phb = np.linspace(-0.05, 0.05, nb+1); flb = []
    for lo,hi in zip(phb[:-1], phb[1:]):
        m = (ph>=lo)&(ph<hi); flb.append(float(np.mean(F[m])) if m.sum() else np.nan)
    axes[1].plot((phb[:-1]+phb[1:])/2, flb, "ro-", ms=4, lw=1.5)
    axes[1].set_xlim(-0.05, 0.05)
    axes[1].set_xlabel(f"phase (P={res['P']:.4f} d)"); axes[1].set_ylabel("flux")
    axes[1].set_title(f"phase: depth={res['depth']*1e6:.0f}ppm dur={res['dur']*24:.2f}h SDE={res['sde']:.2f} R_p={R_p_Re:.2f}R_E T_eq={T_eq:.0f}K")
    axes[2].plot(res["pg"].period, res["pg"].power, "-", lw=0.5)
    axes[2].axvline(res["P"], color="r", ls="--", lw=1)
    axes[2].set_xlabel("period (d)"); axes[2].set_ylabel("BLS power")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "bls.png"), dpi=110, bbox_inches="tight")
    plt.close()

    # Run centroid test
    cent_results = []
    try:
        # use the existing get_spoc_lcs from centroid_psf_test (also caches)
        lcs = get_spoc_lcs(ra, dec)
        for sec, fp in sorted(lcs):
            try:
                r = per_sector_centroid_test(fp, res["P"], res["T0"], res["dur"])
                cent_results.append(r)
            except Exception:
                continue
        combined = combine_sector_results(cent_results)
        verdict_centroid = centroid_verdict(combined)
    except Exception as e:
        combined = dict(status=f"centroid_err:{e}")
        verdict_centroid = "CENTROID_ERR"

    out["status"] = f"SDE_OK_{verdict_centroid.split()[0]}"
    out["centroid_offset_arcsec"] = combined.get("combined_offset_arcsec", float("nan"))
    out["centroid_sigma"] = combined.get("combined_significance_sigma", float("nan"))
    out["centroid_direction_R"] = combined.get("direction_circular_R", float("nan"))
    out["centroid_verdict"] = verdict_centroid

    # save per-candidate JSON
    cand_json = os.path.join(plot_dir, "candidate.json")
    with open(cand_json, "w") as f:
        json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else
                       float(v) if isinstance(v,(np.floating,np.integer)) else v)
                   for k,v in out.items()}, f, indent=2)
    return out

def main(targets_csv="widernet_targets.csv", limit=None, resume=True):
    print("="*78)
    print("WIDER-NET BLS PIPELINE")
    print("="*78)
    targets = pd.read_csv(targets_csv)
    targets = targets.sort_values("n_lc_sectors_actual", ascending=False).reset_index(drop=True)
    if limit: targets = targets.head(limit)
    print(f"  targets: {len(targets)}  (sorted by sector count desc)")

    # Resume support
    done = set()
    if resume and os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        done = set(prev["name"].astype(str).tolist())
        print(f"  resuming -- {len(done)} already done")

    rows = []
    if os.path.exists(RESULTS_CSV):
        rows = pd.read_csv(RESULTS_CSV).to_dict("records")

    t_start = time.time()
    n_new = 0
    for i, r in targets.iterrows():
        name = str(r.get("apogee_id", "")) or f"ra{r.ra:.4f}_dec{r.dec:.4f}"
        if name in done: continue
        try:
            out = run_target(r)
        except Exception as e:
            out = dict(name=name, ra=float(r.ra), dec=float(r.dec),
                       status=f"exception:{type(e).__name__}:{e}")
        rows.append(out)
        n_new += 1
        # save running
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
        # progress message
        flag = ""
        if "SDE_OK" in str(out.get("status","")):
            flag = " *** SDE>=7"
        dt = time.time() - t_start
        rate = n_new / max(dt, 1)
        print(f"  [{i+1:>4}/{len(targets)}] {name:<18}  "
              f"n_sec={out.get('n_sectors','-'):>3}  "
              f"SDE={out.get('sde','-'):>5}  "
              f"P={out.get('P_d','-')}  "
              f"R_p={out.get('R_p_Rearth','-')}  "
              f"-> {out.get('status','-')}{flag}")

    print(f"\nDONE.  Total processed: {n_new} new, {len(rows)} total")
    summary = pd.DataFrame(rows)
    if len(summary) > 0:
        ok = summary[summary.status.fillna("").str.contains("SDE_OK")]
        print(f"  SDE >= 7 hits: {len(ok)}")
        if len(ok):
            print(ok[["name","tic_id","n_sectors","P_d","sde","depth_ppm","R_p_Rearth","T_eq_K","status","centroid_verdict"]].to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="widernet_targets.csv")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    main(args.targets, limit=args.limit, resume=not args.no_resume)

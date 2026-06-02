"""Characterize the signal shape of the two non-transit periodic signals
found by the BLS pipeline:
  HD 271308          P=34.27d  R*=1.54  Teff=6287
  2M18464473+5923422 P=246.01d R*=1.48  Teff=5670

Both pass everything except TLS shape constraint. Goal: figure out what
kind of variability they are. Metrics computed per phase-folded LC:

  1. Peak-to-peak amplitude (ppm)
  2. V-shape vs U-shape metric (transit ratio dur_15/dur_50)
  3. Sinusoidal fit residual vs box fit residual (which model fits better?)
  4. Per-sector amplitude stability (rotation modulates with spot evolution
     -> amplitude changes across sectors; pulsation is amplitude-stable)
  5. Period coherence over the full TESS baseline (O-C residuals)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import widernet_rednoise_fap as F
from astropy.io import fits
import glob

CACHE_ROOT = "/tmp/tess_dl/mastDownload/TESS"

def load_lcs_with_sectors(tic):
    """Same as F.load_lcs_by_tic but keeps sector info."""
    pad = str(tic).split(".")[0].zfill(16)
    files = sorted(glob.glob(f"{CACHE_ROOT}/**/*-{pad}-*_lc.fits", recursive=True))
    out = []
    for fp in files:
        try:
            with fits.open(fp) as h:
                sec = int(h["PRIMARY"].header.get("SECTOR", -1))
                d = h["LIGHTCURVE"].data
                t = d["TIME"]; f = d["PDCSAP_FLUX"]; q = d["QUALITY"]
                m = np.isfinite(t)&np.isfinite(f)&(q==0)
                if m.sum() < 100: continue
                out.append((sec, t[m], f[m]/np.nanmedian(f[m])))
        except Exception:
            continue
    return out

def characterize(name, tic, P, T0, dur_d, R_star):
    print(f"\n{'='*72}\n{name}  TIC={tic}  P={P:.4f}d  R*={R_star:.2f}\n{'='*72}")
    lcs = load_lcs_with_sectors(tic)
    if not lcs:
        print("  no LCs"); return None

    # Per-sector LC stats
    print(f"  loaded {len(lcs)} sectors")

    # Detrend + concat for the global phase fold
    T_all, F_all = F.detrend_concat([(t, f) for _, t, f in lcs])
    Tb, Fb = F.bin_lc(T_all, F_all, 30.0)

    # Phase-fold at the BLS period
    phase = ((Tb - T0) / P) % 1.0
    ph_centered = np.where(phase > 0.5, phase - 1.0, phase)
    order = np.argsort(ph_centered)
    ph_sorted = ph_centered[order]; flux_sorted = Fb[order]

    # Bin the phase-fold into 50 bins for shape analysis
    nb = 50
    phase_edges = np.linspace(-0.5, 0.5, nb + 1)
    phase_centers = 0.5 * (phase_edges[1:] + phase_edges[:-1])
    binned_flux = np.zeros(nb)
    for i in range(nb):
        m = (ph_sorted >= phase_edges[i]) & (ph_sorted < phase_edges[i+1])
        binned_flux[i] = np.median(flux_sorted[m]) if m.sum() > 0 else np.nan

    # Peak-to-peak amplitude
    pp_ppm = (np.nanmax(binned_flux) - np.nanmin(binned_flux)) * 1e6
    print(f"  peak-to-peak amplitude: {pp_ppm:.0f} ppm")

    # Sinusoidal fit vs box fit
    def sinusoid(x, A, phi, c):
        return A * np.sin(2 * np.pi * x + phi) + c
    def two_sinusoid(x, A1, A2, phi1, phi2, c):
        # second harmonic: detects ellipsoidal / RR Lyrae shape (twin peaks)
        return A1 * np.sin(2*np.pi*x + phi1) + A2 * np.sin(4*np.pi*x + phi2) + c

    ok = np.isfinite(binned_flux)
    x = phase_centers[ok]; y = binned_flux[ok]
    try:
        p1, _ = curve_fit(sinusoid, x, y, p0=[pp_ppm/2/1e6, 0, 1.0], maxfev=5000)
        resid_sine = np.sum((y - sinusoid(x, *p1))**2)
    except Exception:
        p1 = None; resid_sine = np.inf
    try:
        p2, _ = curve_fit(two_sinusoid, x, y, p0=[pp_ppm/2/1e6, pp_ppm/4/1e6, 0, 0, 1.0], maxfev=5000)
        resid_two = np.sum((y - two_sinusoid(x, *p2))**2)
        amp_ratio_2to1 = abs(p2[1]) / abs(p2[0]) if abs(p2[0]) > 0 else np.inf
    except Exception:
        p2 = None; resid_two = np.inf; amp_ratio_2to1 = np.nan
    # Box-transit residual
    in_tr = np.abs(x) < (dur_d / P / 2.0)
    out_tr = ~in_tr
    if out_tr.sum() > 0:
        oot_med = np.median(y[out_tr])
        in_med  = np.median(y[in_tr]) if in_tr.sum() > 0 else oot_med
        box_pred = np.where(in_tr, in_med, oot_med)
        resid_box = np.sum((y - box_pred)**2)
    else:
        resid_box = np.inf
    print(f"  fit residuals (lower = better):")
    print(f"    sinusoid (1 harmonic):  {resid_sine:.3e}")
    print(f"    sin+2nd-harm (ellipsoidal-shape): {resid_two:.3e}  amp_2/amp_1={amp_ratio_2to1:.2f}")
    print(f"    box (transit):           {resid_box:.3e}")
    best = min((resid_sine,"sinusoid"),(resid_two,"sin+2nd"),(resid_box,"box"))
    print(f"  best fit:                  {best[1]}")

    # Per-sector amplitude (does it vary -> rotation/spots? or stable -> pulsation/eb?)
    per_sec = []
    for sec, t, f in lcs:
        if len(t) < 100: continue
        # detrend this sector only
        from scipy.signal import medfilt
        dt = np.median(np.diff(t)); dt = dt if dt>0 else 0.02
        box = max(3, int(0.5/dt));  box = box+1 if box%2==0 else box
        fd = f / medfilt(f, box)
        ph_s = ((t - T0) / P) % 1.0
        ph_s = np.where(ph_s > 0.5, ph_s - 1.0, ph_s)
        if abs(ph_s.max() - ph_s.min()) < 0.3: continue  # too little phase coverage
        amp = (np.percentile(fd, 95) - np.percentile(fd, 5)) * 1e6
        per_sec.append((sec, amp))
    if per_sec:
        amps = np.array([a for _, a in per_sec])
        print(f"  per-sector amplitudes (ppm): mean={amps.mean():.0f}  std={amps.std():.0f}  "
              f"min={amps.min():.0f}  max={amps.max():.0f}  rel_std={amps.std()/amps.mean():.2f}")
        if amps.std()/amps.mean() > 0.5:
            print(f"  -> AMPLITUDE VARIES (rel_std>0.5): consistent with rotation+evolving spots")
        else:
            print(f"  -> AMPLITUDE STABLE (rel_std<=0.5): consistent with pulsation or eclipsing geometry")

    # Save phase-fold plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ph_sorted, flux_sorted, ".", color="0.7", ms=1, alpha=0.3)
    ax.plot(phase_centers, binned_flux, "o-", color="k", lw=1.5, ms=4)
    if p1 is not None:
        xx = np.linspace(-0.5, 0.5, 200)
        ax.plot(xx, sinusoid(xx, *p1), "-", color="C0", lw=1, label="sinusoid")
    if p2 is not None:
        ax.plot(xx, two_sinusoid(xx, *p2), "-", color="C1", lw=1, label="sin+2nd-harm")
    ax.set_xlabel("phase"); ax.set_ylabel("normalized flux")
    ax.set_title(f"{name} @ P={P:.3f}d  (pp={pp_ppm:.0f} ppm)")
    ax.legend(loc="best", fontsize=8)
    fn = f"shape_{name.replace(' ','_').replace('+','_')}.png"
    fig.tight_layout(); fig.savefig(fn, dpi=100); plt.close()
    print(f"  phase-fold plot: {fn}")

    return dict(name=name, P_d=P, pp_ppm=pp_ppm,
                resid_sine=resid_sine, resid_two=resid_two, resid_box=resid_box,
                best_model=best[1], amp_ratio_2to1=amp_ratio_2to1,
                n_sectors=len(lcs))

if __name__ == "__main__":
    bls = pd.read_csv("widernet_bls_results.csv")
    results = []
    for nm in ["HD 271308", "2M18464473+5923422"]:
        r = bls[bls.name.astype(str).str.strip() == nm].iloc[0]
        tic = F.tic_for_name(nm)
        res = characterize(nm, tic, float(r.P_d), float(r.T0_BTJD),
                           float(r.dur_h)/24.0, float(r.R_star))
        if res: results.append(res)
    pd.DataFrame(results).to_csv("signal_shape_characterization.csv", index=False)
    print("\nsaved signal_shape_characterization.csv")

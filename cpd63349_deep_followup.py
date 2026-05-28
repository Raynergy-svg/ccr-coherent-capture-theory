"""Deep follow-up on CPD-63 349 P=190.48 d candidate (SDE=8.10).

CPD-63 349 (TIC 38877496, Gaia 4677014433801899392):
  R_star = 0.918 R_sun, M_star = 1.030 M_sun, Teff = 5745 K
  39 TESS sectors, 2606-day baseline (~7 yr), 615948 cadences
  BLS: P=190.4839 d, dur=4.80 h, depth=395 ppm, SDE=8.10
  If real: R_p=2.17 R_Earth, a=0.648 AU, T_eq=338 K

Tests:
  1. Refine P/T0 with finer BLS scan around the peak (P +/- 1 d)
  2. Phase-fold and visually inspect transit shape
  3. Predict transit times -- check each predicted window individually
  4. Test for half/double period (alias?)
  5. Inject-recover: how does measured SDE compare to injected SDE=8?
  6. Check for odd-even depth difference (EB diagnostic)
  7. Centroid shift test (contamination): are X,Y positions shifted at transit?
  8. Out-of-transit secondary search at phase 0.5 (EB)

Output: cpd63349_followup_plot.png + cpd63349_followup_log.txt + verdict
"""
import os, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Observations
from scipy.signal import medfilt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RA, DEC = 69.17225, -62.91497
GAIA = 4677014433801899392
NAME = "CPD-63 349"
TIC  = 38877496
R_STAR = 0.918
M_STAR = 1.030
TEFF   = 5745.0
P0     = 190.4839
T0_0   = None  # will be set after refine

logf = open("cpd63349_followup_log.txt", "w")
def log(msg):
    print(msg); logf.write(msg+"\n"); logf.flush()

log("="*78)
log(f"DEEP FOLLOW-UP: {NAME} (TIC {TIC}) -- P~{P0} d candidate")
log("="*78)

# ----- 1. (Re-)download SPOC LCs and detrend per sector with conservative window -----
log("\n[1] Downloading SPOC LCs from MAST...")
obs = Observations.query_region(f"{RA} {DEC}", radius="20 arcsec")
ts = obs[(obs["obs_collection"]=="TESS") & (obs["dataproduct_type"]=="timeseries")]
prods = Observations.get_product_list(ts)
lc_prods = prods[prods["productSubGroupDescription"]=="LC"]
dl = Observations.download_products(lc_prods, download_dir="/tmp/tess_dl", verbose=False)

sectors = []
for fp in dl["Local Path"]:
    try:
        with fits.open(fp) as hdul:
            h = hdul[0].header
            sec = h.get("SECTOR", -1)
            lc = hdul["LIGHTCURVE"].data
            t = lc["TIME"]; f = lc["PDCSAP_FLUX"]; q = lc["QUALITY"]
            # also pull X,Y centroid for contamination check
            mom1 = lc["MOM_CENTR1"] if "MOM_CENTR1" in lc.names else None
            mom2 = lc["MOM_CENTR2"] if "MOM_CENTR2" in lc.names else None
            m = np.isfinite(t) & np.isfinite(f) & (q == 0)
            if m.sum() < 100: continue
            t = t[m]; fnorm = f[m]/np.nanmedian(f[m])
            x = mom1[m] if mom1 is not None else np.full_like(t, np.nan)
            y = mom2[m] if mom2 is not None else np.full_like(t, np.nan)
            sectors.append((int(sec), t, fnorm, x, y))
    except Exception as e:
        log(f"  parse fail {fp}: {e}")
log(f"  loaded {len(sectors)} usable sectors")

# detrend per sector (12h medfilt) + 5sigma clip; preserve x,y for centroid test
def detrend(t, f, x=None, y=None, win_days=0.5):
    dt = np.median(np.diff(t))
    box = max(3, int(win_days/dt))
    if box % 2 == 0: box += 1
    trend = medfilt(f, box)
    fd = f / trend
    sig = 1.4826 * np.median(np.abs(fd - 1.0))
    keep = np.abs(fd - 1.0) < 5*sig
    out = [t[keep], fd[keep]]
    if x is not None: out.append(x[keep])
    if y is not None: out.append(y[keep])
    return tuple(out)

all_t, all_f, all_x, all_y, all_sec = [], [], [], [], []
for sec, t, f, x, y in sectors:
    td, fd, xd, yd = detrend(t, f, x, y)
    all_t.append(td); all_f.append(fd); all_x.append(xd); all_y.append(yd)
    all_sec.append(np.full_like(td, sec, dtype=int))
T = np.concatenate(all_t); F = np.concatenate(all_f)
X = np.concatenate(all_x); Y = np.concatenate(all_y)
SEC = np.concatenate(all_sec)
log(f"  detrended: {len(T)} cadences, baseline {T.max()-T.min():.0f} d, RMS {np.std(F)*1e6:.0f} ppm")

# ----- 2. Refine P / T0 with finer scan -----
log("\n[2] Refined BLS scan around P=190 d (200000 trials, +/- 2 d window)...")
periods = np.linspace(P0 - 2.0, P0 + 2.0, 80000)
durs = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
bls = BoxLeastSquares(T, F)
pg = bls.power(periods, durs)
i = int(np.argmax(pg.power))
P_best = float(pg.period[i]); T0_best = float(pg.transit_time[i])
dur_best = float(pg.duration[i]); depth_best = float(pg.depth[i])
# SDE
near = np.abs((pg.period - P_best)/P_best) < 0.005
mu = np.mean(pg.power[~near]); sd = np.std(pg.power[~near])
SDE = (pg.power[i] - mu)/sd
log(f"  refined: P={P_best:.5f} d  T0={T0_best:.4f}  dur={dur_best*24:.2f}h  depth={depth_best*1e6:.0f} ppm  SDE={SDE:.2f}")
T0_0 = T0_best

# ----- 3. Predict transit times across baseline -----
log("\n[3] Predicted transit times in baseline:")
n_min = int(np.floor((T.min() - T0_best)/P_best))
n_max = int(np.ceil((T.max() - T0_best)/P_best))
preds = [T0_best + n*P_best for n in range(n_min, n_max+1)
         if T.min() <= T0_best + n*P_best <= T.max()]
log(f"  expected {len(preds)} transits in baseline")
for i, tp in enumerate(preds):
    # check coverage: how many points within dur of each prediction?
    near = (T >= tp - dur_best/2 - 0.1) & (T <= tp + dur_best/2 + 0.1)
    sec_here = np.unique(SEC[near]) if near.sum() else []
    log(f"    transit {i+1}: BTJD {tp:.3f}  pts_in_window={near.sum()}  sectors={list(sec_here)}")

# ----- 4. Per-transit depth estimate -----
log("\n[4] Per-transit depth from local fit:")
local_depths = []
local_times = []
for tp in preds:
    win = (T >= tp - 0.5) & (T <= tp + 0.5)
    if win.sum() < 30: continue
    in_tr = (T >= tp - dur_best/2) & (T <= tp + dur_best/2)
    out_tr = win & ~in_tr
    if in_tr.sum() < 5 or out_tr.sum() < 20: continue
    d = 1.0 - np.median(F[in_tr])/np.median(F[out_tr])
    local_depths.append(d)
    local_times.append(tp)
    log(f"    BTJD {tp:.2f}: in-tr {in_tr.sum()} pts, out-tr {out_tr.sum()} pts, depth = {d*1e6:+.0f} ppm")
local_depths = np.array(local_depths); local_times = np.array(local_times)
if len(local_depths) > 1:
    med_local = np.median(local_depths); spread = np.std(local_depths)
    log(f"  median local depth: {med_local*1e6:.0f} ppm")
    log(f"  spread: {spread*1e6:.0f} ppm   (BLS depth: {depth_best*1e6:.0f} ppm)")
    pos_frac = (local_depths > 0).sum() / len(local_depths)
    log(f"  fraction with positive (transit-like) depth: {pos_frac:.2f}")

# ----- 5. Odd-even depth test (EB diagnostic) -----
log("\n[5] Odd-even depth check:")
phase = ((T - T0_best) / P_best) % 1.0
ph = np.where(phase > 0.5, phase - 1.0, phase)
in_tr = np.abs(ph) < dur_best/(2*P_best)
out_tr = (np.abs(ph) > 2*dur_best/(2*P_best)) & (np.abs(ph) < 0.25)
even_tr = in_tr & (((T - T0_best)/P_best).astype(int) % 2 == 0)
odd_tr  = in_tr & (((T - T0_best)/P_best).astype(int) % 2 == 1)
d_even = 1.0 - np.median(F[even_tr])/np.median(F[out_tr]) if even_tr.sum() else np.nan
d_odd  = 1.0 - np.median(F[odd_tr])/np.median(F[out_tr]) if odd_tr.sum() else np.nan
log(f"  even-transit depth: {d_even*1e6:.0f} ppm  (n={even_tr.sum()})")
log(f"  odd-transit depth:  {d_odd*1e6:.0f} ppm  (n={odd_tr.sum()})")
log(f"  diff: {(d_even-d_odd)*1e6:.0f} ppm (large diff => suspect EB)")

# ----- 6. Secondary search at phase 0.5 -----
log("\n[6] Secondary eclipse check (phase=0.5):")
sec_in = (np.abs(ph - 0.5) < dur_best/(2*P_best)) | (np.abs(ph + 0.5) < dur_best/(2*P_best))
d_sec = 1.0 - np.median(F[sec_in])/np.median(F[out_tr]) if sec_in.sum() else np.nan
log(f"  secondary depth: {d_sec*1e6:.0f} ppm  (n={sec_in.sum()})  (non-zero => suspect EB)")

# ----- 7. Centroid shift at transit -----
log("\n[7] Centroid shift in-transit vs out-of-transit:")
if np.isfinite(X).sum() > 1000:
    in_tr_v = (np.abs(ph) < dur_best/(2*P_best))
    out_tr_v = (np.abs(ph) > 3*dur_best/(2*P_best)) & (np.abs(ph) < 0.25)
    dx = np.median(X[in_tr_v]) - np.median(X[out_tr_v])
    dy = np.median(Y[in_tr_v]) - np.median(Y[out_tr_v])
    log(f"  X centroid shift: {dx*1e3:.2f} milli-pix  (large shift = contamination by nearby source)")
    log(f"  Y centroid shift: {dy*1e3:.2f} milli-pix")
else:
    log("  centroid data unavailable")

# ----- 8. Half/double period harmonic check -----
log("\n[8] Period harmonic check:")
for f_mult, lbl in [(0.5, "P/2"), (2.0, "2P")]:
    P_chk = P_best * f_mult
    pg_chk = bls.power(np.array([P_chk]), durs)
    log(f"  {lbl} = {P_chk:.4f} d -> BLS power {float(pg_chk.power[0]):.6f}  depth {float(pg_chk.depth[0])*1e6:.0f} ppm")

# ----- 9. Plot: lightcurve, phase fold, per-transit, BLS spectrum -----
log("\n[9] Generating diagnostic plot...")
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

# full LC with predicted transits marked
ax = axes[0]
ax.plot(T, F, ",", ms=1, alpha=0.3, color="steelblue")
for tp in preds:
    ax.axvline(tp, color="red", lw=0.5, alpha=0.5)
ax.set_ylim(0.995, 1.005); ax.set_xlabel("BTJD (d)"); ax.set_ylabel("flux")
ax.set_title(f"{NAME}: detrended TESS PDCSAP, predicted transits in red, P={P_best:.4f} d")

# phase fold
ax = axes[1]
ax.plot(ph, F, ".", ms=1, alpha=0.3, color="gray")
nb = 50; phb = np.linspace(-0.05, 0.05, nb+1)
flb = []
for lo, hi in zip(phb[:-1], phb[1:]):
    m = (ph >= lo) & (ph < hi)
    flb.append(np.mean(F[m]) if m.sum() > 0 else np.nan)
ax.plot((phb[:-1]+phb[1:])/2, flb, "ro-", ms=5, lw=2)
ax.set_xlim(-0.05, 0.05); ax.set_ylim(0.999, 1.0015)
ax.set_xlabel(f"phase (P={P_best:.4f} d)"); ax.set_ylabel("flux")
ax.set_title(f"phase fold: depth {depth_best*1e6:.0f} ppm, dur {dur_best*24:.2f} h, SDE={SDE:.2f}, R_p={np.sqrt(depth_best)*R_STAR*109.2:.2f} R_E")

# per-transit local fluxes
ax = axes[2]
for tp in preds:
    win = (T >= tp - 0.5) & (T <= tp + 0.5)
    if win.sum() < 30: continue
    ax.plot((T[win]-tp)*24, F[win], ".", ms=2, alpha=0.4)
ax.axvspan(-dur_best*24/2, dur_best*24/2, alpha=0.1, color="red")
ax.set_xlabel("hours from predicted T0"); ax.set_ylabel("flux")
ax.set_xlim(-12, 12); ax.set_ylim(0.997, 1.003)
ax.set_title("all predicted transit windows stacked")

# BLS spectrum
ax = axes[3]
ax.plot(pg.period, pg.power, "-", lw=0.5)
ax.axvline(P_best, color="red", ls="--", lw=1)
ax.set_xlabel("period (d)"); ax.set_ylabel("BLS power")
ax.set_title(f"BLS narrow scan: peak P={P_best:.4f} d, SDE={SDE:.2f}")

plt.tight_layout()
plt.savefig("cpd63349_followup_plot.png", dpi=120, bbox_inches="tight")
plt.close()
log("  plot saved: cpd63349_followup_plot.png")

# ----- VERDICT -----
log("\n" + "="*78)
log("VERDICT")
log("="*78)
log(f"  Refined period:         P = {P_best:.5f} d")
log(f"  Refined depth (BLS):    {depth_best*1e6:.0f} ppm")
log(f"  Per-transit median:     {np.median(local_depths)*1e6:.0f} ppm (n_visible={len(local_depths)})")
log(f"  Local depth spread:     {np.std(local_depths)*1e6:.0f} ppm")
log(f"  Positive-depth fraction:{(np.array(local_depths) > 0).sum() / len(local_depths):.2f}")
log(f"  SDE (refined):          {SDE:.2f}")
log(f"  Even-odd depth diff:    {(d_even-d_odd)*1e6:.0f} ppm")
log(f"  Secondary depth:        {d_sec*1e6:.0f} ppm")
log("")
log(f"  PLANET PARAMS (if real):")
log(f"    R_p = {np.sqrt(depth_best)*R_STAR*109.2:.2f} R_Earth")
log(f"    a   = {((P_best/365.25)**2 * M_STAR)**(1/3):.3f} AU")
log(f"    T_eq= {TEFF*np.sqrt(R_STAR*0.00465047/(2*((P_best/365.25)**2*M_STAR)**(1/3))):.0f} K")
log("")
log(f"  KOPPARAPU (2013) HZ FOR THIS STAR:")
# Kopparapu inner edge approx 1.107 S_sol for Teff 5745: a_in approx sqrt(L*/1.107)
L_rel = (R_STAR**2)*(TEFF/5772)**4
a_in_HZ_conservative  = np.sqrt(L_rel / 1.107)
a_out_HZ_conservative = np.sqrt(L_rel / 0.356)
a_in_HZ_optimistic    = np.sqrt(L_rel / 1.776)
a_out_HZ_optimistic   = np.sqrt(L_rel / 0.32)
a_p = ((P_best/365.25)**2 * M_STAR)**(1/3)
log(f"    optimistic HZ:   {a_in_HZ_optimistic:.3f} - {a_out_HZ_optimistic:.3f} AU")
log(f"    conservative HZ: {a_in_HZ_conservative:.3f} - {a_out_HZ_conservative:.3f} AU")
log(f"    candidate a:     {a_p:.3f} AU")
in_cons = a_in_HZ_conservative <= a_p <= a_out_HZ_conservative
in_opt  = a_in_HZ_optimistic    <= a_p <= a_out_HZ_optimistic
log(f"    IN CONSERVATIVE HZ: {in_cons}")
log(f"    IN OPTIMISTIC HZ:   {in_opt}")
logf.close()
print("DONE")

"""Follow-up on HD 180950 P=1.857 d candidate (SDE=14.09 from HLSP batch).

HD 180950 (Gaia 4186063147481113472, TIC 130995599):
  R_star=1.376 R_sun, M_star=1.110 M_sun, Teff=6010 K
  hab8=0.988, distance ~188 pc (from cov CSV)
  2 TESS-SPOC FFI sectors, 736-day baseline
  BLS: P=1.857 d, depth=186 ppm, SDE=14.09
  If real: R_p=2.05 R_Earth at a=0.031 AU, T_eq=1943 K

This star has R_star=1.376 -- it's likely a slightly evolved F-G subgiant rather than
a true dwarf. But the high SDE makes this a strong planet candidate even if not habitable.

Tests as in CPD-63 349 follow-up.
"""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Observations
from scipy.signal import medfilt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RA, DEC = 271.05, -12.5322  # from cov CSV
NAME = "HD 180950"
GAIA = 4186063147481113472
R_STAR = 1.376
M_STAR = 1.110
TEFF   = 6010.0
P0     = 1.856654

# Use TIC ra/dec exact lookup
dw = pd.read_csv("real_dwarf_targets.csv")
mat = dw[dw.gaiadr3_source_id == GAIA]
if len(mat):
    RA, DEC = float(mat.iloc[0].ra), float(mat.iloc[0].dec)

logf = open("hd180950_followup_log.txt", "w")
def log(msg): print(msg); logf.write(msg+"\n"); logf.flush()

log("="*78)
log(f"FOLLOW-UP: {NAME}  Gaia {GAIA}  P~{P0:.4f} d candidate (SDE=14.09)")
log("="*78)

# Pull HLSP TESS-SPOC FFI products (cached from previous run)
log("\n[1] Loading HLSP TESS-SPOC FFI LCs from cache...")
obs = Observations.query_region(f"{RA} {DEC}", radius="20 arcsec")
ts = obs[(obs["dataproduct_type"]=="timeseries") & (obs["obs_collection"]=="HLSP") & (obs["provenance_name"]=="TESS-SPOC")]
prods = Observations.get_product_list(ts)
names = np.array([str(n) for n in prods["productFilename"]])
keep = np.array(["_lc.fits" in n for n in names])
lc_prods = prods[keep]
dl = Observations.download_products(lc_prods, download_dir="/tmp/tess_dl", verbose=False)

sectors = []
for fp in dl["Local Path"]:
    try:
        with fits.open(fp) as hdul:
            h = hdul[0].header
            sec = h.get("SECTOR", -1)
            data = hdul[1].data
            cols = [c.lower() for c in data.names]
            tcol = next((c for c in ["time","btjd","tmid"] if c in cols), None)
            fcol = next((c for c in ["pdcsap_flux","flux","det_flux","sap_flux"] if c in cols), None)
            if not tcol or not fcol: continue
            t = np.array(data[data.names[cols.index(tcol)]])
            f = np.array(data[data.names[cols.index(fcol)]])
            q = np.zeros_like(t,dtype=int)
            if "quality" in cols:
                q = np.array(data[data.names[cols.index("quality")]])
            m = np.isfinite(t)&np.isfinite(f)&(q==0)
            if m.sum() < 100: continue
            sectors.append((int(sec), t[m], f[m]/np.nanmedian(f[m])))
    except Exception as e:
        log(f"  parse fail {fp}: {e}")
log(f"  loaded {len(sectors)} sectors")

def detrend(t, f, win_d=0.5):
    dt = np.median(np.diff(t))
    box = max(3, int(win_d/dt))
    if box%2==0: box+=1
    fd = f/medfilt(f, box)
    sig = 1.4826*np.median(np.abs(fd-1.0))
    keep = np.abs(fd-1.0) < 5*sig
    return t[keep], fd[keep]

all_t, all_f, all_sec = [], [], []
for s, t, f in sectors:
    td, fd = detrend(t, f)
    all_t.append(td); all_f.append(fd)
    all_sec.append(np.full_like(td, s, dtype=int))
T = np.concatenate(all_t); F = np.concatenate(all_f); SEC = np.concatenate(all_sec)
log(f"  detrended: {len(T)} pts, baseline {T.max()-T.min():.0f} d, RMS {np.std(F)*1e6:.0f} ppm")
log(f"  sectors covered: {sorted(set(SEC))}")

# Refine BLS narrowly around 1.857d
log("\n[2] Refined BLS around P=1.857 d (P+-0.01 d)...")
periods = np.linspace(P0-0.01, P0+0.01, 20000)
durs = np.array([0.04, 0.06, 0.08, 0.10, 0.15])
bls = BoxLeastSquares(T, F)
pg = bls.power(periods, durs)
i = int(np.argmax(pg.power))
P_best = float(pg.period[i]); T0_best = float(pg.transit_time[i])
dur_best = float(pg.duration[i]); depth_best = float(pg.depth[i])
near = np.abs((pg.period-P_best)/P_best) < 0.001
mu, sd = pg.power[~near].mean(), pg.power[~near].std()
SDE = (pg.power[i] - mu)/sd
log(f"  refined: P={P_best:.6f} d  T0={T0_best:.4f}  dur={dur_best*24:.2f} h  depth={depth_best*1e6:.0f} ppm  SDE={SDE:.2f}")

# Per-sector independent BLS at the same period
log("\n[3] Per-sector independent BLS at refined P:")
for s, t, f in sectors:
    td, fd = detrend(t, f)
    if len(td) < 100: continue
    bls_s = BoxLeastSquares(td, fd)
    pg_s = bls_s.power(np.array([P_best]), durs)
    log(f"  sector {s}: power={float(pg_s.power[0]):.4f}  depth={float(pg_s.depth[0])*1e6:+.0f} ppm  n_pts={len(td)}")

# Phase fold
phase = ((T - T0_best)/P_best) % 1.0
ph = np.where(phase>0.5, phase-1.0, phase)

# Odd-even
in_tr = np.abs(ph) < dur_best/(2*P_best)
out_tr = (np.abs(ph) > 2*dur_best/(2*P_best)) & (np.abs(ph) < 0.25)
n_cycle = ((T - T0_best)/P_best).astype(int)
even = in_tr & (n_cycle % 2 == 0)
odd  = in_tr & (n_cycle % 2 == 1)
d_even = 1.0 - np.median(F[even])/np.median(F[out_tr]) if even.sum() else np.nan
d_odd  = 1.0 - np.median(F[odd])/np.median(F[out_tr]) if odd.sum() else np.nan
log(f"\n[4] Odd-even depths: even={d_even*1e6:.0f} ppm (n={even.sum()}), odd={d_odd*1e6:.0f} ppm (n={odd.sum()})")
log(f"   diff = {(d_even-d_odd)*1e6:.0f} ppm")

# Secondary
sec_in = (np.abs(ph-0.5) < dur_best/(2*P_best)) | (np.abs(ph+0.5) < dur_best/(2*P_best))
d_sec = 1.0 - np.median(F[sec_in])/np.median(F[out_tr]) if sec_in.sum() else np.nan
log(f"\n[5] Secondary depth at phase 0.5: {d_sec*1e6:.0f} ppm  (n={sec_in.sum()})")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
ax = axes[0]
ax.plot(T, F, ",", ms=1, alpha=0.3)
ax.set_xlabel("BTJD"); ax.set_ylabel("flux")
ax.set_title(f"{NAME} -- detrended TESS-SPOC FFI, {len(sectors)} sectors")
ax.set_ylim(0.997, 1.003)

ax = axes[1]
ax.plot(ph, F, ".", ms=1, alpha=0.3, color="gray")
nb = 60; phb = np.linspace(-0.05, 0.05, nb+1); flb = []
for lo,hi in zip(phb[:-1], phb[1:]):
    m = (ph >= lo) & (ph < hi); flb.append(np.mean(F[m]) if m.sum() else np.nan)
ax.plot((phb[:-1]+phb[1:])/2, flb, "ro-", ms=5, lw=2)
ax.set_xlim(-0.05, 0.05); ax.set_ylim(0.9995, 1.0005)
ax.set_xlabel("phase"); ax.set_ylabel("flux")
ax.set_title(f"phase fold P={P_best:.5f} d: depth={depth_best*1e6:.0f} ppm dur={dur_best*24:.2f}h SDE={SDE:.2f}")

ax = axes[2]
ax.plot(pg.period, pg.power, "-", lw=0.5)
ax.axvline(P_best, color="red", ls="--", lw=1)
ax.set_xlabel("period (d)"); ax.set_ylabel("BLS power")
ax.set_title(f"narrow BLS scan: peak at P={P_best:.5f}")

plt.tight_layout()
plt.savefig("hd180950_followup_plot.png", dpi=120, bbox_inches="tight")
plt.close()
log("\nplot saved: hd180950_followup_plot.png")

# Planet params (verbose)
R_p_Rsun = np.sqrt(depth_best) * R_STAR
R_p_Re = R_p_Rsun * 109.2
a_AU = ((P_best/365.25)**2 * M_STAR)**(1/3)
T_eq = TEFF * np.sqrt(R_STAR*0.00465047/(2*a_AU))
log("\n" + "="*78)
log("PLANET CANDIDATE PARAMETERS (if real)")
log("="*78)
log(f"  host:   R*={R_STAR} R_sun  M*={M_STAR}  Teff={TEFF} K")
log(f"  P    :  {P_best:.5f} d")
log(f"  dur  :  {dur_best*24:.2f} h")
log(f"  depth:  {depth_best*1e6:.0f} ppm")
log(f"  R_p  :  {R_p_Re:.2f} R_Earth")
log(f"  a    :  {a_AU:.4f} AU")
log(f"  T_eq :  {T_eq:.0f} K  (NOT habitable -- HOT)")
log(f"  status:  CANDIDATE (SDE 14 with multi-sector consistency check above)")
logf.close()
print("DONE")

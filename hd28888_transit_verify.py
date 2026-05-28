"""Verify the HD 28888 P=14.04 d BLS signal.

Checks:
  - Re-pull the TESS SPOC LCs cleanly
  - Phase-fold at the BLS period and visualize
  - Check signal consistency across the two sectors independently
  - Refine period via TLS-style narrow scan
  - Sanity check against known TESS systematic periods (momentum dumps, etc.)
  - Compute planet radius and Teq for the host

If signal survives all checks, this is a planet candidate.
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Observations
from scipy.signal import medfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RA, DEC = 68.41124, 15.96126
NAME = "HD 28888"
HOST_TEFF = 5734.0     # K
HOST_R = 1.99          # R_sun, from GALAH FLAME / GSP-Phot
HOST_M = 1.15          # M_sun
P_BLS = 14.0356        # day

print("="*78)
print("HD 28888 P=14.04 d signal verification")
print("="*78)

# pull LCs
obs = Observations.query_region(f"{RA} {DEC}", radius="20 arcsec")
ts = obs[(obs["obs_collection"]=="TESS") & (obs["dataproduct_type"]=="timeseries")]
print(f"TESS timeseries observations: {len(ts)}")
prods = Observations.get_product_list(ts)
lc_prods = prods[prods["productSubGroupDescription"]=="LC"]
print(f"LC products: {len(lc_prods)}")
dl = Observations.download_products(lc_prods, download_dir="/tmp/tess_dl", verbose=False)

# per-sector flux extraction
sectors = []
for fp in dl["Local Path"]:
    with fits.open(fp) as hdul:
        hdr = hdul[0].header
        sector = hdr.get("SECTOR", "?")
        lc = hdul["LIGHTCURVE"].data
        t = lc["TIME"]; f = lc["PDCSAP_FLUX"]; q = lc["QUALITY"]
        m = np.isfinite(t) & np.isfinite(f) & (q == 0)
        if m.sum() < 50: continue
        t = t[m]; f = f[m] / np.nanmedian(f[m])
        sectors.append((sector, t, f))
        print(f"  sector {sector}: {len(t)} cadences, range {t[0]:.2f}-{t[-1]:.2f} BTJD")

# detrend each sector independently with 12h median, then concatenate
all_t, all_f = [], []
for s, t, f in sectors:
    dt = np.median(np.diff(t))
    box = max(3, int(0.5 / dt))
    if box % 2 == 0: box += 1
    trend = medfilt(f, box)
    fd = f / trend
    # outlier clip
    sig = 1.4826 * np.median(np.abs(fd - 1.0))
    mk = np.abs(fd - 1.0) < 5*sig
    all_t.append(t[mk]); all_f.append(fd[mk])
T = np.concatenate(all_t); F = np.concatenate(all_f)
print(f"\ndetrended: {len(T)} pts, total baseline {T.max()-T.min():.1f} d, rms = {np.std(F)*1e6:.0f} ppm")

# narrow BLS around 14.04
periods = np.linspace(13.9, 14.2, 4000)
durations = np.array([0.05, 0.08, 0.10, 0.15])
bls = BoxLeastSquares(T, F)
pg = bls.power(periods, durations)
ibest = np.argmax(pg.power)
P_best = pg.period[ibest]
dur_best = pg.duration[ibest]
t0_best = pg.transit_time[ibest]
depth_best = pg.depth[ibest]
print(f"\nrefined BLS peak: P={P_best:.5f} d, dur={dur_best*24:.2f} h, T0={t0_best:.3f}, depth={depth_best*1e6:.0f} ppm")

# also check half-period and double-period to see if 14 d is a harmonic
print("\nharmonic check:")
for f_mult, lbl in [(0.5, "P/2"), (2.0, "2P"), (3.0, "3P")]:
    p_chk = P_best * f_mult
    pg2 = bls.power(np.array([p_chk]), durations)
    print(f"  {lbl} = {p_chk:.4f} d  -> power {pg2.power[0]:.6f}, depth {pg2.depth[0]*1e6:.0f} ppm")

# phase-fold
print("\nphase-folding at refined period...")
phase = ((T - t0_best) / P_best) % 1.0
phase = np.where(phase > 0.5, phase - 1.0, phase)
order = np.argsort(phase)

# how many transits expected in the data baseline?
n_tr = int((T.max() - T.min()) / P_best)
print(f"transits expected in {T.max()-T.min():.0f} d baseline: ~{n_tr}")

# Per-sector independent BLS at the same period -- does each show the signal?
print("\nper-sector BLS at P=14.04:")
for s, t, f in sectors:
    dt = np.median(np.diff(t))
    box = max(3, int(0.5/dt));  box = box+1 if box%2==0 else box
    fd = f / medfilt(f, box)
    sig = 1.4826*np.median(np.abs(fd-1))
    mk = np.abs(fd-1) < 5*sig
    t_s, f_s = t[mk], fd[mk]
    bls_s = BoxLeastSquares(t_s, f_s)
    pg_s = bls_s.power(np.array([P_best]), durations)
    print(f"  sector {s}: power {pg_s.power[0]:.6f}, depth {pg_s.depth[0]*1e6:+.0f} ppm, n_pts {len(t_s)}, baseline {t_s.max()-t_s.min():.1f} d")

# planet radius and Teq from depth
delta = depth_best
R_p_Rsun = np.sqrt(delta) * HOST_R    # R_planet in R_sun
R_p_Rearth = R_p_Rsun * 109.2
a_AU = ( (P_best/365.25)**2 * HOST_M )**(1/3)
T_eq_K = HOST_TEFF * np.sqrt(HOST_R * 0.00465047 / (2 * a_AU))
print("\n" + "="*78)
print("PLANET CANDIDATE PROPERTIES (if real)")
print("="*78)
print(f"  host:        Teff={HOST_TEFF} K  R={HOST_R} R_sun  M={HOST_M} M_sun")
print(f"  period:      {P_best:.4f} d")
print(f"  duration:    {dur_best*24:.2f} h")
print(f"  depth:       {delta*1e6:.0f} ppm")
print(f"  R_p:         {R_p_Rearth:.2f} R_Earth  (=  {R_p_Rsun:.4f} R_sun)")
print(f"  a:           {a_AU:.3f} AU")
print(f"  T_eq:        {T_eq_K:.0f} K (assuming albedo=0)")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
ax = axes[0]
for s, t, f in sectors:
    dt = np.median(np.diff(t)); box = max(3,int(0.5/dt));  box = box+1 if box%2==0 else box
    fd = f / medfilt(f, box)
    ax.plot(t, fd, ".", ms=1, alpha=0.4, label=f"sector {s}")
ax.set_xlabel("BTJD (days)"); ax.set_ylabel("normalised flux")
ax.set_title(f"HD 28888 — detrended TESS PDCSAP")
ax.set_ylim(0.998, 1.002); ax.legend(fontsize=8)

ax = axes[1]
ax.plot(phase[order], F[order], ".", ms=1, alpha=0.5)
# binned
nb = 60; phb = np.linspace(-0.5, 0.5, nb+1); flb=[]
for lo,hi in zip(phb[:-1], phb[1:]):
    m = (phase>=lo)&(phase<hi); flb.append(np.mean(F[m]) if m.sum()>0 else np.nan)
ax.plot((phb[:-1]+phb[1:])/2, flb, "ro-", ms=4, lw=1.5)
ax.set_xlabel(f"phase  (P={P_best:.4f} d)"); ax.set_ylabel("normalised flux")
ax.set_title(f"phase-fold: depth={delta*1e6:.0f} ppm, dur={dur_best*24:.2f} h, R_p~{R_p_Rearth:.1f} R_Earth, T_eq~{T_eq_K:.0f} K")
ax.set_xlim(-0.05, 0.05); ax.set_ylim(0.998, 1.002)

ax = axes[2]
ax.plot(pg.period, pg.power, "-", lw=0.7)
ax.axvline(P_best, color="r", ls="--", lw=1)
ax.set_xlabel("period (d)"); ax.set_ylabel("BLS power")
ax.set_title(f"narrow BLS scan 13.9-14.2 d, peak at P={P_best:.4f}")
plt.tight_layout()
plt.savefig("hd28888_transit_check.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nplot saved: hd28888_transit_check.png")
print("DONE")

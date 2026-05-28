"""Test 3 supplement: fix the SDE-at-2P NaN by widening the period grid."""
import warnings, os
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Observations
from scipy.signal import medfilt

RA, DEC = 69.17225, -62.91497
P_BLS = 190.4761
T0_BLS = 1393.918
DUR_BLS = 7.20/24
R_STAR, M_STAR = 0.918, 1.030

OUT = "cpd63349_vetting.md"
out = open(OUT, "a")
def md(s=""): print(s); out.write(s+"\n")

obs = Observations.query_region(f"{RA} {DEC}", radius="20 arcsec")
ts = obs[(obs["obs_collection"]=="TESS") & (obs["dataproduct_type"]=="timeseries")]
prods = Observations.get_product_list(ts)
lc_prods = prods[prods["productSubGroupDescription"]=="LC"]
dl = Observations.download_products(lc_prods, download_dir="/tmp/tess_dl", verbose=False)

sectors = []
for fp in dl["Local Path"]:
    try:
        with fits.open(fp) as hdul:
            lc = hdul["LIGHTCURVE"].data
            t = lc["TIME"]; f = lc["PDCSAP_FLUX"]; q = lc["QUALITY"]
            m = np.isfinite(t)&np.isfinite(f)&(q==0)
            if m.sum() < 100: continue
            sectors.append((int(hdul[0].header.get("SECTOR",-1)), t[m], f[m]/np.nanmedian(f[m])))
    except: pass

def detrend(t, f):
    dt = np.median(np.diff(t)); box = max(3, int(0.5/dt)); box = box+1 if box%2==0 else box
    fd = f/medfilt(f, box); sig = 1.4826*np.median(np.abs(fd-1.0))
    return t[np.abs(fd-1.0)<5*sig], fd[np.abs(fd-1.0)<5*sig]

all_t, all_f = [], []
for s,t,f in sectors:
    td,fd = detrend(t,f); all_t.append(td); all_f.append(fd)
T = np.concatenate(all_t); F = np.concatenate(all_f)

# wider grids around 1P and 2P
md("\n### 3.1 (redo) BLS at 2P vs 1P -- wider periods for proper SDE")
bls = BoxLeastSquares(T, F)
durs = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60])

# scan a +-10% window around each
pers_1 = np.linspace(0.9*P_BLS, 1.1*P_BLS, 10000)
pg_1 = bls.power(pers_1, durs)
i1 = int(np.argmax(pg_1.power))
near1 = np.abs((pg_1.period - pg_1.period[i1])/pg_1.period[i1]) < 0.005
sde1 = (pg_1.power[i1] - pg_1.power[~near1].mean())/pg_1.power[~near1].std()
md(f"  1P peak: P={pg_1.period[i1]:.4f}, depth={pg_1.depth[i1]*1e6:.0f} ppm, dur={pg_1.duration[i1]*24:.2f}h, SDE={sde1:.2f}")

pers_2 = np.linspace(0.9*2*P_BLS, 1.1*2*P_BLS, 10000)
pg_2 = bls.power(pers_2, durs)
i2 = int(np.argmax(pg_2.power))
near2 = np.abs((pg_2.period - pg_2.period[i2])/pg_2.period[i2]) < 0.005
sde2 = (pg_2.power[i2] - pg_2.power[~near2].mean())/pg_2.power[~near2].std()
md(f"  2P peak: P={pg_2.period[i2]:.4f}, depth={pg_2.depth[i2]*1e6:.0f} ppm, dur={pg_2.duration[i2]*24:.2f}h, SDE={sde2:.2f}")

ratio = sde2/sde1 if sde1 > 0 else float("nan")
md(f"  SDE(2P)/SDE(1P) = {ratio:.2f}")
if ratio < 0.7:
    verdict = "PASS: 2P much weaker than 1P -- period is genuinely 1P, EB-at-2P hypothesis disfavoured ✓"
elif ratio < 1.3:
    verdict = "INCONCLUSIVE: SDEs comparable, period could be 1P (planet) or 2P (EB)"
else:
    verdict = "FAIL: 2P SDE exceeds 1P -- true period is 2P with similar primary+secondary => EB ✗"
md(f"  {verdict}")
out.close()
print("Test 3.1 redo done")

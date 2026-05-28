"""TESS BLS transit search on chemistry-priority targets (positional query)."""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Observations
from scipy.signal import medfilt

TARGETS = [
    ("HD 183193",  292.32463, -24.32321, "main-sequence dwarf, 75 pc, G=8.78, hab8=0.991"),
    ("HD 28888",    68.41124,  15.96126, "subgiant, 100 pc, G=8.24, hab9=0.973 (#1 actionable)"),
    ("CD-30 7056", 134.83005, -30.63719, "main-sequence dwarf, 100 pc, G=9.24, hab8=0.995 (MC-robust)"),
    ("BD-03 3321", 189.28275,  -4.08161, "main-sequence dwarf, 168 pc, G=10.32, hab8=0.997 (highest 8D)"),
]

def get_lc(ra, dec, name):
    obs = Observations.query_region(f"{ra} {dec}", radius="20 arcsec")
    if len(obs) == 0: return None, "no MAST hits"
    ts = obs[(obs["obs_collection"] == "TESS") & (obs["dataproduct_type"] == "timeseries")]
    if len(ts) == 0: return None, f"no TESS timeseries (only {len(obs)} other products incl. FFI)"
    print(f"  TESS timeseries observations: {len(ts)}")
    # get LC product list
    products = Observations.get_product_list(ts)
    spoc_lc = products[(products["productSubGroupDescription"] == "LC")]
    if len(spoc_lc) == 0:
        # fall back to anything with "lc.fits"
        spoc_lc = products[products["productFilename"].astype(str).str.contains("_lc.fits", na=False)]
    print(f"  LC products: {len(spoc_lc)}")
    if len(spoc_lc) == 0:
        return None, "no SPOC LC products"
    dl = Observations.download_products(spoc_lc[:8], download_dir="/tmp/tess_dl", verbose=False)
    if dl is None or len(dl) == 0: return None, "download failed"
    times, fluxes = [], []
    for fp in dl["Local Path"]:
        try:
            with fits.open(fp) as hdul:
                lc = hdul["LIGHTCURVE"].data
                t = lc["TIME"]; f = lc["PDCSAP_FLUX"]; q = lc["QUALITY"]
                m = np.isfinite(t) & np.isfinite(f) & (q == 0)
                if m.sum() < 50: continue
                times.append(t[m])
                fluxes.append(f[m] / np.nanmedian(f[m]))
        except Exception as e:
            print(f"    parse fail: {e}")
    if not times: return None, "no usable LC files"
    return (np.concatenate(times), np.concatenate(fluxes)), f"{len(dl)} LC files"

def bls(t, f):
    # outlier clip
    med, mad = np.median(f), np.median(np.abs(f - np.median(f)))
    m = np.abs(f - med) < 5 * 1.4826 * mad
    t, f = t[m], f[m]
    # detrend with 12-hour median
    dt = np.median(np.diff(t))
    box = max(3, int(0.5 / dt))
    if box % 2 == 0: box += 1
    if box >= 5: f = f / medfilt(f, box)
    # BLS scan
    pmax = min(15.0, (t.max() - t.min()) / 2.5)
    periods = np.linspace(0.5, pmax, 8000)
    durations = np.array([0.05, 0.1, 0.15, 0.2])
    bls = BoxLeastSquares(t, f)
    pg = bls.power(periods, durations)
    # top 3 distinct peaks
    sorted_idx = np.argsort(-pg.power)
    picks = []
    for i in sorted_idx:
        P = pg.period[i]
        if any(abs(P - p["P"]) / p["P"] < 0.02 or abs(1/P - 1/p["P"]) < 0.01 for p in picks): continue
        picks.append(dict(P=P, dur=pg.duration[i], depth=pg.depth[i], pwr=pg.power[i]))
        if len(picks) >= 3: break
    rms = np.std(f) * 1e6
    base = t.max() - t.min()
    return picks, rms, base, len(t)

results = []
for name, ra, dec, note in TARGETS:
    print(f"\n=== {name}  ({ra:.4f}, {dec:.4f}) — {note} ===")
    out, msg = get_lc(ra, dec, name)
    if out is None:
        print(f"  -> {msg}")
        results.append(dict(name=name, status=msg))
        continue
    t, f = out
    print(f"  {msg}, {len(t)} cadences, baseline {t.max()-t.min():.1f} d")
    picks, rms, base, npts = bls(t, f)
    print(f"  BLS top-3 distinct candidates:")
    for i, p in enumerate(picks, 1):
        # rough SNR: depth / (rms / sqrt(n_transit_obs))
        n_tr = int(base / p["P"])
        n_pt_per_tr = int(p["dur"] / np.median(np.diff(t)))
        snr = p["depth"] * 1e6 / (rms / np.sqrt(max(n_tr * n_pt_per_tr, 1)))
        print(f"    #{i}: P={p['P']:.4f} d  dur={p['dur']*24:.2f} h  depth={p['depth']*1e6:.0f} ppm  "
              f"pwr={p['pwr']:.4f}  SNR~{snr:.1f}")
        results.append(dict(name=name, rank=i, period_d=p['P'], duration_h=p['dur']*24,
                            depth_ppm=p['depth']*1e6, bls_power=p['pwr'], snr=snr,
                            baseline_d=base, rms_ppm=rms, n_pts=npts))

if results:
    R = pd.DataFrame(results)
    R.to_csv("tess_bls_results.csv", index=False)
    print("\n" + "="*78)
    print("BLS SUMMARY (sorted by SNR)")
    print("="*78)
    sorted_R = R.dropna(subset=["snr"]).sort_values("snr", ascending=False)
    print(sorted_R.to_string(index=False))
print("DONE")

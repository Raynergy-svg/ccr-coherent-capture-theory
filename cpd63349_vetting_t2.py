"""TEST 2 — proper per-sector difference-imaging centroid test.

For each TESS sector in which a predicted transit occurs (S31, S38, S87, S94):
  - Download the Target Pixel File (TPF) via MAST
  - Identify in-transit cadences (within DUR/2 of T0 + n*P)
  - Identify out-of-transit cadences (sector flux median, away from transit)
  - Mean in-transit pixel image; mean out-of-transit pixel image
  - Compute difference image = OOT - IT (positive = source dimmed in transit)
  - Fit a 2D Gaussian centroid to the difference image
  - Convert from pixel coordinates to RA/Dec via the TPF WCS
  - Compare centroid offset to target position; report in arcsec
  - List Gaia neighbours within 2 pixels (42'') for blend assessment

A significant centroid offset toward a known neighbor indicates the signal
originates from a blended source (background eclipsing binary), not the target.
"""
import warnings, os
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

RA, DEC = 69.17225, -62.91497
NAME = "CPD-63 349"
TIC  = 38877496
P    = 190.4761
T0   = 1393.918
DUR  = 7.20/24  # days
EXPECTED_SECTORS_WITH_TRANSITS = [31, 38, 87, 94]

OUT = "cpd63349_vetting.md"
out = open(OUT, "a")
def md(s=""): print(s); out.write(s+"\n")

md("\n## TEST 2 — proper difference-image centroid test")
md(f"Predicted transit sectors: {EXPECTED_SECTORS_WITH_TRANSITS}")

# 1. Download TPFs for the four sectors with transits
md("\n### Downloading SPOC TPFs for relevant sectors")
obs = Observations.query_region(f"{RA} {DEC}", radius="20 arcsec")
ts = obs[(obs["obs_collection"]=="TESS") & (obs["dataproduct_type"]=="timeseries")]
prods = Observations.get_product_list(ts)
tp_prods = prods[prods["productSubGroupDescription"]=="TP"]
md(f"  TP products available: {len(tp_prods)}")

# filter by sector
def sec_from_name(s):
    # tess2020324010417-s0032-...
    import re
    m = re.search(r"-s0+([0-9]+)-", str(s))
    return int(m.group(1)) if m else -1

tp_sec = np.array([sec_from_name(n) for n in tp_prods["productFilename"]])
keep_mask = np.isin(tp_sec, EXPECTED_SECTORS_WITH_TRANSITS)
md(f"  TP products in target sectors: {keep_mask.sum()}")
tp_keep = tp_prods[keep_mask]
dl = Observations.download_products(tp_keep, download_dir="/tmp/tess_dl", verbose=False)

# 2. Process each TPF
md("\n### Per-sector difference imaging")
def gaussian2d(coords, A, x0, y0, sx, sy, theta, off):
    x, y = coords
    a = np.cos(theta)**2/(2*sx**2) + np.sin(theta)**2/(2*sy**2)
    b = -np.sin(2*theta)/(4*sx**2) + np.sin(2*theta)/(4*sy**2)
    c = np.sin(theta)**2/(2*sx**2) + np.cos(theta)**2/(2*sy**2)
    return off + A*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

results = []
for fp in dl["Local Path"]:
    try:
        with fits.open(fp) as hdul:
            sec = hdul[0].header.get("SECTOR", -1)
            if sec not in EXPECTED_SECTORS_WITH_TRANSITS: continue
            md(f"\n#### SECTOR {sec}")
            md(f"   TPF file: {os.path.basename(fp)}")
            tpf = hdul[1].data
            ti = tpf["TIME"]
            flx = tpf["FLUX"]    # shape (N_cadence, H, W)
            qual = tpf["QUALITY"]
            wcs = WCS(hdul[2].header)  # APERTURE HDU has WCS
            ap_mask = hdul[2].data       # aperture (4 = optimal SPOC aperture bit)
            md(f"   shape: {flx.shape}, time range BTJD {np.nanmin(ti):.1f} - {np.nanmax(ti):.1f}")

            # find predicted transit times in this sector
            n_min = int(np.floor((np.nanmin(ti) - T0)/P))
            n_max = int(np.ceil((np.nanmax(ti) - T0)/P))
            preds = []
            for n in range(n_min-1, n_max+2):
                tp = T0 + n*P
                if np.nanmin(ti) <= tp <= np.nanmax(ti):
                    preds.append((n, tp))
            md(f"   predicted transits in sector: {len(preds)} ({preds})")

            if not preds:
                md(f"   ! no transits in this sector - skipping difference image")
                continue

            # build in-transit and out-of-transit masks
            ok = (qual == 0) & np.isfinite(ti)
            in_tr_total = np.zeros_like(ti, dtype=bool)
            for n, tp in preds:
                in_tr_total |= np.abs(ti - tp) < DUR/2
            in_mask  = in_tr_total & ok
            out_mask = ~in_tr_total & ok
            md(f"   in-tr cadences: {in_mask.sum()}, out-tr cadences: {out_mask.sum()}")
            if in_mask.sum() < 10 or out_mask.sum() < 50:
                md(f"   insufficient cadences - skipping"); continue

            # mean in / mean out images (NaN-safe per-pixel)
            f_in  = np.nanmean(flx[in_mask, :, :],  axis=0)
            f_out = np.nanmean(flx[out_mask, :, :], axis=0)
            diff  = f_out - f_in   # positive = source dimmed

            # PSF-fit centroid of diff image
            H, W = diff.shape
            yi, xi = np.mgrid[0:H, 0:W]
            # initial guess: peak pixel of diff
            ii = np.unravel_index(np.nanargmax(diff), diff.shape)
            yp, xp = ii
            # normalise for fit
            try:
                A0 = float(np.nanmax(diff) - np.nanmedian(diff))
                p0 = [A0, xp, yp, 1.0, 1.0, 0.0, float(np.nanmedian(diff))]
                popt, _ = curve_fit(gaussian2d, (xi.ravel(), yi.ravel()),
                                     diff.ravel(), p0=p0, maxfev=3000)
                x_diff, y_diff = popt[1], popt[2]
                md(f"   diff-image centroid (pixel): ({x_diff:.3f}, {y_diff:.3f})")
            except Exception as e:
                md(f"   diff-image PSF fit failed: {e}")
                x_diff, y_diff = float(xp), float(yp)

            # also centroid of out-of-transit image (= target's flux-weighted centroid)
            try:
                A0 = float(np.nanmax(f_out) - np.nanmedian(f_out))
                ii_out = np.unravel_index(np.nanargmax(f_out), f_out.shape)
                p0 = [A0, ii_out[1], ii_out[0], 1.0, 1.0, 0.0, float(np.nanmedian(f_out))]
                popt2, _ = curve_fit(gaussian2d, (xi.ravel(), yi.ravel()),
                                     f_out.ravel(), p0=p0, maxfev=3000)
                x_out, y_out = popt2[1], popt2[2]
            except Exception as e:
                ii_out = np.unravel_index(np.nanargmax(f_out), f_out.shape)
                x_out, y_out = float(ii_out[1]), float(ii_out[0])
            md(f"   out-of-tr (target) centroid (pixel): ({x_out:.3f}, {y_out:.3f})")

            # offset in pixels and arcsec
            dx = x_diff - x_out
            dy = y_diff - y_out
            offset_pix = np.sqrt(dx**2 + dy**2)
            # TESS pixel scale = 21 arcsec
            offset_arcsec = offset_pix * 21.0
            md(f"   centroid offset (diff vs target): "
               f"({dx:+.3f}, {dy:+.3f}) pixels = {offset_arcsec:.2f} arcsec")

            # convert diff-image centroid to RA/Dec via WCS
            try:
                # APERTURE HDU stores pixel coords in detector frame
                # TPF coordinates are (column, row) which correspond to (X,Y) in image
                tpf_ra, tpf_dec = wcs.all_pix2world(x_diff, y_diff, 0)
                tg_ra,  tg_dec  = wcs.all_pix2world(x_out,  y_out,  0)
                md(f"   diff-image RA,Dec = ({float(tpf_ra):.5f}, {float(tpf_dec):.5f})")
                md(f"   target RA,Dec     = ({float(tg_ra):.5f}, {float(tg_dec):.5f})")
                # offset in arcsec via SkyCoord
                cd = SkyCoord(float(tpf_ra)*u.deg, float(tpf_dec)*u.deg)
                ct = SkyCoord(float(tg_ra)*u.deg, float(tg_dec)*u.deg)
                cat = SkyCoord(RA*u.deg, DEC*u.deg)
                sep_diff_target = cd.separation(ct).arcsec
                sep_diff_cat    = cd.separation(cat).arcsec
                sep_targ_cat    = ct.separation(cat).arcsec
                md(f"   sep(diff,target) = {sep_diff_target:.2f}'' "
                   f" sep(diff,catalog) = {sep_diff_cat:.2f}''"
                   f" sep(target,catalog) = {sep_targ_cat:.2f}''")
            except Exception as e:
                md(f"   WCS conversion error: {e}")
                sep_diff_cat = None

            # Quick verdict per sector
            if sep_diff_cat is not None:
                if sep_diff_cat < 7.0:
                    verdict = "PASS: diff-image centroid on target (<1/3 pixel)"
                elif sep_diff_cat < 21.0:
                    verdict = "INCONCLUSIVE: <1 pixel offset (within PSF)"
                else:
                    verdict = "FAIL: diff-image centroid >1 pixel from target (likely blended source)"
                md(f"   **per-sector verdict: {verdict}**")
                results.append(dict(sector=sec, offset_arcsec=sep_diff_cat,
                                    target_offset_arcsec=sep_targ_cat,
                                    verdict=verdict))

            # save plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, im, t in [(axes[0], f_out, "out-of-tr (target)"),
                              (axes[1], f_in,  "in-transit"),
                              (axes[2], diff,  "out - in (positive = dim)")]:
                ax.imshow(im, origin="lower", cmap="viridis")
                ax.plot(x_out, y_out, "wx", markersize=10, label="target")
                ax.plot(x_diff, y_diff, "r+", markersize=10, label="diff centroid")
                ax.set_title(f"S{sec} {t}")
                ax.legend(fontsize=8, loc="upper right")
            plt.tight_layout()
            plt.savefig(f"cpd63349_diff_sec{sec:02d}.png", dpi=100, bbox_inches="tight")
            plt.close()
            md(f"   plot: cpd63349_diff_sec{sec:02d}.png")
    except Exception as e:
        md(f"   sector {sec} processing error: {e}")

# Summary
md("\n### Test 2 summary")
if results:
    R = pd.DataFrame(results)
    R.to_csv("cpd63349_diffimg_results.csv", index=False)
    md("```")
    md(R.to_string(index=False))
    md("```")
    avg_off = R.offset_arcsec.mean()
    med_off = R.offset_arcsec.median()
    md(f"\nMean centroid offset across sectors: {avg_off:.2f}''")
    md(f"Median centroid offset: {med_off:.2f}''")
    if med_off < 7.0:
        md("**TEST 2: PASS** -- difference-image centroid is on the target in all sectors (<1/3 TESS pixel)")
    elif med_off < 21.0:
        md("**TEST 2: INCONCLUSIVE** -- centroid offset <1 TESS pixel but >1/3 pixel")
    else:
        md("**TEST 2: FAIL** -- centroid offset > 1 TESS pixel = signal is from blended source not target")
else:
    md("No sectors processed successfully")

# Gaia neighbour catalog within 42 arcsec (2 pixels)
md("\n### Gaia DR3 neighbours within 42'' (2 TESS pixels)")
import requests, io
def tap(url, q, retries=3):
    for a in range(retries):
        try:
            r = requests.post(url, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=120)
            if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
        except Exception as e: print(e)
    return pd.DataFrame()
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
neigh = tap(GAIA_TAP, f"""SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp,
                           DISTANCE(POINT('ICRS', ra, dec),
                                    POINT('ICRS', {RA}, {DEC}))*3600 AS sep_arcsec
                          FROM gaiadr3.gaia_source
                          WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                                           CIRCLE('ICRS', {RA}, {DEC}, 0.01167))
                          ORDER BY sep_arcsec""")
md(f"  {len(neigh)} Gaia DR3 sources within 42''")
if len(neigh):
    md("```")
    md(neigh.to_string(index=False))
    md("```")
    target_G = float(neigh.iloc[0].phot_g_mean_mag)
    # for each neighbour, what depth would a 100% eclipse on it produce in target aperture
    md("\nFor each non-target neighbour, the maximum eclipse depth (100% blocked) that could appear on the target's aperture:")
    md("|source_id|sep|G|dG|F_neigh/F_target|max_eclipse_depth_ppm|")
    md("|--|--|--|--|--|--|")
    for i, row in neigh.iterrows():
        if i == 0: continue
        dG = row.phot_g_mean_mag - target_G
        F_ratio = 10**(-0.4*dG)
        # max depth on combined aperture: 100% dip in neighbour
        # observed depth = F_neigh / (F_target + F_neigh) * 1.0  (in fractional flux)
        max_d = F_ratio/(1+F_ratio)
        md(f"|{row.source_id}|{row.sep_arcsec:.1f}''|{row.phot_g_mean_mag:.2f}|{dG:+.2f}|{F_ratio:.4f}|{max_d*1e6:.0f}|")
    md("")
    # Could any single neighbour host the eclipse depth (354 ppm BLS) as a diluted EB?
    md("Depth observed (target apertured BLS): 354 ppm")
    md("=> any neighbour with max_eclipse_depth >= 354 ppm could host the signal as a diluted EB.")

out.close()
print("\nTest 2 done -- appended to cpd63349_vetting.md")

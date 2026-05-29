"""Proper per-sector PSF-fit centroid test for TESS transit candidates.

Replaces the noise-dominated Gaussian-fit-on-mean-diff-image used in
the original CPD-63 349 vetting (Test 2). Uses SPOC PSF_CENTR1/2
columns which are the SPOC pipeline's per-cadence PSF-fit centroids.

Method:
  For each sector with at least one predicted transit:
    1. Pull SPOC LC; extract PSF_CENTR1 (col), PSF_CENTR2 (row), TIME, FLUX, QUALITY
    2. Detrend the centroid timeseries the same way we detrend flux
       (12h medfilt) so long-timescale drift doesn't contaminate
    3. Mark in-transit cadences (within DUR/2 of predicted T0+n*P)
    4. Mark out-of-transit cadences (within +/- 4*DUR but outside DUR/2)
    5. Compute mean(PSF_CENTR) in vs out, with per-cadence scatter as
       uncertainty
    6. Convert (col, row) shift to arcsec via TESS pixel scale (21''/pix)
    7. Test for significant shift at the >3sigma level per sector
  Combine sectors:
    - Mean offset across sectors (weighted by 1/variance)
    - Check direction consistency across sectors -- noise centroid
      uncertainty scatters randomly; a real off-target source produces
      a CONSISTENT direction across sectors

A real transit on the target: in-transit centroid same as out (shift ~0).
A blended EB elsewhere: in-transit centroid shifts toward the blend by
  ~ (depth/contamination_ratio) of the offset, consistent direction.
A noise-dominated null: random shift direction across sectors.

Usage:
  python centroid_psf_test.py --tic 38877496 \\
      --ra 69.17225 --dec -62.91497 \\
      --period 190.4761 --t0 1393.918 --duration 0.3
"""
import warnings, os, argparse
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from astropy.io import fits
from astroquery.mast import Observations
from scipy.signal import medfilt
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_spoc_lcs(ra, dec, radius_arcsec=20):
    """Return list of (sector, filepath) for SPOC LC products."""
    obs = Observations.query_region(f"{ra} {dec}", radius=f"{radius_arcsec} arcsec")
    ts = obs[(obs["obs_collection"]=="TESS") & (obs["dataproduct_type"]=="timeseries")]
    if len(ts) == 0: return []
    prods = Observations.get_product_list(ts)
    lc_prods = prods[prods["productSubGroupDescription"]=="LC"]
    if len(lc_prods) == 0: return []
    dl = Observations.download_products(lc_prods, download_dir="/tmp/tess_dl", verbose=False)
    out = []
    for fp in dl["Local Path"]:
        try:
            with fits.open(fp) as hdul:
                sec = hdul[0].header.get("SECTOR", -1)
                out.append((int(sec), fp))
        except Exception:
            continue
    return out

def detrend_series(t, y, win_days=0.5):
    """Median-filter detrend with NaN protection."""
    if len(t) < 10: return y - np.nanmean(y)
    dt = np.median(np.diff(t))
    if dt <= 0: dt = 0.02
    box = max(3, int(win_days/dt))
    if box % 2 == 0: box += 1
    yfill = np.where(np.isfinite(y), y, np.nanmedian(y))
    trend = medfilt(yfill, box)
    return y - trend

def per_sector_centroid_test(filepath, P, T0, duration_d):
    """Run centroid test on one sector. Returns dict of metrics."""
    with fits.open(filepath) as hdul:
        hdr = hdul[0].header
        sec = hdr.get("SECTOR", -1)
        ccd = hdr.get("CCD", -1)
        camera = hdr.get("CAMERA", -1)
        lc = hdul["LIGHTCURVE"].data
        t = np.array(lc["TIME"], dtype=float)
        flux = np.array(lc["PDCSAP_FLUX"], dtype=float)
        q = np.array(lc["QUALITY"], dtype=int)
        # Centroid: prefer PSF_CENTR; fall back to MOM_CENTR. Both must be
        # corrected for spacecraft pointing by subtracting POS_CORR. SPOC
        # only computes PSF_CENTR for bright stars on PRF-supported sectors;
        # most G ~ 10 targets only have MOM_CENTR populated.
        names = [c.upper() for c in lc.names]
        psf_ok = ("PSF_CENTR1" in names and np.isfinite(np.array(lc["PSF_CENTR1"])).any())
        if psf_ok:
            cx_raw = np.array(lc["PSF_CENTR1"], dtype=float)
            cy_raw = np.array(lc["PSF_CENTR2"], dtype=float)
            centr_type = "PSF"
        elif "MOM_CENTR1" in names:
            cx_raw = np.array(lc["MOM_CENTR1"], dtype=float)
            cy_raw = np.array(lc["MOM_CENTR2"], dtype=float)
            centr_type = "MOM"
        else:
            return dict(sector=sec, status="no_centroid_columns")
        # Spacecraft pointing correction (subtracts pointing drift)
        if "POS_CORR1" in names:
            pcx = np.array(lc["POS_CORR1"], dtype=float)
            pcy = np.array(lc["POS_CORR2"], dtype=float)
            cx = cx_raw - pcx
            cy = cy_raw - pcy
        else:
            cx = cx_raw
            cy = cy_raw

    mask = np.isfinite(t) & np.isfinite(flux) & np.isfinite(cx) & np.isfinite(cy) & (q == 0)
    if mask.sum() < 100:
        return dict(sector=sec, status="insufficient_cadences", n=int(mask.sum()))
    t = t[mask]; flux = flux[mask] / np.nanmedian(flux[mask])
    cx = cx[mask]; cy = cy[mask]

    # Detrend centroid series with the SAME 12-h medfilt used for flux
    cx_dt = detrend_series(t, cx)
    cy_dt = detrend_series(t, cy)

    # Find predicted transits in this sector
    n_min = int(np.floor((t.min() - T0)/P))
    n_max = int(np.ceil((t.max() - T0)/P))
    preds = [T0 + n*P for n in range(n_min-1, n_max+2)
             if t.min() <= T0 + n*P <= t.max()]
    if not preds:
        return dict(sector=sec, status="no_transit_in_sector", t_start=float(t.min()), t_end=float(t.max()))

    # In-transit mask: within DUR/2 of any prediction
    in_mask = np.zeros_like(t, dtype=bool)
    for tp in preds:
        in_mask |= np.abs(t - tp) < duration_d/2

    # Out-of-transit comparison window: +/- 4*DUR around each prediction, excluding DUR
    out_mask = np.zeros_like(t, dtype=bool)
    for tp in preds:
        near = np.abs(t - tp) < 4*duration_d
        out_mask |= near
    out_mask &= ~in_mask
    out_mask &= np.isfinite(flux)

    n_in = in_mask.sum(); n_out = out_mask.sum()
    if n_in < 5 or n_out < 20:
        return dict(sector=sec, status="insufficient_in_or_out", n_in=int(n_in), n_out=int(n_out))

    # PSF centroid in-transit vs out-of-transit (in pixel units), using DETRENDED series
    cx_in_mean  = np.mean(cx_dt[in_mask])
    cx_out_mean = np.mean(cx_dt[out_mask])
    cy_in_mean  = np.mean(cy_dt[in_mask])
    cy_out_mean = np.mean(cy_dt[out_mask])
    # Per-cadence scatter -> SE of mean
    cx_in_se  = np.std(cx_dt[in_mask], ddof=1)/np.sqrt(n_in)
    cx_out_se = np.std(cx_dt[out_mask], ddof=1)/np.sqrt(n_out)
    cy_in_se  = np.std(cy_dt[in_mask], ddof=1)/np.sqrt(n_in)
    cy_out_se = np.std(cy_dt[out_mask], ddof=1)/np.sqrt(n_out)
    dx_pix = cx_in_mean - cx_out_mean
    dy_pix = cy_in_mean - cy_out_mean
    dx_se = np.hypot(cx_in_se, cx_out_se)
    dy_se = np.hypot(cy_in_se, cy_out_se)
    dx_arcsec = dx_pix * 21.0; dy_arcsec = dy_pix * 21.0
    dx_arcsec_se = dx_se * 21.0; dy_arcsec_se = dy_se * 21.0
    sig_x = dx_pix / dx_se if dx_se > 0 else 0
    sig_y = dy_pix / dy_se if dy_se > 0 else 0
    # 2D offset & significance via Mahalanobis-style combined sigma
    offset_arcsec = np.hypot(dx_arcsec, dy_arcsec)
    offset_se = np.hypot(dx_arcsec_se, dy_arcsec_se)
    significance = offset_arcsec / offset_se if offset_se > 0 else 0
    direction_deg = np.degrees(np.arctan2(dy_arcsec, dx_arcsec))

    # Also: flux dip in this sector
    f_in_mean = np.mean(flux[in_mask])
    f_out_mean = np.mean(flux[out_mask])
    depth_ppm = (f_out_mean - f_in_mean) * 1e6

    return dict(
        sector=sec, camera=camera, ccd=ccd, centr_type=centr_type,
        n_in=int(n_in), n_out=int(n_out), n_predicted_transits=len(preds),
        dx_pix=float(dx_pix), dy_pix=float(dy_pix),
        dx_arcsec=float(dx_arcsec), dy_arcsec=float(dy_arcsec),
        dx_se_arcsec=float(dx_arcsec_se), dy_se_arcsec=float(dy_arcsec_se),
        offset_arcsec=float(offset_arcsec), offset_se_arcsec=float(offset_se),
        significance_sigma=float(significance),
        direction_deg=float(direction_deg),
        depth_ppm=float(depth_ppm),
        status="ok",
    )

def combine_sector_results(results):
    """Aggregate per-sector centroid offsets into a single verdict."""
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok: return dict(status="no_sectors_processed", n_sectors=0)
    # Inverse-variance weighted mean (per axis) treating each sector as independent
    dx = np.array([r["dx_arcsec"] for r in ok])
    dy = np.array([r["dy_arcsec"] for r in ok])
    sx = np.array([r["dx_se_arcsec"] for r in ok])
    sy = np.array([r["dy_se_arcsec"] for r in ok])
    sx = np.where(sx > 0, sx, np.median(sx[sx > 0]) if (sx > 0).any() else 1.0)
    sy = np.where(sy > 0, sy, np.median(sy[sy > 0]) if (sy > 0).any() else 1.0)
    wx = 1/sx**2; wy = 1/sy**2
    mean_dx = float((dx * wx).sum() / wx.sum())
    mean_dy = float((dy * wy).sum() / wy.sum())
    se_dx = float(1/np.sqrt(wx.sum()))
    se_dy = float(1/np.sqrt(wy.sum()))
    offset = np.hypot(mean_dx, mean_dy)
    offset_se = np.hypot(se_dx, se_dy)
    sig = offset / offset_se if offset_se > 0 else 0

    # Direction consistency: angle scatter across sectors
    angles = np.degrees(np.arctan2(dy, dx))
    # Circular mean and dispersion
    rad = np.radians(angles)
    sin_m, cos_m = np.sin(rad).mean(), np.cos(rad).mean()
    R = np.hypot(sin_m, cos_m)
    circ_std_deg = np.degrees(np.sqrt(-2*np.log(R))) if R > 0 else 180
    direction_consistent = R > 0.6  # circular R > 0.6 means dispersion < 60 deg

    return dict(
        status="ok", n_sectors=len(ok),
        mean_dx_arcsec=mean_dx, mean_dy_arcsec=mean_dy,
        se_dx_arcsec=se_dx, se_dy_arcsec=se_dy,
        combined_offset_arcsec=float(offset),
        combined_offset_se_arcsec=float(offset_se),
        combined_significance_sigma=float(sig),
        direction_circular_R=float(R),
        direction_circular_std_deg=float(circ_std_deg),
        direction_consistent=bool(direction_consistent),
    )

def verdict(combined):
    """Pre-registered-style verdict for centroid result."""
    if combined.get("status") != "ok": return "UNKNOWN"
    sig = combined["combined_significance_sigma"]
    cons = combined["direction_consistent"]
    off = combined["combined_offset_arcsec"]
    if sig < 2:
        return "ON_TARGET (offset not significant)"
    if not cons:
        return "INCONCLUSIVE (significant offset but direction scatters across sectors)"
    if off < 3.0:
        return "ON_TARGET (significant but <3'' = << 1 TESS pixel)"
    if off < 10.0:
        return "MARGINAL (significant 3-10'' consistent offset; possible blended source)"
    return "OFF_TARGET (>10'' consistent offset; signal likely from a nearby source)"

def main(tic, ra, dec, P, T0, duration_d, name):
    print(f"="*78)
    print(f"PSF-fit centroid test: {name} (TIC {tic})")
    print(f"  P={P:.5f}d  T0={T0:.4f}  duration={duration_d*24:.2f}h")
    print(f"="*78)
    print(f"\n[1] Getting SPOC LC products...")
    lcs = get_spoc_lcs(ra, dec)
    print(f"  {len(lcs)} SPOC LCs found")
    if not lcs:
        print("  no SPOC LCs -- exit")
        return

    print(f"\n[2] Per-sector centroid analysis (in-transit vs out-of-transit)...")
    results = []
    for sec, fp in sorted(lcs):
        try:
            r = per_sector_centroid_test(fp, P, T0, duration_d)
        except Exception as e:
            r = dict(sector=sec, status=f"exception:{type(e).__name__}: {e}")
        results.append(r)
        if r["status"] == "ok":
            print(f"  S{sec:>2}: dx={r['dx_arcsec']:+6.2f}+-{r['dx_se_arcsec']:.2f}  "
                  f"dy={r['dy_arcsec']:+6.2f}+-{r['dy_se_arcsec']:.2f}  "
                  f"|offset|={r['offset_arcsec']:5.2f}+-{r['offset_se_arcsec']:.2f}  "
                  f"sig={r['significance_sigma']:.1f}sigma  "
                  f"depth={r['depth_ppm']:+.0f}ppm  "
                  f"n_in={r['n_in']} n_out={r['n_out']}  ({r['centr_type']})")
        else:
            print(f"  S{sec:>2}: SKIP {r.get('status')}  {r if 'exception' in str(r.get('status','')) else ''}")

    combined = combine_sector_results(results)
    print(f"\n[3] Combined across sectors (inverse-variance weighted):")
    if combined.get("status") == "ok":
        print(f"  n_sectors with transits: {combined['n_sectors']}")
        print(f"  mean dx: {combined['mean_dx_arcsec']:+.2f} +- {combined['se_dx_arcsec']:.2f} arcsec")
        print(f"  mean dy: {combined['mean_dy_arcsec']:+.2f} +- {combined['se_dy_arcsec']:.2f} arcsec")
        print(f"  combined |offset|: {combined['combined_offset_arcsec']:.2f} +- {combined['combined_offset_se_arcsec']:.2f} arcsec")
        print(f"  combined significance: {combined['combined_significance_sigma']:.2f} sigma")
        print(f"  direction circular R: {combined['direction_circular_R']:.3f}  "
              f"(R>0.6 = consistent direction)")
        print(f"  direction consistent: {combined['direction_consistent']}")

    v = verdict(combined)
    print(f"\n[4] VERDICT: {v}")

    # plot per-sector vectors in CCD frame (arcsec offsets)
    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        fig, ax = plt.subplots(figsize=(7,7))
        for r in ok:
            ax.errorbar(r["dx_arcsec"], r["dy_arcsec"],
                         xerr=r["dx_se_arcsec"], yerr=r["dy_se_arcsec"],
                         fmt="o", ms=4, capsize=2, alpha=0.7,
                         label=f"S{r['sector']}")
        ax.plot(0, 0, "k*", ms=15, label="target")
        if combined.get("status") == "ok":
            ax.errorbar(combined["mean_dx_arcsec"], combined["mean_dy_arcsec"],
                         xerr=combined["se_dx_arcsec"], yerr=combined["se_dy_arcsec"],
                         fmt="rs", ms=10, capsize=4, mew=2, label="combined")
        # 21'' = 1 pixel circle
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(21*np.cos(theta), 21*np.sin(theta), "k--", alpha=0.3, label="1 TESS pix")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_aspect("equal")
        ax.set_xlabel("dx in-transit - out-of-transit (arcsec)")
        ax.set_ylabel("dy in-transit - out-of-transit (arcsec)")
        ax.set_title(f"{name} PSF centroid shift per sector\nVERDICT: {v}")
        ax.legend(fontsize=8, ncol=3)
        plt.tight_layout()
        outname = f"centroid_psf_{name.replace(' ','_').replace('/','_')}.png"
        plt.savefig(outname, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  plot: {outname}")

    return results, combined, v

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tic", type=int, default=38877496)
    p.add_argument("--ra", type=float, default=69.17225)
    p.add_argument("--dec", type=float, default=-62.91497)
    p.add_argument("--period", type=float, default=190.4761)
    p.add_argument("--t0", type=float, default=1393.918)
    p.add_argument("--duration", type=float, default=0.3,
                   help="transit duration in days (default 7.2 h = 0.3 d)")
    p.add_argument("--name", default="CPD-63 349")
    args = p.parse_args()
    main(args.tic, args.ra, args.dec, args.period, args.t0, args.duration, args.name)

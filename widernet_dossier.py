"""Generate a per-candidate dossier for each SURVIVED + FLAG_REVIEW.

For each candidate, produces a markdown report with:
  - host parameters (R*, M*, Teff, Tmag, distance)
  - BLS detection summary (P, T0, dur, depth, SDE)
  - implied planet parameters (R_p, a, T_eq)
  - centroid verdict + per-sector offsets
  - EB-screen sub-test detail + verdict
  - next 5 predicted transit windows (UTC)
  - catalog cross-checks: ExoFOP-TESS, Gaia DR3 NSS/variability, AAVSO VSX,
    Prsa+2022 TESS-EB, Suarez-Andres ASAS-SN

Also generates an index report listing all dossiers with one-line summaries.

Outputs:
  widernet_dossiers/<safe_name>/dossier.md
  widernet_dossiers/<safe_name>/centroid.png (if available)
  widernet_dossiers/index.md
"""
import os, io, re, json, time, warnings, shutil, requests
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astroquery.vizier import Vizier

NEA      = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
EXOFOP   = "https://exofop.ipac.caltech.edu/tess/target.php?id={tic}&json"

DOSSIER_DIR = "widernet_dossiers"
CAND_DIR    = "widernet_candidates"
os.makedirs(DOSSIER_DIR, exist_ok=True)

def tap(url, q, retries=3, timeout=120):
    for a in range(retries):
        try:
            r = requests.post(url, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=timeout)
            if r.status_code == 200:
                return pd.read_csv(io.StringIO(r.text))
        except Exception: pass
        time.sleep(2**a)
    return pd.DataFrame()

def btjd_to_utc(btjd):
    """Convert BTJD (BJD - 2457000) to UTC string."""
    try:
        jd = btjd + 2457000.0
        t = Time(jd, format="jd", scale="tdb")
        return t.utc.iso[:16]
    except Exception:
        return f"BTJD {btjd:.3f}"

def predict_next_transits(T0_btjd, P_d, n=5, after_btjd=None):
    """Return next n transit predictions after the given BTJD (defaults to now)."""
    if after_btjd is None:
        now_jd = Time.now().jd
        after_btjd = now_jd - 2457000.0
    n_min = int(np.ceil((after_btjd - T0_btjd) / P_d))
    out = []
    for k in range(n_min, n_min + n):
        btjd = T0_btjd + k * P_d
        out.append((k, btjd, btjd_to_utc(btjd)))
    return out

def lookup_tic_id_at_position(ra, dec, radius_arcsec=5):
    """If the BLS pipeline didn't have a TIC, look one up now."""
    from astroquery.mast import Catalogs
    try:
        tic = Catalogs.query_region(f"{ra} {dec}", radius=f"{radius_arcsec} arcsec",
                                     catalog="TIC")
        if len(tic) == 0: return None, None
        tic = tic[np.argsort(tic["dstArcSec"])]
        row = tic[0]
        return str(row["ID"]), float(row["Tmag"]) if row["Tmag"] else None
    except Exception:
        return None, None

def exofop_check(tic_id):
    """Pull ExoFOP-TESS direct page and check for TOI/CTOI."""
    if not tic_id: return {"available": False, "reason": "no TIC"}
    try:
        r = requests.get(f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic_id}&json",
                          timeout=30)
        if r.status_code != 200:
            return {"available": False, "reason": f"HTTP {r.status_code}"}
        d = r.json()
        return {
            "available": True,
            "n_toi": len(d.get("tois", [])),
            "n_ctoi": len(d.get("ctois", [])),
            "n_imaging": len(d.get("imaging", [])),
            "n_spectroscopy": len(d.get("spectroscopy", [])),
            "n_time_series": len(d.get("time_series", [])),
            "is_known_toi": len(d.get("tois", [])) > 0,
            "is_known_ctoi": len(d.get("ctois", [])) > 0,
        }
    except Exception as e:
        return {"available": False, "reason": str(e)[:80]}

def gaia_check(ra, dec):
    """Gaia DR3 metadata + NSS / variability flags."""
    q = f"""SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp, ruwe,
                   phot_variable_flag, non_single_star, ipd_frac_multi_peak,
                   astrometric_excess_noise, astrometric_excess_noise_sig
            FROM gaiadr3.gaia_source
            WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                             CIRCLE('ICRS', {ra}, {dec}, 0.0014))
            ORDER BY phot_g_mean_mag ASC"""
    df = tap(GAIA_TAP, q)
    if len(df) == 0: return {"available": False}
    r = df.iloc[0]
    # NSS table check (one row max each)
    sid = int(r.source_id)
    out = {
        "available": True, "source_id": sid,
        "G": float(r.phot_g_mean_mag), "bp_rp": float(r.bp_rp),
        "ruwe": float(r.ruwe), "variable_flag": str(r.phot_variable_flag),
        "non_single_star_bitmask": int(r.non_single_star or 0),
        "ipd_multi_peak": int(r.ipd_frac_multi_peak or 0),
        "ast_excess_noise": float(r.astrometric_excess_noise or 0),
        "ast_excess_noise_sig": float(r.astrometric_excess_noise_sig or 0),
        "n_sources_within_5arcsec": len(df),
    }
    nss = tap(GAIA_TAP, f"SELECT COUNT(*) AS n FROM gaiadr3.nss_two_body_orbit WHERE source_id = {sid}")
    out["nss_orbit_entries"] = int(nss.iloc[0]["n"]) if len(nss) else 0
    vari = tap(GAIA_TAP, f"SELECT COUNT(*) AS n FROM gaiadr3.vari_eclipsing_binary WHERE source_id = {sid}")
    out["gaia_eb_classifier"] = int(vari.iloc[0]["n"]) if len(vari) else 0
    return out

def vizier_eb_variable_check(ra, dec):
    """Cross-check VSX / Prsa+2022 / ASAS-SN / ATLAS within 30''."""
    c = SkyCoord(ra*u.deg, dec*u.deg)
    v = Vizier(timeout=60, row_limit=3)
    out = {}
    for cat, lbl in [("B/vsx/vsx", "VSX"),
                     ("J/ApJS/258/16/tessebs", "Prsa+2022 TESS-EB"),
                     ("J/A+A/683/A152/tessebs", "Prsa+2024 TESS-EB"),
                     ("II/366/catalog", "ASAS-SN"),
                     ("J/A+A/673/A171/atlasvar", "ATLAS var")]:
        try:
            res = v.query_region(c, radius=30*u.arcsec, catalog=cat)
            out[lbl] = len(res[0]) if (len(res) and len(res[0])) else 0
        except Exception:
            out[lbl] = "err"
    return out

def write_dossier(r, dossier_dir):
    """r is a dict-like row from widernet_bls_results.csv."""
    name = str(r["name"]).strip()
    safe = re.sub(r"\W+", "_", name)
    out_dir = os.path.join(dossier_dir, safe)
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "dossier.md")

    # Copy plots if present
    src_dir = os.path.join(CAND_DIR, name)
    plot_bls = os.path.join(src_dir, "bls.png")
    if os.path.exists(plot_bls):
        shutil.copy(plot_bls, os.path.join(out_dir, "bls.png"))

    # TIC lookup if missing
    tic_id = str(r.get("tic_id", "")).strip()
    if not tic_id or tic_id == "nan":
        tic_id, tmag = lookup_tic_id_at_position(float(r.ra), float(r.dec))
    else:
        tmag = r.get("Tmag")

    # Next transits
    next_tr = predict_next_transits(float(r["T0_BTJD"]), float(r["P_d"]), n=5)

    # Catalog checks
    print(f"  [{safe}] catalog checks...", flush=True)
    exofop = exofop_check(tic_id)
    gaia   = gaia_check(float(r.ra), float(r.dec))
    vizier = vizier_eb_variable_check(float(r.ra), float(r.dec))

    # Write dossier
    with open(md_path, "w") as f:
        f.write(f"# {name}\n\n")
        f.write(f"_Auto-generated dossier from widernet BLS pipeline_\n\n")
        f.write(f"## Identification\n\n")
        f.write(f"- 2MASS / APOGEE: {name}\n")
        f.write(f"- TIC ID: {tic_id or '(not in TIC v8)'}\n")
        f.write(f"- Tmag: {tmag if tmag is not None else '(unknown)'}\n")
        f.write(f"- Position (J2000): RA = {r.ra:.5f}°, Dec = {r.dec:.5f}°\n")
        if gaia.get("available"):
            f.write(f"- Gaia DR3: {gaia['source_id']}, G = {gaia['G']:.2f}, BP-RP = {gaia['bp_rp']:.3f}\n")
        f.write("\n")

        f.write(f"## Host stellar parameters\n\n")
        f.write(f"- R* = {r.R_star:.3f} R_sun, M* = {r.M_star:.3f} M_sun\n")
        f.write(f"- T_eff = {r.Teff:.0f} K\n")
        f.write("\n")

        f.write(f"## BLS detection\n\n")
        f.write(f"- Sectors: {int(r.n_sectors)}, baseline = {r.baseline_d:.0f} d ({r.n_pts} cadences)\n")
        f.write(f"- Photometric RMS: {r.rms_ppm:.0f} ppm\n")
        f.write(f"- Period: **{r.P_d:.5f} d**\n")
        f.write(f"- T0 (BTJD): {r['T0_BTJD']:.4f}\n")
        f.write(f"- Duration: {r.dur_h:.2f} h\n")
        f.write(f"- Depth: {r.depth_ppm:.0f} ppm\n")
        f.write(f"- SDE: **{r.sde:.2f}**\n")
        f.write("\n")

        f.write(f"## Implied planet parameters (if real)\n\n")
        f.write(f"- R_p ≈ {r.R_p_Rearth:.2f} R_Earth\n")
        f.write(f"- a ≈ {r.a_AU:.4f} AU\n")
        f.write(f"- T_eq ≈ {r.T_eq_K:.0f} K (Bond albedo = 0)\n")
        f.write("\n")

        f.write(f"## Vetting\n\n")
        f.write(f"- Centroid verdict: **{r.get('centroid_verdict', 'unknown')}**\n")
        f.write(f"- Combined centroid offset: {r.get('centroid_offset_arcsec', float('nan')):.3f} arcsec (sig = {r.get('centroid_sigma', float('nan')):.2f}σ, direction R = {r.get('centroid_direction_R', float('nan')):.3f})\n")
        f.write(f"- EB screen: **{r.get('eb_screen_verdict','unknown')}** (PASS={int(r.get('eb_n_pass',0))}, INCONC={int(r.get('eb_n_inconc',0))}, FAIL={int(r.get('eb_n_fail',0))})\n")
        f.write(f"  - SDE(2P)/SDE(1P) ratio: {r.get('eb_sde_ratio_2P_1P', float('nan')):.3f}\n")
        f.write(f"  - Odd-even σ: {r.get('eb_oddeven_sigma', float('nan')):.3f}\n")
        f.write(f"  - Positive-depth fraction: {r.get('eb_posfrac', float('nan')):.3f}\n")
        f.write(f"  - Duration/central ratio: {r.get('eb_duration_ratio', float('nan')):.3f}\n")
        f.write(f"  - Companion R: {r.get('eb_companion_RJup', float('nan')):.3f} R_Jup\n")
        f.write(f"  - Secondary @ phase 0.5: {r.get('eb_secondary_ppm', float('nan')):.0f} ppm\n")
        f.write("\n")

        f.write(f"## Catalog cross-checks\n\n")
        f.write(f"### ExoFOP-TESS\n")
        if exofop.get("available"):
            f.write(f"- TOI entries: {exofop['n_toi']} ({'KNOWN TOI' if exofop['is_known_toi'] else 'not a TOI'})\n")
            f.write(f"- CTOI entries: {exofop['n_ctoi']} ({'KNOWN CTOI' if exofop['is_known_ctoi'] else 'not a CTOI'})\n")
            f.write(f"- Imaging follow-up: {exofop['n_imaging']}\n")
            f.write(f"- Spectroscopy follow-up: {exofop['n_spectroscopy']}\n")
            f.write(f"- Time-series follow-up: {exofop['n_time_series']}\n")
        else:
            f.write(f"- ExoFOP lookup unavailable: {exofop.get('reason', 'unknown')}\n")
        f.write("\n")

        f.write(f"### Gaia DR3\n")
        if gaia.get("available"):
            f.write(f"- RUWE: {gaia['ruwe']:.3f} {'(suspect binary if >1.4)' if gaia['ruwe'] > 1.4 else '(clean)'}\n")
            f.write(f"- Variable flag: {gaia['variable_flag']}\n")
            f.write(f"- non_single_star: {gaia['non_single_star_bitmask']}\n")
            f.write(f"- IPD multi-peak: {gaia['ipd_multi_peak']}\n")
            f.write(f"- Astrometric excess noise: {gaia['ast_excess_noise']:.3f} (sig = {gaia['ast_excess_noise_sig']:.2f})\n")
            f.write(f"- NSS two-body orbit entries: {gaia['nss_orbit_entries']}\n")
            f.write(f"- Gaia DR3 EB classifier: {gaia['gaia_eb_classifier']}\n")
            f.write(f"- Sources within 5'': {gaia['n_sources_within_5arcsec']}\n")
        else:
            f.write(f"- not available\n")
        f.write("\n")

        f.write(f"### Vizier (within 30'')\n")
        for k, v in vizier.items():
            flag = " ← HIT" if isinstance(v, (int, np.integer)) and v > 0 else ""
            f.write(f"- {k}: {v}{flag}\n")
        f.write("\n")

        f.write(f"## Next predicted transits (UTC)\n\n")
        for k, btjd, utc in next_tr:
            f.write(f"- n = {k}, BTJD = {btjd:.4f}, UTC ≈ {utc}\n")
        f.write("\n")

        f.write(f"## Files\n\n")
        if os.path.exists(os.path.join(out_dir, "bls.png")):
            f.write(f"- bls.png: BLS spectrum + phase fold + lightcurve\n")
        f.write("\n")
    print(f"    -> {md_path}", flush=True)
    return md_path, exofop, gaia, vizier, next_tr

def main():
    bls = pd.read_csv("widernet_bls_results.csv")
    keep = bls[bls.eb_screen_verdict.fillna("").isin(["SURVIVED", "FLAG_REVIEW"])].copy()
    print(f"Building dossiers for {len(keep)} candidates ({(keep.eb_screen_verdict=='SURVIVED').sum()} SURVIVED, {(keep.eb_screen_verdict=='FLAG_REVIEW').sum()} FLAG_REVIEW)")
    index_lines = ["# Widernet candidate index\n",
                   f"_Auto-generated. {len(keep)} candidates surviving full pipeline (SDE>=7 + ON_TARGET/INCONCLUSIVE centroid + EB screen)._\n",
                   "\n## Candidates\n\n",
                   "| name | verdict | n_sec | P (d) | R⊕ | T_eq | centroid | known TOI? |\n",
                   "|---|---|---:|---:|---:|---:|---|:-:|\n"]
    for _, r in keep.iterrows():
        try:
            md_path, exofop, gaia, vizier, next_tr = write_dossier(r, DOSSIER_DIR)
            tot = "KNOWN" if exofop.get("is_known_toi") or exofop.get("is_known_ctoi") else "no"
            safe = re.sub(r"\W+", "_", str(r["name"]).strip())
            index_lines.append(
                f"| [{r['name']}]({safe}/dossier.md) | {r.eb_screen_verdict} | "
                f"{int(r.n_sectors)} | {r.P_d:.3f} | {r.R_p_Rearth:.2f} | "
                f"{r.T_eq_K:.0f} | {str(r.get('centroid_verdict','')).split()[0]} | {tot} |\n"
            )
        except Exception as e:
            print(f"  [{r['name']}] dossier error: {e}")
            index_lines.append(f"| {r['name']} | ERROR: {e} | | | | | | |\n")
    with open(os.path.join(DOSSIER_DIR, "index.md"), "w") as f:
        f.write("".join(index_lines))
    print(f"\nindex -> {DOSSIER_DIR}/index.md")

if __name__ == "__main__":
    main()

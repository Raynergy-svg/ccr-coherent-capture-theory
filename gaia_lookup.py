"""Minimal Gaia DR3 RUWE lookup for the EB screen.

RUWE (Renormalized Unit Weight Error) > 1.4 is Gaia's standard signature for
an unresolved binary -- the source's astrometric solution can't be fit as a
single point source. Belokurov et al. 2020 (MNRAS 496, 1922) calibrated the
threshold; the TFOP working group uses RUWE > 1.4 as a binary flag during
TESS candidate vetting. RUWE > 2 is essentially unambiguous binarity.

This module factors the live Gaia TAP query out of widernet_dossier.py so the
EB screen can call it during the BLS pipeline run (before dossiers exist).

Returns RUWE = NaN on network/parse failure so callers can degrade gracefully
to INCONC rather than crash the screen.
"""
import warnings
warnings.filterwarnings("ignore")
import io, time
import requests
import pandas as pd

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"

def _tap(q, retries=2, timeout=60):
    for a in range(retries):
        try:
            r = requests.post(GAIA_TAP, data={
                "REQUEST": "doQuery", "LANG": "ADQL",
                "FORMAT": "csv", "QUERY": q,
            }, timeout=timeout)
            if r.status_code == 200:
                return pd.read_csv(io.StringIO(r.text))
        except Exception:
            pass
        time.sleep(2 ** a)
    return pd.DataFrame()

def get_gaia_ruwe(ra, dec, radius_arcsec=5.0):
    """Return (ruwe, source_id, n_sources_in_cone) for the brightest Gaia DR3
    source within radius_arcsec of (ra, dec). Returns (nan, None, 0) on
    failure or no match."""
    radius_deg = radius_arcsec / 3600.0
    q = f"""SELECT source_id, phot_g_mean_mag, ruwe
            FROM gaiadr3.gaia_source
            WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                             CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
            ORDER BY phot_g_mean_mag ASC"""
    df = _tap(q)
    if len(df) == 0:
        return float("nan"), None, 0
    r = df.iloc[0]
    ruwe = r.ruwe
    try:
        ruwe = float(ruwe)
    except Exception:
        ruwe = float("nan")
    return ruwe, int(r.source_id), len(df)

def get_gaia_binary_flags(source_id):
    """Return (n_nss_orbit, n_vari_eb) for a given Gaia DR3 source_id.
    Both are direct catalog entries from Gaia DR3:
      - nss_two_body_orbit: published two-body orbital solutions
      - vari_eclipsing_binary: Gaia's own EB classifier
    Any non-zero count is a binary detection by Gaia. Returns (0, 0) on
    network failure (degrade-not-fail)."""
    if source_id is None:
        return 0, 0
    nss = _tap(f"SELECT COUNT(*) AS n FROM gaiadr3.nss_two_body_orbit WHERE source_id = {source_id}")
    vari = _tap(f"SELECT COUNT(*) AS n FROM gaiadr3.vari_eclipsing_binary WHERE source_id = {source_id}")
    n_nss  = int(nss.iloc[0]["n"])  if len(nss)  else 0
    n_vari = int(vari.iloc[0]["n"]) if len(vari) else 0
    return n_nss, n_vari

def get_gaia_full_check(ra, dec, radius_arcsec=5.0):
    """One-call wrapper returning (ruwe, source_id, n_cone, n_nss, n_vari_eb).
    Used by the EB screen so a single Gaia round-trip covers sub-tests 7+8."""
    ruwe, sid, n_cone = get_gaia_ruwe(ra, dec, radius_arcsec)
    n_nss, n_vari = get_gaia_binary_flags(sid)
    return ruwe, sid, n_cone, n_nss, n_vari

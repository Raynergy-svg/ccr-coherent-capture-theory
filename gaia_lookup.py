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

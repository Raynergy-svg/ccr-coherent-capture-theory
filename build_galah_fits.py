"""
Reconstruct galah_dr4_allstar_240705.fits from the Data Central TAP service
(the 723 MB official FITS is not in this container). Pulls the union of
columns the affected pipeline scripts need, for all 917,588 stars, via keyset
pagination on sobject_id, and writes a FITS with the exact expected name and
column names so t9/t16b/t20c/habitability_v2 run unmodified.
"""
import io
import time
import requests
import pandas as pd
from astropy.table import Table

DC_TAP = "https://datacentral.org.au/vo/tap/sync"
COLS = [
    "ra", "dec", "sobject_id", "gaiadr3_source_id", "tmass_id",
    "snr_px_ccd3", "flag_sp", "flag_fe_h",
    "c_fe", "e_c_fe", "flag_c_fe", "o_fe", "e_o_fe", "flag_o_fe",
    "mg_fe", "e_mg_fe", "flag_mg_fe", "si_fe", "e_si_fe", "flag_si_fe",
    "fe_h", "e_fe_h", "al_fe", "e_al_fe", "flag_al_fe",
    "ca_fe", "e_ca_fe", "flag_ca_fe", "ba_fe", "e_ba_fe", "flag_ba_fe",
    "teff", "logg", "age", "mass",
    "rv_gaia_dr3", "rv_comp_1", "e_rv_comp_1", "parallax", "parallax_error",
    "phot_g_mean_mag",
]
collist = ",".join(COLS)
CHUNK = 200000


def fetch(adql, retries=4):
    for a in range(retries):
        try:
            r = requests.post(DC_TAP, data={"REQUEST": "doQuery", "LANG": "ADQL",
                              "FORMAT": "csv", "QUERY": adql}, timeout=300)
            if r.status_code == 200:
                return pd.read_csv(io.StringIO(r.text))
            print(f"  HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"  err: {e!r}")
        time.sleep(2 ** (a + 1))
    raise RuntimeError("TAP fetch failed")


parts = []
last = 0
total = 0
while True:
    q = (f"SELECT TOP {CHUNK} {collist} FROM galah_dr4.mainstartable "
         f"WHERE sobject_id > {last} ORDER BY sobject_id")
    df = fetch(q)
    if len(df) == 0:
        break
    parts.append(df)
    total += len(df)
    last = int(df["sobject_id"].max())
    print(f"  pulled {len(df):>6}  cumulative {total:>7}  last_sobject_id={last}", flush=True)
    if len(df) < CHUNK:
        break

galah = pd.concat(parts, ignore_index=True)
print(f"TOTAL rows: {len(galah)} (expected 917588)")
print(f"columns: {len(galah.columns)}")

# write FITS with the exact expected filename
tab = Table.from_pandas(galah)
tab.write("galah_dr4_allstar_240705.fits", overwrite=True)
import os
print(f"Wrote galah_dr4_allstar_240705.fits "
      f"({os.path.getsize('galah_dr4_allstar_240705.fits')/1e6:.0f} MB)")

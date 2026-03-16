#!/usr/bin/env python3
"""
Fetch APOGEE DR17 abundances for OCCAM member stars via SDSS SkyServer SQL API.
Produces apogee_occam_abundances.csv for use by tapogee.py.

Queries the apogeeStar + aspcapStar tables in batches of 200 APOGEE_IDs.
No authentication required.
"""

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
from io import StringIO
from astropy.table import Table

SKYSERVER_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
OCCAM_FILE = "occam_member-DR17.fits"
OUTPUT_CSV = "apogee_occam_abundances.csv"
BATCH_SIZE = 200
RETRY_MAX = 3
RETRY_DELAY = 5

def log(msg):
    print("[FETCH] " + str(msg), flush=True)


def query_skyserver(sql):
    """Submit SQL query to SDSS SkyServer, return DataFrame."""
    params = {"cmd": sql, "format": "csv"}
    resp = requests.get(SKYSERVER_URL, params=params, timeout=120)
    resp.raise_for_status()
    text = resp.text
    # Skip comment lines starting with #
    lines = [line for line in text.split("\n") if not line.startswith("#")]
    cleaned = "\n".join(lines)
    if "error" in cleaned.lower()[:300]:
        raise RuntimeError("SkyServer error: " + cleaned[:500])
    df = pd.read_csv(StringIO(cleaned))
    return df


def build_query(apogee_ids):
    """Build SQL to fetch abundances for a batch of APOGEE_IDs."""
    id_list = ", ".join("'" + aid + "'" for aid in apogee_ids)
    sql = """
    SELECT
        s.apogee_id AS APOGEE_ID,
        s.ra AS RA,
        s.dec AS DEC,
        s.vhelio_avg AS VHELIO_AVG,
        s.vscatter AS VSCATTER,
        s.snr AS SNR,
        a.teff AS TEFF,
        a.logg AS LOGG,
        a.fe_h AS FE_H,
        a.fe_h_err AS FE_H_ERR,
        a.c_fe AS C_FE,
        a.c_fe_err AS C_FE_ERR,
        a.o_fe AS O_FE,
        a.o_fe_err AS O_FE_ERR,
        a.mg_fe AS MG_FE,
        a.si_fe AS SI_FE,
        a.al_fe AS AL_FE,
        a.aspcapflag AS ASPCAPFLAG
    FROM apogeeStar s
    JOIN aspcapStar a ON a.apstar_id = s.apstar_id
    WHERE s.apogee_id IN (""" + id_list + ")"
    return sql


def main():
    log("Loading OCCAM member catalog: " + OCCAM_FILE)
    occam = Table.read(OCCAM_FILE).to_pandas()

    # Strip APOGEE_ID — handle FITS byte strings
    def clean_id(x):
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace").strip()
        return str(x).strip()
    occam["APOGEE_ID"] = occam["APOGEE_ID"].apply(clean_id)

    all_ids = sorted(occam["APOGEE_ID"].unique())
    log("Unique APOGEE_IDs to fetch: " + str(len(all_ids)))

    # Check for existing partial results
    existing = pd.DataFrame()
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        existing_ids = set(existing["APOGEE_ID"].str.strip().values)
        all_ids = [aid for aid in all_ids if aid not in existing_ids]
        log("Resuming: " + str(len(existing)) + " already fetched, " +
            str(len(all_ids)) + " remaining")

    if len(all_ids) == 0:
        log("All IDs already fetched. Nothing to do.")
        return

    n_batches = (len(all_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    results = []
    failed_batches = []

    for i in range(0, len(all_ids), BATCH_SIZE):
        batch = all_ids[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        for attempt in range(RETRY_MAX):
            try:
                sql = build_query(batch)
                df = query_skyserver(sql)
                results.append(df)
                log("Batch " + str(batch_num) + "/" + str(n_batches) +
                    ": " + str(len(df)) + " results")
                break
            except Exception as e:
                if attempt < RETRY_MAX - 1:
                    log("Batch " + str(batch_num) + " attempt " +
                        str(attempt + 1) + " failed: " + str(e)[:100] +
                        " — retrying in " + str(RETRY_DELAY) + "s")
                    time.sleep(RETRY_DELAY)
                else:
                    log("Batch " + str(batch_num) + " FAILED after " +
                        str(RETRY_MAX) + " attempts: " + str(e)[:200])
                    failed_batches.append(batch_num)

        # Courtesy delay between batches
        if batch_num < n_batches:
            time.sleep(0.5)

        # Save progress every 25 batches
        if batch_num % 25 == 0 and results:
            partial = pd.concat(results, ignore_index=True)
            if len(existing) > 0:
                partial = pd.concat([existing, partial], ignore_index=True)
            partial.to_csv(OUTPUT_CSV, index=False)
            log("Progress saved: " + str(len(partial)) + " total rows")

    # Final save
    if results:
        final = pd.concat(results, ignore_index=True)
        if len(existing) > 0:
            final = pd.concat([existing, final], ignore_index=True)
        final.to_csv(OUTPUT_CSV, index=False)
        log("Saved: " + OUTPUT_CSV + " (" + str(len(final)) + " rows)")
    elif len(existing) > 0:
        log("No new results. Existing file has " + str(len(existing)) + " rows.")
    else:
        log("ERROR: No results fetched at all.")
        sys.exit(1)

    if failed_batches:
        log("WARNING: " + str(len(failed_batches)) + " batches failed: " +
            str(failed_batches))
        log("Re-run this script to retry (it resumes from where it left off).")

    log("DONE")


if __name__ == "__main__":
    main()

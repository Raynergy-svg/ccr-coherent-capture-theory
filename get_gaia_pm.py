import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia

def log(msg):
    print("[INFO] " + str(msg), flush=True)

# Load the 4744 coherent stars
stars = pd.read_csv("t6b_coherent_stars.csv")
log("Coherent stars loaded: " + str(len(stars)))
log("Columns available: " + str(list(stars.columns[:10])))

# Gaia TAP query — batch by source using positional crossmatch
# We'll use ADQL with a cross-match against gaiadr3.gaia_source
# Upload the ra/dec as a user table and join within 1 arcsec

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1

# Build coordinate list
coords = SkyCoord(ra=stars["ra"].values * u.degree,
                  dec=stars["dec"].values * u.degree,
                  frame="icrs")

log("Running Gaia DR3 positional crossmatch (1 arcsec radius)...")
log("This may take a few minutes for " + str(len(stars)) + " stars...")

# Astroquery cone search is slow for many targets — use upload ADQL instead
from astropy.table import Table as AstroTable

upload_table = AstroTable({
    "ra":  stars["ra"].values,
    "dec": stars["dec"].values,
    "row_id": np.arange(len(stars))
})

query = """
SELECT u.row_id, g.source_id, g.ra, g.dec,
       g.pmra, g.pmdec, g.pmra_error, g.pmdec_error,
       g.parallax, g.parallax_error,
       g.radial_velocity, g.radial_velocity_error,
       g.ruwe
FROM TAP_UPLOAD.input AS u
JOIN gaiadr3.gaia_source AS g
ON 1=CONTAINS(
    POINT('ICRS', u.ra, u.dec),
    CIRCLE('ICRS', g.ra, g.dec, 0.000278)
)
"""

job = Gaia.launch_job_async(
    query,
    upload_resource=upload_table,
    upload_table_name="input",
    verbose=False
)
result = job.get_results()
log("Gaia matches returned: " + str(len(result)))

gaia_df = result.to_pandas()
log("Columns returned: " + str(list(gaia_df.columns)))

# Keep best match per row_id (closest by parallax agreement if duplicates)
gaia_df = gaia_df.sort_values("ruwe").drop_duplicates(subset="row_id", keep="first")
log("After dedup: " + str(len(gaia_df)))

# Merge back onto coherent stars
merged = stars.merge(
    gaia_df[["row_id","source_id","pmra","pmdec","pmra_error","pmdec_error","ruwe"]],
    left_index=True,
    right_on="row_id",
    how="left"
)
merged = merged.drop(columns=["row_id"])

pm_found = merged["pmra"].notna().sum()
log("Stars with proper motions: " + str(pm_found) + "/" + str(len(merged)))

merged.to_csv("t6b_coherent_stars_with_pm.csv", index=False)
log("Saved: t6b_coherent_stars_with_pm.csv")

# Quick summary per coherent group
log("Per-group PM availability:")
grp_pm = merged.groupby("chem_group").agg(
    N=("pmra","count"),
    pmra_mean=("pmra","mean"),
    pmra_std=("pmra","std"),
    pmdec_mean=("pmdec","mean"),
    pmdec_std=("pmdec","std")
).reset_index()
log(grp_pm.to_string(index=False))
grp_pm.to_csv("t6b_coherent_pm_summary.csv", index=False)
log("Saved: t6b_coherent_pm_summary.csv")
log("DONE")

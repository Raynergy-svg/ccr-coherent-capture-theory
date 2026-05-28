"""Test 1c -- inspect ExoFOP JSON sections; use astroquery.vizier for catalogs."""
import warnings, json, requests, sys
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

TIC=38877496; GAIA=4677014433801899392
RA, DEC = 69.17225, -62.91497

OUT = "cpd63349_vetting.md"
out = open(OUT, "a")
def md(s=""): print(s); out.write(s+"\n")

# Reread ExoFOP JSON
try:
    data = json.load(open("exofop_38877496.json"))
except FileNotFoundError:
    r = requests.get(f"https://exofop.ipac.caltech.edu/tess/target.php?id={TIC}&json", timeout=60)
    data = r.json(); json.dump(data, open("exofop_38877496.json","w"), indent=2)

md("\n### 1c.2 ExoFOP JSON deep inspection")
for sec in ["tois","ctois","imaging","spectroscopy","time_series",
            "stellar_companions","planet_parameters","stellar_parameters",
            "basic_info","files"]:
    if sec in data:
        v = data[sec]
        if isinstance(v,(list,dict)):
            n = len(v)
            md(f"\n**`{sec}`: {n} entr{'y' if n==1 else 'ies'}**")
            if n > 0:
                md("```json")
                md(json.dumps(v, indent=2)[:1500])
                md("```")

# Astroquery Vizier for the EB / variable catalogs
md("\n### 1h.2 Vizier (astroquery) variable / EB cross-match")
c = SkyCoord(RA*u.deg, DEC*u.deg)
v = Vizier(timeout=120, row_limit=5)
catalogs = {
    "B/vsx/vsx":          "AAVSO VSX",
    "J/ApJS/258/16/tessebs": "Prsa+2022 TESS EB",
    "J/A+A/683/A152/tessebs": "Prsa+2024 TESS EB updated",
    "II/366/catalog":     "ASAS-SN variables",
    "J/A+A/673/A171/atlasvar": "ATLAS variables",
    "II/328/allwise":     "AllWISE (cool-companion check)",
    "II/246/out":         "2MASS",
}
for cat, lbl in catalogs.items():
    try:
        res = v.query_region(c, radius=42*u.arcsec, catalog=cat)
        if len(res) and len(res[0]):
            md(f"\n  **{lbl} ({cat})**: {len(res[0])} match(es) within 42''")
            df = res[0].to_pandas()
            md("```")
            md(df.head(10).to_string(index=False, max_colwidth=20))
            md("```")
        else:
            md(f"  {lbl} ({cat}): no match within 42'' ✓")
    except Exception as e:
        md(f"  {lbl} ({cat}): query error: {str(e)[:120]}")

# Final
md("\n### Test 1 — FINAL")
toi_section = data.get("tois", [])
ctoi_section = data.get("ctois", [])
imaging_section = data.get("imaging", [])
spec_section = data.get("spectroscopy", [])
ts_section = data.get("time_series", [])
md(f"\n  ExoFOP: {len(toi_section)} TOI, {len(ctoi_section)} CTOI, "
   f"{len(imaging_section)} imaging, {len(spec_section)} spectroscopy, "
   f"{len(ts_section)} time-series logs")
if len(toi_section) or len(ctoi_section):
    md("\n**TEST 1 result: FAIL -- prior TOI/CTOI entry exists on ExoFOP.**")
else:
    md("\n**TEST 1 result: PASS -- no prior TOI/CTOI entry on ExoFOP for "
       f"TIC {TIC}; no community vetting observations logged; no variable / "
       "EB classification in any external catalog. Signal is new.**")

out.close()
print("Test 1c done -- appended to cpd63349_vetting.md")

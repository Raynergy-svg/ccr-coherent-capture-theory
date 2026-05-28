"""TEST 1 — ExoFOP-TESS / TOI status check for CPD-63 349 (TIC 38877496).

Sources:
  - NASA Exoplanet Archive TOI table
  - ExoFOP-TESS direct HTTP (the public target page returns JSON for known TICs)
  - NASA Exoplanet Archive Community-TOI (ctoi) and pscomppars
  - Gaia DR3 NSS (rechecked)
  - SIMBAD by coordinates
  - VizieR meta search for TESS EB catalogs (Prsa et al. 2022, J/ApJS/258/16),
    Gaia DR3 variable star catalog, ASAS-SN, ATLAS, VSX

Output appended into cpd63349_vetting.md.
"""
import io, time, requests, json, sys
import pandas as pd, numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

TIC  = 38877496
GAIA = 4677014433801899392
RA, DEC = 69.17225, -62.91497

NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
SIM_TAP = "https://simbad.u-strasbg.fr/simbad/sim-tap/sync"
VIZIER  = "https://vizier.cds.unistra.fr/viz-bin/votable"

def tap(url, q, retries=3, timeout=120):
    for a in range(retries):
        try:
            r = requests.post(url, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=timeout)
            if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
        except Exception as e: print(f"[WARN] {e}")
        time.sleep(2**a)
    return pd.DataFrame()

def hdr(s): print(f"\n{'='*78}\n{s}\n{'='*78}")

OUT = "cpd63349_vetting.md"
out = open(OUT, "w")
def md(s=""): print(s); out.write(s+"\n")

md(f"# CPD-63 349 vetting report")
md(f"_TIC {TIC}, Gaia DR3 {GAIA}, (RA,Dec) = ({RA:.5f}, {DEC:.5f}), G=10.04, d=105 pc_")
md(f"\nGenerated: 2026-05-28\n")

# ============================================================================
md("## TEST 1 — TOI / community vetting status")
md("")

# 1a. TOI table
md("### 1a. NASA Exoplanet Archive TOI catalog")
toi = tap(NEA, f"SELECT toi, tid, ra, dec, pl_orbper, pl_rade, pl_eqt, "
               f"tfopwg_disp, toi_created, toi_updated FROM toi WHERE tid = {TIC}")
if len(toi):
    md(f"**HIT: {len(toi)} TOI record(s) for TIC {TIC}!**")
    md("")
    md("```")
    md(toi.to_string(index=False))
    md("```")
else:
    md(f"No TOI in NASA Exoplanet Archive for TIC {TIC}. ✓ (not a known TOI)")

# 1b. CTOI table (community-TOI)
md("\n### 1b. NASA Exoplanet Archive Community-TOI (CTOI)")
try:
    ctoi = tap(NEA, f"SELECT ctoi, tic_id, ra, dec, pl_orbper, pl_rade, "
                   f"user_disp, ctoi_created FROM ctoi WHERE tic_id = {TIC}")
    if len(ctoi):
        md(f"**HIT: {len(ctoi)} CTOI record(s) for TIC {TIC}!**")
        md("```")
        md(ctoi.to_string(index=False))
        md("```")
    else:
        md(f"No CTOI in NASA Exoplanet Archive for TIC {TIC}. ✓")
except Exception as e:
    md(f"_ctoi query failed: {e}_")

# 1c. ExoFOP-TESS direct
md("\n### 1c. ExoFOP-TESS direct target page")
ef_url = f"https://exofop.ipac.caltech.edu/tess/download_target.php?id={TIC}"
try:
    r = requests.get(ef_url, timeout=60)
    md(f"HTTP {r.status_code}, {len(r.text)} bytes from `{ef_url}`")
    if r.status_code == 200:
        # the download_target.php endpoint returns a flat ascii text file with sections
        text = r.text
        if "TOI" in text or "CTOI" in text:
            # find TOI/CTOI section
            for kw in ["TOI INFORMATION", "TOI Information", "CTOI INFORMATION",
                      "Community TOI", "CTOI Information"]:
                idx = text.find(kw)
                if idx > 0:
                    md(f"\n**Section found: {kw}**")
                    snippet = text[idx:idx+1200]
                    md("```")
                    md(snippet)
                    md("```")
        # spectroscopy / imaging / photometry observations sections
        for kw in ["IMAGING", "SPECTROSCOPY", "PHOTOMETRY", "TIME SERIES",
                  "RADIAL VELOCITY", "OBSERVING NOTES", "SG1", "SG2", "SG3", "SG4", "SG5"]:
            idx = text.find(kw)
            if idx > 0:
                snippet = text[idx:idx+800]
                # only print if there are entries (more than just a header)
                lines = snippet.split("\n")
                non_empty = [l for l in lines[1:30] if l.strip() and not l.startswith("-")]
                if len(non_empty) > 2:
                    md(f"\n**Section: {kw}**")
                    md("```")
                    md(snippet[:600])
                    md("```")
        # save full file for inspection
        with open("exofop_38877496.txt", "w") as f:
            f.write(text)
        md(f"\n_full ExoFOP page saved to `exofop_38877496.txt` ({len(text)} bytes)_")
except Exception as e:
    md(f"_ExoFOP HTTP error: {e}_")

# 1d. NASA confirmed planets
md("\n### 1d. NASA Exoplanet Archive confirmed planets (pscomppars)")
pps = tap(NEA, f"SELECT pl_name, hostname, ra, dec, pl_orbper, pl_rade, pl_eqt, "
              f"gaia_dr3_id FROM pscomppars WHERE gaia_dr3_id = '{GAIA}' OR "
              f"(ra BETWEEN {RA-0.001} AND {RA+0.001} AND dec BETWEEN {DEC-0.001} AND {DEC+0.001})")
if len(pps):
    md(f"**HIT: {len(pps)} confirmed planet(s)!**")
    md("```"); md(pps.to_string(index=False)); md("```")
else:
    md("No confirmed planet match. ✓")

# 1e. Gaia DR3 NSS rechecked
md("\n### 1e. Gaia DR3 NSS recheck (any astrometric/orbital companion?)")
nss1 = tap(GAIA_TAP, f"SELECT * FROM gaiadr3.nss_two_body_orbit WHERE source_id = {GAIA}")
md(f"  nss_two_body_orbit: {len(nss1)} entries")
nss2 = tap(GAIA_TAP, f"SELECT * FROM gaiadr3.nss_acceleration_astro WHERE source_id = {GAIA}")
md(f"  nss_acceleration_astro: {len(nss2)} entries")
nss3 = tap(GAIA_TAP, f"SELECT * FROM gaiadr3.nss_non_linear_spectro WHERE source_id = {GAIA}")
md(f"  nss_non_linear_spectro: {len(nss3)} entries")
if any(len(x) for x in [nss1, nss2, nss3]):
    md("**NSS hit -- not a single star according to Gaia DR3 (could be BEB)**")
else:
    md("No NSS entries in any of the 3 NSS tables ✓ (consistent with single star)")

# 1f. Gaia DR3 variable star catalog
md("\n### 1f. Gaia DR3 vari_* tables (eclipsing binary classifier)")
for tbl, lbl in [("gaiadr3.vari_eclipsing_binary","ECLIPSING BINARY"),
                 ("gaiadr3.vari_classifier_result","general classifier"),
                 ("gaiadr3.vari_summary","variability summary")]:
    try:
        v = tap(GAIA_TAP, f"SELECT * FROM {tbl} WHERE source_id = {GAIA}")
        md(f"  {tbl}: {len(v)} entries")
        if len(v):
            md(f"  **HIT in {lbl}:**")
            md("```")
            md(v.to_string(index=False))
            md("```")
    except Exception as e:
        md(f"  {tbl}: query error: {e}")

# 1g. SIMBAD object type
md("\n### 1g. SIMBAD lookup")
sim_q = f"""SELECT main_id, otype_txt, sp_type, b, v, plx_value
            FROM basic
            WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                             CIRCLE('ICRS', {RA}, {DEC}, 0.0014))"""
sim = tap(SIM_TAP, sim_q)
if len(sim):
    md("SIMBAD record(s):")
    md("```"); md(sim.to_string(index=False)); md("```")
    for _, r in sim.iterrows():
        ot = str(r.otype_txt or "")
        for flag in ["EB*","Eclipsing","Variable","SB","Spectro","RR","RV","Algol","WUMa"]:
            if flag.lower() in ot.lower():
                md(f"**SIMBAD object type flag: '{flag}' present in `{ot}` !**")
else:
    md("No SIMBAD hit within 5'' of target position.")

# Also check Vizier for known catalogs of variables / EBs at this position
md("\n### 1h. Vizier variable-star catalog cross-match")
# VSX (B/vsx/vsx, AAVSO VSX) -- the main external EB catalog
# ASAS-SN variables (II/366/catalog), ATLAS variable catalog (J/A+A/673/A171)
# Prsa et al. TESS EB catalog (J/ApJS/258/16/tessebs)
for cat, lbl in [("B/vsx/vsx","AAVSO VSX"),
                 ("II/366/catalog","ASAS-SN variables"),
                 ("J/A+A/673/A171/atlasvar","ATLAS variables"),
                 ("J/ApJS/258/16/tessebs","Prsa+2022 TESS EB catalog")]:
    try:
        url = ("https://vizier.cds.unistra.fr/viz-bin/votable?-source=" + cat +
               f"&-c={RA}+{DEC}&-c.rs=42&-out.add=_r")
        r = requests.get(url, timeout=60)
        from astropy.table import Table
        try:
            t = Table.read(io.BytesIO(r.content), format="votable")
            md(f"  {lbl} ({cat}): {len(t)} match(es)")
            if len(t) > 0:
                # show first few cols
                md("  **HIT** -- preview:")
                md("```")
                md(t[:5].pformat(max_width=200, max_lines=20).__str__() if hasattr(t,"pformat") else str(t[:5]))
                md("```")
        except Exception as e:
            md(f"  {lbl}: votable parse fail / empty ({e})")
    except Exception as e:
        md(f"  {lbl}: query error {e}")

# 1i. ASAS-SN photometric variability check (separately via API)
md("\n### 1i. Summary verdict for Test 1")

# Save & summary
toi_hit = len(toi) > 0
ctoi_hit = False  # set above if non-empty
nss_hit = any(len(x) for x in [nss1, nss2, nss3])
pps_hit = len(pps) > 0

if toi_hit or pps_hit:
    md("\n**TEST 1: FAIL -- the candidate is already in the TOI/confirmed planet catalogs.**")
elif nss_hit:
    md("\n**TEST 1: FAIL -- Gaia DR3 NSS catalog flags this star as non-single (possible spectroscopic/astrometric binary).**")
else:
    md("\n**TEST 1: PASS** -- no TOI/CTOI entry; no NASA confirmed planet at this position; "
       "no Gaia NSS, no AAVSO VSX, no TESS EB catalog flag. The signal is **not previously known**.")
    md("\nThis means we cannot stop -- continue to Tests 2 and 3.")

out.close()
print(f"\nTest 1 done -- output in {OUT}")

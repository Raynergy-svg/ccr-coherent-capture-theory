"""CCT population test — STAGE 1: pull NEA hosts and find APOGEE matches.

Sealed pre-registration: PRE_REGISTRATION.md @ commit 1441551
Frozen scorer:           habitability_v2.py @ commit cfa1249

Strategy:
  - NEA pscomppars one TAP query (small)
  - APOGEE DR17 ASPCAP allStarLite -- download once via direct SDSS URL
    and cross-match locally (avoids 4000 individual Vizier queries)
  - Output: cct_test_hosts.csv (APOGEE chemistry for confirmed planet hosts)
"""
import io, time, requests, os
import numpy as np, pandas as pd
import warnings
warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
import astropy.units as u

NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
APOGEE_DR17 = "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits"
APOGEE_FILE = "/tmp/allStar-dr17-synspec_rev1.fits"

def tap(url, q, retries=4, timeout=300):
    for a in range(retries):
        try:
            r = requests.post(url, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=timeout)
            if r.status_code == 200:
                return pd.read_csv(io.StringIO(r.text))
            print(f"[WARN] HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e: print(f"[WARN] {e}")
        time.sleep(2**a)
    return pd.DataFrame()

print("="*78)
print("CCT POPULATION TEST -- STAGE 1: cross-match data pull")
print("="*78)

# 1. NEA confirmed planets
print("\n[1] Querying NEA pscomppars for confirmed planet hosts...")
pps = tap(NEA, """
    SELECT pl_name, hostname, ra, dec,
           pl_orbper, pl_rade, pl_eqt, pl_orbsmax,
           st_teff, st_logg, st_mass, st_rad, st_met, st_age,
           gaia_dr3_id
    FROM pscomppars
    WHERE pl_rade IS NOT NULL AND pl_eqt IS NOT NULL""")
print(f"  confirmed planets with R_p and T_eq: {len(pps)}")
print(f"  unique hosts: {pps.hostname.nunique()}")

# pre-registered category assignment
pps["category"] = "other"
pps.loc[(pps.pl_rade < 1.6) & (pps.pl_eqt >= 200) & (pps.pl_eqt <= 340), "category"] = "HZ_rocky"
pps.loc[(pps.pl_rade < 1.6) & ~((pps.pl_eqt >= 200) & (pps.pl_eqt <= 340)), "category"] = "non_HZ_rocky"
pps.loc[(pps.pl_rade >= 1.6) & (pps.pl_rade < 4.0), "category"] = "sub_Neptune"
pps.loc[(pps.pl_rade >= 6.0) & (pps.pl_eqt > 1000), "category"] = "hot_Jupiter"
print("\n  category counts (per planet):")
print(pps.category.value_counts().to_string())

# host-level: best category at each host (HZ_rocky > non_HZ > sub_N > hot_J > other)
prio = {"HZ_rocky": 4, "non_HZ_rocky": 3, "sub_Neptune": 2, "hot_Jupiter": 1, "other": 0}
pps["cat_prio"] = pps["category"].map(prio)
hosts = pps.sort_values("cat_prio", ascending=False).drop_duplicates("hostname")[
    ["hostname", "ra", "dec", "st_teff", "st_logg", "st_mass", "st_rad",
     "st_met", "st_age", "gaia_dr3_id", "pl_rade", "pl_eqt", "category"]
].reset_index(drop=True)
print(f"\n  unique hosts: {len(hosts)}")
print(f"  host category counts:")
print(hosts.category.value_counts().to_string())
hosts.to_csv("cct_test_hosts_pre_apogee.csv", index=False)

# 2. APOGEE DR17 allStar -- download once, parse locally
print(f"\n[2] Downloading APOGEE DR17 allStar (only chemistry & coordinates cols)...")
if not os.path.exists(APOGEE_FILE):
    print(f"  fetching {APOGEE_DR17}")
    try:
        with requests.get(APOGEE_DR17, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            print(f"  size: {total/1e6:.0f} MB")
            with open(APOGEE_FILE, "wb") as f:
                done = 0
                for chunk in r.iter_content(chunk_size=2**20):
                    f.write(chunk); done += len(chunk)
                    if done % (50*1024*1024) < 2**20:
                        print(f"    {done/1e6:.0f} MB / {total/1e6:.0f} MB")
        print(f"  saved -> {APOGEE_FILE}")
    except Exception as e:
        print(f"  download error: {e}")
        raise
else:
    print(f"  already have {APOGEE_FILE}")

print(f"\n[3] Reading APOGEE allStar...")
with fits.open(APOGEE_FILE, memmap=True) as hdul:
    print(f"  HDU 1 dimensions: {hdul[1].header.get('NAXIS2', '?')} rows")
    data = hdul[1].data
    cols = data.columns.names
    print(f"  total columns: {len(cols)}")
    # extract the ones we need
    keep_cols = ["APOGEE_ID", "RA", "DEC", "TEFF", "LOGG", "M_H", "FE_H", "ALPHA_M",
                 "C_FE", "CI_FE", "N_FE", "O_FE", "MG_FE", "SI_FE", "CA_FE",
                 "AL_FE", "CE_FE", "TI_FE", "TIII_FE", "NI_FE", "MN_FE", "NA_FE",
                 "VSCATTER", "SNR", "ASPCAPFLAG", "STARFLAG"]
    actual_keep = [c for c in keep_cols if c in cols]
    missing = [c for c in keep_cols if c not in cols]
    print(f"  keeping {len(actual_keep)} of {len(keep_cols)} preferred columns")
    if missing: print(f"  missing: {missing}")
    apg = pd.DataFrame({c: np.array(data[c]) for c in actual_keep})
    print(f"  APOGEE rows: {len(apg)}")
    print(f"  RA range: {apg.RA.min():.2f} to {apg.RA.max():.2f}")

# Quality cuts
print(f"\n[4] APOGEE quality cuts...")
print(f"  TEFF finite: {apg.TEFF.between(3500, 8000).sum()}")
print(f"  LOGG finite: {apg.LOGG.between(0, 6).sum()}")
print(f"  FE_H finite: {apg.FE_H.between(-2, 1).sum()}")
qual = (apg.TEFF.between(3500, 8000) & apg.LOGG.between(0, 6) & apg.FE_H.between(-2, 1) &
        (apg.SNR > 70) & ((apg.ASPCAPFLAG.astype("int64") & (1 << 23)) == 0))   # STARBAD bit 23 in ASPCAPFLAG
apg = apg[qual].reset_index(drop=True)
print(f"  after quality cuts: {len(apg)}")

# 5. Cross-match hosts → APOGEE
print(f"\n[5] Cross-matching {len(hosts)} hosts -> APOGEE ({len(apg)})...")
hosts_coords = SkyCoord(hosts.ra.values*u.deg, hosts.dec.values*u.deg)
apg_coords = SkyCoord(apg.RA.values*u.deg, apg.DEC.values*u.deg)
idx, sep, _ = hosts_coords.match_to_catalog_sky(apg_coords)
match_mask = sep.arcsec < 5.0
print(f"  matches within 5'': {match_mask.sum()}")
print(f"  match rate: {100*match_mask.sum()/len(hosts):.1f}%")

# build host-APOGEE table
match_rows = []
for i, m in enumerate(match_mask):
    if not m: continue
    h = hosts.iloc[i]
    a = apg.iloc[idx[i]]
    row = dict(host_name=h.hostname, host_category=h.category,
               host_pl_rade=h.pl_rade, host_pl_eqt=h.pl_eqt,
               host_ra=h.ra, host_dec=h.dec, host_st_teff=h.st_teff,
               host_st_met=h.st_met, host_st_age=h.st_age,
               sep_arcsec=float(sep[i].arcsec))
    for c in actual_keep:
        row[f"ap_{c}"] = a[c] if pd.notna(a[c]) else np.nan
    match_rows.append(row)
HOSTS = pd.DataFrame(match_rows)
HOSTS.to_csv("cct_test_hosts.csv", index=False)
print(f"  saved -> cct_test_hosts.csv ({len(HOSTS)} rows)")
print(f"\n  matched-host category counts:")
print(HOSTS.host_category.value_counts().to_string())

# 6. Build APOGEE field control: FGK dwarfs, not in host list, exclude any star
# within 30'' of any NEA host
print(f"\n[6] Building APOGEE field control (FGK dwarfs not in host list)...")
fgk = apg[(apg.TEFF.between(4500, 7000)) & (apg.LOGG > 3.8)].copy()
print(f"  FGK dwarfs total: {len(fgk)}")

# exclude any APOGEE star within 30'' of any host
fgk_coords = SkyCoord(fgk.RA.values*u.deg, fgk.DEC.values*u.deg)
host_idx, host_sep, _ = fgk_coords.match_to_catalog_sky(hosts_coords)
not_host = host_sep.arcsec > 30.0
field = fgk[not_host].reset_index(drop=True)
print(f"  field control sample (FGK dwarfs not near hosts): {len(field)}")
field.to_csv("cct_test_field.csv", index=False)
print(f"  saved -> cct_test_field.csv")

print(f"\n{'='*78}\nSTAGE 1 COMPLETE")
print(f"  hosts in APOGEE: {len(HOSTS)}")
print(f"  field control:   {len(field)}")
print(f"  ready for STAGE 2 (apply frozen scorer)\n{'='*78}")

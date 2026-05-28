"""Pull TIC v8 stellar parameters for the 32 actionable dwarfs via MAST Catalogs API.

TIC v8 has homogenised Teff, R_star, M_star, log g used by SPOC.
We need these to convert BLS transit depth into planet radius.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from astroquery.mast import Catalogs

dw = pd.read_csv("real_dwarf_targets.csv")
print(f"32 actionable dwarfs loaded\n")

results = []
for i, r in dw.iterrows():
    ra, dec = float(r.ra), float(r.dec)
    name = r.main_id
    try:
        tic = Catalogs.query_region(f"{ra} {dec}", radius="0.0014 deg", catalog="TIC")
    except Exception as e:
        print(f"  [{i+1:>2}] {name}: query failed: {e}")
        results.append(dict(gaia=int(r.gaiadr3_source_id), name=name, status=f"query_error"))
        continue
    if len(tic) == 0:
        print(f"  [{i+1:>2}] {name}: no TIC match")
        results.append(dict(gaia=int(r.gaiadr3_source_id), name=name, status="no_match"))
        continue
    # sort by distance / GAIA tag
    tic = tic[np.argsort(tic["dstArcSec"])]
    row = tic[0]
    teff = float(row["Teff"]) if row["Teff"] else np.nan
    rad  = float(row["rad"])  if row["rad"]  else np.nan
    mass = float(row["mass"]) if row["mass"] else np.nan
    logg = float(row["logg"]) if row["logg"] else np.nan
    Tmag = float(row["Tmag"]) if row["Tmag"] else np.nan
    print(f"  [{i+1:>2}] {name:<22}: TIC {row['ID']}  Teff={teff:.0f}  R={rad:.3f}  M={mass:.3f}  Tmag={Tmag:.2f}  (sep {row['dstArcSec']:.1f}'')")
    results.append(dict(
        gaia=int(r.gaiadr3_source_id), name=name,
        tic_id=str(row["ID"]), Tmag=Tmag,
        Teff_tic=teff, logg_tic=logg, R_star=rad, M_star=mass,
        sep_arcsec=float(row["dstArcSec"]),
        status="matched"
    ))

R = pd.DataFrame(results)
R.to_csv("tic_params_32dwarfs.csv", index=False)
print(f"\nsaved {len(R)} rows to tic_params_32dwarfs.csv")
print(f"  with R_star: {R.R_star.notna().sum()}/32" if "R_star" in R.columns else "")
print(f"  with M_star: {R.M_star.notna().sum()}/32" if "M_star" in R.columns else "")
print(f"  with Teff:   {R.Teff_tic.notna().sum()}/32" if "Teff_tic" in R.columns else "")
print("DONE")

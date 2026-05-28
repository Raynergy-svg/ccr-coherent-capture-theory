"""For the 12 actionable dwarfs with no SPOC LC but with FFI data,
check what HLSP lightcurves are available (TESS-SPOC FFI, QLP, etc).

These are the actionable dwarfs we WOULD search if we had LCs:
  BD-03 3321  (hab8=0.997, highest 8D scorer) - 2 FFI
  CD-30 7056  (hab8=0.995) - 5 FFI
  HD 271200   (hab8=0.991) - 44 FFI products (CVZ?)
  ...
"""
import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from astroquery.mast import Observations

cov = pd.read_csv("tess_coverage_32dwarfs.csv")
dw  = pd.read_csv("real_dwarf_targets.csv")
no_lc = cov[(cov.n_ts == 0) & (cov.n_ffi > 0)].copy()
print(f"{len(no_lc)} dwarfs with no SPOC LC but FFI data available\n")

# Build ra lookup
ra_lookup = {int(r.gaiadr3_source_id): float(r.ra) for _, r in dw.iterrows()}

results = []
for i, r in no_lc.iterrows():
    sid = int(r.gaia)
    name = r["name"]
    dec = float(r.dec)
    ra = ra_lookup.get(sid)
    if ra is None: continue
    n_ffi = int(r.n_ffi)
    print(f"=== {name}  Gaia {sid}  RA,Dec=({ra:.4f},{dec:.4f})  n_ffi={n_ffi} ===")

    obs = Observations.query_region(f"{ra} {dec}", radius="20 arcsec")
    ts = obs[obs["dataproduct_type"] == "timeseries"]
    # group by provenance_name
    if len(ts) == 0:
        print(f"  no timeseries products at all")
        results.append(dict(name=name, gaia=sid, status="no_ts"))
        continue
    provs = ts["provenance_name"].astype(str)
    print(f"  timeseries products by provenance: {dict(pd.Series(provs).value_counts())}")

    # Pull HLSP products specifically
    hlsp_mask = ts["obs_collection"] == "HLSP"
    hlsp_ts = ts[hlsp_mask]
    if len(hlsp_ts):
        print(f"  HLSP timeseries: {len(hlsp_ts)} rows")
        # what HLSPs?
        h_provs = pd.Series(hlsp_ts["provenance_name"].astype(str)).value_counts()
        print(f"  HLSP providers: {dict(h_provs)}")

        # Pull the product list (only HLSP)
        try:
            prods = Observations.get_product_list(hlsp_ts)
            lc_like = prods[prods["productFilename"].astype(str).str.contains("_lc.fits|llc.fits", na=False, regex=True)]
            print(f"  HLSP _lc.fits products: {len(lc_like)}")
            if len(lc_like):
                # show first few names
                for fn in lc_like["productFilename"].astype(str)[:3]:
                    print(f"    e.g. {fn}")
        except Exception as e:
            print(f"  product list error: {e}")
    else:
        print(f"  no HLSP timeseries -- only FFI cutouts (no derived LCs)")

    results.append(dict(
        name=name, gaia=sid, n_ffi=n_ffi,
        n_ts_total=len(ts),
        n_hlsp=int(hlsp_mask.sum()),
        hlsp_providers="|".join([f"{p}({c})" for p,c in pd.Series(ts[hlsp_mask]["provenance_name"].astype(str)).value_counts().items()]) if hlsp_mask.sum() else "",
    ))
    print()

R = pd.DataFrame(results)
R.to_csv("ffi_hlsp_availability.csv", index=False)
print(f"saved {len(R)} -> ffi_hlsp_availability.csv")
print(R.to_string(index=False))

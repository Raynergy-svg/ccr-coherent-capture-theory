"""Run eb_screen.py backwards on the 8 SDE>=7 hits from the widernet+smoke runs.

Expectation per the research-partner read:
  - CPD-63 349 should SURVIVE (matches the manual Test #3 vetting)
  - The 4 subgiant hits should fail on duration vs density and/or implied R
    (Saturn/Jupiter-sized inferences on R*~1.5-1.9 stars are EB territory)
  - 2M19420596+6546497 at SDE 17 R_p=7 R_E P=5d will be the most interesting
    back-validation -- the centroid was INCONCLUSIVE, so EB suspect; the screen
    should either confirm that suspicion via odd-even/2P or clear it.

Each candidate gets all 6 sub-tests + an overall verdict.
"""
import warnings, os, re
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from astropy.io import fits
from astroquery.mast import Observations
from eb_screen import run_eb_screen

results_df = pd.read_csv("widernet_bls_results.csv")
hits = results_df[results_df.status.fillna("").str.contains("SDE_OK")].copy()
print(f"Found {len(hits)} SDE_OK hits in widernet_bls_results.csv")

def load_lcs(ra, dec):
    obs = Observations.query_region(f"{ra} {dec}", radius="20 arcsec")
    ts = obs[(obs["obs_collection"]=="TESS") & (obs["dataproduct_type"]=="timeseries")]
    if len(ts) == 0: return []
    prods = Observations.get_product_list(ts)
    lc_prods = prods[prods["productSubGroupDescription"]=="LC"]
    if len(lc_prods) == 0: return []
    dl = Observations.download_products(lc_prods, download_dir="/tmp/tess_dl", verbose=False)
    out = []
    for fp in dl["Local Path"]:
        try:
            with fits.open(fp) as hdul:
                hdr = hdul[0].header
                sec = hdr.get("SECTOR", -1)
                lc = hdul["LIGHTCURVE"].data
                t = lc["TIME"]; f = lc["PDCSAP_FLUX"]; q = lc["QUALITY"]
                m = np.isfinite(t) & np.isfinite(f) & (q == 0)
                if m.sum() < 100: continue
                out.append((int(sec), t[m], f[m]/np.nanmedian(f[m])))
        except Exception:
            continue
    return out

screen_rows = []
for _, r in hits.iterrows():
    name = r["name"]
    print(f"\n{'='*78}\nSCREENING {name}  P={r['P_d']:.4f} d  SDE={r['sde']:.2f}\n{'='*78}")
    lcs = load_lcs(r.ra, r.dec)
    if not lcs:
        print("  no LCs -- skip")
        continue
    R_star = float(r.R_star) if pd.notna(r.R_star) else 1.0
    M_star = float(r.M_star) if pd.notna(r.M_star) else 1.0
    Teff   = float(r.Teff)   if pd.notna(r.Teff)   else 5700.0
    res = run_eb_screen(
        sectors=lcs,
        P=float(r.P_d), T0=float(r.T0_BTJD), dur_d=float(r.dur_h)/24.0,
        depth=float(r.depth_ppm)*1e-6,
        R_star=R_star, M_star=M_star, Teff=Teff,
    )
    print(f"  n_predicted_transits={res['n_predicted_transits']}  n_pts={res['n_pts']}")
    for s in res["sub_tests"]:
        flag = {"PASS":"PASS","INCONC":"----","FAIL":"FAIL"}[s["verdict"]]
        extras = []
        for k, v in s.items():
            if k in ("name","verdict","note"): continue
            if isinstance(v, float): extras.append(f"{k}={v:.3f}")
            else: extras.append(f"{k}={v}")
        print(f"  {s['name']:<20} {flag}  " + "  ".join(extras))
    print(f"  OVERALL: {res['verdict']}  (P={res['n_pass']} I={res['n_inconc']} F={res['n_fail']})")
    screen_rows.append(dict(
        name=name, P_d=r["P_d"], sde=r["sde"], depth_ppm=r["depth_ppm"],
        R_p=r["R_p_Rearth"], T_eq=r["T_eq_K"], R_star=R_star, M_star=M_star,
        n_pass=res["n_pass"], n_inconc=res["n_inconc"], n_fail=res["n_fail"],
        verdict=res["verdict"],
        sub1_sderatio=res["sub_tests"][0].get("value"), sub1_verdict=res["sub_tests"][0]["verdict"],
        sub2_oddeven=res["sub_tests"][1].get("value"),  sub2_verdict=res["sub_tests"][1]["verdict"],
        sub3_posfrac=res["sub_tests"][2].get("value"),  sub3_verdict=res["sub_tests"][2]["verdict"],
        sub4_duration=res["sub_tests"][3].get("value"), sub4_verdict=res["sub_tests"][3]["verdict"],
        sub5_companionR=res["sub_tests"][4].get("value"),sub5_verdict=res["sub_tests"][4]["verdict"],
        sub6_secondary=res["sub_tests"][5].get("value"),sub6_verdict=res["sub_tests"][5]["verdict"],
    ))

if screen_rows:
    df = pd.DataFrame(screen_rows)
    df.to_csv("widernet_eb_screen_results.csv", index=False)
    print(f"\n{'='*78}\nSUMMARY -- {len(df)} candidates back-validated\n{'='*78}")
    print(df[["name","P_d","sde","verdict","n_pass","n_inconc","n_fail"]].to_string(index=False))
    print(f"\nsaved widernet_eb_screen_results.csv")
print("\nDONE")

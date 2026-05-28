"""Multi-planet vs single-planet host chemistry — the 'coherence' angle.

CCT's strongest specifically-CCT claim is about COHERENCE: stars
formed in chemically coherent environments host more / better planets.

If this is real, multi-planet hosts should show stronger chemistry
peaking (less scatter, more selective) than single-planet hosts.

NOT pre-registered. Diagnostic. Asking: is there ANY angle where CCT-
specific predictions emerge?
"""
import io, time, requests
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def tap(url, q, retries=3, timeout=300):
    for a in range(retries):
        try:
            r = requests.post(url, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=timeout)
            if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
        except Exception as e: print(f"  WARN {e}")
        time.sleep(2**a)
    return pd.DataFrame()

print("="*78)
print("MULTI-PLANET vs SINGLE-PLANET host chemistry test")
print("="*78)

# Pull per-planet table to count planets per host
pps = tap(NEA, "SELECT pl_name, hostname, pl_rade, pl_eqt FROM pscomppars")
print(f"  per-planet rows: {len(pps)}")
counts = pps.groupby("hostname").size().reset_index(name="n_planets")
print(f"  host multiplicity distribution:")
print(counts.n_planets.value_counts().sort_index().to_string())

hosts = pd.read_csv("cct_test_hosts_scored.csv")
hosts = hosts.merge(counts, left_on="host_name", right_on="hostname", how="left")
hosts["n_planets"] = hosts["n_planets"].fillna(1).astype(int)
print(f"\n  APOGEE-matched hosts: {len(hosts)}")
print(f"  multiplicity distribution in APOGEE sample:")
print(hosts.n_planets.value_counts().sort_index().to_string())

# Bin: single (n=1), multi (n>=2), high-mult (n>=4)
single = hosts[hosts.n_planets == 1]
multi  = hosts[hosts.n_planets >= 2]
highmult = hosts[hosts.n_planets >= 4]

field = pd.read_csv("cct_test_field_scored.csv")
field_v = field.hab_score.dropna().values
field_med = np.median(field_v)
print(f"\n  field median hab_score: {field_med:.4f}")

for label, sub in [("single (n=1)", single), ("multi (n>=2)", multi), ("high-mult (n>=4)", highmult)]:
    if len(sub) < 5: continue
    sv = sub.hab_score.dropna().values
    md = np.median(sv) - field_med
    U, p = stats.mannwhitneyu(sv, field_v, alternative="greater")
    print(f"  {label:<22}: n={len(sv):4d}  median_shift={md:+.4f}  p_MW={p:.3e}")

# CCT specific prediction: multi-planet hosts should have TIGHTER chemistry
# (less scatter) -- formed in more coherent environment
print(f"\n  Score-distribution scatter test (CCT coherence): IQR of hab_score per group")
print(f"  field: IQR = {np.percentile(field_v, 75) - np.percentile(field_v, 25):.4f}")
for label, sub in [("single", single), ("multi >=2", multi), ("high-mult >=4", highmult)]:
    if len(sub) < 5: continue
    sv = sub.hab_score.dropna().values
    iqr = np.percentile(sv, 75) - np.percentile(sv, 25)
    print(f"  {label:<18}: n={len(sv):4d}  IQR={iqr:.4f}")

# Also test the raw chemistry scatter (sigma of each element)
print(f"\n  Raw chemistry scatter: standard dev of each element per multiplicity bin")
elems = ["ap_FE_H", "ap_MG_FE", "ap_SI_FE", "ap_CA_FE", "ap_AL_FE", "ap_TI_FE", "ap_ALPHA_M"]
for elem in elems:
    if elem not in hosts.columns: continue
    print(f"\n  --- {elem} ---")
    f_field = field[elem.replace("ap_","")].dropna()
    print(f"  field    n={len(f_field):6d}  std={f_field.std():.4f}")
    for label, sub in [("single", single), ("multi >=2", multi), ("high-mult >=4", highmult)]:
        if len(sub) < 5: continue
        v = sub[elem].dropna()
        if len(v) < 3: continue
        # Levene test: are variances different?
        try:
            stat, p = stats.levene(v, f_field)
        except: stat, p = float("nan"), float("nan")
        print(f"  {label:<14} n={len(v):4d}  std={v.std():.4f}  Levene p={p:.3e}")

print(f"\n{'='*78}\nMULTI-PLANET TEST DONE\n{'='*78}")

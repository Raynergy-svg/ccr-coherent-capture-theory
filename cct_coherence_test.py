"""Pre-registered CCT-coherence test on multi-planet host chemistry scatter.

Sealed pre-registration: PRE_REGISTRATION_2.md @ commit 1d58aa7
This analysis script committed BEFORE Levene tests run.

Design (frozen):
  1. Multi-planet hosts: APOGEE x NEA, n_planets >= 2, FGK dwarfs (Teff 4500-7000, logg > 3.8)
  2. Single-planet hosts: same, n_planets = 1
  3. Matched control: for each host, k=10 nearest neighbours in (Teff/100, logg/0.1)
     drawn from APOGEE FGK dwarfs not in any host list (SNR > 70). Pool deduplicated.

  Test: Levene (median-centered) on sigma_multi vs sigma_match for each of 12 elements
        Bonferroni alpha = 0.05/12 = 0.0042
        CONFIRMED iff >=3 elements with sigma_multi < sigma_match at p < 0.0042

  Auxiliary: also test single vs match (gradient check) and KS on stellar params
             (matching diagnostic).
"""
import io, time, requests
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def tap(url, q, retries=3, timeout=300):
    for a in range(retries):
        try:
            r = requests.post(url, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=timeout)
            if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
        except Exception as e: print(f"  WARN {e}")
        time.sleep(2**a)
    return pd.DataFrame()

# pre-registered element list (FROZEN -- do not edit during test)
ELEMENTS_H = ["ap_FE_H", "ap_MG_FE", "ap_SI_FE", "ap_CA_FE", "ap_AL_FE",
              "ap_TI_FE", "ap_MN_FE", "ap_NI_FE", "ap_C_FE", "ap_O_FE",
              "ap_N_FE", "ap_ALPHA_M"]
ELEMENTS_F = [e.replace("ap_","") for e in ELEMENTS_H]
N_ELEMENTS = len(ELEMENTS_H)
ALPHA_BONFERRONI = 0.05 / N_ELEMENTS
print(f"Pre-registered: {N_ELEMENTS} elements, Bonferroni alpha = {ALPHA_BONFERRONI:.5f}")

# 1) Load hosts + field already in CSVs
print("\n[1] Loading hosts and field (from cct_test_hosts.csv, cct_test_field.csv)...")
hosts = pd.read_csv("cct_test_hosts.csv")
field = pd.read_csv("cct_test_field.csv")
print(f"  all hosts in APOGEE: {len(hosts)}")
print(f"  field pool:          {len(field)}")

# 2) Get per-planet table to count multiplicity
print("\n[2] Pulling per-planet count from NEA pscomppars...")
pps = tap(NEA, "SELECT pl_name, hostname FROM pscomppars")
counts = pps.groupby("hostname").size().reset_index(name="n_planets")
print(f"  per-planet rows: {len(pps)}, distinct hosts: {len(counts)}")

hosts = hosts.merge(counts, left_on="host_name", right_on="hostname", how="left")
hosts["n_planets"] = hosts["n_planets"].fillna(1).astype(int)

# 3) FGK restriction (PRE-REGISTERED) and split into multi/single
print("\n[3] FGK restriction (Teff 4500-7000, logg > 3.8)...")
fgk = hosts[(hosts.ap_TEFF.between(4500, 7000)) & (hosts.ap_LOGG > 3.8)].copy()
print(f"  FGK hosts: {len(fgk)}")
multi   = fgk[fgk.n_planets >= 2].copy().reset_index(drop=True)
single  = fgk[fgk.n_planets == 1].copy().reset_index(drop=True)
print(f"  FGK multi-planet hosts (n>=2): {len(multi)}")
print(f"  FGK single-planet hosts (n=1): {len(single)}")

# 4) Restrict field control to FGK as well
print("\n[4] FGK field pool (Teff 4500-7000, logg > 3.8, SNR > 70)...")
fgk_field = field[(field.TEFF.between(4500, 7000)) & (field.LOGG > 3.8) & (field.SNR > 70)].copy().reset_index(drop=True)
print(f"  FGK field pool: {len(fgk_field)}")

# 5) Nearest-neighbour matching in (Teff/100, logg/0.1) for each host
print("\n[5] Nearest-neighbour matched control: k=10 per host, dist on (Teff/100, logg/0.1)...")

def build_matched_control(hosts_df, host_teff_col, host_logg_col,
                          field_df, field_teff_col, field_logg_col, k=10):
    """For each host, find k nearest field stars in (Teff/100, logg/0.1) coords."""
    h_coords = np.column_stack([hosts_df[host_teff_col].values / 100,
                                 hosts_df[host_logg_col].values / 0.1])
    f_coords = np.column_stack([field_df[field_teff_col].values / 100,
                                 field_df[field_logg_col].values / 0.1])
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(f_coords)
    _, idx = nn.kneighbors(h_coords)
    all_idx = np.unique(idx.flatten())
    return field_df.iloc[all_idx].copy().reset_index(drop=True)

match_for_multi  = build_matched_control(multi, "ap_TEFF", "ap_LOGG", fgk_field, "TEFF", "LOGG")
match_for_single = build_matched_control(single, "ap_TEFF", "ap_LOGG", fgk_field, "TEFF", "LOGG")
print(f"  matched control for multi: {len(match_for_multi)} (deduplicated from {10*len(multi)} draws)")
print(f"  matched control for single: {len(match_for_single)} (deduplicated from {10*len(single)} draws)")

# 6) KS diagnostic -- is the matching good? (auxiliary, descriptive)
print("\n[6] KS diagnostic: are matched controls well-matched on (Teff, logg, [Fe/H])?")
for col_h, col_f, lbl in [("ap_TEFF", "TEFF", "Teff"),
                          ("ap_LOGG", "LOGG", "logg"),
                          ("ap_FE_H", "FE_H", "[Fe/H]")]:
    for nm, sub, ctrl in [("multi", multi, match_for_multi),
                          ("single", single, match_for_single)]:
        D, p = stats.ks_2samp(sub[col_h].dropna(), ctrl[col_f].dropna())
        print(f"  {nm} vs matched_control [{lbl}]: D={D:.3f}  p={p:.3e}  "
              f"(higher p = better matched)")

# 7) PRE-REGISTERED LEVENE TESTS
print(f"\n{'='*78}\n[7] LEVENE TESTS -- multi vs matched control (PRE-REGISTERED)\n{'='*78}")
print(f"  alpha (Bonferroni over {N_ELEMENTS} elements) = {ALPHA_BONFERRONI:.5f}")
print(f"\n  element     n_multi   n_match   sigma_multi  sigma_match  ratio   Levene_p  pass?")
results = []
for eh, ef in zip(ELEMENTS_H, ELEMENTS_F):
    m_vals = multi[eh].dropna().values
    c_vals = match_for_multi[ef].dropna().values
    if len(m_vals) < 10 or len(c_vals) < 100:
        print(f"  {ef:<10s}  n_multi={len(m_vals)}  n_match={len(c_vals)}  -- skip (sample)")
        continue
    sigma_m = m_vals.std(ddof=1); sigma_c = c_vals.std(ddof=1)
    try:
        stat, p = stats.levene(m_vals, c_vals, center="median")
    except: stat, p = float("nan"), float("nan")
    passed = (p < ALPHA_BONFERRONI) and (sigma_m < sigma_c)
    flag = "PASS" if passed else "----"
    print(f"  {ef:<10s}  {len(m_vals):>7d}  {len(c_vals):>7d}  "
          f"{sigma_m:>11.4f}  {sigma_c:>11.4f}  {sigma_m/sigma_c:>5.3f}  {p:>9.3e}  {flag}")
    results.append(dict(element=ef, n_multi=len(m_vals), n_match=len(c_vals),
                        sigma_multi=sigma_m, sigma_match=sigma_c,
                        ratio=sigma_m/sigma_c, p_levene=p, pass_pre_reg=passed))

# 8) Auxiliary: single vs matched control (gradient check, descriptive)
print(f"\n{'='*78}\n[8] AUXILIARY: single-planet hosts vs matched control (gradient)\n{'='*78}")
print(f"  element     n_single  n_match   sigma_single  sigma_match  ratio   Levene_p")
single_results = []
for eh, ef in zip(ELEMENTS_H, ELEMENTS_F):
    s_vals = single[eh].dropna().values
    c_vals = match_for_single[ef].dropna().values
    if len(s_vals) < 10 or len(c_vals) < 100: continue
    sigma_s = s_vals.std(ddof=1); sigma_c = c_vals.std(ddof=1)
    try:
        stat, p = stats.levene(s_vals, c_vals, center="median")
    except: stat, p = float("nan"), float("nan")
    print(f"  {ef:<10s}  {len(s_vals):>7d}  {len(c_vals):>7d}  "
          f"{sigma_s:>11.4f}  {sigma_c:>11.4f}  {sigma_s/sigma_c:>5.3f}  {p:>9.3e}")
    single_results.append(dict(element=ef, n_single=len(s_vals), n_match=len(c_vals),
                               sigma_single=sigma_s, sigma_match=sigma_c,
                               ratio=sigma_s/sigma_c, p_levene=p))

# 9) Auxiliary: multi vs single direct comparison (no field control)
print(f"\n{'='*78}\n[9] AUXILIARY: multi vs single direct (no field control)\n{'='*78}")
print(f"  element      n_multi   n_single  sigma_multi  sigma_single  ratio    Levene_p")
mvs_results = []
for eh, ef in zip(ELEMENTS_H, ELEMENTS_F):
    m_vals = multi[eh].dropna().values
    s_vals = single[eh].dropna().values
    if len(m_vals) < 10 or len(s_vals) < 10: continue
    sigma_m = m_vals.std(ddof=1); sigma_s = s_vals.std(ddof=1)
    try:
        stat, p = stats.levene(m_vals, s_vals, center="median")
    except: stat, p = float("nan"), float("nan")
    print(f"  {ef:<10s}  {len(m_vals):>8d}  {len(s_vals):>8d}  "
          f"{sigma_m:>11.4f}  {sigma_s:>12.4f}  {sigma_m/sigma_s:>5.3f}  {p:>9.3e}")
    mvs_results.append(dict(element=ef, n_multi=len(m_vals), n_single=len(s_vals),
                            sigma_multi=sigma_m, sigma_single=sigma_s,
                            ratio=sigma_m/sigma_s, p_levene=p))

# 10) FINAL pre-registered verdict
print(f"\n{'='*78}\nPRE-REGISTERED VERDICT\n{'='*78}")
n_pass = sum(1 for r in results if r["pass_pre_reg"])
print(f"\nelements passing (sigma_multi < sigma_match at p < {ALPHA_BONFERRONI:.5f}): {n_pass}/{N_ELEMENTS}")
if n_pass >= 3:
    print(f"\n  *** CCT COHERENCE CONFIRMED (>={n_pass} of {N_ELEMENTS} elements pass) ***")
    print(f"  Passing elements:")
    for r in results:
        if r["pass_pre_reg"]:
            print(f"    {r['element']}: sigma_multi/sigma_match = {r['ratio']:.3f}  p = {r['p_levene']:.3e}")
elif n_pass >= 1:
    print(f"\n  PARTIAL: only {n_pass} elements pass; descriptive finding, not coherence confirmation")
    for r in results:
        if r["pass_pre_reg"]:
            print(f"    {r['element']}: sigma_multi/sigma_match = {r['ratio']:.3f}  p = {r['p_levene']:.3e}")
else:
    # check if direction is consistently inverted
    inverted = sum(1 for r in results if r["ratio"] > 1.0)
    print(f"\n  REJECTED: no elements pass pre-registered threshold")
    print(f"  Direction inverted (sigma_multi > sigma_match) in {inverted}/{len(results)} elements")
    if inverted >= len(results) * 0.7:
        print(f"  Matching killed the signal -> the diagnostic-sweep finding was Kepler selection bias")
    else:
        print(f"  Power-limited or unmatched-effect; no confirmation but not clearly bias either")

# Save outputs
pd.DataFrame(results).to_csv("cct_coherence_multi_vs_match.csv", index=False)
pd.DataFrame(single_results).to_csv("cct_coherence_single_vs_match.csv", index=False)
pd.DataFrame(mvs_results).to_csv("cct_coherence_multi_vs_single.csv", index=False)
print(f"\n  CSV outputs: cct_coherence_*.csv")
print(f"\nDONE")

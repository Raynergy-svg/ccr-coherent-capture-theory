"""Supplementary: re-run coherence test with STRICTER (Teff, logg, [Fe/H]) matching.

The pre-registered test PARTIALLY confirmed (4 of 12 elements pass Bonferroni)
but the third sub-criterion (KS on [Fe/H] matching) failed -- matching on
(Teff, logg) only left hosts systematically metal-richer than match.

This supplement adds [Fe/H] to the matching. If the sigma-reduction in
[α/M], [Ca/Fe], [Mn/Fe] SURVIVES strict (Teff, logg, [Fe/H]) matching,
that's clean coherence -- because by construction the [Fe/H] distribution
is matched, so the sigma reduction in OTHER elements can't be a [Fe/H]
selection artifact.

Also do the multi-vs-single direct comparison properly using only sigma-paired
matched samples.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

ELEMENTS_H = ["ap_FE_H", "ap_MG_FE", "ap_SI_FE", "ap_CA_FE", "ap_AL_FE",
              "ap_TI_FE", "ap_MN_FE", "ap_NI_FE", "ap_C_FE", "ap_O_FE",
              "ap_N_FE", "ap_ALPHA_M"]
ELEMENTS_F = [e.replace("ap_","") for e in ELEMENTS_H]
ALPHA_BONFERRONI = 0.05 / len(ELEMENTS_H)

hosts = pd.read_csv("cct_test_hosts.csv")
field = pd.read_csv("cct_test_field.csv")

# Re-pull multiplicity
import io, time, requests
NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
def tap(q, retries=3):
    for a in range(retries):
        try:
            r = requests.post(NEA, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=300)
            if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
        except: pass
        time.sleep(2**a)
    return pd.DataFrame()
pps = tap("SELECT pl_name, hostname FROM pscomppars")
counts = pps.groupby("hostname").size().reset_index(name="n_planets")
hosts = hosts.merge(counts, left_on="host_name", right_on="hostname", how="left")
hosts["n_planets"] = hosts["n_planets"].fillna(1).astype(int)

# FGK restriction
fgk = hosts[(hosts.ap_TEFF.between(4500, 7000)) & (hosts.ap_LOGG > 3.8)].copy()
multi  = fgk[fgk.n_planets >= 2].copy().reset_index(drop=True)
single = fgk[fgk.n_planets == 1].copy().reset_index(drop=True)
print(f"FGK multi: {len(multi)}  single: {len(single)}")

fgk_field = field[(field.TEFF.between(4500, 7000)) & (field.LOGG > 3.8) &
                  (field.SNR > 70) & field.FE_H.between(-2, 1)].copy().reset_index(drop=True)
print(f"FGK field pool: {len(fgk_field)}")

# STRICT matching: (Teff/100, logg/0.1, [Fe/H]/0.05)
def build_strict_match(hosts_df, host_teff, host_logg, host_feh,
                       field_df, fteff, flogg, ffeh, k=10):
    h_coords = np.column_stack([hosts_df[host_teff].values / 100.0,
                                 hosts_df[host_logg].values / 0.1,
                                 hosts_df[host_feh].values / 0.05])
    f_coords = np.column_stack([field_df[fteff].values / 100.0,
                                 field_df[flogg].values / 0.1,
                                 field_df[ffeh].values / 0.05])
    # filter NaN
    mh = np.all(np.isfinite(h_coords), axis=1)
    mf = np.all(np.isfinite(f_coords), axis=1)
    h_coords = h_coords[mh]
    f_coords = f_coords[mf]
    field_filt = field_df[mf].reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=k).fit(f_coords)
    dist, idx = nn.kneighbors(h_coords)
    all_idx = np.unique(idx.flatten())
    return field_filt.iloc[all_idx].copy().reset_index(drop=True), dist

match_multi, d_multi = build_strict_match(multi.dropna(subset=["ap_TEFF","ap_LOGG","ap_FE_H"]),
                                           "ap_TEFF","ap_LOGG","ap_FE_H",
                                           fgk_field, "TEFF","LOGG","FE_H")
match_single, d_single = build_strict_match(single.dropna(subset=["ap_TEFF","ap_LOGG","ap_FE_H"]),
                                             "ap_TEFF","ap_LOGG","ap_FE_H",
                                             fgk_field, "TEFF","LOGG","FE_H")
print(f"strict-matched control for multi:  {len(match_multi)}")
print(f"strict-matched control for single: {len(match_single)}")
print(f"  median match distance (multi):  {np.median(d_multi):.3f}")
print(f"  median match distance (single): {np.median(d_single):.3f}")

# KS diagnostic
print(f"\nSTRICT match KS diagnostic (Teff, logg, [Fe/H]):")
for col_h, col_f, lbl in [("ap_TEFF","TEFF","Teff"),
                          ("ap_LOGG","LOGG","logg"),
                          ("ap_FE_H","FE_H","[Fe/H]")]:
    for nm, sub, ctrl in [("multi", multi, match_multi),
                          ("single", single, match_single)]:
        D, p = stats.ks_2samp(sub[col_h].dropna(), ctrl[col_f].dropna())
        flag = "OK" if p > 0.01 else "FAIL"
        print(f"  {nm} vs ctrl [{lbl}]: D={D:.3f}  p={p:.3e}  {flag}")

# Levene tests with STRICT matching
print(f"\n{'='*78}\nSTRICT-MATCHED LEVENE TESTS -- multi vs strict-match\n{'='*78}")
print(f"alpha (Bonferroni) = {ALPHA_BONFERRONI:.5f}")
print(f"\nelement      n_multi   n_match   sigma_multi  sigma_match  ratio   Levene_p  pass")
results = []
for eh, ef in zip(ELEMENTS_H, ELEMENTS_F):
    m = multi[eh].dropna().values
    c = match_multi[ef].dropna().values
    if len(m) < 10 or len(c) < 50: continue
    sm = m.std(ddof=1); sc = c.std(ddof=1)
    try: stat, p = stats.levene(m, c, center="median")
    except: stat, p = float("nan"), float("nan")
    passed = (p < ALPHA_BONFERRONI) and (sm < sc)
    flag = "PASS" if passed else "----"
    print(f"  {ef:<10s}  {len(m):>7d}  {len(c):>7d}  "
          f"{sm:>11.4f}  {sc:>11.4f}  {sm/sc:>5.3f}  {p:>9.3e}  {flag}")
    results.append((ef, sm, sc, sm/sc, p, passed))

n_pass = sum(1 for r in results if r[5])
print(f"\nSTRICT match: {n_pass} of 12 elements pass at p<{ALPHA_BONFERRONI:.5f}")
if n_pass >= 3:
    print(f"  *** STRICT COHERENCE CONFIRMED ***")
elif n_pass >= 1:
    print(f"  PARTIAL ({n_pass} elements; needs >=3)")
else:
    print(f"  REJECTED after strict matching -> the unstricted finding was [Fe/H] artifact")

# Same for single
print(f"\n{'='*78}\nSTRICT-MATCHED LEVENE TESTS -- single vs strict-match\n{'='*78}")
print(f"element      n_single  n_match   sigma_single sigma_match  ratio   Levene_p")
single_results = []
for eh, ef in zip(ELEMENTS_H, ELEMENTS_F):
    s = single[eh].dropna().values
    c = match_single[ef].dropna().values
    if len(s) < 10 or len(c) < 50: continue
    ss = s.std(ddof=1); sc = c.std(ddof=1)
    try: stat, p = stats.levene(s, c, center="median")
    except: stat, p = float("nan"), float("nan")
    sig = "**" if p < ALPHA_BONFERRONI else "ns"
    print(f"  {ef:<10s}  {len(s):>7d}  {len(c):>7d}  "
          f"{ss:>11.4f}  {sc:>11.4f}  {ss/sc:>5.3f}  {p:>9.3e}  {sig}")
    single_results.append((ef, ss, sc, ss/sc, p))

# Final
print(f"\n{'='*78}\nFINAL VERDICT (strict matching)\n{'='*78}")
pass_set = [r[0] for r in results if r[5]]
single_sig = [r[0] for r in single_results if r[4] < ALPHA_BONFERRONI]
print(f"\nElements with sigma_multi < sigma_match at Bonferroni: {pass_set}")
print(f"Elements with sigma_single < sigma_match at Bonferroni: {single_sig}")

if not pass_set and not single_sig:
    print(f"\n>> Coherence signal disappears after strict matching.")
    print(f">> The unstricted finding was a metallicity-distribution artifact.")
    print(f">> CCT-coherence REJECTED at population level.")
elif pass_set and single_sig and set(pass_set) <= set(single_sig):
    print(f"\n>> Planet hosts (BOTH multi and single) have tighter scatter than matched field.")
    print(f">> Not specifically multi-planet -- a 'host' effect, not a 'multiplicity' effect.")
    print(f">> CCT-coherence as 'multi-planet coherence' is NOT confirmed.")
    print(f">> CCT-style as 'host-vs-field chemistry tightness' IS confirmed.")
elif pass_set and not single_sig:
    print(f"\n>> Multi-planet hosts show tighter scatter; single-planet hosts do not.")
    print(f">> True multi-planet coherence signal -- CCT prediction CONFIRMED.")
elif single_sig and not pass_set:
    print(f"\n>> Single-planet hosts tighter, multi-planet hosts not -- unexpected.")
    print(f">> May indicate underpowered multi-planet test or non-CCT mechanism.")
else:
    print(f"\n>> Mixed: multi passes {pass_set}, single passes {single_sig}")
    print(f">> Read carefully -- partial pattern.")

print(f"\nDONE")

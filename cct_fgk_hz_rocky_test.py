"""Pre-registered test #3: HZ-rocky in scorer's own regime (FGK).

Sealed: PRE_REGISTRATION_3.md @ commit b20613f
Frozen scorer: habitability_v2.py @ cfa1249 (unchanged)

Design (frozen, no edits allowed during run):
  - FGK restriction: Teff 4500-7000, logg > 3.8
  - HZ rocky hosts in this regime (N=3 expected): Kepler-1126, -442, -62
  - For comparison: FGK non-HZ rocky (~271), FGK sub-Neptune (~417)
  - Strict-matched field control: k=10 NN in (Teff/100, logg/0.1, [Fe/H]/0.05)
  - Test: one-sided MW-U on hab_score (host > matched control)
  - Bonferroni alpha = 0.05/3 = 0.0167
  - Strong confirm: HZ_rocky shift > 0 AND p<0.0167 AND perm p<0.05 AND effect>0.3
  - Clean reject: shift <= 0 OR p > 0.5
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

print("="*78)
print("PRE-REGISTERED TEST #3: HZ-rocky in FGK regime")
print("="*78)

# Load already-scored hosts and field
hosts = pd.read_csv("cct_test_hosts_scored.csv")
field = pd.read_csv("cct_test_field_scored.csv")
print(f"\nLoaded {len(hosts)} hosts, {len(field)} field stars (from scored CSVs)")

# FGK restriction (pre-registered)
fgk_hosts = hosts[(hosts.ap_TEFF.between(4500, 7000)) & (hosts.ap_LOGG > 3.8)].copy()
fgk_field = field[(field.TEFF.between(4500, 7000)) & (field.LOGG > 3.8) &
                  (field.SNR > 70) & field.FE_H.between(-2, 1)].copy().reset_index(drop=True)
print(f"FGK hosts: {len(fgk_hosts)}")
print(f"FGK field pool: {len(fgk_field)}")

print(f"\nFGK host category counts:")
print(fgk_hosts.host_category.value_counts().to_string())

# Identify HZ_rocky FGK hosts explicitly
hz_fgk = fgk_hosts[fgk_hosts.host_category == "HZ_rocky"].copy().reset_index(drop=True)
print(f"\nFGK HZ-rocky hosts ({len(hz_fgk)}):")
print(hz_fgk[["host_name", "host_pl_rade", "host_pl_eqt", "ap_TEFF", "ap_LOGG",
              "ap_FE_H", "ap_MG_FE", "hab_score"]].to_string(index=False))

# Strict-matched control: k=10 NN on (Teff/100, logg/0.1, [Fe/H]/0.05)
def build_strict_match(hosts_df, host_teff, host_logg, host_feh,
                       field_df, fteff, flogg, ffeh, k=10):
    h_cols = hosts_df[[host_teff, host_logg, host_feh]].dropna()
    h_coords = np.column_stack([h_cols[host_teff].values / 100.0,
                                 h_cols[host_logg].values / 0.1,
                                 h_cols[host_feh].values / 0.05])
    f_cols = field_df[[fteff, flogg, ffeh]].dropna()
    f_coords = np.column_stack([f_cols[fteff].values / 100.0,
                                 f_cols[flogg].values / 0.1,
                                 f_cols[ffeh].values / 0.05])
    field_filt = field_df.loc[f_cols.index].reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=k).fit(f_coords)
    _, idx = nn.kneighbors(h_coords)
    all_idx = np.unique(idx.flatten())
    return field_filt.iloc[all_idx].copy().reset_index(drop=True)

# Run pre-registered analysis for each category
ALPHA_BONFERRONI = 0.05 / 3
print(f"\n{'='*78}")
print(f"PRE-REGISTERED RESULTS (Bonferroni alpha = {ALPHA_BONFERRONI:.4f})")
print(f"{'='*78}")

verdict_records = []
np.random.seed(42)
N_PERM = 10000

for cat in ["HZ_rocky", "non_HZ_rocky", "sub_Neptune"]:
    sub = fgk_hosts[fgk_hosts.host_category == cat].copy().reset_index(drop=True)
    if len(sub) < 2:
        print(f"\n--- {cat}: N={len(sub)} too small ---")
        continue

    match = build_strict_match(sub, "ap_TEFF", "ap_LOGG", "ap_FE_H",
                                fgk_field, "TEFF", "LOGG", "FE_H")

    # KS check on matching (diagnostic)
    print(f"\n--- {cat}: N_host={len(sub)}  N_match={len(match)} ---")
    for col_h, col_f, lbl in [("ap_TEFF","TEFF","Teff"),
                              ("ap_LOGG","LOGG","logg"),
                              ("ap_FE_H","FE_H","[Fe/H]")]:
        D, p_ks = stats.ks_2samp(sub[col_h].dropna(), match[col_f].dropna())
        ok = "OK" if p_ks > 0.01 else "MARGINAL"
        print(f"  KS match [{lbl}]: D={D:.3f}  p={p_ks:.3e}  {ok}")

    # Primary test: MW-U on hab_score
    host_scores = sub.hab_score.dropna().values
    match_scores = match.hab_score.dropna().values
    if len(host_scores) < 2 or len(match_scores) < 20:
        print(f"  insufficient scored data")
        continue

    med_host = np.median(host_scores)
    med_match = np.median(match_scores)
    iqr_match = np.percentile(match_scores, 75) - np.percentile(match_scores, 25)
    shift = med_host - med_match
    effect_size = shift / (iqr_match / 1.349)

    U, p_mw = stats.mannwhitneyu(host_scores, match_scores, alternative="greater")

    # Permutation null
    pool = np.concatenate([host_scores, match_scores])
    perm_diffs = []
    for _ in range(N_PERM):
        idx = np.random.choice(len(pool), len(host_scores), replace=False)
        perm_med_h = np.median(pool[idx])
        perm_med_m = np.median(np.delete(pool, idx))
        perm_diffs.append(perm_med_h - perm_med_m)
    perm_diffs = np.array(perm_diffs)
    p_perm = (perm_diffs >= shift).mean()

    print(f"  median host:        {med_host:.4f}")
    print(f"  median match:       {med_match:.4f}")
    print(f"  shift (host-match): {shift:+.4f}")
    print(f"  effect size:        {effect_size:+.2f} (shift / IQR_match*1.349)")
    print(f"  MW-U (one-sided greater): U={U:.0f}, p_MW = {p_mw:.4e}")
    print(f"  permutation null: p_perm(>=obs) = {p_perm:.4e}")

    # Pre-registered verdict
    if shift > 0 and p_mw < ALPHA_BONFERRONI and p_perm < 0.05 and effect_size > 0.3:
        verdict = "CONFIRMED (>0 shift, p<Bonferroni, perm<.05, effect>0.3)"
    elif shift <= 0 or p_mw > 0.5:
        verdict = "CLEANLY REJECTED (shift<=0 OR p_MW>0.5)"
    elif shift > 0 and p_mw > ALPHA_BONFERRONI:
        verdict = "UNDERPOWERED (correct direction but below Bonferroni)"
    else:
        verdict = "PARTIAL"
    print(f"\n  >>> {cat} verdict: {verdict}")

    verdict_records.append(dict(category=cat, n=len(sub), n_match=len(match),
                                 med_host=med_host, med_match=med_match,
                                 shift=shift, effect_size=effect_size,
                                 p_MW=p_mw, p_perm=p_perm, verdict=verdict))

# Final summary
print(f"\n{'='*78}")
print("PRE-REGISTERED FINAL DISPOSITION (test #3)")
print(f"{'='*78}")
print()
hz_row = next((r for r in verdict_records if r["category"]=="HZ_rocky"), None)
if hz_row is None:
    print("No HZ-rocky data; test failed to run")
else:
    print(f"HZ-rocky (the test that matters):")
    print(f"  shift     = {hz_row['shift']:+.4f}")
    print(f"  p_MW      = {hz_row['p_MW']:.4e}")
    print(f"  p_perm    = {hz_row['p_perm']:.4e}")
    print(f"  effect    = {hz_row['effect_size']:+.2f}")
    print(f"  verdict   = {hz_row['verdict']}")
    print()
    if "CONFIRMED" in hz_row["verdict"]:
        print("  *** CCT-FGK CONFIRMED -- scorer predicts HZ-rocky in its own regime ***")
        print("  Original pre-reg #1 failure was sample mismatch (M-dwarf contamination)")
    elif "CLEANLY REJECTED" in hz_row["verdict"]:
        print("  *** CCT-FGK CLEANLY REJECTED ***")
        print("  Scorer fails to predict HZ-rocky even in its design regime (FGK)")
        print("  This is a real falsification, not a sample issue")
    else:
        print(f"  Reported as: {hz_row['verdict']}")

print()
print("For context (non-HZ-rocky and sub-Neptune, same test):")
for r in verdict_records:
    if r["category"] != "HZ_rocky":
        print(f"  {r['category']:<15s} n={r['n']:>3d}  shift={r['shift']:+.4f}  "
              f"p_MW={r['p_MW']:.3e}  -> {r['verdict']}")

pd.DataFrame(verdict_records).to_csv("cct_fgk_hz_rocky_results.csv", index=False)
print(f"\nCSV: cct_fgk_hz_rocky_results.csv")
print("\nDONE")

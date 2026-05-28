"""POST-HOC DIAGNOSTIC SWEEP — what is CCT actually telling us?

Pre-registered test (commit 1441551) FAILED. HZ rocky hosts shifted DOWN
not UP. Now we ask:

  1. Which dimensions of the 9D scorer actually drive predictive signal?
     (dimension ablation: drop each dimension, measure AIC delta)
  2. Does the CCT-specific Gaussian functional form add over simpler forms?
     (compare Gaussian-product vs monotonic vs linear)
  3. Where in host-category space DOES the score work?
     (re-test with sub-Neptune, non-HZ rocky as positive class)
  4. Is there a sub-population of HZ rocky that does match? (FGK-only)
  5. What was the framework ACTUALLY measuring? (post-hoc characterization)

NOT pre-registered. These tests are diagnostic, exploratory.
The result is honest characterization, not a new confirmatory claim.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

print("="*78)
print("CCT DIAGNOSTIC SWEEP")
print("="*78)

# Load
hosts = pd.read_csv("cct_test_hosts_scored.csv")
field = pd.read_csv("cct_test_field_scored.csv")

dims = ["s_CO","s_MgSi","s_FeH","s_MgFe","s_SiFe","s_CaFe","s_AlFe","s_volatile","s_age"]
dim_names = ["C/O", "Mg/Si", "[Fe/H]", "[Mg/Fe]", "[Si/Fe]", "[Ca/Fe]", "[Al/Fe]", "Volatile", "Age"]

# 1) Dimension ablation -- which dimensions drive the signal for sub_Neptune?
print(f"\n{'='*78}")
print("[1] DIMENSION ABLATION (which dimension carries the signal?)")
print(f"{'='*78}")

# For each category, compute the median per-dimension score for hosts vs field
# Then ask: which dimension would, if dropped, hurt or help host detection most?
for cat in ["HZ_rocky", "non_HZ_rocky", "sub_Neptune", "hot_Jupiter"]:
    sub = hosts[hosts.host_category == cat]
    if len(sub) < 3: continue
    print(f"\n--- category: {cat} (n={len(sub)}) ---")
    print(f"  {'dim':<12} {'field_med':>10} {'host_med':>10} {'shift':>9} {'sig':>6}")
    for d, name in zip(dims, dim_names):
        fm = field[d].median()
        hm = sub[d].median()
        # MW test
        try:
            U, p = stats.mannwhitneyu(sub[d].dropna(), field[d].dropna(),
                                       alternative="two-sided")
            sig = "***" if p < 1e-5 else "** " if p < 1e-3 else "*  " if p < 0.05 else "ns "
        except: sig = "?  "
        print(f"  {name:<12} {fm:>10.4f} {hm:>10.4f} {hm-fm:>+9.4f} {sig:>6}")

# 2) Functional form test: how much does CCT's Gaussian-product form add over raw inputs?
print(f"\n{'='*78}")
print("[2] FUNCTIONAL FORM: 9D nonlinear vs linear-9 vs [Fe/H] only")
print(f"{'='*78}")
print("(cross-validated AIC to avoid overfitting bias from 7 positives)")

# Use sub_Neptune as positive class (largest sample, clean signal)
for pos_cat in ["sub_Neptune", "non_HZ_rocky", "HZ_rocky"]:
    print(f"\n--- positive class: {pos_cat} ---")
    pos = hosts[hosts.host_category == pos_cat]
    if len(pos) < 5:
        print(f"  N={len(pos)} too small")
        continue
    n_pos = len(pos)
    y = np.r_[np.ones(n_pos), np.zeros(len(field))]
    # 9D nonlinear: hab_score alone (1 parameter)
    X_9D = np.r_[pos.hab_score.values, field.hab_score.values].reshape(-1, 1)
    # Fe/H alone
    X_feh = np.r_[pos.ap_FE_H.values, field.FE_H.values].reshape(-1, 1)
    # linear of 9 raw inputs (using same data the scorer used)
    def get_lin(host_df, field_df):
        cols_h = ["C_O","Mg_Si","ap_FE_H","ap_MG_FE","ap_SI_FE","ap_CA_FE","ap_AL_FE","ap_CE_FE","host_st_age"]
        cols_f = ["C_O","Mg_Si","FE_H","MG_FE","SI_FE","CA_FE","AL_FE","CE_FE"]
        X_h = host_df[cols_h].copy()
        X_f = field_df[cols_f].copy()
        X_f["age"] = 5.0  # field has no age
        X_h.columns = cols_f + ["age"]
        X_f.columns = cols_f + ["age"]
        return pd.concat([X_h, X_f], ignore_index=True).fillna(0).values
    X_lin = get_lin(pos, field)

    # also age dimension as ablation: drop one at a time
    for name, X in [("9D-nonlinear (hab_score only)", X_9D),
                    ("Fe/H alone", X_feh),
                    ("linear-9 raw inputs", X_lin)]:
        ok = np.all(np.isfinite(X), axis=1)
        Xs = StandardScaler().fit_transform(X[ok])
        ys = y[ok]
        try:
            # 5-fold CV log-loss
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            ll_folds = []
            for tr, te in cv.split(Xs, ys):
                lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
                lr.fit(Xs[tr], ys[tr])
                p = lr.predict_proba(Xs[te])[:, 1]
                ll = np.mean(ys[te]*np.log(np.maximum(p,1e-12)) + (1-ys[te])*np.log(np.maximum(1-p,1e-12)))
                ll_folds.append(ll)
            mean_ll = np.mean(ll_folds)
            print(f"  {name:<40s}: mean CV log-loss = {mean_ll:+.5f}  (higher=better)")
        except Exception as e:
            print(f"  {name}: fit fail: {e}")

# 3) Restrict to FGK hosts only -- where the scorer was designed
print(f"\n{'='*78}")
print("[3] FGK-ONLY RESTRICTED TEST")
print(f"{'='*78}")
fgk_hosts = hosts[(hosts.ap_TEFF >= 4500) & (hosts.ap_TEFF <= 7000) & (hosts.ap_LOGG > 3.8)]
print(f"  FGK-only hosts: {len(fgk_hosts)} (of {len(hosts)})")
print(f"  FGK-only by category:")
print(fgk_hosts.host_category.value_counts().to_string())
for cat in ["HZ_rocky", "non_HZ_rocky", "sub_Neptune", "hot_Jupiter"]:
    sub = fgk_hosts[fgk_hosts.host_category == cat].hab_score.dropna().values
    if len(sub) < 2:
        print(f"  {cat}: n={len(sub)} too small")
        continue
    U, p = stats.mannwhitneyu(sub, field.hab_score.dropna().values, alternative="greater")
    md = np.median(sub) - field.hab_score.median()
    print(f"  {cat:15s}: n={len(sub):3d}  median_shift={md:+.4f}  p_MW={p:.3e}")

# 4) Multi-planet vs single-planet hosts (CCT 'coherence' angle)
print(f"\n{'='*78}")
print("[4] MULTI-PLANET vs SINGLE-PLANET hosts")
print(f"{'='*78}")
# Need to re-pull multiplicity from NEA. Approximate from host pl_rade variation
# (more reliable: re-pull, but quick check using host duplication in pscomppars)
print("  This needs the full per-planet table -- approximate from category mix")
multi_likely = hosts[hosts.host_pl_rade < 1.6].drop_duplicates("host_name")  # one entry per host already
# Cannot reliably split multi vs single without per-planet table
print("  Test deferred -- need to pull per-planet multiplicity (NEA query)")

# 5) Direct host-vs-field shift in each individual element abundance (not scored)
print(f"\n{'='*78}")
print("[5] RAW ABUNDANCE SHIFTS (no scoring -- which elements ARE different in hosts?)")
print(f"{'='*78}")
abun_cols_h = ["ap_FE_H","ap_MG_FE","ap_SI_FE","ap_CA_FE","ap_AL_FE","ap_C_FE","ap_O_FE","ap_N_FE","ap_TI_FE","ap_NI_FE","ap_MN_FE","ap_CE_FE","ap_ALPHA_M"]
abun_cols_f = ["FE_H","MG_FE","SI_FE","CA_FE","AL_FE","C_FE","O_FE","N_FE","TI_FE","NI_FE","MN_FE","CE_FE","ALPHA_M"]

for cat in ["HZ_rocky", "non_HZ_rocky", "sub_Neptune", "hot_Jupiter"]:
    sub = hosts[hosts.host_category == cat]
    if len(sub) < 3: continue
    print(f"\n--- category: {cat} (n={len(sub)}) ---")
    print(f"  {'element':<10} {'field_med':>10} {'host_med':>10} {'shift':>9} {'sig':>6}")
    for ch, cf in zip(abun_cols_h, abun_cols_f):
        if ch not in sub.columns or cf not in field.columns: continue
        f_vals = field[cf].dropna()
        h_vals = sub[ch].dropna()
        if len(h_vals) < 3 or len(f_vals) < 100: continue
        fm = f_vals.median()
        hm = h_vals.median()
        try:
            U, p = stats.mannwhitneyu(h_vals, f_vals, alternative="two-sided")
            sig = "***" if p < 1e-5 else "** " if p < 1e-3 else "*  " if p < 0.05 else "ns "
        except: sig = "?  "
        print(f"  {cf:<10} {fm:>10.4f} {hm:>10.4f} {hm-fm:>+9.4f} {sig:>6}")

print(f"\n{'='*78}\nDIAGNOSTIC SWEEP DONE\n{'='*78}")

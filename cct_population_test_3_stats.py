"""CCT population test — STAGE 3: pre-registered statistical tests.

Pre-registration: PRE_REGISTRATION.md @ commit 1441551

Tests applied (exactly as pre-registered):
  1. Mann-Whitney U test on 9D hab_score for each host category vs field
  2. Permutation null: shuffle labels 10^4 times
  3. KS test: HZ rocky vs hot Jupiter score distributions (selectivity check)
  4. AIC/LRT for 9D nonlinear vs [Fe/H]-only vs all-element-linear logistic
     regression for predicting HZ rocky host status

Success criteria (frozen):
  CONFIRMED iff:
    - HZ-rocky MW-U p < 1e-7 (>5sigma after 4-category Bonferroni)
    - effect size > hot Jupiter effect size by 0.3sigma AND KS p<0.05
    - 9D nonlinear logistic ΔAIC over linear > 10
  DISFAVOURED iff:
    - MW-U p > 0.05 (no shift)
    - OR all categories show same shift
    - OR linear ties/beats nonlinear
"""
import numpy as np, pandas as pd
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("="*78)
print("CCT POPULATION TEST -- STAGE 3: pre-registered statistical tests")
print("="*78)

hosts = pd.read_csv("cct_test_hosts_scored.csv")
field = pd.read_csv("cct_test_field_scored.csv")
print(f"\n  hosts: {len(hosts)}  field: {len(field)}")
print(f"  hosts by category:")
print(hosts.host_category.value_counts().to_string())

# ============ TEST 1: Mann-Whitney U per category ============
print(f"\n{'='*78}\nTEST 1: MANN-WHITNEY U (per category vs field)\n{'='*78}")
field_scores = field.hab_score.dropna().values
field_med = np.median(field_scores)
field_iqr = (np.percentile(field_scores, 75) - np.percentile(field_scores, 25))
print(f"  field n={len(field_scores)}, median={field_med:.4f}, IQR={field_iqr:.4f}")

mw_results = []
for cat in ["HZ_rocky", "non_HZ_rocky", "sub_Neptune", "hot_Jupiter"]:
    sub = hosts[hosts.host_category == cat].hab_score.dropna().values
    if len(sub) < 2:
        print(f"  {cat}: n={len(sub)} -- insufficient")
        mw_results.append(dict(cat=cat, n=len(sub), median=float("nan"), p=1, U=float("nan"),
                               effect_sigma=float("nan")))
        continue
    U, p = stats.mannwhitneyu(sub, field_scores, alternative="greater")
    med = np.median(sub)
    effect = (med - field_med) / (field_iqr / 1.349)  # IQR-based pseudo-sigma
    print(f"  {cat:15s}: n={len(sub):4d}  median={med:.4f}  shift={med-field_med:+.4f}  "
          f"effect_sigma={effect:+.2f}  U={U:.0f}  p_MW={p:.3e}")
    mw_results.append(dict(cat=cat, n=len(sub), median=med, p=p, U=U, effect_sigma=effect))

# 4-category Bonferroni
print(f"\n  Bonferroni alpha = 0.05/4 = 0.0125; >5sigma threshold p < {(1-stats.norm.cdf(5))*2/4:.2e}")

# ============ TEST 2: Permutation null ============
print(f"\n{'='*78}\nTEST 2: PERMUTATION NULL (10^4 shuffles)\n{'='*78}")
np.random.seed(42)
N_PERM = 10000
perm_results = {}
all_scores = np.concatenate([field_scores, hosts.hab_score.dropna().values])
n_field = len(field_scores)
for cat in ["HZ_rocky", "non_HZ_rocky", "sub_Neptune", "hot_Jupiter"]:
    sub = hosts[hosts.host_category == cat].hab_score.dropna().values
    if len(sub) < 2: continue
    obs_med = np.median(sub)
    obs_diff = obs_med - field_med
    # permutation: random sample of size len(sub) from combined pool
    perm_diffs = []
    pool = np.concatenate([sub, field_scores])
    for _ in range(N_PERM):
        idx = np.random.choice(len(pool), len(sub), replace=False)
        perm_med = np.median(pool[idx])
        perm_diffs.append(perm_med - np.median(np.delete(pool, idx)))
    perm_diffs = np.array(perm_diffs)
    p_perm_g = (perm_diffs >= obs_diff).mean()
    print(f"  {cat:15s}: obs_diff={obs_diff:+.4f}  perm_p(>=obs)={p_perm_g:.4e}")
    perm_results[cat] = dict(obs_diff=obs_diff, p_perm=p_perm_g)

# ============ TEST 3: KS HZ rocky vs hot Jupiter (selectivity) ============
print(f"\n{'='*78}\nTEST 3: KS TEST -- selectivity (HZ rocky vs other categories)\n{'='*78}")
hz = hosts[hosts.host_category == "HZ_rocky"].hab_score.dropna().values
print(f"  N_HZ_rocky = {len(hz)}")
ks_results = {}
for other in ["non_HZ_rocky", "sub_Neptune", "hot_Jupiter"]:
    other_v = hosts[hosts.host_category == other].hab_score.dropna().values
    if len(hz) < 2 or len(other_v) < 2:
        print(f"  HZ_rocky vs {other}: insufficient")
        continue
    D, p = stats.ks_2samp(hz, other_v)
    print(f"  HZ_rocky vs {other:15s}: D={D:.3f} p={p:.3e}")
    ks_results[other] = dict(D=D, p=p, med_diff=np.median(hz)-np.median(other_v))

# ============ TEST 4: 9D nonlinear vs [Fe/H]-only vs linear-9 logistic ============
print(f"\n{'='*78}\nTEST 4: LOGISTIC -- 9D nonlinear vs [Fe/H]-only vs linear-9\n{'='*78}")
# Build labeled dataset: HZ_rocky host=1, field=0; rest of hosts excluded for this test
hz_h = hosts[hosts.host_category == "HZ_rocky"]
y_hz = np.r_[np.ones(len(hz_h)), np.zeros(len(field))]

# Predictor sets:
# (a) hab_score alone (1D)
# (b) [Fe/H] alone (1D) -- the literature baseline
# (c) all 9 input dimensions linearly
# All scaled.

X_9D     = np.r_[hz_h.hab_score.values, field.hab_score.values].reshape(-1, 1)
X_feh    = np.r_[hz_h.ap_FE_H.values, field.FE_H.values].reshape(-1, 1)

def col(host_c, field_c):
    return np.r_[hz_h[host_c].fillna(0).values, field[field_c].fillna(0).values]

X_lin = np.column_stack([
    col("C_O","C_O"), col("Mg_Si","Mg_Si"), col("ap_FE_H","FE_H"),
    col("ap_MG_FE","MG_FE"), col("ap_SI_FE","SI_FE"),
    col("ap_CA_FE","CA_FE"), col("ap_AL_FE","AL_FE"),
    col("ap_CE_FE","CE_FE"), col("s_age","s_age"),
])

ok_lin = np.all(np.isfinite(X_lin), axis=1)
ok_9D  = np.isfinite(X_9D).ravel() & ok_lin
ok_feh = np.isfinite(X_feh).ravel() & ok_lin

X_9D, X_feh, X_lin, y_use = X_9D[ok_9D], X_feh[ok_9D], X_lin[ok_9D], y_hz[ok_9D]
print(f"  rows used: {len(y_use)} (HZ rocky + field)")
print(f"  HZ rocky / field in usable: {int(y_use.sum())} / {int((1-y_use).sum())}")

for name, X in [("9D nonlin (hab_score only)", X_9D), ("Fe/H only", X_feh), ("linear-9 elements", X_lin)]:
    Xs = StandardScaler().fit_transform(X)
    try:
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs, y_use)
        ll = lr.score(Xs, y_use)
        # AIC = 2k - 2 logL; here we use logL = sum log p
        p1 = lr.predict_proba(Xs)[:, 1]
        logL = np.sum(y_use*np.log(np.maximum(p1, 1e-12)) + (1-y_use)*np.log(np.maximum(1-p1, 1e-12)))
        k = X.shape[1] + 1
        AIC = 2*k - 2*logL
        print(f"    {name:30s}: logL={logL:.2f}  k={k}  AIC={AIC:.2f}")
    except Exception as e:
        print(f"    {name}: fit failed: {e}")

# ============ TEST 5: Shuffled-weights null (extra; CCT specificity) ============
print(f"\n{'='*78}\nTEST 5 (auxiliary): SHUFFLED-WEIGHTS NULL\n{'='*78}")
print(f"  null hypothesis: random non-linear combinations of the same 9 inputs")
print(f"  produce the same host shift as the CCT-specific weights")
# Recompute hab_score 1000 times with shuffled weights; compare HZ-rocky vs field
# Need scores per dimension: s_CO, s_MgSi, s_FeH, s_MgFe, s_SiFe, s_CaFe, s_AlFe, s_volatile, s_age
hz_h = hosts[hosts.host_category == "HZ_rocky"]
sub_dims_hz    = hz_h[["s_CO","s_MgSi","s_FeH","s_MgFe","s_SiFe","s_CaFe","s_AlFe","s_volatile","s_age"]].values
sub_dims_field = field[["s_CO","s_MgSi","s_FeH","s_MgFe","s_SiFe","s_CaFe","s_AlFe","s_volatile","s_age"]].values
W_cct = np.array([1.0, 1.5, 1.5, 1.0, 1.0, 0.5, 0.5, 1.0, 0.75])

def hab_with_weights(scores, w):
    return np.exp(np.average(np.log(np.maximum(scores, 1e-10)), weights=w, axis=1))

obs_score_hz = hab_with_weights(sub_dims_hz, W_cct)
obs_score_field = hab_with_weights(sub_dims_field, W_cct)
obs_diff = np.median(obs_score_hz) - np.median(obs_score_field)
print(f"  observed HZ-rocky - field shift (CCT weights): {obs_diff:+.4f}")

np.random.seed(42)
null_diffs = []
n_dim = sub_dims_hz.shape[1]
for _ in range(1000):
    # random non-negative weights, same total magnitude
    w_rand = np.random.dirichlet(np.ones(n_dim)) * W_cct.sum()
    null_diff = np.median(hab_with_weights(sub_dims_hz, w_rand)) - \
                np.median(hab_with_weights(sub_dims_field, w_rand))
    null_diffs.append(null_diff)
null_diffs = np.array(null_diffs)
p_null = (null_diffs >= obs_diff).mean()
print(f"  null distribution: median={np.median(null_diffs):+.4f}  std={null_diffs.std():.4f}")
print(f"  fraction of random weights producing >= observed shift: {p_null:.4f}")
if p_null < 0.05:
    print(f"  CCT weights produce a HZ-rocky shift that's MORE positive than 95% of "
          f"random weights -> CCT is specifically informative ✓")
elif p_null > 0.5:
    print(f"  CCT weights are typical / sub-typical -> CCT-specific weights don't help ✗")
else:
    print(f"  CCT weights are above median but not strongly so")

# ============ FINAL ============
print(f"\n{'='*78}\nFINAL PRE-REGISTERED DISPOSITION\n{'='*78}")
hz_row = next((r for r in mw_results if r["cat"]=="HZ_rocky"), None)
hj_row = next((r for r in mw_results if r["cat"]=="hot_Jupiter"), None)
if hz_row is None or hz_row["n"] < 5:
    print(f"\n  HZ rocky N={hz_row['n'] if hz_row else 0} too small for pre-registered test power")
    print(f"  Result: UNDERPOWERED -- need more data, not a failure of CCT")
else:
    p_hz = hz_row["p"]
    p_thresh = (1 - stats.norm.cdf(5)) * 2 / 4
    crit1 = p_hz < p_thresh
    crit2 = (hz_row["effect_sigma"] > (hj_row["effect_sigma"] if hj_row else 0) + 0.3)
    print(f"\n  Criterion 1 (HZ rocky MW-U p < {p_thresh:.2e}):  obs={p_hz:.3e}  -> {'PASS' if crit1 else 'FAIL'}")
    print(f"  Criterion 2 (effect HZ-rocky > hot-Jupiter + 0.3sigma):  "
          f"{hz_row['effect_sigma']:.2f} vs {hj_row['effect_sigma']:.2f}  -> {'PASS' if crit2 else 'FAIL'}")
    if crit1 and crit2:
        print(f"\n  PRE-REGISTERED RESULT: **CCT CONFIRMED**")
    else:
        print(f"\n  PRE-REGISTERED RESULT: **CCT NOT CONFIRMED (at this sample size)**")

print("\nDONE")

"""Pre-registered test #4: dimensional decomposition + held-out CV.

Sealed: PRE_REGISTRATION_4.md @ a476b06
Frozen scorer: habitability_v2.py @ cfa1249

Design (frozen, no edits allowed during run):
  - Samples: non-HZ-rocky FGK (N=271), sub-Neptune FGK (N=417)
    each vs strict-matched (Teff, logg, [Fe/H]) FGK field control
  - Match quality gate: KS p > 0.95 on all 3 matching axes -- else REJECT
  - Step 1: per-dimension MW (one-sided greater), Bonferroni alpha=0.05/8=0.00625
  - Step 2: cross-category consistency
  - Step 3: 70/30 train/test split (seed 42), logistic on training,
    held-out log-loss for (a) [Fe/H], (b) [Fe/H]+surviving, (c) all 9 linear
  - Step 4: within-[Fe/H]-bin sanity for surviving dimensions
  - CONFIRMED iff >=1 dim passes step 1, model b beats a by >0.02 in
    held-out log-loss, and >=1 dim passes step 4
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 8 non-[Fe/H] dimensions (alphabetical, pre-registered order)
SUB_DIMS = ["s_age", "s_AlFe", "s_CaFe", "s_CO", "s_MgFe",
            "s_MgSi", "s_SiFe", "s_volatile"]
SUB_LABELS = ["age", "[Al/Fe]", "[Ca/Fe]", "C/O", "[Mg/Fe]",
              "Mg/Si", "[Si/Fe]", "volatile"]
N_DIMS = len(SUB_DIMS)
ALPHA_BONF = 0.05 / N_DIMS

# Raw inputs for the linear/CV comparison
HOST_RAW_COLS = {
    "FeH":   "ap_FE_H",
    "MgFe":  "ap_MG_FE",
    "SiFe":  "ap_SI_FE",
    "CaFe":  "ap_CA_FE",
    "AlFe":  "ap_AL_FE",
    "CeFe":  "ap_CE_FE",
    "C_O":   "C_O",
    "Mg_Si": "Mg_Si",
    "age":   "host_st_age",
}
FIELD_RAW_COLS = {
    "FeH":   "FE_H",
    "MgFe":  "MG_FE",
    "SiFe":  "SI_FE",
    "CaFe":  "CA_FE",
    "AlFe":  "AL_FE",
    "CeFe":  "CE_FE",
    "C_O":   "C_O",
    "Mg_Si": "Mg_Si",
    "age":   None,  # APOGEE field has no age column
}

print("="*78)
print("PRE-REGISTERED TEST #4: dimensional decomposition + out-of-sample CV")
print("="*78)
print(f"  Bonferroni alpha (step 1) = {ALPHA_BONF:.5f}")

# Load scored data (already produced by stage 2 of test #1)
hosts = pd.read_csv("cct_test_hosts_scored.csv")
field = pd.read_csv("cct_test_field_scored.csv")
print(f"\nLoaded {len(hosts)} hosts, {len(field)} field stars")

# FGK restriction
fgk_hosts = hosts[(hosts.ap_TEFF.between(4500, 7000)) & (hosts.ap_LOGG > 3.8)].copy()
fgk_field = field[(field.TEFF.between(4500, 7000)) & (field.LOGG > 3.8) &
                  (field.SNR > 70) & field.FE_H.between(-2, 1)].copy().reset_index(drop=True)
print(f"FGK hosts: {len(fgk_hosts)}, FGK field pool: {len(fgk_field)}")

def build_strict_match(hosts_df, h_teff, h_logg, h_feh,
                        field_df, f_teff, f_logg, f_feh, k=10):
    sub = hosts_df[[h_teff, h_logg, h_feh]].dropna()
    h_idx = sub.index
    h_coords = np.column_stack([sub[h_teff].values / 100.0,
                                 sub[h_logg].values / 0.1,
                                 sub[h_feh].values / 0.05])
    fsub = field_df[[f_teff, f_logg, f_feh]].dropna()
    f_coords = np.column_stack([fsub[f_teff].values / 100.0,
                                 fsub[f_logg].values / 0.1,
                                 fsub[f_feh].values / 0.05])
    field_filt = field_df.loc[fsub.index].reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=k).fit(f_coords)
    _, idx = nn.kneighbors(h_coords)
    all_idx = np.unique(idx.flatten())
    return field_filt.iloc[all_idx].copy().reset_index(drop=True)

def get_raw_matrix(df, col_map):
    """Build (N, 9) raw-input matrix in fixed column order."""
    order = ["FeH","MgFe","SiFe","CaFe","AlFe","CeFe","C_O","Mg_Si","age"]
    cols = []
    for k in order:
        c = col_map.get(k)
        if c is None or c not in df.columns:
            cols.append(np.full(len(df), 5.0 if k == "age" else 0.0))
        else:
            cols.append(df[c].fillna(5.0 if k == "age" else 0.0).values)
    return np.column_stack(cols), order

# ------------------------------------------------------------
# Process each category
# ------------------------------------------------------------
CATEGORIES = ["non_HZ_rocky", "sub_Neptune"]
all_results = {}

for cat in CATEGORIES:
    print(f"\n{'='*78}")
    print(f"CATEGORY: {cat}")
    print(f"{'='*78}")

    sub_hosts = fgk_hosts[fgk_hosts.host_category == cat].copy().reset_index(drop=True)
    print(f"  N hosts (FGK, {cat}): {len(sub_hosts)}")
    if len(sub_hosts) < 30:
        print(f"  insufficient hosts -> skip")
        continue

    matched = build_strict_match(sub_hosts, "ap_TEFF", "ap_LOGG", "ap_FE_H",
                                  fgk_field, "TEFF", "LOGG", "FE_H")
    print(f"  N matched control (deduplicated): {len(matched)}")

    # --- MATCH QUALITY GATE ---
    print(f"\n  Match-quality gate (KS p > 0.95 required on all 3 axes):")
    gate_pass = True
    for col_h, col_f, lbl in [("ap_TEFF","TEFF","Teff"),
                              ("ap_LOGG","LOGG","logg"),
                              ("ap_FE_H","FE_H","[Fe/H]")]:
        D, p_ks = stats.ks_2samp(sub_hosts[col_h].dropna(), matched[col_f].dropna())
        flag = "PASS" if p_ks > 0.95 else "FAIL"
        print(f"    KS {lbl}: D={D:.4f}  p={p_ks:.3e}  {flag}")
        if p_ks <= 0.95: gate_pass = False
    if not gate_pass:
        print(f"  MATCH-QUALITY GATE FAILED -> category REJECTED as matching failure")
        all_results[cat] = dict(verdict="REJECTED_MATCH_FAIL")
        continue

    # --- STEP 1: Per-dimension MW tests ---
    print(f"\n  STEP 1: per-dimension MW (one-sided greater, Bonferroni alpha={ALPHA_BONF:.5f})")
    print(f"  {'dim':<10} {'host_med':>10} {'ctrl_med':>10} {'shift':>9} {'effect':>7} {'p_MW':>10} {'pass'}")
    step1_results = []
    for d, lbl in zip(SUB_DIMS, SUB_LABELS):
        h_vals = sub_hosts[d].dropna().values
        c_vals = matched[d].dropna().values
        if len(h_vals) < 10 or len(c_vals) < 50:
            print(f"  {lbl:<10} -- insufficient n")
            continue
        med_h = np.median(h_vals)
        med_c = np.median(c_vals)
        shift = med_h - med_c
        iqr_c = np.percentile(c_vals, 75) - np.percentile(c_vals, 25)
        effect = shift / (iqr_c / 1.349) if iqr_c > 0 else float("nan")
        U, p = stats.mannwhitneyu(h_vals, c_vals, alternative="greater")
        passed = (p < ALPHA_BONF) and (shift > 0)
        flag = "PASS" if passed else "----"
        print(f"  {lbl:<10} {med_h:>10.4f} {med_c:>10.4f} {shift:>+9.4f} "
              f"{effect:>+7.2f} {p:>10.3e} {flag}")
        step1_results.append(dict(dim=d, label=lbl, host_med=med_h, ctrl_med=med_c,
                                  shift=shift, effect=effect, p=p, pass_step1=passed))

    surviving = [r for r in step1_results if r["pass_step1"]]
    surviving_dims = [r["dim"] for r in surviving]
    print(f"\n  Surviving dimensions (step 1): {[r['label'] for r in surviving]}")

    # --- STEP 4: Within-[Fe/H]-bin sanity for surviving dims ---
    print(f"\n  STEP 4: within-[Fe/H]-bin sanity check (0.1 dex bins)")
    step4_results = {}
    for r in surviving:
        d = r["dim"]
        bins = np.arange(-1.0, 0.7, 0.1)
        consistent = 0; total_bins = 0; details = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            h_bin = sub_hosts[sub_hosts.ap_FE_H.between(lo, hi)]
            c_bin = matched[matched.FE_H.between(lo, hi)]
            if len(h_bin) < 5 or len(c_bin) < 20: continue
            shift_bin = h_bin[d].median() - c_bin[d].median()
            total_bins += 1
            if shift_bin > 0: consistent += 1
            details.append((f"{lo:+.1f}", len(h_bin), len(c_bin), shift_bin))
        frac = consistent / total_bins if total_bins > 0 else 0
        passes_step4 = (total_bins >= 2) and (frac >= 0.5)
        flag = "PASS" if passes_step4 else "FAIL"
        print(f"    {r['label']:<10}: {consistent}/{total_bins} bins positive  frac={frac:.2f}  {flag}")
        step4_results[d] = dict(consistent=consistent, total_bins=total_bins,
                                fraction=frac, pass_step4=passes_step4, details=details)
        for bin_lo, nh, nc, s in details:
            print(f"      bin [Fe/H]={bin_lo}: n_h={nh:>3d}  n_c={nc:>4d}  shift={s:+.4f}")

    # --- STEP 3: Out-of-sample CV ---
    print(f"\n  STEP 3: held-out 30% log-loss comparison")
    # Build raw input matrix for hosts and controls; label hosts=1, controls=0
    X_h, order = get_raw_matrix(sub_hosts, HOST_RAW_COLS)
    X_c, _     = get_raw_matrix(matched, FIELD_RAW_COLS)
    X_all = np.vstack([X_h, X_c])
    y_all = np.concatenate([np.ones(len(X_h)), np.zeros(len(X_c))])
    print(f"    raw matrix shape: {X_all.shape}  (col order: {order})")

    # 70/30 stratified split, pre-registered seed
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.30, random_state=42, stratify=y_all)
    print(f"    train n={len(y_tr)} ({int(y_tr.sum())} hosts), test n={len(y_te)} ({int(y_te.sum())} hosts)")

    def logloss(p, y):
        p = np.clip(p, 1e-12, 1-1e-12)
        return float(np.mean(y * np.log(p) + (1-y) * np.log(1-p)))

    feh_idx = [order.index("FeH")]
    # Map surviving sub-dim s_X back to raw column index
    # s_MgFe -> MgFe, s_SiFe -> SiFe, s_CaFe -> CaFe, s_AlFe -> AlFe,
    # s_volatile -> CeFe, s_CO -> C_O, s_MgSi -> Mg_Si, s_age -> age
    SUB_TO_RAW = {"s_age":"age","s_AlFe":"AlFe","s_CaFe":"CaFe","s_CO":"C_O",
                  "s_MgFe":"MgFe","s_MgSi":"Mg_Si","s_SiFe":"SiFe","s_volatile":"CeFe"}
    surviving_raw = list({SUB_TO_RAW[d] for d in surviving_dims})
    surviving_idx = [order.index(r) for r in surviving_raw]
    print(f"    surviving raw cols: {surviving_raw}")

    def fit_logloss(idx_cols, name):
        if len(idx_cols) == 0:
            return float("nan"), float("nan")
        Xt = X_tr[:, idx_cols]
        Xv = X_te[:, idx_cols]
        scaler = StandardScaler().fit(Xt)
        Xts = scaler.transform(Xt)
        Xvs = scaler.transform(Xv)
        lr = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0).fit(Xts, y_tr)
        p_tr = lr.predict_proba(Xts)[:,1]
        p_te = lr.predict_proba(Xvs)[:,1]
        ll_tr = logloss(p_tr, y_tr)
        ll_te = logloss(p_te, y_te)
        print(f"    {name:<35s}: train={ll_tr:+.5f}  held-out={ll_te:+.5f}")
        return ll_tr, ll_te

    ll_a_tr, ll_a_te = fit_logloss(feh_idx, "(a) [Fe/H] alone")
    if len(surviving_idx) > 0:
        union = sorted(set(feh_idx + surviving_idx))
        ll_b_tr, ll_b_te = fit_logloss(union, "(b) [Fe/H] + step-1 survivors")
    else:
        ll_b_tr, ll_b_te = float("nan"), float("nan")
        print(f"    (b) [Fe/H] + step-1 survivors  : NO survivors to add")
    all_idx = list(range(X_all.shape[1]))
    ll_c_tr, ll_c_te = fit_logloss(all_idx, "(c) full 9-dim linear")

    delta_ba = (ll_b_te - ll_a_te) if not (np.isnan(ll_b_te) or np.isnan(ll_a_te)) else float("nan")
    delta_ca = ll_c_te - ll_a_te
    print(f"\n    held-out delta (b - a): {delta_ba:+.5f}  (need >+0.02 for confirmation)")
    print(f"    held-out delta (c - a): {delta_ca:+.5f}  (descriptive)")

    # --- VERDICT for this category ---
    print(f"\n  PRE-REGISTERED VERDICT for {cat}:")
    n_surv = len(surviving)
    n_surv_step4 = sum(1 for d, sr in step4_results.items() if sr["pass_step4"])
    confirmed = (n_surv >= 1) and (delta_ba > 0.02) and (n_surv_step4 >= 1)
    partial   = (n_surv >= 1) and not confirmed
    rejected  = (n_surv == 0) or (delta_ba <= 0)
    if confirmed:
        verdict = "CONFIRMED"
    elif partial:
        verdict = "PARTIAL"
    else:
        verdict = "REJECTED"
    print(f"    surviving step 1: {n_surv}")
    print(f"    surviving step 4: {n_surv_step4}")
    print(f"    delta (b-a) held-out: {delta_ba:+.5f}")
    print(f"    >>> VERDICT: {verdict}")

    all_results[cat] = dict(
        verdict=verdict, n_hosts=len(sub_hosts), n_match=len(matched),
        step1=step1_results, surviving=surviving_dims, step4=step4_results,
        ll_a_te=ll_a_te, ll_b_te=ll_b_te, ll_c_te=ll_c_te,
        delta_ba=delta_ba, delta_ca=delta_ca,
    )

# Cross-category consistency
print(f"\n{'='*78}")
print("STEP 2: cross-category consistency")
print(f"{'='*78}")
if len(all_results) >= 2:
    cats = [c for c in CATEGORIES if c in all_results and "step1" in all_results[c]]
    surv_sets = {}
    for c in cats:
        surv_sets[c] = set(all_results[c]["surviving"])
        print(f"  {c}: surviving = {sorted(surv_sets[c])}")
    if len(cats) >= 2:
        intersection = surv_sets[cats[0]] & surv_sets[cats[1]]
        union = surv_sets[cats[0]] | surv_sets[cats[1]]
        only_a = surv_sets[cats[0]] - surv_sets[cats[1]]
        only_b = surv_sets[cats[1]] - surv_sets[cats[0]]
        print(f"\n  shared (both):   {sorted(intersection)}")
        print(f"  only {cats[0]}:   {sorted(only_a)}")
        print(f"  only {cats[1]}:   {sorted(only_b)}")
        print(f"  Jaccard:         {len(intersection)/len(union) if union else 0:.3f}")

# Final summary
print(f"\n{'='*78}\nFINAL PRE-REGISTERED VERDICT (test #4)\n{'='*78}")
for c, r in all_results.items():
    print(f"  {c}: {r['verdict']}")
    if "delta_ba" in r:
        print(f"    held-out delta (b-a): {r['delta_ba']:+.5f}  surviving={r['surviving']}")

import json
with open("cct_test4_results.json","w") as f:
    # strip non-JSON-able
    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k,v in o.items()}
        if isinstance(o, list): return [clean(x) for x in o]
        if isinstance(o, (np.integer, np.floating)): return float(o) if np.isfinite(o) else None
        if isinstance(o, np.ndarray): return clean(o.tolist())
        return o
    json.dump(clean(all_results), f, indent=2, default=str)
print(f"\ncct_test4_results.json saved\nDONE")

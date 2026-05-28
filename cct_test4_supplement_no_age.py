"""Supplementary check: re-run test #4 with s_age EXCLUDED.

The pre-registered test found s_age as the sole surviving dimension,
but s_age is a data-availability artifact (hosts have NEA stellar ages,
APOGEE field has no age column, both default differently). The +0.3
constant shift across every [Fe/H] bin is the smoking gun.

This supplement tests whether ANY of the 7 actual chemistry sub-dimensions
(Mg/Fe, Si/Fe, Ca/Fe, Al/Fe, Ce/Fe, C/O, Mg/Si) carry real predictive
content out-of-sample, with age removed entirely from the analysis.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 7 chemistry sub-dimensions (s_age removed)
SUB_DIMS = ["s_AlFe", "s_CaFe", "s_CO", "s_MgFe", "s_MgSi", "s_SiFe", "s_volatile"]
SUB_LABELS = ["[Al/Fe]", "[Ca/Fe]", "C/O", "[Mg/Fe]", "Mg/Si", "[Si/Fe]", "volatile"]
N_DIMS = len(SUB_DIMS)
ALPHA_BONF = 0.05 / N_DIMS

# Raw columns excluding age
RAW_ORDER = ["FeH","MgFe","SiFe","CaFe","AlFe","CeFe","C_O","Mg_Si"]
HOST_COLS = {"FeH":"ap_FE_H","MgFe":"ap_MG_FE","SiFe":"ap_SI_FE","CaFe":"ap_CA_FE",
             "AlFe":"ap_AL_FE","CeFe":"ap_CE_FE","C_O":"C_O","Mg_Si":"Mg_Si"}
FIELD_COLS = {"FeH":"FE_H","MgFe":"MG_FE","SiFe":"SI_FE","CaFe":"CA_FE",
              "AlFe":"AL_FE","CeFe":"CE_FE","C_O":"C_O","Mg_Si":"Mg_Si"}

print("="*78)
print("SUPPLEMENT: Test #4 with s_age REMOVED (8 -> 7 chemistry-only dims)")
print(f"Bonferroni alpha = 0.05/{N_DIMS} = {ALPHA_BONF:.5f}")
print("="*78)

hosts = pd.read_csv("cct_test_hosts_scored.csv")
field = pd.read_csv("cct_test_field_scored.csv")
fgk_hosts = hosts[(hosts.ap_TEFF.between(4500, 7000)) & (hosts.ap_LOGG > 3.8)].copy()
fgk_field = field[(field.TEFF.between(4500, 7000)) & (field.LOGG > 3.8) &
                  (field.SNR > 70) & field.FE_H.between(-2, 1)].copy().reset_index(drop=True)

def build_strict_match(hosts_df, k=10):
    sub = hosts_df[["ap_TEFF","ap_LOGG","ap_FE_H"]].dropna()
    h_coords = np.column_stack([sub.ap_TEFF/100, sub.ap_LOGG/0.1, sub.ap_FE_H/0.05])
    fsub = fgk_field[["TEFF","LOGG","FE_H"]].dropna()
    f_coords = np.column_stack([fsub.TEFF/100, fsub.LOGG/0.1, fsub.FE_H/0.05])
    fld = fgk_field.loc[fsub.index].reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=k).fit(f_coords)
    _, idx = nn.kneighbors(h_coords)
    return fld.iloc[np.unique(idx.flatten())].copy().reset_index(drop=True)

def raw_matrix(df, col_map):
    return np.column_stack([df[col_map[k]].fillna(0).values
                            if (col_map.get(k) in df.columns) else np.zeros(len(df))
                            for k in RAW_ORDER])

def logloss(p, y):
    p = np.clip(p, 1e-12, 1-1e-12)
    return float(np.mean(y*np.log(p) + (1-y)*np.log(1-p)))

for cat in ["non_HZ_rocky", "sub_Neptune"]:
    sub_h = fgk_hosts[fgk_hosts.host_category == cat].copy().reset_index(drop=True)
    if len(sub_h) < 30: continue
    matched = build_strict_match(sub_h)
    print(f"\n--- {cat} (N_host={len(sub_h)}, N_match={len(matched)}) ---")

    # Step 1 (chemistry only)
    print(f"\nStep 1: per-dimension MW (chemistry only)")
    surviving = []
    for d, lbl in zip(SUB_DIMS, SUB_LABELS):
        h = sub_h[d].dropna().values
        c = matched[d].dropna().values
        if len(h) < 10 or len(c) < 50: continue
        shift = np.median(h) - np.median(c)
        U, p = stats.mannwhitneyu(h, c, alternative="greater")
        passed = (p < ALPHA_BONF) and (shift > 0)
        flag = "PASS" if passed else "----"
        print(f"  {lbl:<10}: shift={shift:+.4f}  p={p:.3e}  {flag}")
        if passed: surviving.append(d)
    print(f"\n  Surviving chemistry dimensions: {surviving}")

    # Held-out CV (chemistry only)
    X_h = raw_matrix(sub_h, HOST_COLS)
    X_c = raw_matrix(matched, FIELD_COLS)
    X = np.vstack([X_h, X_c])
    y = np.concatenate([np.ones(len(X_h)), np.zeros(len(X_c))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    def fit(idx_cols, name):
        if not idx_cols:
            print(f"  {name:<35s}: empty -- skip"); return float("nan")
        Xt = X_tr[:, idx_cols]; Xv = X_te[:, idx_cols]
        sc = StandardScaler().fit(Xt)
        lr = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0).fit(sc.transform(Xt), y_tr)
        ll = logloss(lr.predict_proba(sc.transform(Xv))[:,1], y_te)
        print(f"  {name:<35s}: held-out log-loss = {ll:+.5f}")
        return ll

    print(f"\nStep 3 held-out CV (age column REMOVED):")
    feh_idx = [RAW_ORDER.index("FeH")]
    ll_a = fit(feh_idx, "(a) [Fe/H] alone")
    SUB_TO_RAW = {"s_AlFe":"AlFe","s_CaFe":"CaFe","s_CO":"C_O","s_MgFe":"MgFe",
                  "s_MgSi":"Mg_Si","s_SiFe":"SiFe","s_volatile":"CeFe"}
    surv_raw = list({SUB_TO_RAW[d] for d in surviving})
    surv_idx = [RAW_ORDER.index(r) for r in surv_raw]
    if surv_idx:
        ll_b = fit(sorted(set(feh_idx + surv_idx)), "(b) [Fe/H] + chem survivors")
    else:
        print(f"  (b) [Fe/H] + chem survivors        : no survivors -> skip")
        ll_b = float("nan")
    ll_c = fit(list(range(len(RAW_ORDER))), "(c) all 8 chemistry linear")
    db = ll_b - ll_a if not np.isnan(ll_b) else float("nan")
    dc = ll_c - ll_a
    print(f"\n  delta (b - a): {db:+.5f}  (need > +0.02 for confirmation)")
    print(f"  delta (c - a): {dc:+.5f}  (full chemistry linear vs [Fe/H])")

    if not surviving:
        print(f"\n  >>> {cat} chemistry-only verdict: REJECTED (no step-1 dims)")
    elif np.isnan(db) or db <= 0.02:
        print(f"\n  >>> {cat} chemistry-only verdict: PARTIAL or REJECTED (delta <= 0.02)")
    else:
        print(f"\n  >>> {cat} chemistry-only verdict: CONFIRMED")

print("\nDONE")

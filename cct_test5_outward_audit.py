"""Pre-registered test #5: outward falsification audit of published claims.

Sealed pre-registration: PRE_REGISTRATION_5.md @ commit b196357
Frozen scorer: not used here -- testing literature claims directly

Same APOGEE x NEA dataset, same matching, same threshold:
  - FGK restriction (Teff 4500-7000, logg > 3.8)
  - k=10 NN strict matching on (Teff/100, logg/0.1, [Fe/H]/0.05)
  - Match-quality gate: KS p > 0.95 on all three axes
  - One-sided MW, Bonferroni alpha = 0.05/4 = 0.0125
  - Held-out CV: 70/30 split, seed 42, LogReg class_weight=balanced
  - Threshold for held-out improvement: > 0.02 vs [Fe/H] alone
  - Within-bin sanity: >= 50% of populated 0.1-dex [Fe/H] bins in correct direction

Four claims:
  A. Adibekyan Mg/Si in hot Jupiter hosts (host < control)
  B. Adibekyan [Mg/Fe] in small planet hosts at [Fe/H] < -0.2 (host > control)
  C. Brewer & Fischer C/O in any planet host (host < control)
  D. Suarez-Andres C/O > 0.8 in rocky planet hosts (host frac < field frac)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ALPHA_BONF = 0.05 / 4
HELD_OUT_THRESHOLD = 0.02

print("="*78)
print("PRE-REGISTERED TEST #5: outward falsification audit")
print(f"Bonferroni alpha = {ALPHA_BONF:.4f}, held-out threshold = +{HELD_OUT_THRESHOLD}")
print("="*78)

hosts = pd.read_csv("cct_test_hosts.csv")
field = pd.read_csv("cct_test_field.csv")
fgk_hosts = hosts[(hosts.ap_TEFF.between(4500, 7000)) & (hosts.ap_LOGG > 3.8)].copy()
fgk_field = field[(field.TEFF.between(4500, 7000)) & (field.LOGG > 3.8) &
                  (field.SNR > 70) & field.FE_H.between(-2, 1)].copy().reset_index(drop=True)
print(f"FGK hosts: {len(fgk_hosts)}, FGK field pool: {len(fgk_field)}")

# Compute Mg/Si and C/O on field (already on hosts) -- the strict matching needs them
# (hosts have C_O and Mg_Si already from cct_test_hosts.csv -- field has raw [X/Fe])
SOLAR_CO = 0.549
if "C_O" not in field.columns:
    field["C_O"] = np.where(field.C_FE.notna() & field.O_FE.notna(),
                            (10.0 ** (field.C_FE - field.O_FE)) * SOLAR_CO, np.nan)
if "Mg_Si" not in field.columns:
    field["Mg_Si"] = np.where(field.MG_FE.notna() & field.SI_FE.notna(),
                              10.0 ** (field.MG_FE - field.SI_FE), np.nan)

# Refresh fgk_field with these
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

def match_quality(host_df, ctrl_df):
    out = {}
    for col_h, col_f, lbl in [("ap_TEFF","TEFF","Teff"),
                              ("ap_LOGG","LOGG","logg"),
                              ("ap_FE_H","FE_H","[Fe/H]")]:
        D, p = stats.ks_2samp(host_df[col_h].dropna(), ctrl_df[col_f].dropna())
        out[lbl] = (D, p)
    return out

def logloss(p, y):
    p = np.clip(p, 1e-12, 1-1e-12)
    return float(np.mean(y*np.log(p) + (1-y)*np.log(1-p)))

def held_out_cv(host_df, ctrl_df, host_col, ctrl_col, extra_cols=None):
    """Compare held-out log-loss: (a) [Fe/H] only vs (b) [Fe/H] + claim element."""
    # Build feature matrix: just [Fe/H] and the claim element
    h_feh = host_df["ap_FE_H"].fillna(0).values
    c_feh = ctrl_df["FE_H"].fillna(0).values
    h_el  = host_df[host_col].fillna(0).values
    c_el  = ctrl_df[ctrl_col].fillna(0).values
    X = np.column_stack([np.r_[h_feh, c_feh], np.r_[h_el, c_el]])
    y = np.concatenate([np.ones(len(host_df)), np.zeros(len(ctrl_df))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    def fit(idx):
        Xt = X_tr[:, idx]; Xv = X_te[:, idx]
        sc = StandardScaler().fit(Xt)
        lr = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0).fit(sc.transform(Xt), y_tr)
        return logloss(lr.predict_proba(sc.transform(Xv))[:,1], y_te)
    ll_a = fit([0])      # [Fe/H] alone
    ll_b = fit([0, 1])   # [Fe/H] + claim element
    return ll_a, ll_b

def within_bin_sanity(host_df, ctrl_df, host_col, ctrl_col, predicted_direction):
    """Check direction holds in >=50% of populated 0.1-dex [Fe/H] bins."""
    bins = np.arange(-1.0, 0.7, 0.1)
    consistent = 0; total = 0; details = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        h_bin = host_df[host_df.ap_FE_H.between(lo, hi)]
        c_bin = ctrl_df[ctrl_df.FE_H.between(lo, hi)]
        h_vals = h_bin[host_col].dropna()
        c_vals = c_bin[ctrl_col].dropna()
        if len(h_vals) < 3 or len(c_vals) < 10: continue
        shift = np.median(h_vals) - np.median(c_vals)
        total += 1
        ok = (predicted_direction == "less" and shift < 0) or \
             (predicted_direction == "greater" and shift > 0)
        if ok: consistent += 1
        details.append((lo, len(h_vals), len(c_vals), shift, ok))
    frac = consistent / total if total > 0 else 0
    return frac, consistent, total, details

def report_claim(label, host_df, ctrl_df, host_col, ctrl_col, direction):
    """Run all pre-registered tests for a single claim and report."""
    print(f"\n--- {label} ---")
    print(f"  N_host = {len(host_df)}, N_ctrl = {len(ctrl_df)}")

    # match quality
    mq = match_quality(host_df, ctrl_df)
    print(f"  KS Teff:   D={mq['Teff'][0]:.4f}  p={mq['Teff'][1]:.3e}")
    print(f"  KS logg:   D={mq['logg'][0]:.4f}  p={mq['logg'][1]:.3e}")
    print(f"  KS [Fe/H]: D={mq['[Fe/H]'][0]:.4f}  p={mq['[Fe/H]'][1]:.3e}")
    gate_pass = all(mq[k][1] > 0.95 for k in mq)
    if not gate_pass:
        print(f"  MATCH-QUALITY GATE FAILED -- claim REJECTED as matching failure")
        return dict(label=label, verdict="REJECTED_MATCH_FAIL")

    # MW
    h_vals = host_df[host_col].dropna().values
    c_vals = ctrl_df[ctrl_col].dropna().values
    if len(h_vals) < 10:
        print(f"  insufficient host data (n={len(h_vals)}) -- UNDERPOWERED")
        return dict(label=label, verdict="UNDERPOWERED")
    med_h = np.median(h_vals); med_c = np.median(c_vals)
    shift = med_h - med_c
    iqr_c = np.percentile(c_vals, 75) - np.percentile(c_vals, 25)
    effect = shift / (iqr_c / 1.349) if iqr_c > 0 else float("nan")
    alt = "less" if direction == "less" else "greater"
    U, p_mw = stats.mannwhitneyu(h_vals, c_vals, alternative=alt)
    print(f"  shift (host - ctrl): {shift:+.4f}   effect: {effect:+.2f}")
    print(f"  MW one-sided ({alt}): p = {p_mw:.3e}  ({'PASS' if p_mw < ALPHA_BONF else 'FAIL'} Bonferroni)")

    # held-out CV
    ll_a, ll_b = held_out_cv(host_df, ctrl_df, host_col, ctrl_col)
    delta = ll_b - ll_a
    print(f"  held-out log-loss: [Fe/H] alone = {ll_a:+.5f}, +{host_col} = {ll_b:+.5f}")
    print(f"  delta = {delta:+.5f}  ({'PASS' if delta > HELD_OUT_THRESHOLD else 'FAIL'} threshold {HELD_OUT_THRESHOLD})")

    # within-bin sanity
    frac, k, n, details = within_bin_sanity(host_df, ctrl_df, host_col, ctrl_col, direction)
    print(f"  within-[Fe/H]-bin sanity: {k}/{n} bins direction-consistent ({frac:.2f})  ({'PASS' if frac >= 0.5 and n >= 2 else 'FAIL'})")
    for lo, nh, nc, s, ok in details:
        flag = "OK" if ok else "--"
        print(f"    bin [Fe/H]={lo:+.1f}: n_h={nh:>3d}  n_c={nc:>4d}  shift={s:+.4f}  {flag}")

    # verdict
    mw_pass = p_mw < ALPHA_BONF
    cv_pass = delta > HELD_OUT_THRESHOLD
    bin_pass = (frac >= 0.5) and (n >= 2)

    if mw_pass and cv_pass and bin_pass:
        verdict = "SURVIVES"
    elif mw_pass and not (cv_pass and bin_pass):
        verdict = "PARTIAL"
    elif shift * (-1 if direction == "less" else 1) <= 0:
        verdict = "REJECTED (direction)"
    else:
        verdict = "REJECTED"
    print(f"  >>> verdict: {verdict}")

    return dict(label=label, n_host=len(host_df), n_ctrl=len(ctrl_df),
                shift=shift, effect=effect, p_mw=p_mw,
                ll_a=ll_a, ll_b=ll_b, delta=delta,
                bin_frac=frac, bin_n=n,
                mw_pass=mw_pass, cv_pass=cv_pass, bin_pass=bin_pass,
                verdict=verdict)

all_results = {}

# CLAIM A: Adibekyan Mg/Si in hot Jupiter hosts (host < control)
print("\n" + "="*78)
print("CLAIM A: Adibekyan 2012 -- Mg/Si lower in hot Jupiter hosts")
print("="*78)
hj = fgk_hosts[fgk_hosts.host_category == "hot_Jupiter"].copy().reset_index(drop=True)
match_hj = build_strict_match(hj)
all_results["A_Adibekyan_MgSi_HJ"] = report_claim(
    "Mg/Si in hot Jupiter hosts (predicted: host < control)",
    hj, match_hj, "Mg_Si", "Mg_Si", "less")

# CLAIM B: Adibekyan [Mg/Fe] in small-planet hosts at [Fe/H] < -0.2 (host > control)
print("\n" + "="*78)
print("CLAIM B: Adibekyan 2012 -- [Mg/Fe] higher in small planet hosts at [Fe/H] < -0.2")
print("="*78)
small = fgk_hosts[(fgk_hosts.host_pl_rade < 4.0) & (fgk_hosts.ap_FE_H < -0.2)].copy().reset_index(drop=True)
print(f"  small planet (R<4 R_E) hosts at [Fe/H]<-0.2: {len(small)}")
# match control restricted to same [Fe/H] regime
fgk_field_lowmet = fgk_field[fgk_field.FE_H < -0.2].copy().reset_index(drop=True)
print(f"  field pool at [Fe/H]<-0.2: {len(fgk_field_lowmet)}")
def build_strict_match_lowmet(hosts_df, k=10):
    sub = hosts_df[["ap_TEFF","ap_LOGG","ap_FE_H"]].dropna()
    h_coords = np.column_stack([sub.ap_TEFF/100, sub.ap_LOGG/0.1, sub.ap_FE_H/0.05])
    fsub = fgk_field_lowmet[["TEFF","LOGG","FE_H"]].dropna()
    f_coords = np.column_stack([fsub.TEFF/100, fsub.LOGG/0.1, fsub.FE_H/0.05])
    fld = fgk_field_lowmet.loc[fsub.index].reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=k).fit(f_coords)
    _, idx = nn.kneighbors(h_coords)
    return fld.iloc[np.unique(idx.flatten())].copy().reset_index(drop=True)
match_small = build_strict_match_lowmet(small)
all_results["B_Adibekyan_MgFe_lowmet"] = report_claim(
    "[Mg/Fe] in small planet hosts at [Fe/H]<-0.2 (predicted: host > control)",
    small, match_small, "ap_MG_FE", "MG_FE", "greater")

# CLAIM C: Brewer & Fischer C/O in any planet host (host < control)
print("\n" + "="*78)
print("CLAIM C: Brewer & Fischer 2018 -- C/O lower in any planet host")
print("="*78)
all_hosts = fgk_hosts.copy().reset_index(drop=True)
match_all = build_strict_match(all_hosts)
all_results["C_BrewerFischer_CO_any"] = report_claim(
    "C/O in any planet host (predicted: host < control)",
    all_hosts, match_all, "C_O", "C_O", "less")

# CLAIM D: Suarez-Andres C/O > 0.8 in rocky hosts (host frac < field frac)
print("\n" + "="*78)
print("CLAIM D: Suarez-Andres 2018 -- C/O > 0.8 fraction in rocky planet hosts")
print("="*78)
rocky_hosts = fgk_hosts[fgk_hosts.host_pl_rade < 1.6].copy()
co_h = rocky_hosts.C_O.dropna()
co_f = fgk_field.C_O.dropna()
n_h_high = (co_h > 0.8).sum()
n_f_high = (co_f > 0.8).sum()
print(f"  rocky planet hosts (R<1.6 R_E, FGK): N = {len(co_h)}, C/O > 0.8: {n_h_high}")
print(f"  FGK field:                            N = {len(co_f)}, C/O > 0.8: {n_f_high}")
if len(co_h) < 30:
    print(f"  N_host < 30 -- UNDERPOWERED")
    verd = "UNDERPOWERED"
elif n_f_high < 50:
    print(f"  field N at C/O > 0.8 = {n_f_high} (< 50) -- UNDERPOWERED")
    verd = "UNDERPOWERED"
else:
    table = np.array([[n_h_high, len(co_h)-n_h_high],
                      [n_f_high, len(co_f)-n_f_high]])
    odds, p = stats.fisher_exact(table, alternative="less")
    frac_h = n_h_high/len(co_h); frac_f = n_f_high/len(co_f)
    print(f"  host fraction = {frac_h:.4f}, field fraction = {frac_f:.4f}")
    print(f"  ratio: host/field = {frac_h/frac_f if frac_f>0 else float('inf'):.3f}")
    print(f"  Fisher exact (one-sided host < field): p = {p:.3e}")
    if p < ALPHA_BONF and frac_h < 0.5 * frac_f:
        verd = "SURVIVES"
    elif p < 0.05:
        verd = "PARTIAL"
    else:
        verd = "REJECTED"
    print(f"  >>> verdict: {verd}")
    all_results["D_SuarezAndres_CO_high"] = dict(
        label="C/O > 0.8 fraction in rocky planet hosts",
        n_h=len(co_h), n_f=len(co_f), n_h_high=n_h_high, n_f_high=n_f_high,
        frac_h=frac_h, frac_f=frac_f, fisher_p=p, verdict=verd)

# Final summary
print("\n" + "="*78)
print("PRE-REGISTERED FINAL AUDIT VERDICT (test #5)")
print("="*78)
print(f"\n  Bonferroni alpha = {ALPHA_BONF}, held-out threshold = {HELD_OUT_THRESHOLD}")
print()
for k, r in all_results.items():
    print(f"  {k}: {r.get('verdict', '?')}")
    if "delta" in r and not isinstance(r.get("delta"), str):
        print(f"     shift={r['shift']:+.4f}  p_mw={r['p_mw']:.3e}  held-out delta={r['delta']:+.5f}  bin_frac={r['bin_frac']:.2f}")

# Save results
import json
with open("cct_test5_results.json","w") as f:
    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k,v in o.items()}
        if isinstance(o, list): return [clean(x) for x in o]
        if isinstance(o, (np.integer, np.floating)): return float(o) if np.isfinite(o) else None
        if isinstance(o, np.ndarray): return clean(o.tolist())
        return o
    json.dump(clean(all_results), f, indent=2, default=str)
print(f"\ncct_test5_results.json saved")
print("DONE")

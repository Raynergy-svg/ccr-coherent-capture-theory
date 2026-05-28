"""
T18 audit: is 'alpha 2x tighter than s-process in 98% of clusters' a real
intrinsic-ISM signal, or driven by GALAH's element-specific measurement errors?

Method: re-compute per-cluster scatter as INTRINSIC scatter,
    sigma_int = sqrt(max(0, std_raw^2 - mean(error)^2))
where mean(error) is the per-cluster mean of the per-star quoted abundance
error. Re-run the Wilcoxon signed-rank test (alpha_intrinsic < s-process_
intrinsic) and compare to the raw-std version.

If the raw-std 98% / p~10^-98 survives the error-subtraction, T18 is real.
If the intrinsic version collapses (drops below ~70%, p > 0.05), the
'nucleosynthetic hierarchy' was largely a measurement-precision artifact.
"""
import numpy as np, pandas as pd
from astropy.table import Table
from scipy import stats
from scipy.spatial import cKDTree

# load and cross-match (same as t18)
t9 = pd.read_csv("t9_matched_stars.csv")
g = Table.read("galah_dr4_allstar_240705.fits", memmap=True).to_pandas()
cols = ["ra","dec","snr_px_ccd3","flag_sp",
        "mg_fe","e_mg_fe","flag_mg_fe","si_fe","e_si_fe","flag_si_fe",
        "ba_fe","e_ba_fe","flag_ba_fe",
        "fe_h","e_fe_h","flag_fe_h"]
g = g[cols].copy()
gc = np.deg2rad(np.column_stack([g.ra.values, g.dec.values]))
gx = np.column_stack([np.cos(gc[:,1])*np.cos(gc[:,0]),
                       np.cos(gc[:,1])*np.sin(gc[:,0]), np.sin(gc[:,1])])
tree = cKDTree(gx)
tc = np.deg2rad(np.column_stack([t9.ra.values, t9.dec.values]))
tx = np.column_stack([np.cos(tc[:,1])*np.cos(tc[:,0]),
                       np.cos(tc[:,1])*np.sin(tc[:,0]), np.sin(tc[:,1])])
d, idx = tree.query(tx, k=1)
tol = 2*np.sin(np.deg2rad(0.5/3600)/2)
m = d < tol
df = t9[m].reset_index(drop=True).copy()
for c in ["mg_fe","e_mg_fe","flag_mg_fe","si_fe","e_si_fe","flag_si_fe",
          "ba_fe","e_ba_fe","flag_ba_fe",
          "snr_px_ccd3","flag_sp"]:
    df[c] = g.iloc[idx[m]][c].values
df = df[(df.snr_px_ccd3>30)&(df.flag_sp==0)]
print(f"GALAH-matched cluster members (snr>30, flag_sp=0): {len(df)}")

# typical errors per element across the matched sample
print("\nTYPICAL GALAH PER-STAR ABUNDANCE ERRORS (median, in this cluster-matched sample):")
for col,ec in [("mg_fe","e_mg_fe"),("si_fe","e_si_fe"),
               ("ba_fe","e_ba_fe")]:
    v = df[ec].dropna()
    print(f"  {col:6s}  median error = {v.median():.4f} dex  (p10 {v.quantile(0.1):.4f}, p90 {v.quantile(0.9):.4f}, N={len(v)})")

# per-cluster scatter both raw and intrinsic
MIN_N = 5
def cluster_scatter(d, col, ecol, flag_col=None):
    """For each cluster, return raw std and error-subtracted intrinsic scatter."""
    out = {}
    for cname, grp in d.groupby("cluster_name"):
        vals = grp[col].dropna()
        if flag_col and flag_col in grp.columns:
            flags = grp.loc[vals.index, flag_col]
            vals = vals[flags == 0]
        vals = vals[(vals>-5)&(vals<5)]
        if len(vals) < MIN_N: continue
        errs = grp.loc[vals.index, ecol].fillna(0).values
        # mean variance of measurement errors
        e2_mean = np.mean(errs**2)
        s_raw = vals.std(ddof=1)
        s_int = np.sqrt(max(0.0, s_raw**2 - e2_mean))
        out[cname] = dict(n=len(vals), std_raw=s_raw, std_int=s_int,
                          err_mean=np.sqrt(e2_mean))
    return out

mg = cluster_scatter(df, "mg_fe","e_mg_fe","flag_mg_fe")
si = cluster_scatter(df, "si_fe","e_si_fe","flag_si_fe")
ba = cluster_scatter(df, "ba_fe","e_ba_fe","flag_ba_fe")
ce = {}
print(f"\nclusters with each element (>= {MIN_N} members): Mg {len(mg)}  Si {len(si)}  Ba {len(ba)}  (Ce not in rebuilt FITS)")

# per-cluster alpha and s-process scatter (mean of two)
clusters = sorted(set(mg.keys())|set(si.keys())|set(ba.keys())|set(ce.keys()))
rows=[]
for c in clusters:
    a_raw=[]; a_int=[]; s_raw=[]; s_int=[]
    if c in mg: a_raw.append(mg[c]["std_raw"]); a_int.append(mg[c]["std_int"])
    if c in si: a_raw.append(si[c]["std_raw"]); a_int.append(si[c]["std_int"])
    if c in ba: s_raw.append(ba[c]["std_raw"]); s_int.append(ba[c]["std_int"])
    if c in ce: s_raw.append(ce[c]["std_raw"]); s_int.append(ce[c]["std_int"])
    if a_raw and s_raw:
        rows.append(dict(cluster=c, alpha_raw=np.mean(a_raw), alpha_int=np.mean(a_int),
                         sproc_raw=np.mean(s_raw), sproc_int=np.mean(s_int)))
C = pd.DataFrame(rows)
print(f"\nclusters with both alpha and s-process: {len(C)}")

# RAW comparison (reproduce T18)
print("\n" + "="*78)
print("RAW STD comparison (reproduce T18)")
print("="*78)
w,p = stats.wilcoxon(C.alpha_raw, C.sproc_raw, alternative="less")
frac = (C.alpha_raw < C.sproc_raw).mean()
print(f"  median alpha_raw : {C.alpha_raw.median():.4f}")
print(f"  median sproc_raw : {C.sproc_raw.median():.4f}")
print(f"  median ratio (alpha/s-proc): {(C.alpha_raw/C.sproc_raw).median():.3f}")
print(f"  fraction alpha < s-process: {frac:.1%}")
print(f"  Wilcoxon (alpha<s-proc): W={w:.0f}, p={p:.4e}")

# INTRINSIC (error-subtracted) comparison
print("\n" + "="*78)
print("INTRINSIC SCATTER (error-subtracted) -- the honest test")
print("="*78)
w_i,p_i = stats.wilcoxon(C.alpha_int, C.sproc_int, alternative="less")
frac_i = (C.alpha_int < C.sproc_int).mean()
print(f"  median alpha_int : {C.alpha_int.median():.4f}")
print(f"  median sproc_int : {C.sproc_int.median():.4f}")
print(f"  median ratio (alpha/s-proc): {(C.alpha_int/(C.sproc_int+1e-6)).median():.3f}")
print(f"  fraction alpha < s-process: {frac_i:.1%}")
print(f"  Wilcoxon (alpha<s-proc): W={w_i:.0f}, p={p_i:.4e}")

# what fraction of clusters have intrinsic scatter = 0 (raw < error)?
a_zero = (C.alpha_int < 1e-4).sum(); s_zero = (C.sproc_int < 1e-4).sum()
print(f"\n  clusters with alpha_int ~ 0 (raw std absorbed by error): {a_zero}/{len(C)} ({a_zero/len(C):.1%})")
print(f"  clusters with sproc_int ~ 0 (raw std absorbed by error): {s_zero}/{len(C)} ({s_zero/len(C):.1%})")

# look at the error contribution explicitly
print("\n" + "="*78)
print("How much of the raw std is ERROR vs INTRINSIC scatter?")
print("="*78)
err_a = np.sqrt(np.maximum(C.alpha_raw**2 - C.alpha_int**2, 0))
err_s = np.sqrt(np.maximum(C.sproc_raw**2 - C.sproc_int**2, 0))
print(f"  Alpha     : median raw {C.alpha_raw.median():.4f} = sqrt(err^2 {err_a.median():.4f}^2 + int^2 {C.alpha_int.median():.4f}^2)")
print(f"  S-process : median raw {C.sproc_raw.median():.4f} = sqrt(err^2 {err_s.median():.4f}^2 + int^2 {C.sproc_int.median():.4f}^2)")
print(f"  Alpha error/raw ratio:   median {(err_a/C.alpha_raw).median():.2f}  (1.0 = all error)")
print(f"  S-proc error/raw ratio:  median {(err_s/C.sproc_raw).median():.2f}")

# verdict
print("\n" + "="*78)
print("VERDICT")
print("="*78)
print(f"RAW       : {frac:.1%} of clusters have alpha < s-proc; Wilcoxon p = {p:.2e}")
print(f"INTRINSIC : {frac_i:.1%} of clusters have alpha < s-proc; Wilcoxon p = {p_i:.2e}")
if frac_i >= 0.9 and p_i < 1e-20:
    print("=> The 'alpha tighter' finding SURVIVES error-subtraction. Real ISM signal.")
elif frac_i >= 0.7:
    print("=> The finding SURVIVES but is WEAKER under error-subtraction.")
    print("   The raw fraction overstated the intrinsic case; real but smaller.")
elif frac_i >= 0.55:
    print("=> The finding is MARGINAL after error-subtraction. Most of the raw")
    print("   signal was measurement precision; some intrinsic signal remains.")
else:
    print("=> The finding does NOT survive error-subtraction. The raw 'alpha")
    print("   tighter than s-process' was driven by GALAH measuring alpha")
    print("   elements more precisely than s-process, not by an intrinsic ISM")
    print("   coherence-channel difference.")

C.to_csv("t18_audit_scatter.csv", index=False)
print("\nDONE  (saved t18_audit_scatter.csv)")

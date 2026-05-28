"""
T16d done properly: placebo-controlled test for genuine Ba/Fe alignment beyond
the chemodynamical-disk background.

Original T16d: matched-to-X stars are closer in Ba/Fe to cluster X than
random field stars are. CRITIQUE: 4D-chemistry-matched stars are 23% closer
in Ba/Fe to ANY chemistry-correlated value by background; the comparison
should be against a chemistry-matched control, not whole-field.

This audit:
For each cluster X, compute median |Ba_match - B_X| where matched are stars
within fixed-tol of X's 4D template. Compare to PERMUTED |Ba_match - B_Y|
where Y is a different randomly-picked cluster (so we're asking the same
matched stars to align with a different cluster's Ba/Fe). If matched-to-X
stars are systematically closer to X's Ba/Fe than to other clusters' Ba/Fe,
there is a genuine signal beyond background.
"""
import numpy as np, pandas as pd
from astropy.table import Table
from scipy import stats
from scipy.spatial import cKDTree

T9 = pd.read_csv("t9_matched_stars.csv")
g = Table.read("galah_dr4_allstar_240705.fits", memmap=True).to_pandas()
need = ["ra","dec","snr_px_ccd3","flag_sp",
        "c_fe","flag_c_fe","o_fe","flag_o_fe",
        "mg_fe","flag_mg_fe","si_fe","flag_si_fe",
        "fe_h","flag_fe_h","ba_fe","flag_ba_fe","parallax"]
g = g[need].copy()
SOLAR_CO = 0.549
g["C_O"] = 10**(g.c_fe - g.o_fe) * SOLAR_CO
# quality cuts: snr>30, flag_sp=0, valid + unflagged in all matching dims, valid Ba/Fe (flag=0)
g = g[(g.snr_px_ccd3>30)&(g.flag_sp==0)]
for c, fc in [("C_O",None),("mg_fe","flag_mg_fe"),("si_fe","flag_si_fe"),
              ("fe_h","flag_fe_h"),("ba_fe","flag_ba_fe")]:
    g = g[g[c].notna()]
    if fc and fc in g: g = g[g[fc]==0]
g = g[(g.C_O>0.05)&(g.C_O<2.0)].reset_index(drop=True)
print(f"Field star pool (8D-clean for chem match + Ba/Fe valid): {len(g)}")

# cluster templates (centroid + Ba_centroid from t9 GALAH-matched members)
# Need to merge t9_matched_stars with GALAH ba_fe via positional match
print("\nBuilding cluster templates (4D centroid + Ba/Fe centroid from members)...")
# cross-match t9 -> GALAH
g_xyz = np.deg2rad(np.column_stack([g.ra.values, g.dec.values]))
g_xyz = np.column_stack([np.cos(g_xyz[:,1])*np.cos(g_xyz[:,0]),
                          np.cos(g_xyz[:,1])*np.sin(g_xyz[:,0]),
                          np.sin(g_xyz[:,1])])
tree = cKDTree(g_xyz)
t9c = np.deg2rad(np.column_stack([T9.ra.values, T9.dec.values]))
t9_xyz = np.column_stack([np.cos(t9c[:,1])*np.cos(t9c[:,0]),
                           np.cos(t9c[:,1])*np.sin(t9c[:,0]),
                           np.sin(t9c[:,1])])
tol = 2*np.sin(np.deg2rad(0.5/3600)/2)
d, idx = tree.query(t9_xyz, k=1)
m = d < tol
tmem = T9[m].copy()
tmem["ba_fe"] = g.iloc[idx[m]].ba_fe.values
print(f"t9 cluster members successfully matched to GALAH (with ba_fe): {len(tmem)}")

# cluster templates: require >=5 members AND C_O_std<0.10 (T16d selection)
templates = tmem.groupby("cluster_name").agg(
    n=("C_O","count"),
    C_O=("C_O","mean"), mg_fe=("mg_fe","mean"), si_fe=("si_fe","mean"),
    fe_h=("fe_h","mean"), ba_fe=("ba_fe","mean"),
    C_O_std=("C_O","std")
).reset_index()
templates = templates[(templates.n>=5)&(templates.C_O_std<0.10)].reset_index(drop=True)
print(f"templates (N>=5, C_O_std<0.10): {len(templates)}")

# T16b-style fixed tolerances
TOL = np.array([0.08, 0.05, 0.05, 0.10])  # C/O, Mg/Fe, Si/Fe, Fe/H
DIMS = ["C_O","mg_fe","si_fe","fe_h"]
field = g[DIMS+["ba_fe"]].values

results = []
for _, tmpl in templates.iterrows():
    cent = np.array([tmpl[d] for d in DIMS])
    delta = np.abs(field[:,:4] - cent[None,:])
    chem_match = (delta < TOL).all(axis=1)
    n_match = chem_match.sum()
    if n_match < 10: continue
    Ba_match = field[chem_match, 4]
    # |Ba - cluster Ba|
    d_true = np.abs(Ba_match - tmpl.ba_fe)
    med_true = np.median(d_true)
    results.append(dict(cluster=tmpl.cluster_name, n_match=n_match,
                        B_X=tmpl.ba_fe, med_d_true=med_true,
                        match_idx=np.where(chem_match)[0]))
R = pd.DataFrame(results)
print(f"\nclusters with >=10 matches: {len(R)}")

# placebo null: for each cluster X, randomly pick another cluster Y, and compute
# median |Ba_match_X - B_Y|. Do this 500 times per cluster, take the average.
print("\nPlacebo null: median |Ba_match_X - B_Y| where Y is randomly chosen from")
print("OTHER templates (no cluster matches itself)...")
rng = np.random.default_rng(42)
N_PERM = 200
Ba_centroids = templates.ba_fe.values
med_null = np.zeros(len(R))
for i, r in R.iterrows():
    match_Ba = field[r.match_idx, 4]
    null_meds = []
    for _ in range(N_PERM):
        # pick random template Y different from X
        y = rng.integers(0, len(templates))
        # ensure not the same cluster
        while templates.iloc[y].cluster_name == r.cluster:
            y = rng.integers(0, len(templates))
        B_Y = Ba_centroids[y]
        null_meds.append(np.median(np.abs(match_Ba - B_Y)))
    med_null[i] = np.mean(null_meds)
R["med_d_null"] = med_null
R["ratio"] = R.med_d_true / R.med_d_null
print(f"\nResults (Wilcoxon true vs null):")
print(f"  median |Ba_match - B_X|   (true cluster):     {R.med_d_true.median():.4f}")
print(f"  median |Ba_match - B_Y|   (placebo cluster):  {R.med_d_null.median():.4f}")
print(f"  median ratio true/null:                       {R.ratio.median():.4f}")
print(f"  fraction of clusters with true < null:        {(R.med_d_true < R.med_d_null).mean():.1%}")
W, p = stats.wilcoxon(R.med_d_true, R.med_d_null, alternative="less")
print(f"  Wilcoxon (true < null): W={W:.0f}, p={p:.4e}")
print(f"  N clusters: {len(R)}")
if R.ratio.median() < 0.95 and p < 0.01:
    print("\n  => SURVIVES: matched stars are systematically closer to TRUE cluster Ba/Fe")
    print("     than to random other clusters' Ba/Fe. Real signal beyond chemodynamical background.")
elif R.ratio.median() < 0.99:
    print("\n  => MARGINAL: small residual signal beyond background; not strong.")
else:
    print("\n  => NO RESIDUAL: matched stars are no closer to true cluster Ba/Fe than to random")
    print("     other clusters' Ba/Fe. T16d's signal is fully chemodynamical background.")

R.to_csv("t16d_proper_audit.csv", index=False)
print("DONE")

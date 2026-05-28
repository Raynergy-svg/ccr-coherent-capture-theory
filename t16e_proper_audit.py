"""
T16e done properly: placebo-controlled test for genuine RV alignment beyond
the chemodynamical-disk background.

Same logic as t16d_proper_audit.py but for RV: for each cluster X, do
matched-to-X stars have RV closer to X's mean RV than to a randomly-picked
other cluster's mean RV?
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
        "fe_h","flag_fe_h","rv_gaia_dr3","parallax"]
g = g[need].copy()
SOLAR_CO = 0.549
g["C_O"] = 10**(g.c_fe - g.o_fe) * SOLAR_CO
g = g[(g.snr_px_ccd3>30)&(g.flag_sp==0)]
for c, fc in [("C_O",None),("mg_fe","flag_mg_fe"),("si_fe","flag_si_fe"),
              ("fe_h","flag_fe_h")]:
    g = g[g[c].notna()]
    if fc and fc in g: g = g[g[fc]==0]
g = g[g.rv_gaia_dr3.notna()&(np.abs(g.rv_gaia_dr3)<200)]
g = g[(g.C_O>0.05)&(g.C_O<2.0)].reset_index(drop=True)
print(f"Field star pool (chem-clean + RV valid): {len(g)}")

# Build cluster templates with member RVs
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
tmem["rv"] = g.iloc[idx[m]].rv_gaia_dr3.values

# templates require N>=5, C_O_std<0.10, and a stable mean RV (sigma_RV/sqrt(N) reasonable)
templates = tmem.groupby("cluster_name").agg(
    n=("C_O","count"),
    C_O=("C_O","mean"), mg_fe=("mg_fe","mean"), si_fe=("si_fe","mean"), fe_h=("fe_h","mean"),
    rv=("rv","mean"), rv_std=("rv","std"), C_O_std=("C_O","std")
).reset_index()
templates = templates[(templates.n>=5)&(templates.C_O_std<0.10)&(templates.rv_std<20)].reset_index(drop=True)
print(f"templates (N>=5, C_O_std<0.10, internal RV scatter <20 km/s): {len(templates)}")

TOL = np.array([0.08, 0.05, 0.05, 0.10])
DIMS = ["C_O","mg_fe","si_fe","fe_h"]
field = g[DIMS+["rv_gaia_dr3"]].values

results = []
for _, tmpl in templates.iterrows():
    cent = np.array([tmpl[d] for d in DIMS])
    chem_match = (np.abs(field[:,:4] - cent[None,:]) < TOL).all(axis=1)
    n_match = chem_match.sum()
    if n_match < 10: continue
    RV_match = field[chem_match, 4]
    med_true = np.median(np.abs(RV_match - tmpl.rv))
    results.append(dict(cluster=tmpl.cluster_name, n_match=n_match,
                        V_X=tmpl.rv, med_d_true=med_true,
                        match_idx=np.where(chem_match)[0]))
R = pd.DataFrame(results)
print(f"clusters with >=10 chem matches: {len(R)}")

# Placebo null: median |RV_match_X - V_Y|
rng = np.random.default_rng(42)
N_PERM = 200
RVs = templates.rv.values
med_null = np.zeros(len(R))
for i, r in R.iterrows():
    match_RV = field[r.match_idx, 4]
    nulls = []
    for _ in range(N_PERM):
        y = rng.integers(0, len(templates))
        while templates.iloc[y].cluster_name == r.cluster:
            y = rng.integers(0, len(templates))
        nulls.append(np.median(np.abs(match_RV - RVs[y])))
    med_null[i] = np.mean(nulls)
R["med_d_null"] = med_null
R["ratio"] = R.med_d_true / R.med_d_null

print(f"\n  median |RV_match - V_X|   (true cluster):     {R.med_d_true.median():.2f} km/s")
print(f"  median |RV_match - V_Y|   (placebo cluster):  {R.med_d_null.median():.2f} km/s")
print(f"  median ratio true/null:                       {R.ratio.median():.4f}")
print(f"  fraction true<null:                           {(R.med_d_true < R.med_d_null).mean():.1%}")
W, p = stats.wilcoxon(R.med_d_true, R.med_d_null, alternative="less")
print(f"  Wilcoxon (true < null): W={W:.0f}, p={p:.4e}")
print(f"  N clusters: {len(R)}")
if R.ratio.median() < 0.95 and p < 0.01:
    print("\n  => SURVIVES: matched stars are systematically closer to TRUE cluster RV")
    print("     than to random other clusters' RV. Real birth-cluster kinematic signal.")
elif R.ratio.median() < 0.99:
    print("\n  => MARGINAL: small residual signal beyond background.")
else:
    print("\n  => NO RESIDUAL: matched stars are no closer to true cluster RV than to random")
    print("     other clusters' RV. T16e's signal is fully chemodynamical background.")
R.to_csv("t16e_proper_audit.csv", index=False)
print("DONE")

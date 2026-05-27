"""
Partial Mantel test: is the corrected spatial-chemical correlation (T10,
r=+0.067) real locality, or just the radial [Fe/H] gradient leaking in?

Builds three inter-cluster distance matrices over the 655 GALAH clusters:
  D_spatial  = 3D Euclidean separation (corrected kpc distances)
  D_chem     = |delta C/O_mean|
  D_feh      = |delta [Fe/H]_mean|
and computes the partial Mantel correlation r(spatial, chem | feh) with a
9999-permutation null.  If r_partial collapses toward 0, the spatial signal
is the metallicity gradient, and "chemically similar clusters are not spatial
neighbours" still holds once [Fe/H] is removed.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

N_PERM = 9999
rng = np.random.default_rng(42)

stars = pd.read_csv("t9_matched_stars.csv")           # dist_cl already corrected to kpc
clstats = pd.read_csv("t9_cluster_stats_with_age.csv").rename(columns={"cluster": "cluster_name"})
pos = (stars.groupby("cluster_name")
       .agg(ra_cl=("ra_cl", "first"), dec_cl=("dec_cl", "first"),
            dist_cl=("dist_cl", "first")).reset_index())
df = clstats.merge(pos, on="cluster_name", how="inner").dropna(
    subset=["C_O_mean", "feh_mean", "ra_cl", "dec_cl", "dist_cl"])
df = df[df["dist_cl"] > 0].reset_index(drop=True)
N = len(df)
print(f"Clusters: {N}")

ra = np.radians(df["ra_cl"].values)
dec = np.radians(df["dec_cl"].values)
d_pc = df["dist_cl"].values * 1000.0
x = d_pc * np.cos(dec) * np.cos(ra)
y = d_pc * np.cos(dec) * np.sin(ra)
z = d_pc * np.sin(dec)

D_sp = squareform(pdist(np.column_stack([x, y, z]), "euclidean"))
D_ch = squareform(pdist(df["C_O_mean"].values.reshape(-1, 1), "cityblock"))
D_fe = squareform(pdist(df["feh_mean"].values.reshape(-1, 1), "cityblock"))

iu = np.triu_indices(N, k=1)
sp, ch, fe = D_sp[iu], D_ch[iu], D_fe[iu]

def partial_r(a, b, c):
    rab = pearsonr(a, b)[0]
    rac = pearsonr(a, c)[0]
    rbc = pearsonr(b, c)[0]
    return (rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2)), rab, rac, rbc

r_part, r_sc, r_sf, r_cf = partial_r(sp, ch, fe)

print("\nBivariate Mantel correlations (upper-triangle Pearson):")
print(f"  spatial  vs chem  : r = {r_sc:+.4f}   (this is the T10 result)")
print(f"  spatial  vs [Fe/H]: r = {r_sf:+.4f}")
print(f"  chem     vs [Fe/H]: r = {r_cf:+.4f}")

# permutation null for the partial correlation: permute cluster order of spatial matrix
order = np.arange(N)
perm = np.empty(N_PERM)
for i in range(N_PERM):
    p = rng.permutation(order)
    sp_p = D_sp[np.ix_(p, p)][iu]
    rab = pearsonr(sp_p, ch)[0]
    rac = pearsonr(sp_p, fe)[0]
    perm[i] = (rab - rac * r_cf) / np.sqrt((1 - rac ** 2) * (1 - r_cf ** 2))

p_one = (np.sum(perm >= r_part) + 1) / (N_PERM + 1)
p_two = (np.sum(np.abs(perm) >= abs(r_part)) + 1) / (N_PERM + 1)

print(f"\nPARTIAL Mantel  r(spatial, chem | [Fe/H]) = {r_part:+.4f}")
print(f"  permutation p (one-sided) = {p_one:.4f}")
print(f"  permutation p (two-sided) = {p_two:.4f}")
print(f"  null mean={perm.mean():+.4f} sd={perm.std():.4f}")

print("\nINTERPRETATION")
print(f"  Raw spatial-chem Mantel r = {r_sc:+.4f} (p~0.013, published-buggy was -0.010)")
print(f"  After removing [Fe/H]:    r = {r_part:+.4f}")
drop = (r_sc - r_part) / r_sc * 100 if r_sc != 0 else 0
print(f"  Reduction from metallicity control: {drop:.0f}% of the raw correlation")
if p_two >= 0.05:
    print("  => Spatial-chemical link is NOT significant after [Fe/H] control:")
    print("     the corrected T10 'signal' is the radial metallicity gradient.")
    print("     'Chemically similar clusters are not spatial neighbours' HOLDS")
    print("     once metallicity is removed -> cleaner tagging argument.")
else:
    print("  => A residual spatial-chemical correlation SURVIVES [Fe/H] control:")
    print("     genuine weak locality confound, independent of metallicity.")
print("DONE")

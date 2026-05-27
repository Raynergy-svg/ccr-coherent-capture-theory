"""
Audit of the dist_cl units bug and its impact on T10 (Mantel) and T19 (radial
gradient).  Does NOT overwrite any canonical result file or CSV.

Root cause (t9_cluster_coherence.py:61):
    dist_col = next((c for c in cg20.columns
                     if "dist" in c.lower() or "plx" in c.lower()), None)
The Cantat-Gaudin 2020 catalog exposes both `plx` (mas) and `DistPc` (pc);
`next()` returned `plx`, so dist_cl is the cluster PARALLAX in mas.  Downstream
code then treats it as a distance in kpc (`* u.kpc`), and t10 multiplies it by
1000 to "convert kpc->pc".  Net: cluster distances are the reciprocal of the
truth (near clusters pushed far, far clusters pulled near).

Corrected distance:  d_kpc = 1 / plx_mas = 1 / dist_cl.
"""
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
from scipy import stats as st
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

rng = np.random.default_rng(42)
stars = pd.read_csv("t9_matched_stars.csv")
clstats = pd.read_csv("t9_cluster_stats_with_age.csv").rename(columns={"cluster": "cluster_name"})

cl = (stars.groupby("cluster_name")
      .agg(ra_cl=("ra_cl", "first"), dec_cl=("dec_cl", "first"),
           dist_cl=("dist_cl", "first")).reset_index())
cl = cl.dropna(subset=["ra_cl", "dec_cl", "dist_cl"])
cl = cl[cl["dist_cl"] > 0]

print("=" * 78)
print("PROOF: dist_cl is parallax (mas), not distance (kpc)")
print("=" * 78)
known_plx = {"Melotte_22": 7.36, "Blanco_1": 4.2, "Ruprecht_147": 3.25,
             "Teutsch_80": 0.349, "NGC_2632": 5.36}
for name, plx in known_plx.items():
    r = cl[cl["cluster_name"] == name]
    if len(r):
        stored = r["dist_cl"].iloc[0]
        print(f"  {name:14s} stored dist_cl={stored:7.3f}  CG20 plx={plx:6.3f} mas  "
              f"true dist={1000/plx:6.0f} pc  -> stored==plx: {abs(stored-plx)<0.05}")
print(f"  dist_cl range: {cl['dist_cl'].min():.3f} to {cl['dist_cl'].max():.2f} "
      f"(absurd as kpc; sensible as mas)")

cl["d_kpc_buggy"] = cl["dist_cl"]            # what the pipeline used (mas-as-kpc)
cl["d_kpc_fixed"] = 1.0 / cl["dist_cl"]      # corrected: 1/plx

# =====================================================================  T19
print("\n" + "=" * 78)
print("T19 — radial coherence gradient: BUGGY vs CORRECTED distances")
print("=" * 78)

def t19_rgal(dist_kpc):
    c = SkyCoord(ra=cl["ra_cl"].values * u.deg, dec=cl["dec_cl"].values * u.deg,
                 distance=dist_kpc.values * u.kpc, frame="icrs").transform_to(Galactocentric())
    R = np.sqrt(c.x.to(u.kpc).value ** 2 + c.y.to(u.kpc).value ** 2)
    return pd.Series(R, index=cl.index)

def t19_run(label, dist_kpc):
    cl["R"] = t19_rgal(dist_kpc)
    rmap = dict(zip(cl["cluster_name"], cl["R"]))
    d = clstats.copy()
    d["R_gal"] = d["cluster_name"].map(rmap)
    d = d[d["R_gal"].notna() & (d["N"] >= 3)]
    rho, p = spearmanr(d["R_gal"], d["C_O_std"])
    inner = d[d["R_gal"] < 7.5]["C_O_std"]
    outer = d[d["R_gal"] >= 8.5]["C_O_std"]
    U, p_mw = mannwhitneyu(inner, outer, alternative="two-sided") if (len(inner) >= 10 and len(outer) >= 10) else (np.nan, np.nan)
    print(f"\n[{label}]  N={len(d)}  R_gal range {d['R_gal'].min():.2f}-{d['R_gal'].max():.2f} kpc "
          f"(median {d['R_gal'].median():.2f})")
    print(f"   Spearman(R_gal, C/O scatter): rho={rho:+.4f}  p={p:.3e}")
    print(f"   Inner(R<7.5) N={len(inner)} med={inner.median():.4f} | "
          f"Outer(R>8.5) N={len(outer)} med={outer.median():.4f} | MW p={p_mw:.3e}")
    return rho, p

t19_run("BUGGY (reproduce published: expect rho=-0.228, p=3.4e-9)", cl["d_kpc_buggy"])
t19_run("CORRECTED (d = 1/plx)", cl["d_kpc_fixed"])

# =====================================================================  T10
print("\n" + "=" * 78)
print("T10 — Mantel spatial-decorrelation: BUGGY vs CORRECTED distances")
print("=" * 78)
d10 = clstats.merge(cl[["cluster_name", "ra_cl", "dec_cl", "dist_cl",
                        "d_kpc_buggy", "d_kpc_fixed"]], on="cluster_name", how="inner")
d10 = d10.dropna(subset=["C_O_mean", "ra_cl", "dec_cl", "dist_cl"])
ra = np.radians(d10["ra_cl"].values); dec = np.radians(d10["dec_cl"].values)
ch_v = squareform(pdist(d10["C_O_mean"].values.reshape(-1, 1), metric="cityblock"))[
    np.triu_indices(len(d10), k=1)]

def mantel(dist_pc, label, n_perm=1999):
    x = dist_pc * np.cos(dec) * np.cos(ra)
    y = dist_pc * np.cos(dec) * np.sin(ra)
    z = dist_pc * np.sin(dec)
    Ds = squareform(pdist(np.column_stack([x, y, z]), metric="euclidean"))
    idx = np.triu_indices(len(d10), k=1)
    sp_v = Ds[idx]
    r_obs, _ = pearsonr(sp_v, ch_v)
    order = np.arange(len(d10))
    perm = np.array([pearsonr(Ds[np.ix_(p, p)][idx], ch_v)[0]
                     for p in (rng.permutation(order) for _ in range(n_perm))])
    p_val = (np.sum(perm >= r_obs) + 1) / (n_perm + 1)
    print(f"\n[{label}]  Mantel r={r_obs:+.4f}  p={p_val:.4f}  "
          f"(spatial sep range {sp_v.min()/1e3:.2f}-{sp_v.max()/1e3:.1f} kpc)")
    return r_obs, p_val

# buggy: pipeline multiplied mas-as-kpc by 1000 (its mean<50 heuristic)
mantel(d10["d_kpc_buggy"].values * 1000.0, "BUGGY (reproduce published: expect r=-0.010, p=0.60)")
# corrected: real distance in pc
mantel(1000.0 / d10["dist_cl"].values, "CORRECTED (d_pc = 1000/plx)")
print("\nDONE")

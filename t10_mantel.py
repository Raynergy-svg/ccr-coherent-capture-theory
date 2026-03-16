#!/usr/bin/env python3
# CCT T10 - Spatial Decorrelation Mantel Test
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/root/ccr_crossmatch"
N_PERMS  = 9999
SEED     = 42
rng      = np.random.default_rng(SEED)

print("=" * 60)
print("CCT T10 - Spatial Decorrelation Mantel Test")
print("=" * 60)

stats = pd.read_csv(DATA_DIR + "/t9_cluster_stats_with_age.csv")
stats = stats.rename(columns={"cluster": "cluster_name"})
print("Cluster stats: " + str(len(stats)) + " clusters")

stars = pd.read_csv(DATA_DIR + "/t9_matched_stars.csv")
print("Matched stars: " + str(len(stars)) + " rows")

coords_df = (stars.groupby("cluster_name")
                  .agg(ra_cl=("ra_cl","first"), dec_cl=("dec_cl","first"), dist_cl=("dist_cl","first"))
                  .reset_index())

df = stats.merge(coords_df, on="cluster_name", how="inner")
df = df.dropna(subset=["C_O_mean","ra_cl","dec_cl","dist_cl"])
print("After merge: " + str(len(df)) + " clusters")

ra   = np.radians(df["ra_cl"].values)
dec  = np.radians(df["dec_cl"].values)
dist = df["dist_cl"].values
if dist.mean() < 50:
    dist = dist * 1000.0
    print("dist converted kpc->pc, mean=" + str(int(dist.mean())) + " pc")

x = dist * np.cos(dec) * np.cos(ra)
y = dist * np.cos(dec) * np.sin(ra)
z = dist * np.sin(dec)
coords = np.column_stack([x, y, z])

D_spatial = squareform(pdist(coords, metric="euclidean"))
D_chem    = squareform(pdist(df["C_O_mean"].values.reshape(-1,1), metric="cityblock"))

idx  = np.triu_indices(len(df), k=1)
sp_v = D_spatial[idx]
ch_v = D_chem[idx]
print("N pairs: " + str(len(sp_v)))

print("Running Mantel (" + str(N_PERMS) + " perms)...")
r_obs, _ = pearsonr(sp_v, ch_v)
perm_r   = np.empty(N_PERMS)
order    = np.arange(len(df))
for i in range(N_PERMS):
    p = rng.permutation(order)
    perm_r[i], _ = pearsonr(D_spatial[np.ix_(p,p)][idx], ch_v)

p_val = (np.sum(perm_r >= r_obs) + 1) / (N_PERMS + 1)

sig = "ns p>=0.05"
if p_val < 0.001: sig = "*** p<0.001"
elif p_val < 0.01: sig = "** p<0.01"
elif p_val < 0.05: sig = "* p<0.05"

print("Mantel r = " + str(round(r_obs,4)) + "  p = " + str(round(p_val,4)) + "  " + sig)

if r_obs > 0.1 and p_val < 0.05:
    verdict = "ISM null SUPPORTED - spatial confound present"
elif p_val >= 0.05 or r_obs <= 0:
    verdict = "CCT CONSISTENT - chemistry spatially independent"
else:
    verdict = "WEAK/AMBIGUOUS"
print("Verdict: " + verdict)

r_partial = None
if "age_gyr" in df.columns and not df["age_gyr"].isna().all():
    ages  = df["age_gyr"].fillna(df["age_gyr"].median()).values.reshape(-1,1)
    v_age = squareform(pdist(ages, metric="cityblock"))[idx]
    def resid(y, x):
        A = np.column_stack([x, np.ones_like(x)])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ c
    r_partial, p_part = pearsonr(resid(sp_v,v_age), resid(ch_v,v_age))
    print("Partial r (age-controlled) = " + str(round(r_partial,4)) + " p~" + str(round(p_part,4)))

bins = np.unique(np.percentile(sp_v, np.linspace(0,100,11)))
bm, bc, bs = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (sp_v >= lo) & (sp_v < hi)
    if mask.sum() > 5:
        v = ch_v[mask]
        bm.append((lo+hi)/2); bc.append(v.mean()); bs.append(v.std()/np.sqrt(len(v)))
bm = np.array(bm); bc = np.array(bc); bs = np.array(bs)

fig = plt.figure(figsize=(14,10))
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0,0])
hb  = ax1.hexbin(sp_v/1e3, ch_v, gridsize=40, cmap="viridis", mincnt=1)
ax1.set_xlabel("Spatial separation (kpc)")
ax1.set_ylabel("|delta C/O| (dex)")
ax1.set_title("Spatial vs Chemical\nr=" + str(round(r_obs,4)) + " p=" + str(round(p_val,4)) + " " + sig)
plt.colorbar(hb, ax=ax1, label="pairs")

ax2 = fig.add_subplot(gs[0,1])
ax2.hist(perm_r, bins=60, color="steelblue", alpha=0.75, edgecolor="none", label="Permuted r")
ax2.axvline(r_obs, color="crimson", lw=2.5, label="Obs r=" + str(round(r_obs,4)))
ax2.axvline(np.percentile(perm_r,95), color="orange", lw=1.5, ls="--", label="95th pctile")
ax2.set_xlabel("Mantel r"); ax2.set_ylabel("Count")
ax2.set_title("Permutation Null Distribution")
ax2.legend(fontsize=8)

ax3 = fig.add_subplot(gs[1,0])
ax3.errorbar(bm/1e3, bc, yerr=bs, fmt="o-", color="darkorange", lw=2, capsize=4)
ax3.axhline(ch_v.mean(), color="grey", ls="--", lw=1, label="Global mean")
ax3.set_xlabel("Mean separation (kpc)"); ax3.set_ylabel("Mean |delta C/O|")
ax3.set_title("Distance-Decay (flat=ISM null rejected)")
ax3.legend(fontsize=8)

ax4 = fig.add_subplot(gs[1,1])
ax4.axis("off")
lines = ["T10 RESULT SUMMARY", "-"*28,
         "N clusters: " + str(len(df)),
         "N pairs:    " + str(len(sp_v)),
         "Mantel r:   " + str(round(r_obs,4)),
         "p-value:    " + str(round(p_val,4)),
         "Sig: " + sig, "", verdict]
if r_partial is not None:
    lines += ["", "Partial r (age): " + str(round(r_partial,4))]
ax4.text(0.05, 0.95, "\n".join(lines), transform=ax4.transAxes,
         fontsize=9, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.suptitle("CCT T10 - Spatial Decorrelation (Mantel Test)", fontsize=13, fontweight="bold")
plt.savefig(DATA_DIR + "/t10_mantel_plot.png", dpi=150, bbox_inches="tight")
print("Plot: t10_mantel_plot.png")

with open(DATA_DIR + "/t10_mantel_results.txt", "w") as f:
    f.write("CCT T10 - Mantel Test\n" + "="*40 + "\n")
    f.write("N clusters: " + str(len(df)) + "\n")
    f.write("N pairs:    " + str(len(sp_v)) + "\n")
    f.write("Mantel r:   " + str(round(r_obs,6)) + "\n")
    f.write("p-value:    " + str(round(p_val,6)) + "\n")
    f.write("Sig:        " + sig + "\n")
    if r_partial is not None:
        f.write("Partial r (age): " + str(round(r_partial,6)) + "\n")
    f.write("\nVerdict: " + verdict + "\n\nDecay bins:\n")
    for m,c,s in zip(bm,bc,bs):
        f.write("  " + str(round(m/1e3,2)) + " kpc | " + str(round(c,4)) + " +/- " + str(round(s,4)) + "\n")

print("Results: t10_mantel_results.txt")
print("T10 complete.")

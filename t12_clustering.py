#!/usr/bin/env python3
# CCT T12 - Chemical Family Clustering (Ward Linkage) - fixed
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import kruskal, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

DATA_DIR  = "/root/ccr_crossmatch"
MIN_STARS = 3
N_CLUSTERS_TEST = [3, 4, 5, 6, 8, 10]

print("=" * 60)
print("CCT T12 - Chemical Family Clustering (Ward Linkage)")
print("=" * 60)

print("Loading T9 matched stars...")
stars = pd.read_csv(DATA_DIR + "/t9_matched_stars.csv")
print("Stars: " + str(len(stars)))

print("Loading GALAH DR4...")
hdu   = fits.open(DATA_DIR + "/galah_dr4_allstar_240705.fits", memmap=True)
galah = pd.DataFrame(hdu[1].data)
hdu.close()
galah.columns = [c.lower() for c in galah.columns]

ELEMENTS = [
    ("c_fe",  "flag_c_fe",  "C/Fe"),
    ("o_fe",  "flag_o_fe",  "O/Fe"),
    ("mg_fe", "flag_mg_fe", "Mg/Fe"),
    ("si_fe", "flag_si_fe", "Si/Fe"),
    ("al_fe", "flag_al_fe", "Al/Fe"),
    ("ba_fe", "flag_ba_fe", "Ba/Fe"),
    ("eu_fe", "flag_eu_fe", "Eu/Fe"),
    ("ce_fe", "flag_ce_fe", "Ce/Fe"),
    ("y_fe",  "flag_y_fe",  "Y/Fe"),
    ("mn_fe", "flag_mn_fe", "Mn/Fe"),
    ("ni_fe", "flag_ni_fe", "Ni/Fe"),
    ("fe_h",  "flag_fe_h",  "Fe/H"),
]

stars["ra_r"]  = stars["ra"].round(4)
stars["dec_r"] = stars["dec"].round(4)
galah["ra_r"]  = galah["ra"].round(4)
galah["dec_r"] = galah["dec"].round(4)

need = ["ra_r","dec_r"]
for abund_col, flag_col, _ in ELEMENTS:
    if abund_col in galah.columns:
        need.append(abund_col)
    if flag_col in galah.columns:
        need.append(flag_col)
need = list(dict.fromkeys(need))

merged = stars.merge(galah[need], on=["ra_r","dec_r"], how="inner")
print("After join: " + str(len(merged)) + " stars")
print("Cols in merged: " + str([c for c in [e[0] for e in ELEMENTS] if c in merged.columns]))

stats = pd.read_csv(DATA_DIR + "/t9_cluster_stats_with_age.csv")
stats = stats.rename(columns={"cluster": "cluster_name"})
coords_df = (stars.groupby("cluster_name")
                  .agg(ra_cl=("ra_cl","first"), dec_cl=("dec_cl","first"), dist_cl=("dist_cl","first"))
                  .reset_index())

print("Building cluster abundance vectors...")
records = []
for cname, grp in merged.groupby("cluster_name"):
    rec = {"cluster_name": cname, "n_stars": len(grp)}
    for abund_col, flag_col, label in ELEMENTS:
        if abund_col not in grp.columns:
            rec[abund_col] = np.nan
            continue
        if flag_col in grp.columns:
            clean = grp[grp[flag_col] == 0][abund_col].dropna()
        else:
            clean = grp[abund_col].dropna()
        rec[abund_col] = float(clean.mean()) if len(clean) >= MIN_STARS else np.nan
    records.append(rec)

clust = pd.DataFrame(records)
clust = clust.merge(stats[["cluster_name","age_gyr","age_bin","C_O_mean","C_O_std"]], on="cluster_name", how="left")
clust = clust.merge(coords_df, on="cluster_name", how="left")
print("Clusters with vectors: " + str(len(clust)))

elem_cols = [e[0] for e in ELEMENTS if e[0] in clust.columns]
print("Element cols available: " + str(elem_cols))

clust_sub = clust.dropna(subset=elem_cols, thresh=6).copy()
print("Clusters with >=6 elements: " + str(len(clust_sub)))

feat = clust_sub[elem_cols].copy()

# Robust fill: per-column median, fallback to 0 if median is NaN
for col in elem_cols:
    med = feat[col].median()
    fill_val = med if np.isfinite(med) else 0.0
    feat[col] = feat[col].fillna(fill_val)

# Kill any remaining inf/-inf
feat = feat.replace([np.inf, -np.inf], 0.0)
feat = feat.fillna(0.0)

print("NaN in feat: " + str(feat.isna().sum().sum()))
print("Inf in feat: " + str(np.isinf(feat.values).sum()))

scaler = StandardScaler()
X = scaler.fit_transform(feat.values)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print("Feature matrix: " + str(X.shape) + " finite=" + str(np.all(np.isfinite(X))))

print("Running Ward linkage...")
Z = linkage(X, method="ward")

pca    = PCA(n_components=2)
X_pca  = pca.fit_transform(X)
var1   = round(pca.explained_variance_ratio_[0]*100, 1)
var2   = round(pca.explained_variance_ratio_[1]*100, 1)
print("PCA var: PC1=" + str(var1) + "% PC2=" + str(var2) + "%")

print("Testing k values...")
k_results = []
for k in N_CLUSTERS_TEST:
    labels = fcluster(Z, k, criterion="maxclust")
    age_divs = []
    sp_spreads = []
    for fam in range(1, k+1):
        mask = labels == fam
        if mask.sum() < 2:
            continue
        ages = clust_sub["age_gyr"][mask].dropna()
        if len(ages) >= 2:
            age_divs.append(ages.std())
        ra_f  = clust_sub["ra_cl"][mask].dropna().values
        dec_f = clust_sub["dec_cl"][mask].dropna().values
        if len(ra_f) >= 2:
            sp_spreads.append(np.sqrt(ra_f.std()**2 + dec_f.std()**2))
    mean_age_div = np.mean(age_divs) if age_divs else np.nan
    mean_sp      = np.mean(sp_spreads) if sp_spreads else np.nan
    kw_ps = []
    for col in ["fe_h","mg_fe","ba_fe"]:
        if col in clust_sub.columns:
            groups = [clust_sub[col][labels==f].dropna().values for f in range(1,k+1) if (labels==f).sum()>1]
            groups = [g for g in groups if len(g)>1]
            if len(groups) >= 2:
                try:
                    _, p = kruskal(*groups)
                    kw_ps.append(p)
                except:
                    pass
    mean_kw_p = np.mean(kw_ps) if kw_ps else np.nan
    print("k=" + str(k) + " age_div=" + str(round(mean_age_div,3)) +
          " sp=" + str(round(mean_sp,2)) + " kw_p=" + str(round(mean_kw_p,4)))
    k_results.append({"k":k,"mean_age_div":mean_age_div,"mean_sp_spread":mean_sp,"mean_kw_p":mean_kw_p})

kr_df = pd.DataFrame(k_results)

K_MAIN  = 5
labels5 = fcluster(Z, K_MAIN, criterion="maxclust")
clust_sub = clust_sub.copy()
clust_sub["family"] = labels5

print("\nFamily summary (k=5):")
all_age_divs = []
for fam in range(1, K_MAIN+1):
    mask = labels5 == fam
    ages = clust_sub["age_gyr"][mask].dropna()
    age_str = str(round(ages.mean(),2)) + " +/- " + str(round(ages.std(),2)) if len(ages)>1 else "N/A"
    print("Family " + str(fam) + ": n=" + str(mask.sum()) + " age=" + age_str + " Gyr")
    if len(ages) >= 2:
        all_age_divs.append(ages.std())

mean_within = np.mean(all_age_divs)
global_std  = clust_sub["age_gyr"].dropna().std()
ratio       = mean_within / global_std
print("\nWithin-family age std: " + str(round(mean_within,3)) + " Gyr")
print("Global age std:        " + str(round(global_std,3)) + " Gyr")
print("Ratio:                 " + str(round(ratio,3)))
if ratio > 0.7:
    verdict = "CCT CONSISTENT - age-diverse families"
else:
    verdict = "AGE-CLUSTERED - families age-segregated"
print("Verdict: " + verdict)

r_sp, p_sp = spearmanr(clust_sub["family"], clust_sub["ra_cl"].fillna(0))
print("Family vs RA: r=" + str(round(r_sp,4)) + " p=" + str(round(p_sp,4)))

# Plot
fig = plt.figure(figsize=(16,12))
gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

ax1 = fig.add_subplot(gs[0,:2])
dendrogram(Z, ax=ax1, truncate_mode="lastp", p=30,
           leaf_rotation=90, leaf_font_size=7, color_threshold=0.7*max(Z[:,2]))
ax1.set_title("Ward Linkage Dendrogram (last 30 merges)")
ax1.set_xlabel("Cluster"); ax1.set_ylabel("Distance")

ax2 = fig.add_subplot(gs[0,2])
ax2.plot(kr_df["k"], kr_df["mean_age_div"], "o-", color="steelblue", label="Age div")
ax2.set_xlabel("k"); ax2.set_ylabel("Mean within-family age std (Gyr)")
ax2.set_title("Age Diversity vs k")
ax2b = ax2.twinx()
ax2b.plot(kr_df["k"], kr_df["mean_kw_p"], "s--", color="coral", label="KW p")
ax2b.set_ylabel("KW p-value")
ax2.legend(loc="upper left", fontsize=7); ax2b.legend(loc="upper right", fontsize=7)

ax3 = fig.add_subplot(gs[1,0])
colors5 = ["steelblue","coral","green","purple","orange"]
for fam in range(1, K_MAIN+1):
    mask = labels5 == fam
    ax3.scatter(X_pca[mask,0], X_pca[mask,1], s=20, alpha=0.6,
                color=colors5[fam-1], label="F"+str(fam))
ax3.set_xlabel("PC1 (" + str(var1) + "%)"); ax3.set_ylabel("PC2 (" + str(var2) + "%)")
ax3.set_title("Chemical Families in PCA Space"); ax3.legend(fontsize=7)

ax4 = fig.add_subplot(gs[1,1])
fam_ages = [clust_sub["age_gyr"][labels5==f].dropna().values for f in range(1,K_MAIN+1)]
fam_ages = [a for a in fam_ages if len(a)>0]
ax4.boxplot(fam_ages, labels=["F"+str(i+1) for i in range(len(fam_ages))])
ax4.set_xlabel("Family"); ax4.set_ylabel("Age (Gyr)")
ax4.set_title("Age per Family\n(wide=age-diverse=CCT)")
ax4.axhline(1.0, color="red", ls="--", lw=1)

ax5 = fig.add_subplot(gs[1,2])
ax5.axis("off")
lines = ["T12 RESULT SUMMARY","-"*26,
         "Clusters: "+str(len(clust_sub)),
         "Elements: "+str(len(elem_cols)),
         "K: "+str(K_MAIN),"",
         "Within-age std: "+str(round(mean_within,3))+" Gyr",
         "Global age std: "+str(round(global_std,3))+" Gyr",
         "Ratio: "+str(round(ratio,3)),"",
         "Family vs RA:",
         "r="+str(round(r_sp,4))+" p="+str(round(p_sp,4)),"",
         verdict]
ax5.text(0.05,0.95,"\n".join(lines),transform=ax5.transAxes,
         fontsize=8,va="top",fontfamily="monospace",
         bbox=dict(boxstyle="round",facecolor="lightyellow",alpha=0.8))

plt.suptitle("CCT T12 - Chemical Family Clustering (Ward Linkage)",fontsize=13,fontweight="bold")
plt.savefig(DATA_DIR+"/t12_clustering_plot.png",dpi=150,bbox_inches="tight")
print("Plot: t12_clustering_plot.png")

out_cols = ["cluster_name","family","age_gyr","age_bin","ra_cl","dec_cl","dist_cl","C_O_mean","C_O_std"] + elem_cols
clust_sub[[c for c in out_cols if c in clust_sub.columns]].to_csv(DATA_DIR+"/t12_cluster_families.csv",index=False)

with open(DATA_DIR+"/t12_results.txt","w") as f:
    f.write("CCT T12 - Chemical Family Clustering\n"+"="*40+"\n")
    f.write("Clusters: "+str(len(clust_sub))+"\n")
    f.write("Elements: "+str(elem_cols)+"\n")
    f.write("K: "+str(K_MAIN)+"\n\n")
    f.write("Within-family age std: "+str(round(mean_within,3))+" Gyr\n")
    f.write("Global age std: "+str(round(global_std,3))+" Gyr\n")
    f.write("Ratio: "+str(round(ratio,3))+"\n")
    f.write("Family vs RA: r="+str(round(r_sp,4))+" p="+str(round(p_sp,4))+"\n\n")
    f.write("Verdict: "+verdict+"\n\n")
    f.write("Family summary:\n")
    for fam in range(1,K_MAIN+1):
        mask = labels5==fam
        ages = clust_sub["age_gyr"][mask].dropna()
        age_str = str(round(ages.mean(),2))+" +/- "+str(round(ages.std(),2)) if len(ages)>1 else "N/A"
        f.write("  F"+str(fam)+": n="+str(mask.sum())+" age="+age_str+"\n")
    f.write("\nk-sweep:\n")
    for _,row in kr_df.iterrows():
        f.write("  k="+str(int(row["k"]))+" age_div="+str(round(row["mean_age_div"],3))+
                " sp="+str(round(row["mean_sp_spread"],2))+" kw_p="+str(round(row["mean_kw_p"],4))+"\n")

print("Results: t12_results.txt")
print("T12 complete.")

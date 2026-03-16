import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

an = pd.read_csv("matched_ccr_cleaned.csv")
an["CCR"] = an["CCR"].clip(lower=0.0)

rho, pval = spearmanr(np.log10(an["pl_orbsmax"]), an["CCR"])
geoms = sorted(an["obs_geometry"].unique())
gt = an[an["obs_geometry"]=="Transit"]["CCR"].values
gd = an[an["obs_geometry"]=="Direct"]["CCR"].values
_, mwp = mannwhitneyu(gt, gd, alternative="two-sided")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
fig.patch.set_facecolor("white")

COLORS  = {"Direct":"#2ca02c","Eclipse":"#d62728","Transit":"#1f77b4"}
MARKERS = {"Direct":"D","Eclipse":"s","Transit":"o"}

for g, grp in an.groupby("obs_geometry"):
    ax1.errorbar(grp["pl_orbsmax"],grp["CCR"],yerr=grp["sigma_CCR"],
        fmt=MARKERS[g],color=COLORS[g],label=g+" (N="+str(len(grp))+")",
        capsize=3,alpha=0.85,markersize=7,elinewidth=0.8,ecolor=COLORS[g])

ss_names=["Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune"]
ss_a   =[0.387,0.723,1.000,1.524,5.203,9.537,19.191,30.069]
ss_ccr =[0.541,0.524,0.514,0.494,0.009,0.071,0.101,0.131]
ax1.scatter(ss_a,ss_ccr,marker="*",s=220,color="gold",zorder=10,
    label="Solar system",edgecolors="black",linewidths=0.7)
for name,a,ccr in zip(ss_names,ss_a,ss_ccr):
    if name in {"Mercury","Earth","Jupiter","Neptune"}:
        ax1.annotate(name,xy=(a,ccr),xytext=(a*1.4,ccr+0.03),
            fontsize=7.5,color="saddlebrown",fontweight="bold",
            arrowprops=dict(arrowstyle="-",color="saddlebrown",lw=0.7,alpha=0.7))

outlier_labels={"WASP-178 b":"WASP-178 b (T~2200K)","HD 209458 b":"HD 209458 b (retrieval scatter)","KELT-20 b":"KELT-20 b (T~2300K)"}
for _, row in an.iterrows():
    if row["pl_name"] in outlier_labels:
        ax1.annotate(outlier_labels[row["pl_name"]],
            xy=(row["pl_orbsmax"],row["CCR"]),
            xytext=(row["pl_orbsmax"]*3.0,row["CCR"]-0.15),
            fontsize=7,color="dimgray",
            arrowprops=dict(arrowstyle="-",color="dimgray",lw=0.8,alpha=0.8),ha="left")

ax1.axhline(0.0,color="black",linestyle="--",linewidth=0.9,alpha=0.5,zorder=1)
ax1.axhline(0.518,color="darkorange",linestyle=":",linewidth=1.5,alpha=0.8,
    label="Solar rocky mean (0.518)",zorder=1)
ax1.set_xscale("log")
ax1.set_xlim(0.007,400)
ax1.set_ylim(-0.05,2.1)
ax1.set_xlabel("Orbital Distance (AU)",fontsize=12)
ax1.set_ylabel("CCR = |log$_{10}$(C/O$_{planet}$ / C/O$_{star}$)|",fontsize=12)
ax1.set_title("H$_{UNIVERSAL}$ CCR Gradient Test\nSpearman rho="+str(round(rho,3))+"  p="+str(round(pval,4))+"  N="+str(len(an)),fontsize=12)
ax1.legend(fontsize=8.5,loc="upper right",framealpha=0.9)
ax1.grid(True,alpha=0.15)
ax1.tick_params(labelsize=10)

bp=ax2.boxplot([an[an["obs_geometry"]==g]["CCR"].values for g in geoms],
    tick_labels=geoms,patch_artist=True,
    medianprops=dict(color="white",linewidth=2),
    whiskerprops=dict(linewidth=1.2),capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o",markerfacecolor="gray",markersize=6,linestyle="none",alpha=0.6))
for patch,g in zip(bp["boxes"],geoms):
    patch.set_facecolor(COLORS[g]);patch.set_alpha(0.75)

np.random.seed(42)
for i,g in enumerate(geoms):
    vals=an[an["obs_geometry"]==g]["CCR"].values
    jitter=np.random.uniform(-0.12,0.12,len(vals))
    ax2.scatter(np.full(len(vals),i+1)+jitter,vals,
        color=COLORS[g],alpha=0.6,s=30,zorder=5,edgecolors="white",linewidths=0.4)

ax2.axhline(0.0,color="black",linestyle="--",linewidth=0.9,alpha=0.5)
ax2.axhline(0.518,color="darkorange",linestyle=":",linewidth=1.5,alpha=0.8,label="Solar rocky CCR = 0.518")
for i,g in enumerate(geoms):
    mv=an[an["obs_geometry"]==g]["CCR"].mean()
    ax2.text(i+1,-0.03,"mean="+str(round(mv,3)),ha="center",fontsize=8,color=COLORS[g],fontstyle="italic")

ax2.text(0.97,0.97,"Transit vs Direct\nMann-Whitney p="+str(round(mwp,3)),
    transform=ax2.transAxes,ha="right",va="top",fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3",facecolor="lightyellow",edgecolor="gray",alpha=0.8))
ax2.set_ylabel("CCR",fontsize=12)
ax2.set_title("CCR Distribution by Observation Geometry",fontsize=12)
ax2.legend(fontsize=8.5,loc="upper left",framealpha=0.9)
ax2.grid(True,alpha=0.15,axis="y")
ax2.set_ylim(-0.08,2.1)
ax2.tick_params(labelsize=10)

plt.suptitle("Certan (2026) - Carbon-to-Oxygen Ratio Coherence in Exoplanet Atmospheres",
    fontsize=10,style="italic",color="gray",y=1.01)
plt.tight_layout()
plt.savefig("CCR_gradient_plot_v2.png",dpi=300,bbox_inches="tight",facecolor="white")
plt.close()
print("Saved: CCR_gradient_plot_v2.png")

#!/usr/bin/env python3
# CCT T13 - Off-locus Mg/Fe vs Fe/H Anomaly Test (fixed)
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

DATA_DIR     = "/root/ccr_crossmatch"
MIN_STARS    = 3
SIGMA_THRESH = 2.0

print("=" * 60)
print("CCT T13 - Off-locus Mg/Fe vs Fe/H Anomaly Test")
print("=" * 60)

# t9_matched_stars already has the alpha elements --- use directly
print("Loading T9 matched stars...")
stars = pd.read_csv(DATA_DIR + "/t9_matched_stars.csv")
print("Stars: " + str(len(stars)) + "  cols: " + str(list(stars.columns)))

# Alpha/base elements already in stars
STARS_COLS = ["mg_fe","fe_h","c_fe","o_fe","si_fe","al_fe"]

# S-process / extra elements need GALAH join
GALAH_EXTRA = [
    ("ba_fe", "flag_ba_fe"),
    ("eu_fe", "flag_eu_fe"),
    ("mn_fe", "flag_mn_fe"),
    ("ni_fe", "flag_ni_fe"),
]

print("Loading GALAH DR4 for extra elements...")
hdu   = fits.open(DATA_DIR + "/galah_dr4_allstar_240705.fits", memmap=True)
galah = pd.DataFrame(hdu[1].data)
hdu.close()
galah.columns = [c.lower() for c in galah.columns]

stars["ra_r"]  = stars["ra"].round(4)
stars["dec_r"] = stars["dec"].round(4)
galah["ra_r"]  = galah["ra"].round(4)
galah["dec_r"] = galah["dec"].round(4)

need = ["ra_r","dec_r"]
for abund_col, flag_col in GALAH_EXTRA:
    if abund_col in galah.columns: need.append(abund_col)
    if flag_col  in galah.columns: need.append(flag_col)
need = list(dict.fromkeys(need))

merged = stars.merge(galah[need], on=["ra_r","dec_r"], how="inner")
print("After join: " + str(len(merged)) + " stars")

# Load stats and coords
stats = pd.read_csv(DATA_DIR + "/t9_cluster_stats_with_age.csv")
stats = stats.rename(columns={"cluster": "cluster_name"})
coords_df = (stars.groupby("cluster_name")
                  .agg(ra_cl=("ra_cl","first"), dec_cl=("dec_cl","first"), dist_cl=("dist_cl","first"))
                  .reset_index())

# Build per-cluster mean abundances
print("Building cluster abundance vectors...")
records = []
for cname, grp in merged.groupby("cluster_name"):
    rec = {"cluster_name": cname, "n_stars": len(grp)}
    # Stars-native cols --- no flag, just clean finite values
    for col in STARS_COLS:
        if col in grp.columns:
            vals = grp[col].replace([np.inf,-np.inf], np.nan).dropna()
            rec[col] = float(vals.mean()) if len(vals) >= MIN_STARS else np.nan
        else:
            rec[col] = np.nan
    # GALAH extra cols --- use flag
    for abund_col, flag_col in GALAH_EXTRA:
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
print("Clusters: " + str(len(clust)))

mg_df = clust.dropna(subset=["mg_fe","fe_h"]).copy()
print("Clusters with Mg/Fe + Fe/H: " + str(len(mg_df)))

# Fit MW locus --- Huber poly2
X_feh  = mg_df["fe_h"].values.reshape(-1,1)
y_mg   = mg_df["mg_fe"].values
poly   = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X_feh)
huber  = HuberRegressor(epsilon=1.5, max_iter=500)
huber.fit(X_poly, y_mg)
mg_pred   = huber.predict(X_poly)
residuals = y_mg - mg_pred
resid_std = np.std(residuals)
print("MW locus residual std = " + str(round(resid_std,4)) + " dex")

mg_df = mg_df.copy()
mg_df["mg_pred"]   = mg_pred
mg_df["mg_resid"]  = residuals
mg_df["mg_zscore"] = residuals / resid_std

mg_df["off_locus"]      = (np.abs(mg_df["mg_zscore"]) >= SIGMA_THRESH)
mg_df["off_locus_high"] = (mg_df["mg_zscore"] >= SIGMA_THRESH)
mg_df["off_locus_low"]  = (mg_df["mg_zscore"] <= -SIGMA_THRESH)

n_total    = len(mg_df)
n_off      = int(mg_df["off_locus"].sum())
n_off_high = int(mg_df["off_locus_high"].sum())
n_off_low  = int(mg_df["off_locus_low"].sum())
n_on       = n_total - n_off

print("Total: " + str(n_total) + "  On: " + str(n_on) + "  Off: " + str(n_off) +
      " (high=" + str(n_off_high) + " low=" + str(n_off_low) + ")")

on_mask  = ~mg_df["off_locus"]
off_mask =  mg_df["off_locus"]

# Test 1: C/O coherence
on_co  = mg_df[on_mask]["C_O_std"].dropna()
off_co = mg_df[off_mask]["C_O_std"].dropna()
print("\n--- Test 1: C/O coherence ---")
print("On  C/O std mean=" + str(round(on_co.mean(),4)) + " n=" + str(len(on_co)))
print("Off C/O std mean=" + str(round(off_co.mean(),4)) + " n=" + str(len(off_co)))
if len(on_co) > 1 and len(off_co) > 1:
    _, p1 = mannwhitneyu(on_co, off_co, alternative="two-sided")
    print("MWU p=" + str(round(p1,6)))
else:
    p1 = np.nan; print("n/a")

# Test 2: Age
on_age  = mg_df[on_mask]["age_gyr"].dropna()
off_age = mg_df[off_mask]["age_gyr"].dropna()
print("\n--- Test 2: Age ---")
print("On  age mean=" + str(round(on_age.mean(),3)) + " std=" + str(round(on_age.std(),3)))
print("Off age mean=" + str(round(off_age.mean(),3)) + " std=" + str(round(off_age.std(),3)))
if len(on_age) > 1 and len(off_age) > 1:
    _, p2 = mannwhitneyu(on_age, off_age, alternative="two-sided")
    print("MWU p=" + str(round(p2,6)))
else:
    p2 = np.nan; print("n/a")

# Test 3: Multi-element deltas
extra_els = ["si_fe","al_fe","ba_fe","eu_fe","mn_fe","ni_fe"]
extra_els = [e for e in extra_els if e in mg_df.columns]
print("\n--- Test 3: Multi-element deltas ---")
for el in extra_els:
    on_v  = mg_df[on_mask][el].dropna()
    off_v = mg_df[off_mask][el].dropna()
    if len(on_v) > 1 and len(off_v) > 1:
        _, p_el = mannwhitneyu(on_v, off_v, alternative="two-sided")
        delta = off_v.mean() - on_v.mean()
        sig = "*" if p_el < 0.05 else " "
        print(sig + " " + el + ": delta=" + str(round(delta,4)) + " p=" + str(round(p_el,4)))

# Test 4: Mg residual vs C/O std
has_both = mg_df["mg_resid"].notna() & mg_df["C_O_std"].notna()
r_co, p_co = spearmanr(mg_df["mg_zscore"][has_both], mg_df["C_O_std"][has_both])
print("\n--- Test 4: Mg/Fe z-score vs C/O std ---")
print("Spearman r=" + str(round(r_co,4)) + " p=" + str(round(p_co,6)))

# Extreme clusters
print("\n--- Most extreme clusters ---")
top = mg_df.nlargest(8,"mg_zscore")[["cluster_name","fe_h","mg_fe","mg_zscore","age_gyr"]]
bot = mg_df.nsmallest(8,"mg_zscore")[["cluster_name","fe_h","mg_fe","mg_zscore","age_gyr"]]
print("Mg-rich:")
for _, row in top.iterrows():
    age_s = str(round(row["age_gyr"],2)) if not np.isnan(row["age_gyr"]) else "?"
    print("  " + str(row["cluster_name"]) + " z=" + str(round(row["mg_zscore"],2)) + " age=" + age_s)
print("Mg-poor:")
for _, row in bot.iterrows():
    age_s = str(round(row["age_gyr"],2)) if not np.isnan(row["age_gyr"]) else "?"
    print("  " + str(row["cluster_name"]) + " z=" + str(round(row["mg_zscore"],2)) + " age=" + age_s)

# Verdict
cct_signals = 0
if not np.isnan(p1)  and p1  < 0.05: cct_signals += 1
if not np.isnan(p2)  and p2  < 0.05: cct_signals += 1
if not np.isnan(p_co) and abs(r_co) > 0.1 and p_co < 0.05: cct_signals += 1
if cct_signals >= 2:
    verdict = "CCT SUPPORTED - off-locus clusters show multi-element coherence"
elif cct_signals == 1:
    verdict = "CCT PARTIAL - weak off-locus signal"
else:
    verdict = "NULL - off-locus indistinguishable from on-locus"
print("\nCCT signals: " + str(cct_signals) + "/3")
print("Verdict: " + verdict)

# Plot
feh_sm = np.linspace(mg_df["fe_h"].min()-0.1, mg_df["fe_h"].max()+0.1, 300)
mg_sm  = huber.predict(poly.transform(feh_sm.reshape(-1,1)))

fig = plt.figure(figsize=(18,14))
gs  = gridspec.GridSpec(3,3,hspace=0.5,wspace=0.4)

ax1 = fig.add_subplot(gs[0,:2])
ax1.scatter(mg_df["fe_h"][on_mask],  mg_df["mg_fe"][on_mask],  s=12, alpha=0.4, color="steelblue", label="On-locus (n=" + str(n_on) + ")")
ax1.scatter(mg_df["fe_h"][off_mask], mg_df["mg_fe"][off_mask], s=30, alpha=0.8, color="red",       label="Off-locus (n=" + str(n_off) + ")", zorder=5)
ax1.plot(feh_sm, mg_sm, "k-", lw=2, label="MW locus (Huber poly2)")
ax1.plot(feh_sm, mg_sm + SIGMA_THRESH*resid_std, "k--", lw=1, alpha=0.5, label=str(SIGMA_THRESH) + "sig envelope")
ax1.plot(feh_sm, mg_sm - SIGMA_THRESH*resid_std, "k--", lw=1, alpha=0.5)
ax1.set_xlabel("[Fe/H]"); ax1.set_ylabel("[Mg/Fe]")
ax1.set_title("MW Chemical Locus: [Mg/Fe] vs [Fe/H]  |  Red = off-locus (|z|>=" + str(SIGMA_THRESH) + ")")
ax1.legend(fontsize=7)

ax2 = fig.add_subplot(gs[0,2])
ax2.hist(mg_df["mg_zscore"][on_mask],  bins=30, alpha=0.6, color="steelblue", density=True, label="On")
ax2.hist(mg_df["mg_zscore"][off_mask], bins=10, alpha=0.7, color="red",       density=True, label="Off")
ax2.axvline( SIGMA_THRESH, color="k", ls="--", lw=1)
ax2.axvline(-SIGMA_THRESH, color="k", ls="--", lw=1)
ax2.set_xlabel("Mg/Fe z-score"); ax2.set_title("Residual Distribution"); ax2.legend(fontsize=7)

ax3 = fig.add_subplot(gs[1,0])
ax3.boxplot([on_co.values, off_co.values], labels=["On","Off"])
ax3.set_ylabel("C/O std"); ax3.set_title("C/O Coherence\n(lower=tighter)")
p1s = "p=" + str(round(p1,4)) if not np.isnan(p1) else "n/a"
ax3.text(0.5,0.95,p1s,transform=ax3.transAxes,ha="center",fontsize=9,color="red" if (not np.isnan(p1) and p1<0.05) else "black")

ax4 = fig.add_subplot(gs[1,1])
ax4.boxplot([on_age.values, off_age.values], labels=["On","Off"])
ax4.set_ylabel("Age (Gyr)"); ax4.set_title("Age On vs Off locus")
p2s = "p=" + str(round(p2,4)) if not np.isnan(p2) else "n/a"
ax4.text(0.5,0.95,p2s,transform=ax4.transAxes,ha="center",fontsize=9,color="red" if (not np.isnan(p2) and p2<0.05) else "black")

ax5 = fig.add_subplot(gs[1,2])
ax5.scatter(mg_df["mg_zscore"][has_both], mg_df["C_O_std"][has_both],
            s=10, alpha=0.4, c=mg_df["age_gyr"][has_both], cmap="plasma")
ax5.axvline( SIGMA_THRESH, color="k", ls="--", lw=1)
ax5.axvline(-SIGMA_THRESH, color="k", ls="--", lw=1)
ax5.set_xlabel("Mg/Fe z-score"); ax5.set_ylabel("C/O std")
ax5.set_title("Mg residual vs C/O scatter\nr=" + str(round(r_co,3)) + " p=" + str(round(p_co,4)))

ax6 = fig.add_subplot(gs[2,:2])
el_d=[]; el_l=[]; el_p=[]
for el in extra_els:
    on_v  = mg_df[on_mask][el].dropna()
    off_v = mg_df[off_mask][el].dropna()
    if len(on_v)>1 and len(off_v)>1:
        delta = off_v.mean()-on_v.mean()
        _, p_el = mannwhitneyu(on_v, off_v, alternative="two-sided")
        el_d.append(delta); el_l.append(el.replace("_fe","")); el_p.append(p_el)
cbars = ["red" if p<0.05 else "steelblue" for p in el_p]
ax6.bar(el_l, el_d, color=cbars, alpha=0.75)
ax6.axhline(0,color="k",lw=1)
ax6.set_ylabel("Off - On mean (dex)"); ax6.set_xlabel("Element")
ax6.set_title("Multi-element deviation: Off vs On locus  (red=p<0.05)")

ax7 = fig.add_subplot(gs[2,2])
ax7.axis("off")
lines = ["T13 RESULT SUMMARY","-"*24,
         "Clusters: "+str(n_total),
         "On-locus: "+str(n_on),
         "Off-locus: "+str(n_off),
         "  Mg-rich: "+str(n_off_high),
         "  Mg-poor: "+str(n_off_low),
         "",
         "Locus resid std: "+str(round(resid_std,4)),
         "",
         "T1 C/O  p="+( str(round(p1,4)) if not np.isnan(p1) else "n/a"),
         "T2 Age  p="+( str(round(p2,4)) if not np.isnan(p2) else "n/a"),
         "T4 r="+str(round(r_co,3))+" p="+str(round(p_co,4)),
         "",
         "CCT signals: "+str(cct_signals)+"/3",
         "",
         verdict]
ax7.text(0.05,0.97,"\n".join(lines),transform=ax7.transAxes,
         fontsize=7.5,va="top",fontfamily="monospace",
         bbox=dict(boxstyle="round",facecolor="lightyellow",alpha=0.8))

plt.suptitle("CCT T13 - Off-locus [Mg/FE] vs [Fe/H] Anomaly Test",fontsize=13,fontweight="bold")
plt.savefig(DATA_DIR+"/t13_offlocus_plot.png",dpi=150,bbox_inches="tight")
print("Plot: t13_offlocus_plot.png")

out_cols = ["cluster_name","fe_h","mg_fe","mg_pred","mg_resid","mg_zscore","off_locus",
            "age_gyr","age_bin","C_O_std","ra_cl","dec_cl","dist_cl"] + extra_els
mg_df[[c for c in out_cols if c in mg_df.columns]].to_csv(DATA_DIR+"/t13_offlocus_clusters.csv",index=False)

with open(DATA_DIR+"/t13_results.txt","w") as f:
    f.write("CCT T13 - Off-locus Mg/Fe vs Fe/H Anomaly Test\n"+"="*50+"\n\n")
    f.write("Clusters: "+str(n_total)+"\nOn: "+str(n_on)+"\nOff: "+str(n_off)+" (high="+str(n_off_high)+" low="+str(n_off_low)+")\n\n")
    f.write("Locus resid std: "+str(round(resid_std,4))+" dex\n\n")
    f.write("T1 C/O p="+( str(round(p1,6)) if not np.isnan(p1) else "n/a")+"\n")
    f.write("T2 Age p="+( str(round(p2,6)) if not np.isnan(p2) else "n/a")+"\n")
    f.write("T4 r="+str(round(r_co,4))+" p="+str(round(p_co,6))+"\n\n")
    f.write("Multi-element deltas:\n")
    for el in extra_els:
        on_v  = mg_df[on_mask][el].dropna()
        off_v = mg_df[off_mask][el].dropna()
        if len(on_v)>1 and len(off_v)>1:
            delta = off_v.mean()-on_v.mean()
            _, p_el = mannwhitneyu(on_v, off_v, alternative="two-sided")
            f.write("  "+el+": delta="+str(round(delta,4))+" p="+str(round(p_el,4))+"\n")
    f.write("\nVerdict: "+verdict+"\n")

print("Results: t13_results.txt  |  Catalog: t13_offlocus_clusters.csv")
print("T13 complete.")

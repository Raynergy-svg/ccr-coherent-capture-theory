import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import kruskal, levene
from sklearn.preprocessing import StandardScaler
try:
    import hdbscan
    USE_HDBSCAN = True
except ImportError:
    from sklearn.cluster import KMeans
    USE_HDBSCAN = False

GALAH_FILE = "galah_dr4_allstar_240705.fits"
OUTPUT_CSV  = "t6_chem_cluster.csv"
OUTPUT_PLOT = "t6_chem_cluster_plot.png"
SNR_MIN     = 30
SOLAR_CO    = 0.549
HDBSCAN_MIN_CLUSTER = 100
HDBSCAN_MIN_SAMPLES = 30
N_CLUSTERS_FALLBACK = 12

def log(msg):
    print("[INFO] " + str(msg), flush=True)

log("Loading GALAH DR4...")
galah = Table.read(GALAH_FILE).to_pandas()
log("Raw rows: " + str(len(galah)))

galah = galah[
    (galah["flag_sp"]   == 0) &
    (galah["flag_fe_h"] == 0) &
    (galah["flag_c_fe"] == 0) &
    (galah["flag_o_fe"] == 0) &
    (galah["snr_px_ccd3"] > SNR_MIN)
].copy()
log("After quality cuts: " + str(len(galah)))

CHEM_COLS = ["fe_h", "c_fe", "o_fe", "mg_fe", "si_fe", "al_fe"]
ERR_COLS  = ["e_fe_h", "e_c_fe", "e_o_fe", "e_mg_fe", "e_si_fe", "e_al_fe"]
FLAG_COLS = ["flag_fe_h", "flag_c_fe", "flag_o_fe", "flag_mg_fe", "flag_si_fe", "flag_al_fe"]

available_chem = [c for c in CHEM_COLS if c in galah.columns]
available_err  = [c for c in ERR_COLS  if c in galah.columns]
available_flag = [c for c in FLAG_COLS if c in galah.columns]
log("Available chemistry cols: " + str(available_chem))

for fc in available_flag:
    galah = galah[galah[fc] == 0].copy()

galah = galah.dropna(subset=available_chem + available_err).copy()
log("Stars with clean full chemistry: " + str(len(galah)))

galah["delta_co"] = galah["c_fe"] - galah["o_fe"]
galah["C_O"]      = (10.0 ** galah["delta_co"]) * SOLAR_CO
galah["sigma_CO"] = galah["C_O"] * np.log(10) * np.sqrt(
    galah["e_c_fe"]**2 + galah["e_o_fe"]**2)

galah = galah[
    (galah["C_O"] > 0.05) & (galah["C_O"] < 2.0) &
    (galah["sigma_CO"] > 0) & (galah["sigma_CO"] < 0.3)
].copy()
log("After C/O sanity filter: " + str(len(galah)))

KIN_COLS = ["rv_gaia_dr3", "parallax"]
has_kin = all(c in galah.columns for c in KIN_COLS)
if has_kin:
    galah = galah.dropna(subset=KIN_COLS).copy()
    galah = galah[galah["parallax"] > 0.5].copy()
    log("Stars after kinematic filter: " + str(len(galah)))

X_chem  = galah[available_chem].values
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_chem)

if USE_HDBSCAN:
    log("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom")
    galah["chem_group"] = clusterer.fit_predict(X_scaled)
    n_noise = (galah["chem_group"] == -1).sum()
    log("HDBSCAN noise points: " + str(n_noise))
    galah_cl = galah[galah["chem_group"] >= 0].copy()
else:
    log("KMeans fallback N=" + str(N_CLUSTERS_FALLBACK))
    km = KMeans(n_clusters=N_CLUSTERS_FALLBACK, random_state=42, n_init=20)
    galah["chem_group"] = km.fit_predict(X_scaled)
    galah_cl = galah.copy()

N_GROUPS  = galah_cl["chem_group"].nunique()
group_ids = sorted(galah_cl["chem_group"].unique())
log("N groups: " + str(N_GROUPS) + "  N stars: " + str(len(galah_cl)))

group_stats = []
for g in group_ids:
    grp = galah_cl[galah_cl["chem_group"] == g]
    co_vals = grp["C_O"].values
    w       = 1.0 / (grp["sigma_CO"].values ** 2)
    w_mean  = np.sum(co_vals * w) / np.sum(w)
    w_std   = np.sqrt(np.sum(w * (co_vals - w_mean)**2) / np.sum(w))
    intrinsic = np.sqrt(max(0, w_std**2 - np.mean(grp["sigma_CO"].values**2)))
    row = {"group": g, "N": len(grp),
           "C_O_mean": round(w_mean, 4), "C_O_std": round(w_std, 4),
           "intrinsic_scatter": round(intrinsic, 4),
           "feh_mean": round(grp["fe_h"].mean(), 3),
           "feh_std":  round(grp["fe_h"].std(),  3)}
    if has_kin:
        row["rv_mean"]       = round(grp["rv_gaia_dr3"].mean(), 2)
        row["rv_std"]        = round(grp["rv_gaia_dr3"].std(),  2)
        row["parallax_mean"] = round(grp["parallax"].mean(),    3)
    group_stats.append(row)

stats_df = pd.DataFrame(group_stats)
log("Group stats:\n" + stats_df[["group","N","C_O_mean","C_O_std","intrinsic_scatter","feh_mean"]].to_string(index=False))

groups_co = [galah_cl[galah_cl["chem_group"]==g]["C_O"].values for g in group_ids]
kw_stat, kw_p = kruskal(*groups_co)
lev_stat, lev_p = levene(*groups_co)
log("KW H=" + str(round(kw_stat,3)) + " p=" + str(round(kw_p,6)))
log("Levene W=" + str(round(lev_stat,3)) + " p=" + str(round(lev_p,6)))

within_scatter  = stats_df["C_O_std"].mean()
between_scatter = stats_df["C_O_mean"].std()
CCR = between_scatter / within_scatter
log("CCR=" + str(round(CCR,3)) + "  within=" + str(round(within_scatter,4)) + "  between=" + str(round(between_scatter,4)))

if has_kin:
    groups_rv = [galah_cl[galah_cl["chem_group"]==g]["rv_gaia_dr3"].values for g in group_ids]
    kw_rv_stat, kw_rv_p = kruskal(*groups_rv)
    within_rv  = stats_df["rv_std"].mean()
    between_rv = stats_df["rv_mean"].std()
    KCR = between_rv / within_rv
    log("KCR=" + str(round(KCR,3)) + "  within_rv=" + str(round(within_rv,2)) + " km/s  between_rv=" + str(round(between_rv,2)) + " km/s")
    log("CCT prediction: KCR > 1 means chemistry traces birth kinematics")

n_coherent = (stats_df["C_O_std"] < 0.05).sum()
log("Coherent groups (<0.05 dex): " + str(n_coherent) + "/" + str(N_GROUPS))

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.patch.set_facecolor("white")
cmap   = plt.cm.tab20
colors = [cmap(i / max(N_GROUPS, 1)) for i in range(N_GROUPS)]

ax1 = axes[0]
for idx, g in enumerate(group_ids):
    grp = galah_cl[galah_cl["chem_group"] == g]
    ax1.scatter(grp["fe_h"], grp["C_O"], s=1.5, alpha=0.25, color=colors[idx], rasterized=True)
ax1.axhline(SOLAR_CO, color="darkorange", linestyle=":", linewidth=1.5, label="Solar C/O=0.549")
ax1.set_xlabel("[Fe/H]"); ax1.set_ylabel("C/O")
ax1.set_title("Chemistry Clusters\n[Fe/H] vs C/O"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.12)

ax2 = axes[1]
ax2.bar(range(N_GROUPS), stats_df["C_O_std"].values, color=colors[:N_GROUPS], alpha=0.8, edgecolor="white")
ax2.axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="Threshold 0.05")
ax2.axhline(within_scatter, color="navy", linestyle=":", linewidth=1.5, label="Mean=" + str(round(within_scatter,3)))
ax2.set_xlabel("Chemistry Group"); ax2.set_ylabel("C/O Std Dev")
ax2.set_title("Within-Group C/O Scatter\n(CCT coherence, T6)")
ax2.set_xticks(range(N_GROUPS)); ax2.set_xticklabels([str(g) for g in group_ids], fontsize=7)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.12, axis="y")

ax3 = axes[2]
if has_kin:
    ax3.bar(range(N_GROUPS), stats_df["rv_std"].values, color=colors[:N_GROUPS], alpha=0.8, edgecolor="white")
    ax3.axhline(within_rv, color="navy", linestyle=":", linewidth=1.5, label="Mean=" + str(round(within_rv,1)) + " km/s")
    ax3.set_ylabel("RV Std Dev (km/s)"); ax3.set_title("Within-Group RV Scatter\n(kinematic coherence)")
    kv_str = "KCR=" + str(round(KCR,3))
else:
    ax3.text(0.5, 0.5, "No kinematic data", ha="center", va="center", transform=ax3.transAxes)
    kv_str = "no kinematics"
ax3.set_xlabel("Chemistry Group")
ax3.set_xticks(range(N_GROUPS)); ax3.set_xticklabels([str(g) for g in group_ids], fontsize=7)
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.12, axis="y")

method = "HDBSCAN" if USE_HDBSCAN else "KMeans"
plt.suptitle("T6 Chemistry-First | Certan (2025) | " + method + " | CCR=" + str(round(CCR,3)) + " | " + kv_str, fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log("Plot saved: " + OUTPUT_PLOT)
stats_df.to_csv(OUTPUT_CSV, index=False)
log("Results saved: " + OUTPUT_CSV)
log("=== DONE ===")

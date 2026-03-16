import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import kruskal, levene
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

from sklearn.cluster import KMeans

GALAH_FILE = "galah_dr4_allstar_240705.fits"
OUTPUT_CSV  = "t6b_umap_cluster.csv"
OUTPUT_PLOT = "t6b_umap_cluster_plot.png"
SNR_MIN     = 30
SOLAR_CO    = 0.549
UMAP_N_NEIGHBORS  = 30
UMAP_MIN_DIST     = 0.05
UMAP_N_COMPONENTS = 2
UMAP_METRIC       = "euclidean"
UMAP_RANDOM_STATE = 42
HDBSCAN_MIN_CLUSTER = 50
HDBSCAN_MIN_SAMPLES = 3
HDBSCAN_EPSILON     = 0.0
N_CLUSTERS_FALLBACK = 15

def log(msg):
    print("[INFO] " + str(msg), flush=True)

log("Loading GALAH DR4...")
galah = Table.read(GALAH_FILE).to_pandas()
log("Raw rows: " + str(len(galah)))

galah = galah[
    (galah["flag_sp"]     == 0) &
    (galah["flag_fe_h"]   == 0) &
    (galah["flag_c_fe"]   == 0) &
    (galah["flag_o_fe"]   == 0) &
    (galah["snr_px_ccd3"] > SNR_MIN)
].copy()
log("After quality cuts: " + str(len(galah)))

CHEM_COLS = ["fe_h", "c_fe", "o_fe", "mg_fe", "si_fe", "al_fe"]
ERR_COLS  = ["e_fe_h", "e_c_fe", "e_o_fe", "e_mg_fe", "e_si_fe", "e_al_fe"]
FLAG_COLS = ["flag_fe_h", "flag_c_fe", "flag_o_fe", "flag_mg_fe", "flag_si_fe", "flag_al_fe"]

available_chem = [c for c in CHEM_COLS if c in galah.columns]
available_err  = [c for c in ERR_COLS  if c in galah.columns]
available_flag = [c for c in FLAG_COLS if c in galah.columns]
log("Chemistry cols: " + str(available_chem))

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
has_kin  = all(c in galah.columns for c in KIN_COLS)
if has_kin:
    galah = galah.dropna(subset=KIN_COLS).copy()
    galah = galah[galah["parallax"] > 0.5].copy()
    log("Stars after kinematic filter (parallax>0.5): " + str(len(galah)))

X_chem   = galah[available_chem].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_chem)

if HAS_UMAP:
    log("Running UMAP " + str(len(available_chem)) + "D -> 2D  (n_neighbors=" +
        str(UMAP_N_NEIGHBORS) + ", min_dist=" + str(UMAP_MIN_DIST) + ")...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        low_memory=False,
        verbose=False
    )
    X_embedded = reducer.fit_transform(X_scaled)
    galah["umap1"] = X_embedded[:, 0]
    galah["umap2"] = X_embedded[:, 1]
    log("UMAP done.")
else:
    log("WARNING: umap-learn not installed. Install: pip install umap-learn")
    log("Falling back to raw scaled chemistry.")
    X_embedded = X_scaled
    galah["umap1"] = X_scaled[:, 0]
    galah["umap2"] = X_scaled[:, 1]

if HAS_HDBSCAN:
    log("Running HDBSCAN on embedding (min_cluster=" +
        str(HDBSCAN_MIN_CLUSTER) + ", min_samples=" + str(HDBSCAN_MIN_SAMPLES) + ")...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_epsilon=HDBSCAN_EPSILON,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )
    galah["chem_group"] = clusterer.fit_predict(X_embedded)
    n_found = (galah["chem_group"] >= 0).sum()
    n_noise = (galah["chem_group"] == -1).sum()
    log("HDBSCAN: " + str(galah["chem_group"].max() + 1) +
        " clusters | " + str(n_found) + " assigned | " +
        str(n_noise) + " noise (" + str(round(100 * n_noise / len(galah), 1)) + "%)")
    galah_cl = galah[galah["chem_group"] >= 0].copy()
else:
    log("hdbscan not installed — KMeans fallback N=" + str(N_CLUSTERS_FALLBACK))
    km = KMeans(n_clusters=N_CLUSTERS_FALLBACK, random_state=42, n_init=20)
    galah["chem_group"] = km.fit_predict(X_embedded)
    galah_cl = galah.copy()

N_GROUPS  = galah_cl["chem_group"].nunique()
group_ids = sorted(galah_cl["chem_group"].unique())
log("Final: " + str(N_GROUPS) + " groups, " + str(len(galah_cl)) + " stars")
counts = galah_cl["chem_group"].value_counts().sort_index()
for g, n in counts.items():
    log("  Group " + str(g) + ": N=" + str(n))

log("Computing per-group statistics...")
group_stats = []
for g in group_ids:
    grp     = galah_cl[galah_cl["chem_group"] == g]
    co_vals = grp["C_O"].values
    w       = 1.0 / np.maximum(grp["sigma_CO"].values ** 2, 1e-6)
    w_mean  = np.sum(co_vals * w) / np.sum(w)
    w_std   = np.sqrt(np.sum(w * (co_vals - w_mean)**2) / np.sum(w))
    intrinsic = np.sqrt(max(0.0, w_std**2 - np.mean(grp["sigma_CO"].values**2)))
    row = {
        "group": g, "N": len(grp),
        "C_O_mean": round(w_mean, 4), "C_O_std": round(w_std, 4),
        "intrinsic_scatter": round(intrinsic, 4),
        "feh_mean": round(grp["fe_h"].mean(), 3),
        "feh_std":  round(grp["fe_h"].std(),  3),
        "mg_mean":  round(grp["mg_fe"].mean(), 3) if "mg_fe" in grp.columns else np.nan,
        "umap1_mean": round(grp["umap1"].mean(), 3),
        "umap2_mean": round(grp["umap2"].mean(), 3),
    }
    if has_kin:
        row["rv_mean"]       = round(grp["rv_gaia_dr3"].mean(), 2)
        row["rv_std"]        = round(grp["rv_gaia_dr3"].std(),  2)
        row["parallax_mean"] = round(grp["parallax"].mean(),    3)
    group_stats.append(row)

stats_df = pd.DataFrame(group_stats)
show_cols = ["group","N","C_O_mean","C_O_std","intrinsic_scatter","feh_mean","feh_std"]
if "mg_mean" in stats_df.columns:
    show_cols.append("mg_mean")
log("Group statistics:\n" + stats_df[show_cols].to_string(index=False))

groups_co = [galah_cl[galah_cl["chem_group"]==g]["C_O"].values for g in group_ids]
kw_stat,  kw_p  = kruskal(*groups_co)
lev_stat, lev_p = levene(*groups_co)
log("Kruskal-Wallis (C/O) H=" + str(round(kw_stat,3)) + "  p=" + str(round(kw_p,6)))
log("Levene (variance)    W=" + str(round(lev_stat,3)) + "  p=" + str(round(lev_p,6)))

within_scatter  = stats_df["C_O_std"].mean()
between_scatter = stats_df["C_O_mean"].std()
CCR = between_scatter / within_scatter if within_scatter > 0 else 0
log("CCR=" + str(round(CCR,3)) +
    "  within=" + str(round(within_scatter,4)) +
    "  between=" + str(round(between_scatter,4)))

KCR = None
if has_kin:
    groups_rv = [galah_cl[galah_cl["chem_group"]==g]["rv_gaia_dr3"].values for g in group_ids]
    kw_rv_stat, kw_rv_p = kruskal(*groups_rv)
    within_rv  = stats_df["rv_std"].mean()
    between_rv = stats_df["rv_mean"].std()
    KCR = between_rv / within_rv if within_rv > 0 else 0
    log("KCR=" + str(round(KCR,3)) +
        "  within_rv=" + str(round(within_rv,2)) + " km/s" +
        "  between_rv=" + str(round(between_rv,2)) + " km/s")
    log("KW(RV) H=" + str(round(kw_rv_stat,3)) + "  p=" + str(round(kw_rv_p,6)))
    log("RESULT: " + ("KCR>1 SUPPORTS CCT" if KCR > 1 else "KCR<1 kinematics mixed"))

n_coherent = (stats_df["C_O_std"] < 0.05).sum()
log("Coherent groups (<0.05 dex): " + str(n_coherent) + "/" + str(N_GROUPS))

cmap   = plt.cm.tab20
n_tab  = min(N_GROUPS, 20)
colors = [cmap(i / max(n_tab, 1)) for i in range(n_tab)]
if N_GROUPS > 20:
    extra  = plt.cm.tab20b(np.linspace(0, 1, N_GROUPS - 20))
    colors = colors + list(extra)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.patch.set_facecolor("white")

ax1 = axes[0, 0]
if HAS_HDBSCAN:
    noise = galah[galah["chem_group"] == -1]
    if len(noise) > 0:
        ax1.scatter(noise["umap1"], noise["umap2"],
            s=0.5, alpha=0.08, color="lightgray", rasterized=True, label="noise")
for idx, g in enumerate(group_ids):
    grp = galah_cl[galah_cl["chem_group"] == g]
    ax1.scatter(grp["umap1"], grp["umap2"],
        s=1.0, alpha=0.3, color=colors[idx % len(colors)], rasterized=True,
        label="G"+str(g))
ax1.set_xlabel("UMAP 1", fontsize=11); ax1.set_ylabel("UMAP 2", fontsize=11)
ax1.set_title("UMAP Chemistry Embedding\n(coloured by cluster)", fontsize=11)
if N_GROUPS <= 12:
    ax1.legend(markerscale=4, fontsize=7, ncol=2)
ax1.grid(True, alpha=0.1)

ax2 = axes[0, 1]
for idx, g in enumerate(group_ids):
    grp = galah_cl[galah_cl["chem_group"] == g]
    ax2.scatter(grp["fe_h"], grp["C_O"],
        s=1.5, alpha=0.25, color=colors[idx % len(colors)], rasterized=True)
ax2.axhline(SOLAR_CO, color="darkorange", linestyle=":", linewidth=1.5, label="Solar C/O")
ax2.set_xlabel("[Fe/H]", fontsize=11); ax2.set_ylabel("C/O", fontsize=11)
ax2.set_title("[Fe/H] vs C/O by Cluster", fontsize=11)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.12)

ax3 = axes[1, 0]
bar_c = [colors[i % len(colors)] for i in range(N_GROUPS)]
ax3.bar(range(N_GROUPS), stats_df["C_O_std"].values,
    color=bar_c, alpha=0.8, edgecolor="white", linewidth=0.5)
ax3.axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="Threshold 0.05 dex")
ax3.axhline(within_scatter, color="navy", linestyle=":", linewidth=1.5,
    label="Mean=" + str(round(within_scatter, 3)))
ax3.set_xlabel("Group", fontsize=11); ax3.set_ylabel("C/O Std Dev", fontsize=11)
ax3.set_title("Within-Group C/O Scatter\n(CCT coherence, T6b)", fontsize=11)
ax3.set_xticks(range(N_GROUPS))
ax3.set_xticklabels([str(g) for g in group_ids], fontsize=max(4, 8 - N_GROUPS//5))
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.12, axis="y")

ax4 = axes[1, 1]
if has_kin and KCR is not None:
    ax4.bar(range(N_GROUPS), stats_df["rv_std"].values,
        color=bar_c, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax4.axhline(within_rv, color="navy", linestyle=":", linewidth=1.5,
        label="Mean=" + str(round(within_rv, 1)) + " km/s")
    ax4.set_ylabel("RV Std Dev (km/s)", fontsize=11)
    ax4.set_title("Within-Group RV Scatter\n(KCR=" + str(round(KCR, 3)) + ")", fontsize=11)
    ax4.set_xticks(range(N_GROUPS))
    ax4.set_xticklabels([str(g) for g in group_ids], fontsize=max(4, 8 - N_GROUPS//5))
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.12, axis="y")
else:
    ax4.text(0.5, 0.5, "No kinematic data", ha="center", va="center",
        transform=ax4.transAxes, fontsize=12, color="gray")
ax4.set_xlabel("Group", fontsize=11)

method  = ("UMAP+" + ("HDBSCAN" if HAS_HDBSCAN else "KMeans")) if HAS_UMAP else ("HDBSCAN" if HAS_HDBSCAN else "KMeans")
kw_str  = "p<0.0001" if kw_p < 0.0001 else "p=" + str(round(kw_p, 4))
kcr_str = "KCR=" + str(round(KCR, 3)) if KCR is not None else "no kinematics"
plt.suptitle(
    "T6b Chemistry-First Cluster Coherence  |  Certan (2026)  |  " + method + "\n"
    "KW(C/O) " + kw_str + "  |  CCR=" + str(round(CCR, 3)) +
    "  |  " + kcr_str + "  |  N=" + str(len(galah_cl)) +
    " stars  |  " + str(n_coherent) + "/" + str(N_GROUPS) + " groups <0.05 dex",
    fontsize=9.5, style="italic", color="gray", y=1.01
)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log("Plot saved: " + OUTPUT_PLOT)
stats_df.to_csv(OUTPUT_CSV, index=False)
log("Results saved: " + OUTPUT_CSV)
log("=== T6b SUMMARY ===")
log("Method: " + method)
log("N groups: " + str(N_GROUPS) + "  |  N stars: " + str(len(galah_cl)))
log("CCR=" + str(round(CCR,3)) + ("  [SIGNAL]" if CCR > 1 else "  [NO SIGNAL]"))
if KCR is not None:
    log("KCR=" + str(round(KCR,3)) + ("  [SUPPORTS CCT]" if KCR > 1 else "  [MIXED]"))
log("Coherent groups (<0.05 dex): " + str(n_coherent) + "/" + str(N_GROUPS))
log("DONE")

# ── Coherent group deep-dive ──────────────────────────────────────────────────
coherent = stats_df[stats_df["C_O_std"] < 0.05].copy()
coherent = coherent.sort_values("C_O_std")
log("=== 42 COHERENT GROUPS (C/O std < 0.05 dex) ===")
log(coherent[["group","N","C_O_mean","C_O_std","intrinsic_scatter",
              "feh_mean","feh_std","rv_mean","rv_std"]].to_string(index=False))

# Save coherent group member stars
coherent_ids = coherent["group"].tolist()
coherent_stars = galah_cl[galah_cl["chem_group"].isin(coherent_ids)].copy()
coherent_stars.to_csv("t6b_coherent_stars.csv", index=False)
log("Coherent star members saved: t6b_coherent_stars.csv  N=" + str(len(coherent_stars)))

# Flag any group with rv_std < 5 km/s (kinematically cold — moving group candidate)
if "rv_std" in coherent.columns:
    cold = coherent[coherent["rv_std"] < 5.0]
    log("Kinematically cold coherent groups (RV std < 5 km/s): " + str(len(cold)))
    if len(cold) > 0:
        log(cold[["group","N","C_O_mean","C_O_std","feh_mean","rv_mean","rv_std"]].to_string(index=False))

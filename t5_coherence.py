import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import kruskal, f_oneway
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

GALAH_FILE = "galah_dr4_allstar_240705.fits"
OUTPUT_CSV = "t5_cluster_coherence.csv"
OUTPUT_PLOT = "t5_cluster_plot.png"
SNR_MIN = 30
N_CLUSTERS = 10
SOLAR_CO = 0.549

def log(msg):
    print("[INFO] " + str(msg))

log("Loading GALAH DR4...")
galah = Table.read(GALAH_FILE).to_pandas()
log("Raw rows: " + str(len(galah)))

# Quality cuts
galah = galah[
    (galah["flag_sp"] == 0) &
    (galah["flag_fe_h"] == 0) &
    (galah["flag_c_fe"] == 0) &
    (galah["flag_o_fe"] == 0) &
    (galah["snr_px_ccd3"] > SNR_MIN)
].copy()
log("After quality cuts: " + str(len(galah)))

# Compute C/O for each star
galah["delta_co"] = galah["c_fe"] - galah["o_fe"]
galah["C_O"] = (10.0 ** galah["delta_co"]) * SOLAR_CO
galah["sigma_CO"] = galah["C_O"] * np.log(10) * np.sqrt(
    galah["e_c_fe"]**2 + galah["e_o_fe"]**2
)

# Require valid kinematics and C/O
kin_cols = ["rv_gaia_dr3","parallax"]
galah = galah.dropna(subset=kin_cols + ["C_O","sigma_CO"]).copy()
galah = galah[
    (galah["parallax"] > 0.5) &   # within ~2 kpc
    (galah["C_O"] > 0.05) &
    (galah["C_O"] < 2.0) &
    (galah["sigma_CO"] > 0) &
    (galah["sigma_CO"] < 0.3)
].copy()
log("Stars with kinematics + C/O: " + str(len(galah)))

# Build kinematic feature matrix: RV, pmra, pmdec, parallax
X = galah[kin_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log("Running KMeans with N_CLUSTERS=" + str(N_CLUSTERS) + "...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
galah["kin_group"] = km.fit_predict(X_scaled)

log("Cluster sizes:")
counts = galah["kin_group"].value_counts().sort_index()
for g, n in counts.items():
    log("  Group " + str(g) + ": N=" + str(n))

# Compute within-group C/O statistics
log("Computing within-group C/O scatter...")
group_stats = []
for g in range(N_CLUSTERS):
    grp = galah[galah["kin_group"] == g]
    co_vals = grp["C_O"].values
    w = 1.0 / (grp["sigma_CO"].values ** 2)
    w_mean = np.sum(co_vals * w) / np.sum(w)
    w_std = np.sqrt(np.sum(w * (co_vals - w_mean)**2) / np.sum(w))
    intrinsic_scatter = np.sqrt(max(0, w_std**2 - np.mean(grp["sigma_CO"].values**2)))
    group_stats.append({
        "group": g,
        "N": len(grp),
        "C_O_mean": round(w_mean, 4),
        "C_O_std": round(w_std, 4),
        "intrinsic_scatter": round(intrinsic_scatter, 4),
        "rv_mean": round(grp["rv_gaia_dr3"].mean(), 2),
        "parallax_mean": round(grp["parallax"].mean(), 3),
    })

stats_df = pd.DataFrame(group_stats)
log("Group C/O statistics:")
log(stats_df[["group","N","C_O_mean","C_O_std","intrinsic_scatter"]].to_string(index=False))

# Statistical tests
groups_co = [galah[galah["kin_group"]==g]["C_O"].values for g in range(N_CLUSTERS)]
kw_stat, kw_p = kruskal(*groups_co)
log("Kruskal-Wallis H=" + str(round(kw_stat,3)) + "  p=" + str(round(kw_p,6)))

within_scatter = stats_df["C_O_std"].mean()
between_scatter = stats_df["C_O_mean"].std()
coherence_ratio = between_scatter / within_scatter
log("Mean within-group scatter: " + str(round(within_scatter,4)) + " dex")
log("Between-group scatter: " + str(round(between_scatter,4)) + " dex")
log("Coherence ratio (between/within): " + str(round(coherence_ratio,3)))

coherence_threshold = 0.05
n_coherent = (stats_df["C_O_std"] < coherence_threshold).sum()
log("Groups with within-scatter < " + str(coherence_threshold) + ": " + str(n_coherent) + "/" + str(N_CLUSTERS))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.patch.set_facecolor("white")

cmap = plt.cm.tab10
colors = [cmap(i/N_CLUSTERS) for i in range(N_CLUSTERS)]

# Panel 1: C/O distribution per group
ax1 = axes[0]
positions = list(range(N_CLUSTERS))
bp = ax1.boxplot([galah[galah["kin_group"]==g]["C_O"].values for g in range(N_CLUSTERS)],
    positions=positions, patch_artist=True,
    medianprops=dict(color="white", linewidth=1.5),
    whiskerprops=dict(linewidth=0.8),
    capprops=dict(linewidth=0.8),
    flierprops=dict(marker=".", markersize=2, alpha=0.3))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax1.axhline(SOLAR_CO, color="darkorange", linestyle=":", linewidth=1.5,
    label="Solar C/O=0.549")
ax1.set_xlabel("Kinematic Group", fontsize=11)
ax1.set_ylabel("C/O", fontsize=11)
ax1.set_title("C/O Distribution per\nKinematic Group", fontsize=11)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.15, axis="y")
ax1.set_xticks(positions)

# Panel 2: Within vs between scatter
ax2 = axes[1]
bar_colors = [colors[i] for i in range(N_CLUSTERS)]
bars = ax2.bar(stats_df["group"], stats_df["C_O_std"],
    color=bar_colors, alpha=0.8, edgecolor="white", linewidth=0.5)
ax2.axhline(coherence_threshold, color="red", linestyle="--", linewidth=1.5,
    label="Coherence threshold (0.05)")
ax2.axhline(within_scatter, color="navy", linestyle=":", linewidth=1.5,
    label="Mean within scatter=" + str(round(within_scatter,3)))
ax2.set_xlabel("Kinematic Group", fontsize=11)
ax2.set_ylabel("C/O Standard Deviation", fontsize=11)
ax2.set_title("Within-Group C/O Scatter\n(CCT coherence test)", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.15, axis="y")

# Panel 3: RV vs C/O mean per group
ax3 = axes[2]
sc = ax3.scatter(stats_df["rv_mean"], stats_df["C_O_mean"],
    c=list(range(N_CLUSTERS)), cmap="tab10",
    s=stats_df["N"]/galah["kin_group"].value_counts().max()*400+50,
    alpha=0.85, edgecolors="black", linewidths=0.6, zorder=5)
ax3.errorbar(stats_df["rv_mean"], stats_df["C_O_mean"],
    yerr=stats_df["C_O_std"], fmt="none",
    ecolor="gray", elinewidth=1, capsize=3, alpha=0.6)
for _, row in stats_df.iterrows():
    ax3.annotate("G"+str(int(row["group"])),
        xy=(row["rv_mean"], row["C_O_mean"]),
        xytext=(3, 3), textcoords="offset points",
        fontsize=7.5, color="black")
ax3.axhline(SOLAR_CO, color="darkorange", linestyle=":", linewidth=1.5,
    label="Solar C/O=0.549")
ax3.set_xlabel("Mean RV - Gaia DR3 (km/s)", fontsize=11)
ax3.set_ylabel("Mean C/O", fontsize=11)
ax3.set_title("Group C/O vs Kinematics\n(size = N stars)", fontsize=11)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.15)

kw_str = "p=" + str(round(kw_p,4)) if kw_p >= 0.0001 else "p<0.0001"
plt.suptitle(
    "T5 Birth Cluster Coherence Test  |  Certan (2026)\n"
    "Kruskal-Wallis " + kw_str + "  |  "
    "Coherence ratio=" + str(round(coherence_ratio,2)) + "  |  "
    "N=" + str(len(galah)) + " stars  |  "
    + str(n_coherent) + "/" + str(N_CLUSTERS) + " groups within 0.05 dex",
    fontsize=9.5, style="italic", color="gray", y=1.02
)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log("Plot saved: " + OUTPUT_PLOT)

stats_df.to_csv(OUTPUT_CSV, index=False)
log("Results saved: " + OUTPUT_CSV)

log("=== T5 SUMMARY ===")
log("Kruskal-Wallis p=" + str(round(kw_p,6)))
if kw_p < 0.001:
    log("RESULT: Groups are HIGHLY SIGNIFICANTLY different in C/O (p<0.001)")
    log("INTERPRETATION: Chemical tagging of birth clusters IS detectable")
elif kw_p < 0.05:
    log("RESULT: Groups significantly different in C/O (p<0.05)")
else:
    log("RESULT: No significant difference between groups")
    log("INTERPRETATION: C/O coherence not detected at this clustering resolution")
log("Within-group scatter: " + str(round(within_scatter,4)))
log("Between-group scatter: " + str(round(between_scatter,4)))
log("Coherent groups (<0.05 dex): " + str(n_coherent) + "/" + str(N_CLUSTERS))
log("DONE")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord, Galactocentric
from astropy.coordinates import galactocentric_frame_defaults
import astropy.units as u
from scipy.stats import kruskal, levene
from sklearn.preprocessing import StandardScaler
from astroquery.gaia import Gaia

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

GALAH_FILE  = "galah_dr4_allstar_240705.fits"
AGE_MAX     = 1.0       # Gyr — primary cut
AGE_YOUNG   = 0.5       # Gyr — secondary cut for ultra-young subset
SNR_MIN     = 30
SOLAR_CO    = 0.549
UMAP_N_NEIGHBORS  = 20  # smaller = more local structure (better for young stars)
UMAP_MIN_DIST     = 0.02
UMAP_N_COMPONENTS = 2
UMAP_RANDOM_STATE = 42
HDBSCAN_MIN_CLUSTER = 20
HDBSCAN_MIN_SAMPLES = 3

def log(msg):
    print("[INFO] " + str(msg), flush=True)

# ── Load ──────────────────────────────────────────────────────────────────────
log("Loading GALAH DR4...")
galah = Table.read(GALAH_FILE).to_pandas()
log("Raw rows: " + str(len(galah)))

# ── Quality + age cuts ────────────────────────────────────────────────────────
galah = galah[
    (galah["flag_sp"]     == 0) &
    (galah["flag_fe_h"]   == 0) &
    (galah["flag_c_fe"]   == 0) &
    (galah["flag_o_fe"]   == 0) &
    (galah["snr_px_ccd3"] >  SNR_MIN) &
    (galah["age"].notna()) &
    (galah["age"]         <= AGE_MAX)
].copy()
log("After quality + age<=" + str(AGE_MAX) + " Gyr cuts: " + str(len(galah)))

n_ultra = (galah["age"] <= AGE_YOUNG).sum()
log("Of which age<=" + str(AGE_YOUNG) + " Gyr: " + str(n_ultra))

# ── Chemistry ─────────────────────────────────────────────────────────────────
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
log("Clean chemistry: " + str(len(galah)))

galah["delta_co"] = galah["c_fe"] - galah["o_fe"]
galah["C_O"]      = (10.0 ** galah["delta_co"]) * SOLAR_CO
galah["sigma_CO"] = galah["C_O"] * np.log(10) * np.sqrt(
    galah["e_c_fe"]**2 + galah["e_o_fe"]**2)

galah = galah[
    (galah["C_O"] > 0.05) & (galah["C_O"] < 2.0) &
    (galah["sigma_CO"] > 0) & (galah["sigma_CO"] < 0.3)
].copy()
log("After C/O sanity filter: " + str(len(galah)))

# ── Kinematic filter ──────────────────────────────────────────────────────────
KIN_COLS = ["rv_gaia_dr3", "parallax"]
has_kin  = all(c in galah.columns for c in KIN_COLS)
if has_kin:
    galah = galah.dropna(subset=KIN_COLS).copy()
    galah = galah[galah["parallax"] > 0.5].copy()
    log("After kinematic filter (parallax>0.5): " + str(len(galah)))

# ── UMAP ──────────────────────────────────────────────────────────────────────
X_chem   = galah[available_chem].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_chem)

if HAS_UMAP:
    log("Running UMAP (n_neighbors=" + str(UMAP_N_NEIGHBORS) +
        ", min_dist=" + str(UMAP_MIN_DIST) + ")...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        metric="euclidean",
        random_state=UMAP_RANDOM_STATE,
        low_memory=False,
        verbose=False
    )
    X_embedded = reducer.fit_transform(X_scaled)
    galah["umap1"] = X_embedded[:, 0]
    galah["umap2"] = X_embedded[:, 1]
    log("UMAP done.")
else:
    log("WARNING: umap-learn not installed.")
    X_embedded   = X_scaled
    galah["umap1"] = X_scaled[:, 0]
    galah["umap2"] = X_scaled[:, 1]

# ── HDBSCAN ───────────────────────────────────────────────────────────────────
if HAS_HDBSCAN:
    log("Running HDBSCAN (min_cluster=" + str(HDBSCAN_MIN_CLUSTER) +
        ", min_samples=" + str(HDBSCAN_MIN_SAMPLES) + ")...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )
    galah["chem_group"] = clusterer.fit_predict(X_embedded)
    n_noise = (galah["chem_group"] == -1).sum()
    n_clust = galah["chem_group"].max() + 1
    log("HDBSCAN: " + str(n_clust) + " clusters | " +
        str(n_noise) + " noise (" +
        str(round(100 * n_noise / len(galah), 1)) + "%)")
    galah_cl = galah[galah["chem_group"] >= 0].copy()
else:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=15, random_state=42, n_init=20)
    galah["chem_group"] = km.fit_predict(X_embedded)
    galah_cl = galah.copy()

N_GROUPS  = galah_cl["chem_group"].nunique()
group_ids = sorted(galah_cl["chem_group"].unique())
log("Final: " + str(N_GROUPS) + " groups, " + str(len(galah_cl)) + " clustered stars")

# ── Per-group stats ───────────────────────────────────────────────────────────
group_stats = []
for g in group_ids:
    grp     = galah_cl[galah_cl["chem_group"] == g]
    co_vals = grp["C_O"].values
    w       = 1.0 / np.maximum(grp["sigma_CO"].values**2, 1e-6)
    w_mean  = np.sum(co_vals * w) / np.sum(w)
    w_std   = np.sqrt(np.sum(w * (co_vals - w_mean)**2) / np.sum(w))
    intrinsic = np.sqrt(max(0.0, w_std**2 - np.mean(grp["sigma_CO"].values**2)))
    row = {
        "group": g, "N": len(grp),
        "C_O_mean": round(w_mean, 4), "C_O_std": round(w_std, 4),
        "intrinsic_scatter": round(intrinsic, 4),
        "feh_mean": round(grp["fe_h"].mean(), 3),
        "feh_std":  round(grp["fe_h"].std(),  3),
        "age_mean": round(grp["age"].mean(),  3),
        "age_std":  round(grp["age"].std(),   3),
    }
    if has_kin:
        row["rv_mean"] = round(grp["rv_gaia_dr3"].mean(), 2)
        row["rv_std"]  = round(grp["rv_gaia_dr3"].std(),  2)
    group_stats.append(row)

stats_df = pd.DataFrame(group_stats)
show = ["group","N","C_O_mean","C_O_std","intrinsic_scatter","feh_mean","age_mean","age_std"]
if has_kin:
    show += ["rv_mean","rv_std"]
log("Group statistics:\n" + stats_df[show].to_string(index=False))

# ── CCR / KCR ─────────────────────────────────────────────────────────────────
groups_co      = [galah_cl[galah_cl["chem_group"]==g]["C_O"].values for g in group_ids]
kw_stat, kw_p  = kruskal(*groups_co)
within_scatter  = stats_df["C_O_std"].mean()
between_scatter = stats_df["C_O_mean"].std()
CCR = between_scatter / within_scatter if within_scatter > 0 else 0
log("KW(C/O) H=" + str(round(kw_stat,3)) + "  p=" + str(round(kw_p,6)))
log("CCR=" + str(round(CCR,3)) +
    "  within=" + str(round(within_scatter,4)) +
    "  between=" + str(round(between_scatter,4)))

KCR = None
if has_kin:
    groups_rv      = [galah_cl[galah_cl["chem_group"]==g]["rv_gaia_dr3"].values for g in group_ids]
    kw_rv, kw_rv_p = kruskal(*groups_rv)
    within_rv      = stats_df["rv_std"].mean()
    between_rv     = stats_df["rv_mean"].std()
    KCR = between_rv / within_rv if within_rv > 0 else 0
    log("KCR=" + str(round(KCR,3)) +
        "  within_rv=" + str(round(within_rv,2)) +
        "  between_rv=" + str(round(between_rv,2)) + " km/s")

n_coherent = (stats_df["C_O_std"] < 0.05).sum()
log("Coherent groups (<0.05 dex): " + str(n_coherent) + "/" + str(N_GROUPS))

# ── Gaia PM crossmatch ────────────────────────────────────────────────────────
log("Fetching Gaia DR3 proper motions...")
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1

from astropy.table import Table as AstroTable
upload_table = AstroTable({
    "ra":     galah_cl["ra"].values,
    "dec":    galah_cl["dec"].values,
    "row_id": np.arange(len(galah_cl))
})

query = """
SELECT u.row_id, g.source_id, g.pmra, g.pmdec,
       g.pmra_error, g.pmdec_error, g.ruwe
FROM TAP_UPLOAD.input AS u
JOIN gaiadr3.gaia_source AS g
ON 1=CONTAINS(
    POINT('ICRS', u.ra, u.dec),
    CIRCLE('ICRS', g.ra, g.dec, 0.000278)
)
"""
job    = Gaia.launch_job_async(query, upload_resource=upload_table,
                                upload_table_name="input", verbose=False)
result = job.get_results().to_pandas()
result = result.sort_values("ruwe").drop_duplicates("row_id", keep="first")
log("Gaia matches: " + str(len(result)))

galah_cl = galah_cl.reset_index(drop=True)
galah_cl["row_id"] = np.arange(len(galah_cl))
galah_cl = galah_cl.merge(
    result[["row_id","pmra","pmdec","pmra_error","pmdec_error","ruwe"]],
    on="row_id", how="left"
).drop(columns=["row_id"])
log("Stars with pmra: " + str(galah_cl["pmra"].notna().sum()))

# ── UVW ───────────────────────────────────────────────────────────────────────
have_6d = galah_cl["pmra"].notna() & galah_cl["parallax"].notna()
g6d = galah_cl[have_6d].copy()
log("Stars with full 6D kinematics: " + str(len(g6d)))

if len(g6d) > 0:
    galactocentric_frame_defaults.set("v4.0")
    coords = SkyCoord(
        ra=g6d["ra"].values * u.deg,
        dec=g6d["dec"].values * u.deg,
        distance=(1000.0 / g6d["parallax"].values) * u.pc,
        pm_ra_cosdec=g6d["pmra"].values * u.mas/u.yr,
        pm_dec=g6d["pmdec"].values * u.mas/u.yr,
        radial_velocity=g6d["rv_gaia_dr3"].values * u.km/u.s,
        frame="icrs"
    )
    gc = coords.galactocentric
    g6d["U"] = gc.v_x.to(u.km/u.s).value
    g6d["V"] = gc.v_y.to(u.km/u.s).value
    g6d["W"] = gc.v_z.to(u.km/u.s).value

    uvw_rows = []
    for g in group_ids:
        grp = g6d[g6d["chem_group"] == g]
        if len(grp) < 3:
            continue
        uvw_rows.append({
            "group":     g,
            "N_uvw":     len(grp),
            "U_mean":    round(grp["U"].mean(), 2),
            "U_std":     round(grp["U"].std(),  2),
            "V_mean":    round(grp["V"].mean(), 2),
            "V_std":     round(grp["V"].std(),  2),
            "W_mean":    round(grp["W"].mean(), 2),
            "W_std":     round(grp["W"].std(),  2),
            "sigma_tot": round(np.sqrt(grp["U"].std()**2 +
                                       grp["V"].std()**2 +
                                       grp["W"].std()**2), 2),
            "age_mean":  round(grp["age"].mean(), 3),
            "C_O_std":   round(grp["C_O"].std(),  4),
            "C_O_mean":  round(grp["C_O"].mean(), 4),
            "feh_mean":  round(grp["fe_h"].mean(), 3),
        })

    uvw_df = pd.DataFrame(uvw_rows).sort_values("sigma_tot")
    log("UVW dispersions (sorted by sigma_tot):\n" + uvw_df.to_string(index=False))
    uvw_df.to_csv("t8_uvw_summary.csv", index=False)

    cold  = uvw_df[uvw_df["sigma_tot"] < 20.0]
    ultra = uvw_df[uvw_df["sigma_tot"] < 10.0]
    log("Kinematically cold (sigma_tot<20): " + str(len(cold)))
    if len(cold) > 0:
        log(cold.to_string(index=False))
    log("Ultra-cold (sigma_tot<10): " + str(len(ultra)))
    if len(ultra) > 0:
        log(ultra.to_string(index=False))

    g6d.to_csv("t8_young_stars_uvw.csv", index=False)
    log("Full table saved: t8_young_stars_uvw.csv")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.patch.set_facecolor("white")
cmap   = plt.cm.tab20
colors = [cmap(i / max(N_GROUPS, 1)) for i in range(min(N_GROUPS, 20))]
if N_GROUPS > 20:
    colors += list(plt.cm.tab20b(np.linspace(0, 1, N_GROUPS - 20)))

ax1 = axes[0]
noise = galah[galah["chem_group"] == -1] if HAS_HDBSCAN else pd.DataFrame()
if len(noise) > 0:
    ax1.scatter(noise["umap1"], noise["umap2"],
        s=0.5, alpha=0.08, color="lightgray", rasterized=True)
for idx, g in enumerate(group_ids):
    grp = galah_cl[galah_cl["chem_group"] == g]
    ax1.scatter(grp["umap1"], grp["umap2"],
        s=3, alpha=0.4, color=colors[idx % len(colors)], rasterized=True)
ax1.set_xlabel("UMAP 1"); ax1.set_ylabel("UMAP 2")
ax1.set_title("UMAP Embedding\n(age <= " + str(AGE_MAX) + " Gyr)", fontsize=11)
ax1.grid(True, alpha=0.1)

ax2 = axes[1]
for idx, g in enumerate(group_ids):
    grp = galah_cl[galah_cl["chem_group"] == g]
    ax2.scatter(grp["fe_h"], grp["C_O"],
        s=3, alpha=0.3, color=colors[idx % len(colors)], rasterized=True)
ax2.axhline(SOLAR_CO, color="darkorange", linestyle=":", linewidth=1.5, label="Solar C/O")
ax2.set_xlabel("[Fe/H]"); ax2.set_ylabel("C/O")
ax2.set_title("[Fe/H] vs C/O\n(young stars)", fontsize=11)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.12)

ax3 = axes[2]
sc = ax3.scatter(galah_cl["fe_h"], galah_cl["C_O"],
    c=galah_cl["age"], cmap="plasma_r", s=2, alpha=0.3,
    vmin=0, vmax=AGE_MAX, rasterized=True)
plt.colorbar(sc, ax=ax3, label="Age (Gyr)")
ax3.set_xlabel("[Fe/H]"); ax3.set_ylabel("C/O")
ax3.set_title("[Fe/H] vs C/O\n(coloured by age)", fontsize=11)
ax3.grid(True, alpha=0.12)

method = ("UMAP+" + ("HDBSCAN" if HAS_HDBSCAN else "KMeans")) if HAS_UMAP else "KMeans"
kcr_str = "KCR=" + str(round(KCR, 3)) if KCR is not None else "no kinematics"
plt.suptitle(
    "T8 Young Star Chemistry Coherence  |  Certan (2025)  |  age<=" +
    str(AGE_MAX) + " Gyr  |  " + method + "\n" +
    "CCR=" + str(round(CCR,3)) + "  |  " + kcr_str +
    "  |  N=" + str(len(galah_cl)) + " stars  |  " +
    str(n_coherent) + "/" + str(N_GROUPS) + " groups <0.05 dex",
    fontsize=9.5, style="italic", color="gray", y=1.01
)
plt.tight_layout()
plt.savefig("t8_young_cluster_plot.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log("Plot saved: t8_young_cluster_plot.png")
galah_cl.to_csv("t8_young_clustered.csv", index=False)
log("=== T8 SUMMARY ===")
log("Age cut: <=" + str(AGE_MAX) + " Gyr")
log("N groups: " + str(N_GROUPS) + "  |  N stars: " + str(len(galah_cl)))
log("CCR=" + str(round(CCR,3)) + ("  [SIGNAL]" if CCR > 1 else "  [NO SIGNAL]"))
if KCR is not None:
    log("KCR=" + str(round(KCR,3)) + ("  [SUPPORTS CCT]" if KCR > 1 else "  [MIXED]"))
log("Coherent groups (<0.05 dex): " + str(n_coherent) + "/" + str(N_GROUPS))
log("DONE")

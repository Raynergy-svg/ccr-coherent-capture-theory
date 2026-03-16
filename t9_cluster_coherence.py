import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import kruskal, levene
from astroquery.vizier import Vizier

def log(msg):
    print("[INFO] " + str(msg), flush=True)

GALAH_FILE           = "galah_dr4_allstar_240705.fits"
SOLAR_CO             = 0.549
SNR_MIN              = 30
MATCH_RADIUS_DEG     = 0.5
CO_COHERENCE_THRESH  = 0.05

log("Loading GALAH DR4...")
galah = Table.read(GALAH_FILE).to_pandas()
log("Raw rows: " + str(len(galah)))

galah = galah[
    (galah["flag_sp"]     == 0) &
    (galah["flag_fe_h"]   == 0) &
    (galah["flag_c_fe"]   == 0) &
    (galah["flag_o_fe"]   == 0) &
    (galah["snr_px_ccd3"] >  SNR_MIN)
].copy()
log("After quality cuts: " + str(len(galah)))

galah["delta_co"] = galah["c_fe"] - galah["o_fe"]
galah["C_O"]      = (10.0 ** galah["delta_co"]) * SOLAR_CO
galah["sigma_CO"] = galah["C_O"] * np.log(10) * np.sqrt(
    galah["e_c_fe"]**2 + galah["e_o_fe"]**2)
galah = galah[
    (galah["C_O"] > 0.05) & (galah["C_O"] < 2.0) &
    (galah["sigma_CO"] > 0) & (galah["sigma_CO"] < 0.3)
].copy()
log("After C/O filter: " + str(len(galah)))

log("Fetching Cantat-Gaudin 2020 from VizieR...")
Vizier.ROW_LIMIT = -1
try:
    catalogs = Vizier.get_catalogs("J/A+A/640/A1")
    cg20 = catalogs[0].to_pandas()
    log("CG20 clusters: " + str(len(cg20)))
    log("CG20 columns: " + str(list(cg20.columns)))
except Exception as e:
    log("VizieR failed: " + str(e))
    raise

ra_col   = next((c for c in cg20.columns if c.lower() in ["radeg","ra_icrs","ra"]),   None)
dec_col  = next((c for c in cg20.columns if c.lower() in ["dedeg","de_icrs","dec","de"]), None)
name_col = next((c for c in cg20.columns if c.lower() in ["cluster","name"]),          None)
pmra_col = next((c for c in cg20.columns if "pmra" in c.lower()),                      None)
pmdec_col= next((c for c in cg20.columns if "pmde" in c.lower() or "pmdec" in c.lower()), None)
age_col  = next((c for c in cg20.columns if "logage" in c.lower() or "log_age" in c.lower()), None)
dist_col = next((c for c in cg20.columns if "dist" in c.lower() or "plx" in c.lower()), None)
log("Mapped: RA=" + str(ra_col) + " Dec=" + str(dec_col) +
    " Name=" + str(name_col) + " pmRA=" + str(pmra_col) +
    " age=" + str(age_col))

rename = {}
if ra_col:   rename[ra_col]   = "ra_cl"
if dec_col:  rename[dec_col]  = "dec_cl"
if name_col: rename[name_col] = "cluster_name"
if pmra_col: rename[pmra_col] = "pmra_cl"
if pmdec_col:rename[pmdec_col]= "pmdec_cl"
if age_col:  rename[age_col]  = "logage_cl"
if dist_col: rename[dist_col] = "dist_cl"
cg20 = cg20.rename(columns=rename).dropna(subset=["ra_cl","dec_cl"]).copy()

if "logage_cl" in cg20.columns:
    cg20["age_gyr"] = 10.0 ** (cg20["logage_cl"].astype(float) - 9.0)
    log("Age range: " + str(round(cg20["age_gyr"].min(),3)) +
        " - " + str(round(cg20["age_gyr"].max(),2)) + " Gyr")

log("Clusters with positions: " + str(len(cg20)))

log("Spatial crossmatch GALAH <-> CG20 (r<" + str(MATCH_RADIUS_DEG) + " deg)...")
galah_coords = SkyCoord(ra=galah["ra"].values  * u.deg,
                        dec=galah["dec"].values * u.deg, frame="icrs")

matches = []
for i, cl in cg20.iterrows():
    cl_coord = SkyCoord(ra=cl["ra_cl"] * u.deg,
                        dec=cl["dec_cl"] * u.deg, frame="icrs")
    seps     = cl_coord.separation(galah_coords).deg
    in_field = np.where(seps < MATCH_RADIUS_DEG)[0]
    for idx in in_field:
        matches.append({"cl_idx": i, "galah_idx": idx,
                        "sep_deg": round(seps[idx], 4)})

match_df = pd.DataFrame(matches)
log("Field matches (positional): " + str(len(match_df)))
if len(match_df) == 0:
    log("ERROR: No matches. Check coordinate columns.")
    raise SystemExit(1)

galah_matched = galah.iloc[match_df["galah_idx"].values].reset_index(drop=True)
cg20_matched  = cg20.iloc[match_df["cl_idx"].values].reset_index(drop=True)

keep_galah = ["ra","dec","C_O","sigma_CO","fe_h","c_fe","o_fe","mg_fe","si_fe","al_fe"]
if "rv_gaia_dr3" in galah.columns: keep_galah.append("rv_gaia_dr3")
if "age"         in galah.columns: keep_galah.append("age")

keep_cg20 = ["cluster_name","ra_cl","dec_cl"]
for col in ["pmra_cl","pmdec_cl","age_gyr","logage_cl","dist_cl"]:
    if col in cg20_matched.columns: keep_cg20.append(col)

merged = pd.concat([
    galah_matched[keep_galah],
    cg20_matched[keep_cg20]
], axis=1)
merged["sep_deg"] = match_df["sep_deg"].values
log("Merged table: " + str(len(merged)) + " star-cluster pairs")

log("Computing per-cluster statistics...")
has_rv     = "rv_gaia_dr3" in merged.columns
cl_stats   = []
for clname in merged["cluster_name"].unique():
    grp = merged[merged["cluster_name"] == clname]
    if len(grp) < 3:
        continue
    co_vals = grp["C_O"].values
    w       = 1.0 / np.maximum(grp["sigma_CO"].values**2, 1e-6)
    w_mean  = np.sum(co_vals * w) / np.sum(w)
    w_std   = np.sqrt(np.sum(w * (co_vals - w_mean)**2) / np.sum(w))
    intrinsic = np.sqrt(max(0.0, w_std**2 - np.mean(grp["sigma_CO"].values**2)))
    row = {
        "cluster":           clname,
        "N":                 len(grp),
        "C_O_mean":          round(float(w_mean),    4),
        "C_O_std":           round(float(w_std),     4),
        "intrinsic_scatter": round(float(intrinsic), 4),
        "feh_mean":          round(float(grp["fe_h"].mean()), 3),
        "feh_std":           round(float(grp["fe_h"].std()),  3),
    }
    if "age_gyr"   in grp.columns: row["age_gyr"]  = round(float(grp["age_gyr"].iloc[0]),   3)
    if "logage_cl" in grp.columns: row["logage"]   = round(float(grp["logage_cl"].iloc[0]), 3)
    if has_rv:
        row["rv_mean"] = round(float(grp["rv_gaia_dr3"].mean()), 2)
        row["rv_std"]  = round(float(grp["rv_gaia_dr3"].std()),  2)
    cl_stats.append(row)

cl_df = pd.DataFrame(cl_stats).sort_values("C_O_std")
log("Clusters with >=3 members: " + str(len(cl_df)))
log(cl_df.to_string(index=False))
cl_df.to_csv("t9_cluster_stats.csv", index=False)

log("=== CCT CORE TEST ===")
within_scatter  = cl_df["C_O_std"].mean()
between_scatter = cl_df["C_O_mean"].std()
CCR = between_scatter / within_scatter if within_scatter > 0 else 0
log("Within-cluster C/O scatter (mean): " + str(round(within_scatter,  4)))
log("Between-cluster C/O scatter (std): " + str(round(between_scatter, 4)))
log("CCR=" + str(round(CCR, 3)) +
    ("  [SIGNAL]" if CCR > 1 else "  [NO SIGNAL]"))

if len(cl_df) >= 2:
    groups_co = [merged[merged["cluster_name"]==c]["C_O"].values
                 for c in cl_df["cluster"].values if
                 len(merged[merged["cluster_name"]==c]) >= 2]
    if len(groups_co) >= 2:
        kw_stat, kw_p = kruskal(*groups_co)
        log("Kruskal-Wallis H=" + str(round(kw_stat,3)) +
            "  p=" + str(round(kw_p,6)))

n_coherent = (cl_df["C_O_std"] < CO_COHERENCE_THRESH).sum()
log("Coherent clusters (<0.05 dex): " + str(n_coherent) + "/" + str(len(cl_df)))
if n_coherent > 0:
    log(cl_df[cl_df["C_O_std"] < CO_COHERENCE_THRESH].to_string(index=False))

if "age_gyr" in cl_df.columns:
    young = cl_df[cl_df["age_gyr"] <  0.5]
    old   = cl_df[cl_df["age_gyr"] >= 0.5]
    log("Young (<0.5 Gyr) mean C/O std: " +
        str(round(young["C_O_std"].mean(), 4)) + "  N=" + str(len(young)))
    log("Older (>=0.5 Gyr) mean C/O std: " +
        str(round(old["C_O_std"].mean(),  4)) + "  N=" + str(len(old)))
    if len(young) > 0 and len(old) > 0:
        if young["C_O_std"].mean() < old["C_O_std"].mean():
            log("RESULT: Younger clusters more coherent -> CONSISTENT WITH CCT")
        else:
            log("RESULT: No age-coherence trend")

if has_rv and "rv_std" in cl_df.columns:
    rv_within  = cl_df["rv_std"].mean()
    rv_between = cl_df["rv_mean"].std()
    KCR = rv_between / rv_within if rv_within > 0 else 0
    log("RV KCR=" + str(round(KCR, 3)) +
        ("  [SUPPORTS CCT]" if KCR > 1 else "  [MIXED]"))

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.patch.set_facecolor("white")

ax1 = axes[0]
bar_colors = ["steelblue" if v < CO_COHERENCE_THRESH else "salmon"
              for v in cl_df["C_O_std"].values]
ax1.bar(np.arange(len(cl_df)), cl_df["C_O_std"].values,
    color=bar_colors, alpha=0.8, edgecolor="white")
ax1.axhline(CO_COHERENCE_THRESH, color="red", linestyle="--",
    linewidth=1.5, label="Threshold 0.05 dex")
ax1.axhline(within_scatter, color="navy", linestyle=":",
    linewidth=1.5, label="Mean=" + str(round(within_scatter, 3)))
ax1.set_xlabel("Open Cluster (sorted by C/O std)", fontsize=11)
ax1.set_ylabel("C/O Std Dev", fontsize=11)
ax1.set_title("Within-Cluster C/O Scatter\n(T9 CCT test)", fontsize=11)
ax1.set_xticks(np.arange(len(cl_df)))
ax1.set_xticklabels(cl_df["cluster"].values,
    rotation=90, fontsize=max(3, 7 - len(cl_df)//10))
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.12, axis="y")

ax2 = axes[1]
cmap     = plt.cm.tab20
cl_names = cl_df["cluster"].values
for idx, clname in enumerate(cl_names):
    grp = merged[merged["cluster_name"] == clname]
    ax2.scatter(grp["fe_h"], grp["C_O"],
        s=15, alpha=0.7,
        color=cmap((idx % 20) / 20),
        label=clname if idx < 10 else None,
        rasterized=True)
ax2.axhline(SOLAR_CO, color="darkorange", linestyle=":", linewidth=1.5,
    label="Solar C/O")
ax2.set_xlabel("[Fe/H]", fontsize=11)
ax2.set_ylabel("C/O", fontsize=11)
ax2.set_title("[Fe/H] vs C/O\n(by open cluster)", fontsize=11)
ax2.legend(fontsize=6, ncol=2)
ax2.grid(True, alpha=0.12)

ax3 = axes[2]
if "age_gyr" in cl_df.columns and cl_df["age_gyr"].notna().sum() >= 3:
    sc = ax3.scatter(cl_df["age_gyr"], cl_df["C_O_std"],
        c=cl_df["feh_mean"], cmap="coolwarm",
        s=cl_df["N"] * 4, alpha=0.75,
        edgecolors="white", linewidth=0.5)
    plt.colorbar(sc, ax=ax3, label="[Fe/H]")
    ax3.axhline(CO_COHERENCE_THRESH, color="red", linestyle="--",
        linewidth=1.5, label="Threshold 0.05")
    valid = cl_df[cl_df["age_gyr"].notna() & cl_df["C_O_std"].notna()]
    if len(valid) >= 3:
        z    = np.polyfit(valid["age_gyr"], valid["C_O_std"], 1)
        xfit = np.linspace(valid["age_gyr"].min(), valid["age_gyr"].max(), 100)
        ax3.plot(xfit, np.polyval(z, xfit), "k--", linewidth=1.2, alpha=0.6,
            label="slope=" + str(round(z[0], 5)))
    ax3.set_xlabel("Cluster Age (Gyr)", fontsize=11)
    ax3.set_ylabel("C/O Std Dev", fontsize=11)
    ax3.set_title("Age vs C/O Scatter\n(CCT: younger = more coherent?)", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.12)
else:
    ax3.text(0.5, 0.5, "No age data available", ha="center", va="center",
        transform=ax3.transAxes, fontsize=12, color="gray")

plt.suptitle(
    "T9 Open Cluster Chemical Coherence  |  Certan (2026)  |  Cantat-Gaudin 2020\n"
    "CCR=" + str(round(CCR, 3)) +
    "  |  N_clusters=" + str(len(cl_df)) +
    "  |  Coherent=" + str(n_coherent) + "/" + str(len(cl_df)) +
    "  |  Within scatter=" + str(round(within_scatter, 4)),
    fontsize=9.5, style="italic", color="gray", y=1.01
)
plt.tight_layout()
plt.savefig("t9_cluster_coherence_plot.png", dpi=300,
    bbox_inches="tight", facecolor="white")
plt.close()
log("Plot saved: t9_cluster_coherence_plot.png")

merged.to_csv("t9_matched_stars.csv", index=False)
log("Matched stars saved: t9_matched_stars.csv")
log("=== T9 SUMMARY ===")
log("Clusters with >=3 GALAH members: " + str(len(cl_df)))
log("CCR=" + str(round(CCR, 3)) + ("  [SIGNAL]" if CCR > 1 else "  [NO SIGNAL]"))
log("Coherent clusters (<0.05 dex): " + str(n_coherent) + "/" + str(len(cl_df)))
log("DONE")

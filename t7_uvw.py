import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
from astropy.coordinates import galactocentric_frame_defaults
import astropy.units as u

def log(msg):
    print("[INFO] " + str(msg), flush=True)

stars = pd.read_csv("t6b_coherent_stars_with_pm.csv")
log("Loaded: " + str(len(stars)) + " stars")

# Drop rows missing any kinematic input
kin_cols = ["ra","dec","parallax","pmra","pmdec","rv_gaia_dr3"]
stars = stars.dropna(subset=kin_cols).copy()
log("Stars with full 6D kinematics: " + str(len(stars)))

# Compute UVW using astropy
galactocentric_frame_defaults.set("v4.0")

coords = SkyCoord(
    ra=stars["ra"].values * u.deg,
    dec=stars["dec"].values * u.deg,
    distance=(1000.0 / stars["parallax"].values) * u.pc,
    pm_ra_cosdec=stars["pmra"].values * u.mas/u.yr,
    pm_dec=stars["pmdec"].values * u.mas/u.yr,
    radial_velocity=stars["rv_gaia_dr3"].values * u.km/u.s,
    frame="icrs"
)

gc = coords.galactocentric
stars["U"] = gc.v_x.to(u.km/u.s).value
stars["V"] = gc.v_y.to(u.km/u.s).value
stars["W"] = gc.v_z.to(u.km/u.s).value
stars["vtot"] = np.sqrt(stars["U"]**2 + stars["V"]**2 + stars["W"]**2)
log("UVW computed.")

# Per-group UVW statistics
group_ids = sorted(stars["chem_group"].unique())
rows = []
for g in group_ids:
    grp = stars[stars["chem_group"] == g]
    row = {
        "group":   g,
        "N":       len(grp),
        "U_mean":  round(grp["U"].mean(), 2),
        "U_std":   round(grp["U"].std(),  2),
        "V_mean":  round(grp["V"].mean(), 2),
        "V_std":   round(grp["V"].std(),  2),
        "W_mean":  round(grp["W"].mean(), 2),
        "W_std":   round(grp["W"].std(),  2),
        "sigma_tot": round(np.sqrt(grp["U"].std()**2 +
                                   grp["V"].std()**2 +
                                   grp["W"].std()**2), 2),
        "C_O_std": round(grp["C_O"].std(), 4),
        "feh_mean": round(grp["fe_h"].mean(), 3),
        "C_O_mean": round(grp["C_O"].mean(), 4),
    }
    rows.append(row)

uvw_df = pd.DataFrame(rows).sort_values("sigma_tot")
log("UVW dispersions (sorted by sigma_tot):")
log(uvw_df.to_string(index=False))

uvw_df.to_csv("t7_uvw_summary.csv", index=False)
log("Saved: t7_uvw_summary.csv")

# Flag kinematically cold groups — sigma_tot < 20 km/s
cold = uvw_df[uvw_df["sigma_tot"] < 20.0]
log("=== KINEMATICALLY COLD GROUPS (sigma_tot < 20 km/s): " + str(len(cold)) + " ===")
if len(cold) > 0:
    log(cold.to_string(index=False))

# Flag ultra-cold < 10 km/s — moving group / CCT candidates
ultra = uvw_df[uvw_df["sigma_tot"] < 10.0]
log("=== ULTRA-COLD (sigma_tot < 10 km/s): " + str(len(ultra)) + " ===")
if len(ultra) > 0:
    log(ultra.to_string(index=False))
else:
    log("None — showing top 5 coldest:")
    log(uvw_df.head(5).to_string(index=False))

stars.to_csv("t7_coherent_stars_uvw.csv", index=False)
log("Full star table with UVW saved: t7_coherent_stars_uvw.csv")
log("DONE")

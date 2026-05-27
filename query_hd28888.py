"""
Live archive query for HD 28888 = Gaia DR3 3312653647717790720.

Pulls:
  * Gaia DR3 gaia_source          -> astrometry, photometry, RV, RUWE
  * Gaia DR3 astrophysical_params -> GSP-Phot/GSP-Spec Teff/logg/[M/H],
                                     GSP-Spec [alpha/Fe] + element abundances,
                                     FLAME mass/age/radius/luminosity
  * GALAH DR4 mainstartable       -> Teff/logg/[Fe/H], age, mass, 30+ [X/Fe]
  * GALAH DR4 vacdynamicstable    -> Galactic kinematics (X,Y,Z,U,V,W,R,...)
  * GALAH DR4 vac3dnltelitable    -> 3D NLTE Li

Outputs:
  * hd28888_query_gaia.csv
  * hd28888_query_galah.csv
  * hd28888_query_galah_dynamics.csv
  * hd28888_query_galah_li.csv
  * hd28888_combined_profile.csv  (long-format key/value/source)
  * prints a structured human-readable report
"""
import io
import sys
import time
import requests
import pandas as pd

SOURCE_ID = 3312653647717790720
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
DC_TAP = "https://datacentral.org.au/vo/tap/sync"

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)


def tap_query(url, adql, label, retries=4):
    """Run a synchronous TAP query, return a pandas DataFrame (CSV format)."""
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(
                url,
                data={
                    "REQUEST": "doQuery",
                    "LANG": "ADQL",
                    "FORMAT": "csv",
                    "QUERY": adql,
                },
                timeout=120,
            )
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                print(f"[OK] {label}: {len(df)} row(s), {len(df.columns)} cols")
                return df
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
        wait = 2 ** (attempt + 1)
        print(f"[WARN] {label} attempt {attempt+1} failed ({last_err}); retry in {wait}s")
        time.sleep(wait)
    print(f"[FAIL] {label}: {last_err}")
    return pd.DataFrame()


# ---------------------------------------------------------------- Gaia DR3
gaia_src = tap_query(
    GAIA_TAP,
    f"""
    SELECT source_id, ra, dec, ref_epoch,
           parallax, parallax_error, parallax_over_error,
           pmra, pmra_error, pmdec, pmdec_error,
           radial_velocity, radial_velocity_error, rv_nb_transits,
           ruwe, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp,
           ag_gspphot, distance_gspphot,
           non_single_star, in_qso_candidates, in_galaxy_candidates
    FROM gaiadr3.gaia_source
    WHERE source_id = {SOURCE_ID}
    """,
    "Gaia DR3 gaia_source",
)

gaia_ap = tap_query(
    GAIA_TAP,
    f"""
    SELECT source_id,
           teff_gspphot, logg_gspphot, mh_gspphot,
           distance_gspphot, radius_gspphot, ag_gspphot,
           teff_gspspec, logg_gspspec, mh_gspspec, alphafe_gspspec,
           fem_gspspec, cafe_gspspec, mgfe_gspspec, sife_gspspec,
           sfe_gspspec, tife_gspspec, crfe_gspspec,
           nife_gspspec, nfe_gspspec, cefe_gspspec, ndfe_gspspec,
           zrfe_gspspec, flags_gspspec,
           mass_flame, age_flame, radius_flame, lum_flame, evolstage_flame
    FROM gaiadr3.astrophysical_parameters
    WHERE source_id = {SOURCE_ID}
    """,
    "Gaia DR3 astrophysical_parameters",
)

# ---------------------------------------------------------------- GALAH DR4
galah = tap_query(
    DC_TAP,
    f"""
    SELECT sobject_id, gaiadr3_source_id, tmass_id, ra, dec, survey_name,
           teff, e_teff, logg, e_logg, fe_h, e_fe_h, vmic, vsini, e_vsini,
           age, mass, lbol, ruwe, parallax, parallax_error,
           rv_gaia_dr3, e_rv_gaia_dr3, rv_comp_1, e_rv_comp_1,
           snr_px_ccd1, snr_px_ccd2, snr_px_ccd3, snr_px_ccd4,
           flag_sp, flag_fe_h, flag_red,
           c_fe, e_c_fe, flag_c_fe, o_fe, e_o_fe, flag_o_fe,
           na_fe, e_na_fe, flag_na_fe, mg_fe, e_mg_fe, flag_mg_fe,
           al_fe, e_al_fe, flag_al_fe, si_fe, e_si_fe, flag_si_fe,
           k_fe, e_k_fe, flag_k_fe, ca_fe, e_ca_fe, flag_ca_fe,
           sc_fe, e_sc_fe, flag_sc_fe, ti_fe, e_ti_fe, flag_ti_fe,
           v_fe, e_v_fe, flag_v_fe, cr_fe, e_cr_fe, flag_cr_fe,
           mn_fe, e_mn_fe, flag_mn_fe, co_fe, e_co_fe, flag_co_fe,
           ni_fe, e_ni_fe, flag_ni_fe, cu_fe, e_cu_fe, flag_cu_fe,
           zn_fe, e_zn_fe, flag_zn_fe, n_fe, e_n_fe, flag_n_fe,
           rb_fe, e_rb_fe, flag_rb_fe, sr_fe, e_sr_fe, flag_sr_fe,
           y_fe, e_y_fe, flag_y_fe, zr_fe, e_zr_fe, flag_zr_fe,
           mo_fe, e_mo_fe, flag_mo_fe, ru_fe, e_ru_fe, flag_ru_fe,
           ba_fe, e_ba_fe, flag_ba_fe, la_fe, e_la_fe, flag_la_fe,
           ce_fe, e_ce_fe, flag_ce_fe, nd_fe, e_nd_fe, flag_nd_fe,
           sm_fe, e_sm_fe, flag_sm_fe, eu_fe, e_eu_fe, flag_eu_fe,
           a_li, flag_a_li, nn_li_fe
    FROM galah_dr4.mainstartable
    WHERE gaiadr3_source_id = {SOURCE_ID}
    """,
    "GALAH DR4 mainstartable",
)

galah_dyn = tap_query(
    DC_TAP,
    f"""
    SELECT *
    FROM galah_dr4.vacdynamicstable
    WHERE gaiadr3_source_id = {SOURCE_ID}
    """,
    "GALAH DR4 vacdynamicstable",
)

sobj = None
if not galah.empty and "sobject_id" in galah.columns:
    sobj = int(galah.iloc[0]["sobject_id"])
galah_li = pd.DataFrame()
if sobj is not None:
    galah_li = tap_query(
        DC_TAP,
        f"""
        SELECT sobject_id, tmass_id, ALi, ALi_upp_lim, flag_ALi,
               e_ALi_low, e_ALi_upp, EW, fe_h, teff, logg, rv, snr, flag_sp
        FROM galah_dr4.vac3dnltelitable
        WHERE sobject_id = {sobj}
        """,
        "GALAH DR4 vac3dnltelitable (3D NLTE Li)",
    )

# ---------------------------------------------------------------- save raw
for df, name in [
    (gaia_src, "hd28888_query_gaia.csv"),
    (gaia_ap, "hd28888_query_gaia_astrophysical.csv"),
    (galah, "hd28888_query_galah.csv"),
    (galah_dyn, "hd28888_query_galah_dynamics.csv"),
    (galah_li, "hd28888_query_galah_li.csv"),
]:
    if not df.empty:
        df.to_csv(name, index=False)
        print(f"[SAVE] {name}")


def g(df, col):
    if not df.empty and col in df.columns:
        v = df.iloc[0][col]
        return None if pd.isna(v) else v
    return None


# ---------------------------------------------------------------- combined long table
rows = []


def add(dim, key, value, src, note=""):
    rows.append({"block": dim, "field": key, "value": value, "source": src, "note": note})


add("ID", "common_name", "HD 28888", "SIMBAD", "")
add("ID", "gaia_dr3_source_id", SOURCE_ID, "Gaia DR3", "")
add("ID", "tmass_id", g(galah, "tmass_id"), "GALAH DR4", "")

# Astrometry (Gaia)
add("Astrometry", "ra_deg", g(gaia_src, "ra"), "Gaia DR3", "")
add("Astrometry", "dec_deg", g(gaia_src, "dec"), "Gaia DR3", "")
add("Astrometry", "parallax_mas", g(gaia_src, "parallax"), "Gaia DR3", "")
add("Astrometry", "parallax_error_mas", g(gaia_src, "parallax_error"), "Gaia DR3", "")
plx = g(gaia_src, "parallax")
if plx:
    add("Astrometry", "distance_pc_1/plx", 1000.0 / plx, "derived", "naive 1/parallax")
add("Astrometry", "distance_gspphot_pc", g(gaia_src, "distance_gspphot"), "Gaia DR3", "")
add("Astrometry", "pmra_mas_yr", g(gaia_src, "pmra"), "Gaia DR3", "")
add("Astrometry", "pmdec_mas_yr", g(gaia_src, "pmdec"), "Gaia DR3", "")
add("Astrometry", "ruwe", g(gaia_src, "ruwe"), "Gaia DR3", "single if <1.4")
add("Astrometry", "non_single_star", g(gaia_src, "non_single_star"), "Gaia DR3", "0 = no NSS flag")

# Photometry
add("Photometry", "phot_g_mean_mag", g(gaia_src, "phot_g_mean_mag"), "Gaia DR3", "")
add("Photometry", "bp_rp", g(gaia_src, "bp_rp"), "Gaia DR3", "")

# Radial velocity
add("RV", "rv_gaia_km_s", g(gaia_src, "radial_velocity"), "Gaia DR3", "")
add("RV", "rv_gaia_err_km_s", g(gaia_src, "radial_velocity_error"), "Gaia DR3", "")
add("RV", "rv_galah_km_s", g(galah, "rv_gaia_dr3"), "GALAH DR4", "rv_gaia_dr3 col")

# Stellar parameters
add("Params", "teff_K_galah", g(galah, "teff"), "GALAH DR4", "")
add("Params", "teff_K_gspphot", g(gaia_ap, "teff_gspphot"), "Gaia DR3", "")
add("Params", "teff_K_gspspec", g(gaia_ap, "teff_gspspec"), "Gaia DR3", "")
add("Params", "logg_galah", g(galah, "logg"), "GALAH DR4", "")
add("Params", "logg_gspphot", g(gaia_ap, "logg_gspphot"), "Gaia DR3", "")
add("Params", "feh_galah", g(galah, "fe_h"), "GALAH DR4", "")
add("Params", "mh_gspphot", g(gaia_ap, "mh_gspphot"), "Gaia DR3", "")
add("Params", "mh_gspspec", g(gaia_ap, "mh_gspspec"), "Gaia DR3", "")
add("Params", "vmic_km_s", g(galah, "vmic"), "GALAH DR4", "")
add("Params", "vsini_km_s", g(galah, "vsini"), "GALAH DR4", "")
add("Params", "mass_msun_galah", g(galah, "mass"), "GALAH DR4", "")
add("Params", "mass_msun_flame", g(gaia_ap, "mass_flame"), "Gaia DR3", "")
add("Params", "lbol_lsun", g(galah, "lbol"), "GALAH DR4", "")

# Age
add("Age", "age_gyr_galah", g(galah, "age"), "GALAH DR4", "")
add("Age", "age_gyr_flame", g(gaia_ap, "age_flame"), "Gaia DR3", "FLAME")

# Quality
add("Quality", "flag_sp", g(galah, "flag_sp"), "GALAH DR4", "0 = clean")
add("Quality", "flag_fe_h", g(galah, "flag_fe_h"), "GALAH DR4", "0 = clean")
add("Quality", "snr_ccd3", g(galah, "snr_px_ccd3"), "GALAH DR4", "green CCD S/N")

# Abundances (GALAH) - element: (value_col, err_col, flag_col)
elems = [
    "c", "n", "o", "na", "mg", "al", "si", "k", "ca", "sc", "ti", "v",
    "cr", "mn", "co", "ni", "cu", "zn", "rb", "sr", "y", "zr", "mo",
    "ru", "ba", "la", "ce", "nd", "sm", "eu",
]
for el in elems:
    vc, ec, fc = f"{el}_fe", f"e_{el}_fe", f"flag_{el}_fe"
    add("Abundance", f"[{el.capitalize()}/Fe]", g(galah, vc), "GALAH DR4",
        f"err={g(galah, ec)} flag={g(galah, fc)}")
add("Abundance", "A(Li)", g(galah, "a_li"), "GALAH DR4", f"flag={g(galah,'flag_a_li')}")

# Derived dimension inputs
c_fe = g(galah, "c_fe")
o_fe = g(galah, "o_fe")
mg_fe = g(galah, "mg_fe")
si_fe = g(galah, "si_fe")
feh = g(galah, "fe_h")
# C/O number ratio from [C/Fe],[O/Fe],[Fe/H] using solar C/O=0.55 (Asplund09)
if c_fe is not None and o_fe is not None:
    import math
    solar_logCO = math.log10(0.55)  # log(N_C/N_O) solar
    logCO = solar_logCO + (c_fe - o_fe)
    add("Derived", "C/O_number_ratio", 10 ** logCO, "derived",
        "solar C/O=0.55; 10^(logCO_sun+[C/Fe]-[O/Fe])")
if mg_fe is not None and si_fe is not None:
    import math
    solar_logMgSi = math.log10(1.05)  # ~solar Mg/Si number ratio (Asplund09)
    logMgSi = solar_logMgSi + (mg_fe - si_fe)
    add("Derived", "Mg/Si_number_ratio", 10 ** logMgSi, "derived",
        "solar Mg/Si=1.05; 10^(logMgSi_sun+[Mg/Fe]-[Si/Fe])")
add("Derived", "[Mg/Fe]", mg_fe, "GALAH DR4", "alpha balance")
add("Derived", "[Si/Fe]", si_fe, "GALAH DR4", "alpha balance")
add("Derived", "[Fe/H]", feh, "GALAH DR4", "bulk metallicity")
add("Derived", "[Ba/Fe]_sproc", g(galah, "ba_fe"), "GALAH DR4", "s-process volatile proxy")
add("Derived", "[Y/Fe]_sproc", g(galah, "y_fe"), "GALAH DR4", "s-process")
add("Derived", "[Eu/Fe]_rproc", g(galah, "eu_fe"), "GALAH DR4", "r-process reference")

# Gaia GSP-Spec abundances (independent cross-check, if present)
add("Gaia-GSPSpec", "teff_gspspec_K", g(gaia_ap, "teff_gspspec"), "Gaia DR3", "")
add("Gaia-GSPSpec", "logg_gspspec", g(gaia_ap, "logg_gspspec"), "Gaia DR3", "")
add("Gaia-GSPSpec", "[M/H]_gspspec", g(gaia_ap, "mh_gspspec"), "Gaia DR3", "")
add("Gaia-GSPSpec", "[alpha/Fe]_gspspec", g(gaia_ap, "alphafe_gspspec"), "Gaia DR3", "")
for col, lbl in [("mgfe_gspspec", "[Mg/Fe]"), ("sife_gspspec", "[Si/Fe]"),
                 ("cafe_gspspec", "[Ca/Fe]"), ("tife_gspspec", "[Ti/Fe]"),
                 ("crfe_gspspec", "[Cr/Fe]"), ("nife_gspspec", "[Ni/Fe]"),
                 ("nfe_gspspec", "[N/Fe]"), ("cefe_gspspec", "[Ce/Fe]"),
                 ("ndfe_gspspec", "[Nd/Fe]"), ("zrfe_gspspec", "[Zr/Fe]"),
                 ("fem_gspspec", "[Fe/M]")]:
    add("Gaia-GSPSpec", f"{lbl}_gspspec", g(gaia_ap, col), "Gaia DR3", "")
add("Gaia-FLAME", "mass_msun", g(gaia_ap, "mass_flame"), "Gaia DR3", "FLAME")
add("Gaia-FLAME", "age_gyr", g(gaia_ap, "age_flame"), "Gaia DR3", "FLAME")
add("Gaia-FLAME", "radius_rsun", g(gaia_ap, "radius_flame"), "Gaia DR3", "FLAME")
add("Gaia-FLAME", "lum_lsun", g(gaia_ap, "lum_flame"), "Gaia DR3", "FLAME")

# 3D NLTE Li (VAC)
add("Lithium", "A(Li)_3D_NLTE", g(galah_li, "ALi"), "GALAH DR4 VAC", "3D NLTE")
add("Lithium", "A(Li)_upp_lim", g(galah_li, "ALi_upp_lim"), "GALAH DR4 VAC", "")
add("Lithium", "EW_Li_mA", g(galah_li, "EW"), "GALAH DR4 VAC", "")

# Galactic kinematics (dynamics VAC) - kinematic-stability dimension
for col, lbl, note in [
    ("X_XYZ", "X_kpc", "Galactocentric"), ("Y_XYZ", "Y_kpc", ""),
    ("Z_XYZ", "Z_kpc", "height above plane"),
    ("R_med", "R_gal_kpc", "guiding radius proxy"),
    ("U_UVW", "U_km_s", ""), ("V_UVW", "V_km_s", ""), ("W_UVW", "W_km_s", ""),
    ("ecc", "eccentricity", "orbital"), ("zmax", "z_max_kpc", "max excursion"),
    ("R_peri", "R_peri_kpc", ""), ("R_ap", "R_apo_kpc", ""),
    ("J_R", "J_R", "radial action"), ("J_Z", "J_Z", "vertical action"),
    ("L_Z", "L_Z", "ang. momentum"), ("Energy", "orbital_energy", ""),
]:
    add("Kinematics", lbl, g(galah_dyn, col), "GALAH DR4 VAC", note)

combined = pd.DataFrame(rows)
combined.to_csv("hd28888_combined_profile.csv", index=False)
print("[SAVE] hd28888_combined_profile.csv")

print("\n" + "=" * 78)
print("HD 28888  /  Gaia DR3 3312653647717790720  — combined live profile")
print("=" * 78)
for blk in combined["block"].unique():
    print(f"\n## {blk}")
    sub = combined[combined["block"] == blk]
    for _, r in sub.iterrows():
        val = r["value"]
        note = f"   ({r['note']})" if r["note"] else ""
        print(f"  {r['field']:<26} {str(val):<22} [{r['source']}]{note}")
print("\nDONE")

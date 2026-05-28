"""CCT population test — STAGE 2: apply the FROZEN 9D scorer to hosts and field.

Pre-registration: PRE_REGISTRATION.md @ commit 1441551
Frozen scorer:    habitability_v2.py @ commit cfa1249

This script:
  - Loads cct_test_hosts.csv (APOGEE chemistry of confirmed planet hosts)
  - Loads cct_test_field.csv (APOGEE FGK dwarf field control)
  - Builds matched controls (host-by-host nearest neighbour in Teff, logg, [Fe/H])
  - Applies the EXACT scoring functions from habitability_v2.py (no tuning)
  - For APOGEE: volatile dimension uses [Ce/Fe] (pre-registered substitution)
  - Outputs scored CSVs for stage 3 statistical tests
"""
import numpy as np, pandas as pd
import warnings
warnings.filterwarnings("ignore")

print("="*78)
print("CCT POPULATION TEST -- STAGE 2: apply frozen scorer")
print("="*78)

# ===== FROZEN SCORING FUNCTIONS from habitability_v2.py @ cfa1249 =====
SOLAR_CO = 0.549
REF_TEFF = 5778.0
CO_TEFF_SLOPE = -0.000527
CO_TEFF_INTERCEPT = 3.5242

def teff_correct_co(co_measured, teff_star):
    expected_at_teff = CO_TEFF_SLOPE * teff_star + CO_TEFF_INTERCEPT
    expected_at_solar = CO_TEFF_SLOPE * REF_TEFF + CO_TEFF_INTERCEPT
    return np.clip(co_measured + (expected_at_solar - expected_at_teff), 0.05, 3.0)

def score_co(co):
    co_unc = 0.15
    score = np.exp(-0.5 * np.maximum(0, co - 0.65)**2 / co_unc**2)
    if hasattr(co, "__len__"):
        score[co < 0.15] *= co[co < 0.15] / 0.15
    return np.clip(score, 0, 1)

def score_mgsi(mgsi): return np.clip(np.exp(-0.5 * ((mgsi - 1.02) / 0.4)**2), 0, 1)
def score_feh(feh):
    score = np.exp(-0.5 * ((feh - 0.0) / 0.3)**2)
    if hasattr(feh, "__len__"):
        score[feh < -0.5] *= np.exp(-((feh[feh < -0.5] + 0.5) / 0.2)**2)
    return np.clip(score, 0, 1)
def score_mgfe(mgfe): return np.clip(np.exp(-0.5 * (mgfe / 0.15)**2), 0, 1)
def score_sife(sife): return np.clip(np.exp(-0.5 * (sife / 0.15)**2), 0, 1)
def score_cafe(cafe): return np.clip(np.exp(-0.5 * (cafe / 0.20)**2), 0, 1)
def score_alfe(alfe): return np.clip(np.exp(-0.5 * ((alfe - 0.05) / 0.15)**2), 0, 1)

def score_volatile(bafe):
    """Volatile budget; same Gaussian, applied to [Ba/Fe] (GALAH) or [Ce/Fe] (APOGEE)."""
    score = np.exp(-0.5 * ((bafe - 0.05) / 0.25)**2)
    if hasattr(bafe, "__len__"):
        depleted = bafe < -0.3
        score[depleted] *= np.exp(-((bafe[depleted] + 0.3) / 0.15)**2)
    return np.clip(score, 0, 1)

def score_age(age_gyr):
    age_arr = np.atleast_1d(age_gyr).astype(float)
    score = np.ones_like(age_arr)
    young = age_arr < 1.0
    score[young] = np.clip(age_arr[young] / 1.0, 0.1, 1.0)
    very_young = age_arr < 0.3
    score[very_young] = 0.1
    old = age_arr > 8.0
    score[old] = np.exp(-0.5 * ((age_arr[old] - 8.0) / 4.0)**2)
    return np.clip(score, 0, 1)

# Pre-registered weights -- DO NOT MODIFY
WEIGHTS = np.array([1.0, 1.5, 1.5, 1.0, 1.0, 0.5, 0.5, 1.0, 0.75])
DIMS    = ["C/O", "Mg/Si", "[Fe/H]", "[Mg/Fe]", "[Si/Fe]", "[Ca/Fe]",
           "[Al/Fe]", "Volatile", "Age"]

def apply_scorer(df, c_fe_col, o_fe_col, mg_fe_col, si_fe_col, fe_h_col,
                 ca_fe_col, al_fe_col, ce_fe_col, age_col, teff_col):
    """Run frozen 9D scorer on a dataframe. Returns dataframe with new columns."""
    df = df.copy()
    # C/O computed from C and O abundances [X/Fe]
    df["C_O_raw"] = (10.0 ** (df[c_fe_col] - df[o_fe_col])) * SOLAR_CO
    df["C_O"] = teff_correct_co(df["C_O_raw"].values, df[teff_col].values)
    df["Mg_Si"] = 10.0 ** (df[mg_fe_col] - df[si_fe_col])

    co_v = df["C_O"].fillna(SOLAR_CO).values
    df["s_CO"] = score_co(co_v)
    df["s_MgSi"] = score_mgsi(df["Mg_Si"].fillna(1.0).values)
    df["s_FeH"] = score_feh(df[fe_h_col].fillna(0.0).values)
    df["s_MgFe"] = score_mgfe(df[mg_fe_col].fillna(0.0).values)
    df["s_SiFe"] = score_sife(df[si_fe_col].fillna(0.0).values)
    # Ca and Al fall back to 0.8 if missing (matches scorer behavior)
    ca_valid = df[ca_fe_col].notna() & (df[ca_fe_col] > -1.0) & (df[ca_fe_col] < 1.0)
    df["s_CaFe"] = np.where(ca_valid, score_cafe(df[ca_fe_col].fillna(0).values), 0.8)
    al_valid = df[al_fe_col].notna() & (df[al_fe_col] > -1.0) & (df[al_fe_col] < 1.0)
    df["s_AlFe"] = np.where(al_valid, score_alfe(df[al_fe_col].fillna(0).values), 0.8)
    # Volatile: in APOGEE use [Ce/Fe]; fall back to 0.7 if missing
    ce_valid = df[ce_fe_col].notna() & (df[ce_fe_col] > -1.5) & (df[ce_fe_col] < 1.5)
    df["s_volatile"] = np.where(ce_valid, score_volatile(df[ce_fe_col].fillna(0).values), 0.7)
    # Age: fall back to 0.7 if missing or no age column
    if age_col is None or age_col not in df.columns:
        df["s_age"] = 0.7
    else:
        age_valid = df[age_col].notna() & (df[age_col] > 0) & (df[age_col] < 15)
        df["s_age"] = np.where(age_valid, score_age(df[age_col].fillna(5.0).values), 0.7)

    scores = np.column_stack([df.s_CO, df.s_MgSi, df.s_FeH, df.s_MgFe, df.s_SiFe,
                              df.s_CaFe, df.s_AlFe, df.s_volatile, df.s_age])
    df["hab_score"] = np.exp(np.average(np.log(np.maximum(scores, 1e-10)),
                                         weights=WEIGHTS, axis=1))
    df["co_valid"] = df["C_O_raw"].between(0.05, 3.0)
    return df

# 1. Load
print("\n[1] Loading hosts and field...")
hosts = pd.read_csv("cct_test_hosts.csv")
field = pd.read_csv("cct_test_field.csv")
print(f"  hosts in APOGEE: {len(hosts)}")
print(f"  field control:   {len(field)}")

# 2. Rename APOGEE columns to scorer's expected names
print("\n[2] Apply scorer to hosts...")
# hosts CSV has columns ap_C_FE, ap_O_FE etc.
hcol = lambda x: f"ap_{x}"
hosts_scored = apply_scorer(
    hosts,
    c_fe_col=hcol("C_FE"), o_fe_col=hcol("O_FE"),
    mg_fe_col=hcol("MG_FE"), si_fe_col=hcol("SI_FE"),
    fe_h_col=hcol("FE_H"),
    ca_fe_col=hcol("CA_FE"), al_fe_col=hcol("AL_FE"),
    ce_fe_col=hcol("CE_FE"),
    age_col="host_st_age",   # planet host catalog age (NEA)
    teff_col=hcol("TEFF"))

# only count hosts where the C/O was computable (otherwise dropping C/O dimension)
print(f"  hosts with valid C/O computation: {hosts_scored.co_valid.sum()} / {len(hosts_scored)}")
hosts_scored.to_csv("cct_test_hosts_scored.csv", index=False)
print(f"  saved -> cct_test_hosts_scored.csv")

# 3. Score field
print("\n[3] Apply scorer to field control...")
field_scored = apply_scorer(
    field,
    c_fe_col="C_FE", o_fe_col="O_FE",
    mg_fe_col="MG_FE", si_fe_col="SI_FE",
    fe_h_col="FE_H",
    ca_fe_col="CA_FE", al_fe_col="AL_FE",
    ce_fe_col="CE_FE",
    age_col=None if "age" not in field.columns else "age",  # APOGEE has no age in allStar
    teff_col="TEFF")
print(f"  field stars with valid C/O: {field_scored.co_valid.sum()} / {len(field_scored)}")
field_scored.to_csv("cct_test_field_scored.csv", index=False)
print(f"  saved -> cct_test_field_scored.csv")

# 4. Summary stats
print(f"\n{'='*78}\nSCORED DISTRIBUTIONS SUMMARY\n{'='*78}")
print(f"\nField (n={len(field_scored)}):")
print(f"  hab_score median: {field_scored.hab_score.median():.4f}")
print(f"  hab_score mean:   {field_scored.hab_score.mean():.4f}")
print(f"  hab_score IQR:    [{field_scored.hab_score.quantile(.25):.4f}, {field_scored.hab_score.quantile(.75):.4f}]")

print(f"\nHosts by category:")
for cat in hosts_scored.host_category.unique():
    sub = hosts_scored[hosts_scored.host_category == cat]
    if len(sub) == 0: continue
    print(f"  {cat} (n={len(sub)}):  median={sub.hab_score.median():.4f}  "
          f"mean={sub.hab_score.mean():.4f}  IQR=[{sub.hab_score.quantile(.25):.4f},"
          f"{sub.hab_score.quantile(.75):.4f}]")

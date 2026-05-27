"""
HD 28888 follow-up analysis (4 deliverables):
  1. 9D scorer rank vs the scored GALAH cohort (habitability_v2_targets.csv)
  2. Teutsch_80 reality check: kinematic/positional test of "membership"
  3. Habitable-zone geometry for the subgiant (period guidance for RV)
  4. Dredge-up correction -> birth C/O

Uses live-queried HD 28888 numbers and existing project outputs.
No external network needed (cluster catalog cross-check done separately).
"""
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- live HD 28888 numbers (from archive query)
HD = dict(
    source_id=3312653647717790720,
    ra=68.41124363572398, dec=15.961256676018776,
    dist_pc=99.992, pmra=65.22837692734834, pmdec=-55.77929544357106, rv=53.70128,
    teff=5734.0254, logg=3.9375741, feh=0.12957397,
    c_fe=-0.0715446, n_fe=0.37618434, o_fe=0.18021967,
    mg_fe=0.020332545, si_fe=0.005609162, al_fe=0.08610666, ca_fe=-0.042535856,
    ba_fe=-0.039925143, age=6.3121543, mass=1.15, lum=3.55,
)

SOLAR_CO = 0.549
REF_TEFF = 5778.0
CO_TEFF_SLOPE = -0.000527
CO_TEFF_INTERCEPT = 3.5242

# Asplund (2009) photospheric solar abundances (log eps, H=12)
A_C_SUN, A_N_SUN, A_O_SUN = 8.43, 7.83, 8.69

def teff_correct_co(co, teff):
    exp_t = CO_TEFF_SLOPE * teff + CO_TEFF_INTERCEPT
    exp_s = CO_TEFF_SLOPE * REF_TEFF + CO_TEFF_INTERCEPT
    return float(np.clip(co + (exp_s - exp_t), 0.05, 3.0))

# scoring functions (scalar versions, identical math to habitability_v2.py)
def score_co(co):
    s = np.exp(-0.5 * max(0, co - 0.65) ** 2 / 0.15 ** 2)
    if co < 0.15:
        s *= co / 0.15
    return float(np.clip(s, 0, 1))
def score_mgsi(x): return float(np.clip(np.exp(-0.5 * ((x - 1.02) / 0.4) ** 2), 0, 1))
def score_feh(x):
    s = np.exp(-0.5 * ((x - 0.0) / 0.3) ** 2)
    if x < -0.5: s *= np.exp(-((x + 0.5) / 0.2) ** 2)
    return float(np.clip(s, 0, 1))
def score_mgfe(x): return float(np.clip(np.exp(-0.5 * (x / 0.15) ** 2), 0, 1))
def score_sife(x): return float(np.clip(np.exp(-0.5 * (x / 0.15) ** 2), 0, 1))
def score_cafe(x): return float(np.clip(np.exp(-0.5 * (x / 0.20) ** 2), 0, 1))
def score_alfe(x): return float(np.clip(np.exp(-0.5 * ((x - 0.05) / 0.15) ** 2), 0, 1))
def score_volatile(x):
    s = np.exp(-0.5 * ((x - 0.05) / 0.25) ** 2)
    if x < -0.3: s *= np.exp(-((x + 0.3) / 0.15) ** 2)
    return float(np.clip(s, 0, 1))
def score_age(a):
    if a < 0.3: return 0.1
    if a < 1.0: return float(np.clip(a / 1.0, 0.1, 1.0))
    if a > 8.0: return float(np.clip(np.exp(-0.5 * ((a - 8.0) / 4.0) ** 2), 0, 1))
    return 1.0

def composite_score(co, mgsi, feh, mgfe, sife, cafe, alfe, vol, age):
    w = np.array([1.0, 1.5, 1.5, 1.0, 1.0, 0.5, 0.5, 1.0, 0.75])
    s = np.array([score_co(co), score_mgsi(mgsi), score_feh(feh), score_mgfe(mgfe),
                  score_sife(sife), score_cafe(cafe), score_alfe(alfe),
                  score_volatile(vol), score_age(age)])
    return float(np.exp(np.average(np.log(np.maximum(s, 1e-10)), weights=w))), s

print("=" * 78)
print("TASK 1 — Re-score HD 28888 from live numbers and rank vs cohort")
print("=" * 78)
co_raw = 10 ** (HD["c_fe"] - HD["o_fe"]) * SOLAR_CO
co_corr = teff_correct_co(co_raw, HD["teff"])
mgsi = 10 ** (HD["mg_fe"] - HD["si_fe"])
hab, subs = composite_score(co_corr, mgsi, HD["feh"], HD["mg_fe"], HD["si_fe"],
                            HD["ca_fe"], HD["al_fe"], HD["ba_fe"], HD["age"])
dim_names = ["C/O", "Mg/Si", "[Fe/H]", "[Mg/Fe]", "[Si/Fe]", "[Ca/Fe]", "[Al/Fe]", "Volatile", "Age"]
print(f"C/O raw={co_raw:.4f}  C/O Teff-corr={co_corr:.4f}  Mg/Si={mgsi:.4f}")
print("Sub-scores:")
for n, s in zip(dim_names, subs):
    print(f"   {n:<10} {s:.4f}")
print(f"Composite hab_score = {hab:.4f}")

coh = pd.read_csv("habitability_v2_targets.csv")
N = len(coh)
row = coh[coh["gaiadr3_source_id"] == HD["source_id"]]
stored = float(row["hab_score"].iloc[0]) if len(row) else np.nan
print(f"\nStored hab_score in cohort file = {stored:.4f}  (recompute matches: {abs(stored-hab)<1e-3})")
better = (coh["hab_score"] > stored).sum()
rank = better + 1
pct_top = rank / N * 100
print(f"Cohort size (FGK dwarfs scored) = {N}")
print(f"HD 28888 rank = {rank} of {N}  ->  top {pct_top:.3f}%  (percentile {100-pct_top:.3f})")
n_excellent = (coh["hab_score"] > 0.9).sum()
rank_in_exc = (coh["hab_score"] > stored).sum() + 1
print(f"Stars scoring >0.9 (excellent) = {n_excellent} ({n_excellent/N:.1%})")
print(f"Within excellent group, HD 28888 is #{rank_in_exc}")
print(f"Top 1% cutoff score   = {coh['hab_score'].quantile(0.99):.4f}")
print(f"Top 0.1% cutoff score = {coh['hab_score'].quantile(0.999):.4f}")
print(f"Max score in cohort   = {coh['hab_score'].max():.4f}")

print("\n" + "=" * 78)
print("TASK 4 — Dredge-up correction: back out BIRTH C/O from CN-processed surface")
print("=" * 78)
# current surface number abundances (log eps)
AC = A_C_SUN + HD["c_fe"] + HD["feh"]
AN = A_N_SUN + HD["n_fe"] + HD["feh"]
AO = A_O_SUN + HD["o_fe"] + HD["feh"]
nC, nN, nO = 10 ** AC, 10 ** AN, 10 ** AO
print(f"Surface: A(C)={AC:.3f} A(N)={AN:.3f} A(O)={AO:.3f}")
print(f"Surface [N/Fe]=+{HD['n_fe']:.3f} (high), [C/Fe]={HD['c_fe']:+.3f} (low) -> CN-processing signature")
co_surface_num = nC / nO
print(f"Surface C/O (number) = {co_surface_num:.4f}")
# birth N assumed scaled-solar ([N/Fe]_0 = 0); CN cycle conserves C+N, O ~ unchanged
AN0 = A_N_SUN + 0.0 + HD["feh"]
nN0 = 10 ** AN0
dN = nN - nN0                      # nitrogen added by dredge-up
nC0 = nC + dN                      # that nitrogen came from carbon
AC0 = np.log10(nC0)
co_birth_num = nC0 / nO
print(f"Assume birth [N/Fe]=0 -> A(N)_birth={AN0:.3f}; N added dN={dN:.3e}")
print(f"Birth A(C)={AC0:.3f}  -> birth [C/Fe]={AC0 - A_C_SUN - HD['feh']:+.3f}")
print(f"BIRTH C/O (number) = {co_birth_num:.4f}   (surface was {co_surface_num:.4f})")
# sensitivity: partial dredge-up (50% of the inferred N excess from C)
nC0_half = nC + 0.5 * dN
print(f"If only 50% of N excess is dredge-up: birth C/O = {nC0_half/nO:.4f}")
# rescore with birth C/O (apply same solar normalization: scorer C/O is number-ratio based)
co_birth_corr = teff_correct_co(co_birth_num, HD["teff"])
print(f"Birth C/O Teff-corr = {co_birth_corr:.4f} -> score_co = {score_co(co_birth_corr):.4f} "
      f"(was {score_co(co_corr):.4f}; threshold for penalty is 0.65)")
print(f"Solar C/O = {SOLAR_CO}; birth C/O {co_birth_num:.3f} is {'closer to' if abs(co_birth_num-SOLAR_CO)<abs(co_surface_num-SOLAR_CO) else 'farther from'} solar than surface")

print("\n" + "=" * 78)
print("TASK 2 — Teutsch_80 'membership' reality check (kinematic + positional)")
print("=" * 78)
# cluster params from VizieR (Hunt & Reffert 2023 / Cantat-Gaudin 2020).
# NOTE: project's t9 'dist_cl=0.349' was the PARALLAX in mas mislabeled as kpc;
#       real distance is ~2.4-2.6 kpc (plx 0.339-0.349 mas).
CL = dict(ra=223.361, dec=-60.478, pmra=-5.553, pmdec=-3.581,
          dist_kpc=2.52, age_gyr_lo=0.092, age_gyr_hi=0.224)
members = pd.DataFrame({
    "rv": [-18.51, -15.04, -41.82, -8.13, -3.77, -30.19],
    "fe_h": [0.0118, -0.1017, 0.0344, 0.0987, 0.0677, 0.2297],
    "C_O": [0.2940, 0.2054, 0.4267, 0.5702, 0.4583, 0.5085],
    "age": [1.295, 1.414, 0.430, 4.620, 2.173, 4.620],
})
def angsep(ra1, de1, ra2, de2):
    r1, d1, r2, d2 = map(np.deg2rad, [ra1, de1, ra2, de2])
    return np.rad2deg(np.arccos(np.clip(
        np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1-r2), -1, 1)))
sep = angsep(HD["ra"], HD["dec"], CL["ra"], CL["dec"])
print(f"Cluster center: RA={CL['ra']} Dec={CL['dec']}  dist={CL['dist_kpc']*1000:.0f} pc")
print(f"Star:           RA={HD['ra']:.3f} Dec={HD['dec']:.3f}  dist={HD['dist_pc']:.0f} pc")
print(f"Angular separation on sky      = {sep:.1f} deg")
print(f"Distance ratio (star/cluster)  = {HD['dist_pc']/(CL['dist_kpc']*1000):.2f}x")
print(f"Proper motion: star=({HD['pmra']:+.1f},{HD['pmdec']:+.1f})  "
      f"cluster=({CL['pmra']:+.1f},{CL['pmdec']:+.1f}) mas/yr")
pm_diff = np.hypot(HD['pmra']-CL['pmra'], HD['pmdec']-CL['pmdec'])
print(f"PM vector difference           = {pm_diff:.1f} mas/yr")
print(f"RV: star={HD['rv']:+.1f}  cluster members range {members['rv'].min():.1f} to "
      f"{members['rv'].max():.1f} (mean {members['rv'].mean():.1f}, std {members['rv'].std():.1f}) km/s")
print(f"AGE: star={HD['age']:.1f} Gyr  vs  cluster={CL['age_gyr_lo']:.2f}-{CL['age_gyr_hi']:.2f} Gyr "
      f"(~{HD['age']/CL['age_gyr_hi']:.0f}-{HD['age']/CL['age_gyr_lo']:.0f}x older -> physically impossible as a member)")
print(f"Member C/O spread: {members['C_O'].min():.2f}-{members['C_O'].max():.2f} "
      f"(std {members['C_O'].std():.3f});  member age spread (project pipeline): "
      f"{members['age'].min():.1f}-{members['age'].max():.1f} Gyr")
print(f"-> Star C/O {co_corr:.3f} sits inside the broad member envelope, but the cluster's")
print(f"   own C/O scatter ({members['C_O'].std():.3f}) and age scatter make it non-diagnostic.")
print("VERDICT: chemical-nearest-template label ONLY. Position (132 deg away), distance")
print("         (~25x), proper motion, RV, AND age (6.3 vs 0.1-0.2 Gyr) all EXCLUDE membership.")
print("         This is exactly the 71,000-background-match regime of the Precision Wall.")

print("\n" + "=" * 78)
print("TASK 3 — Habitable zone geometry for the subgiant (RV target guidance)")
print("=" * 78)
L = HD["lum"]; M = HD["mass"]
# Kopparapu (2013) flux factors ~ for Teff 5734 K (between recent-Venus/early-Mars conservative)
S_inner_cons, S_outer_cons = 1.107, 0.356   # conservative (runaway / max greenhouse)
S_inner_opt, S_outer_opt = 1.776, 0.320     # optimistic (recent Venus / early Mars)
def hz_a(S): return np.sqrt(L / S)
def period_yr(a, m): return np.sqrt(a ** 3 / m)
ai_c, ao_c = hz_a(S_inner_cons), hz_a(S_outer_cons)
ai_o, ao_o = hz_a(S_inner_opt), hz_a(S_outer_opt)
print(f"L = {L} Lsun, M = {M} Msun, Teff = {HD['teff']:.0f} K, R ~ 2 Rsun (logg {HD['logg']:.2f})")
print(f"Conservative HZ: {ai_c:.2f} - {ao_c:.2f} AU   (P = {period_yr(ai_c,M):.2f} - {period_yr(ao_c,M):.2f} yr)")
print(f"Optimistic   HZ: {ai_o:.2f} - {ao_o:.2f} AU   (P = {period_yr(ai_o,M):.2f} - {period_yr(ao_o,M):.2f} yr)")
print(f"-> Elevated luminosity ({L} Lsun) pushes HZ outward; RV campaigns must target")
print(f"   {ai_c:.1f}-{ao_c:.1f} AU companions => {period_yr(ai_c,M):.1f}-{period_yr(ao_c,M):.1f} yr periods (multi-year baseline).")
print("\nDONE")

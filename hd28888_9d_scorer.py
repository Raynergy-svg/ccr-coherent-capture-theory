"""
9D habitability scorer applied to HD 28888, with per-dimension and overall
percentile ranks against the scored GALAH cohort (habitability_v2_targets.csv)
and the 18 actionable targets (target_dossier.csv).

IMPORTANT: the *implemented* scorer (habitability_v2.py) scores 9 dimensions:
  C/O, Mg/Si, [Fe/H], [Mg/Fe], [Si/Fe], [Ca/Fe], [Al/Fe], Volatile(Ba/Fe), Age
This differs from the *conceptual* 9D in the request (which lists birth-cluster
density and Galactic kinematic stability instead of [Ca/Fe]/[Al/Fe]). Only the
implemented dimensions can be ranked against the cohort, which was scored with
them. Cluster density / kinematics are reported descriptively, not scored.
"""
import numpy as np, pandas as pd

SOLAR_CO, REF_TEFF = 0.549, 5778.0
CO_TEFF_SLOPE, CO_TEFF_INTERCEPT = -0.000527, 3.5242
A_C_SUN, A_N_SUN, A_O_SUN = 8.43, 7.83, 8.69

def teff_correct_co(co, teff):
    return float(np.clip(co + ((CO_TEFF_SLOPE*REF_TEFF+CO_TEFF_INTERCEPT)
                               - (CO_TEFF_SLOPE*teff+CO_TEFF_INTERCEPT)), 0.05, 3.0))
def s_co(co):
    s = np.exp(-0.5*np.maximum(0,co-0.65)**2/0.15**2)
    s = np.where(co<0.15, s*co/0.15, s); return np.clip(s,0,1)
def s_mgsi(x): return np.clip(np.exp(-0.5*((x-1.02)/0.4)**2),0,1)
def s_feh(x):
    s=np.exp(-0.5*((x-0.0)/0.3)**2); s=np.where(x<-0.5, s*np.exp(-((x+0.5)/0.2)**2),s); return np.clip(s,0,1)
def s_mgfe(x): return np.clip(np.exp(-0.5*(x/0.15)**2),0,1)
def s_sife(x): return np.clip(np.exp(-0.5*(x/0.15)**2),0,1)
def s_cafe(x): return np.clip(np.exp(-0.5*(x/0.20)**2),0,1)
def s_alfe(x): return np.clip(np.exp(-0.5*((x-0.05)/0.15)**2),0,1)
def s_vol(x):
    s=np.exp(-0.5*((x-0.05)/0.25)**2); s=np.where(x<-0.3, s*np.exp(-((x+0.3)/0.15)**2),s); return np.clip(s,0,1)
def s_age(a):
    a=np.atleast_1d(a).astype(float); s=np.ones_like(a)
    s=np.where(a<1.0, np.clip(a/1.0,0.1,1.0), s); s=np.where(a<0.3,0.1,s)
    s=np.where(a>8.0, np.exp(-0.5*((a-8.0)/4.0)**2), s); return np.clip(s,0,1)

W = np.array([1.0,1.5,1.5,1.0,1.0,0.5,0.5,1.0,0.75])
DIMS = ["C/O","Mg/Si","[Fe/H]","[Mg/Fe]","[Si/Fe]","[Ca/Fe]","[Al/Fe]","Volatile(Ba/Fe)","Age"]

# ---- HD 28888 inputs (from archive profile) ----
HD = dict(co_raw=0.308, mgsi=1.086, feh=0.130, mgfe=0.020, sife=0.006,
          cafe=-0.0425, alfe=0.0861, bafe=-0.040, age=6.31, age_flame=6.05,
          teff=5734.0, c_fe=-0.0715446, n_fe=0.37618434, o_fe=0.18021967)

co_corr = teff_correct_co(HD["co_raw"], HD["teff"])
def composite(s): return float(np.exp(np.average(np.log(np.maximum(s,1e-10)), weights=W)))

# Anchor to HD 28888's STORED cohort sub-scores (cohort-consistent -> 0.9728).
# The scorer's Mg/Si is 10^([Mg/Fe]-[Si/Fe])=1.0345, NOT the profile's solar-
# normalised number ratio 1.086; using the latter would mis-score this dim.
_c = pd.read_csv("habitability_v2_targets.csv")
_hd = _c[_c["gaiadr3_source_id"]==3312653647717790720].iloc[0]
MGSI_SCORER = float(_hd["Mg_Si"])  # 1.0345
sub = np.array([
    float(_hd["s_CO"]), float(_hd["s_MgSi"]), float(_hd["s_FeH"]), float(_hd["s_MgFe"]),
    float(_hd["s_SiFe"]), float(s_cafe(HD["cafe"])), float(s_alfe(HD["alfe"])),
    float(_hd["s_volatile"]), float(_hd["s_age"]),
])
hab = composite(sub)
# sensitivity: scoring with the profile's solar-normalised Mg/Si=1.086 instead
sub_mgsi086 = sub.copy(); sub_mgsi086[1] = float(s_mgsi(HD["mgsi"])); hab_mgsi086 = composite(sub_mgsi086)

# age sensitivity (FLAME)
sub_flame = sub.copy(); sub_flame[8] = float(s_age(HD["age_flame"])[0]); hab_flame = composite(sub_flame)

# birth C/O (dredge-up): conserve C+N, birth [N/Fe]=0
AC=A_C_SUN+HD["c_fe"]+HD["feh"]; AN=A_N_SUN+HD["n_fe"]+HD["feh"]; AO=A_O_SUN+HD["o_fe"]+HD["feh"]
nC,nN,nO=10**AC,10**AN,10**AO; nN0=10**(A_N_SUN+0.0+HD["feh"])
co_birth_full=(nC+(nN-nN0))/nO; co_birth_half=(nC+0.5*(nN-nN0))/nO
co_birth_corr=teff_correct_co(co_birth_full, HD["teff"])
sub_birth=sub.copy(); sub_birth[0]=float(s_co(np.array([co_birth_corr]))[0]); hab_birth=composite(sub_birth)

# ---- cohort ----
coh = pd.read_csv("habitability_v2_targets.csv")
N=len(coh); exc=coh[coh["hab_score"]>0.9].copy(); Ne=len(exc)
# recompute Ca/Fe, Al/Fe subscores across cohort (not saved in csv)
coh["s_cafe"]=np.where(coh["ca_fe"].notna(), s_cafe(coh["ca_fe"].fillna(0).values), 0.8)
coh["s_alfe"]=np.where(coh["al_fe"].notna(), s_alfe(coh["al_fe"].fillna(0).values), 0.8)
exc["s_cafe"]=np.where(exc["ca_fe"].notna(), s_cafe(exc["ca_fe"].fillna(0).values),0.8)
exc["s_alfe"]=np.where(exc["al_fe"].notna(), s_alfe(exc["al_fe"].fillna(0).values),0.8)

colmap={"C/O":"s_CO","Mg/Si":"s_MgSi","[Fe/H]":"s_FeH","[Mg/Fe]":"s_MgFe","[Si/Fe]":"s_SiFe",
        "[Ca/Fe]":"s_cafe","[Al/Fe]":"s_alfe","Volatile(Ba/Fe)":"s_volatile","Age":"s_age"}
def pct(series, val):  # percentile = % of cohort with score <= val
    return 100.0*(series.values<=val).mean()

# overall rank
stored=float(coh.loc[coh["gaiadr3_source_id"]==3312653647717790720,"hab_score"].iloc[0])
rank_full=int((coh["hab_score"]>stored).sum())+1
rank_exc=int((exc["hab_score"]>stored).sum())+1

# ---- actionable 18 ----
dos=pd.read_csv("target_dossier.csv")
dos_sorted=dos.sort_values("hab_score",ascending=False).reset_index(drop=True)
hd_rank_18=int(dos_sorted.index[dos_sorted["gaia_id"]==3312653647717790720][0])+1

# ---------- build report ----------
L=[]
def w(s=""): L.append(s)
w("# HD 28888 — 9D Habitability Scorer Result")
w()
w("**Gaia DR3 3312653647717790720** | scorer: `habitability_v2.py` (Certan 2026)")
w()
w("> **Dimension-set note.** The *implemented* scorer scores "
  "C/O, Mg/Si, [Fe/H], [Mg/Fe], [Si/Fe], **[Ca/Fe], [Al/Fe]**, Volatile([Ba/Fe]), Age. "
  "The request's conceptual 9D listed **birth-cluster density** and **Galactic kinematic "
  "stability** instead of [Ca/Fe]/[Al/Fe]. The cohort was scored with the implemented set, "
  "so only those nine can be ranked; cluster density & kinematics are reported as context only.")
w()
w("## 1. Nine dimension scores + composite")
w()
w(f"C/O: raw {HD['co_raw']:.3f} -> Teff-corrected **{co_corr:.3f}** "
  f"(implemented linear-slope detrend; the quoted rho=-0.68 is the correlation, not the formula).")
w()
w("| # | Dimension | input | score (0-1) | weight |")
w("|---|---|---|---|---|")
inputs=[f"{co_corr:.3f}", f"{MGSI_SCORER:.3f}", f"{HD['feh']:+.3f}", f"{HD['mgfe']:+.3f}",
        f"{HD['sife']:+.3f}", f"{HD['cafe']:+.3f}", f"{HD['alfe']:+.3f}", f"{HD['bafe']:+.3f}", f"{HD['age']:.2f} Gyr"]
for i,d in enumerate(DIMS):
    w(f"| {i+1} | {d} | {inputs[i]} | {sub[i]:.4f} | {W[i]} |")
w(f"\n**Composite hab_score = {hab:.4f}** (weighted geometric mean), matching the "
  f"stored cohort value {stored:.4f}.")
w(f"\n> **Mg/Si definition flag.** The scorer uses Mg/Si = $10^{{[Mg/Fe]-[Si/Fe]}}$ = "
  f"**{MGSI_SCORER:.3f}** (score {sub[1]:.4f}), *not* the profile's solar-normalised number "
  f"ratio **1.086**. Feeding 1.086 into the scorer (Gaussian centred at 1.02) would drop "
  f"that dimension to {float(s_mgsi(HD['mgsi'])):.4f} and the composite to **{hab_mgsi086:.4f}**. "
  f"The cohort was scored with the 1.034 convention, so that is used here for a valid comparison.")
w(f"\nAge sensitivity: FLAME age {HD['age_flame']:.2f} Gyr -> s_age "
  f"{sub_flame[8]:.4f}, composite **{hab_flame:.4f}** (Δ={hab_flame-hab:+.4f}; both ages <8 Gyr so s_age=1.0).")
w()
w("## 2. Per-dimension percentile vs the 4,970 excellent cohort (>0.9)")
w()
w("Percentile = % of the excellent cohort with sub-score ≤ HD 28888's (ties noted for saturated dims).")
w()
w("| Dimension | HD score | pctile vs 4,970 | pctile vs full 12,234 |")
w("|---|---|---|---|")
for i,d in enumerate(DIMS):
    c=colmap[d]; pe=pct(exc[c],sub[i]); pf=pct(coh[c],sub[i])
    w(f"| {d} | {sub[i]:.4f} | {pe:.1f} | {pf:.1f} |")
w()
w("## 3. Overall percentile rank")
w()
w(f"- Composite hab_score = **{stored:.4f}**")
w(f"- vs **full FGK cohort (N={N})**: rank **{rank_full}** -> top **{100*rank_full/N:.2f}%** "
  f"(percentile {100-100*rank_full/N:.1f}). **Top 10%, NOT top 1%** "
  f"(1% cutoff {coh['hab_score'].quantile(0.99):.4f}, 0.1% cutoff {coh['hab_score'].quantile(0.999):.4f}, max {coh['hab_score'].max():.4f}).")
w(f"- vs **4,970 excellent (>0.9)**: rank **{rank_exc}** -> top **{100*rank_exc/Ne:.1f}%** "
  f"(percentile {100-100*rank_exc/Ne:.1f}). **NOT top 10%** within the excellent group.")
w()
w("## 4. Rank among the 18 actionable targets")
w()
w(f"By the full v2 composite hab_score, HD 28888 is **#{hd_rank_18} of 18** "
  f"(hab_score {stored:.4f}).")
w("The README's \"0.973\" **is** this v2 composite (0.9728), not a separate older/simpler score — "
  "so #1 on the actionable list and the 0.973 are the same scorer. Its #1 status comes from the "
  "**filter stack** (RUWE<1.4 single + <200 pc + 2-8 Gyr + zero RV coverage + no known planets), "
  f"NOT from having the highest chemistry score (it is {rank_full}th of {N} on chemistry alone).")
w("\nTop of the actionable list by hab_score:")
w("\n| rank | name | hab_score |")
w("|---|---|---|")
for i,r in dos_sorted.head(5).iterrows():
    w(f"| {i+1} | {r['name'].strip()} | {r['hab_score']:.4f} |")
w()
w("## 5. Strongest / weakest dimensions")
order=np.argsort(sub)
w("**Strongest (saturated at ceiling):** " + ", ".join(
    f"{DIMS[i]} ({sub[i]:.3f})" for i in order[::-1][:4]))
w("\n**Weakest (drag on composite):** " + ", ".join(
    f"{DIMS[i]} ({sub[i]:.3f})" for i in order[:3]))
w(f"\n- **[Fe/H] ({sub[2]:.3f})** is the single biggest drag (super-solar +0.13, weight 1.5).")
w(f"- **Volatile/[Ba/Fe] ({sub[7]:.3f})** is second ([Ba/Fe]=-0.04, just below the +0.05 optimum).")
w("- C/O and Age sit exactly at 1.0 (flat plateaus), contributing nothing to differentiation.")
w()
w("## Caveats")
w()
w("**(a) Subgiant vs calibration regime.** The scorer is not ML-trained; it is a hand-built "
  "geometric scorer validated against FGK planet hosts, with C/O Teff-detrend from a 360k FGK "
  "baseline. Its population cut is Teff 4000-7000 K and **log g > 3.8** (`habitability_v2.py:191`). "
  f"HD 28888 (log g 3.94) **passes** that cut and IS in the scored cohort -- it is NOT outside the "
  "calibration regime. But log g>3.8 admits turnoff/subgiants (true dwarfs are log g>4.2), and the "
  "entire actionable-18 sit at log g~3.8-4.0, so the sample is biased toward evolved stars. R=1.99 Rsun "
  "means the C/O Teff-detrend (calibrated on the broad FGK locus) is applied at the cool-subgiant edge.")
w()
w(f"**(b) First dredge-up / birth C/O.** Surface [N/Fe]=+0.38 with [C/Fe]=-0.07 is the CNO-mixing "
  f"signature. Conserving C+N (birth [N/Fe]=0) gives birth C/O ≈ **{co_birth_full:.3f}** (full) / "
  f"**{co_birth_half:.3f}** (50% mixing), vs surface {10**AC/10**AO:.3f}. Re-scored: birth C/O "
  f"Teff-corrected {co_birth_corr:.3f} -> s_CO {sub_birth[0]:.4f}; composite **{hab_birth:.4f}** "
  f"(Δ={hab_birth-hab:+.4f}). **No change**: C/O 0.31-0.43 all sit on the flat s_CO=1.0 plateau "
  "(0.15-0.65). The correction strengthens the *narrative* (natal C/O nearer solar) but not the score.")
w()
w("**(c) [Zr/Fe]=-0.60 outlier.** The implemented Volatile dimension uses **[Ba/Fe] only**; "
  "[Y/Fe] and [Zr/Fe] are NOT inputs to any scored dimension. So the unflagged Zr outlier has "
  "**zero impact** on the composite. (It would only matter if the s-process dimension used Y/Zr; "
  "the code does not.)")
w()
w("## Context dimensions (not in the implemented scorer)")
w("- **Birth cluster density (Teutsch_80):** a chemical-nearest-template label only; kinematics/"
  "position/age exclude real membership (cluster at 2.4 kpc, 132° away, 0.1-0.2 Gyr). Not scored.")
w("- **Galactic kinematic stability:** ecc 0.165, R_peri 6.33 / R_apo 8.83 kpc, z_max 0.063 kpc, "
  "J_R 34.4, J_Z 0.135, L_Z 1723 -- a quiet, near-circular thin-disk orbit. Not scored.")

open("hd28888_9d_scorer_result.md","w").write("\n".join(L))
print("\n".join(L))
print("\n[saved hd28888_9d_scorer_result.md]")

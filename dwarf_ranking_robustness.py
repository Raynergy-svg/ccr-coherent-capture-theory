"""
Phase 1: stress-test the 8D dwarf ranking.

Question: under reasonable perturbations to the scorer (weights, scoring-function
widths, dim subsets), does HD 183193 -- and the broader top-10 -- remain a
consistent pick, or is the ranking fragile?

Variants tested:
  V0  reference (the implemented 8D with original weights)
  V1  uniform weights (all 8 dims = 1.0)
  V2  weights doubled on [Fe/H] (test metallicity dominance)
  V3  weights doubled on alpha (Mg/Fe, Si/Fe)
  V4  drop volatile dim entirely
  V5  drop age dim entirely
  V6  drop both volatile and age (chemistry-only 6D)
  V7  Gaussian widths +/- 25% (looser tolerance)
  V8  Gaussian widths -25% (stricter)
  V9  Monte-Carlo: perturb each star's abundances by its GALAH errors (200 draws)

Reports:
  - top 10 from each variant
  - HD 183193's rank under each variant
  - Spearman rho of each variant's full ranking vs V0
  - MC: HD 183193's rank distribution; fraction of MC draws where HD 183193 in
    top-10 / top-5 / top-1
"""
import numpy as np, pandas as pd
from astropy.table import Table
from scipy.stats import spearmanr

# --- base scoring fns (vectorised) ---
def s_mgsi(x, w=0.4):  return np.clip(np.exp(-0.5*((x-1.02)/w)**2),0,1)
def s_feh(x, w=0.3):
    s=np.exp(-0.5*(x/w)**2); s=np.where(x<-0.5, s*np.exp(-((x+0.5)/0.2)**2),s); return np.clip(s,0,1)
def s_mgfe(x, w=0.15): return np.clip(np.exp(-0.5*(x/w)**2),0,1)
def s_sife(x, w=0.15): return np.clip(np.exp(-0.5*(x/w)**2),0,1)
def s_cafe(x, w=0.20): return np.clip(np.exp(-0.5*(x/w)**2),0,1)
def s_alfe(x, w=0.15): return np.clip(np.exp(-0.5*((x-0.05)/w)**2),0,1)
def s_vol(x, w=0.25):
    s=np.exp(-0.5*((x-0.05)/w)**2); s=np.where(x<-0.3, s*np.exp(-((x+0.3)/0.15)**2),s); return np.clip(s,0,1)
def s_age(a):
    s=np.ones_like(a,dtype=float)
    s=np.where(a<1, np.clip(a/1.0,0.1,1.0), s); s=np.where(a<0.3,0.1,s)
    s=np.where(a>8, np.exp(-0.5*((a-8.0)/4.0)**2), s); return np.clip(s,0,1)

# --- load dwarf pool ---
g=Table.read("galah_dr4_allstar_240705.fits",memmap=True).to_pandas()
g=g[(g.snr_px_ccd3>30)&(g.flag_sp==0)&(g.teff>4000)&(g.teff<7000)&(g.logg>3.8)]
g=g[g.mg_fe.notna()&g.si_fe.notna()&g.fe_h.notna()&(g.flag_mg_fe==0)&(g.flag_si_fe==0)&(g.flag_fe_h==0)]
g=g[g.age.notna()&(g.age>0)&(g.age<15)].reset_index(drop=True)
g["dist"]=np.where(g.parallax>0.2,1000/g.parallax,np.nan)
g["Mg_Si"]=10**(g.mg_fe-g.si_fe)
# nearby true dwarf sample
dw=g[(g.logg>=4.2)&(g.dist<200)].reset_index(drop=True)
print(f"nearby true-dwarf pool (logg>=4.2, <200 pc): {len(dw)}")
HD183193=6771181697822279936

def composite(subs, w):
    return np.exp(np.average(np.log(np.maximum(subs,1e-10)), weights=w, axis=1))

# build base sub-score matrix (8 dims)
ca_ok=dw.ca_fe.notna()&(dw.get("flag_ca_fe",pd.Series(0,index=dw.index))==0)
al_ok=dw.al_fe.notna()&(dw.get("flag_al_fe",pd.Series(0,index=dw.index))==0)
ba_ok=dw.ba_fe.notna()&(dw.get("flag_ba_fe",pd.Series(0,index=dw.index))==0)
def sub_matrix(width_scale=1.0):
    return np.column_stack([
        s_mgsi(dw.Mg_Si.values, w=0.4*width_scale),
        s_feh(dw.fe_h.values,   w=0.3*width_scale),
        s_mgfe(dw.mg_fe.values, w=0.15*width_scale),
        s_sife(dw.si_fe.values, w=0.15*width_scale),
        np.where(ca_ok, s_cafe(dw.ca_fe.fillna(0).values, w=0.2*width_scale), 0.8),
        np.where(al_ok, s_alfe(dw.al_fe.fillna(0).values, w=0.15*width_scale), 0.8),
        np.where(ba_ok, s_vol(dw.ba_fe.fillna(0).values, w=0.25*width_scale), 0.7),
        s_age(dw.age.values)
    ])
W0 = np.array([1.5,1.5,1.0,1.0,0.5,0.5,1.0,0.75])   # original
sub0 = sub_matrix(1.0)

# define variants
variants = []
def variant(label, mat, w):  variants.append((label, np.exp(np.average(np.log(np.maximum(mat,1e-10)),weights=w,axis=1))))

variant("V0 reference (8D, original weights)", sub0, W0)
variant("V1 uniform weights",                   sub0, np.ones(8))
variant("V2 2x weight on [Fe/H]",               sub0, np.array([1.5,3.0,1.0,1.0,0.5,0.5,1.0,0.75]))
variant("V3 2x weight on alpha (Mg, Si)",       sub0, np.array([1.5,1.5,2.0,2.0,0.5,0.5,1.0,0.75]))
# drop dims
mask4=np.array([1,1,1,1,1,1,0,1],bool); variant("V4 drop volatile",            sub0[:,mask4], W0[mask4])
mask5=np.array([1,1,1,1,1,1,1,0],bool); variant("V5 drop age",                  sub0[:,mask5], W0[mask5])
mask6=np.array([1,1,1,1,1,1,0,0],bool); variant("V6 drop volatile + age (6D)",  sub0[:,mask6], W0[mask6])
variant("V7 widths +25%",                       sub_matrix(1.25), W0)
variant("V8 widths -25%",                       sub_matrix(0.75), W0)

# --- print top-10 per variant + HD 183193 rank ---
print("\n" + "="*78)
print("TOP-10 NEARBY DWARFS BY VARIANT (composite score)")
print("="*78)
dw["gaia"]=dw.gaiadr3_source_id.astype("int64")
ranks_hd = {}
v0_order = None
for label, score in variants:
    o = pd.DataFrame({"gaia": dw["gaia"].values, "hab": score})
    o = o.sort_values("hab", ascending=False).reset_index(drop=True)
    if v0_order is None: v0_order = o["gaia"].values
    r_hd = int(o.index[o["gaia"]==HD183193][0])+1 if (o["gaia"]==HD183193).any() else None
    ranks_hd[label] = r_hd
    print(f"\n[{label}]  HD 183193 rank = {r_hd}")
    print(f"  top-10 Gaia IDs: {list(o['gaia'].head(10))}")

# --- Spearman rho of each variant's full ranking vs V0 ---
print("\n" + "="*78)
print("SPEARMAN RHO OF EACH VARIANT RANKING vs V0 (full nearby-dwarf pool)")
print("="*78)
v0_score = variants[0][1]
for label, score in variants:
    if label == variants[0][0]: continue
    rho, p = spearmanr(v0_score, score)
    print(f"  {label:42s}  rho = {rho:+.4f}  (p = {p:.2e})")

# --- top-10 overlap with V0 ---
print("\n" + "="*78)
print("TOP-10 OVERLAP WITH V0 (Jaccard)")
print("="*78)
top10_V0 = set(v0_order[:10])
for label, score in variants:
    o = pd.DataFrame({"gaia": dw["gaia"].values, "hab": score})
    o = o.sort_values("hab", ascending=False).reset_index(drop=True)
    t10 = set(o["gaia"].head(10))
    jac = len(top10_V0 & t10) / len(top10_V0 | t10)
    print(f"  {label:42s}  top-10 overlap = {len(top10_V0 & t10)}/10  Jaccard {jac:.2f}")

# --- V9: Monte Carlo over GALAH abundance errors ---
print("\n" + "="*78)
print("V9: MONTE CARLO over GALAH abundance errors (200 draws)")
print("="*78)
rng=np.random.default_rng(42)
ERR = {"mg_fe":"e_mg_fe","si_fe":"e_si_fe","fe_h":"e_fe_h","ca_fe":"e_ca_fe","al_fe":"e_al_fe","ba_fe":"e_ba_fe"}
def perturb(seed):
    r = rng if seed is None else np.random.default_rng(seed)
    d2 = dw.copy()
    for col,ec in ERR.items():
        if ec not in d2: continue
        sig = d2[ec].fillna(0.05).values
        d2[col] = d2[col].values + r.normal(0,sig)
    d2["Mg_Si"] = 10**(d2.mg_fe - d2.si_fe)
    sub = np.column_stack([
        s_mgsi(d2.Mg_Si.values), s_feh(d2.fe_h.values), s_mgfe(d2.mg_fe.values), s_sife(d2.si_fe.values),
        np.where(ca_ok, s_cafe(d2.ca_fe.fillna(0).values), 0.8),
        np.where(al_ok, s_alfe(d2.al_fe.fillna(0).values), 0.8),
        np.where(ba_ok, s_vol(d2.ba_fe.fillna(0).values), 0.7),
        s_age(d2.age.values)
    ])
    return np.exp(np.average(np.log(np.maximum(sub,1e-10)), weights=W0, axis=1))

N_MC=200
mc_ranks = np.zeros(N_MC,dtype=int)
mc_top10 = 0; mc_top5 = 0; mc_top1 = 0
all_top10_counts = {}
for k in range(N_MC):
    s = perturb(seed=k)
    o = pd.DataFrame({"gaia": dw["gaia"].values, "hab": s})
    o = o.sort_values("hab", ascending=False).reset_index(drop=True)
    r = int(o.index[o["gaia"]==HD183193][0])+1 if (o["gaia"]==HD183193).any() else len(o)+1
    mc_ranks[k] = r
    if r<=10: mc_top10 += 1
    if r<=5:  mc_top5  += 1
    if r==1:  mc_top1  += 1
    for gid in o["gaia"].head(10):
        all_top10_counts[gid] = all_top10_counts.get(gid,0)+1

print(f"HD 183193 MC rank distribution (N={N_MC} draws):")
print(f"  median rank: {int(np.median(mc_ranks))}  (5th pctile {int(np.percentile(mc_ranks,5))}, 95th {int(np.percentile(mc_ranks,95))})")
print(f"  in top-10: {mc_top10}/{N_MC} = {100*mc_top10/N_MC:.0f}%")
print(f"  in top-5:  {mc_top5}/{N_MC} = {100*mc_top5/N_MC:.0f}%")
print(f"  rank = 1:  {mc_top1}/{N_MC} = {100*mc_top1/N_MC:.0f}%")

print("\nMost-frequent top-10 members across MC (top 12):")
sorted_counts = sorted(all_top10_counts.items(), key=lambda x:-x[1])
for gid, n in sorted_counts[:12]:
    print(f"  Gaia DR3 {gid:>20}  in top-10 in {n}/{N_MC} draws ({100*n/N_MC:.0f}%)")

# --- consolidated HD 183193 rank table across all variants ---
print("\n" + "="*78)
print("HD 183193 RANK SUMMARY (out of {} nearby true dwarfs)".format(len(dw)))
print("="*78)
for label, r in ranks_hd.items():
    print(f"  {label:42s}  rank {r}")
print(f"  V9 Monte Carlo median rank                  rank {int(np.median(mc_ranks))} (5th-95th: {int(np.percentile(mc_ranks,5))}-{int(np.percentile(mc_ranks,95))})")
print("\nDONE")

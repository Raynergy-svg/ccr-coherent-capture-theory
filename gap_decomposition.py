"""
Phase 2: decompose the 21% nearby-dwarf vs 44% subgiant 8D-excellent gap.

Cohorts:
  subgiants: logg in (3.8, 4.0], any distance, 8D-eligible (the original 12,234
             cohort population, broadly).
  nearby dwarfs: logg >= 4.2, dist < 200 pc, 8D-eligible.

Tests:
  (a) Distributions of each scorer-input dim across the two cohorts.
  (b) Per-dim excellent rate isolated: which dim's sub-score distribution
      differs most between cohorts?
  (c) [Fe/H]-matched comparison: bin both cohorts in [Fe/H] = +/- 0.05
      windows, compute the excellent rate per bin per cohort; if at fixed
      [Fe/H] the excellent rates equalise -> gap is purely [Fe/H]-driven.
  (d) Same for age, [Mg/Fe], [Ba/Fe].
  (e) Multi-dim matched comparison: select subsets of each cohort matched
      on [Fe/H] AND age, compare excellent rates.
"""
import numpy as np, pandas as pd
from astropy.table import Table

def s_mgsi(x): return np.clip(np.exp(-0.5*((x-1.02)/0.4)**2),0,1)
def s_feh(x):
    s=np.exp(-0.5*(x/0.3)**2); s=np.where(x<-0.5,s*np.exp(-((x+0.5)/0.2)**2),s); return np.clip(s,0,1)
def s_mgfe(x): return np.clip(np.exp(-0.5*(x/0.15)**2),0,1)
def s_sife(x): return np.clip(np.exp(-0.5*(x/0.15)**2),0,1)
def s_cafe(x): return np.clip(np.exp(-0.5*(x/0.20)**2),0,1)
def s_alfe(x): return np.clip(np.exp(-0.5*((x-0.05)/0.15)**2),0,1)
def s_vol(x):
    s=np.exp(-0.5*((x-0.05)/0.25)**2); s=np.where(x<-0.3,s*np.exp(-((x+0.3)/0.15)**2),s); return np.clip(s,0,1)
def s_age(a):
    s=np.ones_like(a,dtype=float)
    s=np.where(a<1, np.clip(a/1.0,0.1,1.0), s); s=np.where(a<0.3,0.1,s)
    s=np.where(a>8, np.exp(-0.5*((a-8.0)/4.0)**2), s); return np.clip(s,0,1)
W=np.array([1.5,1.5,1.0,1.0,0.5,0.5,1.0,0.75])

g=Table.read("galah_dr4_allstar_240705.fits",memmap=True).to_pandas()
g=g[(g.snr_px_ccd3>30)&(g.flag_sp==0)&(g.teff>4000)&(g.teff<7000)&(g.logg>3.8)]
g=g[g.mg_fe.notna()&g.si_fe.notna()&g.fe_h.notna()&(g.flag_mg_fe==0)&(g.flag_si_fe==0)&(g.flag_fe_h==0)]
g=g[g.age.notna()&(g.age>0)&(g.age<15)].reset_index(drop=True)
g["dist"]=np.where(g.parallax>0.2,1000/g.parallax,np.nan)
g["Mg_Si"]=10**(g.mg_fe-g.si_fe)
ca_ok=g.ca_fe.notna()&(g.flag_ca_fe==0); al_ok=g.al_fe.notna()&(g.flag_al_fe==0); ba_ok=g.ba_fe.notna()&(g.flag_ba_fe==0)
sub=np.column_stack([s_mgsi(g.Mg_Si.values),s_feh(g.fe_h.values),s_mgfe(g.mg_fe.values),s_sife(g.si_fe.values),
    np.where(ca_ok,s_cafe(g.ca_fe.fillna(0).values),0.8),
    np.where(al_ok,s_alfe(g.al_fe.fillna(0).values),0.8),
    np.where(ba_ok,s_vol(g.ba_fe.fillna(0).values),0.7),
    s_age(g.age.values)])
g["hab8"]=np.exp(np.average(np.log(np.maximum(sub,1e-10)),weights=W,axis=1))
for i,d in enumerate(["sMgSi","sFeH","sMgFe","sSiFe","sCaFe","sAlFe","sVol","sAge"]):
    g[d]=sub[:,i]

sg=g[(g.logg>3.8)&(g.logg<=4.0)]                        # subgiants
nd=g[(g.logg>=4.2)&(g.dist<200)]                         # nearby dwarfs
print(f"subgiants (logg 3.80-4.00):  {len(sg)}  excellent rate {(sg.hab8>0.9).mean():.1%}")
print(f"nearby dwarfs (logg>=4.2, <200 pc): {len(nd)}  excellent rate {(nd.hab8>0.9).mean():.1%}")
print(f"GAP: {(sg.hab8>0.9).mean()-((nd.hab8>0.9).mean()):.1%}")

# (a) distribution stats per dim
print("\n" + "="*78)
print("(a) DIM DISTRIBUTIONS: subgiants vs nearby dwarfs (median +/- p16-p84 half-width)")
print("="*78)
for col,lbl in [("fe_h","[Fe/H]"),("mg_fe","[Mg/Fe]"),("si_fe","[Si/Fe]"),
                ("ca_fe","[Ca/Fe]"),("al_fe","[Al/Fe]"),("ba_fe","[Ba/Fe]"),
                ("age","age (Gyr)"),("Mg_Si","Mg/Si number")]:
    sm=np.nanmedian(sg[col]); sw=(np.nanpercentile(sg[col],84)-np.nanpercentile(sg[col],16))/2
    dm=np.nanmedian(nd[col]); dw=(np.nanpercentile(nd[col],84)-np.nanpercentile(nd[col],16))/2
    print(f"  {lbl:14s}  subgiants {sm:+.3f} +/- {sw:.3f}   nearby dwarfs {dm:+.3f} +/- {dw:.3f}   "
          f"(width ratio dwarf/sg = {dw/sw if sw>0 else 0:.2f})")

# (b) per-dim sub-score means (which dim drags dwarfs?)
print("\n" + "="*78)
print("(b) PER-DIM MEAN SUB-SCORE: subgiants vs nearby dwarfs")
print("="*78)
print(f"  {'dim':10s}  {'subg':>6} {'dwarf':>6} {'diff':>7}")
for d in ["sMgSi","sFeH","sMgFe","sSiFe","sCaFe","sAlFe","sVol","sAge"]:
    sm=sg[d].mean(); dm=nd[d].mean()
    print(f"  {d:10s}  {sm:>6.3f} {dm:>6.3f} {dm-sm:>+7.3f}")

# (c) excellent rate vs [Fe/H] (matched-bin comparison)
print("\n" + "="*78)
print("(c) EXCELLENT RATE vs [Fe/H] BIN (matched)")
print("="*78)
bins=np.arange(-0.6, 0.42, 0.1)
print(f"  {'[Fe/H] bin':12s}  {'N_sg':>6} {'exc_sg':>8} | {'N_dw':>6} {'exc_dw':>8} | {'delta':>7}")
for lo,hi in zip(bins[:-1],bins[1:]):
    msg=sg[(sg.fe_h>=lo)&(sg.fe_h<hi)]; mdw=nd[(nd.fe_h>=lo)&(nd.fe_h<hi)]
    if len(msg)<20 or len(mdw)<20: continue
    es=(msg.hab8>0.9).mean(); ed=(mdw.hab8>0.9).mean()
    print(f"  [{lo:+.2f},{hi:+.2f})  {len(msg):>6} {es:>7.1%}  | {len(mdw):>6} {ed:>7.1%}  | {ed-es:>+6.1%}")

# (d) excellent rate vs age bin
print("\n" + "="*78)
print("(d) EXCELLENT RATE vs AGE BIN (matched)")
print("="*78)
abins=[0,1,2,3,4,5,6,7,8,10,15]
print(f"  {'age (Gyr)':12s}  {'N_sg':>6} {'exc_sg':>8} | {'N_dw':>6} {'exc_dw':>8} | {'delta':>7}")
for lo,hi in zip(abins[:-1],abins[1:]):
    msg=sg[(sg.age>=lo)&(sg.age<hi)]; mdw=nd[(nd.age>=lo)&(nd.age<hi)]
    if len(msg)<20 or len(mdw)<20: continue
    es=(msg.hab8>0.9).mean(); ed=(mdw.hab8>0.9).mean()
    print(f"  [{lo:>4.1f},{hi:>4.1f})  {len(msg):>6} {es:>7.1%}  | {len(mdw):>6} {ed:>7.1%}  | {ed-es:>+6.1%}")

# (e) joint [Fe/H] AND age matching
print("\n" + "="*78)
print("(e) JOINT [Fe/H] + AGE MATCHING: dwarf excellent rate when matched to subgiant distribution")
print("="*78)
# For each subgiant, find a random dwarf in the same (fe_h+-0.05, age+-1) bin
# Stratified sampling: for each fe_h, age bin, compute excellent rate in both cohorts, weighted by subgiant pop
sg_w=sg.copy(); nd_w=nd.copy()
sg_w["bin_fe"]=pd.cut(sg_w.fe_h, bins=np.arange(-0.6,0.5,0.1), labels=False)
sg_w["bin_age"]=pd.cut(sg_w.age, bins=np.arange(0,11,1), labels=False)
nd_w["bin_fe"]=pd.cut(nd_w.fe_h, bins=np.arange(-0.6,0.5,0.1), labels=False)
nd_w["bin_age"]=pd.cut(nd_w.age, bins=np.arange(0,11,1), labels=False)
sg_dist = sg_w.groupby(["bin_fe","bin_age"]).size()
total_sg = sg_dist.sum()
# weighted dwarf excellent rate using subgiant joint distribution as weights
weighted_dwarf_exc = 0.0; weight_sum = 0.0
for (bf,ba), nsg in sg_dist.items():
    dw_bin = nd_w[(nd_w.bin_fe==bf)&(nd_w.bin_age==ba)]
    if len(dw_bin) < 5: continue
    weighted_dwarf_exc += (dw_bin.hab8>0.9).mean() * nsg
    weight_sum += nsg
weighted_dwarf_exc /= weight_sum
print(f"  unweighted subgiant excellent rate: {(sg.hab8>0.9).mean():.1%}")
print(f"  unweighted nearby-dwarf excellent rate: {(nd.hab8>0.9).mean():.1%}")
print(f"  dwarf rate after re-weighting to subgiant [Fe/H]+age joint distribution: {weighted_dwarf_exc:.1%}")
print(f"  -> gap reduction: {((nd.hab8>0.9).mean()):.1%} -> {weighted_dwarf_exc:.1%}  "
      f"(closes by {(weighted_dwarf_exc-(nd.hab8>0.9).mean())/((sg.hab8>0.9).mean()-(nd.hab8>0.9).mean())*100:.0f}%)")
print("\nDONE")

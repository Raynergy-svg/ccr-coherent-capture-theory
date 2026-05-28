"""
Cohort caveat check: the README/T-series claim '4,970 stars score excellent
(>0.9) on 9D habitability ~ 41% of FGK dwarfs' is based on the 12,234 cohort
which is all logg 3.80-4.00 (subgiants/turnoff). What is the excellent fraction
when computed properly on the FULL FGK sample, split by subgiant vs true dwarf,
using the dwarf-compatible 8D scorer?
"""
import numpy as np, pandas as pd
from astropy.table import Table
def s_mgsi(x): return np.clip(np.exp(-0.5*((x-1.02)/0.4)**2),0,1)
def s_feh(x):
    s=np.exp(-0.5*((x-0.0)/0.3)**2); s=np.where(x<-0.5,s*np.exp(-((x+0.5)/0.2)**2),s); return np.clip(s,0,1)
def s_mgfe(x): return np.clip(np.exp(-0.5*(x/0.15)**2),0,1)
def s_sife(x): return np.clip(np.exp(-0.5*(x/0.15)**2),0,1)
def s_cafe(x): return np.clip(np.exp(-0.5*(x/0.20)**2),0,1)
def s_alfe(x): return np.clip(np.exp(-0.5*((x-0.05)/0.15)**2),0,1)
def s_vol(x):
    s=np.exp(-0.5*((x-0.05)/0.25)**2); s=np.where(x<-0.3,s*np.exp(-((x+0.3)/0.15)**2),s); return np.clip(s,0,1)
def s_age(a):
    a=a.astype(float); s=np.ones_like(a)
    s=np.where(a<1.0,np.clip(a/1.0,0.1,1.0),s); s=np.where(a<0.3,0.1,s)
    s=np.where(a>8.0,np.exp(-0.5*((a-8.0)/4.0)**2),s); return np.clip(s,0,1)
W=np.array([1.5,1.5,1.0,1.0,0.5,0.5,1.0,0.75])

g=Table.read("galah_dr4_allstar_240705.fits",memmap=True).to_pandas()
print(f"raw GALAH: {len(g)}")
g=g[(g.snr_px_ccd3>30)&(g.flag_sp==0)&(g.teff>4000)&(g.teff<7000)&(g.logg>3.8)].copy()
print(f"+ FGK quality + logg>3.8: {len(g)}")
g=g[g.mg_fe.notna()&g.si_fe.notna()&g.fe_h.notna()&(g.flag_mg_fe==0)&(g.flag_si_fe==0)&(g.flag_fe_h==0)]
g=g[g.age.notna()&(g.age>0)&(g.age<15)].reset_index(drop=True)
print(f"+ Mg/Si/Fe/age clean (8D-eligible): {len(g)}")

g["Mg_Si"]=10**(g.mg_fe-g.si_fe)
ca_ok=g.ca_fe.notna()&(g.flag_ca_fe==0); al_ok=g.al_fe.notna()&(g.flag_al_fe==0); ba_ok=g.ba_fe.notna()&(g.flag_ba_fe==0)
sc=np.column_stack([s_mgsi(g.Mg_Si.values),s_feh(g.fe_h.values),s_mgfe(g.mg_fe.values),s_sife(g.si_fe.values),
    np.where(ca_ok,s_cafe(g.ca_fe.fillna(0).values),0.8),
    np.where(al_ok,s_alfe(g.al_fe.fillna(0).values),0.8),
    np.where(ba_ok,s_vol(g.ba_fe.fillna(0).values),0.7),
    s_age(g.age.values)])
g["hab8"]=np.exp(np.average(np.log(np.maximum(sc,1e-10)),weights=W,axis=1))

def summarise(label, sub):
    n=len(sub); e=(sub.hab8>0.9).sum(); v=(sub.hab8>0.95).sum()
    print(f"  {label:<48}  N={n:>6}  hab8>0.9: {e:>6} ({100*e/max(n,1):>5.1f}%)   hab8>0.95: {v:>6} ({100*v/max(n,1):>5.1f}%)")

print("\n=== 8D EXCELLENT FRACTION BY EVOLUTIONARY STATE ===")
summarise("ALL 8D-eligible (FGK, logg>3.8)", g)
summarise("subgiants/turnoff (3.80 < logg <= 4.00)", g[(g.logg>3.8)&(g.logg<=4.0)])
summarise("intermediate (4.00 < logg < 4.20)",       g[(g.logg>4.0)&(g.logg<4.2)])
summarise("true MS dwarfs (logg >= 4.20)",            g[g.logg>=4.2])
summarise("solar-twin dwarfs (logg>=4.2, 5600-5900 K)",g[(g.logg>=4.2)&(g.teff>5600)&(g.teff<5900)])

# Also: nearby restricted
gn=g[(g.parallax>5)]  # ~within 200pc
print("\nrestricted to <200 pc (plx>5 mas):")
summarise("ALL 8D-eligible, <200pc", gn)
summarise("subgiants <200pc", gn[(gn.logg>3.8)&(gn.logg<=4.0)])
summarise("true dwarfs (logg>=4.2) <200pc", gn[gn.logg>=4.2])

# original 9D cohort claim recompute
print("\n=== COMPARISON TO README 9D CLAIM ===")
print(f"README: '4,970 stars score excellent on 9D' from 12,234 = 40.6%")
print(f"  --- those 12,234 are all logg 3.80-4.00 (subgiants/turnoff)")
print(f"  --- 9D not computable on true dwarfs (C/O unmeasured)")
print(f"\nProperly framed under 8D:")
all_e=(g.hab8>0.9).sum(); all_n=len(g)
print(f"  full 8D-eligible FGK (logg>3.8): {all_e}/{all_n} = {100*all_e/all_n:.1f}% excellent under 8D")
sg=g[(g.logg>3.8)&(g.logg<=4.0)]; sg_e=(sg.hab8>0.9).sum()
print(f"  subgiants (3.80<logg<=4.00):     {sg_e}/{len(sg)} = {100*sg_e/len(sg):.1f}% (≈ the old '41%' figure)")
dw=g[g.logg>=4.2]; dw_e=(dw.hab8>0.9).sum()
print(f"  true dwarfs (logg>=4.2):         {dw_e}/{len(dw)} = {100*dw_e/len(dw):.1f}% excellent under 8D")
print("DONE")

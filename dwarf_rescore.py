"""
Re-run the 9D habitability scorer (exact habitability_v2.py functions) on the
FULL GALAH FGK sample -- including the log g > 4.0 main-sequence dwarfs that the
committed cohort (capped at log g <= 4.0, Teff <= 5750) excluded. Hunt the best
NEARBY TRUE DWARFS by hab_score and compare to HD 28888 / the actionable 18.
"""
import numpy as np, pandas as pd
from astropy.table import Table

SOLAR_CO, REF_TEFF = 0.549, 5778.0
SLOPE, INTC = -0.000527, 3.5242
def teff_corr(co,teff): return np.clip(co+((SLOPE*REF_TEFF+INTC)-(SLOPE*teff+INTC)),0.05,3.0)
def s_co(co):
    s=np.exp(-0.5*np.maximum(0,co-0.65)**2/0.15**2); s=np.where(co<0.15,s*co/0.15,s); return np.clip(s,0,1)
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
W=np.array([1.0,1.5,1.5,1.0,1.0,0.5,0.5,1.0,0.75])

g=Table.read("galah_dr4_allstar_240705.fits",memmap=True).to_pandas()
print("raw:",len(g))
g=g[(g.snr_px_ccd3>30)&(g.flag_sp==0)].copy()
for col,fc in [("c_fe","flag_c_fe"),("o_fe","flag_o_fe"),("mg_fe","flag_mg_fe"),("si_fe","flag_si_fe"),("fe_h","flag_fe_h")]:
    g=g[g[col].notna()]
    if fc in g: g=g[g[fc]==0]
g["C_O_raw"]=10.0**(g.c_fe-g.o_fe)*SOLAR_CO
g["Mg_Si"]=10.0**(g.mg_fe-g.si_fe)
g=g[(g.C_O_raw>0.05)&(g.C_O_raw<3.0)]
# FGK per habitability_v2.py line 191 (NO upper logg/teff cap)
g=g[(g.teff>4000)&(g.teff<7000)&(g.logg>3.8)].reset_index(drop=True)
print("FGK (teff4000-7000, logg>3.8):",len(g))
g["C_O"]=teff_corr(g.C_O_raw.values,g.teff.values)
sc=np.column_stack([s_co(g.C_O.values),s_mgsi(g.Mg_Si.values),s_feh(g.fe_h.values),
    s_mgfe(g.mg_fe.values),s_sife(g.si_fe.values),
    np.where(g.ca_fe.notna(),s_cafe(g.ca_fe.fillna(0).values),0.8),
    np.where(g.al_fe.notna(),s_alfe(g.al_fe.fillna(0).values),0.8),
    np.where(g.ba_fe.notna()&(g.get("flag_ba_fe",pd.Series(0,index=g.index))==0),s_vol(g.ba_fe.fillna(0).values),0.7),
    np.where(g.age.notna()&(g.age>0)&(g.age<15),s_age(g.age.fillna(5).values),0.7)])
g["hab_score"]=np.exp(np.average(np.log(np.maximum(sc,1e-10)),weights=W,axis=1))
g["dist"]=np.where(g.parallax>0.2,1000/g.parallax,np.nan)

# validate: HD 28888 reproduces?
hd=g[g.gaiadr3_source_id==3312653647717790720]
if len(hd): print(f"validate HD 28888: hab={hd.hab_score.iloc[0]:.4f} logg={hd.logg.iloc[0]:.2f} (expect ~0.9728)")

dwarf=g[g.logg>=4.2]   # main-sequence dwarfs
print(f"\nFull FGK scored: {len(g)} | dwarfs(logg>=4.2): {len(dwarf)} | excellent dwarfs(>0.9): {(dwarf.hab_score>0.9).sum()}")
print(f"excellent dwarfs <200pc: {((dwarf.hab_score>0.9)&(dwarf.dist<200)).sum()}  <100pc: {((dwarf.hab_score>0.9)&(dwarf.dist<100)).sum()}")
hd_hab=0.9728
better_nearby=((dwarf.hab_score>hd_hab)&(dwarf.dist<200)).sum()
print(f"nearby(<200pc) dwarfs beating HD 28888 (hab>{hd_hab}): {better_nearby}")
print(f"nearby(<100pc) dwarfs beating HD 28888: {((dwarf.hab_score>hd_hab)&(dwarf.dist<100)).sum()}")

cols=["gaiadr3_source_id","tmass_id","hab_score","dist","teff","logg","age","fe_h","C_O","Mg_Si","parallax"]
print("\n=== TOP 20 NEARBY (<200pc) TRUE DWARFS (logg>=4.2) by hab_score ===")
nd=dwarf[dwarf.dist<200].sort_values("hab_score",ascending=False)
print(nd.head(20)[cols].to_string(index=False))
nd.head(200).to_csv("dwarf_nearby_top200.csv",index=False)
g[g.logg>=4.2].sort_values("hab_score",ascending=False).head(1000).to_csv("dwarf_all_top1000.csv",index=False)
print(f"\nsaved dwarf_nearby_top200.csv ({len(nd)} nearby dwarfs<200pc), dwarf_all_top1000.csv")
print("DONE")

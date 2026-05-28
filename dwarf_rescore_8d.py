"""
Reduced 8D scorer (drops C/O; the GALAH DR4 flag_c_fe is unreliable for every
FGK dwarf). Uses the implemented habitability_v2.py functions for the remaining
8 dimensions:  Mg/Si, [Fe/H], [Mg/Fe], [Si/Fe], [Ca/Fe], [Al/Fe], Volatile(Ba/Fe), Age.

Ranks the full GALAH FGK (logg>3.8, no upper cap) and hunts the best NEARBY
TRUE DWARFS (logg>=4.2, dist<200 pc) -- the real habitability targets that the
committed 9D scorer excluded by requiring C/O.
"""
import io, time, requests
import numpy as np, pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

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
W=np.array([1.5,1.5,1.0,1.0,0.5,0.5,1.0,0.75])  # Mg/Si,[Fe/H],[Mg/Fe],[Si/Fe],[Ca/Fe],[Al/Fe],Vol,Age

g=Table.read("galah_dr4_allstar_240705.fits",memmap=True).to_pandas()
# FGK quality + the 8D-required clean abundances + age
g=g[(g.snr_px_ccd3>30)&(g.flag_sp==0)&(g.teff>4000)&(g.teff<7000)&(g.logg>3.8)].copy()
need={"mg_fe":"flag_mg_fe","si_fe":"flag_si_fe","fe_h":"flag_fe_h"}
for c,f in need.items():
    g=g[g[c].notna()&(g[f]==0)]
g=g[g.age.notna()&(g.age>0)&(g.age<15)].reset_index(drop=True)
print("FGK 8D-eligible:",len(g))

g["Mg_Si"]=10.0**(g.mg_fe-g.si_fe)
ca_ok=g.ca_fe.notna()&(g.get("flag_ca_fe",pd.Series(0,index=g.index))==0)
al_ok=g.al_fe.notna()&(g.get("flag_al_fe",pd.Series(0,index=g.index))==0)
ba_ok=g.ba_fe.notna()&(g.get("flag_ba_fe",pd.Series(0,index=g.index))==0)
sc=np.column_stack([
    s_mgsi(g.Mg_Si.values), s_feh(g.fe_h.values), s_mgfe(g.mg_fe.values), s_sife(g.si_fe.values),
    np.where(ca_ok, s_cafe(g.ca_fe.fillna(0).values), 0.8),
    np.where(al_ok, s_alfe(g.al_fe.fillna(0).values), 0.8),
    np.where(ba_ok, s_vol(g.ba_fe.fillna(0).values), 0.7),
    s_age(g.age.values)
])
g["hab8"]=np.exp(np.average(np.log(np.maximum(sc,1e-10)),weights=W,axis=1))
g["dist"]=np.where(g.parallax>0.2,1000/g.parallax,np.nan)

# how does HD 28888 rank under reduced 8D?
hd=g[g.gaiadr3_source_id==3312653647717790720]
if len(hd):
    hd_h8=float(hd.hab8.iloc[0])
    hd_rank_full=int((g.hab8>hd_h8).sum())+1
    print(f"\nHD 28888 under reduced 8D: hab8={hd_h8:.4f} | rank {hd_rank_full}/{len(g)}")

dw=g[(g.logg>=4.2)&(g.dist<200)].copy()
print(f"\nnearby (<200pc) true dwarfs (logg>=4.2) scored under 8D: {len(dw)}")
print(f"  excellent under 8D (>0.9): {(dw.hab8>0.9).sum()}")
print(f"  beating HD28888 (hab8>{hd_h8:.4f}): {(dw.hab8>hd_h8).sum()}")
print(f"  also age 2-8 + beating HD 28888: {((dw.hab8>hd_h8)&(dw.age>2)&(dw.age<8)).sum()}")

cands=dw.sort_values("hab8",ascending=False).head(40).copy()
cands.to_csv("dwarf8_top40_nearby.csv",index=False)
print("\nTOP 25 NEARBY DWARFS BY 8D HAB SCORE:")
cols=["gaiadr3_source_id","hab8","dist","teff","logg","age","fe_h","Mg_Si","mg_fe","si_fe","al_fe","ca_fe","ba_fe","ra","dec"]
print(cands.head(25)[["gaiadr3_source_id","hab8","dist","teff","logg","age","fe_h","Mg_Si"]].to_string(index=False))

# Gaia query for the top 25 nearby dwarfs: RUWE, G, NSS  -> apply actionable filters
ids=cands.head(40)["gaiadr3_source_id"].astype("int64").tolist()
def tap(url,q,retries=4):
    for a in range(retries):
        try:
            r=requests.post(url,data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q},timeout=180)
            if r.status_code==200: return pd.read_csv(io.StringIO(r.text))
        except Exception: pass
        time.sleep(2**a)
    return pd.DataFrame()
GAIA="https://gea.esac.esa.int/tap-server/tap/sync"
NEA="https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
print("\nQuerying Gaia for top-40 nearby dwarfs...")
gx=tap(GAIA,f"""SELECT source_id, parallax, phot_g_mean_mag, ruwe, non_single_star
   FROM gaiadr3.gaia_source WHERE source_id IN ({",".join(str(i) for i in ids)})""")
gx=gx.rename(columns={"source_id":"gaiadr3_source_id"})
cands40=cands.head(40).merge(gx,on="gaiadr3_source_id",how="left",suffixes=("","_g"))
print(f"  Gaia rows: {len(gx)}")
print("\nQuerying NASA Exoplanet Archive for planet hosts...")
nea=tap(NEA,"SELECT pl_name,hostname,ra,dec FROM pscomppars")
has_planet=np.zeros(len(cands40),dtype=bool)
if len(nea):
    pc=SkyCoord(nea.ra.values*u.deg,nea.dec.values*u.deg)
    tc=SkyCoord(cands40.ra.values*u.deg,cands40.dec.values*u.deg)
    idx,sep,_=tc.match_to_catalog_sky(pc)
    has_planet=(sep.arcsec<5)
cands40["has_planet"]=has_planet
cands40["constellation"]=[SkyCoord(r.ra*u.deg,r.dec*u.deg).get_constellation() for _,r in cands40.iterrows()]

# Apply actionable filters: RUWE<1.4, NSS=0, dist<200, G<12, age 2-8, no planet
def passes(r):
    return (pd.notna(r.ruwe) and r.ruwe<1.4 and (pd.isna(r.non_single_star) or r.non_single_star==0)
            and pd.notna(r.parallax_g) and 1000/r.parallax_g<200 and pd.notna(r.phot_g_mean_mag) and r.phot_g_mean_mag<12
            and 2.0<=r.age<=8.0 and not r.has_planet)
cands40["actionable"]=cands40.apply(passes,axis=1)
print(f"\nPass ALL actionable filters: {cands40.actionable.sum()}/{len(cands40)}")
print("\n>>> REAL NEARBY DWARF HABITABILITY TARGETS (8D, actionable) <<<")
out=cands40[cands40.actionable].sort_values("hab8",ascending=False)
hd28888_hab=hd_h8
print(f"{'rk':>2} {'Gaia DR3':>20} {'hab8':>6} {'dist':>5} {'G':>5} {'RUWE':>5} {'Teff':>5} {'logg':>5} {'age':>4} {'Dec':>7} {'N?':>3} {'constellation':>14} {'vs HD28888':>10}")
for i,r in out.iterrows():
    dGa=1000/r.parallax_g if pd.notna(r.parallax_g) else np.nan
    print(f"{len(out[:list(out.index).index(i)+1]):>2} {int(r.gaiadr3_source_id):>20} {r.hab8:>6.4f} {dGa:>5.0f} {r.phot_g_mean_mag:>5.1f} "
          f"{r.ruwe:>5.2f} {r.teff:>5.0f} {r.logg:>5.2f} {r.age:>4.1f} {r.dec:>7.2f} {'Y' if r.dec>-30 else 'n':>3} "
          f"{r.constellation:>14} {('+' if r.hab8>hd28888_hab else '-'):>3}{abs(r.hab8-hd28888_hab):>7.4f}")
out.to_csv("real_dwarf_targets.csv",index=False)
print(f"\nsaved real_dwarf_targets.csv ({len(out)} actionable dwarfs)")
print("DONE")

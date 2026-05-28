"""
Relaxation analysis: which excellent-chemistry (>0.9), optimal-age (2-8 Gyr),
single (RUWE<1.4, NSS=0), planet-free FGK stars sit just outside the actionable
cliff -- distance 200-300 pc and/or G 12-12.5 -- and would be reachable with
ESPRESSO/HARPS? Ranked by hab_score.
"""
import io, time, requests
import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

GAIA="https://gea.esac.esa.int/tap-server/tap/sync"
NEA="https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
def tap(url,q,retries=4):
    for a in range(retries):
        try:
            r=requests.post(url,data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q},timeout=180)
            if r.status_code==200: return pd.read_csv(io.StringIO(r.text))
            err=f"HTTP {r.status_code}: {r.text[:150]}"
        except Exception as e: err=repr(e)
        time.sleep(2**a)
    print("[WARN]",err); return pd.DataFrame()

c=pd.read_csv("habitability_v2_targets.csv")
c["dist0"]=np.where(c["parallax"]>0.2,1000/c["parallax"],np.nan)
# candidate pool: excellent + age 2-8 + just outside cliff (cohort-dist 200-350 to be safe)
pool=c[(c["hab_score"]>0.9)&(c["age"]>2)&(c["age"]<8)&(c["dist0"]>200)&(c["dist0"]<=350)].copy()
ids=pool["gaiadr3_source_id"].astype("int64").tolist()
print(f"candidate pool (excellent, age2-8, 200-350pc): {len(ids)}")
idlist=",".join(str(i) for i in ids)

g=tap(GAIA,f"""SELECT source_id, ra, dec, parallax, phot_g_mean_mag, ruwe, non_single_star
   FROM gaiadr3.gaia_source WHERE source_id IN ({idlist})""").rename(columns={"source_id":"gaiadr3_source_id"})
print(f"Gaia rows: {len(g)}")
pool=pool.merge(g,on="gaiadr3_source_id",how="left",suffixes=("","_g"))
pool["dist"]=np.where(pool["parallax_g"]>0.2,1000/pool["parallax_g"],pool["dist0"])

# planet cross-match (positional 5")
nea=tap(NEA,"SELECT pl_name,hostname,ra,dec FROM pscomppars")
pool["has_planet"]=False
if len(nea):
    pc=SkyCoord(nea["ra"].values*u.deg,nea["dec"].values*u.deg)
    tc=SkyCoord(pool["ra"].values*u.deg,pool["dec"].values*u.deg)
    idx,sep,_=tc.match_to_catalog_sky(pc)
    pool["has_planet"]=sep.arcsec<5

# relaxed filters: dist<300, G<12.5, RUWE<1.4, NSS=0, no planet
def col(p,n): return p[n] if n in p else np.nan
pool["ruwe_v"]=col(pool,"ruwe"); pool["G"]=col(pool,"phot_g_mean_mag"); pool["nss_v"]=col(pool,"non_single_star")
keep=pool[(pool["dist"]<300)&(pool["G"]<12.5)&(pool["ruwe_v"]<1.4)&
          ((pool["nss_v"]==0)|pool["nss_v"].isna())&(~pool["has_planet"])].copy()
keep=keep.sort_values("hab_score",ascending=False).reset_index(drop=True)
print(f"\nReachable backups (dist<300pc, G<12.5, single, planet-free, excellent, age2-8): {len(keep)}")

# crude ESPRESSO photon-limited RV exposure for ~1 m/s on a G~Teff5700 dwarf/subgiant
# scale: ESPRESSO reaches ~1 m/s on G~11 in ~900s; flux ~10^(-0.4*(G-11)); t ~ 900*10^(0.4*(G-11)) for fixed SNR
def esp_min(G):
    if pd.isna(G): return None
    return 900*10**(0.4*(G-11))/60.0  # minutes for ~1 m/s-equivalent SNR
coords=SkyCoord(keep["ra"].values*u.deg,keep["dec"].values*u.deg)
keep["constellation"]=[x.get_constellation() for x in coords]

cols=["gaiadr3_source_id","hab_score","dist","G","ruwe_v","nss_v","teff","logg","age","ra","dec","constellation","has_planet"]
keep["esp_min_1ms"]=keep["G"].map(esp_min)
keep.to_csv("relaxation_backups.csv",index=False)
print("\nTOP REACHABLE BACKUPS (ranked by hab_score):")
print(f"{'rk':>2} {'Gaia DR3':>20} {'hab':>6} {'dist':>5} {'G':>5} {'RUWE':>5} {'age':>4} "
      f"{'Teff':>5} {'logg':>5} {'Dec':>7} {'N?':>3} {'ESPmin/1ms':>10} {'constellation':>14}")
for i,r in keep.head(15).iterrows():
    north="Y" if r["dec"]>-30 else "n"
    em=f"{r['esp_min_1ms']:.0f}" if pd.notna(r['esp_min_1ms']) else "--"
    print(f"{i+1:>2} {int(r['gaiadr3_source_id']):>20} {r['hab_score']:>6.4f} {r['dist']:>5.0f} {r['G']:>5.1f} "
          f"{r['ruwe_v']:>5.2f} {r['age']:>4.1f} {r['teff']:>5.0f} {r['logg']:>5.2f} {r['dec']:>7.2f} {north:>3} {em:>10} {r['constellation']:>14}")
print(f"\nNorthern-accessible (Dec>-30) among reachable: {(keep['dec']>-30).sum()}/{len(keep)}")
print(f"logg<4.0 (subgiant): {(keep['logg']<4.0).sum()}/{len(keep)}  | logg>=4.0 (dwarf): {(keep['logg']>=4.0).sum()}/{len(keep)}")
print("DONE")

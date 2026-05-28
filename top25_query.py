"""
Top-25 chemistry leaders from the full FGK cohort (N=12,234), with full
observability + actionable-filter audit. Pulls Gaia DR3 (RUWE, G, parallax,
NSS, radius), SIMBAD (names, RV, reference count), and the NASA Exoplanet
Archive (known planets). Flags any star passing ALL actionable hard filters
that is NOT on the 18-target list (= pipeline gap).
"""
import io, time, requests
import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

GAIA="https://gea.esac.esa.int/tap-server/tap/sync"
SIMBAD="https://simbad.u-strasbg.fr/simbad/sim-tap/sync"
NEA="https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def tap(url, q, fmt="csv", retries=4, params=None):
    base={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":fmt,"QUERY":q}
    if params: base.update(params)
    for a in range(retries):
        try:
            r=requests.post(url, data=base, timeout=180)
            if r.status_code==200:
                return pd.read_csv(io.StringIO(r.text))
            err=f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e: err=repr(e)
        time.sleep(2**a)
    print(f"[WARN] query failed: {err}")
    return pd.DataFrame()

# ---- scoring fns for Ca/Fe, Al/Fe (not saved in cohort csv) ----
def s_cafe(x): return np.clip(np.exp(-0.5*(x/0.20)**2),0,1)
def s_alfe(x): return np.clip(np.exp(-0.5*((x-0.05)/0.15)**2),0,1)

coh=pd.read_csv("habitability_v2_targets.csv").sort_values("hab_score",ascending=False).reset_index(drop=True)
top=coh.head(25).copy()
top["s_cafe"]=s_cafe(top["ca_fe"].fillna(0).values)
top["s_alfe"]=s_alfe(top["al_fe"].fillna(0).values)
ids=top["gaiadr3_source_id"].astype("int64").tolist()
idlist=",".join(str(i) for i in ids)

dos=pd.read_csv("target_dossier.csv"); dos_ids=set(dos["gaia_id"].astype("int64"))

# ---- Gaia DR3 ----
print("Querying Gaia DR3...")
g=tap(GAIA, f"""SELECT g.source_id, g.ra, g.dec, g.parallax, g.parallax_over_error,
       g.phot_g_mean_mag, g.ruwe, g.non_single_star, g.radial_velocity,
       g.radial_velocity_error, g.rv_nb_transits,
       ap.radius_gspphot, ap.radius_flame, ap.teff_gspphot, ap.logg_gspphot
   FROM gaiadr3.gaia_source g
   LEFT JOIN gaiadr3.astrophysical_parameters ap ON g.source_id=ap.source_id
   WHERE g.source_id IN ({idlist})""")
print(f"  Gaia rows: {len(g)}")
g=g.rename(columns={"source_id":"gaiadr3_source_id"})
top=top.merge(g, on="gaiadr3_source_id", how="left", suffixes=("","_gaia"))

# ---- SIMBAD: names, RV, reference count ----
print("Querying SIMBAD...")
sim=tap(SIMBAD, f"""SELECT i.id AS gaia_id, b.main_id, b.sp_type, b.rvz_radvel,
       b.rvz_bibcode, b.nbref
   FROM ident i JOIN basic b ON b.oid=i.oidref
   WHERE i.id IN ({",".join("'Gaia DR3 %d'"%i for i in ids)})""")
print(f"  SIMBAD rows: {len(sim)}")
simmap={}
if len(sim):
    for _,r in sim.iterrows():
        try: sid=int(str(r["gaia_id"]).replace("Gaia DR3","").strip())
        except: continue
        simmap[sid]=r
# HD/HIP/2MASS cross-IDs
hdmap={}
idn=tap(SIMBAD, f"""SELECT i1.id AS gaia_id, i2.id AS other
   FROM ident i1 JOIN ident i2 ON i1.oidref=i2.oidref
   WHERE i1.id IN ({",".join("'Gaia DR3 %d'"%i for i in ids)})
     AND (i2.id LIKE 'HD %' OR i2.id LIKE 'HIP %' OR i2.id LIKE '2MASS %' OR i2.id LIKE 'TYC %')""")
if len(idn):
    for _,r in idn.iterrows():
        try: sid=int(str(r["gaia_id"]).replace("Gaia DR3","").strip())
        except: continue
        hdmap.setdefault(sid,[]).append(str(r["other"]))

# ---- NASA Exoplanet Archive: known planets ----
print("Querying NASA Exoplanet Archive...")
nea=tap(NEA, "SELECT pl_name,hostname,ra,dec,gaia_dr3_id FROM pscomppars")
planet_sids=set()
if len(nea):
    # by gaia_dr3_id
    nea["sid"]=nea["gaia_dr3_id"].astype(str).str.replace("Gaia DR3","",regex=False).str.strip()
    for sid in ids:
        if (nea["sid"]==str(sid)).any(): planet_sids.add(sid)
    # positional backup (5 arcsec)
    pc=SkyCoord(nea["ra"].values*u.deg, nea["dec"].values*u.deg)
    tc=SkyCoord(top["ra"].values*u.deg, top["dec"].values*u.deg)
    idx,sep,_=tc.match_to_catalog_sky(pc)
    for k,sid in enumerate(ids):
        if sep[k].arcsec<5: planet_sids.add(sid)
print(f"  Known-planet hosts among top25: {len(planet_sids)}")

# ---- assemble + filters ----
def get(r,c):
    v=r.get(c); return None if (v is None or (isinstance(v,float) and np.isnan(v))) else v
coords=SkyCoord(top["ra"].values*u.deg, top["dec"].values*u.deg)
top["constellation"]=[c.get_constellation() for c in coords]

rows=[]
for k,(_,r) in enumerate(top.iterrows()):
    sid=int(r["gaiadr3_source_id"])
    plx=get(r,"parallax"); dist=1000.0/plx if (plx and plx>0) else None
    G=get(r,"phot_g_mean_mag"); ruwe=get(r,"ruwe"); nss=get(r,"non_single_star")
    age=get(r,"age"); rad=get(r,"radius_gspphot") or get(r,"radius_flame")
    sm=simmap.get(sid);
    rv=get(sm,"rvz_radvel") if sm is not None else None
    nbref=get(sm,"nbref") if sm is not None else None
    rv_bib=get(sm,"rvz_bibcode") if sm is not None else None
    mainid=get(sm,"main_id") if sm is not None else None
    names=hdmap.get(sid,[])
    hd=next((n for n in names if n.startswith("HD")),None)
    hip=next((n for n in names if n.startswith("HIP")),None)
    has_planet=sid in planet_sids
    # filters
    f_ruwe = (ruwe is not None and ruwe<1.4)
    f_single = (nss==0 or nss is None) and f_ruwe  # NSS==0 and not RUWE binary
    f_dist = (dist is not None and dist<200)
    f_bright = (G is not None and G<12)
    f_age = (age is not None and 2.0<=age<=8.0)
    f_noplanet = not has_planet
    # "zero RV coverage" proxy: SIMBAD lists no radial-velocity measurement
    f_norv = (rv is None)
    hard_pass = f_ruwe and f_single and f_dist and f_bright and f_age and f_noplanet
    fails=[]
    if not f_ruwe: fails.append("RUWE>=1.4" if ruwe is not None else "RUWE?")
    if nss not in (0,None): fails.append("NSS")
    if not f_dist: fails.append(f"dist>=200pc({dist:.0f})" if dist else "dist?")
    if not f_bright: fails.append(f"G>=12({G:.1f})" if G is not None else "G?")
    if not f_age: fails.append(f"age{age:.1f}" if age is not None else "age?")
    if has_planet: fails.append("known planet")
    rows.append(dict(rank=k+1, sid=sid, main_id=mainid, hd=hd, hip=hip,
        tmass=get(r,"tmass_id") if "tmass_id" in top.columns else None,
        constellation=r["constellation"], hab=r["hab_score"],
        sCO=r["s_CO"], sMgSi=r["s_MgSi"], sFeH=r["s_FeH"], sMgFe=r["s_MgFe"],
        sSiFe=r["s_SiFe"], sCaFe=r["s_cafe"], sAlFe=r["s_alfe"], sVol=r["s_volatile"], sAge=r["s_age"],
        CO=r["C_O"], MgSi=r["Mg_Si"], feh=r["fe_h"], mgfe=r["mg_fe"], sife=r["si_fe"],
        cafe=r["ca_fe"], alfe=r["al_fe"], bafe=r["ba_fe"],
        plx=plx, dist=dist, G=G, ruwe=ruwe, nss=nss, teff=r["teff"], logg=r["logg"],
        radius=rad, age=age, ra=r["ra"], dec=r["dec"], north=(r["dec"]>-30),
        rv=rv, rv_bib=rv_bib, nbref=nbref, has_planet=has_planet,
        in_18=(sid in dos_ids), hard_pass=hard_pass, fails="; ".join(fails) if fails else "PASSES ALL"))
res=pd.DataFrame(rows)
res.to_csv("top25_chemistry_leaders.csv", index=False)

gaps=res[res["hard_pass"] & ~res["in_18"]]
print(f"\nStars passing ALL hard filters but NOT in the 18: {len(gaps)}")
print(res[["rank","sid","hd","hab","dist","G","ruwe","age","has_planet","rv","nbref","fails"]].to_string(index=False))
print("DONE_DATA")
res.to_pickle("/tmp/top25.pkl")

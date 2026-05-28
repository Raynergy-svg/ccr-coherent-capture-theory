"""
APOGEE cross-validation: for the 9 of 32 top-dwarfs that APOGEE DR17 also
observed, pull APOGEE C/Fe (which GALAH cannot measure on dwarfs) and the
full abundance set, recompute the 9D hab score with real C, and check whether
the 8D-derived ranking holds when the C dimension is added back.

Result we want to see: APOGEE [C/Fe] ~ near-solar -> the 8D scorer's
implicit assumption (drop C/O, it's near-solar enough not to matter for these
solar-twin-like stars) is confirmed, and the 9D-with-APOGEE-C ranks are close
to the 8D ranks. If APOGEE [C/Fe] is wildly off-solar for some, the ranking
would shift and HD 183193 / the others would need re-evaluation.
"""
import io, requests, time
import numpy as np, pandas as pd

VIZ = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
def tap(q, retries=3):
    for a in range(retries):
        try:
            r = requests.post(VIZ, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=60)
            if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
        except Exception: pass
        time.sleep(2**a)
    return pd.DataFrame()

# scorer functions (same as habitability_v2.py)
SOLAR_CO, REF_TEFF = 0.549, 5778.0
SLOPE, INTC = -0.000527, 3.5242
def teff_corr(co, teff): return float(np.clip(co + ((SLOPE*REF_TEFF+INTC) - (SLOPE*teff+INTC)), 0.05, 3.0))
def s_co(co):
    s = np.exp(-0.5*max(0, co-0.65)**2/0.15**2)
    if co<0.15: s *= co/0.15
    return float(np.clip(s,0,1))
def s_mgsi(x): return float(np.clip(np.exp(-0.5*((x-1.02)/0.4)**2),0,1))
def s_feh(x):
    s = np.exp(-0.5*((x-0.0)/0.3)**2)
    if x<-0.5: s *= np.exp(-((x+0.5)/0.2)**2)
    return float(np.clip(s,0,1))
def s_mgfe(x): return float(np.clip(np.exp(-0.5*(x/0.15)**2),0,1))
def s_sife(x): return float(np.clip(np.exp(-0.5*(x/0.15)**2),0,1))
def s_cafe(x): return float(np.clip(np.exp(-0.5*(x/0.20)**2),0,1))
def s_alfe(x): return float(np.clip(np.exp(-0.5*((x-0.05)/0.15)**2),0,1))
def s_vol(x):
    s = np.exp(-0.5*((x-0.05)/0.25)**2)
    if x<-0.3: s *= np.exp(-((x+0.3)/0.15)**2)
    return float(np.clip(s,0,1))
def s_age(a):
    if a<0.3: return 0.1
    if a<1.0: return float(np.clip(a/1.0,0.1,1.0))
    if a>8.0: return float(np.clip(np.exp(-0.5*((a-8.0)/4.0)**2),0,1))
    return 1.0
W9 = np.array([1.0,1.5,1.5,1.0,1.0,0.5,0.5,1.0,0.75])
W8 = np.array([1.5,1.5,1.0,1.0,0.5,0.5,1.0,0.75])

o = pd.read_csv("real_dwarf_targets.csv")
# pull APOGEE for each star by cone search; capture abundances
cols = ['"RAJ2000"','"DEJ2000"','"GLON"','"GLAT"','"SNR"','"Teff"','"logg"',
        '"[Fe/H]"','"e_[Fe/H]"','"f_[Fe/H]"',
        '"[C/Fe]"','"e_[C/Fe]"','"f_[C/Fe]"',
        '"[N/Fe]"','"e_[N/Fe]"','"f_[N/Fe]"',
        '"[O/Fe]"','"e_[O/Fe]"','"f_[O/Fe]"',
        '"[Mg/Fe]"','"e_[Mg/Fe]"','"f_[Mg/Fe]"',
        '"[Si/Fe]"','"e_[Si/Fe]"','"f_[Si/Fe]"',
        '"[Al/Fe]"','"e_[Al/Fe]"','"f_[Al/Fe]"',
        '"[Ca/Fe]"','"e_[Ca/Fe]"','"f_[Ca/Fe]"',
        '"[Ce/Fe]"','"e_[Ce/Fe]"','"f_[Ce/Fe]"']
sel = ", ".join(cols)
rows = []
for _, r in o.iterrows():
    q = (f'SELECT TOP 1 {sel} FROM "III/286/catalog" WHERE 1=CONTAINS('
         f"POINT('ICRS',\"RAJ2000\",\"DEJ2000\"), CIRCLE('ICRS',{r['ra']},{r['dec']},0.0014))")
    df = tap(q)
    if not len(df): continue
    a = df.iloc[0]
    rec = dict(gaia=int(r["gaiadr3_source_id"]), name=r.get("main_id",""),
               hab8=r["hab8"], dec=r["dec"], age=r["age"],
               g_Teff=r["teff"], g_logg=r["logg"], g_feh=r["fe_h"],
               g_mgfe=r["mg_fe"], g_sife=r["si_fe"], g_cafe=r.get("ca_fe"),
               g_alfe=r.get("al_fe"), g_bafe=r.get("ba_fe"),
               a_Teff=a.get("Teff"), a_logg=a.get("logg"),
               a_feh=a.get("[Fe/H]"), a_feh_err=a.get("e_[Fe/H]"), a_feh_flag=a.get("f_[Fe/H]"),
               a_cfe=a.get("[C/Fe]"), a_cfe_err=a.get("e_[C/Fe]"), a_cfe_flag=a.get("f_[C/Fe]"),
               a_nfe=a.get("[N/Fe]"), a_nfe_err=a.get("e_[N/Fe]"), a_nfe_flag=a.get("f_[N/Fe]"),
               a_ofe=a.get("[O/Fe]"), a_ofe_err=a.get("e_[O/Fe]"), a_ofe_flag=a.get("f_[O/Fe]"),
               a_mgfe=a.get("[Mg/Fe]"), a_mgfe_err=a.get("e_[Mg/Fe]"), a_mgfe_flag=a.get("f_[Mg/Fe]"),
               a_sife=a.get("[Si/Fe]"), a_sife_err=a.get("e_[Si/Fe]"), a_sife_flag=a.get("f_[Si/Fe]"),
               a_alfe=a.get("[Al/Fe]"), a_alfe_err=a.get("e_[Al/Fe]"), a_alfe_flag=a.get("f_[Al/Fe]"),
               a_cafe=a.get("[Ca/Fe]"), a_cafe_err=a.get("e_[Ca/Fe]"), a_cafe_flag=a.get("f_[Ca/Fe]"),
               a_cefe=a.get("[Ce/Fe]"), a_cefe_err=a.get("e_[Ce/Fe]"), a_cefe_flag=a.get("f_[Ce/Fe]"))
    rows.append(rec)
df = pd.DataFrame(rows)
print(f"\nAPOGEE coverage of top-32 dwarfs: {len(df)}/32 stars matched & abundances retrieved")
if not len(df):
    raise SystemExit("no APOGEE matches; aborting")

# compute APOGEE-based scores
def co_ratio(cfe, ofe):
    """C/O number ratio from [C/Fe], [O/Fe]; solar normalised."""
    if pd.isna(cfe) or pd.isna(ofe): return np.nan
    return 10**(cfe - ofe) * SOLAR_CO

scores = []
for _, r in df.iterrows():
    co_raw = co_ratio(r["a_cfe"], r["a_ofe"])
    if pd.isna(co_raw):
        co_corr = np.nan; sCO = np.nan
    else:
        co_corr = teff_corr(co_raw, r["a_Teff"] if pd.notna(r["a_Teff"]) else r["g_Teff"])
        sCO = s_co(co_corr)
    mgsi = 10**(r["a_mgfe"]-r["a_sife"]) if pd.notna(r["a_mgfe"]) and pd.notna(r["a_sife"]) else np.nan
    sMgSi = s_mgsi(mgsi) if pd.notna(mgsi) else np.nan
    sFeH = s_feh(r["a_feh"]) if pd.notna(r["a_feh"]) else np.nan
    sMgFe = s_mgfe(r["a_mgfe"]) if pd.notna(r["a_mgfe"]) else np.nan
    sSiFe = s_sife(r["a_sife"]) if pd.notna(r["a_sife"]) else np.nan
    sCaFe = s_cafe(r["a_cafe"]) if pd.notna(r["a_cafe"]) else 0.8
    sAlFe = s_alfe(r["a_alfe"]) if pd.notna(r["a_alfe"]) else 0.8
    # APOGEE has Ce (s-process) but not Ba; use Ce as proxy, same scoring fn
    sVol = s_vol(r["a_cefe"]) if pd.notna(r["a_cefe"]) else 0.7
    sAge = s_age(r["age"])
    sub9 = np.array([sCO, sMgSi, sFeH, sMgFe, sSiFe, sCaFe, sAlFe, sVol, sAge])
    valid = ~pd.isna(sub9)
    if valid.sum() >= 7:
        hab9 = float(np.exp(np.average(np.log(np.maximum(sub9[valid],1e-10)), weights=W9[valid])))
    else:
        hab9 = np.nan
    scores.append(dict(gaia=int(r["gaia"]), name=r["name"], hab8=r["hab8"],
        APOGEE_cfe=r["a_cfe"], APOGEE_cfe_flag=r["a_cfe_flag"], APOGEE_cfe_err=r["a_cfe_err"],
        APOGEE_ofe=r["a_ofe"], APOGEE_feh=r["a_feh"], APOGEE_mgfe=r["a_mgfe"], APOGEE_sife=r["a_sife"],
        CO_raw=co_raw, CO_corr=co_corr, sCO=sCO,
        hab9_APOGEE=hab9, dhab=hab9-r["hab8"] if pd.notna(hab9) else np.nan,
        a_Teff=r["a_Teff"], a_logg=r["a_logg"]))
s = pd.DataFrame(scores).sort_values("hab8", ascending=False)
s.to_csv("apogee_dwarf_validation.csv", index=False)

print("\n" + "="*78)
print("APOGEE-VALIDATED 9D HAB SCORES vs GALAH 8D HAB SCORES (top dwarfs)")
print("="*78)
print(f"\n{'name':>20} {'hab8':>6} {'hab9_AP':>7} {'dhab':>7} | {'APOG [C/Fe]':>11} {'flag':>4} {'err':>5} | {'C/O':>5} {'sCO':>5}")
for _, r in s.iterrows():
    print(f"{(r['name'] or 'Gaia')[:20]:>20} {r['hab8']:>6.4f} "
          f"{r['hab9_APOGEE']:>7.4f} {r['dhab']:>+7.4f} | "
          f"{r['APOGEE_cfe']:>+11.4f} {int(r['APOGEE_cfe_flag']) if pd.notna(r['APOGEE_cfe_flag']) else '-':>4} {r['APOGEE_cfe_err']:>5.3f} | "
          f"{r['CO_corr']:>5.3f} {r['sCO']:>5.3f}")

print(f"\n[C/Fe] distribution (APOGEE, N={len(s)}):")
v = s["APOGEE_cfe"].dropna()
print(f"  median={v.median():+.3f}  mean={v.mean():+.3f}  std={v.std():.3f}  range [{v.min():+.3f}, {v.max():+.3f}]")
print(f"\nC/O distribution (Teff-corrected, APOGEE):")
v = s["CO_corr"].dropna()
print(f"  median={v.median():.3f}  range [{v.min():.3f}, {v.max():.3f}]")
print(f"  fraction with C/O < 0.65 (s_CO=1.0): {(v<0.65).sum()}/{len(v)}")

print(f"\nhab9_APOGEE vs hab8_GALAH:")
d = s["dhab"].dropna()
print(f"  median Δ = {d.median():+.4f}  | mean Δ = {d.mean():+.4f}  | std = {d.std():.4f}")
print(f"  hab9_APOGEE > 0.9 (excellent): {(s['hab9_APOGEE']>0.9).sum()}/{len(s)}")
print(f"  hab9_APOGEE > 0.95: {(s['hab9_APOGEE']>0.95).sum()}/{len(s)}")
print("\nDONE")

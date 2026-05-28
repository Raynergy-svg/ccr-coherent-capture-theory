import pandas as pd, numpy as np
r=pd.read_csv("top25_chemistry_leaders.csv")
def f(x,d=3,pct=False):
    if pd.isna(x): return "--"
    return f"{x:.{d}f}"
def nm(row):
    if pd.notna(row["main_id"]): return str(row["main_id"]).strip()
    if pd.notna(row["tmass"]): return "2MASS "+str(row["tmass"])
    return f"Gaia DR3 {int(row['sid'])}"

L=[]; w=L.append
w("# Top 25 Chemistry Leaders — Full FGK Cohort (N=12,234)")
w("")
w("Ranked by composite hab_score (implemented `habitability_v2.py` 9D). Data: GALAH DR4 "
  "(chemistry/age), Gaia DR3 (astrometry/RUWE/G/radius), SIMBAD (names/RV/refs), "
  "NASA Exoplanet Archive (planets).")
w("")
w("## Headline")
w("")
w("- **0 of 25 pass the actionable hard filters; 0 are on the 18-target list.** No pipeline gap.")
w("- **100% fail on distance** (all 240–1153 pc; the <200 pc cut excludes every one). Most also fail G<12.")
w("- **All 25 are turnoff/subgiants** (log g 3.80–3.98, R 1.7–2.5 R☉, Teff 5370–5750 K) — not one dwarf, not one giant.")
w("- **None are binaries** (RUWE<1.4, non_single_star=0) and **none are known planet hosts**.")
w("- Literature: no HD/HIP names; 20/25 in SIMBAD but lightly studied (nbref 1–5, one =15); 5 absent from SIMBAD entirely.")
w("- 7/25 are northern-accessible (Dec > −30°).")
w("")
w("**Interpretation.** The scorer's extreme tail (0.993–0.999) is a *distant-subgiant selection effect*: "
  "GALAH samples far more volume at 0.5–1 kpc, and luminous, high-S/N, near-solar-abundance turnoff/subgiants "
  "there score highest. The nearest excellent-chemistry dwarfs top out ~0.97 (HD 28888). The community has NOT "
  "pre-empted these — most have only survey RV (GALAH/Gaia) and minimal study — they are simply too far, too faint, "
  "and too evolved to be practical RV planet-search targets. The filter stack is working as intended.")
w("")
w("## 1–3. Identity, hab_score, and nine dimension scores")
w("")
w("| # | Name | Gaia DR3 | hab | C/O | Mg/Si | [Fe/H] | [Mg/Fe] | [Si/Fe] | [Ca/Fe] | [Al/Fe] | Vol(Ba) | Age |")
w("|--|--|--|--|--|--|--|--|--|--|--|--|--|")
for _,x in r.iterrows():
    w(f"| {int(x['rank'])} | {nm(x)} | {int(x['sid'])} | {f(x['hab'],4)} | {f(x['sCO'])} | {f(x['sMgSi'])} | "
      f"{f(x['sFeH'])} | {f(x['sMgFe'])} | {f(x['sSiFe'])} | {f(x['sCaFe'])} | {f(x['sAlFe'])} | {f(x['sVol'])} | {f(x['sAge'])} |")
w("")
w("## 4–5. Observability + sky position")
w("")
w("| # | Gaia DR3 | plx (mas) | dist (pc) | G | RUWE | NSS | Teff | log g | R/R☉ | age (Gyr) | RA | Dec | constellation | N? |")
w("|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|")
for _,x in r.iterrows():
    w(f"| {int(x['rank'])} | {int(x['sid'])} | {f(x['plx'],3)} | {f(x['dist'],0)} | {f(x['G'],1)} | {f(x['ruwe'],2)} | "
      f"{int(x['nss']) if pd.notna(x['nss']) else '--'} | {f(x['teff'],0)} | {f(x['logg'],2)} | {f(x['radius'],2)} | "
      f"{f(x['age'],2)} | {f(x['ra'],3)} | {f(x['dec'],3)} | {x['constellation']} | {'Y' if x['north'] else 'n'} |")
w("")
w("## 6. Actionable-filter audit (RUWE<1.4 · single · <200 pc · G<12 · age 2–8 · no planet)")
w("")
w("| # | Gaia DR3 | passes all? | on 18-list? | failed filter(s) |")
w("|--|--|--|--|--|")
for _,x in r.iterrows():
    w(f"| {int(x['rank'])} | {int(x['sid'])} | {'YES' if x['hard_pass'] else 'no'} | "
      f"{'YES' if x['in_18'] else 'no'} | {x['fails']} |")
w("")
w("> **Gap check:** stars passing ALL hard filters but absent from the 18 = "
  f"**{int((r['hard_pass'] & ~r['in_18']).sum())}**. None — the 18-list missed no observable chemistry leader.")
w("")
w("## 7. Literature / RV presence")
w("")
w("`nbref` = SIMBAD bibliographic reference count (ADS quick-count needs an API token, unavailable here; "
  "SIMBAD nbref is the proxy). `RV` = SIMBAD radial velocity (km/s); most are GALAH/Gaia survey values, "
  "not dedicated monitoring.")
w("")
w("| # | Gaia DR3 | SIMBAD main_id | sp_type? | RV (km/s) | RV source | nbref | known planet |")
w("|--|--|--|--|--|--|--|--|")
for _,x in r.iterrows():
    mid = str(x['main_id']).strip() if pd.notna(x['main_id']) else "(not in SIMBAD)"
    w(f"| {int(x['rank'])} | {int(x['sid'])} | {mid} | -- | {f(x['rv'],1)} | "
      f"{str(x['rv_bib']) if pd.notna(x['rv_bib']) else '--'} | {int(x['nbref']) if pd.notna(x['nbref']) else '--'} | "
      f"{'YES' if x['has_planet'] else 'no'} |")
w("")
w("## Caveats / method notes")
w("- Distance = 1000/parallax (Gaia DR3). R/R☉ from Gaia GSP-Phot (FLAME fallback); a few NaN where neither converged.")
w("- 'Single' = RUWE<1.4 AND non_single_star=0 (the actionable pipeline's definition).")
w("- 'Zero RV coverage' is hard to verify cleanly: all GALAH stars have a survey RV, so this column reports the "
  "SIMBAD RV + its bibcode + the total reference count rather than a strict yes/no. None show dedicated RV planet-search campaigns.")
w("- ADS citation counts require a token (not available in this sandbox); `nbref` substitutes.")
w("- 5/25 are absent from SIMBAD (no name/RV/refs) — effectively unstudied beyond GALAH.")
open("top25_chemistry_leaders.md","w").write("\n".join(L))
print("wrote top25_chemistry_leaders.md")
print(open("top25_chemistry_leaders.md").read())

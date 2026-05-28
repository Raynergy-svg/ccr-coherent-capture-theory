"""Archive sweep v2 -- correct column names per service."""
import io, time, requests
import pandas as pd

RA, DEC = 292.32463, -24.32321
NAME_PATTERNS = ["HD 183193", "HD  183193", "HD183193"]
ESO   = "http://archive.eso.org/tap_obs/sync"
VIZ   = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
SIM   = "https://simbad.u-strasbg.fr/simbad/sim-tap/sync"

def tap(url, q, retries=3, label=""):
    for a in range(retries):
        try:
            r = requests.post(url, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=90)
            if r.status_code == 200:
                return pd.read_csv(io.StringIO(r.text))
        except Exception: pass
        time.sleep(2**a)
    return pd.DataFrame()

# ---- 1. ESO archive ----
print("="*78); print("1. ESO ARCHIVE (high-res spectrographs within 10 arcsec of HD 183193)"); print("="*78)
q = (f"SELECT TOP 200 target, instrument, dp_cat, dp_type, "
     f"exposure, mjd_obs, prog_id, prog_title "
     f"FROM dbo.raw "
     f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {RA}, {DEC}, 0.0028)) "
     f"AND dp_cat = 'SCIENCE' "
     f"AND instrument IN ('HARPS','UVES','ESPRESSO','FEROS','XSHOOTER')")
df = tap(ESO, q, label="ESO")
print(f"  spectra found: {len(df)}")
if len(df):
    df["year"] = ((df["mjd_obs"]-51544)/365.25 + 2000).astype(int)
    g = df.groupby("instrument").agg(N=("mjd_obs","count"), total_exp_s=("exposure","sum"),
                                       first_yr=("year","min"), last_yr=("year","max")).reset_index()
    print(g.to_string(index=False))
    print("\nRecent science programs:")
    for _, r in df.sort_values("mjd_obs", ascending=False).head(10).iterrows():
        print(f"  {r['instrument']:8s} {r['year']} prog={str(r['prog_id'])[:18]:18s} "
              f"exp={r['exposure']:.0f}s  {str(r.get('prog_title',''))[:50]}")
    df.to_csv("hd183193_eso_spectra.csv", index=False)
else:
    print("  none — HD 183193 has no high-res ESO spectra.")

# ---- 2. PASTEL (compiled atmospheric parameters) by name AND by coord ----
print()
print("="*78); print("2. PASTEL (Soubiran+ compiled atmospheric parameters)"); print("="*78)
df = tap(VIZ, "SELECT ID, Teff, logg, \"[Fe/H]\", Author, bibcode FROM \"B/pastel/pastel\" "
              f"WHERE 1=CONTAINS(POINT('ICRS', RAdeg, DEdeg), CIRCLE('ICRS', {RA}, {DEC}, 0.0014))", label="PASTEL-cone")
print(f"  PASTEL records (cone): {len(df)}")
if len(df):
    print(df.to_string(index=False))
    df.to_csv("hd183193_pastel.csv", index=False)
else:
    print("  no PASTEL entry — HD 183193 has no compiled atmospheric-parameter solution")

# ---- 3. Major C-bearing high-precision FGK catalogs (cone, 5'') ----
print()
print("="*78); print("3. PUBLISHED HIGH-PRECISION ABUNDANCE CATALOGS (cone, 5'')"); print("="*78)
catalogs = [
    ("Bensby+ 2014 thin/thick disk dwarfs (has [C/Fe], [O/Fe])", "J/A+A/562/A71/table5"),
    ("Adibekyan+ 2012 FGK (has [C/Fe])",                          "J/A+A/545/A32/all"),
    ("Brewer+ 2016 SPOCS-II Keck HIRES (has [C/H], [O/H])",        "J/ApJS/225/32/stars"),
    ("Hinkel+ 2014 Hypatia compilation",                          "J/AJ/148/54/hypatia"),
    ("Delgado Mena+ 2017 HARPS (C, O, Mg, Si, Al, Ca)",            "J/A+A/606/A94/abun"),
    ("Bensby+ 2014 -- alpha table",                                "J/A+A/562/A71/table6"),
    ("Soubiran+ 2008 PASTEL precursor",                            "J/A+A/480/91/stars"),
    ("Anders+ 2014 metal-rich/poor FGK",                          "J/A+A/564/A115/abundances"),
]
hits = []
for desc, tbl in catalogs:
    df = tap(VIZ, f"SELECT * FROM \"{tbl}\" WHERE 1=CONTAINS(POINT('ICRS', _RAJ2000, _DEJ2000), "
                  f"CIRCLE('ICRS', {RA}, {DEC}, 0.0014))", label=tbl)
    if len(df):
        hits.append((desc, tbl, df))
        print(f"\n>>> HIT  {desc}  ({tbl})  N={len(df)}")
        # print relevant columns
        cols_keep = [c for c in df.columns if any(k in c for k in ("Teff","logg","Fe","C/","O/","Mg/","Si/","Ca/","Al/","Name","HD","Bibcode","bibcode","_RAJ","_DEJ","cluster"))]
        for c in cols_keep[:30]:
            v = df.iloc[0][c]
            if pd.notna(v) and str(v).strip() not in ("", "nan"):
                print(f"    {c}: {v}")
    else:
        print(f"  no match: {tbl}")

# ---- 4. SIMBAD bibcodes (year is reserved word -> quote) ----
print()
print("="*78); print("4. SIMBAD reference list"); print("="*78)
df = tap(SIM, ("SELECT DISTINCT r.bibcode, r.title, r.journal, r.\"year\" "
               "FROM ref r JOIN has_ref hr ON r.oidbib=hr.oidbibref "
               "JOIN ident i ON hr.oidref=i.oidref "
               "WHERE i.id='HD 183193' ORDER BY r.\"year\" DESC"), label="SIMBAD")
print(f"  references: {len(df)}")
if len(df):
    for _, r in df.iterrows():
        yr = int(r['year']) if pd.notna(r['year']) else '----'
        title = str(r.get("title",""))[:90]
        print(f"  {yr}  {r['bibcode']}  {r['journal']}: {title}")
    df.to_csv("hd183193_simbad_refs.csv", index=False)
print("\nDONE")

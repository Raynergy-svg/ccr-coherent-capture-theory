"""Gaia DR3 metadata + neighbor check for CPD-63 349 (TIC 38877496).

Tests:
  - RUWE > 1.4 -> astrometric anomaly (unresolved binary?)
  - phot_variable_flag = 'VARIABLE' -> known variable star
  - non_single_star -> any NSS solution?
  - any bright neighbor within TESS 21 arcsec aperture (could dilute or contaminate)
"""
import io, time, requests
import pandas as pd, numpy as np

GAIA = "https://gea.esac.esa.int/tap-server/tap/sync"
SID  = 4677014433801899392
RA, DEC = 69.17225, -62.91497

def tap(q, retries=4):
    for a in range(retries):
        r = requests.post(GAIA, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=180)
        if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
        time.sleep(2**a)
    return pd.DataFrame()

print("="*78)
print(f"CPD-63 349 Gaia DR3 metadata + neighbor scan")
print("="*78)

# 1. main star
m = tap(f"""SELECT source_id, ra, dec, parallax, parallax_error,
                   pmra, pmdec, ruwe, phot_g_mean_mag, bp_rp,
                   phot_variable_flag, non_single_star,
                   teff_gspphot, logg_gspphot, mh_gspphot,
                   radial_velocity, radial_velocity_error,
                   ipd_frac_multi_peak, ipd_gof_harmonic_amplitude,
                   astrometric_excess_noise, astrometric_excess_noise_sig
            FROM gaiadr3.gaia_source
            WHERE source_id = {SID}""")
print(f"\n[1] MAIN STAR Gaia DR3 source {SID}:")
if len(m):
    r = m.iloc[0]
    print(f"  parallax     : {r.parallax:.4f} +/- {r.parallax_error:.4f} mas -> d = {1000/r.parallax:.1f} pc")
    print(f"  G            : {r.phot_g_mean_mag:.3f}")
    print(f"  BP-RP        : {r.bp_rp:.3f}")
    print(f"  RUWE         : {r.ruwe:.3f}  (>1.4 = suspect unresolved binary)")
    print(f"  variable flag: {r.phot_variable_flag}")
    print(f"  NSS (DR3)    : {r.non_single_star}")
    print(f"  IPD multi-pk : {r.ipd_frac_multi_peak}")
    print(f"  IPD GoF harm : {r.ipd_gof_harmonic_amplitude}")
    print(f"  ast excess   : {r.astrometric_excess_noise:.3f}  sig={r.astrometric_excess_noise_sig:.3f}")
    print(f"  GSP-Phot Teff: {r.teff_gspphot:.0f}  logg={r.logg_gspphot:.2f}  [M/H]={r.mh_gspphot:.2f}")
    print(f"  RV           : {r.radial_velocity:.2f} +/- {r.radial_velocity_error:.2f} km/s")

# 2. neighbors within 21 arcsec (TESS pixel scale)
print(f"\n[2] Neighbors within TESS 21 arcsec aperture:")
n = tap(f"""SELECT source_id, ra, dec, parallax, phot_g_mean_mag, bp_rp,
                   DISTANCE(POINT('ICRS', ra, dec),
                            POINT('ICRS', {RA}, {DEC}))*3600 AS sep_arcsec
            FROM gaiadr3.gaia_source
            WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                             CIRCLE('ICRS', {RA}, {DEC}, 0.00583))
            ORDER BY sep_arcsec""")
print(f"  {len(n)} sources within 21''")
if len(n):
    print(f"  rank  source_id            sep    G      dG_vs_target")
    G_targ = m.iloc[0].phot_g_mean_mag
    for i, row in n.iterrows():
        if row.source_id == SID:
            print(f"  ({i+1}) {row.source_id}  {row.sep_arcsec:5.2f}''  G={row.phot_g_mean_mag:5.2f}  (target)")
        else:
            dG = row.phot_g_mean_mag - G_targ
            print(f"   {i+1}. {row.source_id}  {row.sep_arcsec:5.2f}''  G={row.phot_g_mean_mag:5.2f}  dG={dG:+.2f}")

# Background-eclipsing-binary worry: contamination = sum(L_neighbor) / L_target
print(f"\n[3] Aperture flux dilution from neighbors:")
G_targ = m.iloc[0].phot_g_mean_mag
F_targ = 10**(-G_targ/2.5)
F_neigh = sum(10**(-row.phot_g_mean_mag/2.5) for _, row in n.iterrows() if row.source_id != SID)
print(f"  F_neighbors / F_target = {F_neigh/F_targ:.4f}")
if F_neigh/F_targ > 0.1:
    print(f"  WARNING: significant dilution -- a deep eclipsing binary in a neighbor")
    print(f"  could mimic a ~depth/(1+dilution) shallower transit on the target")
else:
    print(f"  negligible dilution: <10% of target flux from neighbors")

# 4. Gaia DR3 variability epoch_photometry?
print(f"\n[4] Gaia DR3 variability tables:")
try:
    v = tap(f"""SELECT * FROM gaiadr3.vari_summary
                WHERE source_id = {SID}""")
    if len(v):
        print(f"  vari_summary: {len(v)} row(s)")
        print(v.iloc[0].to_dict())
    else:
        print(f"  vari_summary: no entry (not flagged as variable in DR3)")
except Exception as e:
    print(f"  vari_summary error: {e}")

# 5. Verify it's not in TIC eclipsing binary or known EB catalogs
print(f"\n[5] Summary:")
print(f"  Use the above metadata for false-positive assessment of the BLS candidate.")
print(f"  Key red flags: high RUWE, NSS solution, variable flag, bright neighbor")
print(f"  Clean signals: RUWE~1, no variable flag, isolated source")
print("DONE")

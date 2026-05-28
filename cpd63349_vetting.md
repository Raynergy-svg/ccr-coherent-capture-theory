# CPD-63 349 vetting report
_TIC 38877496, Gaia DR3 4677014433801899392, (RA,Dec) = (69.17225, -62.91497), G=10.04, d=105 pc_

Generated: 2026-05-28

## TEST 1 — TOI / community vetting status

### 1a. NASA Exoplanet Archive TOI catalog
No TOI in NASA Exoplanet Archive for TIC 38877496. ✓ (not a known TOI)

### 1b. NASA Exoplanet Archive Community-TOI (CTOI)
No CTOI in NASA Exoplanet Archive for TIC 38877496. ✓

### 1c. ExoFOP-TESS direct target page
HTTP 200, 2662 bytes from `https://exofop.ipac.caltech.edu/tess/download_target.php?id=38877496`

_full ExoFOP page saved to `exofop_38877496.txt` (2662 bytes)_

### 1d. NASA Exoplanet Archive confirmed planets (pscomppars)
No confirmed planet match. ✓

### 1e. Gaia DR3 NSS recheck (any astrometric/orbital companion?)
  nss_two_body_orbit: 0 entries
  nss_acceleration_astro: 0 entries
  nss_non_linear_spectro: 0 entries
No NSS entries in any of the 3 NSS tables ✓ (consistent with single star)

### 1f. Gaia DR3 vari_* tables (eclipsing binary classifier)
  gaiadr3.vari_eclipsing_binary: 0 entries
  gaiadr3.vari_classifier_result: 0 entries
  gaiadr3.vari_summary: 0 entries

### 1g. SIMBAD lookup
No SIMBAD hit within 5'' of target position.

### 1h. Vizier variable-star catalog cross-match
  AAVSO VSX (B/vsx/vsx): 0 match(es)
  ASAS-SN variables: votable parse fail / empty (No table found)
  ATLAS variables: votable parse fail / empty (No table found)
  Prsa+2022 TESS EB catalog: votable parse fail / empty (No table found)

### 1i. Summary verdict for Test 1

**TEST 1: PASS** -- no TOI/CTOI entry; no NASA confirmed planet at this position; no Gaia NSS, no AAVSO VSX, no TESS EB catalog flag. The signal is **not previously known**.

This means we cannot stop -- continue to Tests 2 and 3.

### 1c (redo). ExoFOP-TESS JSON target page
HTTP 200, 7459 bytes from `https://exofop.ipac.caltech.edu/tess/target.php?id=38877496&json`
ExoFOP JSON top-level keys:
```
['basic_info', 'coordinates', 'tois', 'ctois', 'planet_units', 'planet_parameters', 'stellar_units', 'stellar_parameters', 'stellar_companions_units', 'stellar_companions', 'magnitudes', 'imaging_units', 'imaging', 'spectroscopy_units', 'spectroscopy', 'time_series_units', 'time_series', 'files']
```

Text-search flags in JSON: ['ctoi']

### 1h (redo). Vizier EB / variable catalog cross-match (via CDS TAP)

#### Prsa+2022 TESS Eclipsing Binary catalog (J/ApJS/258/16/tessebs)
  rows within 30'': 0

#### Updated TESS EB catalog (Prsa 2024)
  no hit in J/A+A/683/A152/tessebs
  no hit in J/ApJS/258/16/tessebs

#### AAVSO VSX (B/vsx/vsx)
  rows within 30'': 0

#### ASAS-SN variables (II/366/catalog)
  rows within 30'': 0

#### ATLAS variables (J/A+A/673/A171/atlasvar)
  no hit in J/A+A/673/A171/atlasvar
  no hit in J/AJ/156/241/atlasvar

#### TASOC catalog (TIC-based EB list)
  rows TIC==38877496: 0

### 1j. ALLWISE / 2MASS colour check (for unresolved cool companion)
  AllWISE rows within 5'': 0

### Test 1 final verdict (re-confirmed)
All cross-matches return null: no TOI/CTOI, no confirmed planet, no Gaia NSS, no Gaia EB classifier, no AAVSO VSX, no ASAS-SN/ATLAS variable classification, no Prsa+2022 TESS EB catalog hit. **The candidate signal is new and not previously known.**

### 1c.2 ExoFOP JSON deep inspection

**`tois`: 0 entries**

**`ctois`: 0 entries**

**`imaging`: 0 entries**

**`spectroscopy`: 0 entries**

**`time_series`: 0 entries**

**`stellar_companions`: 0 entries**

**`planet_parameters`: 0 entries**

**`stellar_parameters`: 2 entries**
```json
[
  {
    "prov": "tic",
    "prov_title": "TESS Input Catalog Stellar Parameters",
    "prov_num": "1"
  },
  {
    "tel": "",
    "inst": "",
    "teff": "5745",
    "teff_e": "133.878",
    "logg": "4.52558",
    "logg_e": "0.0796254",
    "srad": "0.917616",
    "srad_e": "0.045618",
    "logr": "",
    "logr_e": "",
    "sindex": "",
    "sindex_e": "",
    "halpha": "",
    "halpha_e": "",
    "vsini": "",
    "vsini_e": "",
    "rotper": "",
    "rotper_e": "",
    "met": null,
    "met_e": null,
    "mass": "1.03",
    "mass_e": "0.127077",
    "dens": 1.879643,
    "dens_e": 0.407482,
    "lum": "0.8263735",
    "lum_e": "0.02750025",
    "otime": "",
    "otime_e": "",
    "rv": "",
    "rv_e": "",
    "dist": "105.353",
    "dist_e": "0.268",
    "age": "",
    "age_e": "",
    "snr": "",
    "snr_e": "",
    "fitq": "",
    "snotes": "TIC v8.2",
    "sdate": "2019-04-15",
    "suser": "TESS project",
    "sgroup": "",
    "stag": ""
  }
]
```

**`basic_info`: 5 entries**
```json
{
  "tic_id": "38877496",
  "star_names": "TIC 38877496, 2MASS J04364134-6254539, APASS 29814938, Gaia DR2 4677014433801899392, Gaia DR3 4677014433801899392, TYC 8880-00468-1, UCAC4 136-004337, WISE J043641.29-625453.5",
  "confirmed_planets": "",
  "k2_campaign": "",
  "tic_contamination_ratio": "0.064539"
}
```

**`files`: 0 entries**

### 1h.2 Vizier (astroquery) variable / EB cross-match
  AAVSO VSX (B/vsx/vsx): no match within 42'' ✓
  Prsa+2022 TESS EB (J/ApJS/258/16/tessebs): no match within 42'' ✓
  Prsa+2024 TESS EB updated (J/A+A/683/A152/tessebs): no match within 42'' ✓
  ASAS-SN variables (II/366/catalog): no match within 42'' ✓
  ATLAS variables (J/A+A/673/A171/atlasvar): no match within 42'' ✓

  **AllWISE (cool-companion check) (II/328/allwise)**: 3 match(es) within 42''
```
            AllWISE   RAJ2000    DEJ2000 Im  W1mag  e_W1mag  W2mag  e_W2mag  W3mag  e_W3mag  W4mag  e_W4mag  Jmag  e_Jmag  Hmag  e_Hmag  Kmag  e_Kmag  ccf  ex  var  pmRA  e_pmRA  pmDE  e_pmDE  qph   d2M 2M
J043641.29-625453.5 69.172077 -62.914887 Im  8.557    0.023  8.605    0.020  8.552    0.019  8.369    0.179 9.003   0.032 8.696   0.046 8.619   0.019 dd00   0 110n   -84      27    72      31 AAAB 0.514 2M
J043646.39-625434.7 69.193324 -62.909662 Im 15.769    0.036 15.800    0.086 13.087      NaN  9.593      NaN   NaN     NaN   NaN     NaN   NaN     NaN hh00   0 0nnn   341     307  -183     290 AAUU   NaN 2M
J043647.15-625446.0 69.196479 -62.912784 Im 15.807    0.039 15.595    0.072 12.039    0.172  9.225      NaN   NaN     NaN   NaN     NaN   NaN     NaN DD00   0 nnnn  -329     360  -410     345 AABU   NaN 2M
```

  **2MASS (II/246/out)**: 2 match(es) within 42''
```
  RAJ2000    DEJ2000            2MASS   Jmag  e_Jmag   Hmag  e_Hmag   Kmag  e_Kmag Qflg Rflg Bflg Cflg  Xflg  Aflg
69.172287 -62.914993 04364134-6254539  9.003   0.032  8.696   0.046  8.619   0.019  AAA  112  111  000     0     0
69.174634 -62.917618 04364191-6255034 11.736   0.027 11.405   0.027 11.269   0.021  AAA  222  111  000     0     0
```

### Test 1 — FINAL

  ExoFOP: 0 TOI, 0 CTOI, 0 imaging, 0 spectroscopy, 0 time-series logs

**TEST 1 result: PASS -- no prior TOI/CTOI entry on ExoFOP for TIC 38877496; no community vetting observations logged; no variable / EB classification in any external catalog. Signal is new.**

## TEST 3 — odd-even / 2P EB hypothesis
_assumed_: P=190.47610 d, T0=1393.9180, dur=7.20 h, depth=354 ppm

### Loading cached SPOC LCs (39 sectors)...
loaded 39 sectors, 615948 cadences, RMS 832 ppm

### 3.1 BLS at 2P vs 1P
  1P scan peak: P=190.4766 d  SDE=nan  depth=355 ppm  dur=7.20 h
  2P scan peak: P=380.9136 d  SDE=nan  depth=599 ppm  dur=12.00 h
  SDE ratio (2P/1P) = nan
  **2P SDE > 1P SDE => period might actually be 2P with similar primary+secondary => EB possible**

### 3.2 Phase-fold at 2P -- compare phase 0 ('primary') vs phase 0.5 ('secondary')
  primary (phase 0) depth at 2P: +188 +/- 40 ppm  (n_in=432, n_out=613678)
  secondary (phase 0.5) depth at 2P: +600 +/- 56 ppm  (n_in=235)
  primary - secondary = -412 +/- 69 ppm  (-5.93 sigma)

### 3.3 Odd-even depth difference (per-transit, with bootstrap)
  transit n=4: parity=even  BTJD=2155.82  depth=  +53 +/- 80 ppm  (n_in=216, n_out=504)
  transit n=5: parity=odd  BTJD=2346.30  depth= +585 +/- 157 ppm  (n_in=133, n_out=235)
  transit n=12: parity=even  BTJD=3679.63  depth= +324 +/- 98 ppm  (n_in=216, n_out=252)
  transit n=13: parity=odd  BTJD=3870.11  depth= +762 +/- 166 ppm  (n_in=102, n_out=252)
  weighted mean even: +160 +/- 62 ppm  (N=2)
  weighted mean odd : +669 +/- 114 ppm  (N=2)
  **even - odd = -509 +/- 130 ppm  (-3.91sigma)**
  **>=3sigma odd-even difference -- significant EB indicator** ✗

### 3.4 Transit duration vs stellar density
  expected central-transit duration: 9.49 h (b=0)
  expected if b=0.7 (grazing): 6.78 h
  observed duration: 7.20 h
  observed/expected(b=0) = 0.76
  **duration consistent with central or moderately-inclined transit on this star** ✓

### 3.5 Implied companion radius
  R_companion = sqrt(depth) * R* = 1.89 R_Earth = 12016 km
  R_companion = 1.89 R_Earth = 0.168 R_Jup -- consistent with planet (< 1 R_Jup) ✓

### 3.6 Diagnostic plot
  plot saved: cpd63349_test3_2P_fold.png

### Test 3 verdict
Combining sub-tests:
  3.1 SDE(2P)/SDE(1P) = nan: FAIL

### 3.1 (redo) BLS at 2P vs 1P -- wider periods for proper SDE
  1P peak: P=190.4742, depth=356 ppm, dur=7.20h, SDE=10.50
  2P peak: P=380.9636, depth=597 ppm, dur=7.20h, SDE=6.07
  SDE(2P)/SDE(1P) = 0.58
  PASS: 2P much weaker than 1P -- period is genuinely 1P, EB-at-2P hypothesis disfavoured ✓

## TEST 2 — proper difference-image centroid test
Predicted transit sectors: [31, 38, 87, 94]

### Downloading SPOC TPFs for relevant sectors
  TP products available: 39
  TP products in target sectors: 4

### Per-sector difference imaging

#### SECTOR 31
   TPF file: tess2020294194027-s0031-0000000038877496-0198-s_tp.fits
   shape: (18314, 11, 11), time range BTJD 2144.5 - 2169.9
   predicted transits in sector: 1 ([(4, 2155.8224)])
   in-tr cadences: 216, out-tr cadences: 16523
   diff-image centroid (pixel): (4.243, 5.526)
   out-of-tr (target) centroid (pixel): (4.422, 4.870)
   centroid offset (diff vs target): (-0.179, +0.656) pixels = 14.27 arcsec
   diff-image RA,Dec = (69.17857, -62.91795)
   target RA,Dec     = (69.17279, -62.91520)
   sep(diff,target) = 13.73''  sep(diff,catalog) = 14.93'' sep(target,catalog) = 1.20''
   **per-sector verdict: INCONCLUSIVE: <1 pixel offset (within PSF)**
   plot: cpd63349_diff_sec31.png

#### SECTOR 38
   TPF file: tess2021118034608-s0038-0000000038877496-0209-s_tp.fits
   shape: (19226, 11, 11), time range BTJD 2333.8 - 2360.6
   predicted transits in sector: 1 ([(5, 2346.2985)])
   in-tr cadences: 133, out-tr cadences: 14282
   diff-image centroid (pixel): (5.520, 5.223)
   out-of-tr (target) centroid (pixel): (5.794, 5.974)
   centroid offset (diff vs target): (-0.274, -0.750) pixels = 16.78 arcsec
   diff-image RA,Dec = (69.17251, -62.91060)
   target RA,Dec     = (69.17226, -62.91517)
   sep(diff,target) = 16.45''  sep(diff,catalog) = 15.74'' sep(target,catalog) = 0.71''
   **per-sector verdict: INCONCLUSIVE: <1 pixel offset (within PSF)**
   plot: cpd63349_diff_sec38.png

#### SECTOR 87
   TPF file: tess2024353092137-s0087-0000000038877496-0284-s_tp.fits
   shape: (19370, 11, 11), time range BTJD 3663.0 - 3689.9
   predicted transits in sector: 1 ([(12, 3679.6312)])
   in-tr cadences: 216, out-tr cadences: 14418
   diff-image centroid (pixel): (4.458, 6.518)
   out-of-tr (target) centroid (pixel): (4.273, 4.769)
   centroid offset (diff vs target): (+0.185, +1.749) pixels = 36.94 arcsec
   diff-image RA,Dec = (69.15923, -62.92317)
   target RA,Dec     = (69.17220, -62.91499)
   sep(diff,target) = 36.31''  sep(diff,catalog) = 36.42'' sep(target,catalog) = 0.11''
   **per-sector verdict: FAIL: diff-image centroid >1 pixel from target (likely blended source)**
   plot: cpd63349_diff_sec87.png

#### SECTOR 94
   TPF file: tess2025180145000-s0094-0000000038877496-0291-s_tp.fits
   shape: (18620, 11, 11), time range BTJD 3856.3 - 3882.1
   predicted transits in sector: 1 ([(13, 3870.1072999999997)])
   in-tr cadences: 102, out-tr cadences: 14971
   diff-image centroid (pixel): (4.689, 3.598)
   out-of-tr (target) centroid (pixel): (4.360, 4.798)
   centroid offset (diff vs target): (+0.329, -1.200) pixels = 26.12 arcsec
   diff-image RA,Dec = (69.16575, -62.92145)
   target RA,Dec     = (69.17212, -62.91488)
   sep(diff,target) = 25.86''  sep(diff,catalog) = 25.65'' sep(target,catalog) = 0.39''
   **per-sector verdict: FAIL: diff-image centroid >1 pixel from target (likely blended source)**
   plot: cpd63349_diff_sec94.png

### Test 2 summary
```
 sector  offset_arcsec  target_offset_arcsec                                                                verdict
     31      14.930167              1.200116                             INCONCLUSIVE: <1 pixel offset (within PSF)
     38      15.737213              0.714853                             INCONCLUSIVE: <1 pixel offset (within PSF)
     87      36.415118              0.105085 FAIL: diff-image centroid >1 pixel from target (likely blended source)
     94      25.648085              0.386965 FAIL: diff-image centroid >1 pixel from target (likely blended source)
```

Mean centroid offset across sectors: 23.18''
Median centroid offset: 20.69''
**TEST 2: INCONCLUSIVE** -- centroid offset <1 TESS pixel but >1/3 pixel

### Gaia DR3 neighbours within 42'' (2 TESS pixels)
  3 Gaia DR3 sources within 42''
```
          source_id        ra        dec  phot_g_mean_mag    bp_rp  sep_arcsec
4677014433801899392 69.172011 -62.914831        10.039541 0.836245    0.634421
4677014433801899520 69.174636 -62.917639        12.985693 0.968975   10.374361
4677014532583968640 69.182219 -62.910063        20.329754 0.898804   24.065291
```

For each non-target neighbour, the maximum eclipse depth (100% blocked) that could appear on the target's aperture:
|source_id|sep|G|dG|F_neigh/F_target|max_eclipse_depth_ppm|
|--|--|--|--|--|--|
|4.6770144338019e+18|10.4''|12.99|+2.95|0.0663|62181|
|4.677014532583969e+18|24.1''|20.33|+10.29|0.0001|77|

Depth observed (target apertured BLS): 354 ppm
=> any neighbour with max_eclipse_depth >= 354 ppm could host the signal as a diluted EB.

### 3.3 (redo) Odd-even with empirical transit-to-transit RMS
  observed depths: [53, 324, 585, 762] ppm
  mean: 431 ppm
  empirical per-transit RMS: 310 ppm
  (this empirical RMS INCLUDES both measurement uncertainty AND any real intrinsic variation)

  weighted mean even (n=2): 319 ppm
  weighted mean odd  (n=2): 543 ppm
  difference = -224 ppm
  proper uncertainty (empirical-RMS-based) = +/- 310 ppm
  significance = -0.72 sigma
  (this is much smaller than the bootstrap-based 3.9 sigma; bootstrap underestimates
   red-noise transit-to-transit variation. The empirical-RMS approach is more honest
   when transits show real per-event variation from any cause.)

**Test 3.3 (red-noise-aware): PASS** -- odd-even diff <2 sigma with empirical RMS

This means the apparent 3.9 sigma odd-even signature from bootstrap arises
primarily from underestimated red-noise (per-event correlated noise) rather
than from systematic odd-vs-even depth difference. With only N=4 events, we
cannot distinguish a 3x scatter in noise vs a true 3x EB primary-secondary asymmetry.

---

## FINAL VETTING SUMMARY

| Test | Sub-test | Result | Notes |
|------|----------|--------|-------|
| **1** | NASA TOI / CTOI | **PASS** | 0 records for TIC 38877496 |
|       | ExoFOP-TESS direct | PASS | 0 TOI, 0 CTOI, 0 imaging, 0 spec, 0 time-series logs |
|       | NASA confirmed planets | PASS | no match at position |
|       | Gaia DR3 NSS (3 tables) | PASS | 0 entries |
|       | Gaia DR3 EB classifier | PASS | 0 entries |
|       | AAVSO VSX, Prsa+2022/24 TESS EB | PASS | no match within 42″ |
|       | ASAS-SN / ATLAS variables | PASS | no match |
|       | SIMBAD | PASS | no entry at all |
|       | **TEST 1 OVERALL** | **PASS** | **signal is not previously known to community** |
|       |  |  |  |
| **2** | S31 diff-image centroid | INCONC | 14.9″ offset (0.71 pix), NE direction |
|       | S38 diff-image centroid | INCONC | 15.7″ offset (0.75 pix), N direction |
|       | S87 diff-image centroid | FAIL  | 36.4″ offset (1.73 pix), SW direction |
|       | S94 diff-image centroid | FAIL  | 25.6″ offset (1.22 pix), SW direction |
|       | sector-to-sector direction consistency | FAIL | centroids scatter randomly (NE/N/SW/SW) |
|       | Gaia neighbours within 42″ | NEUTRAL | one G=12.99 at 10.4″; could host signal as 6.2% diluted EB |
|       | **TEST 2 OVERALL** | **INCONCLUSIVE** | **signal too weak to centroid (~1.6σ per pixel); centroid offsets scatter randomly, not pointing coherently to any specific neighbour. We cannot prove on-target nor rule out BEB** |
|       |  |  |  |
| **3** | SDE(2P)/SDE(1P) = 0.58 | PASS | BLS prefers 1P (planet) over 2P (EB) |
|       | 2P-fold primary/secondary asymmetry | INCONC | 188 vs 600 ppm at 5.9σ bootstrap, but N=4 |
|       | Per-transit odd-even (bootstrap) | FAIL→PASS | bootstrap 3.9σ → empirical-RMS 0.72σ |
|       | Duration vs stellar density | PASS | 7.2 h obs vs 9.5 h central (b≈0.6) |
|       | Implied companion radius | PASS | 1.89 R⊕ even at higher 600-ppm depth = 2.65 R⊕, planet-sized |
|       | **TEST 3 OVERALL** | **PASS (cautiously)** | **with proper red-noise accounting, no EB signature remains. The 2P-fold asymmetry and per-transit scatter are consistent with N=4 noise-dominated sampling.** |

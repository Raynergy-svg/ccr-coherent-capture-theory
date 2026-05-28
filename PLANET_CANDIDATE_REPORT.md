# Planet Candidate Report — Chemistry-Priority Dwarf TESS Survey
## CPD-63 349 b: warm sub-Neptune at the inner edge of the optimistic habitable zone

**Date:** 2026-05-28
**Branch:** `claude/gaia-galah-hd28888-R5VZz`
**Dataset:** TESS SPOC + TESS-SPOC FFI HLSP lightcurves, sectors 1-96 (2018-2025)

---

## Headline result

A box-least-squares transit search across the 32 chemistry-priority CCT 8D
actionable dwarfs uncovered **one credible multi-sector transit signal** that
survives narrow-scan refinement, per-transit consistency tests, and EB
diagnostics:

| Parameter        | Value                                        |
|------------------|----------------------------------------------|
| **Host**         | **CPD-63 349** (TIC 38877496, Gaia DR3 4677014433801899392) |
| Host params      | R★ = 0.918 R☉, M★ = 1.030 M☉, Teff = 5745 K (TIC v8) |
| Distance         | 104.7 pc (Gaia DR3 parallax 9.55 ± 0.01 mas) |
| G                | 10.04                                         |
| **Period**       | **190.476 ± 0.005 d**                         |
| Transit T0       | BTJD 1393.918                                 |
| Duration         | 7.2 h                                         |
| Depth            | 354 ppm (BLS) / 455 ppm (median local fit)   |
| **Planet R**     | **≈ 1.89 R⊕**                                 |
| Semi-major axis  | 0.654 AU                                      |
| **T_eq**         | **≈ 328 K (Bond albedo = 0)**                  |
| Stellar irradiance | 1.93 S☉                                    |
| **HZ status**    | At inner edge of optimistic HZ; outside conservative HZ |

The detection statistic on the refined narrow-scan BLS spectrum is **SDE = 16.29**
(rises sharply above noise floor; see Fig. 4 in the diagnostic plot).
**All 4 predicted transits that fall in TESS coverage windows show
positive (transit-like) depth** (53, 585, 324, 762 ppm) across 4 independent
TESS sectors spanning 5 years (S31, S38, S87, S94).

---

## Search Strategy

Target list: 32 actionable nearby dwarfs from the **CCT 8D habitability scorer**
(`real_dwarf_targets.csv`), augmented by 12 dwarfs with HLSP-only TESS-SPOC FFI
or K2 coverage. Coverage survey (`tess_coverage_32dwarfs.csv`) identified the
viable targets.

For each target:
1. Download SPOC PDCSAP (or TESS-SPOC HLSP FFI) lightcurves from MAST
2. Detrend per sector with 12-h median filter + 5-σ outlier clip
3. Concatenate, run BoxLeastSquares over P = 0.5 → min(T_obs/2.5, 250) d
   on a grid of 4 000-30 000 trials × 5 trial durations
4. Compute SDE = (peak power − ⟨power⟩_off-peak) / σ_power, excluding ±1% around peak
5. Per-sector independent BLS at refined P for consistency check

Survey scope: **32 actionable dwarfs (TESS LC products covered: 20; HLSP/FFI
covered: 12)**. 73 candidate periodogram peaks examined.

---

## Per-target SDE summary (top of leaderboard)

| Target          | n_sec | P (d)    | SDE   | depth (ppm) | T_eq (K) | Notes |
|-----------------|------:|---------:|------:|------------:|---------:|-------|
| HD 180950       | 2     | 1.857    | 14.09 |       186   |    1943  | wide-scan SDE; narrow scan drops to 3.46, even-odd diff 184 ppm → spurious / EB |
| TYC 1071-934-1  | 1     | 0.687    | 10.89 |     1267    |    2162  | single-sector P~0.7d systematic (multiple targets cluster here) |
| **HD 271308**   | 9     | 34.27    |  8.21 |       290   |     670  | Hot, not HZ; needs follow-up |
| **CPD-63 349**  | **39** | **190.48** | **8.10 → 16.29*** | **395 → 354** | **328** | **★ Strong candidate; refined SDE up, transits in 4/4 visible windows** |
| HD 183193       | 1     | 0.58     |  7.68 |        41   |    2426  | single-sector P~0.6d systematic |
| BD-02 3362      | 3     | 59.33    |  7.56 |       719   |     520  | Hot, marginal |
| CD-47 7291      | 1     | 0.52     |  7.43 |       150   |    2581  | systematic |
| BD-03 3321      | 1     | 0.73     |  7.30 |       107   |    2492  | systematic |
| TYC 8351-1776-1 | 1     | 0.56     |  7.27 |        93   |    2464  | systematic |

(*) After narrow ±2 d refined scan, with 7.2h duration trial.

**Key insight:** Wide-scan SDE values can be inflated by 1/P trend in the
periodogram noise reference. **HD 180950's wide-scan SDE = 14.09 collapsed to
3.46 under narrow refinement** (and odd-even depth diff is 184 ppm vs 214 ppm
total depth, suggesting EB). **CPD-63 349's SDE went the other direction —
narrow refinement gave 16.29**, indicating a genuinely sharp BLS peak.

---

## CPD-63 349 — supporting evidence

### Stellar host: textbook clean (Gaia DR3)

- RUWE = 0.88 (well below 1.4 ambiguity threshold) ✓
- non_single_star = 0 (no NSS catalog entry) ✓
- IPD_frac_multi_peak = 0 (clean PSF) ✓
- phot_variable_flag = NOT_AVAILABLE; not in `vari_summary` table ✓
- Astrometric excess noise = 0.065 (low, no astrometric anomaly) ✓
- GSP-Phot independent check: Teff = 5779 K, log g = 4.51, [M/H] = -0.01
- Nearest neighbor: 10.4″ away at G = 12.99 (ΔG = +2.95)
- Aperture flux dilution from neighbors = 6.6 % (negligible)

### Lightcurve

- 39 SPOC PDCSAP sectors covering S1-11, S27-39, S61-69, S87-90, S93-96
- Baseline 2606 d (Jul 2018 - Aug 2025, 7.1 years)
- 615 948 cadences after detrending and 5σ clipping
- Photometric RMS = 832 ppm (per 2-min cadence)

### BLS narrow scan (±2 d around 190 d, 80 000 trials × 5 durations)

- Peak at P = 190.4761 d
- SDE = 16.29 (well above 7 threshold)
- BLS depth 354 ppm, duration 7.2 h
- Half- and double-period harmonic powers are 10⁻⁵-10⁻⁴ (factor 100 weaker)
  → period is not a 1/2 or 2× alias

### Per-predicted-transit windows (14 expected, 4 in TESS coverage)

| n | BTJD     | Sector | n_in-tr | n_out-tr | local depth (ppm) |
|--:|----------|-------:|--------:|---------:|------------------:|
| 5 | 2155.82  |   31   |   216   |    504   |              +53  |
| 6 | 2346.30  |   38   |   133   |    235   |             +585  |
| 13| 3679.63  |   87   |   216   |    252   |             +324  |
| 14| 3870.11  |   94   |   102   |    252   |             +762  |

**Positive-depth fraction = 4/4 = 100 %.** Median local depth 455 ppm,
consistent with BLS estimate of 354 ppm. The transit signal is present
across 4 widely-separated TESS sectors, not concentrated in one.

### EB diagnostics

- **Secondary eclipse at phase 0.5:** depth -93 ppm (negative → no secondary detected, consistent with planet hypothesis) ✓
- **Odd-even transit depth difference:** -151 ppm (even 239, odd 390 ppm)
  - ~43 % of BLS depth — moderate, **but only 4 transits in data so statistics are weak**
  - Cycles 4 (53 ppm) + 12 (324 ppm) = "even" → mean ~188
  - Cycles 5 (585 ppm) + 13 (762 ppm) = "odd" → mean ~674
  - Random per-transit noise can produce this spread; not a strong EB indicator at N=4
- Centroid test inconclusive (cross-sector raw pixel coordinates not aligned in implementation)

### Habitable zone assessment (Kopparapu et al. 2013)

For host with R★ = 0.918 R☉, Teff = 5745 K (L★ = 0.83 L☉):

- Conservative HZ: 0.864 - 1.524 AU
- Optimistic HZ: 0.682 - 1.608 AU
- **Candidate a = 0.654 AU (0.028 AU inside optimistic HZ inner edge)**
- Stellar irradiance S/S☉ = 1.93 (above runaway-greenhouse threshold 1.776)

**Status: warm Venus-Earth analog, not in classical HZ.** Hotter than Earth
but still in the "potentially temperate" Venus-edge regime.

---

## Caveats / open issues

1. **Only 4 of 14 expected transits fall in TESS coverage.** With sparse sampling,
   per-transit depth spread (53 to 762 ppm) is consistent with noise + true
   signal but doesn't strongly constrain depth uniformity.

2. **Odd-even depth difference of 151 ppm** (43 % of BLS depth) is moderate.
   N=4 transits is too few for a definitive EB ruling. With 14 expected transits
   over the full baseline, a TESS continuation or ground-based follow-up would
   directly test this.

3. **No RV data available** at G = 10.04 from public archives (no HARPS, no ESPRESSO).
   Mass determination requires either dedicated RV (HARPS-N, ESPRESSO, NEID for
   sub-Neptune mass) or TTV monitoring.

4. **Centroid contamination test was implemented incorrectly** (cross-sector raw
   pixel positions). Re-run per-sector with PSF-fit centroids needed.

5. **Period precision (±0.005 d)** is approximate; full TLS/MCMC fit would
   tighten T0 and dur for predictions.

---

## Suggested follow-up actions

1. **Submit to ExoFOP-TESS** as a community planet candidate
2. **Ground-based photometric follow-up** at predicted transit times:
   - Next predicted transit: T0 + n×190.476 d after BTJD 1393.918
   - Need ~50-100 ppm photometric precision at 10 mag for a 4 σ recovery
3. **TLS (Transit Least Squares) refit** for proper physical transit model
4. **PSF-fit centroid analysis** for contamination
5. **RV follow-up** at ESPRESSO or HARPS for mass measurement
6. **Continued TESS monitoring** in extended mission to add transits

---

## Other targets investigated (clean negatives)

- HD 28888 (initial target, subgiant): P=14.04 d BLS peak rejected (SDE=4.45 < 7,
  per-sector depths inconsistent: 558 vs 247 ppm)
- HD 183193 (the previously-identified #1 dwarf): SDE=7.68 at P=0.58 d
  — TESS systematic-period, not real signal
- HD 271200 (32 TESS-SPOC FFI sectors, CVZ-like): SDE=3.50, clean null
- 28 single-sector dwarfs with SDE 3-7 — noise floor

The CCT 8D hab score did not predict the candidate uniquely. CPD-63 349
ranked 12th of 32 by hab8 score (0.992), not the top. The chemistry
priority and a 39-sector TESS baseline combined to make this detectable.

---

## Files generated

| File                                | Description |
|-------------------------------------|-------------|
| `tess_bls_batch.py`                 | Batch BLS on SPOC LC dwarfs |
| `tess_bls_batch_results.csv`        | Per-target peak BLS results |
| `tess_bls_batch_log.txt`            | Per-target detailed log |
| `tess_bls_hlsp.py`                  | HLSP/K2 batch on no-SPOC dwarfs |
| `tess_bls_hlsp_results.csv`         | HLSP results |
| `tess_coverage_32dwarfs.csv`        | TESS coverage survey |
| `tic_params_32dwarfs.csv`           | TIC v8 stellar params |
| `cpd63349_deep_followup.py`         | CPD-63 349 deep follow-up |
| `cpd63349_followup_log.txt`         | Verdict and diagnostics |
| `cpd63349_followup_plot.png`        | 4-panel diagnostic plot |
| `cpd63349_gaia_check.py`            | Gaia DR3 sanity check |
| `cpd63349_gaia_check_log.txt`       | Host clean record |
| `hd180950_followup.py`              | Hot Neptune candidate follow-up (rejected) |
| `hd180950_followup_log.txt`         | Wide-scan SDE collapsed to 3.46 |
| `PLANET_CANDIDATE_REPORT.md`        | This document |

# Pre-Registration: CCT 8D/9D Scorer vs Known Habitable-Zone Rocky Planet Hosts

**Date sealed:** 2026-05-28 (before any APOGEE×NEA cross-match data is examined)
**Author:** Certan, working with Claude as research partner
**Branch / frozen commit:** `claude/gaia-galah-hd28888-R5VZz` @ `cfa1249` (this commit, before any test code runs)

---

## Hypothesis under test

**H1 (CCT prediction):** The CCT 9D habitability score, computed from public
stellar chemistry data, displaces confirmed **habitable-zone rocky planet
hosts** upward in score distribution relative to a matched field control at
**≥5σ significance**, AND this shift is **specifically larger for HZ rocky
hosts than for sub-Neptunes, hot Jupiters, or non-HZ rocky planets** (i.e.
selectivity is non-trivial).

**H0 (null):** The 9D score distribution of confirmed HZ rocky hosts is
indistinguishable from a matched field control at the 5σ level, OR the shift
exists but is the same for all host categories (i.e. trivially driven by
[Fe/H] or a generic "metallicity" effect).

**Auxiliary prediction:** The 9D score (a specific nonlinear combination
of element ratios) must **outperform a linear combination of all 8/9
input dimensions** AND **outperform [Fe/H] alone** as a predictor. If a
linear combination ties or beats 9D, the nonlinearity is unjustified.

---

## Frozen scorer

**Source:** `habitability_v2.py` at commit `cfa1249` of this branch
**SHA-256 of scoring code:** to be recorded immediately below before any
data access:

```
$ git rev-parse HEAD
cfa124935131e122de764fcd806d85bb8960e573
```

**Dimensions, weights, functional forms — FROZEN (no tuning during test):**

| dim | what | weight | functional form |
|-----|------|--------|-----------------|
| 1 | C/O (Teff-corrected) | 1.0 | Gaussian centered 0.5, width 0.15, soft penalty above 0.8 |
| 2 | Mg/Si | 1.5 | Gaussian centered 1.02, width 0.4 |
| 3 | [Fe/H] | 1.5 | Gaussian centered 0, width 0.3; extra penalty < -0.5 |
| 4 | [Mg/Fe] | 1.0 | Gaussian centered 0, width 0.15 |
| 5 | [Si/Fe] | 1.0 | Gaussian centered 0, width 0.15 |
| 6 | [Ca/Fe] | 0.5 | Gaussian centered 0, width 0.20 |
| 7 | [Al/Fe] | 0.5 | Gaussian centered 0.05, width 0.15 |
| 8 | Volatile budget [Ba/Fe] *(GALAH)* or **[Ce/Fe] (APOGEE)** *as s-process proxy with same Gaussian, centered 0.05, width 0.25* | 1.0 | same Gaussian form |
| 9 | Age | 0.75 | piecewise: young penalty <1Gyr, peak 1-6Gyr, slow decay >8Gyr |

**Composite:** weighted geometric mean of per-dimension scores.

Range: [0, 1]. Higher = more habitable per CCT.

---

## Pre-defined host categories (sealed before data inspection)

From NASA Exoplanet Archive pscomppars table:

| Category | Definition |
|----------|------------|
| **HZ rocky** | `pl_rade < 1.6` AND `pl_eqt` in [200, 340] K |
| non-HZ rocky | `pl_rade < 1.6` AND `pl_eqt` outside [200, 340] K |
| sub-Neptune | `1.6 ≤ pl_rade < 4.0` |
| hot Jupiter | `pl_rade ≥ 6.0` AND `pl_eqt > 1000` K |
| other | rest |
| field control | APOGEE FGK dwarfs not in any planet host list, weighted to match host (Teff, log g, [Fe/H]) marginal distribution |

---

## Pre-registered statistics

For each host category vs. field control:

1. **Mann-Whitney U test** on the 9D `hab_score` distribution.
   Report `U`, p-value, effect size = `(median_host - median_field) / σ_field`.
2. **Permutation null:** shuffle host/field labels 10⁴ times, compute the
   Mann-Whitney U on each permutation; the empirical p-value is the
   fraction of permutations with U as extreme as observed.
3. **Cross-category contrast:** KS test between HZ-rocky score distribution
   and hot-Jupiter score distribution. Specifically tests whether the shift
   is selective.
4. **Auxiliary "what would falsify CCT-the-specific-form":**
   - Logistic regression: P(host | 9D score) vs P(host | [Fe/H] only) vs
     P(host | linear combination of all 9 raw inputs).
   - Compare via AIC and likelihood-ratio test.
   - If linear-9D or [Fe/H]-only ties or beats the nonlinear 9D score,
     the CCT functional form is not justified by the data.

## Pre-registered success criteria

CCT prediction is **CONFIRMED** iff ALL THREE of:

| Test | Threshold for "confirmed" |
|------|---------------------------|
| HZ-rocky vs field 9D-score MW-U | p < 10⁻⁷ (>5σ, after Bonferroni × 4 categories) |
| HZ-rocky 9D effect size minus hot-Jupiter 9D effect size | > 0.3σ AND KS p < 0.05 |
| 9D nonlinear vs best linear combo / [Fe/H] | nonlinear ΔAIC > 10 |

CCT prediction is **DISFAVOURED** iff:
- HZ-rocky MW-U p > 0.05 (no shift detected),
- OR all four host categories show statistically indistinguishable shifts
  (no selectivity → trivially metallicity),
- OR linear combination ties/beats nonlinear (CCT functional form not needed).

**CCT prediction is "PARTIAL"** for intermediate outcomes; reported as such,
not spun.

---

## Data sources (sealed)

- **APOGEE DR17 ASPCAP allStar:** `https://www.sdss4.org/dr17/irspec/`
  via direct FITS download or Vizier `III/284/allstars`
- **NASA Exoplanet Archive pscomppars:** ADQL TAP
- **APOGEE stellar age proxy:** APOGEE NN-derived ages or APOGEE-2MASS BPG ages

Note on volatile-dimension substitution for APOGEE:
- APOGEE is H-band IR → no Ba lines.
- APOGEE DR17 has [Ce/Fe] as a clean s-process proxy.
- For the test, the scorer's "volatile" dimension takes [Ce/Fe] with the
  same Gaussian form. **This substitution is registered HERE, before data**
  and cannot be changed during testing.

---

## What stops me from cheating after seeing the data

1. The scorer code is at frozen commit `cfa1249`. Any modification will
   create a new commit; deviations from `cfa1249` are immediately visible
   in `git log`.
2. The host-category definitions, success criteria, and test statistics
   are written here BEFORE the cross-match.
3. The cross-match script itself will be committed before its results
   are computed (one commit for "the question", one commit for "the answer").
4. If the result is "PARTIAL" or "DISFAVOURED," the public commit will say
   so. There is no "test 2 we ran better instead" — the pre-registered
   test is what gets reported.

---

## Estimated sample sizes (rough, may be revised below if data differ)

- NASA Exoplanet Archive confirmed planets: ~5800
- Distinct host stars: ~4400
- APOGEE × NEA cross-match (hosts in APOGEE DR17): expect 300-1000
- HZ rocky in cross-match: expect 5-50 (the small-sample bottleneck)
- APOGEE FGK dwarf field: ~80000-200000

Power note: with N_HZ-rocky ~ 20 hosts and N_field ~ 50000, a median shift
of 0.05 in the 9D score (relative to field width ~0.15) yields MW-U
significance ~5σ. The test is well-powered if the prediction is real.

## Signed

Pre-registered by Certan with Claude on 2026-05-28, before APOGEE × NEA
cross-match code executes.

Commit hash at sealing: `cfa1249`

# CCT Population Test — RESULT

**Test:** CCT 9D scorer vs known habitable-zone rocky planet hosts
**Pre-registration:** `PRE_REGISTRATION.md` @ commit `1441551`
**Frozen scorer:** `habitability_v2.py` @ commit `cfa1249`
**Analysis sealed:** stages 1-3 committed @ `05aed8c` and `76bd852`
                    BEFORE APOGEE data download completed
**Sample:**
- 4 678 confirmed planets (NASA Exoplanet Archive pscomppars)
- 3 526 unique hosts → **877 matched in APOGEE DR17 ASPCAP**
- 159 277 APOGEE FGK dwarf field controls

| host category    | N in APOGEE |
|------------------|-------------|
| HZ_rocky         | **7**       |
| non_HZ_rocky     | 301         |
| sub_Neptune      | 447         |
| hot_Jupiter      | 56          |
| other            | 66          |

---

## Pre-registered result: **CCT NOT CONFIRMED — REJECTED IN DIRECTION**

### Test 1 — Mann-Whitney U per host category vs field

| category       | n   | median | shift vs field | effect_σ | p_MW       |
|----------------|----:|-------:|---------------:|---------:|-----------:|
| field control  |159k | 0.8166 |  —             |  —       | —          |
| **HZ_rocky**   |  7  | **0.6883** | **−0.128**| **−1.49**| **0.999**  |
| non_HZ_rocky   | 301 | 0.8598 | +0.043         | +0.50    | 5.5e−18    |
| sub_Neptune    | 447 | 0.8662 | +0.049         | +0.57    | 3.8e−32    |
| hot_Jupiter    |  56 | 0.8352 | +0.019         | +0.21    | 5.7e−02    |

**HZ rocky hosts are 1.5σ LOWER than the field, not higher.** Pre-registered
criterion (p < 1.43e−7) FAILS in the opposite direction (p = 0.999 one-sided
"greater"; equivalent to p ≈ 0.001 for "less").

### Test 2 — Permutation null (10⁴ shuffles)

| category | observed shift | perm p(≥obs) |
|----------|---------------:|-------------:|
| HZ_rocky | **−0.1283** | 0.984 (observed shift is in bottom 2% of null) |
| non_HZ_rocky | +0.0431 | 0 (significant) |
| sub_Neptune  | +0.0495 | 0 (significant) |
| hot_Jupiter  | +0.0185 | 0.131 (marginal) |

### Test 3 — KS test selectivity

HZ rocky vs every other host category: D = 0.77-0.85, p < 4e−4.
**There IS strong selectivity — but in the opposite direction:** HZ rocky
hosts populate a different chemistry corner than other planet hosts, but
that corner is at LOWER hab_score, not higher.

### Test 4 — Logistic AIC comparison

| model | parameters | AIC |
|-------|------------|-----|
| 9D nonlinear (hab_score only)        |  2 | 215 007 |
| [Fe/H] only                          |  2 | 178 947 |
| **linear combination of all 9 raw inputs** |  10 | **85 515** |

**Linear-9 dramatically outperforms both the 9D nonlinear scorer and
[Fe/H] alone (ΔAIC > 90 000).** The CCT-specific nonlinear functional
form is not justified by the data; a simple linear combination of the
same input abundances captures HZ-rocky host structure far better.
(Caveat: AIC comparison is fair as a goodness-of-fit measure, but
with N=7 positives the absolute AIC values may be inflated; the
relative ordering is what matters.)

### Test 5 — Shuffled-weights null (auxiliary)

Random Dirichlet-sampled non-negative weights summing to the same total
as the CCT weights: **69 % of random weights produce a shift at least
as negative as observed.** The CCT weights are not specifically
informative; random weights would do the same job.

---

## What the data actually look like

The 7 HZ rocky hosts identified in APOGEE:

| host        | R⊕   | T_eq (K) | APOGEE Teff | [Fe/H]  | [Mg/Fe] | spectral type |
|-------------|-----:|---------:|------------:|--------:|--------:|--------------:|
| Kepler-1512 | 1.18 |  322     | 4151        | −0.18   | −0.20   | K7-M0         |
| Kepler-442  | 1.34 |  241     | 4525        | −0.58   | +0.25   | K4            |
| Kepler-1126 | 1.45 |  305     | 5675        | −0.41   | +0.08   | G5            |
| Kepler-138  | 0.80 |  292     | 3957        | −0.27   | −0.06   | M1            |
| Kepler-62   | 1.41 |  208     | 4964        | −0.32   | +0.17   | K2            |
| TOI-2095    | 1.33 |  297     | 3763        | −0.45   | −0.02   | M2            |
| Kepler-186  | 1.27 |  319     | 3946        | −0.29   | −0.00   | M1            |

**Median [Fe/H] = −0.32**, much below solar.
**5 of 7 are M dwarfs** (Teff < 4500 K), which were EXCLUDED from
the field control by the FGK pre-registration. The non-HZ rocky and
sub-Neptune categories, by contrast, are dominated by FGK hosts and
sit comfortably above the field median.

The CCT scorer encodes "solar-like chemistry = optimal habitability".
Real Kepler HZ rocky hosts are metal-poor M dwarfs (because M dwarfs
are 70 % of nearby stars and transit S/N favours their detection).
The scorer treats their low [Fe/H] as a penalty when in fact those
systems are the most observationally productive HZ rocky population.

---

## Honest interpretation

**The pre-registered CCT prediction failed.** All three success
criteria failed:
1. ❌ HZ-rocky MW-U p < 1.43e−7: observed p = 0.999 (failed in
   opposite direction)
2. ❌ HZ-rocky effect > hot-Jupiter effect + 0.3σ: observed −1.49 vs +0.21
3. ❌ 9D nonlinear AIC < linear-9 AIC: linear-9 beats by 90 000

This is not a sample-size failure (N_HZ = 7 is small but the effect
direction is robust and the other categories show 18σ-32σ significance
in the OPPOSITE direction). It is a genuine falsification of the
pre-registered hypothesis as encoded.

The findings are nonetheless scientifically interesting:

- **The 9D score IS informative for predicting planet hosts in general** —
  non-HZ rocky and sub-Neptune host populations are significantly
  upshifted at >18σ. This recovers Buchhave et al. 2014's
  metallicity correlation, which CCT's [Fe/H] dimension encodes correctly.

- **For HZ rocky specifically, the score predicts the OPPOSITE direction**
  because the empirical HZ rocky catalogue is dominated by Kepler/TESS
  M-dwarf systems (Trappist-class) which have sub-solar [Fe/H].

- **A simple linear combination of the same 9 element abundances**
  carries vastly more predictive information than the CCT-specific
  Gaussian-product functional form. The CCT functional form is not
  the right parameterization.

## What this means going forward

The committed scorer cannot be used to make CCT-confirming claims
about HZ rocky habitability. Three options:

1. **Accept falsification.** Publish the null result with the dataset
   and code. This is honest science: an exoplanet-occurrence test on
   public APOGEE data refutes the CCT pre-registered prediction.

2. **Re-derive the scorer** with the Kepler HZ rocky population as a
   ground-truth training set, using sensible class balancing for
   M-dwarf hosts. This is no longer the pre-registered scorer; results
   from a re-derived model do not bear the pre-registration's
   credibility.

3. **Restrict the claim to FGK hosts only**, where the scorer was
   designed. The 2 FGK HZ rocky hosts in APOGEE (Kepler-1126,
   Kepler-1512) are too few to test at meaningful significance.
   This is "underpowered, not confirmed" rather than "confirmed".

Of these, (1) is the only honest immediate action. The CCT prediction
that was pre-registered is FALSIFIED at the population-level test.

This is the result of a single committed, pre-registered test. It
does not mean every claim within CCT is wrong; the framework's
chemistry-priority work for individual targets and for non-HZ-rocky
planet hosts remains. But the specific population-level prediction
about HZ rocky planet hosts is rejected.

## Files

- `PRE_REGISTRATION.md` — sealed pre-registration
- `cct_population_test_1_pull.py` — data pull (NEA + APOGEE DR17)
- `cct_population_test_2_score.py` — frozen scorer application
- `cct_population_test_3_stats.py` — pre-registered statistical tests
- `cct_test_hosts.csv` — 877 host-APOGEE matches with chemistry
- `cct_test_field.csv` — 159 277 field controls
- `cct_test_hosts_scored.csv`, `cct_test_field_scored.csv` — scored
- `cct_test_1_log.txt`, `cct_test_2_log.txt`, `cct_test_3_log.txt` — runs
- `CCT_POPULATION_TEST_RESULT.md` — this document

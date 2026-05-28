# HD 28888 — 9D Habitability Scorer Result

**Gaia DR3 3312653647717790720** | scorer: `habitability_v2.py` (Certan 2026)

> **Dimension-set note.** The *implemented* scorer scores C/O, Mg/Si, [Fe/H], [Mg/Fe], [Si/Fe], **[Ca/Fe], [Al/Fe]**, Volatile([Ba/Fe]), Age. The request's conceptual 9D listed **birth-cluster density** and **Galactic kinematic stability** instead of [Ca/Fe]/[Al/Fe]. The cohort was scored with the implemented set, so only those nine can be ranked; cluster density & kinematics are reported as context only.

## 1. Nine dimension scores + composite

C/O: raw 0.308 -> Teff-corrected **0.285** (implemented linear-slope detrend; the quoted rho=-0.68 is the correlation, not the formula).

| # | Dimension | input | score (0-1) | weight |
|---|---|---|---|---|
| 1 | C/O | 0.285 | 1.0000 | 1.0 |
| 2 | Mg/Si | 1.034 | 0.9993 | 1.5 |
| 3 | [Fe/H] | +0.130 | 0.9109 | 1.5 |
| 4 | [Mg/Fe] | +0.020 | 0.9909 | 1.0 |
| 5 | [Si/Fe] | +0.006 | 0.9993 | 1.0 |
| 6 | [Ca/Fe] | -0.043 | 0.9777 | 0.5 |
| 7 | [Al/Fe] | +0.086 | 0.9715 | 0.5 |
| 8 | Volatile(Ba/Fe) | -0.040 | 0.9374 | 1.0 |
| 9 | Age | 6.31 Gyr | 1.0000 | 0.75 |

**Composite hab_score = 0.9728** (weighted geometric mean), matching the stored cohort value 0.9728.

> **Mg/Si definition flag.** The scorer uses Mg/Si = $10^{[Mg/Fe]-[Si/Fe]}$ = **1.034** (score 0.9993), *not* the profile's solar-normalised number ratio **1.086**. Feeding 1.086 into the scorer (Gaussian centred at 1.02) would drop that dimension to 0.9865 and the composite to **0.9706**. The cohort was scored with the 1.034 convention, so that is used here for a valid comparison.

Age sensitivity: FLAME age 6.05 Gyr -> s_age 1.0000, composite **0.9728** (Δ=+0.0000; both ages <8 Gyr so s_age=1.0).

## 2. Per-dimension percentile vs the 4,970 excellent cohort (>0.9)

Percentile = % of the excellent cohort with sub-score ≤ HD 28888's (ties noted for saturated dims).

| Dimension | HD score | pctile vs 4,970 | pctile vs full 12,234 |
|---|---|---|---|
| C/O | 1.0000 | 100.0 | 100.0 |
| Mg/Si | 0.9993 | 88.1 | 92.1 |
| [Fe/H] | 0.9109 | 28.4 | 57.0 |
| [Mg/Fe] | 0.9909 | 69.1 | 78.0 |
| [Si/Fe] | 0.9993 | 89.2 | 92.5 |
| [Ca/Fe] | 0.9777 | 42.3 | 59.5 |
| [Al/Fe] | 0.9715 | 70.0 | 82.3 |
| Volatile(Ba/Fe) | 0.9374 | 66.1 | 76.4 |
| Age | 1.0000 | 100.0 | 100.0 |

## 3. Overall percentile rank

- Composite hab_score = **0.9728**
- vs **full FGK cohort (N=12234)**: rank **692** -> top **5.66%** (percentile 94.3). **Top 10%, NOT top 1%** (1% cutoff 0.9874, 0.1% cutoff 0.9946, max 0.9987).
- vs **4,970 excellent (>0.9)**: rank **692** -> top **13.9%** (percentile 86.1). **NOT top 10%** within the excellent group.

## 4. Rank among the 18 actionable targets

By the full v2 composite hab_score, HD 28888 is **#1 of 18** (hab_score 0.9728).
The README's "0.973" **is** this v2 composite (0.9728), not a separate older/simpler score — so #1 on the actionable list and the 0.973 are the same scorer. Its #1 status comes from the **filter stack** (RUWE<1.4 single + <200 pc + 2-8 Gyr + zero RV coverage + no known planets), NOT from having the highest chemistry score (it is 692th of 12234 on chemistry alone).

Top of the actionable list by hab_score:

| rank | name | hab_score |
|---|---|---|
| 1 | HD  28888 | 0.9728 |
| 2 | CD-51 11403 | 0.9660 |
| 3 | HD  54524 | 0.9587 |
| 4 | CD-73   972 | 0.9583 |
| 5 | HD 133866 | 0.9577 |

## 5. Strongest / weakest dimensions
**Strongest (saturated at ceiling):** Age (1.000), C/O (1.000), Mg/Si (0.999), [Si/Fe] (0.999)

**Weakest (drag on composite):** [Fe/H] (0.911), Volatile(Ba/Fe) (0.937), [Al/Fe] (0.971)

- **[Fe/H] (0.911)** is the single biggest drag (super-solar +0.13, weight 1.5).
- **Volatile/[Ba/Fe] (0.937)** is second ([Ba/Fe]=-0.04, just below the +0.05 optimum).
- C/O and Age sit exactly at 1.0 (flat plateaus), contributing nothing to differentiation.

## Caveats

**(a) Subgiant vs calibration regime.** The scorer is not ML-trained; it is a hand-built geometric scorer validated against FGK planet hosts, with C/O Teff-detrend from a 360k FGK baseline. Its population cut is Teff 4000-7000 K and **log g > 3.8** (`habitability_v2.py:191`). HD 28888 (log g 3.94) **passes** that cut and IS in the scored cohort -- it is NOT outside the calibration regime. But log g>3.8 admits turnoff/subgiants (true dwarfs are log g>4.2), and the entire actionable-18 sit at log g~3.8-4.0, so the sample is biased toward evolved stars. R=1.99 Rsun means the C/O Teff-detrend (calibrated on the broad FGK locus) is applied at the cool-subgiant edge.

**(b) First dredge-up / birth C/O.** Surface [N/Fe]=+0.38 with [C/Fe]=-0.07 is the CNO-mixing signature. Conserving C+N (birth [N/Fe]=0) gives birth C/O ≈ **0.433** (full) / **0.371** (50% mixing), vs surface 0.308. Re-scored: birth C/O Teff-corrected 0.410 -> s_CO 1.0000; composite **0.9728** (Δ=+0.0000). **No change**: C/O 0.31-0.43 all sit on the flat s_CO=1.0 plateau (0.15-0.65). The correction strengthens the *narrative* (natal C/O nearer solar) but not the score.

**(c) [Zr/Fe]=-0.60 outlier.** The implemented Volatile dimension uses **[Ba/Fe] only**; [Y/Fe] and [Zr/Fe] are NOT inputs to any scored dimension. So the unflagged Zr outlier has **zero impact** on the composite. (It would only matter if the s-process dimension used Y/Zr; the code does not.)

## Context dimensions (not in the implemented scorer)
- **Birth cluster density (Teutsch_80):** a chemical-nearest-template label only; kinematics/position/age exclude real membership (cluster at 2.4 kpc, 132° away, 0.1-0.2 Gyr). Not scored.
- **Galactic kinematic stability:** ecc 0.165, R_peri 6.33 / R_apo 8.83 kpc, z_max 0.063 kpc, J_R 34.4, J_Z 0.135, L_Z 1723 -- a quiet, near-circular thin-disk orbit. Not scored.
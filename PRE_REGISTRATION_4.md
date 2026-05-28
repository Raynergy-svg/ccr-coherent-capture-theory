# Pre-Registration #4: Dimensional Decomposition + Out-of-Sample Test

**Date sealed:** 2026-05-28 (before per-dimension Levene/MW tests or CV fits run)
**Branch / sealing commit:** `claude/gaia-galah-hd28888-R5VZz` @ `8676a71`
**Frozen scorer:** `habitability_v2.py` @ commit `cfa1249` (unchanged through tests #1-#4)

**Prior tests:**
- Pre-reg #1 (`1441551`) → REJECTED at `7a7a93e` (HZ-rocky all-Teff failed)
- Pre-reg #2 (`1d58aa7`) → REJECTED at `060803d` (multi-planet coherence was [Fe/H] artifact)
- Pre-reg #3 (`b20613f`) → MIXED at `8676a71`:
    - HZ-rocky FGK: rejected (thick-disk chemistry)
    - **non-HZ-rocky FGK CONFIRMED at p = 5.3 × 10⁻⁸** after strict matching
    - sub-Neptune FGK PARTIAL (p = 5.5 × 10⁻⁸, effect 0.28 just below 0.3 threshold)

This is the dimensional decomposition follow-up: which specific
dimensions carry the test-#3 signal, and does the surviving subset
hold up out of sample.

---

## What this test asks

Pre-reg #3 confirmed that the CCT 9D scorer carries real information
beyond [Fe/H] for non-HZ-rocky and sub-Neptune FGK hosts. The hab_score
shift was +0.018-0.023 at p = 10⁻⁸ after strict (Teff, log g, [Fe/H])
matching.

But the hab_score is a weighted geometric mean of 9 per-dimension
sub-scores. The +0.02 shift could be driven by any subset of the 8
non-[Fe/H] dimensions, or by all of them weakly. And in-sample
significance at p = 10⁻⁸ with N = 271-417 does not guarantee
out-of-sample predictive power — the in-sample fit could be a
data-dependent artifact that does not generalise.

**This test asks two specific questions:**

1. **Which non-[Fe/H] dimensions individually carry the signal?**
   Per-dimension MW test of host-vs-control shift in each
   sub-score (s_CO, s_MgSi, s_MgFe, s_SiFe, s_CaFe, s_AlFe,
   s_volatile, s_age), Bonferroni-corrected across 8 dimensions.

2. **Does the surviving subset improve out-of-sample prediction
   beyond [Fe/H] alone?** 70/30 train/test split on hosts+controls;
   logistic regression fits on training, log-loss on held-out test.
   The CCT-defensible claim requires the surviving-dimensions model
   to beat [Fe/H]-alone by > 0.02 in held-out log-loss.

---

## Pre-registered design (frozen)

### Samples (no change from test #3)

- **non-HZ-rocky FGK hosts:** N ≈ 271 (same set as test #3)
- **sub-Neptune FGK hosts:** N ≈ 417 (same set as test #3)
- **Strict-matched FGK field controls** (k=10 NN in
  Teff/100, log g/0.1, [Fe/H]/0.05) per host category

### Match-quality gate (pre-registered)

Before any analysis runs, KS test on (Teff, log g, [Fe/H]) for each
host category vs its matched control. **Proceed only if all three KS
p > 0.95** (i.e., matching is statistically indistinguishable).

If any axis fails, the test is REJECTED for that category as
"matching failure" with no further analysis.

### Dimensions tested (frozen, alphabetical for reproducibility)

The 8 non-[Fe/H] sub-scores produced by `habitability_v2.py`:

1. `s_age` (age proxy)
2. `s_AlFe` ([Al/Fe])
3. `s_CaFe` ([Ca/Fe])
4. `s_CO` (C/O Teff-corrected)
5. `s_MgFe` ([Mg/Fe])
6. `s_MgSi` (Mg/Si)
7. `s_SiFe` ([Si/Fe])
8. `s_volatile` (Ce/Fe in APOGEE)

Bonferroni alpha = 0.05 / 8 = 0.00625.

### Step 1: Per-dimension decomposition

For each dimension D and each host category C:
- Compute median(D_host) − median(D_match_control)
- Compute effect size = shift / (IQR_control / 1.349)
- Mann-Whitney U one-sided test, alternative = "greater"
- p value recorded

A dimension is **"surviving"** if its host-vs-control p < 0.00625
(Bonferroni) AND the shift is positive (consistent with CCT
prediction that hosts score higher).

### Step 2: Consistency check

Cross-tabulate surviving dimensions between non-HZ-rocky and sub-Neptune
categories. Dimensions that survive in BOTH are the strongest CCT
candidates. Dimensions surviving in only one are reported but treated
as weaker evidence.

### Step 3: Out-of-sample held-out CV (the real test)

For each category separately:

- Combine hosts (label = 1) and matched control (label = 0).
- 70/30 stratified train/test split, seed = 42 (pre-registered).
- Standard-scale features using training-set statistics only.
- Fit three logistic regression models on the training half:
   - **(a) [Fe/H] alone** (the literature baseline)
   - **(b) [Fe/H] + the dimensions that pass step 1** (CCT-narrow)
   - **(c) full 9-dim raw inputs, linear** (everything-linear baseline)
- For each model, compute **mean log-loss on the held-out 30 %**.
- Higher (less negative) is better.

### Step 4: Within-[Fe/H]-bin sanity check

For each surviving dimension D from step 1:
- Bin all (hosts + controls) into [Fe/H] bins of width 0.1 dex.
- Within each bin, re-test host-vs-control shift in D.
- A dimension is **"non-trivially surviving"** if shift direction
  is consistent across at least half the populated bins.
- A dimension whose signal flips sign or disappears within bins is
  **flagged as residual [Fe/H] correlation** despite the strict
  matching, and reported as such.

---

## Success criteria (frozen)

For each category (non-HZ-rocky, sub-Neptune) independently:

| outcome | criterion |
|---|---|
| **CONFIRMED** | (≥ 1 surviving dimension from step 1) AND (model b beats model a by > 0.02 in held-out log-loss) AND (≥ 1 surviving dimension passes step 4 within-bin sanity) |
| **PARTIAL** | dimensions pass step 1 but model b improvement < 0.02 OR within-bin sanity fails for all surviving dimensions |
| **REJECTED** | 0 dimensions pass step 1, OR model b ≤ model a out-of-sample |

The overall test conclusion synthesises the two categories:

- Both confirmed → strong CCT-narrow finding
- One confirmed, one partial → category-specific signal
- Both partial / rejected → CCT-narrow does not hold up; only
  [Fe/H] is the real predictor

The held-out log-loss threshold (Δ > 0.02) is the key pre-registered
gate. Anything that beats [Fe/H] by less is below noise floor of
typical CV experiments for N of this size.

---

## Guards against prior traps

1. **In-sample vs out-of-sample.** Step 3 splits 70/30. The
   surviving-dimensions model could fit training-set noise and lose
   on held out. Pre-registration commits us to the held-out result.

2. **[Fe/H] residual masquerade.** Even after strict matching, [Fe/H]
   distribution may have small remaining differences inside the
   tolerance. Step 4 re-checks each surviving dimension within
   narrower [Fe/H] bins.

3. **Direction-of-effect cherry-picking.** The MW test is one-sided
   "greater" (CCT prediction). Negative-direction dimensions do not
   pass.

4. **Multiple-testing inflation.** Bonferroni across 8 dimensions per
   category. Total of 16 tests across the two categories; no further
   correction because we treat the two as independent confirmations.

---

## Acknowledged limitations

- N_non-HZ-rocky = 271 and N_sub-Neptune = 417 after 70/30 split give
  189 / 292 in train. CV log-loss differences of 0.02 are detectable
  with these sample sizes but with non-trivial uncertainty.
- APOGEE [Ce/Fe] coverage is limited for cool dwarfs; some rows will
  carry the 0.7 default for `s_volatile`. This is a noise injection,
  not a bias in expectation.
- Age dimension uses host_st_age from NEA which is heterogeneous
  across discovery papers. The s_age sub-score is therefore noisier
  in hosts than in field (where age defaults to 0.7).

These are honest limitations, not exclusion criteria. They are
acknowledged here so post-hoc explanations cannot retroactively become
exclusion criteria.

---

## What stops me from cheating

1. Sealing commit `8676a71` precedes any per-dimension Levene/MW test
   or held-out CV.
2. The 8 dimensions are listed here in alphabetical order; the
   "surviving" set is read from data, not chosen.
3. Held-out CV split seed is pre-registered (= 42).
4. The Δ log-loss threshold (> 0.02 to beat [Fe/H]) is pre-registered
   here, not chosen after seeing results.
5. The analysis script will be committed in a separate commit BEFORE
   it produces any numeric output.

## Signed

Pre-registered 2026-05-28 by Certan with Claude as research partner.
Sealing commit: `8676a71`.

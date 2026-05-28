# Pre-Registration #2: Multi-Planet Host Chemistry-Coherence Test

**Date sealed:** 2026-05-28 (before matched-control construction code runs)
**Branch / sealing commit:** `claude/gaia-galah-hd28888-R5VZz` @ `80ab5b8`
**Previous test:** PRE_REGISTRATION.md @ `1441551` — FAILED (HZ-rocky shifted opposite direction)

This is an INDEPENDENT pre-registration. It does not borrow any
credibility from the failed first test. It targets the one finding that
appeared in the post-hoc diagnostic sweep, with the controls needed to
distinguish a real CCT signal from Kepler selection bias.

---

## Hypothesis under test

**H1 (CCT coherence prediction, sharpened):**
Multi-planet hosts (n_planets ≥ 2) have systematically tighter chemistry
scatter than properly-matched single-planet hosts, AND tighter than
properly-matched non-host control, in **multiple element abundances**,
at >5σ after Bonferroni correction over elements tested.

**H0 (null):** After matching on stellar parameters (Teff, log g) and
applying the same quality cuts, multi-planet hosts' chemistry scatter
is statistically indistinguishable from matched controls. Any apparent
scatter reduction in the unrestricted diagnostic sweep was driven by
Kepler selection bias on stellar parameters, not chemistry.

**Auxiliary direction prediction:** the ordering
σ(multi) < σ(single) < σ(matched field)
must hold for the same elements where the multi-vs-control test passes.
If single = field but multi tighter, OK (host coherence specifically).
If multi = single < field, the effect is not multiplicity-driven.

---

## Frozen design (sealed before any analysis)

### Samples

- **Multi-planet hosts:** APOGEE × NEA pscomppars matches with n_planets ≥ 2
  in the per-planet table. Restrict to FGK dwarfs (Teff 4500-7000, logg > 3.8).
- **Single-planet hosts:** same APOGEE × NEA, n_planets = 1, same FGK restriction.
- **Matched non-host control:** for each host (multi and single separately),
  draw k = 10 nearest neighbours from APOGEE FGK dwarfs (Teff 4500-7000,
  logg > 3.8, SNR > 70, not in any planet host list) using Euclidean
  distance in (Teff/100, logg/0.1) coordinates. Pool deduplicated.

### Elements tested (12, fixed before data inspection)

[Fe/H], [Mg/Fe], [Si/Fe], [Ca/Fe], [Al/Fe], [Ti/Fe], [Mn/Fe], [Ni/Fe],
[C/Fe], [O/Fe], [N/Fe], [α/M]

Volatile-dimension element [Ce/Fe] is excluded — too many APOGEE NaNs
on cool dwarfs (separate concern, not relevant here).

### Test statistic

For each element X:
- σ_multi  = standard deviation of [X/Fe] among multi-planet hosts
- σ_single = same for single-planet hosts
- σ_match  = same for matched control
- Levene test (median-centered) for σ_multi vs σ_match → p_multi_X
- Levene test for σ_single vs σ_match → p_single_X
- Levene test for σ_multi vs σ_single → p_mvs_X
- Also report the variance ratios.

### Bonferroni correction
Over 12 elements: α_corrected = 0.05 / 12 = 0.0042 for any single
significant element. For multi-element claim: require ≥3 elements
with p < 0.0042 AND consistent direction (σ_multi < σ_match).

### Auxiliary diagnostics (descriptive only, do not change verdict)

- Stellar parameter distribution overlap: KS test on (Teff, logg,
  [Fe/H]) between multi-planet hosts and matched control. Confirms
  the matching is effective.
- Bootstrap CI on σ_multi - σ_match for the most-affected element.

### Success criteria (pre-registered, hard thresholds)

**CCT-COHERENCE CONFIRMED iff ALL of:**
- ≥3 elements with σ_multi < σ_match at p < 0.0042 (Bonferroni)
- The pre-registered direction (multi tighter than match) holds for
  every passing element
- Stellar parameter KS test (Teff, logg, [Fe/H]) shows control is
  well-matched (p_KS > 0.01 on each axis — i.e., we cannot
  distinguish multi-planet host stellar params from matched control,
  confirming the matching is good)

**REJECTED (not coherence, just selection bias) iff:**
- All 12 elements have p_multi > 0.0042, OR
- The direction is inverted (σ_multi > σ_match) for the only
  significant elements

**PARTIAL (only 1-2 elements pass) iff:**
- Reported descriptively. Counts as "interesting hint, not coherence
  confirmation."

---

## Why this is a CCT-specific prediction and not generic

The unique CCT claim is that **chemical coherence at formation**
produces planet-system architectures. Other frameworks (Buchhave's
metallicity correlation, photoevaporation models, etc.) predict
**central-tendency** shifts in chemistry (host stars are metal-richer
on average) — NOT **scatter** reduction.

If multi-planet hosts have tighter chemistry scatter than single-planet
hosts at the same Teff/logg, this is a coherence-specific signal that
no metallicity-correlation model predicts directly. It would mean:
multi-planet formation requires a specific narrow chemistry window,
not just "above some [Fe/H] threshold."

That's the prediction this test isolates from the general "planet
hosts are metal-rich" effect.

---

## What stops me from cheating

1. This document is committed at the listed commit hash before any
   analysis code touches the cct_test_hosts_scored.csv / field data
   in a Levene-test context.
2. Analysis script will be committed in a separate, clearly-marked
   commit AFTER this pre-registration but BEFORE the Levene tests run.
3. The 12 elements are listed here, not chosen after looking at p-values.
4. The success thresholds (≥3 elements at p<0.0042) are written here.
5. If the result is REJECTED or PARTIAL, the commit message will say so.

## Predictions in advance — to be checked after running

Best guess from the post-hoc diagnostic sweep (which used unmatched
field control):
- [α/M], [Ca/Fe], [Mg/Fe], [Fe/H] showed σ-reduction at p < 1e-4 in
  the unmatched test
- Expectation under H_1: same elements should still show reduction
  after proper matching (effect at least 30 % of unmatched magnitude)
- Expectation under H_0: matching removes the effect, no elements
  significant after Bonferroni

This advance prediction is for accountability — the result might fall
between the two.

---

## Sample-size estimate

From cct_test_hosts.csv (a-priori known counts, written before tests run):
- 275 multi-planet hosts in APOGEE (all-Teff)
- ~250 expected after FGK cut
- ~600 single-planet hosts in APOGEE (all-Teff)
- ~550 expected after FGK cut
- Matched control pool: up to ~2500 stars after deduplication

With N_multi ~ 250 and N_match ~ 2500, Levene test detects σ ratios
≥ 1.15 at >5σ comfortably. The diagnostic sweep showed unmatched
σ ratios of 1.3-1.5 for some elements; even halved by matching, the
test should detect.

If sample is dramatically smaller than predicted (e.g., FGK cut
removes most multi-planet hosts), the test is underpowered and the
result will be reported as such.

## Signed

Pre-registered by Certan with Claude as research partner, 2026-05-28,
before matched-control construction code runs.

Commit hash at sealing: `80ab5b8`

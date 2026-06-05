# Phase C — Pre-Registration: N-Body Exchange Capture σ(κ) Scan

**Date sealed:** 2026-06-05 (BEFORE simulation code runs)
**Author:** Daniel Certan, with Claude
**Standing on:** `phase_a_derivation.md` (sealed, first-principles physics) and
  `THEORY_v3.md` (sealed, v3 theory statement and discipline rules)
**Discipline rule:** Definitions, parameter grids, and decision rules below
  are locked. They cannot be edited after this commit.

---

## Question

For a multi-planet system with internal angular-momentum coherence parameterized
by the von Mises-Fisher concentration κ, what is the exchange-capture cross-
section σ_exch(κ) during a stellar flyby — as a function of κ, encounter
pericenter r_p, encounter velocity v_∞, and mass ratio?

**Why this question:** the two literature surveys (sealed in this session's
audit trail) confirm that no published N-body study has treated κ as an
independent input parameter for exchange-capture cross-section measurement.
The Nov 5 Laws of Coherency asserted a "sweet spot" at 70-75% coherence with
N=5 trials per level. This study tests that assertion with N=100+ trials at
properly-defined κ levels, using a standard symplectic integrator (REBOUND).

---

## Definition of coherence (LOCKED)

For N planets with osculating angular-momentum unit vectors L̂_i, internal
coherence is parameterized by the **von Mises-Fisher concentration parameter
κ** of the distribution from which the L̂_i are drawn:

    p(L̂ | μ, κ) = (κ / (4π sinh κ)) exp(κ μ · L̂)

where μ is the system mean direction (set to ẑ without loss of generality).

For each simulation trial, the N inclination directions are sampled
independently from vMF(μ=ẑ, κ). This sets a controlled mutual-inclination
dispersion σ_i, with the approximate relation σ_i ≈ √(1/κ) radians for
κ > 5.

κ-grid (LOCKED):

| κ | Approx σ_i (deg) | Interpretation |
|---|---|---|
| 500 | 2.6° | very coplanar |
| 100 | 5.7° | nearly coplanar |
| 50 | 8.0° | tight |
| 20 | 13° | moderate |
| 10 | 18° | loose |
| 5 | 25° | broad |
| 2 | 40° | wide |
| 1 | 57° | very wide |
| 0.5 | ~80° | nearly isotropic |
| 0.1 | ~90° | isotropic |

10 κ levels, log-spaced.

This is the methodologically standard descriptor borrowed from the Kuiper-
belt directional-statistics community (Matheson & Saillenfest 2023, Van
Laerhoven+ 2023). It is operationally well-defined and not a free parameter.

---

## Planetary system setup (LOCKED)

Host star A: 1 M_sun.

Four giant planets at Solar-System-analog positions (Li, Mustill & Davies 2019
convention):

| Planet | a (AU) | m (M_Jup) | initial e |
|---|---:|---:|---:|
| J | 5.20 | 1.000 | 0.01 |
| S | 9.55 | 0.300 | 0.02 |
| U | 19.20 | 0.046 | 0.03 |
| N | 30.10 | 0.054 | 0.01 |

Initial inclinations: drawn per-trial from vMF(μ=ẑ, κ) for that κ-cell.
Initial argument of pericenter, longitude of ascending node, mean anomaly:
uniform on [0, 2π) per planet per trial.

---

## Encounter setup (LOCKED)

Intruder star B: 1 M_sun (equal-mass exchange; standard test case).

Encounter parameter grid:

| Parameter | Values |
|---|---|
| Pericenter distance r_p | {100, 200, 500, 1000} AU |
| Encounter velocity v_∞ | {0.5, 2.0} km/s |
| Encounter inclination | isotropic per trial |
| Phase at pericenter | uniform random per trial |

So 4 × 2 = 8 (r_p, v_∞) cells per κ level.

Total simulation grid: 10 κ × 4 r_p × 2 v_∞ = 80 cells.
N = 100 trials per cell → 8,000 simulations.

Pilot run (committed separately for sanity check): 5 κ values × 1 (r_p, v_∞)
cell × 30 trials = 150 simulations. Used only to validate the pipeline; not
counted in the main result.

---

## Integration (LOCKED)

- Integrator: REBOUND IAS15 (adaptive high-order Gauss-Radau; the standard
  for close encounters in the field).
- Pre-encounter setup: 100 years to settle planetary orbits.
- Encounter integration: from -1000 years before pericenter to +1000 years
  after pericenter.
- Post-encounter classification: integrate +10,000 years to identify stable
  bound state.
- Conservation requirement: |ΔE/E| < 10⁻⁸ over the full simulation.
  Trials failing this are excluded; if >5% of trials in a cell fail, the
  cell result is flagged as numerically suspect.

---

## Outcome classification (LOCKED)

At end of simulation, for each planet, compute energy relative to A and to B:
- Bound to A: E_A < 0 AND E_B > 0
- Bound to B: E_B < 0 AND E_A > 0
- Ambiguous (both negative; transient triple): re-classify at +50,000 yr
- Unbound: E_A > 0 AND E_B > 0 → ejected

Per-trial outcomes counted:
- N_retained_by_A
- N_exchanged_to_B
- N_ejected

Cell-level statistics (averaged over 100 trials per cell):
- P_any_exchange(κ, r_p, v_∞) = fraction of trials with N_exchanged_to_B ≥ 1
- P_group_exchange(κ, r_p, v_∞) = fraction with N_exchanged_to_B ≥ 2
- P_full_exchange(κ, r_p, v_∞) = fraction with N_exchanged_to_B = 4

Plus secondary observables (recorded but not the primary decision):
- Post-exchange eccentricity distribution of captured planets
- Post-exchange mutual inclination of any captured pair
- Inclination of captured planet's orbit relative to B's "spin" (taken as
  the original z-axis of A's frame for definiteness, since both stars are
  point masses in this simulation — this is a proxy for obliquity λ)

---

## Decision rules (LOCKED — do not edit after this commit)

The PRIMARY result is the functional shape of P_any_exchange(κ) averaged
across (r_p, v_∞) cells (and individually per cell, for robustness).

Fit a LOWESS smoother to P_any_exchange(κ) on log κ. Identify the location
and prominence of any local maximum:

- **PEAK_AT_INTERMEDIATE_KAPPA**: maximum at κ ∈ [5, 20] with prominence
  >2σ above the values at the κ-grid endpoints. *This is the outcome that
  would support the Nov 5 "70-75% coherence sweet spot" claim,*
  since κ ≈ 10 ↔ σ_i ≈ 18° corresponds to moderate coherence, not
  perfect alignment.

- **PEAK_AT_HIGH_KAPPA** (>20): maximum at the well-aligned end.
  Confirms a "more coherent = more group exchange" intuition; consistent
  with the mechanism but refutes the specific "sweet spot at intermediate"
  claim.

- **PEAK_AT_LOW_KAPPA** (<5): maximum at the incoherent end. Strongly
  counter-intuitive; would suggest that group exchange is actually MORE
  efficient for randomly-oriented systems (perhaps because each planet
  encounters B at different times so 1+ is always near).

- **MONOTONE_INCREASING**: probability increases monotonically with κ
  with no local maximum. Confirms intuition that more aligned groups
  exchange better, without a sweet spot.

- **MONOTONE_DECREASING**: probability decreases monotonically with κ.
  Coherent groups too tightly internally bound to release? Surprising.

- **FLAT**: max/min ratio of P_any_exchange across the κ grid < 1.5.
  Coherence does not measurably affect exchange probability. The
  distinctive CCT claim is refuted.

The same classification applied to P_group_exchange and P_full_exchange
as secondary checks. Discrepancies between the three (e.g., P_any flat
but P_group peaks) are reportable findings.

---

## What we report regardless of result

1. The complete σ_exch(κ, r_p, v_∞) table, in JSON and CSV.
2. The diagnostic plots (P vs κ at each r_p, v_∞).
3. Energy/momentum conservation diagnostics per cell.
4. Per-trial outcome counts.
5. Post-capture eccentricity and obliquity distributions, even though
   they are not part of the primary decision.

If the result is FLAT or MONOTONE_DECREASING, the distinctive CCT claim
(coherence as a parameter controlling exchange) is refuted by direct
N-body simulation. We report that honestly and the theory must either
be withdrawn or revised with a NEW pre-registration before further
empirical claims.

If the result is PEAK_AT_INTERMEDIATE_KAPPA, the Nov 5 claim is supported
by simulation. We then proceed to Phase D-v3: testing the predicted
post-capture observables (e, λ) against the most-current obliquity sample.

---

## Honest expectations recorded BEFORE running

Based on the Phase A derivation, my prior:
- P_any_exchange should increase weakly with κ (more aligned groups
  have higher coherent geometric cross-section), but the dependence
  is likely shallow (factor of <2 across the full κ range).
- The MONOTONE_INCREASING outcome is the most likely.
- No prediction of a sharp peak at intermediate κ from the standard
  three-body literature; the Nov 5 sweet spot claim would be a positive
  surprise.

This expectation is recorded so that any actual result deviating from
it can be evaluated against a stated prior, not retrofitted.

---

## Reproducibility commitment

- Random seed for the κ direction sampling: seed = 20260605 + κ_index +
  100 * trial_index.
- Each trial is fully deterministic given its seed.
- REBOUND version, numpy version, Python version recorded in the result
  JSON.
- Code committed to coherent_capture_v3/ before any run.
- Pilot results committed separately from main scan results, with
  clear labels.

---

End of pre-registration. Sealed.

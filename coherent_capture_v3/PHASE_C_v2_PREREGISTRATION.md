# Phase C v2 — Pre-Registration: Close-Encounter σ_exch(κ) Scan

**Date sealed:** 2026-06-05 (BEFORE Phase C v2 simulations run)
**Stands on:** PHASE_C_PREREGISTRATION.md (v1 sealed; FLAT result for r_p ≥ 100 AU), phase_a_v2_kappa_derivation.md (sealed analytical predictions)

## What changes from v1

Only the r_p grid: from {100, 200, 500, 1000} AU to **{10, 20, 30, 50, 75} AU**.

Justification: v1 sampled the regime where exchange is dynamically suppressed (Hut-Bahcall hard-binary, x ≪ 1 for inner-system planets at v_∞ ≤ 2 km/s). 0/8000 exchanges in v1 confirmed this. The close-encounter regime is where σ_exch is non-negligible and where the κ-dependence claim can be meaningfully tested.

Everything else (κ grid, v_∞, N_trials, vMF definition, planet setup, integrator, classification, conservation requirement) carries forward unchanged from v1.

## Grid

- κ ∈ {0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500} — 10 levels
- r_p ∈ {10, 20, 30, 50, 75} AU — 5 levels (NEW)
- v_∞ ∈ {0.5, 2.0} km/s — 2 levels
- N_trials = 100 per cell
- Total: 10 × 5 × 2 × 100 = **10,000 simulations**

## Decision rule (LOCKED from phase_a_v2_kappa_derivation.md)

Primary outcome: shape of σ_any_exch(κ) and σ_full_exch(κ) averaged over r_p and v_∞ cells.

- **MONOTONE_INCREASING_BOTH:** σ_any(κ) and σ_full(κ) both monotonically increase. Derivation confirmed. Nov 5 sweet-spot claim refuted; mechanism survives.
- **MONOTONE_INCREASING_FULL_FLAT_ANY:** σ_full monotonically increases, σ_any roughly flat. Also derivation-confirmed (within Phase A-v2 prediction).
- **PEAK_AT_INTERMEDIATE_KAPPA_ANY** (peak in σ_any at κ ∈ [5, 20] with 2σ prominence): supports Nov 5 sweet-spot claim, refutes derivation. Publishable surprise.
- **MONOTONE_DECREASING_ANY:** counter-intuitive; would require explanation.
- **FLAT_BOTH** (max/min < 1.5 for both): coherence doesn't matter even at close encounters. CCT distinctive claim refuted.

Honest expectation: MONOTONE_INCREASING_BOTH or MONOTONE_INCREASING_FULL_FLAT_ANY. The derivation does not predict a peak.

## What I report regardless

- Full σ(κ, r_p, v_∞) table in CSV and JSON
- Diagnostic plots
- Energy conservation per cell
- Post-capture e and i distributions conditional on κ (secondary observables; not part of primary decision but pre-committed to record)

Sealed.

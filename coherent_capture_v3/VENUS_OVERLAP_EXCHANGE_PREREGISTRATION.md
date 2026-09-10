# Venus overlap-region stellar-exchange N-body test

**Date:** 2026-09-10  
**Status:** SEALED BEFORE OUTCOME CALCULATION  
**Parent:** `JOINT_GATE_SCREEN_PILOT_RESULT.md` and the locked Venus capture design.

## Question

Within the only continuous gas-damping overlap identified by the joint screen, can a Venus-mass donor planet undergo genuine stellar exchange onto a heliocentric orbit, and can any exchange arrive already cold enough (`e <= 0.05`, `i <= 3.4 deg`) for the calibrated gas-damping route?

## Locked pilot

This is a focused forward N-body tranche, not the full Venus production survey.

- donor star: `1.0 M_sun`
- capturing star (Sun analogue): `1.0 M_sun`
- planet: Venus mass, initially circular at `a_D = 0.723332 AU`
- `q/a_D = {2, 3, 4}`
- `v_inf = {0.10, 0.30, 0.50, 1.00} km/s`
- four independently scrambled Sobol batches per cell
- 128 orientations/phases per batch
- 512 trials per `(q/a_D, v_inf)` cell
- total denominator: `6144` encounters

Each Sobol point samples donor-planet mean anomaly plus an isotropic encounter-plane normal and periapsis orientation. The grid and denominator cannot be changed after outcomes are inspected.

## Integrator and encounter construction

Use REBOUND IAS15 in `AU, yr, Msun`. The donor planet begins bound to the donor star. The Sun analogue approaches on a two-body hyperbola with the locked `q` and `v_inf`. Start radius is `R0 = max(100 AU, 100 a_D, 20 q)` and integration continues until the intruder is outbound beyond `R0`.

A candidate is a **genuine exchange** only when the planet is bound to the Sun analogue and unbound from the donor at the classification epoch. Double-bound states are not counted as exchanges. Candidate exchanges are integrated for 100 heliocentric orbital periods and must retain the same classification.

Numerical gate: relative total-energy error and relative total-angular-momentum error must each be `<= 1e-10`; failed cases are retried once with tighter IAS15 epsilon. Cases still failing are reported as numerical failures and excluded from success numerators but retained in the full attempted denominator.

## Recorded outcomes

Every trial records input cell/batch/Sobol index, seed/scramble provenance, capture class, heliocentric `a,e,i`, donor-relative energy sign, Sun-relative energy sign, persistence result, energy error and angular-momentum error.

Primary counts per cell and globally:

- attempted encounters
- numerically valid encounters
- genuine persistent exchanges
- `cold_exchange`: persistent exchange with `e <= 0.05` and `i <= 3.4 deg`
- `venus_orbit_near`: persistent exchange with `a` within 10% of `0.723332 AU` (diagnostic only)
- joint `cold_near`: cold exchange also within that 10% semimajor-axis diagnostic

Binomial fractions use Wilson 95% intervals. Zero events are reported with a 95% upper bound, never as probability zero.

## Decision labels

- `ROBUST_COLD_OVERLAP`: at least 5 `cold_exchange` events and they occur in at least two adjacent `q/a_D` values for the same `v_inf`.
- `COLD_OVERLAP_OBSERVED`: at least 1 `cold_exchange`, but robust criterion not met.
- `EXCHANGE_ONLY_HOT`: at least 1 persistent exchange but zero `cold_exchange`.
- `EXCHANGE_VANISHES_IN_OVERLAP_REGION`: zero persistent exchanges in all 6144 encounters.
- `NUMERICALLY_UNRESOLVED`: more than 1% of attempts fail the numerical gate.

These labels apply only to this pilot tranche. A positive result establishes dynamical compatibility, not historical occurrence. A null constrains only this locked donor-star/orbit model and sampled phase-space measure.
# Phase A-v2: Analytical κ-dependence of σ_exch

**Date sealed:** 2026-06-05 (before Phase C v2 close-encounter scan)
**Author:** Daniel Certan, with Claude
**Builds on:** phase_a_derivation.md (sealed first-principles, kept as foundation)
**Purpose:** Derive σ_any_exch(κ) and σ_full_exch(κ) analytically, so the Phase C v2 close-encounter pre-registration tests a physics-derived prediction rather than another asserted shape.

---

## Setup

A planetary system of N planets orbits Star A. Each planet's orbital angular momentum direction L̂_i is drawn from vMF(μ=ẑ, κ). Star B passes through with pericenter distance r_p, encounter velocity v_∞, impact direction isotropic.

Define:
- **σ_any_exch(κ)**: cross-section for ≥1 planet ending up bound to B
- **σ_full_exch(κ)**: cross-section for *all* N planets ending up bound to B

The exchange cross-section for a single planet of semi-major axis a, in the slow-encounter regime, is (Hut & Bahcall 1983; Heggie & Hut 2003, Eq. 23.21):

    σ_single_exch ≈ C(M_A, M_B, m_p) · π a² · F(x)

where x = v_∞/v_orb(a), F(x) is the dimensionless dependence (F(x) ~ x⁴ for hard binaries x≪1, F(x) ~ 1 for x ~ 1), and C is a mass-ratio factor of order M_B/(M_A+M_B+m_p).

For a *single* planet, σ_single_exch has no κ-dependence — κ doesn't enter individual two-body cross-sections.

The κ-dependence comes from the *joint* outcome over N planets. This is what we derive.

---

## Geometric setup

When Star B passes through the system at pericenter r_p, define the "exchange shell" as the spatial region around B where a planet could be captured. The shell radius ~ Hill radius of B,

    r_H(B) ≈ (M_B / (3 M_A))^(1/3) · r_p ≈ 0.69 · r_p   for M_A = M_B

For r_p = 30 AU (passes through the J/S/U/N envelope), r_H(B) ~ 20 AU. A planet within ~20 AU of B at closest approach has nontrivial probability of capture.

The geometric question becomes: at the moment of B's pericenter passage, *how many of the N planets are inside the Hill shell*?

This depends on:
1. The radial positions of planets (semi-major axes are fixed; eccentric anomalies are uniformly distributed)
2. The angular positions (κ-dependent — coherent groups have correlated angular positions; incoherent groups have random angular positions)

---

## High-κ limit (coplanar, coherent)

All planets have L̂_i ≈ ẑ. The planets all lie in the same plane.

When B passes with pericenter r_p in a random direction, the geometry decides whether B's trajectory comes inside the planetary plane near the system.

For pericenter direction within angle θ_p of the plane normal: B passes through the plane near pericenter, sweeping a chord through the system. **All planets in the planet are simultaneously in B's encounter region.** Either many are captured (if r_p is small) or none are (if r_p is large), but the outcome is correlated.

For pericenter direction far from the plane normal: B never enters the planetary disk. **No exchange.**

So at high κ, the cross-section is essentially binary on pericenter direction:
- P(any exchange | encounter geometry favorable) ≈ 1 if r_p < a_outermost
- P(any exchange | encounter geometry unfavorable) ≈ 0

The fraction of pericenter directions that are "favorable" is geometric:
    f_favorable(κ → ∞) ≈ (a_outermost / r_p) · sin(θ_p)  
                        for r_p > a_outermost

For r_p = 30 AU, a_outermost = 30 AU, f_favorable ≈ sin(θ_p) ~ 0.5.

So at high κ:
    σ_any_exch(κ → ∞) ≈ π r_p² · f_favorable · F(x) ≈ 0.5 π r_p² · F(x)
    σ_full_exch(κ → ∞) ≈ σ_any_exch (favorable geometry captures all)

---

## Low-κ limit (isotropic, incoherent)

The L̂_i are uniformly distributed. Planet positions are random both radially and angularly.

The angular positions of the N planets at the encounter moment are essentially uncorrelated. Each planet has independent probability p_capture of being inside B's Hill shell at closest approach.

For a single planet at semi-major axis a_i, the probability it's within Hill shell of B during the encounter:
    p_i ≈ r_H(B)² / (4π a_i²)
                        for r_H < a_i

Total expected number of planets in shell:
    E[N_in_shell] = Σ p_i ≈ N · r_H² / (4π · ā²)  where ā is the typical scale

For the J/S/U/N system, ā ~ 16 AU, r_H ~ 0.69 r_p:
    E[N_in_shell] ≈ 4 · (0.69 r_p)² / (4π · 16²) ≈ 5.9e-4 · r_p²/AU²

At r_p = 30 AU: E[N_in_shell] ≈ 0.53. So roughly one planet at a time near B on average.

By Poisson approximation (independent planets):
    σ_any_exch(κ → 0) ≈ π r_p² · [1 - exp(-E[N_in_shell])] · F(x)

For E[N_in_shell] ~ 0.5: P(any) ≈ 0.39

So at low κ:
    σ_any_exch(κ → 0) ≈ 0.39 π r_p² · F(x)

And σ_full_exch is much smaller because P(all 4 in shell simultaneously) ≈ p_i⁴ ~ 10⁻⁴.

    σ_full_exch(κ → 0) ≈ 10⁻⁴ · σ_any_exch(κ → 0)

---

## The two cross-sections, predicted κ-dependence

Combining the two limits and interpolating monotonically:

**σ_any_exch(κ):** Has competing effects:
- High κ favors "if B is in plane, captures many"
- Low κ favors "one of N planets is always somewhere near B"
- The geometric f_favorable factor at high κ (~0.5) vs the independent-probability sum at low κ (~0.39)

**Net prediction:** σ_any_exch(κ) is **roughly flat or weakly increasing with κ** — within a factor of ~1.3 across the full κ range.

**σ_full_exch(κ):** Strongly monotonic:
- At low κ: σ_full ~ p_i^N ~ 10⁻⁴ (vanishing)
- At high κ: σ_full ~ σ_any ~ 0.5 π r_p² · F(x)
- The ratio σ_full(high κ) / σ_full(low κ) ≈ 5000

**Net prediction:** σ_full_exch(κ) is **strongly monotonically increasing with κ**, with no peak at intermediate κ.

**The "sweet spot at 70-75% coherence" claim from Nov 5 has no analytical basis** in this derivation. Neither σ_any nor σ_full has a peak at intermediate κ.

---

## Where a peak could in principle come from (steel-manning Nov 5)

For completeness, let me steel-man the original claim. A peak at intermediate κ could arise if:

1. **At high κ:** the group is so tightly coplanar that internal dynamics during encounter lock the planets together — they either all get exchanged or none do (binary outcome). If the "favorable geometry" condition is restrictive enough, σ_any could drop.

2. **At low κ:** individual planets are at random positions; each has only a small chance of being in the Hill shell. σ_any could be low because P(any in shell) ≈ E[N in shell] which is small.

3. **At intermediate κ:** the group is partially coplanar — some planets near each other, some scattered. Multiple "chances" for one to be in the shell, AND coherent enough that the group doesn't disperse before B arrives. Could plausibly peak.

This is a plausible *qualitative* argument but the analytical estimates above don't show numeric support for a peak. The factor-of-1.3 weak dependence at high κ vs the factor-of-2 increase at low κ probably crosses without producing a strong peak.

**Honest verdict:** Phase A-v2 does not derive a peak. The expected shape is roughly monotonic in both σ_any (weakly increasing) and σ_full (strongly increasing). The Nov 5 claim of a sweet spot at intermediate κ is not analytically supported.

---

## Predictions for the Phase C v2 close-encounter scan (LOCKED)

The Phase C v2 scan will sample r_p ∈ {10, 20, 30, 50, 75} AU — the regime where exchanges actually occur.

**Derived predictions:**

| Observable | Low-κ (0.1) prediction | High-κ (500) prediction | Shape |
|---|---|---|---|
| σ_any_exch | 0.3–0.5 × baseline | 0.4–0.6 × baseline | weakly monotone increasing OR flat |
| σ_full_exch | <0.001 × baseline | 0.3–0.6 × baseline | strongly monotone increasing |
| σ_partial (1<n<N) | mid-range | small | monotone DECREASING with κ |

Where "baseline" ~ π r_p² · F(x).

**Pre-registered decision rule for Phase C v2 (binding):**

- **MONOTONE_INCREASING confirmed** (both σ_any and σ_full): Phase A-v2 derivation confirmed. Coherent groups exchange-capture better, no sweet spot. The Nov 5 specific claim is refuted but the general mechanism survives.

- **MONOTONE_DECREASING in σ_any** (high-κ groups *less* likely to have ≥1 exchanged): would be a real surprise, contradicting the derivation. Would need to be investigated.

- **PEAK_AT_INTERMEDIATE_KAPPA in σ_any** (5-20 range): would *support* the Nov 5 sweet spot claim and *refute* the Phase A-v2 derivation. Would be a publishable positive finding.

- **FLAT in BOTH σ_any AND σ_full** (max/min < 1.5): coherence doesn't matter even at close encounters. Refutes the distinctive CCT claim entirely.

- **PEAK in σ_full at intermediate κ**: physically implausible but recorded as a possibility.

---

## What I am NOT changing from the original Phase C pre-registration

- The vMF κ definition and grid: same as Phase C v1.
- The planetary system setup: same J/S/U/N analog.
- The simulation methodology: REBOUND IAS15, same conservation requirement.

What changes for Phase C v2:
- r_p grid: {10, 20, 30, 50, 75} AU (close-encounter regime, derived to be where σ_exch is non-negligible).
- v_∞ grid: keep {0.5, 2.0} km/s.
- N=100 trials per cell.
- Total: 5 × 2 × 10 × 100 = 10,000 simulations.

Expected wall-clock: close encounters are slower per sim due to small timesteps near pericenter; estimate 1.5–3 s per sim. With 4 workers: 1–2 hours.

End of Phase A-v2 derivation. Sealed.

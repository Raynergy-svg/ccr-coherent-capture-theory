# T16c audit — "fingerprint permanent over 0–10 Gyr" is overclaim

T16c's published headline: chemical fingerprint shows **no decay over 0–10 Gyr**;
Spearman ρ = +0.097, p = 0.14; exp-decay fit τ = 100 ± 280 Gyr; AIC favours
flat over decay by ΔAIC = 2.8. Concluded: "fingerprint is permanent."

**The test cannot support that conclusion. Two structural issues.**

## Issue 1: age coverage is grossly inadequate

The 253 cluster templates that T16c uses have age coverage:

| age bin | N templates | fraction |
|---|---|---|
| 0–1 Gyr | **202** | **87 %** |
| 1–2 Gyr | 13 | 6 % |
| 2–4 Gyr | 10 | 4 % |
| 4–6 Gyr | 5 | 2 % |
| 6–8 Gyr | 1 | 0.4 % |
| **8–10 Gyr** | **0** | **0 %** |

**Maximum age in the sample: 6.31 Gyr.** The "0–10 Gyr" claim is being made
from a sample that doesn't reach 8 Gyr, let alone 10. And the test is
dominated by 0–1 Gyr clusters (87 % of the sample).

The upstream t9_cluster_stats_with_age.csv has 606 clusters with ages, and
the same problem: only **2** clusters above 5 Gyr (and both happen to be in
the 253 templates because they're tight). The age coverage isn't a T16c
problem — GALAH itself doesn't observe many >5 Gyr open clusters.

## Issue 2: power analysis — what tau can the test actually rule out?

I simulated synthetic data with `E(t) = E0·exp(-t/τ)` at the actual ages
of the 231 clusters with enrichment + age, using the actual per-cluster
enrichment noise (median σ ≈ 4.3, from MC + Poisson). 2000 simulations
per τ.

| true τ (Gyr) | frac p<0.05 (power) | frac p<0.01 | median ρ_observed |
|---|---|---|---|
| 0.5 | **90 %** | 75 % | −0.21 |
| 1.0 | **76 %** | 52 % | −0.17 |
| 2.0 | 47 % | 23 % | −0.12 |
| 5.0 | **18 %** | 6 % | −0.07 |
| 10.0 | **10 %** | 3 % | −0.04 |
| 20.0 | 6 % | 2 % | −0.02 |
| ∞ (flat) | 4 % | 0.5 % | +0.004 |

The test has confident power **only for τ ≲ 1 Gyr** (80 % power at τ = 0.7
Gyr). For τ = 5 Gyr the power is 18 %; for τ = 10 Gyr it is 10 %; for
τ = 20 Gyr it is at the 5 % false-positive baseline — i.e., **literally
indistinguishable from flat**.

Translation: an observed p = 0.14 with this dataset is *equally
consistent* with τ = 5 Gyr, τ = 10 Gyr, τ = 100 Gyr, or true permanence.
The test simply doesn't distinguish them.

## Bonus issue: template selection introduces survivorship bias

The 253 templates are selected from 606 age-having clusters by requiring
`C_O_std < 0.10` and N ≥ 5. The pass rate by age:

| age bin | full cohort N | pass C_O_std < 0.10 |
|---|---|---|
| 0–1 Gyr | 511 | 49 % |
| 1–2 Gyr | 61 | **25 %** |
| 2–5 Gyr | 32 | 44 % |
| 5–12 Gyr | 2 | 100 % |

The C_O_std cut systematically excludes the looser 1–2 Gyr clusters at
twice the rate of any other age bin. So the templates are not a fair
sample of old clusters — they're the *tight survivors* at every age,
which is the survivorship bias T17 already documents.

## What the data actually support

A defensible reading of T16c, given the audit:

- The chemical fingerprint is detectable in field-star matches at aggregate
  ~3× enrichment, robustly across matching scales 0.5× to 2× — **this part
  is solid** (the multi-scale check is clean).
- The test rules out decay timescales **τ < ~1 Gyr** at moderate
  significance (76 % power for τ = 1 Gyr).
- The test **cannot distinguish** decay timescales of 5, 10, 100 Gyr, or
  permanence — they all produce p ~ 0.1 with this sample.
- The "0–10 Gyr" age range in the claim is wrong on its face: max age in
  the sample is 6.31 Gyr.
- The AIC ΔAIC = 2.7 in favour of flat is "weak evidence only" by
  conventional thresholds (ΔAIC > 4 wanted).

## Recommended rephrasing

> Original: *"The chemical fingerprint shows no decay over 0–10 Gyr; data
> are consistent with permanent chemical identity (τ = 100 ± 280 Gyr; AIC
> favours flat model by ΔAIC = 2.8)."*

> Revised: *"The chemical fingerprint shows no detectable decay over the
> 0–6 Gyr age range covered by the GALAH template sample (N = 231 clusters
> with ages; Spearman ρ = 0.097, p = 0.14). A power analysis indicates the
> test rules out decay timescales τ < ~1 Gyr at moderate significance but
> cannot distinguish τ ≥ 5 Gyr from permanence with this sample.
> Extending the claim to longer timescales requires older cluster
> coverage than GALAH DR4 provides."*

This is consistent with the data, doesn't overclaim, and is honest about
the precision-wall limitation (which the paper already centres).

## Files
- `t16c_permanence_data.csv` — existing per-cluster output
- (no new script — audit done inline; reproducible from the t16c outputs +
  power-analysis simulation in the conversation log)

# T14 audit — "τ = 1.29 Gyr matches disk heating" is plausible but loose

T14's published claim: intra-cluster C/O scatter weakly grows with cluster age
(Spearman ρ = +0.117, p = 4.1×10⁻³, N = 606), and an exponential-decay fit
gives τ = 1.29 ± 1.58 Gyr, "matching the disk-heating timescale (1–2 Gyr)."

**Verdict: the qualitative correlation is real; the specific τ point estimate
is well within the right ballpark but the data don't constrain it tightly.**

## Reproduction (clean)

Re-ran the same fit on `t9_cluster_stats_with_age.csv`:

| quantity | published | reproduced |
|---|---|---|
| Spearman ρ | +0.117 | +0.117 |
| Spearman p | 4.1×10⁻³ | 4.1×10⁻³ |
| N clusters | 606 | 606 |
| exp fit τ | 1.29 ± 1.58 Gyr | 1.29 ± 1.58 Gyr |
| AIC(exp) vs AIC(power-law) | difference 0.5 | difference 0.5 |

T14 reproduces exactly.

## What the numbers actually mean

**The Spearman correlation is real but tiny.** ρ² = 0.014 — age explains
~1.4 % of the cluster-to-cluster C/O scatter variation. 98.6 % of the
variance is from other sources (intrinsic ISM mixing, measurement noise,
member-impurity contamination). The signal is statistically robust (high N
makes p=0.004 easy) but practically modest.

**The exponential-fit τ ± σ is dominated by curvature uncertainty.** The
formal one-sigma range is [−0.29, 2.86] Gyr — i.e., includes negative
values, meaning the fit is on the edge of being unconstrained on the lower
side. A bootstrap is more honest:

| bootstrap τ distribution (1000 resamples) | value |
|---|---|
| median | 1.33 Gyr |
| 16-84 pctile (1σ) | [0.80, 2.46] |
| 5-95 pctile (2σ) | [0.56, 4.36] |
| fraction in [1, 2] Gyr ("disk heating") window | 47 % |

So the bootstrap-honest 2σ range is τ ∈ [0.6, 4.4] Gyr. That includes the
disk-heating window [1, 2] Gyr but also extends to factor-of-3 faster and
factor-of-2 slower. "τ MATCHES disk heating" is true of the point estimate;
"τ IS the disk-heating timescale" overstates the precision.

## Sample limitation (same as T16c)

| age bin (Gyr) | N clusters |
|---|---|
| 0–1 | 511 (84 %) |
| 1–2 | 61 |
| 2–5 | 32 |
| 5–10 | 2 |
| >10 | 0 |

84 % of the sample is younger than 1 Gyr. Only 2 clusters are >5 Gyr; max
age 6.31 Gyr. The fit's asymptote (σ₀+A) is therefore very poorly
determined, which propagates into the τ uncertainty.

## Recommended rephrasing

> Original: *"Coherence degrades on τ = 1.29 Gyr (cluster dissolution
> timescale), Spearman p = 4 × 10⁻³."*

> Revised: *"Intra-cluster C/O scatter weakly increases with cluster age
> (Spearman ρ = 0.12, p = 4 × 10⁻³, N = 606; ~1 % variance explained). An
> exponential model gives a coherence-decay timescale τ = 1.3 Gyr (bootstrap
> 90 % CI [0.6, 4.4] Gyr), consistent with the disk-heating range [1, 2] Gyr
> but not uniquely constraining it. The sample heavily over-weights young
> clusters (84 % under 1 Gyr; max age 6.3 Gyr); τ values longer than ~5 Gyr
> cannot be ruled out and shorter than ~0.5 Gyr cannot."*

## Cross-comparison with T16c (no contradiction)

T14 (intra-cluster C/O scatter vs age, weakly grows) and T16c (field-star
enrichment vs age, no detectable trend) are not in contradiction even
though they superficially measure opposite things:

- T14: surviving clusters become less internally coherent over time
  (interloper accretion, evaporation, or chemical reshuffling among members).
- T16c: dissolved members retain their birth chemistry recoverable as field-
  star matches.

Both can hold simultaneously: the cluster *loses* members to the field, and
the dissolved members carry the original chemistry with them outside the
cluster boundary. T14 sees the cluster diluting; T16c sees the diluted
material in the field. Different populations, different tests.

Where they share a *limitation* is the same age-coverage problem — both are
running on a sample with virtually no clusters >5 Gyr.

## Files
- (audit done inline, no new script; reproduction code is in the
  conversation log and `t14_results.txt` matches.)

# T9 audit — cluster distinctness is real, but smaller than the p-value implies

T9 published claim: 655 GALAH open clusters have chemically distinct C/O
distributions (Kruskal–Wallis p < 10⁻¹⁰).

**Verdict: claim holds, but the magnitude is much smaller than the p-value
suggests, and a substantial fraction is the Galactic [Fe/H] gradient.**

## Reproduction
- Re-ran KW on `t9_matched_stars.csv`: H = 1170.8, p ≈ 8×10⁻³² (matches T9).
- Sample: 10,479 stars in 655 clusters (N≥3 per cluster).

## Permutation null
Shuffling cluster labels and re-running KW: permuted H median = 652, max = 745
across 200 perms. Observed H = 1171 sits **15.4σ above the permuted-null mean**.
Cluster identity is doing real work; the KW signal is not a statistical fluke.

## But how much of "cluster distinctness" is the [Fe/H] gradient?
Fitting `C/O = 0.296·[Fe/H] + 0.482` globally (across all stars), then re-doing
KW on residuals: **H drops 1170.8 → 887.1** (24 % reduction). Cluster identity
remains highly significant after [Fe/H] removal (p = 3×10⁻⁹), but a quarter
of the raw signal was the metallicity gradient.

At the cluster-mean level: inter-cluster C/O variance has **R² = 0.34** with
inter-cluster [Fe/H] — i.e., a third of cluster-to-cluster C/O differences
are predictable from cluster-to-cluster [Fe/H] differences (the Galactic
disk metallicity gradient).

## Variance partition

| component | variance | % of total |
|---|---|---|
| Total C/O variance | 0.01737 | 100 % |
| Between-cluster (signal) | 0.00186 | **11 %** |
| Within-cluster (noise) | 0.01551 | **89 %** |
| Between/within ratio | 0.12 | — |

The "cluster identity" signal is real but small — between-cluster differences
contribute only ~11 % of the total C/O variance, and after removing the
[Fe/H] gradient ~7 %.

## Recommended framing

> Original: *"Open clusters carry distinct multi-element chemical fingerprints
> (Kruskal–Wallis p < 10⁻¹⁰)."*

> Revised: *"Open clusters carry statistically distinct C/O fingerprints
> (KW p ≈ 8×10⁻³² across 655 GALAH clusters; observed H = 1171, 15σ above the
> permutation-null mean). Cluster identity contributes ~11 % of the total C/O
> variance, of which roughly one third is predictable from the inter-cluster
> [Fe/H] gradient; residual cluster identity beyond the metallicity gradient
> remains highly significant (KW p = 3×10⁻⁹ on [Fe/H]-detrended residuals)."*

This honestly reflects (a) cluster identity is real and statistically secure,
(b) part of the apparent distinctness is the Galactic metallicity gradient,
(c) the remaining signal after metallicity control is still robust.

## What this does NOT change

- The foundation of the T-series is intact: clusters DO have distinguishable
  C/O distributions beyond what the Galactic [Fe/H] gradient explains.
- The downstream tests (T10, T14, T16, T18, T19) rest on a real signal, just
  with smaller effective effect size than "p<10⁻¹⁰" implies.

## Files
- (audit done inline; reproduction in conversation log).

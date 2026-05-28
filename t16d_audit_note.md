# T16d audit (brief) — Ba/Fe alignment is partly chemodynamical background

T16d's published claim: matched field stars' Ba/Fe is closer to parent
cluster's Ba/Fe than random field stars' Ba/Fe is, in 97.2 % of clusters
(Wilcoxon p = 3.6 × 10⁻⁴¹). Framed as "blind cross-check" — an independent
confirmation of dissolved-member recovery via a dimension not used in the
matching.

**The dimension may be different but the underlying chemo-dynamical
background is not removed.** Random GALAH FGK pairs (no cluster involved)
matched on T16b's 3 main chemistry dims show:

| pair selection (random GALAH, 300,000 pairs) | median \|ΔBa/Fe\| | vs unmatched random |
|---|---|---|
| unmatched random | 0.161 dex | 1.000 |
| matched \|Fe/H\|<0.10 only | 0.144 dex | 10.3 % closer |
| matched \|Fe/H\|<0.05 only | 0.143 dex | 11.0 % closer |
| matched \|Mg/Fe\|<0.05 only | 0.161 dex | 0 % (Mg/Fe alone uncorrelated with Ba) |
| **matched all three (\|Fe/H\|<0.10, \|Mg/Fe\|<0.05, \|Si/Fe\|<0.05)** | **0.124 dex** | **23.2 % closer** |

When you select stars on the 4D (Fe/H + Mg/Fe + Si/Fe + C/O) chemistry that
T16b uses, the selected stars are by background **23 % closer in Ba/Fe**
than random — no cluster information required. T16d's "97 % of clusters
show matched closer than random in Ba/Fe" is exactly the sign statistic
you'd predict from this background (when matched stars are systematically
closer by a measurable amount, the per-cluster sign comes out matched <
random in nearly all clusters).

The Wilcoxon p = 10⁻⁴¹ then reflects the *high statistical power* of the
test (253 clusters × millions of matches), not an independent confirmation
beyond the background.

## What's left of the "three-channel dissolved-recovery" claim

| channel | claim | audit status |
|---|---|---|
| **T16b** chemical Mahalanobis | E ~ 2× field excess | **clean** (Mahalanobis vs MC random-center null, no kinematic confound) |
| T16d Ba/Fe blind | 97.2 % alignment | **partly background** (23 % closer Ba/Fe is the 4D-chemistry-matched chemodynamical floor; T16d's signal needs to be compared to that floor, not to all-field) |
| T16e RV kinematic | 5.2 % closer | **fully background** (5–8 % closer RV is the [Fe/H]-matched chemodynamical floor; T16e's signal is inside that) |

The recovery story reduces from three independent channels to one
(T16b) plus a Mahalanobis-matched bonus. T16d and T16e need re-doing with
chemistry-matched control populations to extract any genuine independent
signal.

## Recommended rephrasing for the paper

> Original: *"...recover dissolved members at ~3× enrichment, confirmed
> by barium proximity (p = 3.6 × 10⁻⁴¹) and kinematic coherence
> (p = 3.7 × 10⁻³⁷)."*

> Revised: *"...recover dissolved members at ~2–3× chemical enrichment
> over the Monte Carlo random-center null (T16b). The Ba/Fe blind-channel
> check shows the expected alignment but a substantial fraction of the
> Ba/Fe-closer signal is the chemodynamical background of 4D-chemistry-
> matched stars (random GALAH pairs matched on the same chemistry dims are
> already ~23 % closer in Ba/Fe); the residual signal beyond background
> would require an explicit chemistry-matched control population to
> isolate. The kinematic test on radial velocities shows a 5 % closer-RV
> effect that is fully consistent with the disk's [Fe/H]–RV chemodynamical
> background and does not isolate a birth-cluster signal."*

# T16e audit — proper placebo test: signal is marginal at best

## Two-stage audit

**Stage 1 (whole-field control):** T16e as published compares matched-star
|ΔRV| to whole-field |ΔRV|. My initial audit showed this comparison is
dominated by the disk's chemodynamical [Fe/H]–RV correlation: 4D-chemistry-
matched random pairs are 4–8 % closer in RV than fully random pairs,
without any cluster information. The published "5.2 % closer, p = 10⁻³⁷"
sits inside this background.

**Stage 2 (placebo cluster control):** When I do the test PROPERLY — compare
matched-to-X stars' |ΔRV vs V_X| against |ΔRV vs V_Y| for a random
*other* cluster Y — only a marginal hint of signal survives:

| metric | value |
|---|---|
| N clusters tested (with stable mean RV, internal σ_RV<20 km/s) | **25** |
| median \|RV_match − V_X\| (true cluster) | 25.9 km/s |
| median \|RV_match − V_Y\| (placebo cluster) | 29.7 km/s |
| ratio true/null | 0.837 (16 % closer to true) |
| fraction of clusters with true < null | 56 % |
| Wilcoxon (true < null) | W=128, p = **0.18** (n.s.) |

The median ratio (0.84) hints at a real signal of comparable size to T16d's
Ba/Fe residual (~10 %), but the per-cluster sign distribution is essentially
50/50 (56 %) and the Wilcoxon test is not significant (p = 0.18). The
sample also drops sharply (25 vs 140 for T16d) because most clusters don't
have a stable enough mean RV to support the test.

**Verdict: T16e's residual beyond background is uncertain.** The published
5 % closer signal IS the chemodynamical background; the placebo-controlled
version shows a similar magnitude in the median but not in the per-cluster
sign statistic. Larger samples or a longer-baseline RV measurement would
be needed to confirm a genuine kinematic-memory signal.

## What this means

The dissolved-member-recovery story now has:
- **T16b** (Mahalanobis chemical enrichment): clean and primary.
- **T16d** (Ba/Fe blind channel): real residual signal of ~10 % beyond
  chemodynamical background (Stage 2 audit confirms).
- **T16e** (RV kinematic): marginal at best; published 5 % is background;
  placebo-controlled version shows a 16 % median hint but Wilcoxon n.s.
  Better-baseline (e.g. Gaia DR4 + ESPRESSO follow-up) RVs needed.

So the three-channel "independent confirmation" reduces to **two clean
channels + one to defer until better data**, not the previous "three
strong independent confirmations."

## Recommended rephrasing

> Original: *"...and residual kinematic coherence (p = 3.7 × 10⁻³⁷)."*

> Revised: *"The radial-velocity channel shows a 5 % closer-RV effect of
> matched stars relative to whole-field controls (Wilcoxon p = 3.7 × 10⁻³⁷)
> that is fully consistent with the disk's [Fe/H]–RV chemodynamical
> background (random GALAH pairs matched on [Fe/H] alone are already
> ~5 % closer in RV). A placebo-controlled test (matched-to-X stars'
> |ΔRV vs V_X| vs |ΔRV vs V_Y| for random other cluster Y) gives a 16 %
> median hint of real signal but does not reach per-cluster significance
> (Wilcoxon p = 0.18, N = 25 clusters with stable mean RVs). A
> genuine kinematic-memory test requires a larger sample (Gaia DR4) or
> dedicated precision-RV monitoring."*

## Files
- `t16e_proper_audit.py` — placebo-controlled re-implementation
- `t16e_proper_audit.csv` — per-cluster true vs null medians

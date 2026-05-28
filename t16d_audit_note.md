# T16d audit — proper placebo test reveals real but smaller residual signal

## Two-stage audit

**Stage 1 (whole-field control):** T16d as published compares matched-star
|ΔBa/Fe| to whole-field |ΔBa/Fe|. My initial audit showed this comparison is
dominated by the disk's chemodynamical background: 4D-chemistry-matched
random pairs are 23 % closer in Ba/Fe than fully random pairs, without any
cluster information. The published "97.2 % of clusters, p = 10⁻⁴¹" largely
reflects this background.

**Stage 2 (placebo cluster control):** When I do the test PROPERLY — compare
matched-to-X stars' |ΔBa/Fe vs B_X| against |ΔBa/Fe vs B_Y| for a random
*other* cluster Y — a genuine residual signal survives:

| metric | value |
|---|---|
| N clusters tested (N≥5 members, C_O_std<0.10, ≥10 chem matches) | **140** |
| median \|Ba_match − B_X\| (true cluster) | 0.118 dex |
| median \|Ba_match − B_Y\| (placebo cluster) | 0.130 dex |
| ratio true/null | **0.898** (10 % closer to true) |
| fraction of clusters with true < null | **68.6 %** |
| Wilcoxon (true < null) | W=3111, p = **7.4 × 10⁻⁵** |

**Verdict: T16d has a real residual signal beyond chemodynamical
background — ~10 % closer Ba/Fe to true cluster vs placebo, in 68.6 % of
clusters, p < 10⁻⁴.** This is much smaller than the published 97 % / 10⁻⁴¹
(which had whole-field as control), but it is a real independent
confirmation of dissolved-member recovery beyond what generic 4D-chemistry
matching predicts.

## Recommended rephrasing

> Original: *"...confirmed by barium proximity (p = 3.6 × 10⁻⁴¹)..."*

> Revised: *"...with barium-channel cross-check: matched field stars'
> Ba/Fe is closer to the true parent-cluster Ba/Fe than to a randomly
> chosen other cluster's Ba/Fe in 69 % of templates (median 10 % closer,
> Wilcoxon p = 7.4 × 10⁻⁵). The published 97 % statistic and 10⁻⁴¹ p-value
> use a whole-field control, which double-counts the disk's [Fe/H]-driven
> chemodynamical background; the placebo-controlled comparison given here
> is the residual real signal."*

## Files
- `t16d_proper_audit.py` — placebo-controlled re-implementation
- `t16d_proper_audit.csv` — per-cluster true vs null medians

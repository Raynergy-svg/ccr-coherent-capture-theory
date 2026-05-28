# T16e audit — "5.2% closer RV" is the chemodynamical background, not a
# birth-cluster kinematic signal

T16e's published claim: chemically-matched field stars share kinematics with
the parent cluster more than random field stars; median |ΔRV|_matched is
**5.2 % closer** than |ΔRV|_random across 253 clusters (Wilcoxon p = 3.7×10⁻³⁷).
Interpreted as "kinematic memory of birth cluster" → independent confirmation
of dissolved-member recovery.

**Verdict: this 5.2 % effect is consistent with the disk's underlying
chemodynamical correlation alone. T16e does not isolate a birth-cluster
kinematic signal beyond that background.**

## The test

In random GALAH FGK field pairs (no cluster involved), what does the
median |ΔRV| ratio look like as a function of chemistry-matching tightness?

| pair selection | N pairs (of 200,000) | median \|ΔRV\| (km/s) | ratio vs unmatched |
|---|---|---|---|
| unmatched (random) | 200,000 | 29.2 | 1.000 |
| matched on [Fe/H] only (\|Δ\|<0.05 dex) | 25,406 | 27.9 | **0.953** |
| matched on [Fe/H] only (\|Δ\|<0.03 dex) | 15,170 | 27.9 | 0.955 |
| matched on [Fe/H] only (\|Δ\|<0.02 dex) | 10,104 | 27.9 | 0.955 |
| **matched on [Fe/H]+Mg/Fe+Si/Fe** (T16e-like tolerances) | 13,398 | 26.9 | **0.917** |

[Fe/H] matching alone gives a 4.5–4.7 % closer-RV effect (ratio 0.95).
T16e's published 5.2 % closer (ratio 0.948) sits inside this background.
Adding Mg/Fe and Si/Fe (T16e's fuller chemistry match) gives 8.3 % closer
without any cluster involvement — *more* than T16e's published effect.

## Why this happens — the chemodynamics of the disk

The Milky Way thin disk has a radial [Fe/H] gradient (~−0.05 dex/kpc) and a
circular-velocity gradient (the local shear ~10–15 km/s per kpc). Stars at
similar [Fe/H] therefore tend to live at similar Galactic radii and share
similar circular-rotation velocities, giving similar heliocentric RVs in
the local volume. This produces a few-percent closer-RV signal for any
[Fe/H]-matched stellar pair, no birth-cluster information required.

For dissolved-member recovery to claim a *real* kinematic signal beyond
this background, the test needs to compare matched stars to a control
population drawn from the **same chemistry distribution** rather than from
the whole field. Without that control, you cannot distinguish "this star
remembers its birth cluster's RV" from "this star sits at the same
Galactic radius as the cluster."

## What this changes

T16e cannot be claimed as an *independent* confirmation of dissolved-
member recovery. It's measuring the disk's chemo-kinematic background,
which is real but not the same thing.

What's left of the "three-channel dissolved-member confirmation":
- **T16b** (Mahalanobis chemical excess, E~2×): real — pure chemistry vs
  MC random-center null, no kinematic confound.
- **T16d** (Ba/Fe blind cross-check): probably OK because Ba/Fe correlates
  with R_gal less strongly than [Fe/H] does; worth a similar background
  check before publishing as independent confirmation.
- **T16e** (RV kinematic): chemo-dynamical background, NOT a clean
  independent signal.

## Recommended rephrasing

> Original: *"...confirmed by barium proximity (p = 3.6 × 10⁻⁴¹) and
> residual kinematic coherence (p = 3.7 × 10⁻³⁷)."*

> Revised: *"...with barium-proximity cross-check (p = 3.6 × 10⁻⁴¹). The
> apparent 5.2 % residual closer RV of chemically-matched field stars
> relative to random field stars (Wilcoxon p = 3.7 × 10⁻³⁷) is consistent
> with the disk's underlying chemodynamical [Fe/H]–RV correlation alone
> (random GALAH pair test: [Fe/H]-matched pairs are ~5 % closer; full
> 3-element-matched pairs are ~8 % closer, without any cluster
> involvement). The kinematic test as currently implemented does not
> isolate a birth-cluster signal beyond this background and should not be
> claimed as an independent confirmation; redoing the test with an
> [Fe/H]-matched control population is needed to recover any genuine
> kinematic memory signal."*

## What this does NOT change

T16b's chemical-enrichment claim (E~2× via Mahalanobis vs MC null) is
intact. The dissolved-member recovery as a whole still has one solid
independent leg (T16b) plus the Ba/Fe cross-check (T16d, not separately
audited here).

## Files
- (audit done inline; reproduction code in conversation log)

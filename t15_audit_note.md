# T15 audit (brief) — partial correlations survive, Fisher OR is sample-sensitive

T15 published: cluster-mean abundance correlations (e.g. C/O vs Mg/Fe ρ=+0.58)
and coherent-coherent link (79 % of C/O-coherent clusters are also Mg/Fe-
coherent, Fisher OR=4.69, p=2.7×10⁻³).

## What I found on my (different) sample selection

Using cluster-mean abundances with N≥5 members (593 clusters; T15 used a
stricter 81-cluster sample):

| pair | raw ρ | [Fe/H]-controlled partial ρ |
|---|---|---|
| C/O vs Mg/Fe | −0.19 | **+0.19**, p=2×10⁻⁶ |
| C/O vs Si/Fe | −0.10 | +0.26, p=1.5×10⁻¹⁰ |
| C/O vs Al/Fe | +0.22 | +0.17, p=4×10⁻⁵ |
| Mg/Fe vs Si/Fe | +0.55 | +0.48, p=1.5×10⁻³⁵ |
| Mg/Fe vs Al/Fe | +0.31 | +0.46, p=5×10⁻³² |

Multi-element partial correlations survive [Fe/H] control and remain
significant — the "multi-channel coherence" claim is qualitatively real.

## Where T15 is sensitive

The categorical Fisher-OR claim depends on which clusters are selected:

| metric | T15 (N=81) | my reproduction (N=593) |
|---|---|---|
| % of C/O-coherent that are Mg/Fe-coherent | 79 % (31/39) | 29 % (5/17) |
| Fisher OR | 4.69 | 2.83 |
| Fisher p | 2.7×10⁻³ | 6.3×10⁻² (n.s.) |
| permutation rank | (not reported) | observed at 6th pctile of null |

This is a real disagreement, not just bigger sample noise: my 593-cluster
sample gives a substantially weaker effect. The 81-cluster T15 sample must
be applying a stricter pre-cut (probably C_O_std measurement quality +
multi-element measurement availability) that selects clusters where the
coherence link happens to be strong. **The OR=4.69 claim is sample-
selection-dependent and not a generic property.**

## Recommended rephrasing

> Original: *"Multi-element fingerprint: C/O predicts Mg/Fe (Fisher OR = 4.69,
> p = 2.7 × 10⁻³)."*

> Revised: *"Cluster-mean abundances across dimensions are correlated after
> removing the Galactic [Fe/H] gradient (partial Spearman ρ = 0.17–0.48
> across element pairs, p < 10⁻⁴ in each case, N = 593 clusters with N≥5
> members). A categorical 'coherent-coherent' analysis on a stricter
> sub-sample yields Fisher OR ≈ 3–5 with significance dependent on cluster
> selection."*

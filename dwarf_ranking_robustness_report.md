# 8D dwarf-ranking robustness — Phase 1 stress-test

Goal: before treating HD 183193 (or any specific star) as a defensible
"top dwarf," check whether the 8D scorer's ranking survives reasonable
perturbations to its weights, scoring widths, dim subsets, and the GALAH
abundance errors themselves.

## Headline

**The general "top tier of nearby dwarfs" is robust; specific within-tier
ranks are noise-limited at GALAH precision.** HD 183193 is consistently in
the top tier, but its precise rank is highly variable.

| test | HD 183193 rank (in 13,244 nearby dwarfs) | top-10 Jaccard vs reference |
|---|---|---|
| V0 reference (8D, original weights) | 22 | 1.00 |
| V1 uniform weights | 14 | 0.67 |
| V2 2× weight on [Fe/H] | 13 | 0.82 |
| V3 2× weight on alpha (Mg, Si) | 19 | 0.82 |
| V4 drop volatile | **4** | 0.18 |
| V5 drop age | 24 | 0.82 |
| V6 drop volatile + age (6D chemistry only) | **5** | 0.11 |
| V7 widths +25% | 22 | 1.00 |
| V8 widths −25% | 22 | 1.00 |
| **V9 Monte Carlo over GALAH errors (200 draws)** | **median 45 (5–95 pct: 4–276)** | — |

Spearman ρ vs V0 is ≥0.95 for every weight/width variant — the overall
ranking is qualitatively stable. But the V9 Monte Carlo over realistic
GALAH abundance errors shows HD 183193's rank ranges from **4 to 276**
across 200 draws, with **only 16% of draws placing it in the top-10**
(6% in top-5, 0% as rank 1). The "rank 22" is at the precision-limit noise
floor.

## The real top tier

When I tally how often each star appears in the MC top-10:

| Gaia DR3 | SIMBAD | frequency in MC top-10 |
|---|---|---|
| 5629444975252369024 | **CD−30 7056** (Pyxis, 100 pc, G=9.2) | **40 %** |
| 3680522166465341056 | **BD−03 3321** (Virgo, 168 pc, G=10.3) | **40 %** |
| 6728919696365591040 | TYC 7915-779-1 (Cor.Aus., 151 pc) | 28 % |
| 5920769716418475776 | — | 27 % |
| 2634966799782787584 | HD 217340 (Aquarius, 130 pc) | 24 % |
| 6756649036022752640 | TYC 7413-1369-1 (Sgr, 147 pc) | 24 % |
| 5782016541618200320 | — | 22 % |
| 3787168403447234688 | BD−02 3362 (Leo, 154 pc) | 20 % |
| 5371291464801734912 | CD−47 7291 (Cen, 192 pc) | 18 % |
| 5479947952629976576 | CD−60 1593 (Car, 151 pc) | 18 % |
| 6771181697822279936 | **HD 183193** (Sgr, 75 pc, G=8.78) | **16 %** |

**Two stars stand out as MC-robust top picks:** CD−30 7056 and BD−03 3321
(both at ~40 % top-10 frequency). HD 183193 is in the top-tier crowd but
not a measurement-robust standout — it's tied for ~11th place in robustness
rank.

## What's driving the instability

Drop-volatile (V4) and drop-volatile+age (V6) **swing HD 183193's rank from
22 up to 4 and 5** and change the top-10 dramatically (Jaccard 0.18 and 0.11
respectively). Meaning: the volatile (Ba/Fe) and age dimensions are doing
most of the *differentiation* between stars at the top of the ranking — and
since both saturate near 1.0 for most solar-twin candidates, small
perturbations there flip the order.

This is a **scorer-design weakness**: most of the top tier scores ~1.0 on
several dims (C/O implicit, age in 2–8 Gyr, Mg/Si near-solar). The composite
is then determined by which star happens to have, e.g., Ba/Fe closest to
+0.05 vs −0.04. That's a 0.09 dex difference, which is smaller than the
typical GALAH Ba/Fe error (0.03–0.05 dex). Hence the MC noise dominates.

## What this means

1. **The general claim** — "the 8D scorer identifies ~30 actionable nearby
   dwarfs with chemistry equal to or better than the best subgiant pick"
   — is **robust**. The top-tier set is broadly stable.

2. **The specific "HD 183193 = #1" claim** — is **not robust**. HD 183193 is
   in the top tier but its precise rank is precision-limited. The more
   defensible specific picks at GALAH precision are CD−30 7056 and
   BD−03 3321 (40 % MC top-10 frequency each).

3. **HD 183193's actual appeal is operational, not chemical**: brightest
   (G = 8.78), closest (75 pc), clean astrometric solution, no planets, no
   prior spectroscopy. For an observation proposal, "brightest closest
   member of a precision-tied top tier" is honest framing.

4. **GALAH precision is the bottleneck again.** Same Precision-Wall theme.
   Distinguishing within the top tier needs ~0.01 dex abundance precision —
   4MOST territory — or instrument-specific spectroscopy that beats GALAH's
   per-star errors (ESPRESSO/HARPS-N for the brightest targets).

5. **The volatile and age dims need scrutiny.** They drive ranking
   instability because they're saturated for many solar twins. Either
   sharpen the scoring functions or accept that ranking within the top
   tier is meaningless and report the set, not the rank.

## Files
- `dwarf_ranking_robustness.py` — reproducible script

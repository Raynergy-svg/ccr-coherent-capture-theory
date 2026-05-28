# Phase 2b: gap closure under thin-disk + [Fe/H] matching

Follow-up on the Phase 2 finding that the 21 % nearby-dwarf vs 44 %
subgiant excellent-rate gap is α-driven. The simple test: imposing a
thin-disk α-cut and matching on [Fe/H], does the gap close?

## Thin-disk α-cut (Hayden+2015-style)

| α-cut | subgiants excellent | dwarfs excellent | gap |
|---|---|---|---|
| no cut | 44.1 % (50,717) | 21.3 % (13,244) | **22.8** pp |
| \|Mg/Fe\|, \|Si/Fe\| < 0.20 | 48.2 % | 33.9 % | 14.3 pp |
| \|Mg/Fe\|, \|Si/Fe\| < 0.15 | 50.9 % | 43.4 % | **7.5** pp |
| **\|Mg/Fe\|, \|Si/Fe\| < 0.10** | **58.0 %** | **56.4 %** | **1.6** pp |
| \|Mg/Fe\|, \|Si/Fe\| < 0.05 | 72.1 % | 68.3 % | 3.8 pp |

Cutting at α-width ≈ 0.10 dex collapses the gap to ~2 pp. The α-distribution
mismatch explains nearly all of the raw 22.8 % gap **on aggregate**.

## BUT — at fixed [Fe/H] AND thin-disk membership, a real residual remains

Within [Mg/Fe] < 0.15 thin-disk cut, binned by [Fe/H]:

| [Fe/H] bin | subgiants excellent | dwarfs excellent | residual gap |
|---|---|---|---|
| [−0.2, −0.1) | 65.7 % (6922) | 21.8 % (1036) | **+43.9 pp** |
| [−0.1,  0.0) | 82.3 % (7720) | 46.1 % (2147) | +36.2 pp |
| [ 0.0, +0.1) | 83.0 % (7994) | 60.5 % (1693) | +22.5 pp |
| [+0.1, +0.2) | 64.1 % (5237) | 50.5 % (1036) | +13.6 pp |

The aggregate closure is real, but only because the α-cut shifts the dwarf
[Fe/H] distribution toward higher metallicity (where subgiants and dwarfs
both score well). At fixed [Fe/H], a 20–40 pp gap persists. So the
mechanism is:

- **alpha distribution** (Phase 2 finding) — drives ~80 % of per-dim damage
- **[Fe/H] distribution interaction** — α-cut pushes both populations to a
  higher-[Fe/H] sweet spot, mechanically closing the aggregate gap
- **Residual at fixed [Fe/H]** — 20–40 pp persists even matched on
  thin-disk α and [Fe/H], indicating a third factor

## Hypothesis for the residual

Two candidates worth checking separately:

1. **Age** — Phase 2 (d) showed dwarfs at older ages match subgiants. The
   nearby thin-disk dwarf sample is younger on average; matching on age
   should close more of the residual.
2. **Abundance-flag selection bias** — both populations require valid
   unflagged [Mg/Fe], [Si/Fe], [Fe/H]. Within the dwarf population, those
   with all-clean flags may be a non-random sample (e.g. specific
   Teff/log g regions where GALAH's α retrieval works best) — and the
   selection geometry may differ from subgiants.

## What this means for the paper framing

The "21 % vs 44 %" headline is even more nuanced than Phase 2 indicated:

> *Nearby thin-disk dwarfs (|Mg/Fe|<0.10) match subgiants in excellent
> chemistry rate (~57 % vs ~58 %). The aggregate 21 vs 44 % gap therefore
> reflects (a) α-enhanced thick-disk contamination in the local volume, and
> (b) interleaved age and selection effects that produce residual
> per-[Fe/H]-bin differences. The 'rate of solar-like FGK chemistry' is
> not a single number; it depends on which sub-population is targeted.*

For proposal/target framing: the rate among **nearby thin-disk dwarfs** is
**~57 %** — same as the subgiant rate. The 21 % figure is more strictly
"nearby dwarfs including the α-enhanced tail."

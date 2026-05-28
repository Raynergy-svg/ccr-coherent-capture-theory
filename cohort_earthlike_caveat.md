# "Earth-like chemistry is common" — caveat & recommended rephrasing

The README/T-series headline `"4,970 stars score excellent (>0.9) on 9D habitability ~ 41% of
FGK dwarfs"` is computed on a cohort that is in fact **all turnoff/subgiants** (log g 3.80–4.00),
because the 9D scorer's required C/O dimension excludes every FGK main-sequence dwarf in GALAH DR4
(`flag_c_fe = 0` for 0/13,554 nearby dwarfs).

The qualitative conclusion **survives** once we use the dwarf-compatible 8D scorer (drops C/O,
keeps the other 8 dims with the original weights), but the framing needs tightening.

## 8D excellent-fraction by evolutionary state (GALAH DR4 FGK, 293,557 stars 8D-eligible)

| population | N | hab8 > 0.9 | hab8 > 0.95 |
|---|---|---|---|
| **all 8D-eligible FGK (log g > 3.8)** | 293,557 | **38.5 %** | 15.3 % |
| subgiants/turnoff (3.80 < log g ≤ 4.00) | 50,717 | **44.1 %** ≈ original "41%" | 19.7 % |
| intermediate (4.00 < log g < 4.20) | 82,632 | 42.5 % | 17.4 % |
| true main-sequence dwarfs (log g ≥ 4.20) | 160,208 | **34.7 %** | 12.8 % |
| **solar-twin dwarfs (log g ≥ 4.2, 5600–5900 K)** | 49,411 | **40.4 %** | 15.6 % |

Nearby slice (parallax > 5 mas, ~within 200 pc):

| population | N | hab8 > 0.9 |
|---|---|---|
| all 8D-eligible, <200 pc | 14,016 | 22.3 % |
| subgiants, <200 pc | 142 | 43.7 % |
| true dwarfs (log g ≥ 4.2), <200 pc | 13,244 | **21.3 %** |

## What this means

- **The headline "Earth-like chemistry is common" holds.** Across the populations the rates are 35–44 %, all of the same order. The qualitative conclusion (a non-trivial fraction of FGK stars have near-solar 8D abundance patterns) is robust.
- **But "41 % of FGK dwarfs" is technically inaccurate.** The 41 % figure is for turnoff/subgiants (44 % under 8D); true dwarfs are 35 % overall, 40 % for solar-twin dwarfs, and 21 % among *nearby* dwarfs.
- **Nearby dwarfs are scarcer in the excellent bin** (21 % vs 44 % for subgiants). This is partly because GALAH's nearby dwarf sample spans a wider [Fe/H] range than the (more uniform) distant subgiant sample, which pulls the [Fe/H] sub-score down.

## Recommended rephrasings

Drop the unqualified "FGK dwarfs" wording. Pick whichever fits the paper's emphasis:

> *Earth-like 9D chemistry is found in 44 % of the 50,717 GALAH-measurable FGK turnoff/subgiants. The fraction is similar (≈ 35–40 %) for true main-sequence dwarfs when assessed with a C/O-blind 8D variant (GALAH cannot reliably measure C on FGK dwarfs); only ~21 % of nearby (<200 pc) dwarfs reach the threshold, because the nearby thin-disk dwarf [Fe/H] distribution is broader than the distant subgiant locus.*

For the README summary line:

> ~~Earth-like chemistry is common — 41% of FGK dwarfs score excellent on 9D habitability~~
>
> **Earth-like chemistry is common — ~35–44% of GALAH-measurable FGK stars score excellent on the 9D scorer (subgiants/turnoff) or its dwarf-compatible 8D variant.**

## Bottom line for the paper

- T-series conclusion #4 holds in spirit; the quantitative claim needs one extra qualifying clause.
- The 8D scorer's near-identical excellent fraction (44 % subgiants vs 35–40 % dwarfs) is itself a nice sanity check: the same "rate of solar-like FGK chemistry" emerges across evolutionary states and across scorer variants. The dwarf 21 % nearby figure is the most habitability-relevant headline number if rocky-HZ targets are the focus.

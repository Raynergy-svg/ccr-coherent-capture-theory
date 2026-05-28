# Hidden gems: the dwarfs the 9D scorer never saw

## Headline

The committed habitability scorer's 4,970-excellent / 18-actionable / "HD 28888 = #1" pipeline
is **structurally blind to FGK main-sequence dwarfs**: GALAH DR4's `flag_c_fe = 0` for **zero of
13,554 nearby (<200 pc) FGK dwarfs**, so requiring unflagged C/O excludes every dwarf and leaves
only subgiants. The 12,234 "FGK dwarfs" in `habitability_v2_targets.csv` are *all* turnoff/subgiants
(log g 3.80–4.00, never higher).

Re-running the scorer with C/O dropped (the only dim that fails on dwarfs) — same 8 remaining
implemented dims — and applying the actionable filters gives **32 nearby true dwarfs
(log g ≥ 4.2, dist < 200 pc, RUWE < 1.4, NSS = 0, G < 12, age 2–8 Gyr, no known planets) — every
one of which outscores HD 28888 by 0.018–0.028.** 20 of the 32 are northern (Dec > −30°).

**HD 28888 remains the best *subgiant* within 200 pc. It is not the best *dwarf*.**

## Why C/O is the choke-point

| element | unflagged & valid in 13,554 nearby FGK dwarfs |
|---|---|
| Mg | 99.9% |
| Si | 99.4% |
| Fe | 98.0% |
| O | 91.6% |
| Ca | 99.9% |
| Al | 99.4% |
| Ba | 99.9% |
| age | 99.9% |
| **C** | **0.0%** |

`flag_c_fe ≤ 1` (warning) survives for **0 of 13,554** dwarfs. The CH-band C measurement is
fundamentally unreliable for FGK dwarfs in GALAH DR4. This is a precision-wall issue, not a
pipeline bug — but it has hidden a population of dwarf targets that the 9D scorer couldn't see.

## The 8D scorer (drops C/O)

Weights from `habitability_v2.py`, retained as-is, C/O omitted:

`[Mg/Si]=1.5, [Fe/H]=1.5, [Mg/Fe]=1.0, [Si/Fe]=1.0, [Ca/Fe]=0.5, [Al/Fe]=0.5, Volatile([Ba/Fe])=1.0, Age=0.75`

HD 28888 under the 8D scorer ≈ **0.9693** (vs 0.9728 in 9D — the C/O dim saturates at 1.0 for HD 28888
so the small drop is from re-weighting). Its rank among the 32 nearby dwarfs scored: **33rd** (it is below all of them).

## Top 10 nearby dwarf habitability targets (8D, actionable)

| rk | Gaia DR3 | SIMBAD name | hab8 | dist (pc) | G | Teff | log g | age | Dec | constellation | nbref |
|--|--|--|--|--|--|--|--|--|--|--|--|
| 1 | 3680522166465341056 | BD−03 3321 | 0.9969 | 168 | 10.3 | 5954 | 4.33 | 5.7 | −4.1° | Virgo | 5 |
| 2 | 5629444975252369024 | CD−30 7056 | 0.9954 | 100 | 9.2 | 5876 | 4.34 | 6.7 | −30.6° | Pyxis | 6 |
| 3 | 6756649036022752640 | TYC 7413-1369-1 | 0.9949 | 147 | 10.6 | 5757 | 4.47 | 3.8 | −33.3° | Sagittarius | 0 |
| 4 | 4299049512104000128 | TYC 1071-934-1 | 0.9944 | 194 | 11.4 | 5666 | 4.49 | 4.7 | +8.4° | Aquila | 0 |
| 5 | 3787168403447234688 | BD−02 3362 | 0.9943 | 154 | 10.3 | 5881 | 4.34 | 7.3 | −3.5° | Leo | 2 |
| 6 | 6765392391143702400 | TYC 6897-1616-1 | 0.9939 | 198 | 11.0 | 5894 | 4.38 | 5.1 | −27.2° | Sagittarius | 3 |
| 7 | 4660360749704988672 | **HD 271308** | 0.9936 | 184 | 9.9 | 6311 | 4.25 | 3.4 | −66.0° | Dorado | 14 |
| 8 | 6728919696365591040 | TYC 7915-779-1 | 0.9934 | 151 | 10.3 | 5874 | 4.37 | 6.0 | −39.1° | Corona Australis | 3 |
| 9 | 5479947952629976576 | CD−60 1593 | 0.9934 | 151 | 10.9 | 5509 | 4.47 | 6.9 | −60.3° | Carina | 8 |
| 10 | 5820151934211076736 | TYC 9263-701-1 | 0.9925 | 157 | 10.5 | 5863 | 4.39 | 5.0 | −69.4° | Triangulum Australe | 1 |

(Full 32 in `real_dwarf_targets.csv`.)

### Standout single target — **HD 183193**
Gaia DR3 6771181697822279936, Sagittarius (Dec −24.3°), **75 pc, G = 8.78**, Teff 5874 K, log g 4.37,
age 6.4 Gyr (≈ solar), hab8 = 0.991, 8 SIMBAD refs, no known planets, RUWE 1.07, single.
The **brightest, closest, optimally-aged true dwarf** in the actionable set — and the strongest single
contender for "real #1." It is closer than HD 28888 (75 vs 100 pc), of comparable brightness, but
**main-sequence** (log g 4.37 vs HD 28888's 3.94), so its habitable zone is at ~1 AU with year-scale
periods rather than the wider 1.8–3.2 AU / multi-year subgiant HZ.

## Implications for the paper

1. **The 9D scorer's headline ranking is dwarf-blind.** Any claim involving "best habitability target
   from GALAH FGK" must either (a) state explicitly that it ranks subgiants only because C/O is GALAH-
   unreliable on dwarfs, or (b) use a dwarf-compatible reduced scorer.

2. **HD 28888 keeps a defensible #1 status, but only as "best subgiant within 200 pc."** Under the
   dwarf-compatible 8D scorer, it ranks 33rd among 32 nearby true dwarfs. The paper's "#1 actionable"
   sentence should be qualified.

3. **A dual-ranking proposal target list** is the honest fix: top-N subgiants (9D) + top-N dwarfs (8D),
   with the structural caveat that dwarf C/O is not measured. The combined ladder gives proposers a
   chemistry-ranked menu across evolutionary state.

4. **The reduced 8D scorer is publishable on its own merits** as the GALAH-DR4-compatible habitability
   metric for FGK dwarfs. It uses all the elements GALAH measures cleanly on main-sequence stars; the
   C/O dimension is reserved for spectra where it is meaningful (cool dwarfs in better surveys, or
   subgiants in GALAH).

5. **This is consistent with — and sharpens — the Precision Wall thesis.** It's not just individual
   co-natal tracing that is GALAH-limited; even population-level dwarf habitability scoring needs
   either a reduced dim-set or higher-precision C measurements (4MOST, future surveys).

## Artifacts
- `dwarf_rescore_8d.py` — reduced-scorer script
- `real_dwarf_targets.csv` — 32 actionable nearby dwarfs with full Gaia + SIMBAD fields
- `dwarf8_top40_nearby.csv` — top 40 nearby dwarfs by 8D hab_score (pre-actionable filter)
- `dwarf_all_top1000.csv` — top 1000 dwarfs by 9D hab_score (for reference; mostly distant)

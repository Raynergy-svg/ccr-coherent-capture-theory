# HD 28888 — Follow-up Analysis (4 deliverables)

Gaia DR3 3312653647717790720. Inputs are the live-queried values; computations in [`hd28888_analysis.py`](hd28888_analysis.py).

---

## 1. 9D scorer rank vs the scored cohort

Re-scoring from the live numbers exactly reproduces the stored value: **hab_score = 0.9728**.

| Dimension | Sub-score | Limiting? |
|---|---|---|
| C/O (Teff-corr 0.284) | 1.0000 | maxed (flat 0.15–0.65) |
| Mg/Si (1.034) | 0.9993 | |
| [Fe/H] (+0.130) | 0.9109 | **main drag** (slightly metal-rich) |
| [Mg/Fe] | 0.9909 | |
| [Si/Fe] | 0.9993 | |
| [Ca/Fe] | 0.9776 | |
| [Al/Fe] | 0.9714 | |
| Volatile (Ba/Fe −0.04) | 0.9374 | second drag (Ba below +0.05 optimum) |
| Age (6.3 Gyr) | 1.0000 | maxed |

**Rank: 692 of 12,234 → top 5.7% (94.3rd percentile).**

- **Not top 1%** (cutoff 0.9874) and **not top 0.1%** (cutoff 0.9946). Cohort max = 0.9987.
- 4,970 stars (**40.6%**) score >0.9 — the "excellent" bar is broad, exactly as the README's Precision-Wall section concedes. A 0.97 chemistry score is very good but not rare.

> **Important framing:** HD 28888 is #1 on the *actionable* list because of the **filter stack** (RUWE<1.4 single + <200 pc + 2–8 Gyr + **zero RV coverage** + **no known planets**), not because it has the highest chemistry score. ~691 stars are chemically "better," but they're farther, already observed, or in multiples. The pitch is "excellent-and-unobserved," not "chemically the best in the galaxy." Keep that honest in the proposal.

---

## 2. Teutsch_80 tag — NOT a real birth-cluster association

This is the load-bearing result, and it's a clean negative. The `cct_cluster="Teutsch_80"` tag comes from the scorer's **chemical nearest-template match** in (C/O, Mg/Fe, Si/Fe, [Fe/H]) space (`habitability_v2.py:246`), with no spatial or kinematic constraint.

Catalog cross-check (Hunt & Reffert 2023 `J/A+A/673/A114`; Cantat-Gaudin 2020 `J/A+A/640/A1`, via VizieR TAP):

| | HD 28888 | Teutsch 80 | Verdict |
|---|---|---|---|
| Position | RA 68.4°, Dec +16.0° | RA 223.4°, Dec −60.5° | **132° apart** (opposite sky) |
| Distance | 100 pc | **2.4–2.6 kpc** | **~25× farther** |
| Proper motion | (+65, −56) mas/yr | (−5.6, −3.6) mas/yr | Δ = 88 mas/yr |
| Radial velocity | +53.7 km/s | −14.6 km/s (members −42 to −4) | opposite sign |
| **Age** | **6.3 Gyr** | **0.09–0.22 Gyr** | **28–69× too old** |

Any single row excludes membership; together they're overwhelming. A 6.3 Gyr star cannot belong to a 0.1 Gyr cluster, full stop. The chemical match is the **71,000-background-matches-per-star** regime the paper already documents — a chemical neighbor, not a birth sibling.

> ⚠️ **Bug found:** the project's `t9_matched_stars.csv` lists `dist_cl = 0.349` for Teutsch 80 "kpc". That 0.349 is actually the **parallax in mas** (CG2020 plx = 0.349 mas → 2.6 kpc). The distance column for cluster centers is mislabeled/units-swapped across the t9 pipeline — worth auditing, since any analysis using `dist_cl` as kpc is wrong by the inverse-parallax factor.

**Bottom line:** no smoking gun here. If you want a genuine co-natal claim, it needs a real kinematic+chemical member of a *surviving, nearby, age-matched* cluster — and the paper's own conclusion is that GALAH precision can't deliver that at the individual-star level (it's the 4MOST-era test).

---

## 3. Subgiant caveat — proposal language + HZ geometry

HD 28888 has log g ≈ 3.94 and L ≈ 3.55 L☉ (R ≈ 2 R☉): it has **evolved off the main sequence onto the subgiant branch**. The habitable zone has migrated outward accordingly.

| HZ (Kopparapu 2013 flux scaling, Teff 5734 K, L 3.55 L☉) | Inner | Outer | Periods (M = 1.15 M☉) |
|---|---|---|---|
| Conservative (runaway → max greenhouse) | 1.79 AU | 3.16 AU | **2.2 – 5.2 yr** |
| Optimistic (recent Venus → early Mars) | 1.41 AU | 3.33 AU | 1.6 – 5.7 yr |

**Draft proposal text:**

> *HD 28888 has evolved onto the subgiant branch (log g ≈ 3.9, L ≈ 3.6 L☉), so its present-day habitable zone has migrated outward to ≈1.8–3.2 AU. A radial-velocity campaign should therefore target longer-period companions (P ≈ 2–5 yr, K ≈ a few m/s for sub-Saturn masses) and budget a multi-year baseline; short-cadence searches tuned to ≤1 yr orbits would miss the temperate zone entirely. The star's brightness (G = 8.2), low v sin i (5.5 km/s), and complete absence of prior RV monitoring make it an efficient, uncontested target for such a baseline.*

Note this caveat applies to the **entire** actionable list — every one of the 18 sits at log g ≈ 3.8–4.0 (the scorer's FGK cut is `logg > 3.8`). Worth a one-line disclosure that the sample is biased toward turnoff/subgiant stars, where GALAH ages are most reliable.

---

## 4. Dredge-up correction → birth C/O

The surface shows the classic first-dredge-up CN signature: **[N/Fe] = +0.38 (high), [C/Fe] = −0.07 (low)**, O essentially unchanged. Backing out the natal carbon by conserving C+N (CN cycle converts C→N) and assuming scaled-solar birth nitrogen:

| Quantity | Value |
|---|---|
| Surface C/O (number ratio) | 0.308 |
| **Birth C/O — full CN equilibrium** | **0.433** |
| Birth C/O — 50% partial mixing | 0.371 |
| Birth [C/Fe] (full) | +0.08 (vs −0.07 observed) |

The star is at the *start* of first dredge-up (log g 3.94; FDU completes nearer log g 3.3–3.5), so the realistic birth value is in the **0.37–0.43** range — i.e., the natal C/O was **higher and closer to solar (0.55)** than the depleted surface implies. The user's intuition is correct: it bumps up.

**Effect on the score: none.** C/O 0.31→0.43 stays inside the scorer's flat 0.15–0.65 plateau (s_CO = 1.0 either way) and far below the C/O = 0.8 carbon-planet line. So the correction doesn't raise the 0.9728, but it *does* strengthen the qualitative story: HD 28888's birth disk was silicate-dominated with near-solar C/O — good for rocky, water-bearing planet formation.

---

## TL;DR

1. **Rank top 5.7%, not top 1%.** Genuinely excellent chemistry; #1 status is driven by the observability filters, not the score.
2. **Teutsch_80 is a chemical coincidence, not a birth cluster** — 132° away, 25× more distant, 30–70× too old. Not a paper; it's the Precision Wall in action. (Also flagged a units bug in `dist_cl`.)
3. **Subgiant → HZ at 1.8–3.2 AU, periods 2–5 yr.** Proposal text drafted; caveat applies to all 18 targets.
4. **Birth C/O ≈ 0.37–0.43** (up from 0.31 surface), closer to solar. Doesn't move the score but supports the rocky-planet narrative.

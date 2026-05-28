# CCT Audit + Dwarf-Target Investigation — Consolidated Session Report

**Branch:** `claude/gaia-galah-hd28888-R5VZz` (all artifacts pushed)
**Scope:** Full live-archive query for HD 28888 + 9D habitability scorer audit + complete T-series methodological audit + new dwarf-target finding (HD 183193 et al.)

---

## Executive summary

Started as a single-target archive query (HD 28888) and grew into a comprehensive audit pass of the Coherent Capture Theory (CCT) T-series. Found and fixed **five distinct structural issues** in the published pipeline, surfaced **one major new positive result** (the 8D scorer reveals 32 actionable nearby dwarf habitability targets the original 9D scorer was structurally blind to), and produced ~30 reproducible scripts + audit reports covering every headline T-test except T17 (which the paper itself documents as having survivorship bias).

The CCT thesis ("chemical tagging is the right channel; precision is the bottleneck; 4MOST will deliver") is **strengthened**, not weakened, by the audit — every issue found maps back to the paper's own Precision Wall framing. The chemical fingerprint is real (T9, T18, T16b, T16d residual). But many specific *quantitative* claims need rephrasing.

---

## Five structural issues found and addressed

1. **`dist_cl` units bug** — In `t9_cluster_coherence.py`, the distance column selector matched `plx` instead of `DistPc`, so `dist_cl` was the cluster parallax in mas mislabeled as kpc. This inverted near/far ordering and propagated through T10, T19, T20c. Fixed in source; CSVs patched (1/plx → kpc); affected analyses re-run with `*_distfix` outputs preserving canonical state.

2. **Subgiant-only cohort artifact** — `habitability_v2_targets.csv` (the 12,234 "FGK dwarfs") is actually all log g 3.80–4.00 turnoff/subgiants. Cause: GALAH DR4 `flag_c_fe = 0` for **0 of 13,554 nearby FGK dwarfs**. The 9D scorer's C/O requirement excludes every dwarf. Solution: a reduced 8D scorer (drops C/O, keeps the other 8 implemented dims with original weights) that is GALAH-compatible for dwarfs.

3. **T20c HD 163560 = bug artifact** — Both the cluster "0.56 kpc" distance and HD 163560's "1.2% parallax match" were `dist_cl`-bug artifacts (actual cluster distance 1.65 kpc; HD 163560 is a 547 pc foreground star). The TESS asteroseismic elimination (Hon 2021) was chasing the artifact. Paper reframed: HD 163560 rejected at the parallax stage.

4. **Whole-field controls inflate dissolved-recovery signals** (new pattern surfaced by audits) — T10, T16d, T16e compare matched stars to fully-random field controls, double-counting the disk's chemodynamical correlations (4D-chemistry-matched random pairs are 23% closer in [Ba/Fe], 5–8% closer in RV, without any cluster involvement). Proper placebo-controlled tests (match-to-X vs random other-cluster) implemented and reveal smaller but real residual signals.

5. **T16c age-coverage overclaim** — "Permanent over 0–10 Gyr" computed on a sample where max age is 6.31 Gyr and 87% of clusters are <1 Gyr. Power analysis: the test can only rule out τ < 1 Gyr; τ ≥ 5 Gyr is indistinguishable from permanence.

---

## Major new positive result — the 8D dwarf finding

With C/O dropped, the **reduced 8D scorer applied to the full FGK sample** (not the subgiant-only committed cohort) yields **32 actionable nearby true dwarfs** (log g ≥ 4.2, dist < 200 pc, RUWE < 1.4, NSS = 0, G < 12, age 2–8 Gyr, planet-free) that all outscore HD 28888's hab8 = 0.969 by 0.018–0.028. 20 are northern-accessible.

**HD 183193** (Gaia DR3 6771181697822279936; Sagittarius, Dec −24.3°, 75 pc, G = 8.78, T_eff 5874 K, log g 4.37, R ~ 1 R☉, age 6.37 Gyr, [Fe/H] = −0.003, RUWE 1.07, no planets, no precision-RV history) is the brightest + closest + main-sequence pick. Confirmed against APOGEE for 9 of the 32: APOGEE [C/Fe] uniformly near-solar (median −0.09 dex, all flag = 0), all 9 on the s_CO = 1.0 plateau → 8D scorer's "drop C/O" assumption empirically vindicated.

HD 183193 itself is not in APOGEE; an archive sweep confirmed **no prior high-res spectroscopy exists** (zero ESO HARPS/UVES/ESPRESSO/FEROS/X-Shooter spectra; zero entries in PASTEL, Bensby+14, Adibekyan+12, Brewer+16, Hypatia, Delgado-Mena+17; 8 SIMBAD references, all wide surveys + one independent ELT-biosignature target list paper, Hardegree-Ullman+23). The "untouched solar twin" framing is literally accurate.

**Robustness check** (9 scorer variants + 200-draw Monte Carlo over GALAH abundance errors): the top-tier set is stable (Spearman ρ ≥ 0.95 across variants), but specific within-tier ranks are GALAH-precision-limited. HD 183193's rank varies from 4 to 276 in MC, only 16% top-10 frequency. The MC-robust top picks are **CD−30 7056** and **BD−03 3321** (40% top-10 frequency each). HD 183193's appeal is therefore *operational* (brightest/closest/untouched), not chemistry-uniquely-special.

---

## Test-by-test status (the consolidated audit)

| Test | Headline claim | Status after audit | What changed |
|---|---|---|---|
| T9 | Open clusters chemically distinct, KW p<10⁻¹⁰ | **SOLID** but smaller effect than headline | Permutation null: 15σ above. But 24% of KW signal is the [Fe/H] gradient; only 11% of total C/O variance is between-cluster |
| T10 | Spatially independent (Mantel r=−0.010, p=0.60) | **WAS BUG → FIXED** | Raw r=+0.067, p=0.013; partial Mantel controlling [Fe/H] = +0.018, p=0.46. Spatial independence still holds, but now properly attributed |
| T14 | Coherence decays τ=1.29 ± 1.58 Gyr (disk heating) | qualitative real, τ loose | ρ=0.117, p=0.004 reproduces; 1.4% variance explained; bootstrap 90% CI for τ is [0.6, 4.4] Gyr |
| T15 | Multi-element coherence (Fisher OR=4.69, p=2.7e-3) | partial-corrs solid, Fisher OR sample-sensitive | Partial Spearman across element pairs (controlling [Fe/H]): ρ=0.17–0.48, p<10⁻⁴. Fisher OR on broader N=593 drops to 2.83 (p=0.06) |
| T16b | Dissolved-member recovery E~2× | **SOLID** | Mahalanobis vs MC random-center null; chemistry-only; distance-independent |
| T16c | Permanent over 0–10 Gyr | **OVERCLAIM** | Sample max age 6.31 Gyr; 87% <1 Gyr. Power analysis: test rules out τ<1 Gyr but cannot distinguish τ≥5 from permanence |
| T16d | Ba/Fe blind cross-check (97.2%, p=10⁻⁴¹) | **REAL signal, smaller than headline** | Published 97% is whole-field control = 23% chemodynamical background. Proper placebo test (vs random other-cluster Ba/Fe): 10% closer to true cluster, 69% of clusters, Wilcoxon p=7e-5 |
| T16e | RV residual 5.2% closer (p=10⁻³⁷) | **MARGINAL** | Published 5.2% = [Fe/H]-matched chemodynamical background. Proper placebo: 16% median hint but Wilcoxon p=0.18 (n.s.). Deferred to Gaia DR4 / dedicated RV |
| T17 | No decay in surviving clusters | not separately re-audited | Paper documents the survivorship caveat. C_O_std<0.10 template cut excludes 1–2 Gyr clusters at 2× the rate of other age bins — confirms the caveat |
| T18 | α 2× tighter than s-proc (98%, p=10⁻⁹⁸) | **SOLID** ✓ | Error-subtracted intrinsic version: 96.4%, p=10⁻⁹⁷. Cleanest audit win |
| T19 | Outer disc more coherent (ρ=−0.228, p=10⁻⁹) | **WAS BUG → FIXED, weaker** | Post-`dist_cl` fix: ρ=−0.091, p=0.02; partial ρ\|[Fe/H] = −0.147, p=1.1e-5. Direction holds; ~2× weaker |
| T20a/b | Praesepe / NGC 6791 recovery funnel | reproducible, distance-independent | — |
| T20c | NGC 6253 / HD 163560 asteroseismic elimination | **BUGS FIXED** | Distance 0.56 → 1.65 kpc; HD 163560 reframed as parallax-rejected foreground; script adaptive-tolerance gap aligned to paper's fixed tolerance; Gaia query made deterministic via retry |

**Patterns:** (1) age coverage caps at 6 Gyr → no leverage for "0–10 Gyr" claims; (2) [Fe/H] gradient interleaves with cluster-identity signals → partial versions needed; (3) tight-cluster survivorship bias in template selection; (4) whole-field controls inflate dissolved-recovery effect sizes.

---

## Recommended manuscript revisions (ranked by impact)

1. **T10**: rephrase to partial-Mantel-controlling-for-[Fe/H] framing (drafted; conclusion intact, mechanism clarified).
2. **T19**: ρ=−0.228 → ρ=−0.091; partial ρ\|[Fe/H] = −0.147 (drafted).
3. **T16c**: "0–10 Gyr permanent" → "no detectable decay 0–6 Gyr; rules out τ<1 Gyr; cannot distinguish τ≥5 from permanence."
4. **T20c**: distance, funnel, HD 163560 narrative reframed (drafted and applied).
5. **T14**: "τ=1.29 Gyr matches disk heating" → "bootstrap 90% CI τ=[0.6, 4.4] Gyr; consistent with but not uniquely tied to disk-heating window."
6. **T9 framing**: KW p<10⁻¹⁰ headline → acknowledge 11% between-cluster variance and 24% [Fe/H]-gradient component.
7. **T15 framing**: replace Fisher OR (sample-sensitive) with the more robust partial-correlation framing.
8. **T16d**: replace "97.2%, p=10⁻⁴¹" (whole-field control) with placebo-controlled version ("10% closer Ba/Fe, 69% of clusters, p=7×10⁻⁵").
9. **T16e**: drop "p=10⁻³⁷" as independent confirmation; note 5.2% is the chemodynamical background and the placebo-controlled test does not reach per-cluster significance; defer to Gaia DR4.
10. **Cohort scope**: "12,234 FGK dwarfs" → "GALAH-measurable turnoff/subgiants" + introduce the 8D variant for true dwarfs (README already updated).
11. **Actionable target list**: HD 28888 retains "best subgiant within 200 pc"; dual-track with HD 183193 et al. as 8D dwarf picks, *honest framing that within-tier rank is GALAH-precision-limited*.

The "Earth-like chemistry common — 41% of FGK dwarfs" headline survives qualitatively: 35–44% under both 9D-subgiant and 8D-dwarf framings, ~21% for nearby (<200 pc) dwarfs specifically. README rephrased.

---

## Operational deliverables (proposal-ready)

- **HD 28888** archive profile complete, kinematics computed, 9D scorer rank 692/12,234 (top 5.7%); #1 of 18 actionable list driven by the filter stack (single + <200 pc + 2–8 Gyr + zero RV + no planets), not by raw chemistry score.
- **HD 183193** archive profile complete; literally untouched by prior spectroscopy. Single ESPRESSO snapshot (~10–15 min at G=8.78) would deliver first [C/Fe] + full abundance pattern + precision-RV zero-point.
- **HD 28888 vs HD 183193 co-natal test**: clean NO. Ages match (6.31 vs 6.37 Gyr) but [Fe/H] differs by 0.13 dex (4× intra-cluster scatter), L_Z differs 17%, J_R 47%, J_Z 77%, UVW Δ 55 km/s. Two unrelated solar-neighbourhood thin-disk stars.
- **Co-natal pair search across the 32 actionable dwarfs**: clean null. No pair stands out from the bulk 496-pair distance distribution.
- **47 reachable backup targets** (excellent + age 2–8 + single + planet-free, 200–300 pc, G<12.5) — all ESPRESSO-feasible in <20 min, 21 northern; all subgiants (zero dwarfs in the broader excellent set at GALAH precision).
- **MC-robust top dwarf picks** for proposal naming: **CD−30 7056** (Pyxis, 100 pc, G=9.2, 40% MC top-10) and **BD−03 3321** (Virgo, 168 pc, G=10.3, Dec −4°, hab8=0.997, 40% MC top-10).

---

## Open items I did NOT do

- **T17 audit** (paper documents the survivorship caveat already).
- **APOGEE OCCAM independent confirmation** of cluster distinctness (p=10⁻¹⁰⁴) — not re-derived; probably solid since OCCAM is well-controlled.
- **Archive sweep on the other 31 actionable dwarfs** (HD 183193-style) — operational; identifies which are virgin vs pre-studied.
- **Proper placebo audits on T15 Fisher OR** and any other "matched vs random field" tests not covered.
- **Full re-run of T16b on regenerated CSVs** with `dist_cl` fix applied (the headline E=2.03× is distance-independent and survives; the radial sub-stratification would shift).
- **Apply the recommended manuscript rephrasings** to `certan2026_cct.tex` end-to-end (T10, T19, T20c already applied; the rest are drafted in audit reports awaiting your decision).
- **Tighten the 8D scorer's volatile / age dimensions** (they currently saturate for solar twins and drive MC ranking instability).

---

## Files / commits (all on `claude/gaia-galah-hd28888-R5VZz`)

**Audit reports**
`dist_cl_bug_audit.md` · `revisions_T10_T19.md` · `META_AUDIT.md` ·
`t9_audit_report.md` · `t14_audit_report.md` · `t15_audit_note.md` ·
`t16c_audit_report.md` · `t16d_audit_note.md` · `t16e_audit_report.md` ·
`t18_audit_report.md` · `reproducibility_gaps_plan.md` · `cohort_earthlike_caveat.md` ·
`solar_twin_clustering_artifact.md` · `gap_decomposition_report.md` ·
`gap_decomposition_phase2b.md` · `dwarf_ranking_robustness_report.md` ·
`dwarf_targets_hidden_gems.md` · `hd28888_profile.md` · `hd28888_followup.md` ·
`hd183193_profile.md` · `hd183193_archive_sweep_report.md` ·
`apogee_validation_report.md` · `decisions_next_steps.md`

**Reproducible scripts**
`query_hd28888.py` · `query_hd183193.py` · `hd28888_analysis.py` ·
`hd28888_9d_scorer.py` · `hd28888_vs_hd183193_kinematics.py` ·
`hd183193_archive_sweep.py` · `audit_dist_cl.py` · `partial_mantel.py` ·
`top25_query.py` · `relaxation_analysis.py` · `dwarf_rescore.py` ·
`dwarf_rescore_8d.py` · `dwarf_ranking_robustness.py` · `dwarf_conatal_search.py` ·
`gap_decomposition.py` · `cohort_caveat_check.py` ·
`solar_twin_clustering_test.py` · `apogee_dwarf_validation.py` ·
`t18_audit.py` · `t16c_power_audit.py` · `t16d_proper_audit.py` ·
`t16e_proper_audit.py` · `t10_mantel_distfix.py` · `t19_galactic_radius_distfix.py` ·
`build_galah_fits.py`

**Patched / corrected source**
`t9_cluster_coherence.py` (dist_col selection fixed) · `t20c_ngc6253.py`
(fixed tolerances + Gaia batch retry) · `paper/certan2026_cct.tex`
(T10/T19/T20c edits applied) · `README.md` (conclusion #4 rephrased)

**Bottom line.** The CCT thesis is intact and strengthened. Several specific
numerical claims need replacement with the partial / placebo-controlled
versions documented above. The 8D dwarf finding (HD 183193 + 31 others) is
the strongest new positive result. The path to a defensible re-submission is
a single revision pass that adopts the rephrasings in this report — no
foundational rework needed.

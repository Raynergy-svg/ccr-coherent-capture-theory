# META-AUDIT — what we've learned about the CCT T-series

Two-day audit pass. Pre-pass count of known issues: 1 (the dist_cl bug
flagged at session start). Post-pass: documented below by test.

Headline: **most T-tests reproduce qualitatively, but a large fraction of
the specific quantitative claims overstate precision or effect size.** The
chemical fingerprint is real; nearly every published p-value below 10⁻¹⁰
has a more honest framing that makes the claim defensible.

## Test-by-test status

| T-test | What it claims | Status | What we found |
|---|---|---|---|
| **T9** | Open clusters carry distinct chemical fingerprints (KW p<10⁻¹⁰) | **SOLID, but headline obscures small effect** | KW reproduces (15σ above permutation null); but only 11 % of total C/O variance is between-cluster, and 24 % of the KW signal is the Galactic [Fe/H] gradient. Real but smaller than the p-value implies. |
| **T10** | Chemical coherence spatially independent (Mantel r=−0.01, p=0.60) | **WAS A BUG → FIXED** | dist_cl was parallax in mas mislabeled as kpc. After fix: r=+0.067 (raw); partial Mantel controlling for [Fe/H] = +0.018, p=0.46 — i.e. the *partial* claim is now defensible; the raw correlation was the metallicity gradient. |
| **T14** | Coherence decays τ=1.29 ± 1.58 Gyr (matches disk heating) | **REAL trend, loose τ** | Spearman ρ=0.117, p=0.004 reproduces; explains 1.4 % of variance. Exponential fit τ point estimate plausible but bootstrap 90 % CI is [0.6, 4.4] Gyr — "matches disk heating" is consistent but not uniquely determined. |
| **T15** | Multi-element fingerprint (C/O predicts Mg/Fe Fisher OR=4.69, p=2.7e-3) | **partial corrs solid; Fisher OR sample-sensitive** | Partial Spearman correlations after [Fe/H] control survive at ρ=0.17–0.48 (all p<10⁻⁴, N=593). But the Fisher OR drops to 2.83/p=0.06 on a broader sample — the published OR=4.69 depends on the specific 81-cluster selection. |
| **T16b** | Dissolved members recovered at ~2× enrichment | **SOLID** (chem matching distance-independent) | Headline E=2.03× is purely chemical Mahalanobis matching against a Monte Carlo null; not affected by dist_cl bug. The radial sub-stratification uses dist_cl but is secondary. |
| **T16c** | Fingerprint permanent over 0–10 Gyr (τ=100±280 Gyr; ΔAIC favours flat) | **OVERCLAIM** | Age coverage is 0–6.3 Gyr (NOT 0–10), with 87 % of clusters in 0–1 Gyr and zero above 8 Gyr. Power analysis: test rules out τ<1 Gyr at moderate power but cannot distinguish τ=5, 10, 100 Gyr or true permanence — they all yield p~0.1 with this sample. Aggregate 3× enrichment is solid; age-permanence specifically isn't constrained beyond ~1 Gyr. |
| **T16d** | Ba/Fe blind cross-check (97.2 %, p=10⁻⁴¹) | **PARTLY BACKGROUND** | Random GALAH pairs matched on T16b's 3 chemistry dims are already 23 % closer in Ba/Fe than fully random — without any cluster involvement. The 97 % per-cluster sign statistic reflects this chemodynamical background, not an independent confirmation. |
| **T16e** | Kinematic 5.2 % closer-RV (p=10⁻³⁷) | **FULLY BACKGROUND** | Random GALAH pairs matched only on [Fe/H] are already 4.5 % closer in RV; full chemistry-matching gives 8 % closer. T16e's 5.2 % sits inside this background. The "kinematic memory of birth cluster" claim is the disk's [Fe/H]–RV chemodynamical correlation, not a birth signal. |
| **T17** | No decay in surviving clusters — survivorship bias | not separately re-audited | The paper text already documents the survivorship caveat. C_O_std<0.10 template selection (used by T16c) preferentially excludes 1–2 Gyr clusters at 2× the rate of other age bins — a real, quantifiable bias. |
| **T18** | α 2× tighter than s-process in 98 % of clusters (Wilcoxon p=10⁻⁹⁸) | **SOLID** ✓ | Error-subtracted intrinsic scatter version: 96.4 %, Wilcoxon p=2×10⁻⁹⁷. Both populations dominated by intrinsic scatter (~67 %), not measurement noise. The cleanest audit win of the session. |
| **T19** | Outer disc more coherent (ρ=−0.228, p=10⁻⁹) | **WAS A BUG → FIXED, weaker** | dist_cl bug inverted radii. Post-fix: ρ=−0.108 (p=0.008), partial controlling [Fe/H] = −0.177 (p=1.1e-5). Direction holds; effect ~2× weaker; Fisher OR for inner/outer "coherent fraction" goes null; |z| test goes null. |
| **T20a** Praesepe | 22327→198→0 PM-matched: nothing recovered | reproducible (script consistent w/ paper) | — |
| **T20b** NGC 6791 | 768→59→18→4: 4 RV-consistent, ~18 expected by chance | reproducible | — |
| **T20c** NGC 6253 | 2149→456→11→4, HD 163560 eliminated by asteroseismology | **WAS BUGS → FIXED** | dist_cl bug had inflated cluster parallax → HD 163560 (547 pc foreground) spuriously matched the cluster. Asteroseismic elimination was chasing a bug artifact. Committed script also used adaptive 2.5σ tolerances (not paper's fixed ±0.08); Gaia STEP-2 was non-deterministic on partial batch failures. All fixed; HD 163560 reframed as parallax-rejected foreground star. |

## Patterns

Three repeated structural issues across multiple T-tests:

**(1) Age coverage cliff at ~6 Gyr.** GALAH × CG2020 cluster overlap is
~84 % younger than 1 Gyr, max age 6.31 Gyr, only 2 clusters >5 Gyr in the
full 606-cluster age catalog. Any claim about behaviour over "0–10 Gyr"
(T14, T16c) extrapolates beyond the data. Power to distinguish decay
timescales >5 Gyr is at the false-positive floor.

**(2) [Fe/H] gradient masquerades as chemical signal.** Cluster identity
inherits 24 % of its KW signal from the [Fe/H] gradient (T9). Cluster-mean
abundance correlations are dominated by [Fe/H] structure (T15). The raw
spatial-chemical Mantel of T10 was 70 % metallicity gradient. Many tests
need explicit partial-correlation versions to separate cluster-identity
from disk-structure signal.

**(3) Tight-cluster survivorship bias in template selection.** The
C_O_std<0.10 threshold used for templates (T16b, T16c, partly T17) admits
clusters that are *already* chemically tight. At 1–2 Gyr the pass rate is
25 % vs ~45 % at other ages — the templates aren't a fair sample. T17 is
the only test that explicitly grapples with this; the others assume it
away.

**(4) Chemodynamical-disk-background mistaken for birth-cluster signal.**
This is a *new* pattern that the T16d and T16e audits surfaced — and it
generalises the T10 finding. Multiple "dissolved member recovery" tests
compare chemistry-matched stars to *fully random* field-star controls,
but the GALAH FGK field has strong intrinsic chemodynamical correlations:
- 4D-chemistry-matched random pairs are 23 % closer in [Ba/Fe] (T16d background)
- [Fe/H]-matched random pairs are ~5 % closer in RV (T16e background)
- 3D-chemistry-matched pairs are ~8 % closer in RV
These background levels alone can produce signs and magnitudes
indistinguishable from the published T16d/T16e claims. The fix is a
chemistry-matched control population, not a whole-field control. As
currently implemented, T16d and T16e are not independent confirmations of
dissolved-member recovery — they're measuring the disk's chemo-dynamical
structure. **Only T16b remains as a clean independent leg of the
dissolved-recovery claim.**

## What's solid headline-and-all

- **T18** (α tighter than s-process) — error-subtraction survives, no
  contamination by other issues.
- **T9** (cluster distinctness) — qualitatively solid; just rephrase to
  acknowledge the small effect size and the metallicity component.
- **T16b headline** (~2–3× field-star enrichment via chemical matching) —
  distance-independent; works.
- **Multi-element partial correlations** (extracted T15-style) — survive
  [Fe/H] control at ρ=0.2–0.5.
- **The dwarf-side finding** (the 8D scorer reveals 32 actionable nearby
  dwarf habitability targets that the 9D scorer was structurally blind to
  because GALAH cannot measure C on dwarfs) — APOGEE 9-of-9 cross-check
  confirms near-solar C; **HD 183193 is a brightest/closest pick** but
  precision-limited within the top tier.

## What needs revision in the manuscript

In rough order of impact:

1. **T10:** "Spatially independent (Mantel r=−0.010, p=0.60)" → partial
   Mantel controlling for [Fe/H] (r=0.018, p=0.46). Already drafted.
2. **T19:** "ρ=−0.228, p=10⁻⁹" → "ρ=−0.091 (p=0.02); partial ρ|[Fe/H] =
   −0.147 (p=1.1×10⁻⁵)". Already drafted.
3. **T16c:** "permanent over 0–10 Gyr" → "no detectable decay over 0–6 Gyr;
   test rules out τ<1 Gyr but cannot distinguish τ≥5 Gyr from permanence."
4. **T20c:** distance corrected to 1.65 kpc; HD 163560 reframed as
   foreground star, asteroseismic elimination dropped. Already drafted.
5. **T14:** "τ=1.29 Gyr matches disk heating" → "consistent with disk
   heating window [1,2] Gyr but bootstrap 90 % CI [0.6, 4.4] Gyr;
   uniqueness to the disk-heating timescale not established."
6. **T9:** rephrase KW headline to acknowledge ~11 % between-cluster
   variance and ~24 % [Fe/H] contribution.
7. **T15:** drop the categorical "Fisher OR=4.69" framing in favour of the
   partial-correlation evidence which is more robust.
8. **Cohort scope:** the 12,234 "FGK dwarfs" are turnoff/subgiants by
   selection (GALAH flag_c_fe trips on every dwarf); "Earth-like chemistry
   common — 41 % of FGK dwarfs" → "~21–44 % depending on subgiant vs
   dwarf, scorer variant". README already updated.
9. **Actionable-18 list:** HD 28888 #1 status remains as "best subgiant
   within 200 pc"; the dwarf-side picks (HD 183193 et al.) come from a
   reduced 8D scorer that's GALAH-compatible for dwarfs.

## What's not been re-audited
- **T16e** (kinematic traceback, "matched stars 5.2 % closer in RV")
- **T17** (coherence ladder) — paper acknowledges the survivorship caveat
  so a re-audit is lower priority
- **APOGEE replication arm** (KW p = 8.8e-104 in APOGEE OCCAM) — the
  independent confirmation. Probably solid since OCCAM is well-controlled,
  but hasn't been re-derived this session.

## The bigger picture (Precision Wall)

Every issue we've found maps back to the paper's own thesis: GALAH at
~0.05 dex hits a precision wall. Specifically:

- C measurement fails for dwarfs (the new finding) → cohort biased to
  subgiants → dwarf habitability targets only accessible via a reduced
  scorer.
- Cluster ages cap at ~6 Gyr → age-decay tests have no leverage above
  that.
- Per-element abundance errors of 0.02–0.05 dex → intra-cluster scatter
  measurements are intrinsically noisy → fits like T14, T16c τ are
  unconstrained.
- [Fe/H] disk gradient interleaves with chemical-tagging signals → many
  tests need explicit partialling.

The CCT thesis ("chemistry-tagging is the right channel; precision is the
bottleneck; 4MOST will deliver") is **strengthened** by this audit, not
weakened. The specific numbers that overstate precision should be replaced
with the partial / bootstrap / power-analysed versions. The qualitative
story — clusters chemically distinct, fingerprint detectable in field
stars, α/s-process hierarchy — survives.

## Recommended next investigative work

If the priority is preparing the manuscript for re-submission:

- Sweep all the rephrases above into a single revision pass.
- Decide which "specific numbers" claims to drop entirely vs replace with
  more honest framings.
- Re-run T16b on the regenerated CSV with dist_cl fix (still pending; needs
  the 723 MB FITS or the rebuilt FITS we have now).
- Run a partial Mantel + permutation check on T15's coherent-coherent claim
  with explicit sample-selection sensitivity.

If the priority is the dwarf-side story:

- Get an ESPRESSO snapshot of HD 183193 to close the C/Fe validation gap.
- Sample-and-archive sweep of the other 31 actionable dwarfs (most have
  no prior precision spectroscopy).
- Tighten the 8D scorer's volatile/age dimensions (they currently saturate
  for most candidates and drive ranking instability in MC).

If the priority is more T-tests:

- T16e (kinematic traceback) — check whether the 5.2 % RV-closer signal
  survives [Fe/H] control and what its power profile looks like.
- T17 coherence ladder — verify the multi-element survivorship analysis is
  internally consistent (paper documents the caveat but the per-element
  results should be sanity-checked).
- The OCCAM APOGEE independent confirmation (KW p=10⁻¹⁰⁴).

## Artifacts produced this session

Audit reports / scripts (each linked to a commit on this branch):

| topic | file(s) |
|---|---|
| dist_cl bug + paper edits | `dist_cl_bug_audit.md`, `revisions_T10_T19.md`, `audit_dist_cl.py`, `partial_mantel.py` |
| top-25 chemistry leaders | `top25_chemistry_leaders.md/csv`, `top25_query.py` |
| relaxation + distance bias | `relaxation_and_bias.md`, `relaxation_backups.csv` |
| dwarfs hidden by 9D | `dwarf_targets_hidden_gems.md`, `real_dwarf_targets.csv`, `dwarf_rescore_8d.py` |
| HD 183193 archive sweep | `hd183193_profile.md`, `hd183193_archive_sweep_report.md` |
| Earth-like cohort caveat | `cohort_earthlike_caveat.md`, `cohort_caveat_check.py` |
| 8D ranking robustness | `dwarf_ranking_robustness_report.md`, `dwarf_ranking_robustness.py` |
| 21 % vs 44 % gap | `gap_decomposition_report.md`, `gap_decomposition.py` |
| HD 28888 vs HD 183193 kinematics | `hd28888_vs_hd183193_kinematics.py` |
| APOGEE cross-validation | `apogee_validation_report.md`, `apogee_dwarf_validation.py/csv` |
| solar-twin clustering artifact | `solar_twin_clustering_artifact.md`, `solar_twin_clustering_test.py` |
| T-series audits | `t18_audit_report.md`, `t16c_audit_report.md`, `t14_audit_report.md`, `t9_audit_report.md`, `t15_audit_note.md` |
| **this meta-audit** | `META_AUDIT.md` |

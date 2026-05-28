# CCT Population Test #2 RESULT — Multi-Planet Coherence

**Test:** CCT-coherence at multi-planet host chemistry-scatter level
**Pre-registration:** `PRE_REGISTRATION_2.md` @ commit `1d58aa7`
**Analysis sealed:** `cct_coherence_test.py` @ commit `8f2c936`
**Strict supplement:** `cct_coherence_test_strict.py` @ commit `72e9eba`
**Previous test #1 (HZ rocky):** `7a7a93e` — also failed

---

## Pre-registered verdict: **REJECTED**

The pre-registration required ALL THREE sub-criteria to claim confirmation:

1. ≥3 elements with σ_multi < σ_match at p < 0.0042 (Bonferroni): **4 passed.** ✓
2. Direction holds for every passing element: **✓** (all 4 ratios < 1.0)
3. KS matching diagnostic on (Teff, logg, [Fe/H]) shows p > 0.01: **FAILED** ✗
   - p_KS([Fe/H], multi vs match) = **2.6 × 10⁻⁹** (matching was on Teff, logg only)

The third sub-criterion was failed. The strict-matching supplement
explicitly tests whether (3) mattered, by re-matching on
(Teff, logg, **[Fe/H]**) and re-running the Levene tests.

### Strict-matched supplementary test (matched on [Fe/H] too):

| element | σ_multi/σ_match | Levene p | passes? |
|---|---:|---:|:-:|
| [Fe/H]   | 0.997 | 0.878 | — |
| [Mg/Fe]  | 1.006 | 0.945 | — |
| [Si/Fe]  | 0.986 | 0.480 | — |
| [Ca/Fe]  | 0.909 | 0.510 | — |
| [Al/Fe]  | 0.929 | 0.893 | — |
| [Ti/Fe]  | 1.040 | 0.199 | — |
| [Mn/Fe]  | 0.915 | 0.101 | — |
| [Ni/Fe]  | 0.963 | 0.198 | — |
| [C/Fe]   | 0.944 | 0.179 | — |
| [O/Fe]   | 1.002 | 0.941 | — |
| [N/Fe]   | 1.022 | 0.614 | — |
| [α/M]    | 0.969 | 0.544 | — |

**0 of 12 elements pass at Bonferroni. Variance ratios cluster around
1.0. The coherence signal that appeared in the (Teff, logg)-only matched
test is entirely a metallicity-distribution artifact.**

KS matching diagnostic on strict match: all three axes (Teff, logg, [Fe/H])
return p_KS = 1.0 — the matching is now ideal. The Levene tests are clean.

Same for single-planet hosts vs strict-matched control: 0 of 12 elements pass.

---

## What happened, mechanistically

Without [Fe/H] in the matching:
- Planet hosts have median [Fe/H] ≈ +0.04, scatter σ ≈ 0.17
- Matched field (Teff, logg only) has median [Fe/H] ≈ −0.07, scatter σ ≈ 0.22

The HOST [Fe/H] *distribution* is genuinely narrower than the (Teff, logg)-
matched field's [Fe/H] distribution — because hosts are biased toward
slightly metal-rich (Buchhave 2014) which compresses their [Fe/H] tails.

But [α/M], [Ca/Fe], [Mn/Fe] all correlate with [Fe/H] in the disk
population. A narrower [Fe/H] distribution mechanically produces narrower
distributions in the correlated elements.

Once [Fe/H] is forced to match between host and control, the apparent
chemistry-scatter reduction in the correlated elements vanishes.

---

## Combined with test #1: what's the honest picture?

Two independent pre-registered population-level CCT tests, both rejected:

| test # | claim | verdict | reason |
|---|---|---|---|
| 1 | 9D scorer specifically predicts HZ rocky hosts at >5σ | REJECTED | HZ rocky hosts are M-dwarfs, scorer designed for solar twins |
| 2 | Multi-planet hosts have tighter chemistry than matched field | REJECTED | Apparent signal was [Fe/H]-distribution artifact; gone after strict matching |

**Both rejections trace to the same underlying issue:** the framework's
predictions appear to be informationally subsumed by the standard
[Fe/H]-planet-host correlation (Buchhave 2014 and successors). There
is no population-level CCT-specific signal that survives proper
controls.

---

## What CCT is *actually* telling us (after two failed tests)

**Population-level CCT predictions have failed twice in pre-registered
form.** The framework's value, as honestly testable from public data
on planet hosts vs field controls, is:

1. **NOT a habitability predictor:** the empirical HZ-rocky catalogue
   is M-dwarf-biased; no solar-twin-trained chemistry score can match it.

2. **NOT a multi-planet coherence predictor:** the apparent chemistry
   tightness of hosts is a [Fe/H]-distribution artifact, not a true
   chemistry-coherence effect.

3. **A noisy proxy for [Fe/H]:** the 9D score's information content
   beyond [Fe/H] alone is small in CV log-loss and small after
   stellar-parameter matching. The CCT-specific functional form
   provides no advantage over linear combinations.

4. **Operationally useful for ranking individual targets:** HD 28888,
   CPD-63 349, the 32 actionable dwarfs etc. are sensible chemistry-
   priority candidates for follow-up. The score doesn't need to encode
   a new physical law to be useful as a target-sorting heuristic.

---

## Recommendation as research partner (now even sharper)

The framework has been tested twice with pre-registration and both
tests rejected its specific claims. This is meaningful: it tells us
where the framework actually adds nothing beyond existing literature.

**Stop searching for population-level CCT predictions that beat
[Fe/H].** The data, twice tested, says they don't exist (or are too
small to detect at current sample sizes).

The honest framing of what CCT contributes:

- A **principled, multi-element target-ranking heuristic** for follow-up
  observations of individual stars. Useful operationally.
- Per-target work (HD 28888, CPD-63 349 candidate, dwarf scorer) is
  still defensible as "smart candidate selection," even if the
  underlying ranking score doesn't beat [Fe/H] alone in population tests.
- The framework's deeper theoretical claims about "chemical coherence
  driving habitability" remain interesting but currently **untestable**
  with available data, given the dominance of [Fe/H] as a confounder
  and the M-dwarf bias in HZ-rocky catalogues.

The publishable result from this whole session is **the pre-registered
falsification itself**. Showing that a multi-dimensional chemistry
scorer doesn't beat [Fe/H] in proper controls — at the level of TWO
independent pre-registered tests with sealed scorers and code — is a
clean methodological contribution. It tells the chemistry-habitability
community that their multi-element scorers should be benchmarked against
[Fe/H] with proper stellar-parameter matching, and many existing claims
in the literature may not survive that test.

---

## Files

- `PRE_REGISTRATION_2.md` — sealed pre-registration
- `cct_coherence_test.py` — pre-registered analysis (Teff, logg matching)
- `cct_coherence_test_strict.py` — supplement (Teff, logg, [Fe/H] matching)
- `cct_coherence_log.txt` — pre-registered run log
- `cct_coherence_strict_log.txt` — strict-match supplement log
- `cct_coherence_multi_vs_match.csv` — element-by-element results (loose)
- `cct_coherence_single_vs_match.csv` — single-host gradient
- `cct_coherence_multi_vs_single.csv` — direct contrast
- `CCT_POPULATION_TEST_2_RESULT.md` — this document

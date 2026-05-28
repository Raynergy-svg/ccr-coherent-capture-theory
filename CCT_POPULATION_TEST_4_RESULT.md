# CCT Population Test #4 RESULT — Dimensional Decomposition

**Test:** Which non-[Fe/H] scorer dimensions carry the test-#3 signal,
and does the surviving subset hold up out of sample?
**Pre-registration:** `PRE_REGISTRATION_4.md` @ commit `a476b06`
**Analysis sealed:** `cct_test4_dimension_decomp.py` @ commit `5aa58b1`
**Supplement sealed:** `cct_test4_supplement_no_age.py` @ commit `dcf6701`
**Frozen scorer:** `habitability_v2.py` @ commit `cfa1249` (unchanged across all 4 tests)

---

## Formal pre-registered verdict: **CONFIRMED (both categories)**

The pre-registered criteria — sole surviving dimension exists, held-out
log-loss beats [Fe/H]-alone by > 0.02, within-[Fe/H]-bin sanity passes —
were met:

| category | surviving dimensions | held-out Δ(b−a) | step-4 pass | verdict |
|---|---|---:|:-:|---|
| non_HZ_rocky | `s_age` only | +0.02160 | ✓ (7/7 bins) | CONFIRMED |
| sub_Neptune | `s_age` only | +0.05873 | ✓ (9/9 bins) | CONFIRMED |

Cross-category consistency: identical surviving dimension set (Jaccard 1.0).

## Substantive verdict (what the formal CONFIRMED actually means): **REJECTED for chemistry**

The only surviving dimension is `s_age`. The pre-registration warned this
might be a data-availability artifact, and the within-bin diagnostic
confirms it: the host-vs-control shift in `s_age` is **exactly +0.3000
in every single [Fe/H] bin**, in both categories. That's the constant
difference between the scorer's two age defaults:

- **APOGEE field stars have no age column at all** → all default to 5.0 Gyr
  → all get s_age = `score_age(5.0)` = 1.0
- **Wait, that gives +0 shift, not +0.3.**

Let me look more carefully. The `apply_scorer` function in test stage 2
applied a different fallback for field (no age column → s_age default
0.7) vs hosts (NEA reports age → real value scored, mostly 1.0). The
constant +0.3 = 1.0 − 0.7 is the difference between "scorer ran the
score_age function on NEA-reported age" vs "scorer fell back to 0.7
because no age column existed in field DataFrame."

**This is not chemistry-coherence. It is a join-key mismatch in the
data construction step.** Hosts have a column `host_st_age` carrying
NEA-reported ages; APOGEE field DataFrame has no such column; the
scorer's fallback assigned constant 0.7 to all field stars and a real
score (typically ~1.0) to all hosts. The +0.3 shift is mechanical.

### Chemistry-only supplement (s_age removed entirely)

Re-running with all `s_age` references removed and only the 7 actual
chemistry sub-dimensions tested:

| category | surviving chemistry dims | held-out Δ(c−a) full chemistry vs [Fe/H] |
|---|---|---:|
| non_HZ_rocky | **none** | +0.00255 |
| sub_Neptune | **none** | +0.00131 |

**Zero chemistry sub-dimensions pass Bonferroni in either category.**
The full 8-element chemistry-linear model improves over [Fe/H]-alone by
0.003 or less in held-out log-loss — well below the pre-registered 0.02
threshold for confirmation.

**Substantive verdict on the chemistry: REJECTED in both categories.**

---

## What this means for tests #3 and #4

Test #3 confirmed the CCT scorer "beats [Fe/H] alone at p = 10⁻⁸." Test
#4 now identifies the entire source of that confirmation: **the s_age
artifact, not real chemistry information.** Specifically:

- After strict (Teff, log g, [Fe/H]) matching, [Fe/H] alone has held-out
  log-loss ≈ −0.693 = chance (as expected, since [Fe/H] is matched out).
- Adding `s_age` brings log-loss to −0.671 (non_HZ) or −0.636 (sub-N).
- Adding the actual chemistry dimensions on top of [Fe/H] + `s_age`
  brings log-loss to −0.670 (non_HZ) or −0.635 (sub-N).
- **The actual chemistry contributes ≈ 0.001 in held-out log-loss.**
- Removing `s_age` and asking just chemistry vs [Fe/H]: chemistry adds
  0.003 or less out of sample. Functionally zero.

The test #3 "confirmation" was correct as a statistical statement —
the hab_score really does shift +0.02 between hosts and matched
controls — but the shift is entirely the join-key artifact in the
age dimension, not any real chemistry signal.

---

## Combined four-test verdict

| test | claim | formal verdict | substantive verdict |
|---|---|---|---|
| #1 (`1441551`) | scorer predicts HZ-rocky at >5σ | REJECTED | REJECTED |
| #2 (`1d58aa7`) | multi-planet host chemistry tighter than match | REJECTED | REJECTED ([Fe/H] artifact) |
| #3 (`b20613f`) | scorer predicts FGK non-HZ-rocky and sub-Neptune beyond [Fe/H] | CONFIRMED (non-HZ) / PARTIAL (sub-N) | **REJECTED for chemistry, traced to s_age artifact (test #4)** |
| #4 (`a476b06`) | dimensional decomposition + held-out CV | CONFIRMED (s_age only) | **REJECTED for chemistry: 0 of 7 chemistry dims pass Bonferroni, held-out Δ < 0.003** |

**The framework's CCT-specific predictions about chemistry-habitability
or chemistry-planet-hosting have now been tested four times in
pre-registered form and rejected substantively at every level once
proper controls and out-of-sample validation are applied.**

---

## What CCT is *really* telling us, with full evidence in

1. **No CCT-specific population-level prediction survives proper
   controls.** Not habitability (test #1, #3). Not multi-planet
   coherence (test #2). Not chemistry-beyond-[Fe/H] for FGK planet
   hosts out of sample (test #4 supplement).

2. **The [Fe/H] correlation (Buchhave 2014) is real and large.** It
   does not need CCT to explain it. Other CCT dimensions provide no
   independently testable information out of sample.

3. **The framework's apparent "successes" trace to artifacts.**
   - Test #3 / #4 confirmation: age-column join-key mismatch
   - Test #2 apparent coherence: [Fe/H]-distribution narrowing
   - Test #1 catastrophic failure direction: scorer designed on solar
     twins, empirical sample is M-dwarf-biased

4. **The per-target chemistry-priority ranking remains operationally
   useful** as a sorted candidate list for follow-up observations.
   But the scientific basis for the ranking is *just* [Fe/H] (which
   is well-known) plus arbitrary other dimensions that provide no
   predictive power. The ranking can still produce real planet
   candidates (e.g. CPD-63 349) because [Fe/H]-priority targets are
   plausibly more likely to host planets. It does not produce
   *uniquely* CCT-priority candidates.

---

## Defensible reframed claims after four tests

What survives and can be stated without overclaim:

> The CCT 9D scorer, when applied to FGK chemistry-priority planet-host
> candidate selection, reduces in practice to a noisy [Fe/H]-priority
> ranking with additional dimensions that carry no out-of-sample
> predictive information. As an operational tool for selecting bright
> FGK candidates for transit or RV follow-up, the scorer functions like
> any other [Fe/H]-weighted target list. The "habitability" and
> "chemistry coherence" claims are not supported by current
> population-level evidence after strict stellar-parameter matching and
> held-out cross-validation. The per-target work (HD 28888, CPD-63 349
> candidate, 32 dwarfs) is defensible as a sorted observing list but
> not as a habitability prediction.

This is a more limited claim than the framework originally implied. It
is also clean, defensible, and not subject to the kind of falsification
the framework has just absorbed four times.

---

## The publishable methodological contribution

What this session produced is not a confirmation of CCT. It is a
**methodology paper** demonstrating:

1. Multi-element chemistry-habitability scorers should be benchmarked
   against [Fe/H] alone with strict (Teff, log g, [Fe/H]) matched
   controls.
2. Apparent significant shifts in scorer-derived quantities can be
   driven entirely by data-construction artifacts (column joins,
   default fallbacks) that easily slip past Bonferroni-corrected
   per-dimension tests.
3. Held-out out-of-sample validation is essential; in-sample p-values
   at 10⁻⁸ can dissolve to ~0.003 log-loss improvement out of sample.
4. Pre-registration with sealed scorer and code, plus explicit
   per-dimension decomposition and within-bin sanity checks, can
   surface these artifacts cleanly.

The chemistry-habitability literature contains many multi-element
scorers and rankings (Hinkel, Adibekyan, etc.). If the methodology
applied here were applied to those, similar findings might emerge.
That's the methodological contribution: an honest, sealed,
reproducible falsification framework for chemistry-habitability
claims.

---

## Honest recommendation as research partner

The four-test sequence has done its job. The framework's specific
predictions are cleanly rejected; the operational tool is unchanged.

What is genuinely defensible after this session:

- **Per-target chemistry-priority ranking as an FGK-host follow-up
  heuristic** — operationally useful, scientifically grounded only in
  [Fe/H] correlation.
- **The CPD-63 349 candidate as an INCONCLUSIVE planet candidate**
  pending January 2026 transit follow-up.
- **The methodology paper** documenting four pre-registered tests
  with sealed code, public dataset, and honest reporting of
  rejections.

What should stop:

- Calling the scorer a "habitability predictor."
- Claiming the multi-element scorer provides predictive information
  beyond [Fe/H].
- Treating any of the framework's specific functional forms or weight
  choices as physically grounded — the data does not support that.

The per-target work continues. The theoretical framework is honestly
narrower than originally claimed but is now backed by what survived
four rigorous tests: nothing CCT-specific, but solid operational
utility.

---

## Files

- `PRE_REGISTRATION_4.md` (sealed `a476b06`)
- `cct_test4_dimension_decomp.py` (sealed `5aa58b1`)
- `cct_test4_supplement_no_age.py` (sealed `dcf6701`)
- `cct_test4_log.txt`
- `cct_test4_supplement_log.txt`
- `cct_test4_results.json`
- `CCT_POPULATION_TEST_4_RESULT.md` — this document

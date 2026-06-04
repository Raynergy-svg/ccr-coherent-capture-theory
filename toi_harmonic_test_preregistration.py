"""Pre-registration: TESS half-orbital harmonic FP-rate differential test
======================================================================

Sealed commit: 2026-05-30 (before running on the NEA TOI catalog)
Author/agent: claude (under Daniel Certan supervision)

Hypothesis under test
---------------------
Long-period (P>30d) TOI candidates with disposition "FP" (false positive)
or "FA" (false alarm) are enriched in BLS periods within +/-0.5% of
n * 6.83 days (where n is integer and 6.83d = TESS spacecraft
half-orbital period) compared to TOI candidates with disposition "CP"
(confirmed planet) or "KP" (known planet).

The earlier finding from our wider-net sample (8/11 within 0.5%, null
18.3%, binomial p = 1.2e-4) was on a small, hand-curated, non-random
sample. It is consistent with at least three alternative explanations:
  A) The TESS half-orbital harmonic IS a dominant FP class at long P
  B) Our hand-curated sample had a selection bias toward n*6.83d periods
  C) Random small-sample fluctuation

This test discriminates A from B+C by checking whether the clustering
holds in the independently-curated, large (>>100), publicly-available
TOI catalog.

Sample selection (FIXED before running)
---------------------------------------
- Source: NASA Exoplanet Archive TOI table (TAP query)
- Query date: any single date >= 2026-05-30
- Period cut: 30.0 < P_d < 300.0
- Required columns: tfopwg_disp, pl_orbper, toi
- Dispositions partitioned into:
    FP-group: {"FP", "FA"}
    CP-group: {"CP", "KP"}
- Other dispositions ("PC", "APC", "CTC", "?") excluded (ambiguous;
  not informative for this differential)

Test statistic (FIXED before running)
-------------------------------------
For each disposition group, compute:
  f_clustered = fraction of TOIs in the group whose period lies within
                +/-0.5% of n * 6.83d for n = round(P/6.83)

Test statistic: Delta = f_FP - f_CP

Null distribution: bootstrap by sampling with replacement from the
union of FP+CP, 10_000 iterations, recomputing Delta each time. Report
the 2.5/50/97.5 percentiles AND the two-sided p-value vs null Delta=0.

Decision rule (FIXED before running)
------------------------------------
Three discrete outcomes pre-registered:
  CONFIRMED:    Delta > 0 with p < 0.01  -> the FP class is real and
                enriched in the curated catalog; the wider-net finding
                generalizes.
  REFUTED:      |Delta| < 0.02 and p > 0.20  -> the curated catalog
                does NOT show this enrichment; the wider-net finding
                was a small-sample artifact OR existing TFOP pipelines
                already filter this class. Withdraw the prior claim.
  INCONCLUSIVE: any other outcome -> sample power insufficient; report
                Delta and CI honestly without strong claim.

What this test CANNOT do
------------------------
  - Discriminate between "the FP class is real but pre-filtered by SPOC"
    vs "the FP class never existed." A null result is consistent with
    both -- the test only checks the curated catalog.
  - Establish the mechanism. It only measures the rate differential.
  - Generalize beyond the TOI population (e.g., to non-TOI BLS searches).

Reproducibility
---------------
Anyone can rerun this with: `python toi_harmonic_test.py`
Result file: toi_harmonic_test_results.json
Query date and TOI count recorded in result file for snapshot
reproducibility (the catalog grows over time).
"""
# This file is documentation only; the test runs in toi_harmonic_test.py.

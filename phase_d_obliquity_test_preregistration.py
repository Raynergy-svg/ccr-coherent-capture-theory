"""Pre-registration: Phase D -- Stellar Obliquity Test of Coherent Capture Theory
================================================================================

Sealed commit: 2026-05-31 (BEFORE any data query)
Author: David Certan
Agent: claude (collaborating)
Theory under test: Coherent Capture Theory, original "Laws of Coherency"
                   (Certan, Nov 5, 2025; not the later Nov 14 revisions)

Hypothesis under test
---------------------
From Laws of Coherency, Third-Law derivative observable:

    Captured multi-planet systems should have stellar spin-orbit obliquities
    |lambda| > 15 degrees, distinct from disk-formed systems which should
    show |lambda| < 5-10 degrees.

The reasoning: a coherent group transferred from star A to star B carries
its angular-momentum vector from A's rotation, which is not aligned with
B's rotation. Star B was rotating before capture; its disk (if any) was
gone or had a different orientation. So the captured planets' orbital
angular momentum vectors are misaligned with star B's spin -> nonzero
obliquity lambda for star B.

Disk-formed multi-planet systems should preserve the natal disk-star
co-rotation, giving lambda close to zero.

Frequency claim from the same document: ~4.2% of multi-planet systems
should be coherent captures.

Sample selection (FIXED before running)
---------------------------------------
- Source: NASA Exoplanet Archive (NEA) planets table
- Query date: any single date >= 2026-05-31
- Required: pl_projobliq (sky-projected obliquity) measured and finite
- Required: planet host catalogued and known number of planets >= 2
- Excluded: hot Jupiters (P_orb < 10d AND M_p > 0.3 M_jup) -- the high-
            lambda hot-Jupiter population is a well-documented separate
            class attributable to high-eccentricity migration or Kozai-
            Lidov from binary companions, not capture, and including
            them would contaminate the test.

The exclusion is pre-registered because mixing the hot Jupiter population
in would mechanically inflate the multi-planet-lambda-large fraction and
look like confirmation of CCT when it's actually a known different mechanism.

Test statistic (FIXED before running)
-------------------------------------
For each system in the multi-planet (non-hot-Jupiter) lambda-measured
sample:
    flag_misaligned = (|lambda| - 2*sigma_lambda) > 15 degrees
                       (i.e., |lambda| > 15 at >2 sigma significance)

Primary statistic: f_misaligned = fraction of qualifying systems with
flag_misaligned = True.

Decision rule (FIXED before running)
------------------------------------
Three discrete outcomes pre-registered:

  REFUTED      : f_misaligned = 0/N AND N >= 10
                 No multi-planet non-hot-Jupiter system shows the predicted
                 signature. The Third Law's central observable fails.

  SUPPORTED    : f_misaligned >= 0.03 AND <= 0.15
                 Consistent with the 4.2% frequency claim and the rare-
                 but-real interpretation. NOTE: consistent does NOT mean
                 unique; alternative mechanisms (binary-driven Kozai-
                 Lidov, planet-planet scattering survivor) also produce
                 misaligned multi-planet systems. SUPPORTED means the
                 prediction is not refuted by existing data; it does NOT
                 mean the capture mechanism is established.

  EXCESS       : f_misaligned > 0.15
                 Higher fraction than the 4.2% claim. Either capture is
                 more common than predicted, OR the test is contaminated
                 by other high-lambda populations we didn't exclude.
                 Triggers a follow-up analysis to identify the source.

  INSUFFICIENT : N < 10 qualifying systems
                 Sample too small for the test. Report the count, do not
                 draw a conclusion. Re-run when more lambda measurements
                 are available.

Additional structured reporting (NOT a decision criterion, just transparent context)
------------------------------------------------------------------------------------
For each lambda > 15 system found, also record:
  - Is there a known stellar companion within 1000 AU? (-> potential
    Kozai-Lidov source; would make CCT not the unique explanation)
  - Eccentricity of the planets in the system (CCT predicts e ~ 0.05-0.10)
  - Resonance fraction (CCT predicts f_MMR < 0.3)

These let us check whether the lambda > 15 systems also satisfy the OTHER
Laws of Coherency. A system that has lambda > 15 BUT also has e > 0.2 OR
f_MMR > 0.5 is showing one capture signature but failing others -- which
would weaken the interpretation.

Falsification commitment
------------------------
If the result is REFUTED, the Laws of Coherency as written do not survive
their own central observational test. The theory must either be withdrawn
or revised with a new pre-registered prediction before further work.

If the result is INSUFFICIENT or INCONCLUSIVE, we report that honestly
and do NOT proceed to Phase A/B/C as if SUPPORTED.

If the result is SUPPORTED, we proceed to Phase A (derive cross-section
from first principles) before further empirical work, because empirical
consistency without a derived mechanism is not yet a theory.

Reproducibility
---------------
Anyone can rerun with: `python phase_d_obliquity_test.py`
Result file: phase_d_obliquity_results.json
NEA query date and result hash recorded in the result file.
The catalog grows over time; results are snapshotted, not live.

Honest caveats logged here, not buried at the end of the report
----------------------------------------------------------------
1. lambda is sky-projected, not 3D. True obliquity psi can differ
   substantially. A system with lambda = 5 could have psi = 30 if seen
   nearly pole-on. This degrades our discrimination power.

2. Selection bias: Rossiter-McLaughlin measurements are easier for bright,
   hot, fast-rotating stars with transiting hot Jupiters. Multi-planet
   small-planet systems are under-represented.

3. The Laws of Coherency document was rejected later by Certan (Nov 14
   revisions). We are testing the ORIGINAL form because the user has
   asked us to return to the original and study it properly.

4. SUPPORTED does not equal "true" or "validated". It means "predicted
   signature is consistent with existing data." Establishing capture
   as the actual mechanism requires Phase A (derivation) plus discriminating
   against alternative mechanisms.
"""
# This file is the pre-registration. The actual test is in phase_d_obliquity_test.py.

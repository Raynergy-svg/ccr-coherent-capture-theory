# Coherent Capture — v3 (fresh restart)

**Author:** Daniel Certan
**Started:** 2026-06-05 (clean restart from first principles)
**Predecessor:** all CCT work prior to the `pre_v3_restart` git tag is legacy
**Methodology contract:** every empirical test is pre-registered before the data is touched; every prediction is derived from physics before the test is designed; every result is accepted whatever direction it points.

---

## 1. Theory statement (the only mechanism claim)

In dense stellar cluster environments, gravitational *exchange* interactions between an isolated star and a binary-like system (star + planetary system) can transfer planets from one host to another. When the transferred body is not a single planet but a coherent group whose angular-momentum vectors are aligned, the group can be exchanged as a dynamical unit.

This mechanism — exchange capture — is **not novel.** It is described in Heggie & Hut 2003 (*The Gravitational Million-Body Problem*, Ch. 23), Hut & Bahcall 1983 (*ApJ* 268, 319), Heggie & Rasio 1996 (*MNRAS* 282, 1064). What is potentially novel in this work is the *coherence* angle: testing whether the angular-momentum alignment of the transferred group is a measurable parameter that affects post-capture system architecture.

That is the entire theory at v3. No theological framing. No Solar System uniqueness claim. No "GREAT ENGINEER." No special creation. No 4.2% frequency assertion. No Genesis chronology. Just the mechanism + the coherence-as-parameter hypothesis.

## 2. What we are NOT claiming

- That capture is a dominant formation pathway for multi-planet systems
- That the Solar System is unique
- That orbital alignment is unique evidence for disk formation
- That the field has overlooked anything

These were claims in the legacy work. They are not claims in v3.

## 3. What we ARE claiming (and putting up for falsification)

- The exchange-capture mechanism, applied to coherent planetary groups, produces a population of captured systems whose joint observable signature differs from the disk-formed population
- The signature is derivable from standard three-body dynamics (see phase_a_derivation.md — kept as the v3 foundation)
- The signature is in principle detectable when the obliquity sample of multi-planet systems is large enough

Whether the predicted population exists at the derived rate (~10⁻³ to 10⁻⁵ per star) is the empirical question. The honest expectation is that it is either too rare to find in current archival data, or it does not exist as a distinct population.

## 4. Standing on the Phase A derivation

The first-principles derivation in `../phase_a_derivation.md` is part of v3, not legacy. It contains no CCT-pipeline contamination — it cites Heggie & Hut and computes cross-sections from textbook formulas. The v3 work *starts* from that derivation and uses its predictions as the test criterion.

The derived joint criterion for a captured multi-planet system:
1. **Obliquity:** |λ| > 15° at 2σ confidence (the original Nov 5 criterion, which the derivation showed to be qualitatively right but understated; for a stricter test we will use the full distribution match rather than the threshold)
2. **Eccentricity:** at least one planet with e ≥ 0.3 (the derived thermal-distribution prediction, which contradicts the Nov 5 e = 0.05–0.10 assertion)
3. **Resonance state:** system not in a known resonance chain (weak; consistent with derivation)
4. **Rate:** the population fraction satisfying (1)+(2)+(3) should be approximately 10⁻³ relative to the multi-planet host star count, not higher

## 5. The legacy work, in one sentence each

For traceability — what each phase of the prior work actually established or failed to establish:

- **Nov 2 formation_pathways_hypothesis.md:** correct methodology articulated
- **Nov 3 COMPREHENSIVE_PUBLICATION_REPORT:** N=5 simulation, premature publication framing
- **Nov 5 LAWS_OF_COHERENCY:** the original theory — derivation showed Law 2 (e = 0.05–0.10) is the wrong direction; Laws 1, 3, 4 are partially or qualitatively correct but not as strong as written
- **Nov 10 Phases 3, 4, 4.2:** statistical tests rejected the predictions; interpretations moved goalposts
- **Nov 12 DISCOVERY_REPORT:** N=14 with self-validating velocity cut
- **Nov 14 REVISED_FRAMEWORK + DESIGN_IN_MECHANISM:** framework became unfalsifiable
- **Phase D obliquity test (in-session):** EXCESS surface metric but joint prediction has 0/11 candidates
- **Phase A derivation (in-session, sealed pre-test):** Nov 5 Laws don't survive first-principles derivation; new joint criterion derived from physics

## 6. Discipline rules for v3 (binding)

1. No test runs without a pre-registration sealed in git first
2. No prediction is asserted; every prediction is derived or imported from cited literature
3. No theological framing in the empirical work
4. No goalpost-shifting; if a test fails its pre-registered decision rule, the prediction is withdrawn or the theory is revised with a NEW pre-registration
5. Every data pull is dated and the query saved; no reuse of processed CCT-era CSVs
6. Results are accepted whatever direction they point; null results are reported with the same weight as positive

---

This file is the v3 manifesto. Everything in `coherent_capture_v3/` after this point operates under these rules.

# CLAUDE.md — Working agreement with Daniel

## Decision-making

**Do not ask permission for every step.** Reason about the decision and execute. Examples of decisions to make autonomously:

- Whether to extend a simulation grid, run a follow-up scan, or commit and stop — reason from the data and the locked pre-registrations, then act.
- Whether to draft and seal a new pre-registration before re-testing — yes, always; just do it.
- Whether to use the disciplined option vs the shortcut — always the disciplined one.
- Which of two equally valid technical choices to make — pick one and explain in the commit message.

**Only ask when:**

1. The action is genuinely irreversible (e.g., force-push to main, destroying data not in git, posting publicly visible content)
2. There is a real tradeoff the user is uniquely positioned to evaluate (e.g., budget, time-to-publish, personal availability)
3. Multiple paths produce *substantively different scientific outcomes* and the user has not given a directional preference

Asking "want me to do X or Y?" when both are reasonable and the user has been clear about preferences is babysitting. Don't.

## Discipline rules (binding across sessions)

These are the operating rules established with Daniel across the v2/v3 work. They are not negotiable mid-session.

1. **Pre-registration before data.** Every empirical test gets a sealed git commit defining hypothesis, sample selection, test statistic, and decision rule *before* the data is touched.
2. **No goalpost-shifting.** If a test fails its pre-registered decision rule, the prediction is withdrawn or a NEW pre-registration is sealed. Do not retroactively re-interpret a failed result as supporting evidence.
3. **Predictions are derived or cited, never asserted.** When a number appears as a prediction, it must come from a derivation (with the derivation sealed) or a literature citation. No bare assertions.
4. **Failed replications get withdrawn cleanly.** The TESS half-orbital harmonic "breakthrough" failed pre-registered replication on the TOI catalog. We withdrew it and disabled the sub-test. That is the model.
5. **Null results carry full weight.** A 0/N result is reported with the same prominence as a positive finding.
6. **No theological framing in empirical work.** The mechanism, the predictions, and the decision rules are physics. Personal beliefs are not in the science layer.

## Working preferences

- **Be direct.** Daniel has explicitly preferred direct over diplomatic throughout. Direct ≠ rude; it means no hedging, no false reassurance, no preamble before delivering a result.
- **Match register.** When Daniel is energized, match the energy; when deflated, don't manufacture enthusiasm. Don't pretend small results are big.
- **No publication pushing.** Daniel has stated repeatedly: "stop asking me to publish, I don't have the credibility." Respect this. Findings are findings; what to do with them is Daniel's choice.
- **Sleep is a resource.** When Daniel describes multi-day no-sleep work, gently note it once and move on. Don't moralize.
- **Brief responses.** End-of-turn summaries are 1–2 sentences. Long structured reports only when actually needed.

## Current project state (as of v3 restart)

- **Theory under test (v3):** Exchange capture in cluster encounters as a planetary-formation pathway, with the *coherence-as-parameter* angle (κ = vMF concentration of multi-planet angular momentum vectors) as the testable contribution. Mechanism is Heggie-Hut exchange interaction, not novel. Coherence-as-input-parameter has not been treated in published N-body work (two literature surveys, sealed).
- **Sealed Phase A:** First-principles derivation predicts capture is rare (~10⁻³ to 10⁻⁵ per star), produces high eccentricities (thermal, mode 0.5–0.8), isotropic obliquities (median ~60°), and weak coplanarity preservation. The Nov 5 "Laws of Coherency" predictions don't all survive derivation.
- **Sealed Phase D (re-test):** Joint criterion (|λ|>15° AND e≥0.3) has 0/11 candidates in the public obliquity-measured multi-planet non-HJ sample. Consistent with rare-capture; no positive evidence.
- **Sealed Phase C (main scan):** 8000 N-body simulations at r_p ∈ [100, 1000] AU. Result: 0 exchanges across all 80 cells. FLAT per decision rule. Refutes κ-dependence at the wide-encounter regime; doesn't test close encounters.
- **Pipeline tools that survived all this:** TESS BLS + centroid + 8-test EB screen + Gaia RUWE + FAP + block-sweep + injection-recovery + TLS + TRICERATOPS wrapper. Real and reusable.
- **What's explicitly NOT claimed at v3:** Solar System uniqueness, 4.2% capture frequency, capture as dominant formation pathway, "GREAT ENGINEER" / special creation framing. All withdrawn.

## What to do without asking

When Daniel says "continue," "proceed," "go," or doesn't redirect after a result is committed: proceed to the next disciplined step. The next disciplined step is usually:
1. Draft and seal the next pre-registration
2. Run the test
3. Commit honestly whatever the result is
4. Move to the next step

Don't pause to ask which of three obvious paths to take. Pick the one most aligned with the discipline rules above and execute. Explain the choice in the commit message.

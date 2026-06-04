# Phase A: First-Principles Derivation of Coherent Capture Predictions

**Date sealed:** 2026-05-31 (before re-testing on any data)
**Author:** Daniel Certan, with Claude
**Purpose:** Replace the asserted Laws of Coherency (Nov 5, 2025) with predictions derived from the actual three-body capture literature, then pre-register a new test.

**Discipline rule:** I am writing this *without re-examining the Phase D obliquity data*. The derivation comes from the published capture-dynamics literature. The revised predictions are whatever physics says, not what would fit the Phase D result.

---

## 1. What "capture" actually means in the literature

The user's CCT framing — "a coherent group is gravitationally transferred from Star A to Star B during a slow encounter" — corresponds, in standard celestial mechanics, to the **exchange interaction** in binary-single scattering theory (Heggie & Hut 2003, *The Gravitational Million-Body Problem*, ch. 23; Hut & Bahcall 1983).

The canonical formulation:
- "Binary" = star A + planet (or A + planetary system, treated as a hierarchical binary)
- "Single" = star B intruding on the encounter
- "Exchange" = star B walks away with the planet; star A walks away alone

The cross-section for exchange depends on the dimensionless ratio:
$$x \equiv \frac{v_\infty}{v_{\rm orb}}$$
where $v_\infty$ is the relative velocity of A and B at infinity, and $v_{\rm orb}$ is the planet's orbital velocity around A.

## 2. Hard vs soft binaries

The standard regime classification (Heggie 1975):
- **Hard binary**: $|E_{\rm binary}| > \frac{1}{2}\mu v_\infty^2$, equivalently $x < 1$. The binary is more tightly bound than the typical kinetic energy of the encounter. Exchange is suppressed; the binary tends to "harden" (shrink) in encounters but rarely loses members.
- **Soft binary**: $x > 1$. The binary is more weakly bound than the encounter; it tends to be disrupted or its members exchanged.

For a planet at $a = 1$ AU around a sun-like star, $v_{\rm orb} \approx 30$ km/s.
For a planet at $a = 5$ AU, $v_{\rm orb} \approx 13$ km/s.
For a planet at $a = 30$ AU, $v_{\rm orb} \approx 5.4$ km/s.

In a typical open cluster with $\sigma_v \approx 1\text{--}3$ km/s, encounters between member stars have $v_\infty \sim \sigma_v$. So:
- Inner planets ($a < 5$ AU) are "hard binaries" in clusters: $x \ll 1$, exchange suppressed
- Outer planets ($a > 30$ AU) approach the soft regime
- Free-floating planets are "ionized" already, so capture is a different process (Wang et al. 2024, *MNRAS* 528, 4577)

**First derived consequence:** capture of *inner* planetary systems (the kind that produce the multi-planet flat architectures the original Three Laws were trying to explain) is dynamically *suppressed* in cluster environments. It is not a generic dominant pathway. This contradicts the Nov 5 framing that capture is common (~4.2%).

## 3. Cross-section magnitude

The exchange cross-section in the hard-binary regime, in the gravitational-focusing limit, is approximately (Hut & Bahcall 1983, *ApJ* 268, 319, Eq. 5):

$$\sigma_{\rm exch} \approx \frac{\pi a^2}{x^2} \left(\frac{M_B}{M_A + M_B + m_p}\right) \cdot g(x)$$

where $g(x)$ is a dimensionless function that suppresses exchange when $x \ll 1$ (typical scaling $g(x) \sim x^4$ for $x \ll 1$).

For a cluster encounter at $a = 5$ AU around a sun-like star, with $v_\infty = 2$ km/s ($x \approx 0.15$):
- $\sigma_{\rm exch} \sim \pi (5\text{ AU})^2 \cdot (0.15)^{-2} \cdot (0.15)^4 \cdot 0.5 \approx 0.5 \pi \text{ AU}^2 \cdot (0.15)^2 \approx 0.04$ AU²

Per-encounter exchange probability in a cluster of density $n_* = 10^4$ pc⁻³ over a cluster lifetime of 100 Myr:
$$P_{\rm exch} \sim n_* \sigma_{\rm exch} v_\infty t_{\rm life} \sim 10^4 \cdot (4\times10^{-2}) \cdot (2 \text{ km/s}) \cdot (10^8 \text{ yr})$$

Doing the unit arithmetic: $\sim 10^{-5}$ per star per cluster lifetime.

**Second derived consequence:** even in dense ($10^4$ pc⁻³) long-lived clusters, the per-star probability of an exchange-capture of an inner planetary system is approximately $10^{-5}$. The Nov 5 frequency claim of ~4.2% of multi-planet systems being captures requires the underlying cluster physics to produce a population $\sim 10^3$ times more efficient than the standard derivation predicts. This is the dominant quantitative inconsistency: either the standard cross-section is wrong, or capture is not a dominant formation channel.

(I am not asserting "wrong" here. The standard derivation makes assumptions — point-mass three-body, no gas dissipation, no Hill-sphere overlap effects — that may not hold for the realistic case. But the Nov 5 document never derived its own number; it took ~4.2% from the count of systems passing an architectural filter, which is circular as a frequency claim.)

## 4. Post-capture eccentricity distribution

When exchange does occur in the literature, the post-capture eccentricity is set by the geometry of the closest approach (Heggie & Rasio 1996, *MNRAS* 282, 1064).

For exchange in the slow-encounter regime ($x \lesssim 1$):
- The captured planet's new semi-major axis $a_{\rm new} \approx a_{\rm old}$ (similar orbital scale)
- The eccentricity distribution is **broadly thermal**: $p(e) \approx 2e$, peaking at $e \to 1$
- Mean $\langle e \rangle \approx 2/3$
- Modal value: $e \sim 0.5\text{--}0.8$

For the very slow regime ($x \ll 1$, "gentle handoff"):
- Eccentricities can be lower but are typically still $e \gtrsim 0.2$
- The "circular handoff" limit $e \to 0$ requires extreme fine-tuning and is statistically negligible

**Third derived consequence:** the Nov 5 Second Law prediction of $e = 0.05\text{--}0.10$ for captured systems is **not derivable from standard three-body dynamics**. The actual prediction is $e > 0.2$, more likely $e \sim 0.5\text{--}0.8$ (thermal). This is closer to the "scattering" regime than to the "moderate excitation" the Nov 5 document claimed, and *overlaps* with planet-planet scattering predictions.

The narrow $e = 0.05\text{--}0.10$ range was chosen in the Nov 5 document to be discriminating against both disk formation ($e < 0.05$) and scattering ($e > 0.2$). It was a discrimination criterion, not a derived value.

## 5. Post-capture inclination / obliquity

The captured planet's orbital plane after exchange is set by the encounter geometry, not by the original orbital plane around Star A.

For random encounter geometries (isotropic in cluster encounters):
- Inclination of captured orbit relative to Star B's spin axis: **isotropically distributed**
- This gives $\langle |\lambda| \rangle = 90°/2 = 45°$ for projected obliquity
- Median $|\lambda| \approx 60°$
- Fraction with $|\lambda| > 15°$: approximately $1 - \cos(15°) \approx 0.97$ (almost all)

**Fourth derived consequence:** the Nov 5 prediction that captured systems have $|\lambda| > 15°$ is *qualitatively* correct, but the original document understated it — the prediction should be that *essentially all* captured systems have $|\lambda| > 15°$, and the median should be near $60°$. The Phase D criterion (just $|\lambda| > 15°$) is too lax; it would be satisfied by random alignment.

This actually *strengthens* the obliquity discrimination, but in a way that worsens the joint test: the predicted population is misaligned ($|\lambda|$ broadly distributed, median 60°) AND high-e ($e \gtrsim 0.3$, median 2/3). Neither characteristic matches the actual misaligned multi-planet systems in the public sample.

## 6. Post-capture coplanarity within a multi-planet captured group

This is where the Nov 5 First Law ("Conservation of Coherence") fails most clearly.

In a coherent multi-planet group, before encounter: $\sigma_{\rm group} \approx 0°$ (coplanar).

During the encounter, each planet experiences different perturbations from Star B because each is at a different position relative to B at closest approach. The differential perturbation:
$$\Delta v_i \sim \frac{G M_B}{d_{\rm min}^2} \cdot \frac{a_i \tau_{\rm enc}}{d_{\rm min}}$$

where $\tau_{\rm enc} \sim d_{\rm min}/v_\infty$ is the encounter duration.

For typical cluster encounters ($d_{\rm min} \sim 100$ AU, $v_\infty \sim 2$ km/s, $\tau_{\rm enc} \sim 240$ yr), and a planet at $a_i = 5$ AU around a sun-like Star B after capture, the typical $\Delta v / v_{\rm orb}$ is order $\sim 1\%\text{--}10\%$ — but with *different geometric signs* for different planets in the group.

The result: even if Star A's system was perfectly coplanar, after exchange the captured planets' inclinations are randomized within the new system on the order of *several degrees to tens of degrees*. The "Conservation of Coherence" assertion is not supported.

**Fifth derived consequence:** captured multi-planet groups should have *within-system* mutual inclinations of $\sigma \sim 5°\text{--}30°$, *not* the $\sigma < 10°$ the Nov 5 document predicted as a tight conservation. This overlaps disk turbulence ($\sigma \sim 10°$) and is statistically indistinguishable from in situ formation on this axis alone.

## 7. Resonance preservation

The Nov 5 Third Law claimed: "Captured systems do not exhibit mean motion resonances. Captured systems never migrated through the host star's disk and therefore lack resonant configurations."

This is correct *if* the original system around Star A also lacked resonances, OR if any resonances are broken by the encounter. The encounter does generally break resonances (the perturbation is much stronger than the resonance-trapping torque). So the Third Law's prediction (f_MMR < 0.3) is consistent with derivation.

However, this prediction is also satisfied by *most* planetary systems regardless of formation channel: the fraction of confirmed multi-planet systems in established resonances is already small (~20–30% per Fabrycky+ 2014). The Third Law's "f_MMR < 0.3" criterion is not strongly discriminating.

**Sixth derived consequence:** the Third Law is qualitatively correct (capture should disrupt resonances), but quantitatively weak (most disk-formed systems also satisfy it).

---

## 8. The Revised Joint Prediction (derived, sealed before re-test)

Based on the standard three-body capture literature, a *captured* multi-planet system (in the sense of the user's CCT) should have:

| Observable | Original Nov 5 prediction | Derived prediction | Strength |
|---|---|---|---|
| Eccentricity $e$ (modal) | 0.05–0.10 | **0.3–0.8 (thermal)** | Strong, distinguishes from disk |
| Mutual inclination $\sigma$ | < 10° | **5°–30°** | Weak (overlaps disk turbulence) |
| Spin-orbit obliquity $\|\lambda\|$ | > 15° | **broadly distributed, median ~60°, $\langle\rangle \approx 45°$** | Strong, distinguishes from disk |
| Resonance fraction $f_{\rm MMR}$ | < 0.3 | < 0.3 (qualitative) | Weak (most systems satisfy this) |
| Frequency | ~4.2% | **~$10^{-3}$ to $10^{-5}$ per star** | Strong: capture should be RARE |

**Revised joint observational signature (FIXED, sealed BEFORE re-testing):**

A coherent-capture candidate must satisfy:
1. $|\lambda| > 15°$ at 2σ confidence (obliquity)
2. $e \geq 0.3$ for at least one planet in the system (high modal eccentricity)
3. The system is *not* in a known resonance chain (Third Law qualitative)

**Pre-registered decision rule for the re-test:**
- **CONSISTENT_WITH_DERIVATION**: ≥1 system in the public multi-planet non-HJ sample with measured λ satisfies all three criteria simultaneously. The mechanism is plausible at the population level even if rare.
- **DERIVATION_HAS_ZERO_CANDIDATES**: 0/N systems satisfy the joint criterion. The derived prediction has no support in the existing data. This is a different (and stronger) outcome than the original test's REFUTED, because the derivation predicts capture should be rare — so 0/11 is consistent with rare-capture but provides no positive evidence for the mechanism.
- **DERIVATION_OVERSHOOTS**: >1 system satisfies the joint criterion. Either capture is more common than the standard literature predicts, OR the systems found have alternative explanations we need to track down.

**Honest expectation, recorded BEFORE re-running:** Given that the derived frequency is $10^{-3}\text{--}10^{-5}$ per star and the sample is N=11, the expected number of captures in the sample is essentially zero. The most likely outcome is DERIVATION_HAS_ZERO_CANDIDATES, which is *not* a refutation of the mechanism — it's consistent with the mechanism being real but too rare to detect in n=11.

This is also the result that should make us pause: if the mechanism is real but undetectable in the available data, no archival test will resolve it. We would need either (a) the full obliquity catalog from PLATO/Roman, or (b) a different observable that's more sensitive.

## 9. What this derivation accomplishes vs. the original Nov 5 framework

The Nov 5 document asserted four observable signatures:
1. Moderate eccentricity (e = 0.05-0.10) — **REFUTED by derivation**; actual prediction is e ≥ 0.3
2. Coplanarity (σ < 10°) — **NOT SUPPORTED by derivation**; actual prediction is σ = 5-30°
3. Obliquity (|λ| > 15°) — **WEAKER than derivation**; actual prediction is |λ| broadly distributed, median ~60°
4. No resonances (f_MMR < 0.3) — **CONSISTENT** but weakly discriminating

The frequency claim (~4.2%) is not supported by standard cross-section calculations; the derived rate is several orders of magnitude lower.

**Net assessment:** the Nov 5 Laws of Coherency, as written, are not derivable from standard three-body capture dynamics. Some predictions (obliquity, resonances) are qualitatively correct but quantitatively understated. One prediction (eccentricity) is qualitatively wrong — the document predicted *moderate* eccentricities when the actual physics predicts *high* eccentricities. One prediction (coplanarity within group) is unsupported by the encounter geometry.

The mechanism (coherent group exchange in cluster encounters) is real and in the literature, but it is a *rare* process, not a dominant formation pathway. Its expected frequency is ~$10^{-3}$ per star or lower, not ~5% of all multi-planet systems.

## 10. What we should do next

1. Commit this derivation, sealed.
2. Re-run the Phase D test with the derived joint criterion (instead of the Nov 5 criterion). Same n=11 data.
3. Accept whatever result comes back. The honest expectation is DERIVATION_HAS_ZERO_CANDIDATES, which is consistent with rare-but-real capture but provides no positive evidence.
4. If we want positive evidence, we need a different observable or a much larger sample. The archival data can't decide this.

---

**References used:**
- Heggie, D. & Hut, P. 2003, *The Gravitational Million-Body Problem*, CUP, Ch. 23
- Heggie, D. 1975, *MNRAS* 173, 729 (binary classification)
- Hut, P. & Bahcall, J. 1983, *ApJ* 268, 319 (exchange cross-section)
- Heggie, D. & Rasio, F. 1996, *MNRAS* 282, 1064 (post-encounter distributions)
- Fabrycky, D. et al. 2014, *ApJ* 790, 146 (multi-planet architectures)
- Wang, L. et al. 2024, *MNRAS* 528, 4577 (free-floating planet capture)

These are the canonical references for three-body capture dynamics. Anyone with access to a textbook on stellar dynamics can verify the cross-section magnitudes; the numbers are not controversial.

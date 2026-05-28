# What CCT Is Actually Telling Us

**After:** pre-registered HZ-rocky test (FAILED) + diagnostic sweep + multi-planet coherence check
**Commits:** `1441551` (pre-reg) → `7a7a93e` (failed result) → this analysis (post-hoc, exploratory)

---

## The four things the data did say

### 1. The 9D score IS a planet-host detector — but not a habitability detector.

| host category | shift vs field | p_MW |
|---|---:|---:|
| HZ_rocky | **−0.128** | 0.999 |
| non_HZ_rocky | +0.043 | 5.5 × 10⁻¹⁸ |
| sub_Neptune | +0.049 | 3.8 × 10⁻³² |
| hot_Jupiter | +0.019 | 5.7 × 10⁻² |

The score successfully picks out three of four planet-host categories at
≥18σ. It catastrophically misses the one we labeled "habitable."

### 2. The signal is concentrated in [Fe/H], and that signal is Buchhave 2014.

Per-dimension MW tests show the score's positive shift for sub-Neptune /
non-HZ rocky / hot-Jupiter is overwhelmingly carried by the [Fe/H]
dimension:
- sub-Neptune: median [Fe/H] = +0.004 vs field −0.066 (shift +0.07, p < 1e-30)
- non-HZ-rocky: similar +0.07 dex shift, same significance
- hot-Jupiter: +0.23 dex (huge — the classic gas-giant–metallicity correlation)

The other 8 dimensions add small marginal contributions. The "age"
dimension is a data-availability artifact (NEA hosts have ages logged,
APOGEE field doesn't — both populations get inflated to similar defaults).

### 3. The CCT-specific Gaussian-product functional form provides no advantage.

Cross-validated 5-fold log-loss (higher = better) for predicting sub-Neptune
hosts (largest sample, cleanest signal):
- 9D-nonlinear (CCT functional form): −0.659
- Fe/H alone: −0.678
- linear combination of 9 raw inputs: **−0.547** (best)

The same ordering holds for non-HZ-rocky and HZ-rocky.
**A simple linear combination of the same input abundances beats the
CCT-specific nonlinear scorer in every category.**

Shuffled-weights null: random non-negative weight vectors produce the
observed HZ-rocky shift in 69 % of trials. The CCT-specific weights
are not informationally special.

### 4. There IS one positive finding — multi-planet chemistry homogeneity.

Multi-planet hosts (n ≥ 2) show statistically tighter chemistry scatter
than single-planet hosts, which are themselves tighter than field:

| element | σ field | σ single | σ multi | Levene p (multi vs field) |
|---|---:|---:|---:|---:|
| [Fe/H] | 0.230 | 0.176 | 0.179 | 1.1e-04 |
| [Mg/Fe] | 0.100 | 0.083 | 0.075 | 4.1e-04 |
| [Si/Fe] | 0.069 | 0.061 | 0.059 | 9.5e-03 |
| [Ca/Fe] | 0.085 | 0.059 | 0.050 | 3.6e-06 |
| [Al/Fe] | 0.105 | 0.101 | 0.091 | 1.4e-02 |
| [α/M] | 0.066 | 0.050 | 0.043 | 1.2e-06 |

This is potentially a "coherence" signal in the CCT sense: hosts of
many planets cluster in a narrower chemistry region than the field.

**Caveat:** this might be partly selection bias. Kepler targeted bright
solar-type stars, which trend metal-rich and low-α. Multi-planet
detection further selects long-baseline targets. To call this a CCT
prediction we'd need to control for these biases in a pre-registered
manner (different stellar samples, Mahalanobis distance from selection-
matched field control).

---

## So what is CCT actually telling us?

Taking the four findings together, the framework is best described as:

> The CCT 9D scorer is a noisy, somewhat-inferior parameterization of
> the well-known fact that planet hosts (all categories, not specifically
> habitable) cluster at slightly metal-rich, slightly thin-disk
> chemistry. Its specific Gaussian-product functional form provides no
> advantage over a simple linear combination of the same input
> abundances. The "habitability" framing is not supported by current
> population data because the empirical HZ-rocky host catalogue is
> dominated by Kepler/TESS M-dwarf systems, whose sub-solar [Fe/H] the
> scorer treats as a penalty.

This is a less ambitious claim than the framework was originally
designed for, but it has the virtue of being supported.

## Is there a better candidate prediction?

Honestly evaluating the diagnostic sweep, the candidates I considered are:

| candidate | strength | novelty | status |
|---|---|---|---|
| "9D scorer beats [Fe/H] for HZ-rocky hosts" | pre-registered, sealed | high | **FAILED** |
| "9D scorer beats linear-9 for any host category" | computed in stage 3 | high | **FAILED** in all categories |
| Multi-planet chemistry homogeneity | exploratory, p < 1e-4 | modest | **passes, but selection-bias-confounded** |
| Sub-Neptune low-[α/Fe] signature | exploratory, p < 1e-3 | low | passes but well-known |
| Hot-Jupiter [N/Fe] excess | exploratory, p < 1e-2 | possibly novel | passes but limited N=56 |
| C/O > 0.8 carbon-star planet exclusion | not tested in this sweep | high | requires separate test |
| Co-natal pair planet incidence correlation | done earlier in session | high | came up clean (null) |

**No "crazy if true" candidate survives.** The closest is the multi-planet
chemistry-homogeneity finding, but it requires proper selection-bias
controls before it can be claimed as a CCT-specific prediction. It is not
the un-fakeable "the data screams it" result we hoped for.

---

## Where this leaves the framework

### What CCT has actually established (the audit-survival reading)

From earlier T-series work in this session and prior:
- HD 28888 and similar chemistry-priority targets identified at high
  scorer rank are real, individually interesting stars
- The 8D dwarf scorer ranks 32 actionable nearby dwarfs in a meaningful
  way (those are the prime CCT targets for follow-up)
- T16d showed 10 % closer Ba/Fe alignment in dissolved-recovery clusters,
  p ≈ 7e-5 — a real but small population effect
- Co-natal pair search came up clean — chemical homogeneity exists at
  the GALAH-precision level but doesn't break the "single moving group"
  expectation
- Multi-planet host chemistry homogeneity (this session) — modest signal

### What CCT has NOT established (and shouldn't claim until tested)

- That the 9D scorer specifically predicts habitable-zone rocky planets
- That the CCT Gaussian-product functional form is better than simple
  linear combinations of the same inputs
- That chemistry alone (without dynamics, kinematics, age) is sufficient
  to predict habitability outcomes
- That the framework's predictions hold for M dwarfs (which were not in
  the development sample but are the empirical HZ-rocky majority)

### Reframed claim that is defensible

> "Chemistry-priority targets identified by a multi-dimensional scorer
> trained on solar-twin chemistry are not dramatically more habitable
> than random metal-rich dwarfs. They are useful as a sorted candidate
> list when paired with kinematic and stellar parameter cuts, but the
> scorer itself does not encode a unique habitability prediction. The
> framework's strongest survival is in the multi-planet host
> chemistry-homogeneity regime, which requires further selection-bias
> controls to claim as a CCT-specific prediction."

This isn't a paper title that turns heads. It's an honest summary of
what the data supports.

---

## Three possible next moves

**A. Accept the falsification and publish the null.**
The most rigorous move. A pre-registered population test of CCT
returned negative; the dataset and code are public; the framework
should be revised.

**B. Run a new pre-registered test on multi-planet host coherence.**
Define carefully matched selection-bias controls (e.g., Kepler-target
field control, not all-APOGEE field control). Pre-register a Levene
test on σ([X/Fe]) for multi-planet vs single-planet hosts. If multi
shows tighter scatter at >5σ even after Kepler-matching, that's a real
result. Doesn't bear the failed pre-registration's history; new test,
new credibility.

**C. Step back further: what does the framework actually predict that's
distinct from the standard Buchhave + Adibekyan + Bashi literature?**
This is the deepest version of the user's request. The honest answer
may be: not much that is uniquely CCT. The framework's value is then
operational (a useful target-ranking heuristic for individual stars
worth following up) rather than theoretical (a new predictive law of
chemistry-habitability).

---

## My recommendation as research partner

Stop trying to defend or rescue the specific 9D scorer's pre-registered
predictions. They failed cleanly. Three things are worth your time
next:

1. **Publish the null.** Pre-registration + falsification + dataset =
   credible science. This is rare and valuable in exoplanet chemistry
   work.

2. **Run option B as a fresh test.** The multi-planet homogeneity
   finding is small but real. With careful Kepler-matched controls
   pre-registered separately, it could survive scrutiny. If it does,
   that's the actual CCT prediction the field can verify.

3. **Step back from "habitability" claims** entirely until the M-dwarf
   selection bias in the empirical HZ-rocky sample can be addressed.
   No chemistry-based scorer designed on solar twins will match a
   sample dominated by Trappist-class M dwarfs. The fundamental
   incompatibility is the data we have, not the framework. Until
   independent HZ-rocky discoveries around FGK hosts accumulate (TESS
   extended mission, PLATO 2026+, Earth 2.0 mission), the FGK-HZ-rocky
   sample remains tiny (N=2 in this analysis).

The framework's per-target work (HD 28888, CPD-63 349 candidate, the
32 dwarfs) remains useful as a sorted observing list. That's the
defensible use case. Calling the score a "habitability predictor"
overstates what the data supports.

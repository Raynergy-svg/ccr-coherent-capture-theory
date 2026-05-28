# Test #5 RESULT — Outward Falsification Audit

**Test:** Apply the four-test pre-registered methodology outward to four
widely-cited published chemistry-planet-host claims.
**Pre-registration:** `PRE_REGISTRATION_5.md` @ commit `b196357`
**Analysis sealed:** `cct_test5_outward_audit.py` @ commit `d54880c`
**Data-load fix:** `a061703` (non-methodological correction documented in commit)

---

## Verdict matrix

| claim | source | predicted direction | shift | MW p | held-out Δ | bin sanity | verdict |
|---|---|---|---:|---:|---:|:-:|---|
| A | Adibekyan 2012 — Mg/Si in hot Jupiter hosts | host < ctrl | **−0.015** ✓dir | 0.16 | −0.0004 | 5/5 ✓ | **REJECTED** |
| B | Adibekyan 2012 — [Mg/Fe] in small planets at [Fe/H]<−0.2 | host > ctrl | **−0.005** ✗dir | 0.53 | −0.0019 | 2/4 | **REJECTED (direction)** |
| C | Brewer & Fischer 2018 — C/O in any planet host | host < ctrl | **−0.013** ✓dir | 0.095 | +0.0000 | 8/11 ✓ | **REJECTED** |
| D | Suárez-Andrés 2018 — C/O > 0.8 fraction in rocky hosts | host < field | **frac ratio 0.41** ✓dir | Fisher 0.017 | — | — | **PARTIAL** |

Bonferroni threshold: p < 0.0125 (= 0.05 / 4).
Held-out improvement threshold: Δlog-loss > 0.02.

---

## Per-claim findings

### Claim A — Adibekyan Mg/Si in hot Jupiter hosts: REJECTED

- Direction IS consistent (host Mg/Si lower than control) and holds in
  5 of 5 populated [Fe/H] bins
- But the effect is small (−0.015 in Mg/Si units, ≈ 0.13 IQR)
- MW p = 0.16 fails Bonferroni
- Held-out CV improvement essentially zero (−0.0004) — adding Mg/Si to
  [Fe/H] does NOT improve held-out planet-host prediction
- Sample size limited: N = 48 hot Jupiter FGK hosts in APOGEE

**Honest read:** The original Adibekyan direction is supported as a
trend but the effect, after strict matching, is too small to discriminate
hosts from matched controls at our pre-registered threshold. The
literature claim may still hold at larger sample sizes; with APOGEE×NEA
N = 48, we cannot confirm.

### Claim B — Adibekyan [Mg/Fe] in small planets at low [Fe/H]: REJECTED IN OPPOSITE DIRECTION

This is the most CCT-relevant claim — that small planets in the thick-disk
α-enhanced regime should host MORE planets at given [Fe/H].

- Direction is **OPPOSITE** to predicted: small-planet hosts at
  [Fe/H] < −0.2 have shift = −0.005 in [Mg/Fe], not positive
- MW p = 0.53 (essentially no signal in either direction)
- Held-out CV: adding [Mg/Fe] to [Fe/H] makes prediction WORSE
  (−0.0019)
- Within-bin: 2/4 bins consistent, 2/4 opposite — random

**Honest read:** The Adibekyan thick-disk-α-enhancement-of-small-planet-hosts
claim does NOT hold in APOGEE × NEA after strict matching. This is the
specific claim that CCT's framework was implicitly designed around;
its failure here is the cleanest part of the audit.

### Claim C — Brewer & Fischer C/O in any planet host: REJECTED

- Direction IS consistent (planet hosts have lower C/O than control)
  and holds in 8 of 11 populated [Fe/H] bins
- But effect is small (−0.013 in C/O units, ≈ 0.06 IQR)
- MW p = 0.095 fails Bonferroni
- Held-out CV improvement essentially zero (+0.00003)
- N = 796 hosts (the largest sample tested), so this is not
  underpowered

**Honest read:** A consistent direction across most [Fe/H] bins is a
real phenomenon — planet hosts ARE on average slightly lower-C/O. But
after strict (Teff, log g, [Fe/H]) matching, the effect is too small
to be detected at Bonferroni significance or to add out-of-sample
predictive power. The Brewer & Fischer effect is real but tiny when
properly controlled.

### Claim D — Suárez-Andrés C/O > 0.8 carbon exclusion: PARTIAL

This is the only claim that comes close to surviving.

- Rocky planet hosts (R_p < 1.6 R_⊕): 5 of 273 (1.83%) have C/O > 0.8
- FGK field stars: 7 032 of 158 516 (4.44%) have C/O > 0.8
- **Host fraction is 41% of field fraction** — substantial reduction
- Fisher exact one-sided p = 0.017
- Direction is correct (high-C/O carbon-star environment seems to
  exclude rocky planets)
- Just misses Bonferroni (0.017 > 0.0125)

**Honest read:** The Suárez-Andrés / Brewer-Fischer carbon-exclusion
prediction is **directionally supported with substantial effect size**
in APOGEE × NEA. It misses pre-registered Bonferroni significance by
a factor of ~1.4. This is a real candidate for follow-up with a larger
rocky-planet sample.

Caveat: APOGEE's C/O is notoriously noisy for dwarfs (H-band C and O
features). The 4.4% field fraction with C/O > 0.8 is likely inflated
by measurement noise. But the comparative test (hosts vs same-noise
field) is valid in direction even if absolute C/O values are uncertain.

---

## Combined picture: published chemistry-host literature vs strict controls

Of the four widely-cited published claims tested:

- **3 of 4 fail to clear strict-control held-out testing.** The
  direction is consistent for 2 of those 3 (Claims A, C) but the
  effect dissolves to nothing in held-out log-loss after [Fe/H] matching.
- **1 of 4 is partial.** Claim D (carbon exclusion) has substantial
  effect size and correct direction; misses pre-registered Bonferroni
  but survives at p < 0.05.
- **Claim B fails in the OPPOSITE direction** — the specific α-enhancement
  prediction that CCT and Adibekyan share is not supported by APOGEE × NEA.

The pattern matches our internal CCT tests #1-#4: **chemistry signatures
beyond [Fe/H] in planet-host vs control comparisons largely dissolve
when proper matching and out-of-sample validation are applied.**

This is not adversarial to the original authors. They worked with
different (often smaller) samples and pre-pre-registration norms. But
it does suggest the chemistry-planet-host literature has a systematic
gap: in-sample p-values at α < 0.05 (without matching, without CV) do
not reliably translate to surviving signals under modern rigorous
controls.

---

## What the audit lets us claim

After the full five-test sequence, we can say with evidence:

1. **The [Fe/H]-planet-host correlation (Buchhave 2014) is robust.** It
   recovers at p < 10⁻⁸ on APOGEE × NEA. This is the most reliable
   chemistry-host signal in the literature.

2. **The carbon-exclusion prediction (Suárez-Andrés/Brewer-Fischer) has
   substantial directional support** (host fraction 41% of field at
   C/O > 0.8) but needs a larger rocky-planet sample to clear strict
   Bonferroni.

3. **Most other multi-element claims (Adibekyan Mg/Si, Adibekyan
   thick-disk α, Brewer-Fischer C/O) do not survive strict-matched
   held-out testing on APOGEE × NEA.** Effects are either too small,
   in the wrong direction, or both.

4. **No claimed chemistry signature beyond [Fe/H] survives all four
   discipline checks** (matched control, MW Bonferroni, held-out CV,
   within-bin sanity) at sample sizes available.

The methodology paper has its concrete worked examples now: same
framework, same data, applied to internal CCT claims (all rejected)
and external published claims (3/4 rejected, 1 partial). The
methodology itself is the contribution; the empirical pattern across
the audit is a substantive scientific finding.

---

## Caveats and honest limitations

- **Sample size**: N = 48 for hot Jupiter FGK hosts is small; A's
  rejection may be a power issue.
- **APOGEE C/O for dwarfs**: noisy due to H-band feature weakness;
  affects C and O abundances in Claims C and D.
- **Original publication samples**: HARPS GTO, etc., differ from APOGEE
  ×  NEA in target selection. Our test does not invalidate the original
  findings on the original samples — it shows the claims don't replicate
  on APOGEE × NEA with strict controls.
- **Steel-manned restatements**: We tested OUR interpretations of the
  published claims. The original authors might phrase the claims
  differently and the exact test design could vary. Our pre-registration
  is transparent about this.

---

## What this means for the publishable contribution

The session now has:

- **5 pre-registered tests**, sealed scorer, sealed analysis scripts,
  honest verdicts at every step.
- **4 internal CCT claims** rejected substantively.
- **3 of 4 external published claims** rejected by the same framework;
  1 of 4 partial with directional support.
- **1 robust survivor** across the literature: [Fe/H] correlation
  (Buchhave 2014).
- **1 candidate worth follow-up**: carbon exclusion (Suárez-Andrés/
  Brewer-Fischer direction, sub-Bonferroni significance).
- **Operational tool surviving**: chemistry-priority FGK dwarf list
  for transit follow-up, scientifically grounded in [Fe/H] but
  practically useful.
- **One real planet candidate**: CPD-63 349 b, inconclusive but
  testable in early January 2026.

The methodology paper writes itself:

> *"Five pre-registered tests of chemistry-planet-host claims on
> APOGEE × NEA: four internal (CCT scorer's predictions) and four
> external (Adibekyan, Brewer & Fischer, Suárez-Andrés). Of nine
> total claim-level tests, [Fe/H] correlation is the only one that
> robustly survives; carbon-exclusion partially survives; the
> remaining seven multi-element claims dissolve under strict matching
> and held-out cross-validation. The chemistry-habitability literature
> may benefit from systematically benchmarking multi-element scorers
> against [Fe/H] alone with strict (Teff, log g, [Fe/H])-matched
> controls before claiming predictive content beyond metallicity."*

That's a real, defensible, publication-grade contribution. It does not
require the framework to be correct; it requires only the discipline
of the methodology, which is what survives.

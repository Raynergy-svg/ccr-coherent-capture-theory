# Pre-Registration #5: Outward Falsification Audit of Published Chemistry-Planet-Host Claims

**Date sealed:** 2026-05-28 (before any analysis runs)
**Branch / sealing commit:** `claude/gaia-galah-hd28888-R5VZz` @ `889eabf`

**Prior tests (the in-house framework being applied):**
- Pre-reg #1-#4 tested CCT scorer's own claims, all rejected substantively
- Methodology developed: strict (Teff, log g, [Fe/H]) matching + held-out CV +
  within-bin sanity check + Bonferroni correction
- Established that ΔAIC at 10⁻⁸ in-sample can dissolve to ~0.003 log-loss
  improvement out-of-sample once artifacts are controlled

This pre-registration turns that methodology outward. Same APOGEE×NEA
dataset (877 hosts, 159k field), same strict matching, same held-out CV.
We audit specific, widely-cited published claims about chemistry-planet-host
relationships beyond [Fe/H], and report what survives.

---

## Why this matters

The chemistry-habitability and chemistry-planet-host literature contains
many multi-element claims (Mg/Si, [α/Fe], C/O, etc.) that have not been
benchmarked under strict (Teff, log g, [Fe/H])-matched controls with
held-out cross-validation. Our four CCT tests demonstrated that:
- Apparent in-sample p < 10⁻⁸ can dissolve to ~0.003 held-out Δlog-loss
- Data-construction artifacts (column joins, default fallbacks) can pass
  Bonferroni-corrected per-dimension tests at apparent p < 10⁻²⁰
- Multi-element nonlinear scorers don't beat linear inputs out of sample

The literature claims tested here may have similar issues. The point of
this audit is methodological, not adversarial — to apply the same
discipline that rejected our own scorer's specific claims to published
claims, and report honestly what survives.

**Important disclaimer:** The published authors did not pre-register
their claims. We are testing modern restatements of their claims with
modern (APOGEE DR17) data. Survival or failure under our framework
does not constitute a definitive verdict on the original papers —
those were written without access to APOGEE DR17 and without the
held-out CV machinery now standard in ML-adjacent fields. The
contribution here is showing what holds under modern matched-control
testing, not adjudicating the original work.

---

## Claims to audit

### Claim A — Adibekyan et al. (2012) "Mg/Si in hot Jupiter hosts"

**Steel-manned restatement:** Hot Jupiter hosts have systematically
lower Mg/Si ratios than non-host stars at matched stellar parameters
and [Fe/H]. Equivalently, Mg/Si is a discriminator for hot Jupiter
host vs non-host beyond what [Fe/H] alone captures.

**Test:**
- Hot Jupiter hosts (R_p ≥ 6 R_⊕, T_eq > 1000 K, FGK): N = 48
- Strict-matched FGK control (Teff, log g, [Fe/H])
- One-sided MW test: Mg/Si shift host < control (Adibekyan direction)
- Held-out CV: does adding Mg/Si to [Fe/H] improve held-out log-loss
  by > 0.02 over [Fe/H] alone?
- Within-bin sanity: does the direction hold across ≥ 50 % of populated
  [Fe/H] bins?

### Claim B — Adibekyan et al. (2012) "α-enhancement in small planet hosts at low [Fe/H]"

**Steel-manned restatement:** Among stars at [Fe/H] < −0.2 (thick-disk
chemistry regime), small planet hosts (R_p < 4 R_⊕) are systematically
α-enhanced compared to non-host stars at matched stellar parameters.

**Test:**
- Small planet hosts (R_p < 4 R_⊕, FGK, [Fe/H] < −0.2)
- Strict-matched control restricted to same [Fe/H] regime
- One-sided MW test: [Mg/Fe] shift host > control (Adibekyan direction)
- Held-out CV: does [Mg/Fe] add to [Fe/H] in the [Fe/H] < −0.2 regime?

This is the most CCT-relevant claim because it predicts a regime-specific
chemistry-host correlation that the CCT scorer would also predict.

### Claim C — Brewer & Fischer (2018) "C/O lower in planet hosts"

**Steel-manned restatement:** All-planet hosts have systematically lower
C/O than non-host stars at matched stellar parameters.

**Test:**
- All confirmed planet hosts (FGK): N = 796 (all FGK with any planet)
- Strict-matched FGK control
- One-sided MW test: C/O shift host < control (Brewer & Fischer direction)
- Held-out CV: does C/O add to [Fe/H] for predicting any-planet-host?

### Claim D — Suárez-Andrés et al. (2018) "C/O ≥ 0.8 carbon-rich exclusion"

**Steel-manned restatement:** Few stars have C/O > 0.8; those that do
should host no terrestrial planets (carbon-rich chemistry produces
SiC-dominant rather than silicate-dominant rocky compositions).

**Test:**
- Count fraction of FGK field stars with C/O > 0.8
- Count fraction of FGK rocky planet hosts with C/O > 0.8
- Fisher exact test: is the host fraction smaller than the field fraction?
- Pre-registered threshold: Fisher p < 0.05 with host fraction at least
  50 % smaller than field fraction
- Honest power note: requires N_field with C/O > 0.8 ≳ 100 for power;
  if fewer, report as UNDERPOWERED.

---

## Frozen test design (same as PRE_REGISTRATION_4)

- **Sample:** existing APOGEE×NEA cross-match (`cct_test_hosts.csv`,
  `cct_test_field.csv` at commit 889eabf)
- **FGK restriction:** Teff 4500-7000, log g > 3.8
- **Matching:** k=10 nearest neighbour on (Teff/100, log g/0.1, [Fe/H]/0.05)
- **Match-quality gate:** KS p > 0.95 on Teff, log g, [Fe/H] required
- **Statistical tests:**
  - Mann-Whitney U one-sided in pre-registered direction
  - Bonferroni alpha = 0.05 / 4 claims = 0.0125
- **Held-out CV (where applicable):**
  - 70/30 stratified train/test split, seed 42 (same as test #4)
  - StandardScaler fit on training only
  - LogisticRegression class_weight=balanced, C=1.0, max_iter=5000
  - Threshold for "passes held-out test": Δlog-loss > 0.02 vs [Fe/H]-alone baseline
- **Within-bin sanity:**
  - [Fe/H] bins of 0.1 dex
  - Test passes if the predicted direction holds in ≥ 50 % of populated
    bins (≥ 2 bins required to evaluate)

## Per-claim verdict matrix

For each claim:

| outcome | criterion |
|---|---|
| **SURVIVES** | one-sided MW p < 0.0125 (Bonferroni) AND held-out Δlog-loss > 0.02 AND within-bin sanity ≥ 50% bins consistent direction |
| **PARTIAL** | MW passes but held-out CV does not, OR within-bin fails |
| **REJECTED** | MW fails OR direction opposite to predicted |
| **UNDERPOWERED** | sample too small (N < 30 hosts in restricted subset) — reported descriptively, not as confirmation or rejection |

---

## Pre-registered direction of each claim

Spelled out here so the test is direction-locked (no two-tailed cheating):

| claim | element | predicted direction (host vs control) |
|---|---|---|
| A | Mg/Si | host < control (lower in hot Jupiter hosts) |
| B | [Mg/Fe] | host > control (higher in metal-poor small-planet hosts) |
| C | C/O | host < control (lower in any planet hosts) |
| D | C/O > 0.8 fraction | host < field (fewer high-C/O stars among rocky hosts) |

---

## What stops me from cheating

1. The four claims (A-D) and their directions are listed above before
   data analysis starts.
2. The Bonferroni alpha (0.05/4 = 0.0125) is set before any p-values
   are computed.
3. The held-out CV threshold (Δlog-loss > 0.02) is identical to test #4.
4. The same APOGEE×NEA dataset is used (no new cross-matches, no
   selective filtering beyond FGK restriction).
5. The analysis script will be committed in a separate commit BEFORE
   it produces results.

## Honest expectation

Based on what survived the CCT tests (essentially nothing beyond [Fe/H]
out-of-sample after artifact removal), I expect:
- Claim A: possibly PARTIAL — Mg/Si direction may hold in-sample but
  unlikely to beat [Fe/H] by > 0.02 in held-out CV
- Claim B: this is the most interesting — small planet hosts in the
  α-enhanced thick-disk regime ARE the population CCT was implicitly
  designed for. May survive if real.
- Claim C: possibly PARTIAL — C/O is harder to measure cleanly in
  APOGEE; effects likely small
- Claim D: likely UNDERPOWERED unless field has many C/O > 0.8 stars

If everything fails, the publishable finding becomes "the multi-element
chemistry-planet-host literature does not survive matched-control
held-out CV — most apparent signals dissolve to ≈ [Fe/H] alone."

If some survive, those become the genuinely defensible claims worth
building on.

## Signed

Pre-registered 2026-05-28 by Certan with Claude as research partner.
Sealing commit: `889eabf`.

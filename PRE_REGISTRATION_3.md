# Pre-Registration #3: HZ-Rocky Test Restricted to Scorer's Own Regime (FGK)

**Date sealed:** 2026-05-28 (immediately, before re-running the test)
**Branch / sealing commit:** `claude/gaia-galah-hd28888-R5VZz` @ `060803d`
**Frozen scorer:** `habitability_v2.py` @ commit `cfa1249` (unchanged from pre-reg #1)

**Prior tests:**
- Pre-reg #1 (`1441551`) → REJECTED at `7a7a93e` (HZ rocky shifted opposite direction)
- Pre-reg #2 (`1d58aa7`) → REJECTED at `060803d` (multi-planet coherence was [Fe/H] artifact)

This is a focused re-run of pre-reg #1, restricted to the regime where
the scorer was designed to operate.

---

## What this test asks

The scorer was developed on FGK dwarfs (see `habitability_v2.py` header:
"FGK dwarf filter: teff > 4000 K, teff < 7000 K, logg > 3.8"). The
pre-reg #1 sample inherited M dwarfs because the empirical HZ-rocky
catalogue is M-dwarf-dominated.

**The legitimate scientific question:** does the scorer predict HZ-rocky
hosts in the FGK regime where it was built?

If YES → original failure was sample mismatch; the scorer has real
predictive power for its design population.
If NO → genuine falsification within its own regime; the scorer's
predictions about chemistry-habitability fail even where it should work.

No new dimensions added. No weight tuning. Same `habitability_v2.py`,
same Gaussian functional form, same `hab_score` formula. Only change:
the sample restricted to FGK hosts and FGK matched field control.

This is not "rescuing" the framework — it's testing it cleanly in its
stated regime.

---

## Hypothesis under test

**H1 (CCT-FGK):** Confirmed HZ-rocky planet hosts that are FGK dwarfs
(Teff 4500-7000, log g > 3.8) score systematically higher on the
frozen 9D scorer than a (Teff, log g, [Fe/H])-matched FGK field
control, at significance corrected for prior failed pre-registrations.

**H0 (null):** No detectable shift between FGK HZ-rocky hosts and a
properly matched FGK field control, OR shift is in the opposite
(downward) direction.

---

## Frozen sample (sealed before re-running)

**FGK HZ-rocky hosts in APOGEE:** restricted from the 7 already
identified in pre-reg #1 to those satisfying Teff 4500-7000 and
log g > 3.8.

From the post-hoc diagnostic sweep (already run, committed at `80ab5b8`),
this is known to be **3 hosts**:
- Kepler-1126 (G5, Teff 5675)
- Kepler-442 (K4, Teff 4525)
- Kepler-62 (K2, Teff 4964)

The N=3 is small but it is the population the scorer claims to
predict. The test will be honest about power.

**Matched control:** k=10 nearest-neighbour APOGEE field stars per host,
matched in (Teff/100, log g/0.1, [Fe/H]/0.05). [Fe/H] included in
matching because pre-reg #2 showed [Fe/H] matching is essential to
isolate CCT-specific predictions from generic [Fe/H] correlation.

Field pool: APOGEE FGK dwarfs (Teff 4500-7000, log g > 3.8, SNR > 70)
not in any planet host list, [Fe/H] in [-2, 1].

**For context:** also test FGK non-HZ-rocky (n ≈ 271) and FGK
sub-Neptune (n ≈ 417) hosts vs their own strict-matched controls. This
checks whether the scorer fails specifically on HZ-rocky or
generically on small-planet hosts.

---

## Pre-registered test statistics

For each host category (HZ_rocky, non_HZ_rocky, sub_Neptune):

1. **Mann-Whitney U test** on `hab_score` between hosts and strict-
   matched control, one-sided "greater" (CCT prediction is upward shift).
2. **Effect size:** (median_host − median_match) / IQR_match.
3. **Permutation null:** 10⁴ shuffles of host/match labels, compute
   empirical p-value.
4. **Bonferroni correction:** α = 0.05 / 3 categories = 0.0167.
5. **Strong-confirmation threshold:** p < 1e-3 (3σ) for the HZ_rocky
   category specifically.

## Success criteria (frozen)

**CCT-FGK CONFIRMED iff:**
- HZ_rocky vs matched field: median shift > 0 AND MW-U p_one-sided < 0.0167 (Bonferroni 3σ).
- Direction holds across all permutations (perm-p < 0.05).
- Effect size > 0.3 (median shift > 0.3 × IQR_match).

**CCT-FGK CLEANLY REJECTED iff:**
- HZ_rocky vs matched field: median shift ≤ 0, OR
- MW-U p > 0.5 (suggesting random or opposite-direction).

**UNDERPOWERED iff:**
- HZ_rocky shift is in correct direction but p > 0.0167 with N=3 hosts.
  In this case the result is reported as "directionally consistent but
  not significant; need more FGK HZ-rocky discoveries."

The known result from the post-hoc diagnostic sweep (commit `80ab5b8`)
was: FGK HZ-rocky shift = −0.13, p_MW = 0.987. **If the formal pre-
registered test reproduces this, the verdict is CLEAN REJECTION even
in the scorer's design regime.** That's the strong falsification
outcome.

---

## What stops me from cheating

1. The post-hoc diagnostic sweep result is already publicly committed,
   so I cannot pretend to be surprised by it. The legitimacy of THIS
   test is in formalizing it under proper pre-registered controls
   (matched field control on [Fe/H], permutation null, Bonferroni).
2. The scorer hash (`cfa1249`) is unchanged. No new dimensions, no
   reweighting.
3. The 3 FGK HZ-rocky hosts are publicly known and named here. No
   substitution possible.
4. The expected outcome (per post-hoc finding) is REJECTION. This is
   not a sympathetic test — it's an explicit attempt to verify that
   the scorer fails even in its own regime.

---

## Honest acknowledgment

The result of THIS test is, by virtue of the diagnostic-sweep finding,
already strongly suggested: rejection. The point of running it as a
fresh pre-registered test is to:

(a) Make the falsification formal, with proper [Fe/H]-matched control
    that we lacked in pre-reg #1.
(b) Compare HZ-rocky FGK hosts against non-HZ-rocky and sub-Neptune
    FGK hosts in the same test, to see whether the scorer fails on
    HZ-rocky specifically or fails on all small-planet hosts equally.
(c) Establish that the falsification holds in the scorer's stated
    design regime, not just in M-dwarf-contaminated samples.

If somehow the test does pass — if the 3 FGK HZ-rocky hosts shift
upward against strict-matched controls — that would be a surprise
worth reporting. The pre-registration commits us to reporting
honestly either way.

---

## Sample-size and power note

N_HZ_rocky_FGK = 3 is small. Power analysis with strict-matched control
(n_match ≈ 30 effective for k=10):
- Detectable one-sided effect at p<0.0167 with N_pos=3: ~ +0.7 σ shift in hab_score
- Detectable at p<0.001: ~ +1.2 σ shift
- Observed (post-hoc): -1.1 σ shift -> p_MW will be >> 0.999

So the test has plenty of power to reject (the opposite-direction
shift is large). It does not have power to detect a subtle positive
shift (would need ~10+ FGK HZ-rocky hosts for that, which don't exist
in APOGEE).

## Signed

Pre-registered 2026-05-28 by Certan with Claude as research partner.
Sealing commit: `060803d`.

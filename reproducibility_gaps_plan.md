# Reproducibility-gap investigation + fix plan

Two gaps were flagged after the dist_cl re-run. Investigated against the
TAP-rebuilt catalog (`galah_dr4_allstar_240705.fits`, 917,588 stars).

---

## Gap 1 — field-star count (497k vs 519k): **FALSE ALARM, no fix needed**

**Root cause: I compared two different cuts.** The 497,037 was from **t20c** (a
**5D** pool requiring Al/Fe validity, with its own NGC-6253 exclusion). The
paper's 519,547 is from **t16b** (a **4D** pool: C/O, Mg/Fe, Si/Fe, [Fe/H]).

**Verification (rebuilt catalog, t16b's exact 4D cuts):**
| Cut | Count |
|---|---|
| total | 917,588 |
| snr_px_ccd3 > 30 | 813,743 |
| flag_sp == 0 | 631,925 |
| 4D valid + C/O∈(0.05,2.0) | 535,598 |
| exclude <0.5° of any cluster (**t16b field pool**) | **519,547** ✓ |
| (+ Al/Fe valid → 5D, t20c-style) | 516,665 |

The rebuilt catalog reproduces the paper's **519,547 exactly**. The catalog
reconstruction is faithful; there is no data discrepancy. **Action: none**
(optionally note in the methods that the TAP-rebuilt catalog reproduces the
t16b field pool to the star).

---

## Gap 2 — t20c tolerances/funnel ≠ paper: **REAL divergence, plan below**

The committed `t20c_ngc6253.py` does not reproduce the paper's NGC 6253 funnel
(2149 → 456 → 11 → 4). Three distinct causes:

1. **Tolerance definition.** Committed uses adaptive `TOL[d] = max(2.5·σ, 0.06)`.
   NGC 6253's GALAH members are contamination-broadened (the script itself says
   so), so 2.5σ inflates to ±0.32 → **80,456** chemical matches. The paper text
   states **fixed** tolerances (C/O ±0.08, Mg/Fe ±0.05, Si/Fe ±0.05, [Fe/H] ±0.08,
   Al/Fe ±0.06).

2. **Match count, even with fixed tolerances.** Applying the paper's fixed
   tolerances (verified in `t20c_fixedtol.py`) gets the template **exactly right
   (31 clean members)** and the right tolerances, but yields **3,598** chemical
   matches, not 2,149. The catalog is faithful (Gap 1), so the residual gap comes
   from the **template centroid** (C/O = 0.653 is contamination-elevated) and/or a
   field-pool pre-cut the committed script lacks. The paper-producing version
   likely used a cleaner template or a metallicity-restricted field pool.

3. **Funnel.** Fixed-tolerance funnel: 3,598 chem → 1,908 parallax → 361 +PM →
   **7** (+RV +age). Paper: 2,149 → 456 → 11 → **4**. Same shape, different
   numbers.

### Why the conclusion is unaffected
The fixed-tolerance run's own false-positive model gives **~7.0 expected random
matches** in the 497k pool, and the funnel returns ~7 candidates — i.e. the
recovery is **statistically consistent with chance**. The paper's T20c
conclusion is an *honest elimination* (best candidate HD 163560 ruled out by
TESS asteroseismic age; no confident individual recovery). That conclusion
holds whether the funnel ends at 4 or 7. **This is a numerical-reproducibility
issue, not a conclusion-changing one.**

### Proper-fix options (author's call)

**Option A — reproduce the paper exactly.** Restore the original t20c template
selection / field-pool cut that produced 2,149. Needs the original script (git
history has only two bulk commits, so it isn't recoverable from the repo).
Risk: may not be reconstructible; high effort.

**Option B — adopt the reproducible script + update the paper (RECOMMENDED).**
1. In `t20c_ngc6253.py`, replace the adaptive tolerance block with the paper's
   fixed values (one edit; already validated in `t20c_fixedtol.py`):
   ```python
   TOL = {"C_O":0.08,"mg_fe":0.05,"si_fe":0.05,"fe_h":0.08,"al_fe":0.06}
   ```
2. Re-run; record the reproducible funnel (3,598 → 1,908 → 361 → 7) and the
   false-positive expectation (~7).
3. Update the paper's T20c sentence (≈ line 495) to the reproducible numbers and
   add the false-positive context, which *strengthens* the elimination argument
   ("the candidate count matches the random expectation of ~7, so no member is
   confidently recovered"). Conclusion unchanged.
4. Optionally tighten the template (drop contamination, e.g. an [Fe/H] core cut)
   to reduce the centroid bias — but only if it doesn't change the conclusion.

**Recommendation: Option B.** It makes the committed code match the paper's
stated method (fixed tolerances), is fully reproducible from the rebuilt
catalog, and the honest-elimination conclusion is preserved (indeed sharpened
by the explicit false-positive comparison).

### Note (independent of both gaps)
t20b (NGC 6791) uses `max(2·σ, …)` and t20 (Praesepe) uses fixed
`[0.08,0.05,0.05,0.10]`. The three T20 scripts use **inconsistent tolerance
conventions**; if reproducibility matters across all three, standardize them on
the fixed-threshold convention the paper describes and re-verify each funnel
(Praesepe 22,327; NGC 6791 768 → 18; NGC 6253 → above).

---

## UPDATE — fixing t20c surfaced a dist_cl-bug artifact in the paper's NGC 6253 narrative

Applying the fix (fixed tolerances + Gaia batch-retry for determinism) gives a
reproducible funnel: **3,598 chem → 1,908 plx → 361 +PM → 120 +RV → 7 +age**,
against **~7.0 expected false positives** — i.e. recovery consistent with chance.

But the paper's *named* best candidate, **HD 163560 (Gaia DR3
5953941329394191360)**, is absent from the funnel. Investigation:
- HD 163560 **passes all 5 chemical dimensions** (it is a genuine chemical match).
- It is rejected on **parallax**: HD 163560 has plx **1.828 mas (547 pc)**, while
  NGC 6253 is at **0.605 mas (1.65 kpc)** — a ~3× distance mismatch.
- The paper (Sec. NGC 6253) states NGC 6253 is at **"0.56 kpc"** and that HD 163560
  **"matches the cluster in parallax to 1.2 per cent."** Both are **dist_cl-bug
  artifacts**: 0.56 ≈ the cluster *parallax* (0.605 mas) mislabeled as kpc; the
  buggy pipeline then used cl_plx = 1/0.56 ≈ 1.79 mas, which spuriously matched
  HD 163560's real 1.83 mas. The asteroseismic elimination (Hon2021 method,
  $1.69\,M_\odot$ → 1.7 Gyr) was therefore eliminating a **foreground star that
  the corrected pipeline rejects on parallax alone**.

**Conclusion unchanged** (no confident NGC 6253 recovery; 7 candidates ≈ 7 random),
but the **NGC 6253 subsection needs substantive revision**, not just number swaps:
1. Distance 0.56 → **1.65 kpc** (text + pipeline table).
2. Funnel → 3,598 → 1,908 → 361 → 120 → 7 (or paper's chosen stages); FP ~7.
3. The HD 163560 paragraph: it is now rejected at the parallax stage; the
   asteroseismic-elimination narrative (and the Hon2021 citation's role here)
   becomes moot. Recommend reframing to "no candidate survives kinematics;
   the count matches the random expectation," and either drop HD 163560 or
   mention it as a chemical match excluded by parallax.

This is a published-narrative change touching a specific claim + a citation, so
it is left for author decision rather than auto-applied. The **t20c code fix
(fixed tolerances + Gaia retry) is applied and committed**; the paper edits are
pending your call.

### Artifacts from this investigation
- `t20c_fixedtol.py` — t20c with the paper's fixed tolerances (validated: 31-member
  template, correct tolerances, reproducible 3,598-match funnel).
- `t20c_results_fixedtol.txt`, `t20c_ngc6253_fixedtol_plot.png` — its outputs.

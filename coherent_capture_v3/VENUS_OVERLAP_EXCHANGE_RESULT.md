# Venus overlap-region stellar-exchange N-body result

**Date:** 2026-09-10  
**Preregistration:** `VENUS_OVERLAP_EXCHANGE_PREREGISTRATION.md`  
**Workflow run:** GitHub Actions `34536790421`  
**Status:** PROVISIONAL PENDING CODE REVIEW; numerical run completed successfully.

## Primary result

| Metric | Result |
|---|---:|
| Attempted encounters | 6,144 |
| Numerically valid | 6,144 |
| Numerical failures | 0 |
| Persistent Sun-exchange captures | 145 |
| Persistent exchange fraction | 2.36% |
| Wilson 95% interval | 2.01%–2.77% |
| Cold exchanges (`e <= 0.05`, `i <= 3.4 deg`) | **0** |
| Cold-exchange Wilson 95% upper bound | **0.0625%** |
| Exchanges within 10% of Venus semimajor axis | 40 |
| Cold + near-Venus-a | **0** |

Preregistered verdict: **`EXCHANGE_ONLY_HOT`**.

## Cell-level result

| q/aD | v_inf (km/s) | N | Persistent exchanges | Cold exchanges |
|---:|---:|---:|---:|---:|
| 2 | 0.10 | 512 | 37 | 0 |
| 2 | 0.30 | 512 | 35 | 0 |
| 2 | 0.50 | 512 | 33 | 0 |
| 2 | 1.00 | 512 | 40 | 0 |
| 3 | 0.10 | 512 | 0 | 0 |
| 3 | 0.30 | 512 | 0 | 0 |
| 3 | 0.50 | 512 | 0 | 0 |
| 3 | 1.00 | 512 | 0 | 0 |
| 4 | 0.10 | 512 | 0 | 0 |
| 4 | 0.30 | 512 | 0 | 0 |
| 4 | 0.50 | 512 | 0 | 0 |
| 4 | 1.00 | 512 | 0 | 0 |

Thus all 145 observed exchanges occur at `q/aD = 2`; the two wider disk-survival rows (`q/aD = 3,4`) contain no exchange events in 4,096 attempts combined. Their combined zero-event Wilson 95% upper bound is approximately 0.0937% for this locked sampling measure.

## Orbital-state diagnostics

The two lowest-e persistent exchanges are:

- `e = 0.01127`, `a = 0.7923 AU`, but `i = 94.76 deg`;
- `e = 0.01295`, `a = 0.7298 AU`, but `i = 31.65 deg`.

The minimum inclination among all 145 persistent exchanges is `11.50 deg`, with `e = 0.871`. Therefore the failure of the cold gate is not a threshold-edge artifact: the sampled exchange population does not approach the required joint low-e/low-i corner.

Among the 40 exchange captures within 10% of Venus's semimajor axis, mean eccentricity is approximately 0.343 and mean inclination approximately 83.1 deg. None satisfy the cold gate.

## Interpretation

The previous necessary-condition screen found that a continuous post-flyby gas disk can reach Venus's orbit beginning at `q/aD = 2` only under the most permissive truncation factor (`r_out/q = 0.5`), while the intermediate truncation factor requires `q/aD = 4`.

This N-body test now shows a sharp dynamical tension:

1. `q/aD = 2` still permits exchange at a measurable encounter-conditional rate, but the resulting captures are dynamically hot/inclined;
2. `q/aD = 3` and `4` preserve progressively more disk but produced zero exchange captures in the locked survey;
3. no sampled capture simultaneously arrives with `e <= 0.05` and `i <= 3.4 deg`.

Under this preregistered donor-star/orbit model, the **continuous calibrated gas-damping Venus pathway is therefore strongly disfavored**. This is not a claim that stellar exchange is impossible: exchange clearly occurs at `q/aD = 2`. It is specifically a failure to find overlap between (a) disk survival, (b) genuine exchange, and (c) an already cold enough arrival state for the calibrated damping regime.

## Limits

- This is a focused one-donor-mass, one-donor-semimajor-axis pilot, not the full Venus production survey.
- It samples donor planetary planes isotropically relative to the Solar reference plane, as sealed before outcomes.
- It does not model hydrodynamic damping of the hot captures; those lie outside the calibrated low-e/i screen.
- It does not yet replay Mercury–Neptune survival.
- Historical occurrence probability is not inferred from the encounter-conditional fractions.
- Result remains marked provisional until PR #3 code review is complete and any material review findings are resolved.

Configuration SHA-256: `40a129c479f544b006d936d0315227a984d36511906d2de48a3bbc62d7da8c9d`.
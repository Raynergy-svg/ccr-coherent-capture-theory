# Joint Capture–Damping Screen — Pilot Addendum

**Date:** 2026-09-09  
**Status:** SEALED BEFORE GRID OUTCOME CALCULATION  
**Parent preregistration:** `coherent_capture_v3/JOINT_GATE_SCREEN_PREREGISTRATION.md`

The complete Cartesian screen implied by the parent grid is too large for a first implementation/debugging pass. This addendum fixes a deterministic pilot tranche **before any grid outcomes are inspected**. Pilot results may validate implementation and expose gross incompatibilities, but may not change the parent production grid or final denominator.

## Locked pilot tranche

- donor mass: `1.00 M_sun`
- reference donor semimajor axis: `0.723332 AU`
- donor eccentricity: `0.00` (recorded but analytically inactive in this necessary-condition screen)
- `q/a_D`: all 12 parent values
- `v_inf`: `{0.30, 1.00, 3.00} km/s`
- disk truncation factor: all `{0.2,0.3,0.5}`
- `f_Sigma`: all `{0.01,0.03,0.10,0.30,1.00}`
- surface-density slope: fixed `p=1.0`
- aspect ratio: fixed `h0=0.035`
- flaring: fixed `0.25`
- remaining disk lifetime: all `{0.1,0.3,1.0,3.0} Myr`
- screening eccentricity: all `{0.05,0.10,0.20,0.30,0.50,0.70,0.90}`
- screening inclination: all `{1,3.4,5,10,20,30,60,90} deg`

Total pilot rows: `120,960`.

## Pilot outputs

Report:

1. counts and fractions by final screen label;
2. compatible fraction versus `q/a_D`;
3. compatible fraction versus `f_Sigma` and remaining disk lifetime;
4. the minimum `q/a_D` at which any post-truncation disk reaches Venus's orbit for each truncation factor;
5. maximum eccentricity and inclination that pass the calibrated screen in any pilot cell;
6. whether any contiguous screen region exists under the parent definition.

No REBOUND capture probability, Solar-System survival probability, or historical occurrence rate is inferred from this pilot.

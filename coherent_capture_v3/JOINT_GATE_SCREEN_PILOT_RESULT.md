# Joint Capture–Damping Necessary-Condition Screen — Pilot Result

**Date:** 2026-09-09  
**Preregistration:** `JOINT_GATE_SCREEN_PREREGISTRATION.md`  
**Pilot addendum:** `JOINT_GATE_SCREEN_PILOT_ADDENDUM.md`  
**Rows:** 120,960

## Result

| Label | Count | Fraction |
|---|---:|---:|
| `DISK_TRUNCATED` | 107,520 | 88.8889% |
| `SCREEN_COMPATIBLE` | 480 | 0.3968% |
| `HYDRO_REQUIRED_OR_TOO_SLOW` | 12,960 | 10.7143% |
| `ENERGY_FAIL` | 0 | 0% |

The pilot therefore returns `CONTIGUOUS_SCREEN_REGION` under the preregistered screening definition, but the surviving region is narrow in orbital state and occurs only at the widest encounter-periapsis ratios in the pilot.

## Where the continuous gas disk survives

For a Venus-scale donor orbit (`a_D = 0.723332 AU`):

- `r_out/q = 0.2`: no tested `q/a_D <= 4` retains a disk out to Venus's final orbit.
- `r_out/q = 0.3`: the first surviving disk occurs only at `q/a_D = 4`.
- `r_out/q = 0.5`: the first surviving disk occurs at `q/a_D = 2`.

Every pilot cell with `q/a_D <= 1.5` fails the continuous-history gas route because the flyby truncates the disk inside Venus's final orbit.

## Surviving orbital-state screen

Among `SCREEN_COMPATIBLE` cells:

- minimum `q/a_D`: `2.0`;
- maximum screened arrival eccentricity: `e_i = 0.05`;
- maximum screened arrival inclination: `3.4 deg`;
- a contiguous example exists at `v_inf = 0.3 km/s`, `r_out/q = 0.5`, `e_i = 0.05`, `i_i = 1 deg`, adjacent `q/a_D = {2,3}`, adjacent `f_Sigma = {0.01,0.03}`, and adjacent remaining lifetimes `{0.1,0.3} Myr`.

The low/moderate-e/i Type-I damping screen is fast enough in all of those calibrated pilot cells. The dominant rejection is therefore **not damping time**; it is whether any disk remains at Venus's orbit and whether the arrival state is already mild enough to stay inside the calibrated damping regime.

## Energy bound

No cell that reached the disk-presence gate failed the deliberately generous absolute disk-binding-energy upper bound. This does **not** demonstrate realistic energy dissipation. The parent preregistration deliberately treats this as a necessary-condition upper bound; hydrodynamic efficiency remains unresolved.

## Scientific interpretation

The pilot exposes a direct overlap problem:

> the stellar encounter must be close enough to exchange-capture a Venus-mass planet, yet distant enough that the post-flyby Solar disk still extends to ~0.723 AU.

Under the adopted truncation bracket, the continuous gas route requires approximately `q/a_D >= 2` for the most permissive truncation factor and `q/a_D = 4` for the intermediate factor. Thus the next decisive calculation is the preregistered stellar-exchange N-body survey restricted only for scheduling priority—not denominator—to the `q/a_D = {2,3,4}` overlap region, measuring whether capture into mild (`e ~ 0.05`, low-i) heliocentric states occurs at any meaningful rate.

This pilot does not measure capture probability, terrestrial-planet survival, or historical occurrence rate.

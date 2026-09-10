# CCT v3 Joint Capture–Damping Necessary-Condition Screen

**Date:** 2026-09-09  
**Status:** SEALED BEFORE OUTCOME CALCULATION  
**Parent design:** `docs/superpowers/specs/2026-09-03-venus-capture-design.md`  
**Scope:** Analytic/Monte-Carlo screening layer only. This does **not** replace the preregistered REBOUND encounter survey, Solar-System survival replay, calibrated gas-damping integrations, or spin evolution.

## 1. Question

Does the locked Venus stellar-exchange parameter space contain a non-negligible region that simultaneously satisfies the **necessary** conditions for:

1. a close encounter capable of entering the exchange-capture regime;
2. retention of a gas disk extending to at least Venus's final orbit after flyby truncation;
3. an orbital circularization energy demand that does not exceed the total post-truncation disk orbital-binding-energy reservoir;
4. eccentricity/inclination damping times shorter than the remaining disk lifetime when the state is inside the calibrated low/moderate-e/i regime;
5. a final-state target compatible with the locked Venus semimajor-axis/eccentricity/inclination gates.

A pass means only that the cell is not ruled out by these necessary conditions. It is **not** evidence that exchange capture occurred.

## 2. Locked constants

- Solar mass: `M_sun = 1.98847e30 kg`
- Venus mass: `M_V = 4.8675e24 kg`
- Venus target semimajor axis: `a_V = 0.723332 AU`
- Strict target eccentricity: `e < 0.020`
- Strict target inclination: `i = 3.4 ± 1.0 deg`
- Disk surface density normalization: `Sigma_1AU = 1700 g cm^-2`
- Disk model: `Sigma(r,t)=f_Sigma*Sigma_1AU*(r/AU)^(-p)*exp(-t/tau_d)`
- Disk truncation factors: `r_out/q = {0.2, 0.3, 0.5}`
- Disk density factors: `f_Sigma = {0.01, 0.03, 0.10, 0.30, 1.00}`
- Surface-density slopes: `p = {0.5, 1.0, 1.5}`
- Aspect ratios at 1 AU: `h0 = {0.025, 0.035, 0.050}`
- Flaring index: `{0, 0.25}`
- Remaining disk lifetime: `{0.1, 0.3, 1.0, 3.0} Myr`

Encounter grid is inherited unchanged from the locked Venus design:

- donor mass / solar mass: `{0.50, 0.75, 1.00, 1.25}`
- reference donor semimajor axis / AU: `{0.60, 0.723332, 0.90}`
- actual donor semimajor axis: `M_donor * a_ref`
- donor eccentricity: `{0.00, 0.05}`
- stellar periapsis / donor semimajor axis: `{0.05,0.10,0.20,0.35,0.50,0.75,1.00,1.25,1.50,2.00,3.00,4.00}`
- velocity at infinity / km s^-1: `{0.10,0.30,0.50,1.00,2.00,3.00,5.00,10.00}`

## 3. Derived quantities

### 3.1 Stellar periapsis speed

`v_p = sqrt(v_inf^2 + 2 G (M_sun + M_donor) / q)`.

This is diagnostic only; no capture probability is inferred analytically.

### 3.2 Post-flyby disk radius

`r_out = f_trunc * q`.

**Hard disk-presence gate:** `r_out >= a_V`. If this fails, the continuous-history gas-damping route at Venus's orbit fails for that cell.

### 3.3 Circularization energy demand

For a candidate arriving at Venus's semimajor axis with eccentricity `e_i`, use the locked first-order scale

`DeltaE = G M_sun M_V e_i^2 / (2 a_V)`.

The screening eccentricity grid is `{0.05,0.10,0.20,0.30,0.50,0.70,0.90}`. This is a sensitivity grid, not an assertion about the exchange output distribution.

### 3.4 Absolute disk-energy upper bound

The post-truncation disk orbital binding-energy magnitude between `r_in=0.05 AU` and `r_out` is

`E_disk = integral[ G M_sun/(2r) * 2*pi*r*Sigma(r) dr ]`

with the locked surface-density profile evaluated at the start of the remaining-lifetime interval.

**Hard energy gate:** `DeltaE <= E_disk`.

This is deliberately generous: it treats the entire disk orbital-binding reservoir as available. Passing does not prove the disk can actually absorb the energy; failing rules out that disk cell under this model.

### 3.5 Damping-time screen

Use the standard Type-I wave timescale

`t_wave = (M_sun/M_V) * (M_sun/(Sigma(a_V)*a_V^2)) * h(a_V)^4 / Omega_K`.

For states inside the declared calibration screen (`e_i/h <= 2` and `i_i/h <= 2`, with `i_i` sampled at `{1,3.4,5,10,20,30,60,90} deg`), use

`t_e = t_wave/0.780`, `t_i = t_wave/0.544`.

States outside that screen are labeled `HYDRO_REQUIRED`; they are never counted as damping successes.

**Hard damping gate:** `max(t_e,t_i) <= remaining_disk_lifetime`.

The low-e/i formula is a screen only; the production Venus test retains the two calibrated damping families required by the parent preregistration.

## 4. Primary joint gate

A parameter tuple is `SCREEN_COMPATIBLE` only if all are true:

1. `r_out >= a_V`;
2. `DeltaE <= E_disk`;
3. state is not `HYDRO_REQUIRED`;
4. `max(t_e,t_i) <= t_remaining`.

No analytic capture-success flag is included in the numerator. Encounter cells are carried forward only as coordinates for the later REBOUND survey.

## 5. Decision rule

The screen reports the fraction of locked encounter × disk × `(e_i,i_i)` cells in each of four classes:

- `SCREEN_COMPATIBLE`
- `DISK_TRUNCATED`
- `ENERGY_FAIL`
- `HYDRO_REQUIRED_OR_TOO_SLOW`

It also reports whether any **connected** region exists across adjacent `q/a_D`, `v_inf`, disk-density, and remaining-lifetime cells. A single isolated success is labeled `FINE_TUNED_SCREEN_ONLY`.

Interpretation:

- `NO_SCREEN_REGION`: zero compatible cells.
- `FINE_TUNED_SCREEN_ONLY`: compatible cells exist but only in isolated parameter cells.
- `CONTIGUOUS_SCREEN_REGION`: compatible cells occupy at least two adjacent values in each of `q/a_D`, `f_Sigma`, and remaining disk lifetime for at least one donor-mass/a_ref/v_inf family.

These labels are screening labels only and do not replace the four locked final Venus verdicts.

## 6. Reproducibility

The implementation must be deterministic and write:

- one row per parameter tuple;
- all input parameters;
- `q`, `v_p`, `r_out`, `DeltaE`, `E_disk`, `t_wave`, `t_e`, `t_i`;
- every gate boolean and the final screen label;
- a configuration hash.

The first implementation is unit-tested against analytic limiting cases before any grid result is inspected.

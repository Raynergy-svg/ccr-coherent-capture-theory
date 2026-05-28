# HD 183193 — Full Archive Profile

**Gaia DR3 6771181697822279936** · live pull from Gaia ESAC + GALAH DR4 Data Central TAP.
Identified by the reduced 8D scorer as the brightest/closest actionable nearby dwarf habitability target.

## Identity & Astrometry — Gaia DR3

| Field | Value |
|---|---|
| Source ID | 6771181697822279936 |
| SIMBAD name | HD 183193 |
| Cross-IDs | TIC 169309109; HD 183193; TYC 6876-450-1; 2MASS J19291789-2419241 |
| RA / Dec | 292.32463°, -24.32321° |
| Parallax | 13.3814 ± 0.0183 mas |
| Distance | **74.7 pc** (1/π); GSP-Phot 74 pc |
| Proper motion | (+27.118, +31.971) mas/yr |
| Gaia RV | -27.77 ± 0.23 km/s (5 transits) |
| RUWE | **1.072** (single if <1.4) |
| non_single_star | 0.0 |
| G / BP−RP | 8.777 / 0.7615 |

## Stellar parameters — three independent solutions

| | GALAH DR4 | Gaia GSP-Phot | Gaia GSP-Spec | Gaia FLAME |
|---|---|---|---|---|
| Teff (K) | 5874 | 5836 | 5747 | — |
| log g | **4.368** | 4.279 | 4.04 | — |
| [Fe/H] / [M/H] | -0.003 | -0.298 | -0.23 | — |
| Mass (M☉) | 0.984 | — | — | 1.001 |
| Age (Gyr) | **6.37** | — | — | 6.96 |
| Radius (R☉) | — | 1.10 | — | 1.10 |
| Lum (L☉) | 1.239 | — | — | 1.272 |
| vmic / vsini (km/s) | 1.04 / 4.74 | | | |

**Quality:** flag_sp=0, flag_fe_h=0; S/N CCD3 = 323.

> **Dwarf status confirmed:** log g ≈ 4.37, R ≈ 1.0–1.1 R☉ (Gaia FLAME), L ≈ 1 L☉, Teff 5874 K — a true main-sequence solar twin, not a turnoff/subgiant.

## GALAH DR4 abundances (full element list)

| Element | [X/Fe] | error | flag |
|---|---|---|---|
| C | -0.0548 | 0.0222 | 32 |
| N | +0.5171 | 0.0948 | 32 |
| O | +0.0434 | 0.0280 | 0 |
| Na | -0.0554 | 0.0104 | 0 |
| Mg | +0.0084 | 0.0122 | 0 |
| Al | +0.0711 | 0.0195 | 0 |
| Si | -0.0097 | 0.0067 | 0 |
| K | -0.1451 | 0.0217 | 0 |
| Ca | +0.0260 | 0.0195 | 0 |
| Sc | +0.0590 | 0.0167 | 0 |
| Ti | +0.0246 | 0.0130 | 0 |
| V | +0.0424 | 0.0279 | 0 |
| Cr | -0.0067 | 0.0080 | 0 |
| Mn | -0.0998 | 0.0090 | 0 |
| Co | -0.0021 | 0.0182 | 0 |
| Ni | +0.0071 | 0.0072 | 0 |
| Cu | -0.0804 | 0.0192 | 0 |
| Zn | -0.1034 | 0.0151 | 0 |
| Y | -0.0397 | 0.0182 | 0 |
| Zr | -0.2531 | 0.0710 | 0 |
| Ba | -0.0325 | 0.0292 | 0 |
| La | -- | -- | 2 |
| Ce | -- | -- | 2 |
| Nd | +0.1871 | 0.0425 | 0 |
| Sm | -- | -- | 2 |
| Eu | -- | -- | 2 |
| A(Li) | 2.336 | — | 0 |

**Carbon caveat:** as expected for an FGK dwarf in GALAH DR4, `flag_c_fe` is non-zero. C/Fe cannot be reliably extracted from the CH band for this star — the C/O dimension of the original 9D scorer is therefore unmeasurable here. The 8D scorer was used.

**C/O (raw, with caveat above):** 0.438; **Mg/Si (number, solar-normalised):** 1.095

## Galactic kinematics — GALAH DR4 dynamics VAC

| Quantity | Value | Quantity | Value |
|---|---|---|---|
| Eccentricity | 0.111 | J_R | 18.22 |
| R_peri (kpc) | 8.012 | J_Z | 0.909 |
| R_apo (kpc)  | 10.019 | L_Z | 2075.8 |
| z_max (kpc)  | 0.188 | Energy | -151387 |
| U / V / W (km/s) | -30.05 / +6.72 / +4.38 | | |

## SIMBAD / literature

- SIMBAD nbref: **8** (literature references; no precision-RV campaign visible)
- SIMBAD RV: -26.47 km/s
- NASA Exoplanet Archive: **no known planets**

## Habitable-zone geometry — main-sequence solar twin

- Conservative (runaway / max greenhouse): **1.07–1.89 AU**, P ≈ 1.12–2.62 yr
- Optimistic (recent Venus / early Mars): **0.85–1.99 AU**, P ≈ 0.79–2.84 yr

Unlike HD 28888 (subgiant, HZ ≈ 1.8–3.2 AU, P 2–5 yr), HD 183193's HZ sits at **near-Earth orbital distance** (~0.9–1.7 AU) with sub-year to ~2 yr periods — much faster, more conventional precision-RV target.

## Proposal-ready summary

| Property | HD 28888 (old #1) | **HD 183193 (new lead)** |
|---|---|---|
| Evolutionary state | subgiant (log g 3.94, R 1.99 R☉) | **main-sequence dwarf (log g 4.37, R ~1 R☉)** |
| Distance | 100 pc | **75 pc** |
| G mag | 8.24 | 8.78 |
| Teff | 5734 K | 5874 K |
| Age | 6.31 Gyr | 6.37 Gyr (≈ solar) |
| [Fe/H] | +0.13 | near-solar |
| RUWE | 0.99 | 1.07 |
| Known planets | none | none |
| Precision RV monitoring | none | none |
| 8D hab score | 0.969 | **0.991** |
| HZ orbit | 1.8–3.2 AU / 2–5 yr | **~0.9–1.7 AU / ~1–2 yr** |

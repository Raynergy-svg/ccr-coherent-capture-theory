# HD 28888 — Full Archive Profile

**Gaia DR3 3312653647717790720** · live pull from Gaia ESAC TAP + GALAH DR4 (Data Central TAP)
Generated for the coherent-capture 9D habitability scorer. Reproducible via [`query_hd28888.py`](query_hd28888.py).

## Identity & Astrometry — Gaia DR3

| Field | Value |
|---|---|
| Source ID | 3312653647717790720 |
| Common name | HD 28888 |
| 2MASS | 04333861+1557416 |
| RA / Dec | 68.41124°, +15.96126° (04h33m38.7s, +15°57′41″) |
| Parallax | 10.0008 ± 0.0242 mas → **99.99 pc** (GSP-Phot 99.56 pc) |
| PM (RA, Dec) | +65.228, −55.779 mas/yr |
| Radial velocity | **53.70 ± 0.14 km/s** (identical value in GALAH) |
| RUWE | 0.985 · non_single_star = 0 → **clean single star** |
| G / BP−RP | 8.238 / 0.816 |

## Stellar Parameters — three independent solutions

| | GALAH DR4 | Gaia GSP-Phot | Gaia GSP-Spec | Gaia FLAME |
|---|---|---|---|---|
| Teff (K) | 5734 | 5705 | 5761 | — |
| log g | 3.94 | 4.04 | 3.80 | — |
| [Fe/H] / [M/H] | **+0.130** | −0.099 | +0.06 | — |
| Mass (M☉) | 1.15 | — | — | 1.17 |
| **Age (Gyr)** | **6.31** | — | — | **6.05** |
| Radius (R☉) | — | — | — | 1.99 |
| Luminosity (L☉) | 3.55 | — | — | 3.78 |
| vmic / vsini (km/s) | 1.32 / 5.54 | — | — | — |

> **Note:** log g ≈ 3.9–4.0 with R ≈ 2 R☉ places HD 28888 at the **main-sequence turnoff / early subgiant** stage, not a true dwarf. Relevant for habitability framing.

**Quality:** GALAH `flag_sp = 0`, `flag_fe_h = 0` (clean); S/N (green CCD) = 295.

## GALAH DR4 Abundances — 30 elements

All `flag = 0` (clean) unless noted. Values are [X/Fe] dex.

### Light & α-elements
| El | [X/Fe] | err | El | [X/Fe] | err |
|---|---|---|---|---|---|
| C | −0.072 | 0.020 | K | −0.111 | 0.022 |
| N | +0.376 | 0.052 | Ca | −0.043 | 0.019 |
| O | +0.180 | 0.024 | **Mg** | **+0.020** | 0.010 |
| Na | +0.105 | 0.010 | **Si** | **+0.006** | 0.005 |
| Al | +0.086 | 0.019 | Ti | −0.029 | 0.008 |

### Fe-peak
| El | [X/Fe] | err | El | [X/Fe] | err |
|---|---|---|---|---|---|
| Sc | +0.038 | 0.014 | Co | +0.022 | 0.011 |
| V | +0.015 | 0.014 | Ni | +0.051 | 0.005 |
| Cr | −0.002 | 0.006 | Cu | +0.080 | 0.015 |
| Mn | +0.040 | 0.007 | Zn | −0.010 | 0.014 |

### Neutron-capture (s- / r-process)
| El | [X/Fe] | err | Note |
|---|---|---|---|
| Y | −0.069 | 0.016 | s-process |
| Zr | **−0.598** | 0.052 | unusually low, unflagged |
| Ba | −0.040 | 0.032 | s-process volatile proxy |
| La | +0.257 | 0.020 | |
| Ce | −0.106 | 0.030 | |
| Nd | +0.152 | 0.027 | |
| Sm | +0.049 | 0.038 | |

**Not measured (flag = 2):** Rb, Sr, Mo, Ru, Eu

**Lithium:** A(Li) = 2.255 (3D-NLTE VAC; EW = 51.4 mÅ), flag = 0

### Gaia GSP-Spec cross-check
[α/Fe] +0.02 · [Mg/Fe] +0.10 · [Si/Fe] +0.03 · [Ca/Fe] +0.01 · [Ti/Fe] +0.16 · [Ni/Fe] +0.01 · [N/Fe] +0.12 — consistent with GALAH's near-solar α pattern.

## Nine Scorer Dimensions

| # | Dimension | Input value |
|---|---|---|
| 1 | C/O ratio (Teff-corrected) | **0.308** raw (apply ρ = −0.68 GALAH Teff correction) |
| 2 | Mg/Si | **1.086** (number ratio) |
| 3 | [Fe/H] | **+0.130** |
| 4 | [Mg/Fe] | +0.020 |
| 5 | [Si/Fe] | +0.006 |
| 6 | Volatile budget (s-process) | [Ba/Fe] −0.040, [Y/Fe] −0.069 |
| 7 | Stellar age | **6.3 Gyr** (GALAH) / 6.0 Gyr (FLAME) |
| 8 | Birth-cluster density | *not a direct catalog field* — pipeline-derived (existing profile tags `Teutsch_80`) |
| 9 | Galactic kinematic stability | ecc **0.165**, R_peri 6.33 / R_apo 8.83 kpc, z_max 0.063 kpc, J_R 34.4, J_Z 0.135, L_Z 1723, U/V/W = −52.3 / −40.7 / −11.9 km/s |

### Derived number ratios
- **C/O = 0.308** — solar C/O = 0.55; 10^(log(C/O)☉ + [C/Fe] − [O/Fe])
- **Mg/Si = 1.086** — solar Mg/Si = 1.05; 10^(log(Mg/Si)☉ + [Mg/Fe] − [Si/Fe])

## Galactic Kinematics — GALAH DR4 dynamics VAC

| Quantity | Value | Quantity | Value |
|---|---|---|---|
| Eccentricity | 0.165 | J_R | 34.4 |
| R_peri (kpc) | 6.330 | J_Z | 0.135 |
| R_apo (kpc) | 8.831 | L_Z | 1723.3 |
| z_max (kpc) | 0.063 | Orbital energy | −160679 |
| U / V / W (km/s) | −52.3 / −40.7 / −11.9 | | |

> Dynamics-VAC `X/Y/Z` are heliocentric (≈0.1 kpc = the star's distance); `R_med` returned null. Use R_peri/R_apo for orbital radius (~7.6 kpc guiding, current ~8.2 kpc ≈ solar circle).

## Caveats

1. **Dimension 8 (birth-cluster density)** has no direct Gaia/GALAH field — it comes from the project's clustering pipeline, not these queries.
2. **Evolutionary state:** log g ≈ 3.9–4.0, R ≈ 2 R☉ → turnoff/subgiant, not a dwarf.
3. **[Zr/Fe] = −0.60** is a notable low outlier despite being unflagged.

## Source files
- [`query_hd28888.py`](query_hd28888.py) — rerunnable query script
- [`hd28888_combined_profile.csv`](hd28888_combined_profile.csv) — long-format key/value/source
- `hd28888_query_gaia.csv` · `hd28888_query_gaia_astrophysical.csv` · `hd28888_query_galah.csv` · `hd28888_query_galah_dynamics.csv` · `hd28888_query_galah_li.csv` — raw per-archive pulls

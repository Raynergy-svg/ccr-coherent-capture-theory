# Wider-Net BLS Triage Summary

Total targets processed: 29

SDE >= 7 hits: 8

Candidates surviving full triage: **2**

---

## Ranked candidates

| rank | name | TIC | T | n_sec | P_d | dep_ppm | SDE | R_p | T_eq | cent verdict | score |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | CPD-63   349 | 38877496.0 | 9.59 | 39 | 190.4869 | 444 | 8.27 | 2.11 | 328 | ON_TARGET (significant bu | 3.51 |
| 2 | HD 271308 | 287298304.0 | 9.56 | 9 | 34.2690 | 282 | 7.47 | 2.83 | 800 | ON_TARGET (offset not sig | 2.35 |


## Per-candidate diagnostics

### #1: CPD-63   349

- TIC: 38877496.0
- Position: (69.17225, -62.91497)
- Tmag: 9.59
- R* = 0.918 R_sun, M* = 1.030, Teff = 5745 K
- Sectors: 39, baseline = 2606 d
- BLS: P = 190.48688 d, depth = 444 ppm, dur = 3.60 h, SDE = 8.27
- Planet (if real): R = 2.11 R_Earth, a = 0.6543 AU, T_eq = 328 K
- Centroid verdict: ON_TARGET (significant but <3'' = << 1 TESS pixel)
- Centroid offset: 0.006 arcsec
- Centroid significance: 4.59 sigma
- Triage score: 3.51
- Plot: `widernet_candidates/CPD-63   349/bls.png`

### #2: HD 271308

- TIC: 287298304.0
- Position: (82.44338, -66.04164)
- Tmag: 9.56
- R* = 1.540 R_sun, M* = 1.230, Teff = 6287 K
- Sectors: 9, baseline = 328 d
- BLS: P = 34.26897 d, depth = 282 ppm, dur = 2.40 h, SDE = 7.47
- Planet (if real): R = 2.83 R_Earth, a = 0.2212 AU, T_eq = 800 K
- Centroid verdict: ON_TARGET (offset not significant)
- Centroid offset: 0.003 arcsec
- Centroid significance: 1.74 sigma
- Triage score: 2.35
- Plot: `widernet_candidates/HD 271308/bls.png`


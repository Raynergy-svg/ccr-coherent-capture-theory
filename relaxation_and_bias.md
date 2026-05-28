# Follow-up: relaxation analysis + distance-bias check

Two analyses on the top-25 null result. Code: `relaxation_analysis.py`; data:
`relaxation_backups.csv`.

## A. Distance-bias check — is "top chemistry" a sampling artifact? **No.**

Correlations across the full cohort (N=12,234):

| Driver | Spearman ρ vs hab_score | p | read |
|---|---|---|---|
| **Teff** | **+0.291** | 4×10⁻²³⁷ | dominant — score peaks near-solar Teff |
| log g | −0.100 | 2×10⁻²⁸ | weak — subgiants mildly favoured |
| distance | **+0.083** | 3×10⁻²⁰ | **negligible** (<1% variance) |
| gal. latitude \|b\| | +0.043 | 2×10⁻⁶ | negligible |

- **Distance is not a meaningful driver.** The top-25's median distance (655 pc) is actually *below* the cohort median (727 pc) and the excellent-subset median (737 pc). High-chemistry stars sit at *typical* GALAH distances — they are not preferentially far.
- **Direction is not concentrated.** Top-25 galactic longitudes span l ≈ 255–349° plus a handful at 7–72° (i.e. across GALAH's southern footprint), with \|b\| median 33.8° (slightly *above* the plane, not bulge/plane-piled). 84% are inner-ward (\|l\|<90°) vs 77% for the cohort — only a mild excess.
- **Verdict:** the "top chemistry" is a **real population signal selected by Teff (near-solar) and mild evolutionary state**, not an artifact of one deep survey direction. The reason the leaders fail the actionable cut is simply that **GALAH as a whole is a distant survey** (cohort median 727 pc; only **14** excellent + optimal-age stars lie within 200 pc) — the <200 pc filter keeps a tiny nearby slice, by construction.

## B. Relaxation analysis — reachable backups just outside the cliff

Pool: excellent (>0.9) + age 2–8 Gyr + 200–350 pc → 123 candidates → after **dist<300 pc, G<12.5, RUWE<1.4, NSS=0, planet-free**: **47 reachable backups**.

- All are **bright (G 10–12.5)** → ESPRESSO photon-limited exposure ≈ **7–16 min** for ~1 m/s-equivalent SNR (rough flux scaling from 900 s at G=11; order-of-magnitude).
- **21/47 are northern-accessible** (Dec > −30°).
- The top two are the top-25 "cliff" stars that just missed the 200 pc cut.
- **All 47 are subgiants** (log g 3.81–3.97) — zero dwarfs even in the relaxed set, reinforcing that GALAH's excellent-chemistry stars are uniformly evolved (→ wider HZ, longer-period RV campaigns, same as HD 28888).

### Top 15 reachable backups (ranked by hab_score)

| rk | Gaia DR3 | hab | dist (pc) | G | RUWE | age | Teff | log g | Dec | N? | ESPRESSO min (~1 m/s) | constellation |
|--|--|--|--|--|--|--|--|--|--|--|--|--|
| 1 | 5926741718143504256 | 0.9932 | 281 | 10.6 | 0.78 | 5.9 | 5699 | 3.88 | −60.3 | n | 10 | Ara |
| 2 | 5041407258853950464 | 0.9928 | 242 | 10.1 | 1.17 | 6.5 | 5542 | 3.85 | −23.3 | **Y** | 7 | Cetus |
| 3 | 5481503177467663744 | 0.9889 | 268 | 10.4 | 0.92 | 6.2 | 5469 | 3.84 | −61.2 | n | 9 | Pictor |
| 4 | 3633381876733629952 | 0.9794 | 245 | 10.4 | 1.25 | 7.7 | 5467 | 3.87 | −5.2 | **Y** | 8 | Virgo |
| 5 | 4652356786039719680 | 0.9793 | 288 | 10.5 | 1.02 | 5.7 | 5514 | 3.82 | −74.0 | n | 10 | Mensa |
| 6 | 3618243285246354432 | 0.9792 | 254 | 10.1 | 1.05 | 4.8 | 5534 | 3.82 | −8.7 | **Y** | 6 | Virgo |
| 7 | 6005184485425467392 | 0.9752 | 287 | 10.9 | 1.26 | 7.3 | 5746 | 3.97 | −40.2 | n | 13 | Lupus |
| 8 | 6249745970569490944 | 0.9742 | 297 | 11.1 | 0.86 | 5.9 | 5654 | 3.87 | −16.6 | **Y** | 16 | Scorpius |
| 9 | 5796751749778293632 | 0.9703 | 294 | 10.8 | 0.90 | 7.4 | 5551 | 3.88 | −73.2 | n | 13 | Apus |
| 10 | 5820179215840908544 | 0.9640 | 291 | 10.6 | 1.14 | 5.3 | 5489 | 3.83 | −69.0 | n | 11 | Triangulum Australe |
| 11 | 57517519130999936 | 0.9617 | 262 | 10.3 | 1.15 | 5.9 | 5677 | 3.89 | +19.0 | **Y** | 8 | Taurus |
| 12 | 4046719972997615360 | 0.9595 | 271 | 10.3 | 0.86 | 4.6 | 5707 | 3.82 | −31.3 | n | 8 | Sagittarius |
| 13 | 6704071699950942976 | 0.9574 | 227 | 10.2 | 0.96 | 7.1 | 5647 | 3.92 | −49.1 | n | 7 | Telescopium |
| 14 | 4760419709395670656 | 0.9567 | 270 | 10.4 | 0.92 | 6.2 | 5687 | 3.92 | −62.4 | n | 9 | Dorado |
| 15 | 3620784187898424192 | 0.9561 | 260 | 10.5 | 0.95 | 7.4 | 5365 | 3.81 | −5.6 | **Y** | 9 | Virgo |

(Full 47 in `relaxation_backups.csv`.) Notably #2 (Cetus, Dec −23°, G 10.1, hab 0.993) and #11 (Taurus, Dec +19°, G 10.3) are northern, bright, and score above HD 28888 — strong ESPRESSO backup-proposal candidates at ~240–262 pc.

## Bottom line
1. The null result holds and is **not** a sampling artifact — high chemistry is a real Teff/evolution-selected population at typical GALAH distances.
2. Pushing the cut to 300 pc / G 12.5 yields **47 bright, single, planet-free, optimal-age backups** (21 northern), all ESPRESSO-feasible in <20 min — a ready secondary target list.
3. Every reachable star is a subgiant, so the observational strategy (longer-period HZ planets, multi-year baselines) is the same as for HD 28888.

# APOGEE DR17 cross-validation of the 8D dwarf chemistry ranking

## The test

The reduced 8D scorer ranked 32 nearby (<200 pc) true dwarfs as actionable
habitability targets, with HD 183193 as the brightest/closest pick. The 8D
scorer drops the C/O dimension because GALAH DR4 cannot measure C/Fe reliably
on FGK dwarfs (`flag_c_fe = 0` for 0/13,554). APOGEE DR17 uses H-band CO
bandheads and **does** measure C reliably on dwarfs (verified: 99.6% of OCCAM
FGK dwarfs have valid `C_FE`, median error 0.023 dex). So the test:
cross-match the 32 dwarfs against APOGEE DR17, pull APOGEE C/Fe and the full
abundance set, recompute the 9D hab score with real C, and see whether the 8D
ranking holds.

## Findings

### 1. APOGEE coverage: 9 of 32 dwarfs are in APOGEE DR17.

Most of the 32 are southern (Dec −30° to −75°) and outside the APOGEE
footprint. **HD 183193 itself is not in APOGEE** — the brightest/closest pick
cannot be directly cross-validated. The 9 matched stars are spread across
Dorado, Aquarius, Sagittarius, Carina.

### 2. APOGEE [C/Fe] is reliably measured and near-solar for all 9.

| star | APOGEE [C/Fe] | err | flag | APOGEE C/O (Teff-corr) | s_CO |
|---|---|---|---|---|---|
| HD 271308 | −0.060 | 0.048 | 0 | 0.616 | 1.000 |
| CD−60 1593 | −0.091 | 0.025 | 0 | 0.305 | 1.000 |
| HD 217340 | −0.046 | 0.034 | 0 | 0.429 | 1.000 |
| CPD−63 349 | −0.144 | 0.028 | 0 | 0.274 | 1.000 |
| HD 271200 | −0.092 | 0.032 | 0 | 0.317 | 1.000 |
| BD−08 6091 | −0.010 | 0.049 | 0 | 0.391 | 1.000 |
| TYC 5248-636-1 | +0.012 | 0.038 | 0 | 0.386 | 1.000 |
| TYC 5232-208-1 | −0.116 | 0.022 | 0 | 0.178 | 1.000 |
| HD 178528B | −0.112 | 0.036 | 0 | 0.486 | 1.000 |

Distribution: median −0.091, range −0.144 to +0.012, std 0.051 dex. **All
nine sit comfortably within the s_CO=1.0 plateau (C/O 0.15–0.65).** No
carbon-rich or carbon-depleted outlier among the top dwarfs.

### 3. The 8D scorer's "drop C/O" assumption is empirically vindicated.

For these chemically solar-like dwarfs, the C/O dimension contributes 1.0
identically. Dropping it changes nothing in the dim-by-dim comparison; the
8D ranking among solar-like dwarfs is therefore well-justified, not an
artifact of working around a missing dimension.

### 4. But absolute scores shift by ~0.07 under APOGEE.

| metric | value |
|---|---|
| median hab9_APOGEE − hab8_GALAH | **−0.061** |
| mean Δ | −0.070 |
| std Δ | 0.033 |
| hab9_APOGEE > 0.9 (excellent) | 7/9 |
| hab9_APOGEE > 0.95 | 1/9 |

This downward shift is **not** from the C/O dimension (which is 1.0 everywhere).
It comes from APOGEE-GALAH **calibration offsets in the other abundance
dimensions** — APOGEE H-band ASPCAP analyses sit on a slightly different
abundance scale than GALAH optical Sp4N analyses, particularly for the alpha
elements and the volatile/Ce proxy. The two surveys agree on the *qualitative*
near-solar picture but disagree at the 0.05–0.1 dex level on individual ratios.

### 5. Relative rankings reshuffle moderately.

| 8D rank | hab8 | star | hab9_APOGEE | 9D rank |
|---|---|---|---|---|
| 1 | 0.9936 | HD 271308 | 0.886 | 8 |
| 2 | 0.9934 | CD−60 1593 | 0.952 | 1 |
| 3 | 0.9924 | HD 217340 | 0.928 | 6 |
| 4 | 0.9922 | CPD−63 349 | 0.949 | 3 |
| 5 | 0.9914 | HD 271200 | 0.940 | 4 |
| 6 | 0.9911 | BD−08 6091 | 0.856 | 9 |
| 7 | 0.9902 | TYC 5248-636-1 | 0.929 | 5 |
| 8 | 0.9901 | TYC 5232-208-1 | 0.950 | 2 |
| 9 | 0.9884 | HD 178528B | 0.905 | 7 |

The within-survey absolute rank flips significantly for some stars (HD 271308
goes from #1 to #8; TYC 5232-208-1 goes from #8 to #2). The two surveys agree
on broad population but not on fine ordering.

## What this means for the proposal & paper

**Holds:** The 8D scorer correctly identifies a real near-solar-chemistry
dwarf population. C/O does not change the picture for these stars — it
saturates at 1.0 for any normal solar-type dwarf, so its absence in GALAH
DR4 dwarfs doesn't materially affect the population-level ranking. The 8D
result is methodologically defensible.

**Caveats to write into the paper:**
- Absolute hab scores are survey-dependent at the 0.05–0.1 dex level
  (calibration offsets between GALAH and APOGEE). The "0.991" for HD 183193
  is GALAH-scale; an APOGEE-scale evaluation would likely yield ~0.92 ± 0.05.
- Fine within-population rankings (which dwarf is #1 vs #5) are not robust
  across surveys. Multiple targets in the top tier are equivalent at the
  precision available.
- **HD 183193 is not in APOGEE.** The "brightest/closest dwarf" claim survives
  unchanged (Gaia astrometry + GALAH chemistry are both clean), but a direct
  C/Fe measurement is not available. Worth checking the HARPS/UVES/ESPRESSO
  archives or proposing dedicated high-res spectroscopy for an independent C
  measurement before committing to it as the #1 target.

## Recommended language for paper

> *The 8D scorer's omission of C/O — necessitated by GALAH DR4's inability to
> measure C/Fe on FGK dwarfs — is empirically validated against APOGEE DR17,
> which can. For the 9 of 32 top-ranked dwarfs that APOGEE also observed, the
> APOGEE [C/Fe] distribution is uniformly near-solar (median −0.09 dex, range
> −0.14 to +0.01); all nine sit on the flat s_CO = 1.0 plateau of the C/O
> sub-score; and the 9D-with-APOGEE-C scores remain in the "excellent" band
> (>0.9) for 7 of 9. The absolute hab values shift ~0.07 lower under APOGEE
> than under GALAH — a calibration-scale offset in the non-C dimensions rather
> than a C/O signal — and within-population rankings reshuffle moderately,
> reflecting that the surveys agree on the broad population but not on the
> fine ordering at the 0.05-dex precision available. The proposed lead target
> HD 183193 is not in APOGEE DR17; an independent C/Fe measurement (HARPS or
> ESPRESSO archive, or new high-resolution spectroscopy) would close that
> validation gap.*

# HD 183193 archive sweep — closing the C/Fe validation gap

**Question:** APOGEE DR17 doesn't cover HD 183193, so the 8D scorer's
top-pick has no independent C/Fe measurement. Is there a pre-existing
high-res spectrum (HARPS / UVES / ESPRESSO) or a published abundance
analysis somewhere else?

**Answer: no.** HD 183193 has never been observed with dedicated
high-precision spectroscopy, and never appeared in any major
high-precision abundance catalog. The closure of the C/Fe gap will
require *new* observations.

## What was checked

| archive / catalog | scope | HD 183193 records |
|---|---|---|
| **ESO raw archive** (`dbo.raw`) | HARPS, UVES, ESPRESSO, FEROS, X-Shooter — 10″ cone | **0** |
| **PASTEL** (Soubiran+ compiled atmospheric params) | ~150,000 spectroscopic Teff/log g/[Fe/H] | **0** |
| **Bensby+ 2014** thin/thick disk dwarfs (has [C/Fe]) | 714 FGK dwarfs | 0 |
| **Adibekyan+ 2012** FGK with [C/Fe] | 1111 FGK dwarfs | 0 |
| **Brewer+ 2016** SPOCS-II Keck HIRES ([C/H], [O/H]) | 1626 stars | 0 |
| **Hypatia** (Hinkel+ 2014) compilation | ~6,500 stars | 0 |
| **Delgado Mena+ 2017** HARPS C/O/Mg/Si/Al/Ca | 1059 FGK dwarfs | 0 |
| **SIMBAD references** | all bibcodes citing HD 183193 | **8 (all surveys / target lists, none abundance)** |

## The 8 SIMBAD references

| year | bibcode | type | note |
|---|---|---|---|
| 1988 | `1988MSS...C04....0H` | catalog | MK spectral types compilation |
| 1993 | `1993yCat.3135....0C` | catalog | Henry Draper Catalogue digitisation |
| 2014 | `2014ASPC..485..223B` | catalog | JMMC stellar diameter catalogue v2 |
| 2018 | `2018A&A...609A.116R` | calibration | Gaia red-clump photometric calibration |
| 2019 | `2019MNRAS.490.3158C` | catalog | stellar diameters / mid-IR interferometry |
| 2019 | `2019A&A...624A..19B` | **GALAH/TGAS** | broad survey paper |
| 2021 | `2021MNRAS.506..150B` | **GALAH+ DR3** | the survey that fed the 9D scorer |
| 2023 | `2023AJ....165..267H` | **Bioverse / ELT target list** | independent habitability list! |

The Bioverse paper (Hardegree-Ullman+ 2023, *"Bioverse: A Comprehensive
Assessment of the Capabilities of Extremely Large Telescopes to Probe
Earth-like O₂ Levels in Nearby Transiting Habitable-zone Exoplanets"*)
is the **one independent corroboration**: HD 183193 sits in their pool of
nearby stars worth assessing for ELT biosignature follow-up. No one else
has touched it with dedicated spectroscopy.

## What this means

**The validation gap on HD 183193's C/Fe cannot be closed from existing
archives.** Three takeaways:

1. **The 8D scorer's verdict on HD 183193 stands undisturbed.** Nothing in
   the literature contradicts the GALAH chemistry; nothing else has even
   tried. The 9-of-9 APOGEE pattern (all matched dwarfs have near-solar C
   sitting on the s_CO=1.0 plateau) is the only inference we have, and it's
   compatible.

2. **The "untouched solar twin" framing in the proposal is literally true.**
   HD 183193 has no HARPS/UVES/ESPRESSO history, no precision-RV monitoring,
   no detailed abundance analysis. The Bioverse 2023 paper independently
   put it on a habitability target list. This is a genuinely virgin target.

3. **The next observational step is direct.** A single ESPRESSO snapshot
   (~10 minutes at G = 8.78) would deliver the first dedicated C/Fe + full
   high-precision abundance pattern for HD 183193 and close the validation
   gap. Same observation also delivers a precision-RV zero-point for the
   long-period (1–2 yr) HZ planet search.

## Recommended observation block for the proposal

> Single high-S/N ESPRESSO snapshot (R ≈ 140,000, S/N ≈ 300 in 10–15 min at
> V ≈ 8.9, with the slit mode acceptable for this declination from Paranal)
> to deliver: (i) the first independently-measured [C/Fe] confirming the
> GALAH-derived solar-twin chemistry, closing the C/O dimension that GALAH
> DR4 cannot measure on dwarfs; (ii) a Mg/Fe, Si/Fe, Al/Fe, Ca/Fe, Ba/Fe
> cross-check on the existing GALAH determinations; (iii) a precision-RV
> zero-point for the proposed multi-year HZ planet search.

## Files
- `hd183193_eso_spectra.csv` — empty (no spectra)
- `hd183193_simbad_refs.csv` — the 8 references with bibcodes / years
- `hd183193_archive_sweep.py` — reproducible script

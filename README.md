# Coherent Capture Theory — Empirical Test Series

*Stars carry a permanent chemical record of their birth. This repository contains the analysis that demonstrated it.*

**Certan (2026)**

Multi-instrument empirical analysis of chemical coherence in open clusters and the Milky Way field star population, using GALAH DR4 (917,588 stars) and APOGEE DR17 (OCCAM VAC, 2,515 stars).

## Summary of Results

| Test | Finding | Significance |
|------|---------|--------------|
| **T9** | Open clusters carry distinct C/O fingerprints (GALAH, 655 clusters) | KW p < 10⁻¹⁰ |
| **T-APOGEE** | Confirmed independently in APOGEE (85 clusters) | KW p = 8.8×10⁻¹⁰⁴ |
| **T10** | Chemical coherence is spatially independent | Mantel p = 0.60 |
| **T14** | Coherence degrades on τ = 1.29 Gyr timescale | Spearman p = 4×10⁻³ |
| **T15** | Multi-element fingerprint: C/O predicts Mg/Fe (Fisher OR = 4.69) | p = 2.7×10⁻³ |
| **T17** | No decay in surviving clusters — survivorship bias | All ρ negative |
| **T18** | Alpha 2× tighter than s-process in 98% of clusters | Wilcoxon p = 10⁻⁹⁸ |
| **T19** | Outer disk more coherent; persists after [Fe/H] control | Spearman p = 10⁻⁹ |
| **T16b** | 2× enrichment of dissolved members in field (intra-GALAH) | 5 clusters sig |
| **T16c** | Enrichment flat 0–10 Gyr — fingerprint is permanent (τ > 10 Gyr) | AIC favors flat |
| **T16d** | Ba/Fe (5th dim, not used in matching) confirms 97.2% of clusters | Wilcoxon p = 10⁻⁴¹ |
| **T16e** | Kinematic residual: matched stars 5.2% closer in RV | Wilcoxon p = 10⁻³⁷ |
| **T20c** | Candidate dissolved NGC 6253 members identified (4 stars) | Follow-up needed |

## Key Conclusions

1. **Chemical coherence is real and instrument-independent** — confirmed across GALAH and APOGEE
2. **The fingerprint is multi-dimensional** — correlated across CNO, alpha, and s-process channels
3. **The fingerprint is permanent** — no decay detected over 0–10 Gyr with fixed-threshold matching
4. **Dissolved cluster members are recoverable** — 3× enrichment in field stars, confirmed by independent Ba/Fe and kinematic channels
5. **The Galaxy's chemical archive is complete** — every star still carries its birth certificate

## Data Requirements

The analysis requires:
- `galah_dr4_allstar_240705.fits` — GALAH DR4 allstar catalog (~723 MB)
- `allStar-dr17-synspec_rev1.fits` — APOGEE DR17 allStar (~2.5 GB)
- `occam_member-DR17.fits` — OCCAM membership catalog (~4 MB)

These are not included in the repository due to size. Obtain from:
- GALAH: https://www.galah-survey.org/dr4/
- APOGEE/OCCAM: https://www.sdss.org/dr17/

## Scripts

### Core analysis pipeline
| Script | Description |
|--------|-------------|
| `cross_match_ccr.py` | Initial CCR cross-match with exoplanet data |
| `t5_coherence.py` | Cluster coherence analysis (GALAH) |
| `t6_chem_cluster.py` | Chemical clustering |
| `t6b_umap_cluster.py` | UMAP dimensionality reduction |
| `t7_uvw.py` | UVW velocity analysis |
| `t8_young.py` | Young cluster analysis |
| `t9_cluster_coherence.py` | Full GALAH cluster coherence with ages |
| `tapogee.py` | APOGEE DR17 independent replication |
| `t10_mantel.py` | Mantel spatial independence test |
| `t11_multielement.py` | Multi-element analysis (GALAH) |
| `t12_clustering.py` | Hierarchical clustering |
| `t13_offlocus.py` | Off-locus cluster identification |
| `t14_decay_curve.py` | Chemical coherence decay curve (τ = 1.29 Gyr) |
| `t15_multielement_coherence.py` | Multi-element simultaneous coherence (APOGEE) |
| `t16_dissolved_recovery.py` | Cross-survey dissolved cluster recovery (v1) |
| `t16b_dissolved_intra_galah.py` | Intra-GALAH dissolved recovery (Mahalanobis) |
| `t16c_permanence_test.py` | The permanence test (fixed threshold, no decay) |
| `t16d_sproc_consistency.py` | S-process independent confirmation (Ba/Fe) |
| `t16e_kinematic_traceback.py` | Kinematic RV traceback test |
| `t17_coherence_ladder.py` | Multi-element coherence lifetime ladder |
| `t18_nucleosynthetic_timestamp.py` | Nucleosynthetic timestamp (alpha vs s-process) |
| `t19_galactic_radius.py` | Galactic radius coherence gradient |
| `t20_find_one_star.py` | Individual star recovery — Praesepe |
| `t20b_find_one_star_ngc6791.py` | Individual star recovery — NGC 6791 |
| `t20c_ngc6253.py` | Individual star recovery — NGC 6253 (6D + age) |

## Follow-up Observation Target

**Gaia DR3 5953941329394191360** — Candidate dissolved NGC 6253 member
- G = 9.46 (bright, easily observable)
- Parallax matches NGC 6253 to 1.2%
- 5D chemistry match (GALAH)
- Independently confirmed metal-rich by Gaia GSP-Spec ([M/H] = +0.14)
- Requires UVES/HARPS spectrum for age determination and [Al/Fe] at 0.01 dex

## Author

Certan (2025)

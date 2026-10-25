# CCT: return to the original evidence

Private research audit • 12 October 2026 • Version 2 after independent computational review • No publication or new simulations

## Bottom line

The original capture claim is not established by the earliest recoverable evidence. A smaller, plausible chemical signal survives a bounded reanalysis, but it is conditional evidence in a selected existing sample. It is neither a novel discovery nor evidence that planetary capture occurred.

The most consequential new finding is a sample-definition error: the GALAH T9 “cluster members” are actually stars inside sky apertures. Its statistics reproduce, but those statistics cannot by themselves establish shared birth-cluster chemistry. This issue reaches downstream analyses that reuse T9 templates.

The useful next step is to rebuild the empirical foundation using independently verified membership and a sealed, non-overlapping validation sample. It is premature to claim permanence, recover dissolved siblings, or infer a planetary formation mechanism.

## 1. What “the beginning” actually contains

The oldest recovered private statements are two documents internally dated **5 November 2025**:

- **Final Theory Assessment Report** describes 35 of 379 multi-planet systems (9.2%) passing an architectural classifier: moderate eccentricities, a low resonance fraction and approximate orbital alignment. It calls this supportive evidence for capture.
- **Laws of Coherency** describes a four-diagnostic selection of 16 of 379 systems (4.2%). It proposes preserved coplanarity, eccentricity 0.05–0.10, absent resonances and stellar obliquity above 15°.

Their accessible Library copies were uploaded on 11 October 2026. The internal 2025 dates are not independently authenticated contemporaneous timestamps. The two counts describe different filtering stages; the original classifier code, full candidate tables, dated archive query and original pre-outcome hypothesis record were not located in the searched sources. Their exact historical ordering is therefore less secure than the documents' narrative implies.

The public repository has one root commit, [83f0468, 16 March 2026](https://github.com/Raynergy-svg/ccr-coherent-capture-theory/commit/83f0468ca579c4e9baf690cc7fa94f289d7c3dda). It imports **T5–T20 together with their results**. Chemical coherence is thus a later recoverable empirical branch, not the earliest capture evidence. That history does not demonstrate that each chemical test was pre-registered. This is an absence of verifiable timing evidence, not proof that no earlier record existed.

The audit pins current public main to **ff273145c6edf855a810992d6f8afbdf7916a7de**. It preserves, rather than reactivates, superseded claims. Venus is outside this audit.

## 2. Claim-to-evidence map

| Claim | Recoverable evidence | What it currently establishes |
|---|---|---|
| 9.2%, or 4.2%, of multi-planet systems formed by capture | Historical documents count systems passing chosen orbital thresholds | Candidate-classification fractions. No independently identified capture truth set or calibrated population inference was recovered. |
| Capture candidates lack resonance chains/randomize period ratios | Historical candidate examples and asserted thresholds | An early example, K2-138, conflicts with the claimed diagnostic; see below. The later 16-system list was not available to check. |
| T5 kinematic groups reveal birth-cluster chemistry | Root code and output: ten broad KMeans groups, differing C/O distributions; zero groups below the stated coherence threshold | Differences among kinematic populations, compatible with ordinary Galactic structure. These are not independently established birth clusters. |
| T6 chemically selected groups are chemically tight | The clustering inputs include carbon and oxygen abundances themselves | Tightness in variables used to construct groups is not an independent validation. External membership or withheld chemistry is needed. |
| T9: 655 open clusters have distinct C/O fingerprints | GALAH stars selected within 0.5° of cluster centres; reproduced nominal Kruskal–Wallis statistic | Differences between overlapping sky-aperture samples. Membership and independence were not established. |
| APOGEE independently confirms T9 | Original table has 2,515 rows, 1,994 IDs and a chemistry-influenced membership rule | A separate instrument, but repeated objects, differing selection, survey overlap and uncalibrated abundances prevent a simple claim of independent confirmation. |
| Chemical signal survives stricter controls | New retained-data, unique-ID, training-only controlled check below | Modest conditional predictive information in selected catalog-membership candidates. Not universal fingerprints, birth-site identification, capture evidence or novelty. |
| T16c proves permanent chemistry over 0–10 Gyr | 253 rows; 231 ages; 202 under 1 Gyr; maximum age 6.31 Gyr | The stated 0–10 Gyr coverage is false. Failure to detect decay is not evidence of permanence. |
| T16/T18 identify dissolved members or cloud mixing structure | Code reuses the T9 sky-aperture sample/templates | These physical interpretations need a membership-correct reconstruction. Their prior numerical audits do not fix the upstream sample definition. |
| Later dynamical tests settle every version of capture | Repository has later derivations, simulations and withdrawals | Historical results and claims are preserved. Their full physical derivations and numerical experiments were not independently rerun in this bounded audit. |

### One direct check of the original orbital diagnostic

The 35-candidate report explicitly lists **K2-138** with a resonance fraction of 0.27. Yet its 2018 discovery paper already describes an inner five-planet near-3:2 chain. Its quoted periods, 2.35, 3.56, 5.40, 8.26 and 12.76 days, yield successive ratios 1.5149, 1.5169, 1.5296 and 1.5448. These are about 1–3% above 3:2. [Christiansen et al. (2018)](https://arxiv.org/abs/1801.03874)

This contradicts using that early candidate as evidence of random period ratios or absence of a near-resonant chain. Near-commensurate periods alone do not prove libration of resonance angles, and this check does not refute every possible capture event. Without the historical code, the cause of its reported fraction is unverified. We must not invent an all-pairs-versus-adjacent-pairs denominator explanation.

## 3. The GALAH foundation: arithmetic reproducible, interpretation unsupported

The [T9 selection code](https://github.com/Raynergy-svg/ccr-coherent-capture-theory/blob/ff273145c6edf855a810992d6f8afbdf7916a7de/t9_cluster_coherence.py#L100-L149) admits every quality-filtered GALAH star within 0.5° of each cluster centre. It does not require stellar parallax, proper-motion, radial-velocity or catalog membership agreement. Stored proper-motion columns describe the cluster centre, not each star. The exported table lacks stable stellar IDs and stellar astrometric membership fields.

Independently recomputed from both root and current retained tables:

- 10,479 star–aperture rows in 655 apertures with at least three rows
- 8,749 unique sky coordinates; 1,324 reused coordinates and 1,730 extra rows
- A single coordinate can occur in seven apertures
- Three pairs of differently named apertures contain exactly identical coordinate sets
- Nominal Kruskal–Wallis H = 1170.814, nominal p = 8.30 × 10⁻³²
- After a global linear metallicity adjustment, nominal H = 887.140 and p = 2.83 × 10⁻⁹
- Median within-aperture radial-velocity standard deviation ≈26.56 km/s; 648 of 655 exceed 10 km/s

These small p-values are arithmetic reproductions, **not calibrated significance claims**: repeated stars violate independent-group assumptions, and sky position, stellar populations and survey selection provide alternative explanations. The large velocity spreads are a contamination warning, not an independent membership classifier by themselves.

The 0.05 “dex” threshold in this pipeline is applied to the linear number ratio C/O. A linear-ratio standard deviation is not measured in dex. Our new check uses [C/Fe]−[O/Fe], the solar-relative logarithmic ratio [C/O] in dex. It differs from absolute log10(C/O) by a constant, which does not change these prediction errors.

The position-only selection exists in the first public commit. It was not repaired by the later cluster-distance unit correction. Previous repository assessments calling T9 “solid” did not address this membership issue.

## 4. Bounded APOGEE reanalysis

### Protocol and data semantics

A private analysis specification was written before calculating the new result. It is included as `TEST_SPEC.md`; this is not a public historical preregistration. Only small retained tables were used, on one CPU, without dependency installation or external data mutation.

The hypothesis is narrow: **Does a cluster label improve held-out stellar [C/O] prediction beyond fixed temperature, surface-gravity and metallicity controls?**

Admission uses unique APOGEE IDs, both RV and PM membership scores above 0.5, S/N ≥70, temperatures 4000–5200 K, 1≤log g<3.8, and at least five retained stars per cluster. Ambiguous multiple-cluster IDs are removed. Metallicities are not used to admit members in the new check. This is a custom conservative screen: the published OCCAM rule is scores >0.01 in all three RV, PM and metallicity spaces, not the original script's claimed standard of two-of-three scores >0.5. OCCAM scores reflect distances from fitted cluster means, not calibrated posterior membership probabilities. A quadratic ridge nuisance model is fit only on each training fold; no star ID appears in both training and validation. Cluster residual means are learned on training stars and evaluated on held-out stars. The statistic equally weights clusters. This predicts held-out stars in clusters already represented in training. It does not identify the birthplace of an unknown field star or predict the chemistry of a previously unseen cluster.

The initial 420-star run included 35 stars with PM_PROB=2. The primary OCCAM documentation explains that 2 is a fallback to a separate Gaia membership determination when its PM analysis fails. It is not an ordinary probability. The retained table omits the underlying CG_PROB and cluster quality flags. The correction and all initial results are preserved explicitly. [Myers et al. (2022), Table 3 and membership discussion](https://arxiv.org/html/2206.13650v1)

### All attempted comparisons, without selecting the best result

| Check | Stars / clusters | Held-out MSE improvement | Conditional permutation p |
|---|---:|---:|---:|
| Initial frozen procedure, including sentinel | 420 / 27 | 11.29% | 0.0005 |
| Correct probability semantics: exclude scores above 1 | 382 / 26 | 8.85% | 0.0010 |
| Same corrected data; additionally preserve original fold allocation in permutations | 382 / 26 | 8.85% | 0.0065 |

Each run used 1,999 permutations. The latter two are transparently post-result validity checks, not additional confirmatory tests or independent replications.

In the corrected sample, equally cluster-weighted RMSE is **0.0852 dex without labels and 0.0813 dex with labels**. Fifteen of 26 clusters improve. A descriptive bootstrap of fixed per-cluster prediction gains gives a 95% interval of approximately **−0.00061 to +0.00204 dex²**. It spans zero and does not establish a broadly transferable average improvement. This bootstrap does not refit models and is not a formal population-generalization interval.

### Why this is still limited

- Preserving both covariates and folds produces sparse bins: only 212/382 stars (55.5%) lie in bins with multiple cluster labels. The p-value is conditional on limited exchanges, not a general significance certificate. Coarse bins only approximate conditional exchangeability.
- The retained source table was already response-selected by the original pipeline to 0.05<C/O<2.0; this reanalysis cannot recover excluded stars. Earlier undocumented trial or sample-selection history cannot be reconstructed from the root commit.
- These are membership-score-selected candidates. Parallaxes, Gaia membership probabilities and cluster-quality designations still need checking against original catalogs.
- Stellar evolution can change surface carbon; common age may help predict C/O without implying a permanent birth fingerprint. Age and dredge-up have not been separately identified by these controls.
- Survey calibration/systematics can correlate within observing groups. APOGEE itself advises caution across temperature and gravity, uses separate giant/dwarf zero-points and does not supply abundance covariances. [SDSS abundance guidance](https://www.sdss4.org/dr17/irspec/abundances/)
- Thirty-one corrected APOGEE stars position-match the GALAH aperture table within 1 arcsecond. This is not a full epoch-aware ID crossmatch, but it is sufficient to reject an assumption of complete star-level independence. No cross-survey abundance calibration was fitted here.
- GALAH abundance trends with temperature can be substantial. A modern cluster-based calibration study explicitly investigates this problem. [Kos et al. (2025)](https://arxiv.org/abs/2501.06140)

**Defensible result:** cluster labels carry conditional predictive information for C/O in this selected retained APOGEE sample. The effect is modest, heterogeneous and not yet independently validated on new stars.

## 5. What is real, what would be new, and what to test next

Chemical similarities among stars formed together are an established scientific expectation and research topic. GALAH work published years before CCT demonstrated chemical tagging of known clusters and candidate members. Recovering related structure in another sample would be useful replication, not evidence unique to capture. [Kos et al. (2018)](https://academic.oup.com/mnras/article/473/4/4612/4555384)

The next foundation test should be specified before accessing its chemical outcomes:

1. Rebuild a membership table using stable survey/Gaia IDs, proper motions, parallaxes and published membership probabilities. Retain quality designations and every rejection reason.
2. Seal a star-level holdout that does not overlap the present analysis. Prefer an independently selected survey sample and harmonize abundance scales without using holdout outcomes.
3. Predict withheld element measurements or held-out C/O with measurement uncertainties, stellar evolutionary state, age, metallicity and survey/field effects addressed. Compare genuine clusters to physically matched non-member controls.
4. Report effect size and predictive uncertainty. Failure, or inability to distinguish ordinary common formation from the proposed mechanism, remains an acceptable outcome.
5. Only after this foundation survives should dissolved-sibling recovery or unusually distinctive fingerprints be tested. Capture needs an additional, derived observable that competing formation mechanisms do not reproduce.

For the exact 2025 capture classification, the essential missing artifacts are its original code, complete 35- and 16-system tables, input archive snapshot and uncertainty/filter definitions. That gap does not block the membership rebuild, but it blocks an exact replication of the historical candidate percentages.

## Independent computational review

A fresh review independently reproduced the 382-ID/26-cluster selection, checked that no retained ID or exact coordinate repeats, refit every training-only nuisance and cluster model, and matched the stored held-out predictions. It independently reproduced the first five permutation statistics, confirmed covariate/fold count preservation and counted 12 of 1,999 null statistics at least as large as observed, giving (12+1)/2000 = 0.0065. The descriptive bootstrap interval also reproduced. This verifies the implemented calculation; it does not establish exact exchangeability, secure membership, novel astrophysics or capture.

## Reproducibility contents

The companion bundle contains the frozen specification, post-result validity addendum, pinned input tables/code, three result sets, per-star held-out predictions, permutation statistics, source hashes, repository chronology and independent arithmetic checks. Scientific narrative and machine-readable results are kept separate. No public files or repository commits were changed.

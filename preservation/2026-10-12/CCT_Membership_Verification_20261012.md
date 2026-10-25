# CCT: membership-only verification

12 October 2026 • Private result • No chemical outcomes evaluated in this step

## Result

An alternative public Gaia-based catalog supports **291 of the existing 382 candidate assignments** at its membership-strength score of at least 0.5. **253** also belong to its published high-quality bound-cluster subset. This is a firmer membership foundation than calling stars “members” because they lie near a cluster on the sky.

It does not independently validate the chemical finding. These stars were already chemically selected and inspected in the previous work, and the two membership methods share Gaia astrometric ancestry. No new chemical test or capture inference was performed.

| Frozen-protocol outcome | Stars |
|---|---:|
| Same cluster, unambiguous Gaia identity, score ≥0.5 | 291 |
| Same cluster, score below 0.5 | 6 |
| Not listed, with sound identity and cluster alias | 31 |
| Unresolved cluster alias or stellar identity | 54 |
| Unambiguously assigned to a different catalog cluster | 0 |
| Total | 382 |

The 291 supported assignments cover **21 original cluster labels**, each with at least five stars, and all 291 are marked within the catalog's Jacobi radius. The 253 high-quality bound-cluster assignments are a stricter subset, not a new optimized chemical sample.

## Catalog and exact identity route

We used **Hunt & Reffert (2024), VizieR J/A+A/686/A42**, whose member records contain Gaia DR3 source IDs, cluster IDs, membership-strength scores, astrometry and radial-boundary flags. Its cluster table supplies published aliases, classifications and quality measures. Membership originates in Gaia phase-space clustering rather than C/O abundance. The catalog also contains unbound moving groups; those are kept distinct. [Catalog documentation](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A%2BA/686/A42?format=html&tex=true)

The fixed input contained only APOGEE ID, original cluster label, RA and Dec. It contained no abundance, chemical prediction, residual, statistic or gain fields.

1. Convert the position-encoded APOGEE identifier to its exact 2MASS identifier by removing only the literal “2M” prefix.
2. Resolve it through Gaia DR3's official 2MASS best-neighbour table. Require one Gaia counterpart, one external neighbour, zero mates and a unique reverse identity. Preserve all ambiguous records.
3. Match that Gaia DR3 ID exactly to the HR2024 member table. IDs are never converted to floating point.
4. Compare the original cluster label with the catalog's explicitly listed aliases. Normalize only spelling separators, letter case and leading numeric zeros. No nearest-cluster or fuzzy-name substitution is used.

The Gaia identity bridge is an epoch-aware positional crossmatch, using proper motions when available and positional uncertainties otherwise. This avoids directly equating 2MASS-era coordinates with Gaia positions at epoch 2016.0. Gaia's multiplicity and special-treatment flags were retained. None of the 291 supported assignments has the bridge's multiplicity/duplicate/resolved-source/error-broadening warning bits. [Gaia DR3 crossmatch method](https://gea.esac.esa.int/archive/documentation/GDR3/Catalogue_consolidation/chap_crossmatch/sec_crossmatch_algorithm/)

## Reliability and unresolved cases

The score threshold is a conservative descriptive screen, not a posterior assertion of ≥50% true membership. Hunt & Reffert explain that HDBSCAN scores reflect proximity to the bulk of a cluster and do not incorporate Gaia measurement uncertainties in the clustering. Their census can miss distant, low-density or tail members. A star's absence is therefore **not proof of non-membership**. [Membership method and limitations](https://arxiv.org/html/2303.13424v2)

The high-quality bound-cluster flag uses the published criteria: object Type=o, astrometric significance CST>5 and median CMD class>0.5. The member's radial-boundary flag is recorded separately. [Cluster classification and quality criteria](https://arxiv.org/html/2403.05143v2)

The 54 unresolved cases are deliberately not forced into a favorable answer:

- **52 alias ambiguities:** 10 under ASCC 19, 10 under NGC 2420 and 32 under NGC 7789. The published alias lists map each of these labels to more than one HR internal cluster ID. For example, the NGC 2420 alias appears under both NGC 2420 and Theia 6042. NGC 7789 appears under its main cluster and OCSN 40. The frozen protocol leaves these unresolved even when a canonical main name looks plausible.
- **2 stellar identity ambiguities:** one candidate each under ESO 211 03 and NGC 2158 has two Gaia counterparts in the official 2MASS bridge. Neither is resolved by choosing the nearest or the chemically better match.

Additional astrometric quality values, including RUWE, remain in the per-star table. They were not retrospectively tuned to improve chemistry. Catalog agreement strengthens an assignment; it does not provide certainty of common birth.

## Frozen protocol and bounded retrieval

The protocol was saved before candidate membership results were queried. Its SHA-256 is:

0a28d6415d41e95d1a2dd1b6d9ae5b917b607fb40403715dcc131491b17bd04c

The identity-only input SHA-256 is:

65c827f91de23fafb77789f8961273dcecd677be5af3caff852d1748c78df762

The substantive retrieval comprised **11 bounded public requests and 608,828 response bytes**: four exact-ID bridge queries, four exact-ID member queries, one small cluster identity/quality table and two astrometry queries. No complete stellar catalog was downloaded. Every response, ADQL query, byte count and checksum is preserved. No credentials, paid access, dependency installation or heavy computation were used.

## What this unlocks, and the stopping point

This step supplies a defensible list of catalog-supported assignments and explicitly unresolved cases. It does not re-evaluate whether the earlier chemical prediction gain survives on these subsets. Doing that now would be a post-selection robustness analysis on known data, not a sealed independent validation.

A separate review of the recovered original theory statements found no numerical chemical prediction that separates coherent capture from ordinary common-origin tagging. The companion CCT_Discriminating_Prediction_Gap_20261012.md records that gap. Therefore no additional chemical test is proposed as a capture test. A genuinely uninspected holdout would matter only after a discriminating model prediction and explicit alternative baseline are defined.

The work stops at membership verification, as requested.

## Reproducibility

`MEMBERSHIP_PROTOCOL.md` is the fixed protocol. `candidate_identity_only.csv` is the bounded input. `fetch_membership.py` records the retrieval procedure, while saved CSV responses allow offline reproduction with `classify_membership.py`. The per-star table, per-cluster counts, summary and checksums are included. Reading or re-running these files does not require access to chemical outcomes.

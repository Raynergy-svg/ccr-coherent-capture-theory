# Membership-only verification protocol

Frozen 2026-10-12 16:12 UTC before retrieving or evaluating membership matches for retained candidates.

## Scope and separation

Verify the existing 382-star, 26-label APOGEE candidate assignments using an alternative public astrometric cluster catalog. Input is candidate_identity_only.csv containing only APOGEE_ID, CLUSTER, RA and DEC. Read no chemical abundance, prediction, residual, gain or chemical-selection columns in this step. The sample was already selected and inspected chemically in the previous analysis; it is not a sealed independent chemical validation set.

Use Hunt & Reffert (2024), VizieR J/A+A/686/A42/members and clusters, an updated version of their Gaia DR3 astrometric census. Membership is determined from Gaia phase-space clustering rather than [C/O]. OCCAM also uses Gaia astrometry, so this is an alternative chemical-outcome-independent assignment check, not completely independent measurements. The HR catalog retains unbound moving groups as well as bound clusters; record their classifications.

## Exact identities and epochs

1. Require each APOGEE_ID to match 2M followed by the standard 2MASS position-encoded identifier. Strip only that literal 2M prefix to obtain Gaia's original_ext_source_id. Keep all numeric Gaia source IDs as strings/integers, never floating point.
2. Query Gaia DR3's official tmass_psc_xsc_best_neighbour table by those exact external IDs. Retain source_id, original_ext_source_id, angular_distance, number_of_neighbours, number_of_mates and xm_flag. An unambiguous bridge requires exactly one candidate Gaia ID, one external neighbor, zero mates, and no reverse mapping of the same Gaia ID to a second candidate. Other cases stay ambiguous; do not silently pick the nearest.
3. The published crossmatch handles disparate survey epochs. Do not apply a naive fixed-distance match between APOGEE/2MASS coordinates and Gaia's epoch 2016.0 coordinates. Retrieve Gaia astrometric_params_solved, reference epoch, positions, proper motions, errors, parallax and RUWE for audit where available, but not chemical columns. A position-only fallback is not authorized for final assignment in this step. Missing or ambiguous exact-ID bridges remain unresolved.
4. Query HR members by exact Gaia DR3 IDs. Retrieve only cluster Name and ID, GaiaDR3, Prob, inrj, inrt, position, proper motion, parallax, uncertainties and astrometric quality fields. HR positions are ICRS epoch 2016.0; Gaia DR3 uses the same source IDs and epoch. No DR2-to-DR3 numeric-ID assumption is permitted.

## Cluster identity and decision labels

Retrieve the small (7,167-row) HR cluster name/quality table only: Name, ID, AllNames, Type, CST, CMDCl50 and flags defining merged or GMM memberships. Do not download the 1.29-million-row member catalog. Name normalization uppercases alphabetic tokens, removes spacing/underscores/hyphens and normalizes leading zeros within separate numeric tokens. Match exact normalized aliases listed in AllNames or the main Name. Do not identify clusters from sky proximity or a fuzzy name. Multiple IDs matching an alias remain ambiguous.

For each existing candidate, report one of:
- SAME_CLUSTER_HIGH_SCORE: unambiguous identity bridge and same published cluster/alias, HR Prob >=0.5.
- SAME_CLUSTER_LOW_SCORE: same identity and cluster, 0<=Prob<0.5.
- DIFFERENT_CATALOG_CLUSTER: returned member belongs to a different unambiguously identified catalog cluster.
- NOT_LISTED: identity bridge is sound but Gaia ID absent from HR member table. This is not proof of non-membership because the catalog is incomplete.
- UNRESOLVED_IDENTITY_OR_ALIAS: missing/ambiguous ID bridge or unresolved expected cluster alias.

Retain any multiple membership records; do not choose the chemistry-favorable assignment. HDBSCAN Prob is a proximity-based cluster-strength score, not a calibrated posterior probability. Separately flag support in the published high-quality bound-cluster subset: Type=o, CST>5, CMDCl50>0.5. Record inrj (within Jacobi radius) and inrt separately; do not reject outer members automatically.

## Bounds and stopping point

Small public read-only TAP requests only: up to four batches of 100 identity IDs, four corresponding membership queries, one small cluster name/quality table, and at most two astrometry batches if needed. Total response budget 5 MB; individual responses max 3 MB; TOP/MAXREC caps checked for truncation. Network timeouts 30 seconds, at most one retry per failed read with 2-second backoff. No full catalogs, paid services, credentials, chemistry analyses, N-body runs, installation or publication. Stop after the per-star assignment table and per-cluster summary, or report the exact unresolved data/service gap.

## Primary sources

- HR2024 metadata: https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A%2BA/686/A42?format=html&tex=true
- HR2024 paper: https://arxiv.org/html/2403.05143v2 (high-quality cluster definition)
- HR2023 method: https://arxiv.org/html/2303.13424v2 (astrometric clustering, meaning of membership scores, incompleteness)
- Gaia DR3 2MASS bridge: https://gaia.aip.de/metadata/gaiadr3/tmass_psc_xsc_best_neighbour/
- Gaia crossmatch epoch handling: https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_crossmatches/ssec_dm_tmass_neighbourhood.html (documents the propagation concept; DR3 schema/algorithm flags must remain release-specific)

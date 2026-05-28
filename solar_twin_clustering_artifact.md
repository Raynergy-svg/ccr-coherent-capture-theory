# Negative result: the "15× solar-twin clustering" is a geometric artifact

## The flagged claim

From the cohort sanity check: subgiant solar-twin-like stars
(Teff 5600–5900 K, [Fe/H] ±0.1) in `habitability_v2_targets.csv` show
**0.6 % `cct_cluster = "field"` rate vs 9.0 % baseline → 15× lower**.
Provisional interpretation considered: "solar-twin chemistry concentrates into
cluster templates at 15× the rate of typical chemistry → the Sun's birth
chemistry may not be ordinary."

## Why it had to be tested

`cct_cluster` is assigned by
`habitability_v2.py:246` via a tolerance-normalised Mahalanobis distance in 4D
(C_O, [Mg/Fe], [Si/Fe], [Fe/H]) to each cluster template centroid, with
`min_dist < 4 → assigned` else `field`. If the cluster templates themselves
concentrate near solar (most GALAH-observed open clusters are local thin-disk
near-solar metallicity), then any star at solar abundances is by construction
close to many templates — and the low field rate becomes a geometric tautology.

## What the tests showed

### (1) Where do the 593 cluster templates actually sit?

Tolerance-normalised distance of each template centroid from solar:
**median 1.90, 25th pctile 1.67, 75th 2.25 — all 593 templates lie within 4
(the assignment threshold) of solar.** Per-dim:

| dim | median template | solar | template std |
|---|---|---|---|
| C_O | +0.461 | 0.549 | 0.045 |
| [Mg/Fe] | +0.019 | 0.000 | 0.035 |
| [Si/Fe] | +0.072 | 0.000 | 0.028 |
| [Fe/H] | −0.050 | 0.000 | 0.095 |

The templates cluster around (mostly slightly-sub-)solar abundances. The
tolerance window centred on solar already contains every template.

### (2) Field rate vs distance from solar (rebuilt-template assignment)

| 4D-tol distance from solar | N | field rate |
|---|---|---|
| 0–2 (solar-twin core) | 1,395 | **0.0 %** |
| 2–4 | 10,062 | 4.0 % |
| 4–6 | 777 | **90.0 %** |

Smooth, monotonic, no anomaly at solar — exactly the falloff expected from
template density alone.

### (3) Permutation null

Shuffle the cluster-membership labels among `t9_matched_stars` (preserves the
overall abundance pool and the number of templates and members-per-template,
but destroys any real cluster-coherence signal), rebuild centroids, recompute
the field rate for solar-twin-like cohort stars. 200 permutations:

- **real solar-twin field rate: 0.617 %**
- permuted null mean: **0.760 %**, std 0.088 %, range [0.494 %, 0.988 %]
- **z-score: −1.63, one-sided p = 0.090**

The real value is within the random null. Whatever real cluster-coherence
signal exists at the solar locus is invisible against the geometric
template-centroid concentration.

## Conclusion

**The 15× contrast is a geometric artifact of the `cct_cluster` assignment
method.** It tells us only that:

1. Open-cluster templates are themselves concentrated near solar abundances
   (which is well established and unsurprising for local thin-disk clusters).
2. Solar-abundance stars therefore always sit near some template by 4D
   proximity, regardless of whether they are or are not co-natal members.

There is **no detectable signal of "solar-twin chemistry concentrates into
clusters above geometric expectation."** The headline "Sun not ordinary"
claim cannot be made from this analysis.

To detect a real solar-twin-clustering signal one would need: a method that
controls for template-centroid density (e.g. local-density-corrected
assignment, or comparing chemical-space density at the solar locus against
randomly-drawn populations) and ideally a survey whose cluster sample is not
concentrated near the locus being tested.

## What this does NOT change

- Conclusion #4 (Earth-like chemistry common, 35–44 %) — unaffected.
- Conclusion #5 (Jupiter analog bottleneck) — unaffected.
- The 9D / 8D scorer outputs themselves — unaffected; only the auxiliary
  `cct_cluster` summary column is implicated.
- The actionable list and HD 28888 / HD 183193 picks — unaffected.

## What this does change

- The cohort-caveat note (`cohort_earthlike_caveat.md`) mentioned the 0.6 %
  vs 9 % rate as a "bonus observation worth filing." That note should be
  amended to record that the bonus observation was tested and found to be a
  geometric artifact. (Update applied here.)

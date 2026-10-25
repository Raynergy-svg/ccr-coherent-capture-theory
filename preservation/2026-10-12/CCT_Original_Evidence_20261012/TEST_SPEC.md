# Bounded chemical-foundation reanalysis

Frozen 2026-10-12, before calculating the new chemical test outcome. This is a private prospective analysis specification, not a claim of independent public preregistration. Input: repository commit ff273145c6edf855a810992d6f8afbdf7916a7de.

The earlier T9 sample must be called a sky-aperture sample: its script admits positions within 0.5 degrees of cluster centres, not independently established members. Reproduce its Kruskal-Wallis statistic and quantify repeated sky positions, overlapping apertures, units, and available provenance. Do not interpret this as demonstrated common birth.

Primary falsifiable chemical-branch hypothesis: in the retained APOGEE data, cluster identity improves held-out log10(C/O) prediction beyond specified stellar-parameter and metallicity controls.

Admission: use the retained APOGEE_ID, RV_PROB and PM_PROB. Require both kinematic probabilities >0.5; ignore FEH_PROB for admission. Keep highest-SNR row per ID/cluster, then discard IDs assigned to multiple eligible clusters. Require SNR >=70, 4000 <= TEFF <=5200, 1 <= LOGG <3.8, valid FE_H, C_FE and O_FE, and >=5 retained stars per cluster. If fewer than 5 clusters or 30 stars remain, report insufficient data. These are catalog-probability-selected candidates, not certainty of birth membership.

Outcome is C_FE-O_FE, a logarithmic abundance ratio in dex (not linear C/O mislabeled dex). Use fixed five folds, assigned deterministically within observed clusters with seed 20261012. Nuisance model: standardized TEFF, LOGG, FE_H with all polynomial terms through degree 2 and Ridge(alpha=1). Fit in four folds, predict the fifth. For each held-out star, add the mean training-fold nuisance residual of its cluster to its nuisance prediction. The statistic is the equally cluster-weighted held-out MSE improvement over the nuisance-only model.

Null: 1,999 label permutations, seed 20261013, within fixed bins of width 200 K, 0.5 dex log g and 0.1 dex FE_H anchored at zero. Keep nuisance predictions fixed and recompute cluster-residual predictors for every shuffle. Report bin occupancy and proportion of stars in bins containing more than one cluster. If mixing is weak, call the conditional test limited; do not loosen bins after seeing results. One-sided p=(1+number(null>=observed))/2000. A positive finding requires positive held-out improvement and p<0.01. Otherwise the chosen controlled check fails or is inconclusive; no alternative threshold is substituted.

Interpretation limits even after success: retained data and original survey selection; APOGEE calibration systematics and stellar evolution; bins and quadratic controls are approximate; no raw-spectrum remeasurement, no new observing campaign, no unique capture mechanism, no proof of permanence, no proof of novelty. Describe source overlap and do not multiply GALAH/APOGEE p-values.

Computation budget: one CPU, no dependency installation, 1,999 permutations on small local tables, no N-body simulations, no publication or external mutation.

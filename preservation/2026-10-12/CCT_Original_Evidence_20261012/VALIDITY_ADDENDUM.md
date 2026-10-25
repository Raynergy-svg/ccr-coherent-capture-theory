# Data-validity correction and sensitivity check

2026-10-12 15:59 UTC. The first test result has already been observed and is retained unchanged. These checks are transparently post-result, not additional preregistration.

The OCCAM DR17 primary paper (Myers et al. 2022, arXiv:2206.13650, Table 3) states that PM_PROB=2 is a special fallback to Cantat-Gaudin membership when the proper-motion analysis failed. The retained table lacks CG_PROB. It must not be treated as an ordinary proper-motion probability greater than 0.5. Exclude RV_PROB or PM_PROB outside [0,1] for a corrected strict-probability run; otherwise retain the complete original selection and method. This correction is required by data semantics, not chosen to improve the outcome. Catalog cluster-quality flags are also absent, so the sample remains probability-selected candidates rather than verified members.

Run one additional conservative sensitivity: permute labels within covariate bins AND original fold. This preserves each cluster's fold allocation as well as its bin allocation. The original fixed stratified folds otherwise make observed cluster training counts better balanced than shuffled ones. Report resulting weak mixing if bins become sparse. No new models, thresholds, or outcome definitions are selected.

The computations are limited to two additional 1,999-permutation runs using the same small tables and one CPU. Do not replace the first outcome; report all outcomes and limitations.

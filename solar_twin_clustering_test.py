"""
Honest test of the '15x' solar-twin-clustering finding.

Claim to test: subgiant solar-twin-like stars (T 5600-5900, [Fe/H] ±0.1) have
0.6% field rate vs 9.0% baseline in habitability_v2_targets.csv -> 15x lower
field rate -> claimed as "Sun not ordinary."

Structural artifact hypothesis: cct_cluster is assigned by Mahalanobis distance
in 4D (C_O, Mg/Fe, Si/Fe, [Fe/H]) to the 655 cluster TEMPLATES. If those
templates themselves concentrate near solar abundances (because most GALAH
open clusters are local thin-disk near-solar), then any solar-abundance star
will sit close to many templates by pure geometry -- and the low field rate
is a tautology, not a signal.

Tests run:
  (1) Where do the cluster template centroids actually sit in abundance space?
      Compare template centroid distribution to the solar point.
  (2) Is field rate a smooth function of distance from solar / from the
      template-centroid locus?  If monotonic / smooth -> geometric artifact.
  (3) Permutation null: shuffle cluster *memberships* (preserving N per cluster
      and overall abundance pool) -> rebuild centroids -> recompute field rate
      for solar-twin-like stars.  If the permuted null gives similar 0.6%,
      the 15x is artifact; if the real cohort gives much lower, real signal.
"""
import numpy as np, pandas as pd

stars = pd.read_csv("t9_matched_stars.csv").dropna(subset=["C_O","mg_fe","si_fe","fe_h","cluster_name"])
cohort = pd.read_csv("habitability_v2_targets.csv")
TOL = np.array([0.08, 0.05, 0.05, 0.10])
DIMS = ["C_O","mg_fe","si_fe","fe_h"]
SOLAR = np.array([0.549, 0.0, 0.0, 0.0])  # solar C/O (number), Mg/Fe, Si/Fe, [Fe/H]
THRESH = 4.0

# ----- build the actual templates (replicate habitability_v2.py logic) -----
templates_real = (stars.groupby("cluster_name")[DIMS]
                       .filter(lambda x: len(x) >= 3)
                       .groupby(stars["cluster_name"])
                       .mean())
# the above is brittle; just rebuild cleanly
g = stars.groupby("cluster_name")
templates_real = pd.DataFrame({
    "n": g.size(),
    **{d: g[d].mean() for d in DIMS}
}).query("n >= 5")  # match T9 min
T_real = templates_real[DIMS].values
print(f"templates rebuilt: {len(T_real)}")

# ----- (1) where do templates sit relative to solar? -----
solar_norm_dist = np.sqrt(((T_real - SOLAR) / TOL)**2).sum(axis=1)**0.5
print("\n(1) cluster-template centroid distances from solar (tolerance-normalised):")
print(f"  median: {np.median(solar_norm_dist):.2f}  | 25th pctile: {np.percentile(solar_norm_dist,25):.2f}  | 75th: {np.percentile(solar_norm_dist,75):.2f}")
print(f"  fraction of templates within 4 (the assignment threshold) of solar: "
      f"{(solar_norm_dist<4).sum()}/{len(T_real)} = {(solar_norm_dist<4).mean():.1%}")
# template means in each dim
for i,d in enumerate(DIMS):
    print(f"  {d:6s} template centroids: median={np.median(T_real[:,i]):+.4f}  "
          f"std={T_real[:,i].std():.4f}  solar={SOLAR[i]:.4f}")

# ----- assigment function (vectorised) -----
def assign_field(stars_xyz, templates):
    # stars_xyz shape (N,4), templates (M,4). return fraction with min_dist >= THRESH
    diff = stars_xyz[:,None,:] - templates[None,:,:]
    d = np.sqrt(((diff/TOL)**2).sum(axis=-1))   # (N,M)
    mind = d.min(axis=1)
    return (mind >= THRESH)

# Use full subgiant cohort (cct_cluster column already gives reality)
# But to run permutations we need raw 4D values. They're in cohort:
cohort4D = cohort[["C_O","mg_fe","si_fe","fe_h"]].dropna().values
field_real = assign_field(cohort4D, T_real)
print(f"\nreproduce cohort field rate from rebuilt templates: {field_real.mean():.3%}  (cohort file: {(cohort.cct_cluster=='field').mean():.3%})")

# solar-twin-like subset (need teff & feh from cohort + abundances)
m = cohort.dropna(subset=["C_O","mg_fe","si_fe","fe_h","teff","fe_h"])
sun = m[(m.teff>5600)&(m.teff<5900)&(m.fe_h.between(-0.1,0.1))]
sun4D = sun[DIMS].values
field_sun_real = assign_field(sun4D, T_real)
print(f"\nsolar-twin-like (subgiant, T5600-5900, [Fe/H]+/-0.1): N={len(sun4D)}, field rate={field_sun_real.mean():.3%}")

# ----- (2) field rate as a function of distance from solar -----
print("\n(2) field rate by distance from solar (using rebuilt templates):")
all_dist_solar = np.sqrt(((cohort4D-SOLAR)/TOL)**2).sum(axis=1)**0.5
all_field = assign_field(cohort4D, T_real)
bins = [0,2,4,6,8,10,15,25,1e3]
for lo,hi in zip(bins[:-1],bins[1:]):
    mask = (all_dist_solar>=lo) & (all_dist_solar<hi)
    if mask.sum()>20:
        print(f"  dist 4D-solar in [{lo:>2g},{hi:>4g}):  N={mask.sum():>5}  field rate={all_field[mask].mean():.3%}")

# ----- (3) permutation null: shuffle which cluster each star belongs to -----
print("\n(3) PERMUTATION NULL: shuffle cluster memberships (preserving sizes), rebuild centroids, re-assess")
rng = np.random.default_rng(42)
n_perm = 200
real_field_sun = field_sun_real.mean()
perm_field_sun = np.empty(n_perm)
sizes = templates_real["n"].values
for k in range(n_perm):
    # shuffle the cluster_name labels among the t9 matched_stars
    perm_labels = rng.permutation(stars["cluster_name"].values)
    pg = pd.DataFrame({"cluster_name": perm_labels, **{d: stars[d].values for d in DIMS}}).groupby("cluster_name")
    p_tmpl = pd.DataFrame({"n": pg.size(), **{d: pg[d].mean() for d in DIMS}}).query("n>=5")[DIMS].values
    if len(p_tmpl)==0: continue
    perm_field_sun[k] = assign_field(sun4D, p_tmpl).mean()
print(f"  real solar-twin field rate: {real_field_sun:.3%}")
print(f"  permuted null mean: {perm_field_sun.mean():.3%}  std: {perm_field_sun.std():.3%}  range [{perm_field_sun.min():.3%}, {perm_field_sun.max():.3%}]")
zscore = (real_field_sun - perm_field_sun.mean())/perm_field_sun.std() if perm_field_sun.std()>0 else 0
print(f"  z-score (real vs permuted null): {zscore:+.2f}")
p_one = (perm_field_sun <= real_field_sun).mean()
print(f"  one-sided p (fraction of perms with field rate <= real): {p_one:.3f}")
if abs(zscore)<2:
    print("  => REAL FIELD RATE IS WITHIN PERMUTATION NULL: low solar-twin field rate is a GEOMETRIC ARTIFACT")
    print("     (cluster templates concentrate near solar, so any solar-twin star is always near some template)")
else:
    print("  => Real field rate is OUTSIDE the permuted null -> there is a residual signal beyond geometry")
print("\nDONE")

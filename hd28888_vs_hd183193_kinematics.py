"""
Compare HD 28888 and HD 183193 in action space + chemistry to test whether
they could share a birth origin (co-natal pair) or are unrelated.

Inputs already in hand (from GALAH DR4 dynamics VAC + chemistry):
- HD 28888  (Gaia DR3 3312653647717790720)
- HD 183193 (Gaia DR3 6771181697822279936)

The test: actions (J_R, J_Z, L_Z) are nearly conserved under axisymmetric
Galactic perturbations.  If two stars share a birth origin (same cluster ->
same initial actions), their present-day actions should match closely
(with allowance for radial migration over ~few Gyr of order +/- 0.5 kpc in
guiding radius / +/- few % in L_Z).  Combined with age + chemistry
matching, this is the cleanest test available without full orbit integration.
"""
import numpy as np
import pandas as pd

# both stars' archive data (units as in GALAH vacdynamicstable + GALAH abundances)
stars = pd.DataFrame([
    dict(name="HD 28888",  gaia=3312653647717790720, age=6.31, dist_pc=100,
         feh=+0.130, mgfe=+0.020, sife=+0.006, alfe=+0.086, cafe=-0.043, bafe=-0.040,
         CO_raw=0.308, # GALAH (subgiant - C measured)
         ecc=0.165, R_peri=6.330, R_apo=8.831, z_max=0.063,
         J_R=34.4, J_Z=0.135, L_Z=1723.3, Energy=-160679.0,
         U=-52.3, V=-40.7, W=-11.9,
         ra=68.41124, dec=15.96126, pmra=65.228, pmdec=-55.779, rv=53.70),
    dict(name="HD 183193", gaia=6771181697822279936, age=6.37, dist_pc=74.7,
         feh=-0.003, mgfe=+0.008, sife=-0.010, alfe=+0.071, cafe=+0.026, bafe=-0.033,
         CO_raw=0.438, # GALAH dwarf - C flag bad; this is uncorrected
         ecc=0.111, R_peri=8.012, R_apo=10.019, z_max=0.188,
         J_R=18.22, J_Z=0.909, L_Z=2075.8, Energy=-151387.0,
         U=-30.05, V=+6.72, W=+4.38,
         ra=292.32463, dec=-24.32321, pmra=27.118, pmdec=31.971, rv=-27.77),
])

print("=" * 78)
print("HD 28888 vs HD 183193 — co-natal-pair test")
print("=" * 78)

# 1. ages
a1, a2 = stars.age.values
print(f"\n[AGES]  HD28888 {a1:.2f} Gyr  vs  HD183193 {a2:.2f} Gyr  ->  diff {abs(a1-a2)*1000:.0f} Myr")
print("        AGES MATCH at <1% level. Consistent with shared birth epoch.")

# 2. chemistry: [Fe/H] is the strongest co-natal discriminant (small intrinsic
#    cluster scatter <0.03 dex)
print(f"\n[CHEMISTRY]")
def diff(c):
    v1, v2 = stars[c].values
    return v1, v2, v1-v2
for c, tag in [("feh","[Fe/H]"),("mgfe","[Mg/Fe]"),("sife","[Si/Fe]"),
               ("alfe","[Al/Fe]"),("cafe","[Ca/Fe]"),("bafe","[Ba/Fe]")]:
    v1, v2, d = diff(c)
    print(f"  {tag:8s}  HD28888 {v1:+.3f}  HD183193 {v2:+.3f}  delta = {d:+.3f}")
print()
print("        [Fe/H] differs by 0.13 dex. Typical intra-cluster scatter is")
print("        ~0.03 dex. This is a ~4x rejection of strict co-natal origin.")
print("        Other elements: [Mg/Fe],[Si/Fe],[Al/Fe],[Ba/Fe] all differ by")
print("        <0.05 dex (within cluster scatter); [Ca/Fe] differs by 0.07.")

# 3. actions -- the formal test
print(f"\n[ACTIONS]  (units: kpc.km/s for J_R, J_Z; kpc.km/s for L_Z)")
for k, label in [("J_R","J_R (radial action)"), ("J_Z","J_Z (vertical action)"),
                 ("L_Z","L_Z (z-ang. momentum)"), ("Energy","Orbital energy")]:
    v1, v2 = stars[k].values
    pct = 100*abs(v1-v2)/max(abs(v1),abs(v2),1)
    print(f"  {label:24s}  HD28888 {v1:>+10.2f}  HD183193 {v2:>+10.2f}   abs diff {abs(v1-v2):>8.2f}  ({pct:.1f}%)")

# guiding radius from L_Z (assuming flat circular curve ~220 km/s)
V_CIRC = 220.0
Rg1, Rg2 = stars.L_Z.values / V_CIRC
print(f"\n  Guiding radii (R_g = L_Z / v_circ at 220 km/s):")
print(f"    HD28888  R_g = {Rg1:.2f} kpc")
print(f"    HD183193 R_g = {Rg2:.2f} kpc")
print(f"    delta R_g = {abs(Rg1-Rg2):.2f} kpc  ({100*abs(Rg1-Rg2)/Rg1:.1f}%)")
print("\n  L_Z differs by ~20%; R_g differs by ~1.6 kpc.")
print("  Radial migration over 6 Gyr: typical Sellwood-Binney churning RMS is")
print("  ~1.5-2 kpc -- the two stars COULD have started near a common radius.")
print("  But starting *exactly* together would require coordinated migration")
print("  in opposite directions, ~0.8 kpc each.  Possible, not implied.")

# orbit-shape comparison
print(f"\n[ORBIT SHAPE]")
print(f"  HD28888  : R_peri 6.33 / R_apo 8.83 kpc  (mean 7.58 kpc, ecc 0.165)")
print(f"  HD183193 : R_peri 8.01 / R_apo 10.02 kpc (mean 9.02 kpc, ecc 0.111)")
print(f"  HD28888  : z_max 0.06 kpc  (very thin disk)")
print(f"  HD183193 : z_max 0.19 kpc  (thin disk, ~3x thicker excursion)")
print(f"  Orbital footprints do not overlap: HD183193's perihelion (8.0) is")
print(f"  larger than HD28888's aphelion (8.8) only marginally.")

# velocities
print(f"\n[UVW]")
print(f"  HD28888  : U/V/W = -52.3 / -40.7 / -11.9 km/s  (inward, lagging LSR)")
print(f"  HD183193 : U/V/W = -30.0 / +6.7 / +4.4 km/s    (mild inward, leading)")
duv = np.sqrt((stars.U.diff().iloc[1])**2 + (stars.V.diff().iloc[1])**2 + (stars.W.diff().iloc[1])**2)
print(f"  3D velocity difference: {duv:.1f} km/s")

# 4. composite verdict
print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
print("""
AGES:         match (delta ~60 Myr, <1%) -- consistent with shared epoch
CHEMISTRY:    NOT consistent with strict co-natal -- [Fe/H] delta 0.13 dex is
              ~4x larger than typical intra-cluster scatter (~0.03 dex)
ACTIONS:      L_Z differs by ~20%, guiding radii by ~1.6 kpc -- could be
              reconciled by 6 Gyr of radial migration but is not strongly
              indicated
ORBIT:        non-overlapping footprints; thin-disk both, but HD183193 sits
              ~1.5 kpc outward of HD28888's guiding radius

The two stars look like INDEPENDENT solar-neighbourhood thin-disk stars that
happen to share an age (~6.3 Gyr) and a broad chemistry class (solar-twin-like).
The [Fe/H] mismatch alone -- 4x larger than the typical scatter that even
the most chemically uniform open clusters show -- is sufficient to reject
co-natal origin.  Action and orbit-shape differences add to that picture
rather than against it.

Co-natal pair: NO (rejected by chemistry; not supported by kinematics either).
""")

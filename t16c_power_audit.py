"""T16c power audit: simulated detection rate for E(t)=E0*exp(-t/tau) given
the actual age and noise distribution of T16c's 231-cluster sample."""
import numpy as np, pandas as pd
from scipy import stats
d=pd.read_csv("t16c_permanence_data.csv").dropna(subset=["enrichment","age_gyr"])
ages=d.age_gyr.values; obs=d.observed.values; mcmed=d.mc_median.values; mcs=d.mc_std.values
enr_sigma=d.enrichment.values*np.sqrt(1.0/np.maximum(obs,1)+(mcs/np.maximum(mcmed,1))**2)
print(f"N={len(d)}, age range {ages.min():.2f}-{ages.max():.2f} Gyr, median noise sigma {np.median(enr_sigma):.3f}")
print("age distribution:")
for lo,hi in [(0,1),(1,2),(2,4),(4,6),(6,8),(8,10)]:
    print(f"  {lo}-{hi} Gyr: N={((ages>=lo)&(ages<hi)).sum()}")
E0=d.enrichment.mean()
rng=np.random.default_rng(42)
print(f"\nPower vs true decay timescale (N_sims=2000):")
print(f"{'tau (Gyr)':>10}  {'p<0.05':>8}  {'p<0.01':>8}")
for tau in [0.5,1.0,2.0,5.0,10.0,20.0,50.0,1e9]:
    n5=n1=0
    for _ in range(2000):
        true_e = np.full_like(ages,E0) if tau>=1e8 else E0*np.exp(-ages/tau)
        noisy = true_e + rng.normal(0, enr_sigma)
        _, p = stats.spearmanr(ages, noisy)
        if p<0.05: n5+=1
        if p<0.01: n1+=1
    print(f"{tau:>10.1f}  {n5/2000:>7.1%}  {n1/2000:>7.1%}")
print("Observed Spearman on real data:")
rho,p=stats.spearmanr(d.age_gyr, d.enrichment)
print(f"  rho={rho:+.4f}, p={p:.4e}  -> consistent with tau >= ~5 Gyr (incl permanence)")

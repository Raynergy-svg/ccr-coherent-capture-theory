"""Independent output identities and descriptive (not confirmatory) uncertainty."""
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parent
checks={}
for mode in ['original','strict_probability','strict_probability_fold_preserved']:
    folder=R if mode=='original' else R/mode
    r=json.loads((folder/'results.json').read_text())
    p=pd.read_csv(folder/'qualified_apogee_heldout_predictions.csv')
    gains=(p.log_CO-p.prediction_nuisance)**2-(p.log_CO-p.prediction_cluster)**2
    gain=gains.groupby(p.CLUSTER).mean().mean()
    null=pd.read_csv(folder/'permutation_null.csv').mse_gain
    perm_p=(1+(null>=gain).sum())/(len(null)+1)
    assert p.APOGEE_ID.is_unique
    assert np.isclose(gain,r['test']['mse_gain'],rtol=1e-12,atol=1e-15)
    assert np.isclose(perm_p,r['test']['p_permutation'])
    if mode!='original': assert p.RV_PROB.between(.5,1,inclusive='right').all() and p.PM_PROB.between(.5,1,inclusive='right').all()
    checks[mode]={'unique_ids':True,'mse_identity':True,'permutation_p_identity':True,'n':len(p)}

c=pd.read_csv(R/'strict_probability/cluster_gains.csv')
v=c.mse_gain.to_numpy(); rng=np.random.default_rng(20261014)
b=rng.choice(v,size=(10000,len(v)),replace=True).mean(axis=1)
descriptive={'not_preregistered':'fixed-prediction cluster bootstrap; does not refit selection or nuisance models and is not a formal generalization interval','n_resamples':10000,'mean_mse_gain_dex2':float(v.mean()),'percentile_95_interval_dex2':np.quantile(b,[.025,.975]).tolist(),'positive_clusters':int((v>0).sum()),'clusters':len(v),'leave_one_cluster_fixed_prediction_mean_range_dex2':[float(((v.sum()-v)/(len(v)-1)).min()),float(((v.sum()-v)/(len(v)-1)).max())]}
# Rounded periods quoted by the 2018 discovery paper. This is an arithmetic
# consistency check, not an independent proof of librating resonant angles.
periods=np.array([2.35,3.56,5.40,8.26,12.76])
ratios=periods[1:]/periods[:-1]
k2={'source':'https://arxiv.org/abs/1801.03874','periods_days':periods.tolist(),'adjacent_ratios':ratios.tolist(),'relative_offset_from_3_to_2':(ratios/1.5-1).tolist(),'interpretation':'The historical report lists K2-138 as a low-resonance capture candidate, but its inner five planets were already described as a near-3:2 chain in the discovery paper. Near commensurability is not by itself dynamical resonance proof.'}
age=pd.read_csv(R/'repo/t16c_permanence_data.csv').age_gyr
coverage={'rows':len(age),'finite_ages':int(age.notna().sum()),'max_age_gyr':float(age.max()),'under1_gyr':int((age<1).sum())}
out={'verification':checks,'descriptive_uncertainty':descriptive,'k2_138_arithmetic':k2,'t16c_age_coverage':coverage}
(R/'verification_and_diagnostics.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

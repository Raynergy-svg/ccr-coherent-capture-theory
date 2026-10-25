"""Independent, read-only checks of frozen CCT tables. No source-file writes or network."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
R=Path('/workspace/scratch/5f1b1bd8b2b3/cct_original_audit_20261012')
O=R/'strict_probability_fold_preserved'
raw=pd.read_csv(R/'repo/tapogee_matched_stars.csv')
p=pd.read_csv(O/'qualified_apogee_heldout_predictions.csv')
null=pd.read_csv(O/'permutation_null.csv')
result=json.loads((O/'results.json').read_text())
# Independently reconstruct admission from raw source, avoiding the author helper.
elig=raw.loc[raw.RV_PROB.gt(.5)&raw.RV_PROB.le(1)&raw.PM_PROB.gt(.5)&raw.PM_PROB.le(1)].copy()
elig=elig.loc[elig.groupby(['APOGEE_ID','CLUSTER'],sort=False).SNR.idxmax()]
unique_id=elig.groupby('APOGEE_ID').CLUSTER.nunique()
elig=elig[elig.APOGEE_ID.isin(unique_id[unique_id.eq(1)].index)]
elig=elig[elig.SNR.ge(70)&elig.TEFF.ge(4000)&elig.TEFF.le(5200)&elig.LOGG.ge(1)&elig.LOGG.lt(3.8)]
elig=elig[np.isfinite(elig[['FE_H','C_FE','O_FE']]).all(axis=1)]
elig=elig[elig.CLUSTER.map(elig.CLUSTER.value_counts()).ge(5)].sort_values(['CLUSTER','APOGEE_ID']).reset_index(drop=True)
assert list(zip(elig.APOGEE_ID,elig.CLUSTER))==list(zip(p.APOGEE_ID,p.CLUSTER))
assert p.APOGEE_ID.is_unique
assert len(p.drop_duplicates(['RA','DEC']))==len(p)
assert np.allclose(p.log_CO,p.C_FE-p.O_FE,rtol=0,atol=1e-14)
assert np.isfinite(p[['FE_H','C_FE','O_FE','TEFF','LOGG','SNR']]).all().all()
assert not (p[['FE_H','C_FE','O_FE']]<=-9000).any().any()
y=(p.C_FE-p.O_FE).to_numpy(); X=p[['TEFF','LOGG','FE_H']].to_numpy()
# Rebuild every training-only nuisance fit and cluster residual offset.
base=np.zeros(len(p)); pred=np.zeros(len(p)); cache=[]
for f in range(5):
    train=p.fold.ne(f).to_numpy(); test=~train
    assert not set(p.loc[train,'APOGEE_ID'])&set(p.loc[test,'APOGEE_ID'])
    m=make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),Ridge(alpha=1))
    m.fit(X[train],y[train]); base[test]=m.predict(X[test])
    resid=y[train]-m.predict(X[train]); means=pd.Series(resid,index=p.CLUSTER[train]).groupby(level=0).mean()
    pred[test]=base[test]+p.CLUSTER[test].map(means)
    cache.append((train,test,resid))
assert np.allclose(base,p.prediction_nuisance,rtol=0,atol=1e-12)
assert np.allclose(pred,p.prediction_cluster,rtol=0,atol=1e-12)
b_mse=pd.Series((y-base)**2).groupby(p.CLUSTER).mean().mean()
c_mse=pd.Series((y-pred)**2).groupby(p.CLUSTER).mean().mean()
gain_by_cluster=pd.Series((y-base)**2-(y-pred)**2).groupby(p.CLUSTER).mean()
gain=float(gain_by_cluster.mean()); nge=int((null.mse_gain>=gain).sum()); perm_p=(nge+1)/(len(null)+1)
assert abs(gain-result['test']['mse_gain'])<1e-14
assert perm_p==result['test']['p_permutation']
# Independently recreate all fold labels and covariate/fold bin keys.
rng=np.random.default_rng(20261012); rebuilt=np.zeros(len(p),dtype=int)
for c in sorted(p.CLUSTER.unique()):
    idx=np.where(p.CLUSTER.eq(c))[0]; idx=rng.permutation(idx); rebuilt[idx]=np.arange(len(idx))%5
assert np.array_equal(rebuilt,p.fold.to_numpy())
key=np.column_stack([np.floor(X/np.array([200,.5,.1])).astype(int),rebuilt])
_,bin_id=np.unique(key,axis=0,return_inverse=True)
assert np.array_equal(bin_id,p.bin)
label,_=pd.factorize(p.CLUSTER,sort=True); counts=np.bincount(label)
indices=[np.where(bin_id==i)[0] for i in sorted(np.unique(bin_id))]
mobile=sum(len(i) for i in indices if len(set(label[i]))>1)
# Five deterministic shuffles verify the published RNG, normalization and scoring.
rng=np.random.default_rng(20261013); spot=[]
for k in range(5):
    lab=label.copy()
    for ii in indices: lab[ii]=rng.permutation(lab[ii])
    assert np.array_equal(np.bincount(lab),counts)
    for f in range(5):
        assert np.array_equal(np.bincount(lab[p.fold.eq(f)],minlength=len(counts)),np.bincount(label[p.fold.eq(f)],minlength=len(counts)))
    shuf_pred=base.copy()
    for train,test,resid in cache:
        means=pd.Series(resid,index=lab[train]).groupby(level=0).mean()
        shuf_pred[test]=base[test]+pd.Series(lab[test]).map(means).to_numpy()
    sg=float(pd.Series((y-base)**2-(y-shuf_pred)**2).groupby(lab).mean().mean())
    assert abs(sg-null.mse_gain.iloc[k])<1e-14
    spot.append(sg)
# The first-observed run included 35 sentinel-bearing stars, plus 3 valid-probability
# stars lost in the corrected run because their cluster falls below five stars.
old=pd.read_csv(R/'qualified_apogee_heldout_predictions.csv')
old_sentinel=int(((old.PM_PROB>1)|(old.RV_PROB>1)).sum())
# Verify retained abundance provenance and original STAR_BAD cut with the available table.
a=pd.read_csv(R/'repo/apogee_occam_abundances.csv')
for row in p.itertuples():
    matches=a[a.APOGEE_ID.eq(row.APOGEE_ID)&a.SNR.eq(row.SNR)]
    same=matches[np.isclose(matches.C_FE,row.C_FE)&np.isclose(matches.O_FE,row.O_FE)&np.isclose(matches.FE_H,row.FE_H)]
    assert len(same)>0
    assert any((int(v)&(1<<23))==0 for v in same.ASPCAPFLAG)
# Descriptive cluster resampling; does not refit anything or provide new-cluster inference.
rng=np.random.default_rng(20261014)
boot=rng.choice(gain_by_cluster.to_numpy(),size=(10000,len(gain_by_cluster)),replace=True).mean(axis=1)
ci=np.quantile(boot,[.025,.975]).tolist()
source_files=[R/'TEST_SPEC.md',R/'VALIDITY_ADDENDUM.md',R/'reanalysis.py',R/'repo/tapogee_matched_stars.csv',R/'repo/t9_matched_stars.csv',R/'repo/apogee_occam_abundances.csv']+[O/x for x in ['results.json','qualified_apogee_heldout_predictions.csv','permutation_null.csv','cluster_gains.csv']]
report={'independent_check_scope':'source selection, identity and position uniqueness, training-only fits, predictions, weights, 5 deterministic null draws, complete stored-null exceedance count, fixed-prediction cluster resampling; not full null regeneration or raw catalog/spectra audit','selected_stars':len(p),'clusters':p.CLUSTER.nunique(),'unique_ids':p.APOGEE_ID.nunique(),'unique_coordinates':len(p.drop_duplicates(['RA','DEC'])),'minimum_cluster_n':int(p.CLUSTER.value_counts().min()),'first_run_sentinel_stars':old_sentinel,'fold_sizes':p.fold.value_counts().sort_index().to_dict(),'balanced_baseline_mse':float(b_mse),'balanced_cluster_mse':float(c_mse),'balanced_mse_gain':gain,'fraction_mse_improvement':float(gain/b_mse),'baseline_rmse':float(np.sqrt(b_mse)),'cluster_rmse':float(np.sqrt(c_mse)),'stored_null_n':len(null),'stored_null_exceedances':nge,'permutation_p':perm_p,'bin_fold_strata':len(indices),'mixed_strata_stars':mobile,'fixed_strata_stars':len(p)-mobile,'first_5_nulls_verified':spot,'clusters_positive_gain':int(gain_by_cluster.gt(0).sum()),'descriptive_cluster_bootstrap_95_interval':ci,'source_hashes':{str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}}
print(json.dumps(report,indent=2))

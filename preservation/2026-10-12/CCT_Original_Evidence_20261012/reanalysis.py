"""Private bounded check; see TEST_SPEC.md. No network or source-data mutation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json, hashlib, sys, platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

ROOT=Path(__file__).resolve().parent
REPO=ROOT/'repo'
MODE=sys.argv[1] if len(sys.argv)>1 else 'original'
OUT=ROOT if MODE=='original' else ROOT/MODE
OUT.mkdir(exist_ok=True)
result={'scope':'retained-data reanalysis, not new observations', 'spec_sha256':hashlib.sha256((ROOT/'TEST_SPEC.md').read_bytes()).hexdigest()}
result['versions']={'python':platform.python_version(), 'numpy':np.__version__, 'pandas':pd.__version__}
result['inputs']={f:hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in ['t9_matched_stars.csv','tapogee_matched_stars.csv','tapogee.py','t9_cluster_coherence.py']}

def t9_summary(path):
    d=pd.read_csv(path)
    counts=d.groupby('cluster_name').size()
    d=d[d.cluster_name.isin(counts[counts>=3].index)].copy()
    groups=[v.C_O.to_numpy() for _,v in d.groupby('cluster_name')]
    kw=stats.kruskal(*groups)
    sizes=d.groupby(['ra','dec']).size()
    rv=d.groupby('cluster_name').rv_gaia_dr3.std()
    coordinate_sets=d.groupby('cluster_name').apply(lambda g:tuple(sorted(zip(g.ra,g.dec))),include_groups=False)
    identical={}
    for name,signature in coordinate_sets.items(): identical.setdefault(signature,[]).append(name)
    X=np.column_stack([np.ones(len(d)),d.fe_h])
    residual=d.C_O-np.linalg.lstsq(X,d.C_O,rcond=None)[0]@X.T
    kr=stats.kruskal(*[residual[d.cluster_name==n] for n in sorted(d.cluster_name.unique())])
    return {'rows':len(d),'clusters':len(groups),'unique_coordinates':len(sizes),'coordinates_reused':int((sizes>1).sum()),'extra_rows_due_reuse':len(d)-len(sizes),'max_apertures_per_coordinate':int(sizes.max()),'kw_H':float(kw.statistic),'kw_p':float(kw.pvalue),'feh_residual_kw_H':float(kr.statistic),'feh_residual_kw_p':float(kr.pvalue),'median_aperture_rv_sd_kms':float(rv.median()),'apertures_rv_sd_gt10':int((rv>10).sum()),'identical_coordinate_groups':[v for v in identical.values() if len(v)>1]}

result['t9_current']=t9_summary(REPO/'t9_matched_stars.csv')
if (ROOT/'t9_root_matched_stars.csv').stat().st_size:
    result['t9_root']=t9_summary(ROOT/'t9_root_matched_stars.csv')

raw=pd.read_csv(REPO/'tapogee_matched_stars.csv')
eligible=raw.groupby('CLUSTER').size()
groups=[g.C_O.values for _,g in raw[raw.CLUSTER.isin(eligible[eligible>=3].index)].groupby('CLUSTER')]
kw=stats.kruskal(*groups)
result['apogee_original']={'rows':len(raw),'unique_ids':raw.APOGEE_ID.nunique(),'clusters_all':raw.CLUSTER.nunique(),'clusters_ge3':len(groups),'kw_H':float(kw.statistic),'kw_p':float(kw.pvalue),'same_id_multi_cluster':int((raw.groupby('APOGEE_ID').CLUSTER.nunique()>1).sum()),'rows_without_both_kinematic_probabilities':int((~((raw.RV_PROB>.5)&(raw.PM_PROB>.5))).sum())}
d=raw[(raw.RV_PROB>.5)&(raw.PM_PROB>.5)].copy()
if MODE!='original':
    d=d[(d.RV_PROB<=1)&(d.PM_PROB<=1)].copy()
result['mode']=MODE
selection={'rv_and_pm_rows':len(d)}
d=d.sort_values(['SNR','APOGEE_ID','CLUSTER'],ascending=[False,True,True],kind='stable').drop_duplicates(['APOGEE_ID','CLUSTER'])
selection['deduplicated_id_cluster']=len(d)
ambiguity=d.groupby('APOGEE_ID').CLUSTER.nunique()
selection['ambiguous_ids_removed']=int((ambiguity>1).sum())
d=d[d.APOGEE_ID.isin(ambiguity[ambiguity==1].index)]
d=d[(d.SNR>=70)&d.TEFF.between(4000,5200)&(d.LOGG>=1)&(d.LOGG<3.8)].dropna(subset=['FE_H','C_FE','O_FE'])
selection['parameter_selected_stars']=len(d)
counts=d.groupby('CLUSTER').size()
d=d[d.CLUSTER.isin(counts[counts>=5].index)].sort_values(['CLUSTER','APOGEE_ID']).reset_index(drop=True)
selection['final_stars']=len(d); selection['final_clusters']=d.CLUSTER.nunique()
result['selection']=selection
if len(d)<30 or d.CLUSTER.nunique()<5:
    result['decision']='INSUFFICIENT_QUALIFIED_SUBSET'
else:
    assert not d.APOGEE_ID.duplicated().any()
    labels,names=pd.factorize(d.CLUSTER,sort=True)
    nc=len(names); n=len(d); counts=np.bincount(labels,minlength=nc)
    rng=np.random.default_rng(20261012)
    folds=np.empty(n,dtype=int)
    for g in range(nc):
        ii=np.where(labels==g)[0]; ii=rng.permutation(ii); folds[ii]=np.arange(len(ii))%5
    X=d[['TEFF','LOGG','FE_H']].to_numpy(); y=(d.C_FE-d.O_FE).to_numpy()
    baseline=np.empty(n); fit_cache=[]
    for f in range(5):
        test=folds==f; train=~test
        model=make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),Ridge(alpha=1))
        model.fit(X[train],y[train])
        pred=model.predict(X[test]); baseline[test]=pred
        fit_cache.append((train,test,y[train]-model.predict(X[train]),pred))
    # All nuisance fits use X/y/folds only, not labels. Therefore recomputing
    # them inside every label permutation gives exactly the cached values.
    # The label-dependent cluster training and held-out scoring are repeated.
    def score(lab, return_pred=False):
        pred=baseline.copy()
        for train,test,resid,basepred in fit_cache:
            cnt=np.bincount(lab[train],minlength=nc)
            sums=np.bincount(lab[train],weights=resid,minlength=nc)
            mean=np.divide(sums,cnt,out=np.zeros(nc),where=cnt>0)
            pred[test]=basepred+mean[lab[test]]
        gain=(y-baseline)**2-(y-pred)**2
        balanced=np.bincount(lab,weights=gain,minlength=nc)/counts
        return (float(np.mean(balanced)),pred,balanced) if return_pred else float(np.mean(balanced))
    observed,pred,by_cluster=score(labels,True)
    bin_values=np.floor(X/np.array([200,.5,.1])).astype(int)
    _,bins=np.unique(bin_values,axis=0,return_inverse=True)
    if MODE=='strict_probability_fold_preserved':
        _,bins=np.unique(np.column_stack([bin_values,folds]),axis=0,return_inverse=True)
    idx_bins=[np.where(bins==i)[0] for i in np.unique(bins)]
    mix_bins=[ii for ii in idx_bins if len(np.unique(labels[ii]))>1]
    mobile=sum(map(len,mix_bins))
    rng=np.random.default_rng(20261013)
    null=np.empty(1999)
    for k in range(1999):
        lab=labels.copy()
        for ii in idx_bins: lab[ii]=rng.permutation(lab[ii])
        null[k]=score(lab)
    p=(1+np.count_nonzero(null>=observed))/2000
    b_mse=float(np.mean(np.bincount(labels,weights=(y-baseline)**2,minlength=nc)/counts))
    c_mse=float(np.mean(np.bincount(labels,weights=(y-pred)**2,minlength=nc)/counts))
    result['test']={'balanced_mse_baseline':b_mse,'balanced_mse_cluster':c_mse,'mse_gain':observed,'fraction_mse_improvement':observed/b_mse,'balanced_rmse_baseline_dex':float(np.sqrt(b_mse)),'balanced_rmse_cluster_dex':float(np.sqrt(c_mse)),'p_permutation':float(p),'null_median':float(np.median(null)),'null_95':float(np.quantile(null,.95)),'null_99':float(np.quantile(null,.99)),'n_permutations':1999,'bins':len(idx_bins),'mixed_cluster_bins':len(mix_bins),'stars_in_mixed_bins':mobile,'fraction_stars_permutable_across_clusters':mobile/n,'clusters_with_positive_gain':int((by_cluster>0).sum()),'interpretation':'membership-probability-selected retained APOGEE giants; approximate conditional inference, not independent new observations'}
    result['decision']='SURVIVES_SELECTED_CONDITIONAL_CHECK' if observed>0 and p<.01 else 'DOES_NOT_SURVIVE_SELECTED_CONDITIONAL_CHECK'
    d['fold']=folds; d['log_CO']=y; d['prediction_nuisance']=baseline; d['prediction_cluster']=pred; d['bin']=bins
    d.to_csv(OUT/'qualified_apogee_heldout_predictions.csv',index=False)
    pd.DataFrame({'permutation':np.arange(1999),'mse_gain':null}).to_csv(OUT/'permutation_null.csv',index=False)
    pd.DataFrame({'cluster':names,'n':counts,'mse_gain':by_cluster}).to_csv(OUT/'cluster_gains.csv',index=False)
    # Quantify positional overlap, not formal calibrated instrument agreement.
    g=pd.read_csv(REPO/'t9_matched_stars.csv').drop_duplicates(['ra','dec'])
    def xyz(ra,dec):
        a=np.deg2rad(ra); b=np.deg2rad(dec)
        return np.column_stack([np.cos(b)*np.cos(a),np.cos(b)*np.sin(a),np.sin(b)])
    tree=cKDTree(xyz(g.ra,g.dec))
    dd,ii=tree.query(xyz(d.RA,d.DEC))
    sec=np.rad2deg(2*np.arcsin(np.minimum(dd/2,1)))*3600
    result['cross_survey_overlap']={'qualified_apogee_stars_with_galah_aperture_position_within_1arcsec':int((sec<1).sum()),'caveat':'positions at their retained catalog epochs; not a full-ID matched or zero-point-calibrated replication'}

(OUT/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

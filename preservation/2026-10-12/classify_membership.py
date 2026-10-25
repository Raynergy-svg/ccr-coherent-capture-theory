"""Evaluate only the frozen identity/astrometry crossmatch protocol."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,re
import pandas as pd

R=Path(__file__).resolve().parent
d=pd.read_csv(R/'candidate_identity_only.csv',dtype=str)
assert set(d.columns)=={'APOGEE_ID','CLUSTER','RA','DEC'}
b=pd.read_csv(R/'gaia_bridge_all.csv',dtype=str)
m=pd.read_csv(R/'hr_members_all.csv',dtype=str)
c=pd.read_csv(R/'hr_cluster_identity_quality.csv',dtype=str)
ast=pd.concat([pd.read_csv(p,dtype=str) for p in sorted(R.glob('gaia_astrometry_[0-9].csv'))],ignore_index=True)
def clean(x):return str(x).strip().strip("'")
for frame in [b,m,c,ast]:
 for col in frame.columns:frame[col]=frame[col].map(clean)
def norm(x):
 return '_'.join(str(int(t)) if t.isdigit() else t.upper() for t in re.findall('[A-Za-z]+|[0-9]+',str(x)))
aliases=defaultdict(set)
for row in c.itertuples(index=False):
 for alias in [row.Name]+str(row.AllNames).split(','):
  aliases[norm(alias)].add(row.ID)
lookup=c.set_index('ID').to_dict(orient='index')
reverse=b.groupby('source_id').original_ext_source_id.nunique()
rows=[]
for candidate in d.itertuples(index=False):
 oid=candidate.APOGEE_ID[2:]
 bridge=b[b.original_ext_source_id==oid]
 expected=sorted(aliases.get(norm(candidate.CLUSTER),set()))
 out={'APOGEE_ID':candidate.APOGEE_ID,'original_cluster':candidate.CLUSTER,'expected_hr_ids':'|'.join(expected),'bridge_rows':len(bridge),'bridge_source_ids':'|'.join(sorted(bridge.source_id.unique()))}
 if len(bridge)!=1 or bridge.source_id.nunique()!=1:
  out.update(status='UNRESOLVED_IDENTITY_OR_ALIAS',reason='missing or multiple Gaia identity bridge');rows.append(out);continue
 br=bridge.iloc[0];gid=br.source_id
 out.update(gaia_dr3_id=gid,bridge_angular_distance_arcsec=br.angular_distance,xm_flag=br.xm_flag,number_of_neighbours=br.number_of_neighbours,number_of_mates=br.number_of_mates)
 flags=int(float(br.xm_flag));out['crossmatch_special_treatment_warning']=bool(flags & (1|2|4|16|32|64))
 ast_row=ast[ast.source_id==gid]
 if len(ast_row)==1:
  for k in ['ref_epoch','astrometric_params_solved','ruwe','parallax','parallax_error','pmra','pmdec']:out['gaia_'+k]=ast_row.iloc[0][k]
 if float(br.number_of_neighbours)!=1 or float(br.number_of_mates)!=0 or reverse.loc[gid]!=1:
  out.update(status='UNRESOLVED_IDENTITY_OR_ALIAS',reason='ambiguous neighbor/mate/reverse-identity mapping');rows.append(out);continue
 mm=m[m.GaiaDR3==gid].drop_duplicates()
 out['hr_member_records']=len(mm)
 out['returned_hr_ids']='|'.join(mm.ID.unique())
 out['returned_hr_names']='|'.join(mm.Name.unique())
 if len(expected)!=1:
  out.update(status='UNRESOLVED_IDENTITY_OR_ALIAS',reason='expected cluster alias missing or ambiguous');rows.append(out);continue
 if len(mm)==0:
  out.update(status='NOT_LISTED',reason='sound ID bridge; absent from this incomplete membership catalog');rows.append(out);continue
 if len(mm)!=1:
  out.update(status='UNRESOLVED_IDENTITY_OR_ALIAS',reason='multiple membership records retained');rows.append(out);continue
 mem=mm.iloc[0];qc=lookup[mem.ID]
 out.update(hr_id=mem.ID,hr_name=mem.Name,hr_score=float(mem.Prob),in_jacobi_radius=int(float(mem.inrj)),in_profile_radius=int(float(mem.inrt)),hr_cluster_type=qc['Type'],hr_cluster_CST=float(qc['CST']),hr_cluster_CMD_score=float(qc['CMDCl50']),hr_cluster_merged=qc['isMerged'],hr_cluster_GMM_membership=qc['isGMMMemb'])
 out['hr_high_quality_bound_cluster']=qc['Type']=='o' and float(qc['CST'])>5 and float(qc['CMDCl50'])>.5
 if mem.ID!=expected[0]:out.update(status='DIFFERENT_CATALOG_CLUSTER',reason='Gaia ID listed under another catalog cluster')
 else:out.update(status='SAME_CLUSTER_HIGH_SCORE' if float(mem.Prob)>=.5 else 'SAME_CLUSTER_LOW_SCORE',reason='exact Gaia ID and published cluster alias agree')
 rows.append(out)
out=pd.DataFrame(rows)
out.to_csv(R/'membership_assignments.csv',index=False)
by=pd.crosstab(out.original_cluster,out.status);by.to_csv(R/'membership_by_cluster.csv')
support=out.status.eq('SAME_CLUSTER_HIGH_SCORE')
summary={'protocol_sha256':hashlib.sha256((R/'MEMBERSHIP_PROTOCOL.md').read_bytes()).hexdigest(),'input_sha256':hashlib.sha256((R/'candidate_identity_only.csv').read_bytes()).hexdigest(),'input_rows':len(d),'input_clusters':d.CLUSTER.nunique(),'bridge_rows':len(b),'bridge_unique_input_ids':b.original_ext_source_id.nunique(),'bridge_unique_gaia_ids':b.source_id.nunique(),'retrieved_member_rows':len(m),'catalog_cluster_identity_rows':len(c),'counts':out.status.value_counts().to_dict(),'same_cluster_high_score_and_highquality_bound':int((support&out.hr_high_quality_bound_cluster.eq(True)).sum()),'same_cluster_high_score_inside_jacobi_radius':int((support&out.in_jacobi_radius.eq(1)).sum()),'same_cluster_high_score_and_highquality_bound_inside_jacobi':int((support&out.hr_high_quality_bound_cluster.eq(True)&out.in_jacobi_radius.eq(1)).sum()),'supported_with_special_crossmatch_warning':int((support&out.crossmatch_special_treatment_warning.eq(True)).sum()),'same_cluster_highscore_clusters':out.loc[support,'original_cluster'].nunique(),'clusters_with_ge5_same_cluster_highscore':int((out.loc[support].groupby('original_cluster').size()>=5).sum()),'limitations':['No chemistry evaluated; existing chemically inspected sample, not independent chemical validation.','Alternative Gaia astrometric method shares measurement ancestry with OCCAM.','HDBSCAN membership scores are not calibrated membership posteriors.','Absence is not non-membership; distant/low-density/tail members may be missed.','Cluster alias ambiguity remains unresolved; no spatial or fuzzy-name substitutions.','Identity-bridge special-treatment flags and astrometric quality retained, not hidden.']}
(R/'membership_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(by.to_string())

"""Bounded identity/astrometry-only public retrieval; no chemistry is read."""
import csv, hashlib, io, json, re, time, urllib.request, urllib.parse
from pathlib import Path
import pandas as pd

R=Path(__file__).resolve().parent
G='https://gea.esac.esa.int/tap-server/tap/sync'
V='https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync'
TOTAL=0
receipts=[]
def query(name,base,adql,maxrec=2000):
    global TOTAL
    (R/(name+'.adql')).write_text(adql)
    args={'REQUEST':'doQuery','LANG':'ADQL','FORMAT':'csv','MAXREC':str(maxrec),'QUERY':adql}
    # POST keeps public object lists out of very long request URLs.
    data=urllib.parse.urlencode(args).encode()
    for attempt in range(2):
        try:
            req=urllib.request.Request(base,data=data)
            with urllib.request.urlopen(req,timeout=30) as response:
                b=response.read(3000001)
            if len(b)>3000000: raise ValueError('Individual response budget exceeded')
            TOTAL+=len(b)
            if TOTAL>5000000: raise ValueError('Total response budget exceeded')
            if b.lstrip().startswith(b'<?xml') or b'<VOTABLE' in b[:200]: raise ValueError('Service returned XML/error rather than CSV')
            p=R/(name+'.csv');p.write_bytes(b)
            d=pd.read_csv(io.BytesIO(b),dtype=str)
            if len(d)>=maxrec: raise ValueError('Row cap reached; possible truncation')
            receipts.append({'name':name,'endpoint':base,'query_sha256':hashlib.sha256(adql.encode()).hexdigest(),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'rows':len(d)})
            (R/'retrieval_receipts.json').write_text(json.dumps(receipts,indent=2))
            print(name,len(d),'rows',len(b),'bytes',flush=True)
            return d
        except Exception as e:
            print(name,'attempt',attempt+1,type(e).__name__,str(e),flush=True)
            if attempt or isinstance(e,ValueError): raise
            time.sleep(2)

ident=pd.read_csv(R/'candidate_identity_only.csv',dtype={'APOGEE_ID':str})
assert set(ident.columns)=={'APOGEE_ID','CLUSTER','RA','DEC'}
assert ident.APOGEE_ID.is_unique
assert ident.APOGEE_ID.str.fullmatch(r'2M\d{8}[+-]\d{7}').all()
ids=ident.APOGEE_ID.str[2:].tolist()
frames=[]
for i in range(0,len(ids),100):
    quoted=','.join("'"+x+"'" for x in ids[i:i+100])
    q='SELECT TOP 500 source_id,original_ext_source_id,angular_distance,xm_flag,number_of_neighbours,number_of_mates FROM gaiadr3.tmass_psc_xsc_best_neighbour WHERE original_ext_source_id IN ('+quoted+')'
    frames.append(query('gaia_bridge_'+str(i//100),G,q,500))
bridge=pd.concat(frames,ignore_index=True);bridge.to_csv(R/'gaia_bridge_all.csv',index=False)
gaia_ids=sorted(bridge.source_id.dropna().unique().tolist())
assert all(re.fullmatch(r'\d+',x) for x in gaia_ids)
frames=[]
for i in range(0,len(gaia_ids),100):
    values=','.join(gaia_ids[i:i+100])
    q='SELECT TOP 1000 Name,ID,GaiaDR3,Prob,inrj,inrt,RA_ICRS,DE_ICRS,pmRA,pmDE,Plx,e_Plx,RUWE,FidelityV1,Solved FROM "J/A+A/686/A42/members" WHERE GaiaDR3 IN ('+values+')'
    frames.append(query('hr_members_'+str(i//100),V,q,1000))
members=pd.concat(frames,ignore_index=True);members.to_csv(R/'hr_members_all.csv',index=False)
q='SELECT TOP 7200 Name,ID,AllNames,Type,CST,CMDCl50,isMerged,isGMMMemb FROM "J/A+A/686/A42/clusters"'
query('hr_cluster_identity_quality',V,q,7200)
for i in range(0,len(gaia_ids),200):
    values=','.join(gaia_ids[i:i+200])
    q='SELECT TOP 1000 source_id,ref_epoch,ra,dec,ra_error,dec_error,pmra,pmdec,pmra_error,pmdec_error,parallax,parallax_error,astrometric_params_solved,ruwe FROM gaiadr3.gaia_source WHERE source_id IN ('+values+')'
    query('gaia_astrometry_'+str(i//200),G,q,1000)
print('TOTAL_BYTES',TOTAL,flush=True)

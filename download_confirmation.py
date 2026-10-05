"""Download a label-independent, previously unused patient cohort separately."""
import argparse,ast,concurrent.futures,hashlib,json,re,time
from pathlib import Path
import numpy as np,pandas as pd
from download_data import BASE,LEADS,fetch
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'confirmation_results'
def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT.parent/'confirmation_cache');p.add_argument('--source-cache',type=Path,default=ROOT.parent/'metadata_cache');a=p.parse_args()
    protocol=json.loads((ROOT/'confirmation_protocol.json').read_text());OUT.mkdir(exist_ok=True);a.cache.mkdir(exist_ok=True)
    a.source_cache.mkdir(exist_ok=True)
    for name in ['ptbxl_database.csv','scp_statements.csv']:fetch(BASE+name,a.source_cache/name)
    df=pd.read_csv(a.source_cache/'ptbxl_database.csv');sc=pd.read_csv(a.source_cache/'scp_statements.csv',index_col=0);old=pd.read_csv(ROOT/'results/manifest.csv')
    excluded=set(old.patient_id);excluded.update(df.loc[df.ecg_id.isin([1,101,1013]),'patient_id'])
    unique=df.sort_values('ecg_id').drop_duplicates('patient_id');pool=unique[unique.strat_fold.isin(protocol['cohort']['folds'])&~unique.patient_id.isin(excluded)]
    rng=np.random.default_rng(protocol['cohort']['selection_seed']);manifest=pool.iloc[rng.choice(len(pool),protocol['cohort']['n'],replace=False)].copy();manifest['split']='confirmation'
    mapping=sc.loc[sc.diagnostic==1,'diagnostic_class'].to_dict()
    for task in protocol['audit']['tasks']:manifest[task]=manifest.scp_codes.map(lambda s:int(task in {mapping.get(c) for c in ast.literal_eval(s)}))
    assert manifest.patient_id.is_unique and not set(manifest.patient_id)&excluded
    manifest.to_csv(OUT/'manifest.csv',index=False)
    sources={'eligible_patient_count':len(pool),'excluded_patient_count':len(excluded),'overlap_with_previous_manifest':0,'protocol_sha256':hashlib.sha256((ROOT/'confirmation_protocol.json').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((OUT/'manifest.csv').read_bytes()).hexdigest(),'metadata_sha256':hashlib.sha256((a.source_cache/'ptbxl_database.csv').read_bytes()).hexdigest(),'base':BASE}
    (OUT/'data_sources.json').write_text(json.dumps(sources,indent=2));print(json.dumps(sources),flush=True)
    out=np.lib.format.open_memmap(a.cache/'signals.npy',mode='w+',dtype='float32',shape=(len(manifest),12,1000));begin=time.monotonic()
    def record(item):
        i,row=item;name=row.filename_lr;stem=name.rsplit('/',1)[-1];header=fetch(BASE+name+'.hea',a.cache/(stem+'.hea')).decode().splitlines();data=fetch(BASE+name+'.dat',a.cache/(stem+'.dat'))
        assert header[0].split()[1:]==['12','100','1000'] and len(data)==24000
        values=np.frombuffer(data,dtype='<i2').reshape(1000,12).T.astype('float32')
        for j,line in enumerate(header[1:13]):
            fields=line.split();assert fields[1]=='16' and fields[-1]==LEADS[j];m=re.fullmatch(r'([0-9.]+)\((-?[0-9]+)\)/mV',fields[2]);assert m;values[j]=(values[j]-int(m[2]))/float(m[1])
        return i,values
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for done,future in enumerate(concurrent.futures.as_completed([pool.submit(record,(i,row)) for i,(_,row) in enumerate(manifest.iterrows())]),1):
            i,values=future.result();out[i]=values
            if done%100==0:print(f'downloaded {done}/{len(manifest)} in {time.monotonic()-begin:.0f}s',flush=True)
    out.flush();assert np.isfinite(out).all();print('Complete: no exclusions or replacements.',flush=True)
if __name__=='__main__':main()

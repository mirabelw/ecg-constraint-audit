"""Download a fixed, label-independent patient sample. No credentials required."""
import argparse, ast, concurrent.futures, hashlib, json, re, time, threading
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
BASE = 'https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/'
LEADS = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
LOCAL=threading.local()

def fetch(url, path):
    if path.exists():
        return path.read_bytes()
    for attempt in range(4):
        try:
            if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session()
            response=LOCAL.session.get(url,timeout=40);response.raise_for_status();value=response.content
            tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_bytes(value); tmp.replace(path)
            return value
        except Exception:
            if attempt==3: raise
            time.sleep(1+attempt)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,default=ROOT.parent/'data_cache');p.add_argument('--workers',type=int,default=24);a=p.parse_args()
    a.cache.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    metadata=fetch(BASE+'ptbxl_database.csv',a.cache/'ptbxl_database.csv')
    statements=fetch(BASE+'scp_statements.csv',a.cache/'scp_statements.csv')
    df=pd.read_csv(a.cache/'ptbxl_database.csv');sc=pd.read_csv(a.cache/'scp_statements.csv',index_col=0)
    # Patients cannot cross the supplied folds; detect an incompatible dataset.
    assert df.groupby('patient_id').strat_fold.nunique().max()==1
    unique=df.sort_values('ecg_id').drop_duplicates('patient_id').copy()
    rng=np.random.default_rng(protocol['selection_seed']); parts=[]
    for split, folds,n in [('train',range(1,9),protocol['n_train']),('validation',[9],protocol['n_validation']),('test',[10],protocol['n_test'])]:
        pool=unique[unique.strat_fold.isin(folds)];chosen=pool.iloc[rng.choice(len(pool),n,replace=False)].copy();chosen['split']=split;parts.append(chosen)
    manifest=pd.concat(parts,ignore_index=True)
    mapping=sc.loc[sc.diagnostic==1,'diagnostic_class'].to_dict()
    for task in protocol['tasks']:
        manifest[task]=manifest.scp_codes.map(lambda s:int(task in {mapping.get(c) for c in ast.literal_eval(s)}))
    manifest.to_csv(ROOT/'results'/'manifest.csv',index=False)
    (ROOT/'results'/'data_sources.json').write_text(json.dumps({'base':BASE,'metadata_sha256':hashlib.sha256(metadata).hexdigest(),'statements_sha256':hashlib.sha256(statements).hexdigest(),'manifest_sha256':hashlib.sha256((ROOT/'results'/'manifest.csv').read_bytes()).hexdigest()},indent=2))
    out=np.lib.format.open_memmap(a.cache/'signals.npy',mode='w+',dtype='float32',shape=(len(manifest),12,1000))
    begin=time.monotonic();done=0
    def record(item):
        i,row=item; name=row.filename_lr; stem=name.rsplit('/',1)[-1]
        header=fetch(BASE+name+'.hea',a.cache/(stem+'.hea')).decode().splitlines()
        data=fetch(BASE+name+'.dat',a.cache/(stem+'.dat'))
        assert header[0].split()[1:]==['12','100','1000'], header[0]
        assert len(data)==24000
        values=np.frombuffer(data,dtype='<i2').reshape(1000,12).T.astype('float32')
        for j,line in enumerate(header[1:13]):
            fields=line.split();assert fields[1]=='16' and fields[-1]==LEADS[j],line
            m=re.fullmatch(r'([0-9.]+)\((-?[0-9]+)\)/mV',fields[2]);assert m,line
            values[j]=(values[j]-int(m[2]))/float(m[1])
        return i,values
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(record,item) for item in manifest.iterrows()]):
            i,values=future.result();out[i]=values;done+=1
            if done%100==0: print(f'downloaded {done}/{len(manifest)} in {time.monotonic()-begin:.0f}s',flush=True)
    out.flush();print('COMPLETE: all prespecified records downloaded; no exclusions.',flush=True)

if __name__=='__main__':main()

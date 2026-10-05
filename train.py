import argparse,copy,json,os,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from torch.utils.data import DataLoader,TensorDataset
from geometry import project,validate
from model import ECGNet
ROOT=Path(__file__).resolve().parent

def load_data(cache):
    meta=pd.read_csv(ROOT/'results'/'manifest.csv');raw=np.load(cache/'signals.npy',mmap_mode='r')
    assert len(raw)==len(meta) and raw.shape[1:]==(12,1000)
    assert np.isfinite(raw).all()
    residual=np.asarray(raw[:,:6])-project(np.asarray(raw))[:,:6]
    geometry={'raw_projection_rms_mv':float(np.sqrt(np.mean(residual**2))),'raw_projection_max_mv':float(np.abs(residual).max()),'one_record_per_patient':bool(meta.patient_id.is_unique),'n_by_split':meta.groupby('split').size().to_dict(),'label_counts':meta.groupby('split')[['MI','CD']].sum().to_dict()}
    (ROOT/'results'/'data_quality.json').write_text(json.dumps(geometry,indent=2))
    centered=np.asarray(raw).copy();centered-=centered.mean(axis=-1,keepdims=True);x=project(centered).astype('float32')
    train=meta.split.eq('train').to_numpy();scale=float(np.sqrt(np.mean(x[train]**2)))
    (ROOT/'results'/'preprocessing.json').write_text(json.dumps({'global_train_rms_mv':scale,'center_each_lead':True,'project_limb':True},indent=2))
    x/=scale;return meta,x,meta[['MI','CD']].to_numpy(dtype='float32'),scale

def predict(model,x,batch=128):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):out.append(model(torch.from_numpy(x[start:start+batch])).sigmoid().numpy())
    return np.concatenate(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT.parent/'data_cache');p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(a.threads);torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'protocol.json').read_text());assert all(v=='pass' for v in validate().values())
    (ROOT/'results'/'geometry_checks.json').write_text(json.dumps(validate(),indent=2))
    meta,x,y,scale=load_data(a.cache);tr=meta.split.eq('train').to_numpy();va=meta.split.eq('validation').to_numpy();history=[];summaries=[]
    for seed in protocol['model_seeds']:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);model=ECGNet()
        loader=DataLoader(TensorDataset(torch.from_numpy(x[tr]),torch.from_numpy(y[tr])),batch_size=64,shuffle=True,num_workers=0)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        # Label prevalence adjustment uses training data only.
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy((1-y[tr].mean(0))/y[tr].mean(0)))
        best=-1;best_state=None;best_epoch=None;start=time.monotonic()
        for epoch in range(protocol['epochs']):
            model.train();loss_total=0
            for xb,yb in loader:
                optimizer.zero_grad();loss=criterion(model(xb),yb);loss.backward();optimizer.step();loss_total+=float(loss.detach())*len(xb)
            pv=predict(model,x[va]);auc=roc_auc_score(y[va],pv,average=None)
            row={'seed':seed,'epoch':epoch+1,'loss':loss_total/tr.sum(),'MI_val_auroc':float(auc[0]),'CD_val_auroc':float(auc[1])};history.append(row)
            if auc.mean()>best:best=float(auc.mean());best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            print(f'seed={seed} epoch={epoch+1} loss={row["loss"]:.3f} val={auc.round(3).tolist()} elapsed={time.monotonic()-start:.0f}s',flush=True)
        model.load_state_dict(best_state);pv=predict(model,x[va]);auc=roc_auc_score(y[va],pv,average=None);ap=average_precision_score(y[va],pv,average=None)
        torch.save({'state_dict':best_state,'seed':seed,'scale':scale,'best_epoch':best_epoch},ROOT/'checkpoints'/f'seed_{seed}.pt')
        summaries.append({'seed':seed,'best_epoch':best_epoch,'MI_val_auroc':float(auc[0]),'CD_val_auroc':float(auc[1]),'MI_val_auprc':float(ap[0]),'CD_val_auprc':float(ap[1]),'gate_pass':bool(np.all(auc>=protocol['validation_auroc_gate']))})
        pd.DataFrame(history).to_csv(ROOT/'results'/'training_history.csv',index=False);pd.DataFrame(summaries).to_csv(ROOT/'results'/'validation_metrics.csv',index=False)
    (ROOT/'results'/'environment.json').write_text(json.dumps({'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'threads':a.threads,'device':'cpu','protocol_sha256':__import__('hashlib').sha256((ROOT/'protocol.json').read_bytes()).hexdigest()},indent=2))
    assert all(s['gate_pass'] for s in summaries),'Validation gate failed; do not run test explanations.'

if __name__=='__main__':main()

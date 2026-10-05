import argparse,copy,hashlib,json,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score
from torch.utils.data import DataLoader,TensorDataset
from model import ResidualECGNet
from train import load_data,predict
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'extension_results'

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT.parent/'data_cache');a=p.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);OUT.mkdir(exist_ok=True)
    meta,x,y,scale=load_data(a.cache);tr=meta.split.eq('train').to_numpy();va=meta.split.eq('validation').to_numpy();history=[];summaries=[];config=json.loads((ROOT/'extension_protocol.json').read_text())
    for seed in config['seeds']:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);model=ResidualECGNet();loader=DataLoader(TensorDataset(torch.from_numpy(x[tr]),torch.from_numpy(y[tr])),batch_size=64,shuffle=True,num_workers=0)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy((1-y[tr].mean(0))/y[tr].mean(0)))
        best=-1;start=time.monotonic()
        for epoch in range(20):
            model.train();loss_total=0
            for xb,yb in loader:
                optimizer.zero_grad();loss=criterion(model(xb),yb);loss.backward();optimizer.step();loss_total+=float(loss.detach())*len(xb)
            prediction=predict(model,x[va]);aucs=roc_auc_score(y[va],prediction,average=None);row={'seed':seed,'epoch':epoch+1,'loss':loss_total/tr.sum(),'MI_val_auroc':float(aucs[0]),'CD_val_auroc':float(aucs[1])};history.append(row)
            if aucs.mean()>best:best=float(aucs.mean());state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            print(f'residual seed={seed} epoch={epoch+1} val={aucs.round(3)} elapsed={time.monotonic()-start:.0f}s',flush=True)
        model.load_state_dict(state);prediction=predict(model,x[va]);aucs=roc_auc_score(y[va],prediction,average=None);ap=average_precision_score(y[va],prediction,average=None)
        torch.save({'state_dict':state,'seed':seed,'scale':scale,'best_epoch':best_epoch,'architecture':'residual_cnn'},ROOT/'checkpoints'/f'residual_seed_{seed}.pt')
        summaries.append({'seed':seed,'best_epoch':best_epoch,'MI_val_auroc':float(aucs[0]),'CD_val_auroc':float(aucs[1]),'MI_val_auprc':float(ap[0]),'CD_val_auprc':float(ap[1]),'gate_pass':bool(np.all(aucs>=.75))})
        pd.DataFrame(history).to_csv(OUT/'residual_training_history.csv',index=False);pd.DataFrame(summaries).to_csv(OUT/'residual_validation.csv',index=False)
    assert all(r['gate_pass'] for r in summaries)
    (OUT/'protocol_sha256.txt').write_text(hashlib.sha256((ROOT/'extension_protocol.json').read_bytes()).hexdigest()+'\n')

if __name__=='__main__':main()

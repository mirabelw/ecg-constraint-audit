import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score
from model import ECGNet,ResidualECGNet
from train import load_data
from evaluate import make_mask
from extension_methods import maps,METHODS,amplitude_random_mask,dose_operators
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'extension_results'
KINDS=['guided','uniform_random','amplitude_matched_random'];OPS=['independent','energy_matched','projected_unscaled']

def logits(model,x):
    with torch.no_grad():return model(torch.from_numpy(x.astype('float32'))).numpy()
def sigmoid(v):return 1/(1+np.exp(-np.clip(v,-60,60)))

def evaluate_model(model,xt,ids,seed,architecture,batch=64):
    n=len(xt);shape=(2,7,3,3,n);response=np.empty(shape,dtype='float32');logit_response=np.empty(shape,dtype='float32');normal=np.empty((2,7,3,n),dtype='float32');off_response=np.empty((2,7,3,n),dtype='float32');amp_ratio=np.empty((2,7,n),dtype='float32');overlap=np.empty_like(amp_ratio)
    secondary=np.empty((2,4,2,3,n),dtype='float32');saved=np.empty((2,7,n,12,20),dtype='float32');original=logits(model,xt);random_scores=np.stack([np.random.default_rng(1000000+int(e)).random((12,20)) for e in ids]);clock=time.monotonic()
    for task in range(2):
        for start in range(0,n,batch):
            end=min(n,start+batch);x=xt[start:end];ranks=maps(model,x,task,ids[start:end]);saved[task,:,start:end]=ranks.transpose(1,0,2,3);original_logit=original[start:end,task];original_prob=sigmoid(original_logit);random_mask=make_mask(random_scores[start:end],'limb',.1,50)
            def compute(delta_full,mask,dose=None):
                ds=dose_operators(delta_full,mask,dose);pred=[logits(model,x+d)[:,task] for d in ds[:3]];absolute=[np.abs(sigmoid(v)-original_prob) for v in pred];absolute_logit=[np.abs(v-original_logit) for v in pred]
                return np.stack(absolute),np.stack(absolute_logit),ds[3],np.abs(pred[0]-pred[2])
            uniform=compute(-x,random_mask)
            for m in range(7):
                mask=make_mask(ranks[:,m],'limb',.1,50);matched,rat,ov=amplitude_random_mask(-x,mask,ids[start:end],m);dose=np.sqrt(((x*mask)**2).sum((1,2)));amp_ratio[task,m,start:end]=rat;overlap[task,m,start:end]=ov
                for kind,result in enumerate([compute(-x,mask),uniform,compute(-x,matched,dose)]):
                    response[task,m,kind,:,start:end]=result[0];logit_response[task,m,kind,:,start:end]=result[1];normal[task,m,kind,start:end]=result[2];off_response[task,m,kind,start:end]=result[3]
            full_mean=np.repeat(x.reshape(len(x),12,20,50).mean(-1),50,-1)-x;ur=compute(full_mean,random_mask)
            for sm,m in enumerate([0,1,2,6]):
                mask=make_mask(ranks[:,m],'limb',.1,50);secondary[task,sm,0,:,start:end]=compute(full_mean,mask)[0];secondary[task,sm,1,:,start:end]=ur[0]
            print(f'{architecture} seed={seed} task={task} n={end}/{n} elapsed={time.monotonic()-clock:.0f}s',flush=True)
    np.savez_compressed(OUT/f'{architecture}_seed_{seed}.npz',response=response,logit_response=logit_response,normal_energy_fraction=normal,off_direction_logit_response=off_response,matched_pre_scale_dose_ratio=amp_ratio,matched_mask_overlap=overlap,mean_replacement_response=secondary,window_scores=saved,ecg_id=ids,original_logits=original)
    return original

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT.parent/'data_cache');p.add_argument('--architecture',choices=['original_cnn','residual_cnn','both'],default='both');a=p.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);OUT.mkdir(exist_ok=True)
    assert pd.read_csv(ROOT/'results/validation_metrics.csv').gate_pass.all()
    if a.architecture!='original_cnn':assert pd.read_csv(OUT/'residual_validation.csv').gate_pass.all()
    meta,x,y,scale=load_data(a.cache);test=meta.split.eq('test').to_numpy();xt=x[test];yt=y[test];ids=meta.loc[test,'ecg_id'].to_numpy();performance=[]
    for architecture in (['original_cnn','residual_cnn'] if a.architecture=='both' else [a.architecture]):
        for seed in [17,29,43]:
            path=ROOT/'checkpoints'/(f'seed_{seed}.pt' if architecture=='original_cnn' else f'residual_seed_{seed}.pt');model=ECGNet() if architecture=='original_cnn' else ResidualECGNet();model.load_state_dict(torch.load(path,weights_only=True)['state_dict']);model.eval()
            predictions=evaluate_model(model,xt,ids,seed,architecture);auc=roc_auc_score(yt,sigmoid(predictions),average=None);ap=average_precision_score(yt,sigmoid(predictions),average=None)
            for t,task in enumerate(['MI','CD']):performance.append({'architecture':architecture,'seed':seed,'task':task,'test_auroc':float(auc[t]),'test_auprc':float(ap[t])})
            pd.DataFrame(performance).to_csv(OUT/'extension_test_metrics.csv',index=False)
    (OUT/'axes.json').write_text(json.dumps({'methods':METHODS,'kinds':KINDS,'operators':OPS,'tasks':['MI','CD'],'seeds':[17,29,43],'primary_dimensions':['task','method','kind','operator','patient'],'secondary_mean_methods':[METHODS[i] for i in [0,1,2,6]],'secondary_mean_kinds':['guided','uniform_random']},indent=2))

if __name__=='__main__':main()

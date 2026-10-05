"""Frozen-mask, paired perturbation audit. No attribution superiority claims."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from geometry import interventions,diagnostics,project
from model import ECGNet
from train import load_data,predict
ROOT=Path(__file__).resolve().parent
METHODS=['saliency','input_gradient','integrated_gradients','random']
STRATA=['limb','chest','all'];OPERATORS=['independent_zero','target_matched','energy_matched']

def explanations(model, x, task, steps):
    t=torch.from_numpy(x).requires_grad_(True)
    logit=model(t)[:,task];gradient=torch.autograd.grad(logit.sum(),t)[0]
    maps={'saliency':gradient.detach().abs().numpy(),'input_gradient':(gradient*t).detach().abs().numpy()}
    integral=torch.zeros_like(t)
    for i in range(steps):
        u=(t.detach()*((i+.5)/steps)).requires_grad_(True)
        integral+=torch.autograd.grad(model(u)[:,task].sum(),u)[0].detach()/steps
    signed_ig=(integral*t.detach()).numpy();maps['integrated_gradients']=np.abs(signed_ig)
    with torch.no_grad():gap=(logit.detach()-model(torch.zeros_like(t))[:,task]).numpy()
    error=np.abs(signed_ig.sum(axis=(1,2))-gap)
    return maps,error,gap

def make_mask(scores,stratum,budget,window):
    n,lead,windows=scores.shape;mask=np.zeros_like(scores,dtype=np.float32)
    eligible=np.arange(6) if stratum=='limb' else np.arange(6,12) if stratum=='chest' else np.arange(12)
    subset=scores[:,eligible].reshape(n,-1);count=int(round(budget*subset.shape[1]));assert count>=1
    selected=np.argsort(-subset,axis=1,kind='stable')[:,:count]
    for b in range(n):mask[b,eligible[selected[b]//windows],selected[b]%windows]=1
    return np.repeat(mask,window,axis=-1)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,default=ROOT.parent/'data_cache');parser.add_argument('--threads',type=int,default=4);parser.add_argument('--batch',type=int,default=64);a=parser.parse_args()
    torch.set_num_threads(a.threads);torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'protocol.json').read_text());metrics=pd.read_csv(ROOT/'results'/'validation_metrics.csv');assert len(metrics)==3 and metrics.gate_pass.all(),'Validation gate failed.'
    meta,x,y,scale=load_data(a.cache);test=meta.split.eq('test').to_numpy();xt=x[test];yt=y[test];testmeta=meta.loc[test].reset_index(drop=True);n=len(xt);window=protocol['window_samples'];budgets=protocol['budgets']
    shape=(3,2,4,3,len(budgets),3,n);responses=np.empty(shape,dtype=np.float32)
    diagnostic_arrays={k:np.empty(shape,dtype=np.float32) for k in ['dose_ratio','collateral_fraction','selected_relative_error','off_subspace_rms_mv']}
    completeness=[];performance=[];clock=time.monotonic()
    # Random comparator is fixed per patient, shared across tasks and seeds.
    random_scores=np.stack([np.random.default_rng(1000000+int(e)).random((12,1000//window)) for e in testmeta.ecg_id])
    saved_maps=np.zeros((3,2,3,n,12,1000//window),dtype=np.float32)
    for si,seed in enumerate(protocol['model_seeds']):
        state=torch.load(ROOT/'checkpoints'/f'seed_{seed}.pt',map_location='cpu',weights_only=True);model=ECGNet();model.load_state_dict(state['state_dict']);model.eval()
        original=predict(model,xt);auc=roc_auc_score(yt,original,average=None);ap=average_precision_score(yt,original,average=None)
        for t,task in enumerate(protocol['tasks']):performance.append({'seed':seed,'task':task,'test_auroc':float(auc[t]),'test_auprc':float(ap[t]),'test_positive_n':int(yt[:,t].sum()),'test_n':n})
        for task in range(2):
            errors=[];gaps=[]
            for start in range(0,n,a.batch):
                end=min(n,start+a.batch);xb=xt[start:end];scores,error,gap=explanations(model,xb,task,protocol['ig_midpoint_steps']);errors.extend(error);gaps.extend(gap)
                for mi,method in enumerate(METHODS):
                    if method=='random':score=random_scores[start:end]
                    else:
                        score=scores[method].reshape(len(xb),12,-1,window).mean(-1);saved_maps[si,task,mi,start:end]=score
                    for sti,stratum in enumerate(STRATA):
                        for bi,budget in enumerate(budgets):
                            mask=make_mask(score,stratum,budget,window);deltas=interventions(xb,mask)
                            for oi,operator in enumerate(OPERATORS):
                                delta=deltas[operator].astype('float32')
                                if stratum=='chest' and oi>0:
                                    responses[si,task,mi,sti,bi,oi,start:end]=responses[si,task,mi,sti,bi,0,start:end]
                                else:
                                    perturbed=predict(model,xb+delta)[:,task]
                                    responses[si,task,mi,sti,bi,oi,start:end]=np.abs(perturbed-original[start:end,task])
                                di=diagnostics(xb*scale,mask,delta*scale)
                                for k,value in di.items():diagnostic_arrays[k][si,task,mi,sti,bi,oi,start:end]=value
                print(f'evaluate seed={seed} task={protocol["tasks"][task]} patients={end}/{n} elapsed={time.monotonic()-clock:.0f}s',flush=True)
            completeness.append({'seed':seed,'task':protocol['tasks'][task],'ig_absolute_logit_error_median':float(np.median(errors)),'ig_absolute_logit_error_p95':float(np.quantile(errors,.95)),'median_absolute_logit_gap':float(np.median(np.abs(gaps)))})
        # Checkpoint completed seed only; never mistake uninitialized arrays for results.
        np.savez_compressed(ROOT/'results'/f'evaluation_seed_{seed}.npz',response=responses[si],**{k:v[si] for k,v in diagnostic_arrays.items()},patient_id=testmeta.patient_id.to_numpy(),ecg_id=testmeta.ecg_id.to_numpy(),labels=yt)
        pd.DataFrame(performance).to_csv(ROOT/'results'/'test_metrics.csv',index=False);pd.DataFrame(completeness).to_csv(ROOT/'results'/'ig_completeness.csv',index=False)
    np.savez_compressed(ROOT/'results'/'window_attributions.npz',scores=saved_maps,ecg_id=testmeta.ecg_id.to_numpy())
    (ROOT/'results'/'evaluation_axes.json').write_text(json.dumps({'seeds':protocol['model_seeds'],'tasks':protocol['tasks'],'methods':METHODS,'strata':STRATA,'budgets':budgets,'operators':OPERATORS,'per_seed_array_dimensions':['task','method','stratum','budget','operator','patient']},indent=2))
    print('Evaluation complete.',flush=True)

if __name__=='__main__':main()

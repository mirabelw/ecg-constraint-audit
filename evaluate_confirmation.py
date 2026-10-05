"""Frozen models, unused cohort, shared time shifts with measured geometry."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score
from model import ECGNet,ResidualECGNet
from geometry import project
from evaluate import make_mask,explanations
from evaluate_extension import logits,sigmoid
from extension_methods import amplitude_random_mask,dose_operators
from positive_control import run as wrapper_run
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'confirmation_results'
METHODS=['saliency','input_gradient','integrated_gradients']
KINDS=['guided','uniform_random','amplitude_matched_random','time_shift_dose_matched','time_shift_natural'];OPS=['independent','energy_matched','projected_unscaled']
def load_confirmation(cache):
    meta=pd.read_csv(OUT/'manifest.csv');old=pd.read_csv(ROOT/'results/manifest.csv');raw=np.load(cache/'signals.npy');assert raw.shape==(len(meta),12,1000) and np.isfinite(raw).all() and meta.patient_id.is_unique
    assert not set(meta.patient_id)&set(old.patient_id)
    scale=json.loads((ROOT/'results/preprocessing.json').read_text())['global_train_rms_mv'];centered=raw-raw.mean(-1,keepdims=True);x=project(centered).astype('float32')/scale
    (OUT/'data_quality.json').write_text(json.dumps({'n':len(meta),'label_counts':meta[['MI','CD']].sum().to_dict(),'human_validated_fraction':float(meta.validated_by_human.mean()),'global_train_rms_mv':scale,'raw_projection_rms_mv':float(np.sqrt(np.mean((raw-project(raw))**2))),'no_patient_overlap':True},indent=2))
    return meta,x
def shifted_mask(mask,offsets):
    cells=mask[:,:,::50];index=(np.arange(20)[None,:]-offsets[:,None])%20
    shifted=np.take_along_axis(cells,index[:,None,:],axis=2)
    # Exact structural control, including pairwise cross-lead alignment.
    assert np.array_equal(cells.sum(-1),shifted.sum(-1))
    assert np.array_equal(np.einsum('blt,bkt->blk',cells,cells),np.einsum('blt,bkt->blk',shifted,shifted))
    return np.repeat(shifted,50,-1)
def evaluate_model(model,x,ids,offsets,architecture,seed):
    n=len(x);response=np.empty((2,3,5,3,n),dtype='float32');normal=np.empty((2,3,5,n),dtype='float32');shift_response=np.empty((2,3,5,2,3,n),dtype='float32');shift_normal=np.empty((2,3,5,n),dtype='float32');shift_ratio=np.empty_like(shift_normal);shift_overlap=np.empty_like(shift_normal);scores_saved=np.empty((2,3,n,12,20),dtype='float32');dose_saved=np.empty((2,3,n),dtype='float32');dose_error=0.;original=logits(model,x);clock=time.monotonic()
    rs=np.stack([np.random.default_rng(1000000+int(e)).random((12,20)) for e in ids])
    for task in range(2):
        for start in range(0,n,64):
            end=min(n,start+64);xb=x[start:end];b=len(xb);maps,_,_=explanations(model,xb,task,32);base=sigmoid(original[start:end,task]);umask=make_mask(rs[start:end],'limb',.1,50)
            def compute(mask,dose=None):
                ds=dose_operators(-xb,mask,dose);pred=np.stack([np.abs(sigmoid(logits(model,xb+d)[:,task])-base) for d in ds[:3]])
                if dose is not None:
                    nonlocal dose_error
                    for d in ds[:2]:dose_error=max(dose_error,float(np.max(np.abs(np.sqrt((d*d).sum((1,2)))/dose-1))))
                return pred,ds[3]
            uniform=compute(umask)
            for m,method in enumerate(METHODS):
                score=maps[method].reshape(b,12,20,50).mean(-1);scores_saved[task,m,start:end]=score;mask=make_mask(score,'limb',.1,50);dose=np.sqrt(((xb*mask)**2).sum((1,2)));assert (dose>1e-10).all();dose_saved[task,m,start:end]=dose
                matched,_,_=amplitude_random_mask(-xb,mask,ids[start:end],m)
                for k,result in enumerate([compute(mask),uniform,compute(matched,dose)]):response[task,m,k,:,start:end]=result[0];normal[task,m,k,start:end]=result[1]
                for j in range(5):
                    smask=shifted_mask(mask,offsets[start:end,j]);natural=np.sqrt(((xb*smask)**2).sum((1,2)));assert (natural>1e-10).all()
                    shift_ratio[task,m,j,start:end]=natural/dose;shift_overlap[task,m,j,start:end]=(smask*mask).sum((1,2))/mask.sum((1,2))
                    for policy,ds in enumerate([compute(smask,dose),compute(smask)]):
                        shift_response[task,m,j,policy,:,start:end]=ds[0]
                        if policy==0:shift_normal[task,m,j,start:end]=ds[1]
                for policy,k in enumerate([3,4]):response[task,m,k,:,start:end]=shift_response[task,m,:,policy,:,start:end].mean(0);normal[task,m,k,start:end]=shift_normal[task,m,:,start:end].mean(0)
            if end==n or end%256==0:print(f'{architecture} seed={seed} task={task} n={end}/{n} elapsed={time.monotonic()-clock:.0f}s',flush=True)
    assert dose_error<2e-5
    np.savez_compressed(OUT/f'{architecture}_seed_{seed}.npz',response=response,normal_energy_fraction=normal,shift_response=shift_response,shift_normal_energy_fraction=shift_normal,shift_natural_dose_ratio=shift_ratio,shift_mask_overlap=shift_overlap,window_scores=scores_saved,guided_dose=dose_saved,offsets=offsets,ecg_id=ids,original_logits=original)
    return original,dose_error
def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT.parent/'confirmation_cache');a=p.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assert pd.read_csv(ROOT/'results/validation_metrics.csv').gate_pass.all() and pd.read_csv(ROOT/'extension_results/residual_validation.csv').gate_pass.all()
    meta,x=load_confirmation(a.cache);ids=meta.ecg_id.to_numpy();offsets=np.stack([np.random.default_rng(2200000+int(e)).choice(np.arange(1,20),5,replace=False) for e in ids]);metrics=[];checks={};baseline={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'checkpoints').glob('*.pt')}
    (OUT/'execution_freeze.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256((ROOT/'confirmation_protocol.json').read_bytes()).hexdigest(),'model_sha256':baseline,'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    for architecture in ['original_cnn','residual_cnn']:
        for seed in [17,29,43]:
            model=ECGNet() if architecture=='original_cnn' else ResidualECGNet();path=ROOT/'checkpoints'/(f'seed_{seed}.pt' if architecture=='original_cnn' else f'residual_seed_{seed}.pt');model.load_state_dict(torch.load(path,weights_only=True)['state_dict']);model.eval()
            pred,err=evaluate_model(model,x,ids,offsets,architecture,seed);checks[f'{architecture}_{seed}_max_relative_dose_error']=err
            for t,task in enumerate(['MI','CD']):metrics.append({'architecture':architecture,'seed':seed,'task':task,'auroc':float(roc_auc_score(meta[task],sigmoid(pred[:,t]))),'auprc':float(average_precision_score(meta[task],sigmoid(pred[:,t])))})
            pd.DataFrame(metrics).to_csv(OUT/'model_metrics.csv',index=False)
    # Confirm only the previously selected lambda10; no new grid or template search.
    model=ECGNet();model.load_state_dict(torch.load(ROOT/'checkpoints/seed_17.pt',weights_only=True)['state_dict']);model.eval();w=np.load(ROOT/'extension_results/positive_control_test.npz')['w'];frozen,recomputed,invariants=wrapper_run(model,x,ids,w,lambdas=[0,10]);np.savez_compressed(OUT/'wrapper_confirmation.npz',frozen=frozen,recomputed=recomputed,w=w,lambdas=np.array([0,10]),ecg_id=ids);checks['wrapper']=invariants
    assert invariants['on_consistent_max_logit_difference']<1e-5 and invariants['fixed_mask_consistent_max_response_difference']<1e-5 and invariants['tangent_gradient_max_difference']<1e-5
    assert baseline=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'checkpoints').glob('*.pt')}
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2));(OUT/'axes.json').write_text(json.dumps({'methods':METHODS,'kinds':KINDS,'operators':OPS,'tasks':['MI','CD'],'seeds':[17,29,43],'response_dimensions':['task','method','kind','operator','patient'],'shift_response_dimensions':['task','method','shift','dose_policy','operator','patient'],'dose_policy':['matched','natural']},indent=2))
    print('Confirmation evaluation complete. Model states unchanged.',flush=True)
if __name__=='__main__':main()

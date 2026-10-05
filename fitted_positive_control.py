import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from geometry import project,interventions
from train import load_data
from model import ECGNet
from evaluate import make_mask
from evaluate_extension import logits,sigmoid
from positive_control import signed,run,LAMBDAS
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'extension_results'

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT.parent/'data_cache');a=p.parse_args();torch.set_num_threads(2)
    meta,x,y,scale=load_data(a.cache);model=ECGNet();model.load_state_dict(torch.load(ROOT/'checkpoints/seed_17.pt',weights_only=True)['state_dict']);model.eval();va=meta.split.eq('validation').to_numpy();xv=project(x[va].astype('float64'));base=logits(model,xv)[:,0];normal=[[],[]];perturbed=[[],[]]
    for start in range(0,len(xv),64):
        xb=xv[start:start+64];g,integral=signed(model,xb,0)
        for k,score in enumerate([np.abs(g),np.abs(xb*integral)]):
            mask=make_mask(score.reshape(len(xb),12,20,50).mean(-1),'limb',.1,50);raw=interventions(xb,mask)['independent_zero'];normal[k].append((raw-project(raw)).reshape(len(xb),-1));perturbed[k].append(logits(model,project(xb+raw))[:,0])
    d=[torch.from_numpy(np.concatenate(v).astype('float32')) for v in normal];b=[torch.from_numpy(np.concatenate(v)) for v in perturbed];p0=torch.from_numpy(sigmoid(base));rng=np.random.default_rng(872);v=project(rng.normal(size=(12,1000)));v=rng.normal(size=(12,1000))-v
    # Start from an actual normal template; gradients stay in the normal span.
    v=np.zeros((12,1000));v[:6]=rng.normal(size=(6,1000));v-=project(v);v/=np.linalg.norm(v);parameter=torch.nn.Parameter(torch.from_numpy(v.reshape(-1).astype('float32')));optimizer=torch.optim.Adam([parameter],lr=.05);history=[]
    for step in range(200):
        optimizer.zero_grad();response=[torch.abs(torch.sigmoid(b[k]+d[k]@parameter)-p0).mean() for k in range(2)];loss=-(response[0]-response[1]);loss.backward();optimizer.step()
        with torch.no_grad():parameter*=min(1.,30/float(parameter.norm()))
        if step%10==0:history.append({'step':step+1,'saliency_minus_IG_validation_gap':float((-loss).detach()),'vector_norm':float(parameter.detach().norm())})
    fitted=parameter.detach().numpy().reshape(12,1000).astype('float64');fitted-=project(fitted);w=fitted/np.linalg.norm(fitted);pd.DataFrame(history).to_csv(OUT/'fitted_control_optimization.csv',index=False)
    vf,vr,vc=run(model,x[va],meta.loc[va,'ecg_id'].to_numpy(),w);means=vf.mean(-1);old=means[0,0,0,0]-means[0,2,0,0];chosen=None
    for li,lam in enumerate(LAMBDAS[1:],1):
        if abs(old)>.001 and old*(means[0,0,li,0]-means[0,2,li,0])<0:chosen=li;break
    selection={'validation_selected_lambda':None if chosen is None else LAMBDAS[chosen],'validation_base_saliency_minus_IG':float(old),'validation_new_gap':None if chosen is None else float(means[0,0,chosen,0]-means[0,2,chosen,0]),'fitted_direction_only_used_validation':True,'selection_failure':chosen is None}
    (OUT/'fitted_control_selection.json').write_text(json.dumps(selection,indent=2));np.savez_compressed(OUT/'fitted_control_validation.npz',frozen=vf,recomputed=vr,w=w,lambdas=np.array(LAMBDAS))
    test=meta.split.eq('test').to_numpy();tf,tr,tc=run(model,x[test],meta.loc[test,'ecg_id'].to_numpy(),w);np.savez_compressed(OUT/'fitted_control_test.npz',frozen=tf,recomputed=tr,w=w,lambdas=np.array(LAMBDAS),ecg_id=meta.loc[test,'ecg_id'].to_numpy())
    result={'selection':selection,'validation_checks':vc,'test_checks':tc}
    if chosen is not None:
        old=tf[0,0,0,0]-tf[0,2,0,0];new=tf[0,0,chosen,0]-tf[0,2,chosen,0];rng=np.random.default_rng(901);inds=rng.integers(len(old),size=(5000,len(old)),dtype=np.int32)
        for name,values in [('raw_old',old),('raw_new',new)]:
            bs=np.concatenate([values[i].mean(1) for i in np.array_split(inds,20)]);result[name]={'gap':float(values.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()}
        result['test_fixed_mask_point_reversal']=bool(old.mean()*new.mean()<0)
    assert tc['on_consistent_max_logit_difference']<1e-5 and tc['fixed_mask_consistent_max_response_difference']<1e-5
    (OUT/'fitted_control_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

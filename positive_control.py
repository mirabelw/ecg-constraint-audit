"""Real-model off-subspace wrapper; validation-selected control, fixed/recomputed masks.

This is a verification control implementing established Anders et al. logic,
not a proposed novel attack. Projection alone cannot fix altered attribution
selection; frozen masks and recomputed masks are separate estimands.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from geometry import project,interventions
from train import load_data
from model import ECGNet
from evaluate import make_mask
from evaluate_extension import logits,sigmoid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'extension_results';LAMBDAS=[0,1,3,10,30]

def signed(model,x,task):
    t=torch.from_numpy(x.astype('float32')).requires_grad_(True);g=project(torch.autograd.grad(model(t)[:,task].sum(),t)[0].detach().numpy());integral=np.zeros_like(x)
    for i in range(32):
        u=(t.detach()*((i+.5)/32)).requires_grad_(True);integral+=project(torch.autograd.grad(model(u)[:,task].sum(),u)[0].detach().numpy())/32
    return g,integral

def run(model,x,ids,w,lambdas=None):
    lambdas=LAMBDAS if lambdas is None else lambdas
    x=project(x.astype('float64'));n=len(x);shape=(2,3,len(lambdas),2,n);frozen=np.empty(shape);recomputed=np.empty(shape);max_logit_difference=0.;tangent_difference=0.;base=logits(model,project(x));base_probability=sigmoid(base)
    for task in range(2):
        for start in range(0,n,64):
            end=min(n,start+64);xb=x[start:end];g,integral=signed(model,xb,task)
            scores0=[np.abs(g),np.abs(xb*g),np.abs(xb*integral)];mask0=[make_mask(s.reshape(len(xb),12,20,50).mean(-1),'limb',.1,50) for s in scores0]
            for li,lam in enumerate(lambdas):
                original=base[start:end,task]+lam*np.einsum('lt,blt->b',w,xb-project(xb));max_logit_difference=max(max_logit_difference,float(np.abs(original-base[start:end,task]).max()))
                tangent_difference=max(tangent_difference,float(np.abs(project(g+lam*w)-project(g)).max()))
                scores=[np.abs(g+lam*w),np.abs(xb*(g+lam*w)),np.abs(xb*(integral+lam*w))]
                for m in range(3):
                    new_mask=make_mask(scores[m].reshape(len(xb),12,20,50).mean(-1),'limb',.1,50)
                    for out,mask in [(frozen,mask0[m]),(recomputed,new_mask)]:
                        ds=interventions(xb,mask)
                        for o,name in enumerate(['independent_zero','energy_matched']):
                            z=xb+ds[name];pert=logits(model,project(z))[:,task]+lam*np.einsum('lt,blt->b',w,z-project(z));out[task,m,li,o,start:end]=np.abs(sigmoid(pert)-sigmoid(original))
    return frozen,recomputed,{'on_consistent_max_logit_difference':max_logit_difference,'tangent_gradient_max_difference':tangent_difference,'fixed_mask_consistent_max_response_difference':float(np.abs(frozen[:,:,:,1]-frozen[:,:,0:1,1]).max())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT.parent/'data_cache');a=p.parse_args();torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    meta,x,y,scale=load_data(a.cache);model=ECGNet();model.load_state_dict(torch.load(ROOT/'checkpoints/seed_17.pt',weights_only=True)['state_dict']);model.eval()
    rng=np.random.default_rng(872);w=np.zeros((12,1000));w[:6,:200]=rng.normal(size=(6,200));w=w-project(w);w/=np.linalg.norm(w);assert np.linalg.norm(project(w))<1e-12
    validations=meta.split.eq('validation').to_numpy();vf,vr,vc=run(model,x[validations],meta.loc[validations,'ecg_id'].to_numpy(),w)
    vmean=vf.mean(-1);chosen=None;selection=None
    for li,lam in enumerate(LAMBDAS[1:],start=1):
        for task in range(2):
            for aidx,bidx in [(0,1),(0,2),(1,2)]:
                old=vmean[task,aidx,0,0]-vmean[task,bidx,0,0];new=vmean[task,aidx,li,0]-vmean[task,bidx,li,0]
                if abs(old)>.001 and old*new<0 and chosen is None:chosen=li;selection={'lambda':lam,'task':['MI','CD'][task],'method_a':aidx,'method_b':bidx,'validation_old_gap':float(old),'validation_new_gap':float(new),'selection':'first fixed grid level with a validation point-order reversal and baseline gap >0.001'}
        if chosen is not None:break
    if chosen is None:selection={'selection':'control failed to find a validation reversal in frozen lambda grid'}
    # Record validation-only selection before test evaluation.
    (OUT/'positive_control_selection.json').write_text(json.dumps(selection,indent=2));np.savez_compressed(OUT/'positive_control_validation.npz',frozen=vf,recomputed=vr,w=w,lambdas=np.array(LAMBDAS))
    test=meta.split.eq('test').to_numpy();tf,tr,tc=run(model,x[test],meta.loc[test,'ecg_id'].to_numpy(),w);np.savez_compressed(OUT/'positive_control_test.npz',frozen=tf,recomputed=tr,w=w,lambdas=np.array(LAMBDAS),ecg_id=meta.loc[test,'ecg_id'].to_numpy())
    rows=[]
    for kind,values in [('fixed_masks',tf),('recomputed_masks',tr)]:
        for task in range(2):
            for m,method in enumerate(['saliency','input_gradient','integrated_gradients']):
                for li,lam in enumerate(LAMBDAS):
                    for o,operator in enumerate(['independent','energy_matched']):rows.append({'mask_policy':kind,'task':['MI','CD'][task],'method':method,'lambda':lam,'operator':operator,'response':float(values[task,m,li,o].mean())})
    pd.DataFrame(rows).to_csv(OUT/'positive_control_scores.csv',index=False)
    assert tc['on_consistent_max_logit_difference']<1e-5 and tc['fixed_mask_consistent_max_response_difference']<1e-5 and tc['tangent_gradient_max_difference']<1e-5
    result={'validation_checks':vc,'test_checks':tc,'validation_selected':selection,'control_found_validation_reversal':chosen is not None}
    if chosen is not None:
        t=['MI','CD'].index(selection['task']);ai=selection['method_a'];bi=selection['method_b'];old=tf[t,ai,0,0]-tf[t,bi,0,0];new=tf[t,ai,chosen,0]-tf[t,bi,chosen,0];rng=np.random.default_rng(901);indices=rng.integers(len(old),size=(5000,len(old)),dtype=np.int32)
        for label,v in [('raw_old',old),('raw_new',new)]:
            bs=np.concatenate([v[ind].mean(1) for ind in np.array_split(indices,20)]);lo,hi=np.quantile(bs,[.025,.975]);result[label]={'gap':float(v.mean()),'ci95':[float(lo),float(hi)]}
        result['test_fixed_mask_point_reversal']=bool(old.mean()*new.mean()<0)
    (OUT/'positive_control_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

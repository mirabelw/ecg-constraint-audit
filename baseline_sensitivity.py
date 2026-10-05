"""Disclosed data-quality sensitivity; not an additional primary endpoint."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from geometry import project
from train import predict
from model import ECGNet
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT.parent/'data_cache');a=p.parse_args();torch.set_num_threads(4)
    meta=pd.read_csv(ROOT/'results'/'manifest.csv');raw=np.load(a.cache/'signals.npy');res=raw[:,:6]-project(raw)[:,:6]
    peak=np.abs(res).max((1,2));rms=np.sqrt((res*res).mean((1,2)))
    quality=pd.DataFrame({'ecg_id':meta.ecg_id,'patient_id':meta.patient_id,'split':meta.split,'projection_max_mv':peak,'projection_rms_mv':rms});quality.to_csv(ROOT/'results'/'recording_constraint_residuals.csv',index=False)
    test=meta.split.eq('test').to_numpy();original=raw[test].copy();original-=original.mean(-1,keepdims=True);canonical=project(original).astype('float32');rows=[]
    for seed in [17,29,43]:
        checkpoint=torch.load(ROOT/'checkpoints'/f'seed_{seed}.pt',weights_only=True);model=ECGNet();model.load_state_dict(checkpoint['state_dict']);model.eval();s=checkpoint['scale']
        difference=np.abs(predict(model,original/s)-predict(model,canonical/s))
        for t,task in enumerate(['MI','CD']):rows.append({'seed':seed,'task':task,'mean_abs_probability_change':float(difference[:,t].mean()),'p95_abs_probability_change':float(np.quantile(difference[:,t],.95)),'max_abs_probability_change':float(difference[:,t].max())})
    pd.DataFrame(rows).to_csv(ROOT/'results'/'baseline_projection_prediction_changes.csv',index=False)
    (ROOT/'results'/'recording_quality_summary.json').write_text(json.dumps({'threshold_mv':.01,'threshold_purpose':'posthoc recording-quality sensitivity, not patient exclusion from primary analysis','n_over_threshold_by_split':quality.assign(over=peak>.01).groupby('split').over.sum().astype(int).to_dict(),'projection_peak_quantiles_mv':{str(q):float(np.quantile(peak,q)) for q in [.5,.95,.99,.999,1]}},indent=2))
    # Exclude only the one high-residual test recording for a disclosed sensitivity
    # calculation. Original primary results retain all 1000 test patients.
    if (ROOT/'results'/'evaluation_axes.json').exists():
        axes=json.loads((ROOT/'results'/'evaluation_axes.json').read_text());keep=peak[test]<=.01
        response=np.stack([np.load(ROOT/'results'/f'evaluation_seed_{seed}.npz')['response'] for seed in axes['seeds']]).mean(0);b=axes['budgets'].index(.1);contrasts=[]
        for t,task in enumerate(axes['tasks']):
            for m,method in enumerate(axes['methods'][:3]):
                v=(response[t,m,0,b,2]-response[t,3,0,b,2])-(response[t,m,0,b,0]-response[t,3,0,b,0])
                contrasts.append({'task':task,'method':method,'all_patients_delta_advantage':float(v.mean()),'low_residual_patients_delta_advantage':float(v[keep].mean()),'sensitivity_n':int(keep.sum())})
        pd.DataFrame(contrasts).to_csv(ROOT/'results'/'low_residual_sensitivity.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()

"""Validation-only integration convergence diagnostic; does not tune test results."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from evaluate import explanations
from train import load_data
from model import ECGNet
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT.parent/'data_cache');a=p.parse_args();torch.set_num_threads(4)
    meta,x,y,s=load_data(a.cache);v=x[meta.split.eq('validation').to_numpy()][:32];rows=[]
    for seed in [17,29,43]:
        model=ECGNet();model.load_state_dict(torch.load(ROOT/'checkpoints'/f'seed_{seed}.pt',weights_only=True)['state_dict']);model.eval()
        for t,task in enumerate(['MI','CD']):
            maps32,e32,gap=explanations(model,v,t,32);maps64,e64,_=explanations(model,v,t,64)
            scores=[m['integrated_gradients'].reshape(32,12,20,50).mean(-1).reshape(32,-1) for m in [maps32,maps64]]
            corr=[spearmanr(a,b).statistic for a,b in zip(*scores)]
            rows.append({'seed':seed,'task':task,'n_validation_records':32,'median_window_rank_correlation_32_vs_64':float(np.median(corr)),'median_absolute_logit_error_32':float(np.median(e32)),'median_absolute_logit_error_64':float(np.median(e64)),'median_absolute_logit_gap':float(np.median(np.abs(gap)))})
    pd.DataFrame(rows).to_csv(ROOT/'results'/'ig_validation_convergence.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()

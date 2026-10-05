"""Explicit posthoc verification analysis of the already fixed control grid.

All failed controls are retained. This analysis must not be presented as an
independent preregistered control success. It separates model identity on the
subspace, evaluator invariance for fixed masks, and changed map selection.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'extension_results'

def main():
    val=np.load(OUT/'positive_control_validation.npz');test=np.load(OUT/'positive_control_test.npz');lambdas=test['lambdas'];validation=val['recomputed'][0,0,:,0].mean(-1)-val['recomputed'][0,2,:,0].mean(-1)
    eligible=np.flatnonzero((validation>0.005)&(lambdas>0));chosen=None if len(eligible)==0 else int(eligible[0]);selection={'analysis_status':'Posthoc verification reanalysis of the fixed random-template grid, after fixed-mask controls failed; original grid and failures retained. Choice computed from validation only, but test grid had already been examined. Not an independent validation.', 'mask_policy':'recomputed','pair':'MI saliency minus IG','minimum_validation_new_gap':.005,'chosen_lambda':None if chosen is None else int(lambdas[chosen])}
    rng=np.random.default_rng(901);inds=rng.integers(1000,size=(5000,1000),dtype=np.int32);rows=[]
    for policy in ['frozen','recomputed']:
        for li,lam in enumerate(lambdas):
            for oi,operator in enumerate(['independent','energy_matched']):
                gap=test[policy][0,0,li,oi]-test[policy][0,2,li,oi];bs=np.concatenate([gap[i].mean(1) for i in np.array_split(inds,20)]);lo,hi=np.quantile(bs,[.025,.975]);rows.append({'policy':policy,'lambda':int(lam),'operator':operator,'saliency_minus_IG_gap':float(gap.mean()),'ci95_low':float(lo),'ci95_high':float(hi)})
    pd.DataFrame(rows).to_csv(OUT/'control_grid_pair_gaps.csv',index=False);(OUT/'control_grid_interpretation.json').write_text(json.dumps(selection,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True);df=pd.DataFrame(rows)
    for ax,policy in zip(axs,['frozen','recomputed']):
        for operator,label in [('independent','Independent'),('energy_matched','Consistent, matched energy')]:
            d=df[(df.policy==policy)&(df.operator==operator)];center=d.saliency_minus_IG_gap.to_numpy()*100;lo=d.ci95_low.to_numpy()*100;hi=d.ci95_high.to_numpy()*100;ax.errorbar(np.arange(5),center,yerr=np.stack([center-lo,hi-center]),marker='o',label=label,capsize=3)
        ax.axhline(0,color='gray',ls='--');ax.set_xticks(np.arange(5),lambdas);ax.set_xlabel('Normal-direction coefficient λ');ax.set_title('Fixed masks' if policy=='frozen' else 'Recomputed explanations');ax.grid(alpha=.2)
    axs[0].set_ylabel('Saliency − IG response gap (percentage points)');axs[1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'control_grid.png',dpi=200);plt.close(fig);print(json.dumps(selection,indent=2))

if __name__=='__main__':main()

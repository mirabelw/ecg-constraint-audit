"""Patient bootstrap, paired across operators and methods; seeds averaged first."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());axes=json.loads((ROOT/'results'/'evaluation_axes.json').read_text())
    files=[np.load(ROOT/'results'/f'evaluation_seed_{s}.npz') for s in axes['seeds']]
    for f in files[1:]:assert np.array_equal(f['patient_id'],files[0]['patient_id'])
    assert len(np.unique(files[0]['patient_id']))==len(files[0]['patient_id'])
    r=np.stack([f['response'] for f in files]);assert np.isfinite(r).all();avg=r.mean(0);n=r.shape[-1]
    rng=np.random.default_rng(protocol['bootstrap_seed']);indices=rng.integers(0,n,size=(protocol['bootstrap_replicates'],n),dtype=np.int32)
    def ci(values,alpha=.05):
        # Small chunks avoid materializing a huge all-endpoint bootstrap tensor.
        bs=np.concatenate([values[i].mean(1) for i in np.array_split(indices,20)])
        return np.quantile(bs,[alpha/2,1-alpha/2])
    rows=[];diagnostic=[]
    for t,task in enumerate(axes['tasks']):
        for m,method in enumerate(axes['methods']):
            for st,stratum in enumerate(axes['strata']):
                for b,budget in enumerate(axes['budgets']):
                    for o,operator in enumerate(axes['operators']):
                        v=avg[t,m,st,b,o];lo,hi=ci(v)
                        key={'task':task,'method':method,'stratum':stratum,'budget':budget,'operator':operator}
                        rows.append({**key,'response':float(v.mean()),'ci95_low':float(lo),'ci95_high':float(hi)})
                        di={**key}
                        for k in ['dose_ratio','collateral_fraction','selected_relative_error','off_subspace_rms_mv']:
                            values=np.stack([f[k][t,m,st,b,o] for f in files]).mean(0);di[k+'_mean']=float(values.mean());di[k+'_p95']=float(np.quantile(values,.95))
                        diagnostic.append(di)
    pd.DataFrame(rows).to_csv(ROOT/'results'/'response_summary.csv',index=False);pd.DataFrame(diagnostic).to_csv(ROOT/'results'/'intervention_diagnostics.csv',index=False)
    contrasts=[];rank_contrasts=[];bi=axes['budgets'].index(protocol['primary_budget']);st=axes['strata'].index('limb');margin=protocol['equivalence_margin_probability'];alpha=.05/protocol['primary_family_size']
    for t,task in enumerate(axes['tasks']):
        for m,method in enumerate(axes['methods'][:3]):
            for o,operator in enumerate(axes['operators'][1:],start=1):
                value=(avg[t,m,st,bi,o]-avg[t,3,st,bi,o])-(avg[t,m,st,bi,0]-avg[t,3,st,bi,0])
                low,high=ci(value,alpha if o==2 else .05)
                result='equivalent' if low>-margin and high<margin else 'beyond_margin' if low>margin or high<-margin else 'inconclusive'
                contrasts.append({'task':task,'method':method,'operator':operator,'delta_advantage':float(value.mean()),'ci_low':float(low),'ci_high':float(high),'interval_coverage':1-(alpha if o==2 else .05),'equivalence_margin':margin,'conclusion':result,**{f'equivalent_margin_{s:g}':bool(low>-s and high<s) for s in [.005,.02]}})
        for a,b in [(0,1),(0,2),(1,2)]:
            for o,operator in enumerate(axes['operators']):
                value=avg[t,a,st,bi,o]-avg[t,b,st,bi,o];lo,hi=ci(value)
                rank_contrasts.append({'task':task,'method_a':axes['methods'][a],'method_b':axes['methods'][b],'operator':operator,'a_minus_b':float(value.mean()),'ci95_low':float(lo),'ci95_high':float(hi),'ordering':'a>b' if lo>0 else 'a<b' if hi<0 else 'unresolved'})
    pd.DataFrame(contrasts).to_csv(ROOT/'results'/'primary_contrasts.csv',index=False);pd.DataFrame(rank_contrasts).to_csv(ROOT/'results'/'method_rank_contrasts.csv',index=False)
    # Per-seed contrasts expose model-to-model heterogeneity that the conditional
    # patient bootstrap does not include as population-level uncertainty.
    seedrows=[]
    for s,seed in enumerate(axes['seeds']):
        for t,task in enumerate(axes['tasks']):
            for m,method in enumerate(axes['methods'][:3]):
                v=(r[s,t,m,st,bi,2]-r[s,t,3,st,bi,2])-(r[s,t,m,st,bi,0]-r[s,t,3,st,bi,0])
                seedrows.append({'seed':seed,'task':task,'method':method,'delta_advantage':float(v.mean())})
    pd.DataFrame(seedrows).to_csv(ROOT/'results'/'per_seed_primary.csv',index=False)
    chest=axes['strata'].index('chest');err=float(np.max(np.abs(r[:,:,:,chest,:,:,]-r[:,:,:,chest,:,0:1,])))
    assert err<1e-7,'Chest control differs across operators.'
    (ROOT/'results'/'statistical_checks.json').write_text(json.dumps({'unique_test_patients':n,'chest_max_operator_response_difference':err,'bootstrap_replicates':len(indices),'primary_family_size':protocol['primary_family_size'],'patient_intervals_conditional_on_three_fitted_models':True},indent=2))
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for t,task in enumerate(axes['tasks']):
        ax=axs[t]
        for m,method in enumerate(axes['methods']):
            ys=[avg[t,m,st,bi,o].mean() for o in range(3)];ax.plot(range(3),ys,marker='o',label=method.replace('_',' '))
        ax.set_xticks(range(3),['Independent','Target matched','Energy matched']);ax.set_title(f'{task}: limb-only, 10% budget');ax.set_ylabel('Mean absolute probability change');ax.grid(alpha=.2)
    axs[1].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'results'/'operator_comparison.png',dpi=200);plt.close(fig)
    primary=pd.DataFrame(contrasts).query("operator=='energy_matched'").reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(9,4.8));ypos=np.arange(len(primary));center=primary.delta_advantage.to_numpy()*100;low=primary.ci_low.to_numpy()*100;high=primary.ci_high.to_numpy()*100
    ax.axvspan(-margin*100,margin*100,color='#dcebdc',alpha=.7,label='Prespecified operational equivalence range')
    ax.axvline(0,color='gray',linestyle='--',linewidth=1)
    ax.errorbar(center,ypos,xerr=np.stack([center-low,high-center]),fmt='o',color='#176b92',capsize=4)
    ax.set_yticks(ypos,[f'{row.task}: {row.method.replace("_"," ")}' for row in primary.itertuples()]);ax.invert_yaxis();ax.set_xlim(-1.1,1.1)
    ax.set_xlabel('Change in method advantage over random (percentage points)');ax.set_title('Historical absolute-margin check: all six intervals within ±1 point')
    ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',fontsize=8);fig.tight_layout();fig.savefig(ROOT/'results'/'primary_equivalence.png',dpi=200);plt.close(fig)
    print(pd.DataFrame(contrasts).to_string(index=False));print('chest identity',err)

if __name__=='__main__':main()

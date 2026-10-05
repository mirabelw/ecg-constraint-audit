import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score
from evaluate_extension import sigmoid
from extension_methods import METHODS
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'extension_results'

def main():
    rng=np.random.default_rng(881);indices=rng.integers(1000,size=(5000,1000),dtype=np.int32)
    def ci(v):return np.quantile(np.concatenate([v[i].mean(1) for i in np.array_split(indices,20)]),[.025,.975])
    summaries=[];comparisons=[];geometry=[];pairs=[];change_vectors=[];performance=[];checks=[];replacement=[];meta=pd.read_csv(ROOT/'results/manifest.csv');yt=meta.loc[meta.split.eq('test'),['MI','CD']].to_numpy();ids=meta.loc[meta.split.eq('test'),'ecg_id'].to_numpy()
    for architecture in ['original_cnn','residual_cnn']:
        files=[np.load(OUT/f'{architecture}_seed_{seed}.npz') for seed in [17,29,43]]
        assert all(np.array_equal(f['ecg_id'],ids) for f in files);r=np.stack([f['response'] for f in files]);avg=r.mean(0)
        for f,seed in zip(files,[17,29,43]):
            p=sigmoid(f['original_logits']);auc=roc_auc_score(yt,p,average=None);ap=average_precision_score(yt,p,average=None)
            for t,task in enumerate(['MI','CD']):performance.append({'architecture':architecture,'seed':seed,'task':task,'test_auroc':float(auc[t]),'test_auprc':float(ap[t])})
            if architecture=='original_cnn':
                old=np.load(ROOT/'results'/f'evaluation_seed_{seed}.npz')['response'][:,:3,0,1][:,:,[0,2]];new=f['response'][:,:3,0,:2];error=float(np.abs(old-new).max());assert error<2e-6;checks.append({'architecture':architecture,'seed':seed,'original_reproduction_max_difference':error})
        for t,task in enumerate(['MI','CD']):
            for m,method in enumerate(METHODS):
                for kind,kname in enumerate(['guided','uniform_random','amplitude_matched_random']):
                    for o,oname in enumerate(['independent','energy_matched','projected_unscaled']):
                        v=avg[t,m,kind,o];lo,hi=ci(v);summaries.append({'architecture':architecture,'task':task,'method':method,'kind':kname,'operator':oname,'response':float(v.mean()),'ci95_low':float(lo),'ci95_high':float(hi)})
                    nfrac=np.stack([f['normal_energy_fraction'][t,m,kind] for f in files]).mean(0);off=np.stack([f['off_direction_logit_response'][t,m,kind] for f in files]).mean(0);rawlog=np.stack([f['logit_response'][t,m,kind,0] for f in files]).mean(0)
                    geometry.append({'architecture':architecture,'task':task,'method':method,'kind':kname,'normal_energy_fraction':float(nfrac.mean()),'mean_off_direction_abs_logit_response':float(off.mean()),'mean_raw_abs_logit_response':float(rawlog.mean()),'off_to_raw_logit_response_ratio':float(off.mean()/max(rawlog.mean(),1e-12)),'raw_probability_response':float(avg[t,m,kind,0].mean()),'projected_unscaled_probability_response':float(avg[t,m,kind,2].mean()),'energy_matched_probability_response':float(avg[t,m,kind,1].mean())})
                for kind,kname in [(1,'uniform_random'),(2,'amplitude_matched_random')]:
                    a=avg[t,m,0,0]-avg[t,m,kind,0];c=avg[t,m,0,1]-avg[t,m,kind,1];delta=c-a;lo,hi=ci(delta)
                    comparisons.append({'architecture':architecture,'task':task,'method':method,'comparator':kname,'raw_advantage':float(a.mean()),'consistent_advantage':float(c.mean()),'delta_advantage':float(delta.mean()),'ci95_low':float(lo),'ci95_high':float(hi),'relative_change':float(delta.mean()/a.mean()) if abs(a.mean())>1e-6 else np.nan,'matched_pre_scale_dose_ratio':float(np.stack([f['matched_pre_scale_dose_ratio'][t,m] for f in files]).mean()) if kind==2 else np.nan,'matched_mask_overlap':float(np.stack([f['matched_mask_overlap'][t,m] for f in files]).mean()) if kind==2 else np.nan})
            # Near competitors: all pair gaps are descriptive, with changes paired.
            for a in range(7):
                for b in range(a+1,7):
                    for endpoint in ['guided_response','amplitude_matched_advantage']:
                        raw=avg[t,a,0,0]-avg[t,b,0,0];con=avg[t,a,0,1]-avg[t,b,0,1]
                        if endpoint=='amplitude_matched_advantage':raw-=avg[t,a,2,0]-avg[t,b,2,0];con-=avg[t,a,2,1]-avg[t,b,2,1]
                        lo0,hi0=ci(raw);lo1,hi1=ci(con);lod,hid=ci(con-raw)
                        change_vectors.append(con-raw)
                        pairs.append({'architecture':architecture,'task':task,'endpoint':endpoint,'method_a':METHODS[a],'method_b':METHODS[b],'raw_gap':float(raw.mean()),'raw_ci95_low':float(lo0),'raw_ci95_high':float(hi0),'consistent_gap':float(con.mean()),'consistent_ci95_low':float(lo1),'consistent_ci95_high':float(hi1),'gap_change':float((con-raw).mean()),'gap_change_ci95_low':float(lod),'gap_change_ci95_high':float(hid),'point_reversal':bool(raw.mean()*con.mean()<0),'resolved_opposite_order':bool((lo0>0 and hi1<0) or (hi0<0 and lo1>0))})
        mean_response=np.stack([f['mean_replacement_response'] for f in files]).mean(0)
        for t,task in enumerate(['MI','CD']):
            for m,method in enumerate([METHODS[i] for i in [0,1,2,6]]):
                raw=mean_response[t,m,0,0]-mean_response[t,m,1,0];con=mean_response[t,m,0,1]-mean_response[t,m,1,1];lo,hi=ci(con-raw)
                replacement.append({'architecture':architecture,'task':task,'method':method,'raw_advantage':float(raw.mean()),'consistent_advantage':float(con.mean()),'delta_advantage':float((con-raw).mean()),'ci95_low':float(lo),'ci95_high':float(hi)})
    # Exploratory simultaneous intervals for all 168 gap-change endpoints,
    # standardized by observed patient SEs, with one paired bootstrap family.
    vectors=np.stack(change_vectors);se=vectors.std(1,ddof=1)/np.sqrt(1000);weights=np.stack([np.bincount(i,minlength=1000) for i in indices]).astype('float32')/1000
    centered=weights@vectors.T-vectors.mean(1);standardized=np.divide(centered,se[None,:],out=np.zeros_like(centered),where=se[None,:]>1e-12);critical=float(np.quantile(np.abs(standardized).max(1),.95))
    for row,s in zip(pairs,se):
        row['simultaneous_gap_change_low']=row['gap_change']-critical*s;row['simultaneous_gap_change_high']=row['gap_change']+critical*s;row['simultaneous_gap_change_excludes_zero']=bool(row['simultaneous_gap_change_low']>0 or row['simultaneous_gap_change_high']<0)
    (OUT/'simultaneous_interval_design.json').write_text(json.dumps({'status':'exploratory multiplicity diagnostic, added after reviewing descriptive extension outputs','family_size':len(pairs),'bootstrap_replicates':5000,'critical_standardized_max_deviation':critical,'patient_SE_standardization':'observed SE, seeds averaged before bootstrap; conditional on fitted models'},indent=2))
    for name,data in [('response_summary',summaries),('comparator_contrasts',comparisons),('geometry_mechanism',geometry),('pairwise_ranking',pairs),('extension_test_metrics',performance),('reproduction_checks',checks),('mean_replacement_contrasts',replacement)]:pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)
    relative=pd.read_csv(OUT/'original_relative_decomposition.csv');fig,ax=plt.subplots(figsize=(9,4.6));center=relative.relative_advantage_change.to_numpy()*100;low=relative.relative_adjusted_low.to_numpy()*100;high=relative.relative_adjusted_high.to_numpy()*100;y=np.arange(6)
    ax.axvspan(-10,10,color='#eeeeee');ax.axvline(0,color='gray',ls='--');ax.errorbar(center,y,xerr=np.stack([center-low,high-center]),fmt='o',capsize=4,color='#994533');ax.set_yticks(y,[f'{r.task}: {r.method.replace("_"," ")}' for r in relative.itertuples()]);ax.invert_yaxis();ax.set_xlabel('Relative change in advantage over uniform random (%)');ax.set_title('Original audit: absolute-margin result does not imply relative equivalence');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(OUT/'relative_effects.png',dpi=200);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ai,architecture in enumerate(['original_cnn','residual_cnn']):
        data=pd.DataFrame(summaries);data=data[(data.architecture==architecture)&(data.kind=='guided')&(data.operator.isin(['independent','energy_matched']))]
        for task,marker in [('MI','o'),('CD','s')]:
            rows=data[data.task==task];ys0=[];ys1=[]
            for method in METHODS:ys0.append(float(rows[(rows.method==method)&(rows.operator=='independent')].response.iloc[0]));ys1.append(float(rows[(rows.method==method)&(rows.operator=='energy_matched')].response.iloc[0]))
            axs[ai].plot(np.array(ys0)*100,np.array(ys1)*100,marker,ls='none',label=task)
            for m in range(7):axs[ai].annotate(str(m+1),(ys0[m]*100,ys1[m]*100),xytext=(3,3),textcoords='offset points',fontsize=8)
        axs[ai].plot([0,15],[0,15],ls='--',color='gray');axs[ai].set_xlabel('Independent response (percentage points)');axs[ai].set_ylabel('Energy-matched response (percentage points)');axs[ai].set_title(architecture.replace('_',' '));axs[ai].legend();axs[ai].grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'expanded_responses.png',dpi=200);plt.close(fig)
    print('Extension summary complete.',flush=True)

if __name__=='__main__':main()

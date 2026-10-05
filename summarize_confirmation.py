"""Patient-level paired confirmation summaries; no retuning."""
import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'confirmation_results'
def main():
    axes=json.loads((OUT/'axes.json').read_text());n=len(pd.read_csv(OUT/'manifest.csv'));rng=np.random.default_rng(20261007);weights=rng.multinomial(n,np.full(n,1/n),size=5000).astype('float64')/n
    def stats(v,alpha=.05):
        samples=weights@v;lo,hi=np.quantile(samples,[alpha/2,1-alpha/2]);return float(v.mean()),float(lo),float(hi)
    rows=[];geometry=[];difference=[];diagnostics=[]
    for architecture in ['original_cnn','residual_cnn']:
        files=[np.load(OUT/f'{architecture}_seed_{seed}.npz') for seed in [17,29,43]];r=np.stack([f['response'] for f in files]).mean(0).astype('float64');q=np.stack([f['normal_energy_fraction'] for f in files]).mean(0).astype('float64');ratio=np.stack([f['shift_natural_dose_ratio'] for f in files]).mean((0,3));overlap=np.stack([f['shift_mask_overlap'] for f in files]).mean((0,3))
        for t,task in enumerate(axes['tasks']):
            for m,method in enumerate(axes['methods']):
                for k,kind in enumerate(axes['kinds']):
                    mean,lo,hi=stats(q[t,m,k]);geometry.append({'architecture':architecture,'task':task,'method':method,'mask':kind,'normal_energy_fraction':mean,'ci95_lo':lo,'ci95_hi':hi})
                mean,lo,hi=stats(q[t,m,3]-q[t,m,0]);diagnostics.append({'architecture':architecture,'task':task,'method':method,'shift_minus_guided_normal_fraction':mean,'ci95_lo':lo,'ci95_hi':hi,'within_operational_5pp':bool(lo>=-.05 and hi<=.05),'mean_patient_absolute_fraction_mismatch':float(np.abs(q[t,m,3]-q[t,m,0]).mean()),'natural_shift_dose_ratio':float(ratio[t,m].mean()),'shift_mask_overlap':float(overlap[t,m].mean())})
                changes={}
                for k,kind in enumerate(axes['kinds'][1:],1):
                    raw=r[t,m,0,0]-r[t,m,k,0];con=r[t,m,0,1]-r[t,m,k,1];delta=con-raw;changes[k]=delta;mean,lo,hi=stats(delta);adjusted=stats(delta,alpha=.05/12);denom=stats(raw);bsraw=weights@raw;bsdelta=weights@delta;relative=100*bsdelta/bsraw;rel=np.quantile(relative,[.025,.975]);rela=np.quantile(relative,[.05/24,1-.05/24]);rows.append({'architecture':architecture,'task':task,'method':method,'comparator':kind,'raw_advantage':float(raw.mean()),'consistent_advantage':float(con.mean()),'advantage_change':mean,'change_ci95_lo':lo,'change_ci95_hi':hi,'change_adjusted_lo':adjusted[1],'change_adjusted_hi':adjusted[2],'relative_change_percent':float(100*delta.mean()/raw.mean()),'relative_ci95_lo':float(rel[0]),'relative_ci95_hi':float(rel[1]),'relative_adjusted_lo':float(rela[0]),'relative_adjusted_hi':float(rela[1]),'denominator_ci95_includes_zero':bool(denom[1]<=0<=denom[2]),'guided_raw_response':float(r[t,m,0,0].mean()),'guided_consistent_response':float(r[t,m,0,1].mean()),'random_raw_response':float(r[t,m,k,0].mean()),'random_consistent_response':float(r[t,m,k,1].mean())})
                mean,lo,hi=stats(changes[3]-changes[2]);difference.append({'architecture':architecture,'task':task,'method':method,'shift_minus_amplitude_advantage_change':mean,'ci95_lo':lo,'ci95_hi':hi,'absolute_loss_attenuation_fraction':float(1-changes[3].mean()/changes[2].mean())})
    for name,values in [('effects',rows),('geometry',geometry),('shift_diagnostics',diagnostics),('paired_comparator_difference',difference)]:pd.DataFrame(values).to_csv(OUT/f'{name}.csv',index=False)
    w=np.load(OUT/'wrapper_confirmation.npz');recomputed=w['recomputed'];fixed=w['frozen'];control={};gaps={'raw_lambda0':recomputed[0,0,0,0]-recomputed[0,2,0,0],'raw_lambda10':recomputed[0,0,1,0]-recomputed[0,2,1,0],'consistent_lambda10':recomputed[0,0,1,1]-recomputed[0,2,1,1]}
    for name,v in gaps.items():
        mean,lo,hi=stats(v,alpha=.05/3);control[name]={'gap':mean,'adjusted_ci':[lo,hi],'ci95':list(stats(v)[1:])}
    control['confirmed_predeclared_recomputed_reversal']=bool(control['raw_lambda0']['adjusted_ci'][1]<0 and control['raw_lambda10']['adjusted_ci'][0]>0 and control['consistent_lambda10']['adjusted_ci'][1]<0);control['raw_gap_change']=list(stats(gaps['raw_lambda10']-gaps['raw_lambda0']));control['fixed_consistent_max_response_difference']=float(np.abs(fixed[:,:,1,1]-fixed[:,:,0,1]).max());(OUT/'wrapper_summary.json').write_text(json.dumps(control,indent=2))
    effects=pd.DataFrame(rows);fig,axs=plt.subplots(1,2,figsize=(11,5),sharey=True);labels=[f'{t}: {m}' for t in axes['tasks'] for m in ['Saliency','Input×gradient','IG']]
    for ax,architecture in zip(axs,['original_cnn','residual_cnn']):
        for k,(kind,color,offset) in enumerate([('uniform_random','#6f7b86',-.24),('amplitude_matched_random','#bd5941',0),('time_shift_dose_matched','#247d72',.24)]):
            e=effects[(effects.architecture==architecture)&(effects.comparator==kind)];means=e.relative_change_percent.to_numpy();lo=e.relative_ci95_lo.to_numpy();hi=e.relative_ci95_hi.to_numpy();ax.errorbar(means,np.arange(6)+offset,xerr=np.array([means-lo,hi-means]),fmt='o',color=color,label=['Uniform','Amplitude + dose','Time shift + dose'][k],capsize=3)
        ax.axvline(0,color='black',lw=.8);ax.set_title('Original CNN' if architecture=='original_cnn' else 'Residual CNN');ax.set_yticks(np.arange(6),labels);ax.set_xlabel('Relative change in advantage (%)');ax.grid(axis='x',alpha=.2)
    axs[0].invert_yaxis();handles,legend_labels=axs[1].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='lower center',ncol=3,fontsize=9,frameon=False);fig.suptitle('Untouched 1,000-patient cohort: comparator choice matters');fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(OUT/'confirmation_effects.png',dpi=180);plt.close(fig)
    print(effects[effects.comparator.eq('time_shift_dose_matched')][['architecture','task','method','relative_change_percent','change_ci95_lo','change_ci95_hi','denominator_ci95_includes_zero']].to_string(index=False));print(pd.DataFrame(diagnostics).to_string(index=False));print(json.dumps(control,indent=2))
if __name__=='__main__':main()

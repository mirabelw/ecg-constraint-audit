"""Independent stored-array audit of cohort separation, dose and headline results."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'confirmation_results'
def main():
    old=pd.read_csv(ROOT/'results/manifest.csv');fresh=pd.read_csv(OUT/'manifest.csv');assert len(fresh)==1000 and fresh.patient_id.is_unique and not set(old.patient_id)&set(fresh.patient_id);assert fresh.strat_fold.between(1,8).all()
    sources=json.loads((OUT/'data_sources.json').read_text());assert sources['manifest_sha256']==hashlib.sha256((OUT/'manifest.csv').read_bytes()).hexdigest();assert sources['protocol_sha256']==hashlib.sha256((ROOT/'confirmation_protocol.json').read_bytes()).hexdigest()
    effects=pd.read_csv(OUT/'effects.csv');geometry=pd.read_csv(OUT/'geometry.csv');axes=json.loads((OUT/'axes.json').read_text());count=0
    for architecture in ['original_cnn','residual_cnn']:
        files=[np.load(OUT/f'{architecture}_seed_{s}.npz') for s in [17,29,43]]
        for f in files:
            assert np.array_equal(f['ecg_id'],fresh.ecg_id);assert np.isfinite(f['response']).all() and np.isfinite(f['normal_energy_fraction']).all();assert (f['offsets']>0).all() and (f['offsets']<20).all();assert all(len(set(row))==5 for row in f['offsets'])
            assert np.allclose(f['response'][:,:,3],f['shift_response'][:,:,:,0].mean(2),rtol=1e-6,atol=1e-7);assert np.allclose(f['response'][:,:,4],f['shift_response'][:,:,:,1].mean(2),rtol=1e-6,atol=1e-7)
        r=np.mean([f['response'] for f in files],axis=0).astype('float64');q=np.mean([f['normal_energy_fraction'] for f in files],axis=0)
        for row in effects[effects.architecture.eq(architecture)].itertuples():
            t=axes['tasks'].index(row.task);m=axes['methods'].index(row.method);k=axes['kinds'].index(row.comparator);raw=(r[t,m,0,0]-r[t,m,k,0]).mean();con=(r[t,m,0,1]-r[t,m,k,1]).mean();assert abs(raw-row.raw_advantage)<1e-10 and abs(con-row.consistent_advantage)<1e-10;assert abs(100*(con/raw-1)-row.relative_change_percent)<1e-8;count+=1
        for row in geometry[geometry.architecture.eq(architecture)].itertuples():
            t=axes['tasks'].index(row.task);m=axes['methods'].index(row.method);k=axes['kinds'].index(row.mask);assert abs(q[t,m,k].astype('float64').mean()-row.normal_energy_fraction)<1e-8
    wrapper=np.load(OUT/'wrapper_confirmation.npz');original=np.load(ROOT/'extension_results/positive_control_test.npz');assert np.array_equal(wrapper['w'],original['w']) and np.array_equal(wrapper['lambdas'],[0,10]);summary=json.loads((OUT/'wrapper_summary.json').read_text());r=wrapper['recomputed'];pairs={'raw_lambda0':r[0,0,0,0]-r[0,2,0,0],'raw_lambda10':r[0,0,1,0]-r[0,2,1,0],'consistent_lambda10':r[0,0,1,1]-r[0,2,1,1]}
    for name,v in pairs.items():assert abs(v.mean()-summary[name]['gap'])<1e-12
    checks=json.loads((OUT/'checks.json').read_text());assert all(v<2e-5 for k,v in checks.items() if k!='wrapper');assert checks['wrapper']['fixed_mask_consistent_max_response_difference']<1e-5
    # Recompute all primary adjusted intervals from independently indexed bootstrap draws.
    rng=np.random.default_rng(20261007);weights=rng.multinomial(1000,np.full(1000,.001),size=5000)/1000
    for architecture in ['original_cnn','residual_cnn']:
        r=np.mean([np.load(OUT/f'{architecture}_seed_{s}.npz')['response'] for s in [17,29,43]],axis=0).astype('float64')
        for row in effects[(effects.architecture.eq(architecture))&effects.comparator.eq('time_shift_dose_matched')].itertuples():
            t=axes['tasks'].index(row.task);m=axes['methods'].index(row.method);delta=r[t,m,0,1]-r[t,m,3,1]-r[t,m,0,0]+r[t,m,3,0];ci=np.quantile(weights@delta,[.05/24,1-.05/24]);assert np.allclose(ci,[row.change_adjusted_lo,row.change_adjusted_hi],rtol=1e-7,atol=1e-9)
    result={'status':'pass','patient_overlap':0,'reproduced_effect_rows':count,'primary_adjusted_intervals_reproduced':12,'wrapper_template_unchanged':True};(OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

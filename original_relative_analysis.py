import json
from pathlib import Path
import numpy as np
import pandas as pd
from train import load_data
from evaluate import make_mask
from geometry import project
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'extension_results'

def main():
    OUT.mkdir(exist_ok=True);axes=json.loads((ROOT/'results/evaluation_axes.json').read_text())
    response=np.stack([np.load(ROOT/'results'/f'evaluation_seed_{s}.npz')['response'] for s in axes['seeds']]);r=response.mean(0);rng=np.random.default_rng(811);n=r.shape[-1];indices=rng.integers(n,size=(5000,n),dtype=np.int32);rows=[]
    for t,task in enumerate(axes['tasks']):
        random_raw=r[t,3,0,1,0];random_cons=r[t,3,0,1,2];random_gain=random_cons-random_raw
        for m,method in enumerate(axes['methods'][:3]):
            a=r[t,m,0,1,0]-random_raw;c=r[t,m,0,1,2]-random_cons;delta=c-a;g=r[t,m,0,1,2]-r[t,m,0,1,0]
            bs=np.concatenate([((c[i].mean(1)-a[i].mean(1))/a[i].mean(1)) for i in np.array_split(indices,20)]);lo,hi=np.quantile(bs,[.05/6/2,1-.05/6/2])
            rows.append({'task':task,'method':method,'raw_advantage':float(a.mean()),'consistent_advantage':float(c.mean()),'relative_advantage_change':float(delta.mean()/a.mean()),'relative_adjusted_low':float(lo),'relative_adjusted_high':float(hi),'point_within_10pct':bool(abs(delta.mean()/a.mean())<.1),'adjusted_interval_within_10pct':bool(lo>-.1 and hi<.1),'guided_response_gain':float(g.mean()),'guided_response_relative_gain':float(g.mean()/r[t,m,0,1,0].mean()),'random_response_gain':float(random_gain.mean()),'random_response_relative_gain':float(random_gain.mean()/random_raw.mean()),'random_gain_fraction_of_advantage_loss':float(random_gain.mean()/(-delta.mean()))})
    pd.DataFrame(rows).to_csv(OUT/'original_relative_decomposition.csv',index=False)
    ordering=[]
    for task in range(2):
        for st,stratum in enumerate(axes['strata']):
            for b,budget in enumerate(axes['budgets']):
                raw_order=np.argsort(-r[task,:3,st,b,0].mean(-1));cons_order=np.argsort(-r[task,:3,st,b,2].mean(-1));ordering.append({'task':axes['tasks'][task],'stratum':stratum,'budget':budget,'same_point_order':bool(np.array_equal(raw_order,cons_order)),'raw_order':str(raw_order.tolist()),'consistent_order':str(cons_order.tolist())})
    pd.DataFrame(ordering).to_csv(OUT/'original_ordering_18cells.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()

"""Statistical resolution of method orderings on the confirmation cohort (descriptive; paper Section III-B).
For each architecture x task x operator, tests saliency vs input x gradient vs integrated gradients on guided response
(seeds averaged within patient; 5,000 patient-bootstrap replicates; 95% percentile intervals; no multiplicity adjustment)."""
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path("confirmation_results"); M = ["saliency", "input_gradient", "integrated_gradients"]; T = ["MI", "CD"]; OP = ["independent", "energy_matched"]
idx = np.random.default_rng(20261008).integers(0, 1000, size=(5000, 1000)); rows = []
for arch in ("original_cnn", "residual_cnn"):
    R = np.stack([np.load(OUT / f"{arch}_seed_{s}.npz")["response"] for s in (17, 29, 43)]).mean(0)   # task, method, kind, operator, patient
    for t in range(2):
        for a, b in ((1, 2), (1, 0), (2, 0)):
            for oi in range(2):
                d = R[t, a, 0, oi] - R[t, b, 0, oi]; lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
                rows.append(dict(architecture=arch, task=T[t], method_a=M[a], method_b=M[b], operator=OP[oi], diff_pp=100 * d.mean(), ci95_lo_pp=100 * lo, ci95_hi_pp=100 * hi, resolved=bool(lo > 0 or hi < 0)))
df = pd.DataFrame(rows); df.to_csv(OUT / "ranking_resolution.csv", index=False)
pair = df[(df.method_a == "input_gradient") & (df.method_b == "integrated_gradients")]
print(f"resolved comparisons: {int(df.resolved.sum())} of {len(df)}; input_gradient vs integrated_gradients: {int(pair.resolved.sum())} of {len(pair)}")
sal = df[df.method_b == "saliency"]; print("saliency lowest, resolved, in every comparison:", bool((sal.resolved & (sal.diff_pp > 0)).all()))

"""Supplementary analyses computed only from the saved patient-level arrays (no model or waveform access needed).

Run from the repository root:  python supplementary_analyses.py
These are descriptive and exploratory; none is part of the frozen confirmation protocol. Outputs:
  confirmation_results/patient_level_geometry_effect.csv    patient-level link between off-subspace energy gap and advantage change
  confirmation_results/comparator_level_geometry_effect.csv per-comparator summary of that link
  confirmation_results/mask_structure.csv                   guided-mask lead/time structure and guided-vs-shifted overlap
  results/budget_sensitivity.csv                            development cohort: advantage change at 5/10/20% budgets, limb stratum
  extension_results/projected_gradient_comparison.csv       development cohort: standard vs projected-gradient variants, per cell
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "confirmation_results"
METHODS = ["saliency", "input_gradient", "integrated_gradients"]
TASKS = ["MI", "CD"]
KINDS = ["guided", "uniform_random", "amplitude_matched_random", "time_shift_dose_matched", "time_shift_natural"]
COMPS = {"uniform_random": 1, "amplitude_matched_random": 2, "time_shift_dose_matched": 3, "time_shift_natural": 4}
B = 2000
idx = np.random.default_rng(20261010).integers(0, 1000, size=(B, 1000))


def load(arch):
    z = [np.load(OUT / f"{arch}_seed_{s}.npz") for s in (17, 29, 43)]
    return {k: np.stack([a[k] for a in z]) for k in ["response", "normal_energy_fraction", "window_scores", "offsets",
                                                     "shift_mask_overlap", "shift_natural_dose_ratio"]}


def fast_spearman(x, y):
    rx = x.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))



# ---------------------------------------------------------------- A. patient-level geometry/effect link
rows, cells = [], []
for arch in ("original_cnn", "residual_cnn"):
    d = load(arch)
    R = d["response"].mean(0)                    # task, method, kind, operator, patient (seeds averaged)
    N = d["normal_energy_fraction"].mean(0)      # task, method, kind, patient
    for t, task in enumerate(TASKS):
        for m, method in enumerate(METHODS):
            g_ind, g_con = R[t, m, 0, 0], R[t, m, 0, 1]
            for comp, k in COMPS.items():
                c_ind, c_con = R[t, m, k, 0], R[t, m, k, 1]
                d_adv = (g_con - c_con) - (g_ind - c_ind)          # patient-level change in advantage
                gap = N[t, m, k] - N[t, m, 0]                       # comparator minus guided raw off-subspace fraction
                rho = fast_spearman(gap, d_adv)
                # bootstrap on patient indices
                bs = np.array([fast_spearman(gap[i], d_adv[i]) for i in idx])
                lo, hi = np.percentile(bs, [2.5, 97.5])
                slope = float(np.polyfit(gap * 100, d_adv * 100, 1)[0])  # response points per percentage point of energy gap
                bs_s = np.array([np.polyfit(gap[i] * 100, d_adv[i] * 100, 1)[0] for i in idx[:500]])
                slo, shi = np.percentile(bs_s, [2.5, 97.5])
                adv_ind = (g_ind - c_ind).mean()
                rows.append(dict(architecture=arch, task=task, method=method, comparator=comp,
                                 mean_energy_gap_pp=100 * gap.mean(), sd_energy_gap_pp=100 * gap.std(),
                                 mean_adv_change_points=100 * d_adv.mean(), rel_change_percent=100 * d_adv.mean() / adv_ind,
                                 spearman_rho=rho, rho_lo=lo, rho_hi=hi,
                                 slope_points_per_pp=slope, slope_lo=slo, slope_hi=shi))
pl = pd.DataFrame(rows)
pl.to_csv(OUT / "patient_level_geometry_effect.csv", index=False)

# ---------------------------------------------------------------- B. comparator-level (cell) summary
cl = pl.groupby("comparator").agg(cells=("method", "size"), gap_min=("mean_energy_gap_pp", "min"), gap_max=("mean_energy_gap_pp", "max"),
                                  rel_min=("rel_change_percent", "min"), rel_max=("rel_change_percent", "max"),
                                  rho_median=("spearman_rho", "median"), rho_min=("spearman_rho", "min"), rho_max=("spearman_rho", "max"),
                                  cells_ci_excl0=("rho_lo", lambda s: int(((s > 0) | (pl.loc[s.index, "rho_hi"] < 0)).sum())),
                                  slope_median=("slope_points_per_pp", "median")).reset_index()
cl.to_csv(OUT / "comparator_level_geometry_effect.csv", index=False)
dm = pl[pl.comparator.isin(["amplitude_matched_random", "time_shift_dose_matched"])]
rho_dm = spearmanr(dm.mean_energy_gap_pp, dm.rel_change_percent)
rho_all = spearmanr(pl[pl.comparator != "time_shift_natural"].mean_energy_gap_pp, pl[pl.comparator != "time_shift_natural"].rel_change_percent)

# ---------------------------------------------------------------- C. guided-mask structure and shift overlap
srows = []
rng = np.random.default_rng(7)
sim = np.zeros((20000, 6, 20), dtype=bool)
for i in range(20000):
    sim[i].flat[rng.choice(120, 12, replace=False)] = True
def structure(mask):                       # mask: (n, 6, 20) bool
    per_time = mask.sum(1)                 # leads selected in each window
    shared = ((per_time[:, None, :] >= 2) & mask).sum((1, 2)) / mask.sum((1, 2))   # fraction of selected cells sharing a window with another lead
    n_windows = (per_time > 0).sum(1)      # distinct windows used by the mask
    return shared, n_windows
sh_u, nw_u = structure(sim)
for arch in ("original_cnn", "residual_cnn"):
    d = load(arch)
    for t, task in enumerate(TASKS):
        for m, method in enumerate(METHODS):
            shared_l, nw_l, edge_l, lead_l, ov_l, ovrec_l, ov_ind = [], [], [], [], [], [], []
            for s in range(3):
                scores = d["window_scores"][s, t, m][:, :6, :]            # limb leads, 20 windows
                flat = scores.reshape(len(scores), -1)
                sel = np.argsort(-flat, axis=1, kind="stable")[:, :12]
                mask = np.zeros((len(scores), 120), dtype=bool)
                np.put_along_axis(mask, sel, True, axis=1); mask = mask.reshape(-1, 6, 20)
                sh, nw = structure(mask)
                shared_l.append(sh.mean()); nw_l.append(nw.mean())
                edge_l.append(mask[:, :, [0, 19]].sum((1, 2)).mean() / 12)    # fraction of cells in first/last window
                lead_l.append(mask.sum((0, 2)) / (12 * len(mask)))
                # reconstruct shift overlap exactly from offsets; compare with the saved diagnostic
                off = d["offsets"][s]
                ov = np.stack([(mask & np.take_along_axis(mask, ((np.arange(20)[None, :] - off[:, j][:, None]) % 20)[:, None, :].repeat(6, 1), axis=2)).sum((1, 2)) / 12 for j in range(5)], 1)
                saved = d["shift_mask_overlap"][s, t, m].T                     # (n, 5)
                assert np.allclose(ov, saved, atol=1e-6), "overlap reconstruction mismatch"
                ov_l.append(ov.mean(1)); ov_ind.append(ov.reshape(-1))
            ov_all = np.concatenate(ov_l); ov_i = np.concatenate(ov_ind)
            lead_share = np.mean(lead_l, 0)
            srows.append(dict(architecture=arch, task=task, method=method,
                              share_cells_sharing_window=np.mean(shared_l), distinct_windows=np.mean(nw_l), edge_window_fraction=np.mean(edge_l),
                              lead_share_max=lead_share.max(), lead_share_min=lead_share.min(),
                              shift_overlap_mean=ov_all.mean(), indiv_shift_overlap_p95=np.percentile(ov_i, 95), indiv_shift_overlap_zero=float((ov_i == 0).mean()),
                              avg_shift_overlap_p95=np.percentile(ov_all, 95), avg_shift_overlap_zero=float((ov_all == 0).mean()),
                              natural_dose_ratio_mean=float(d["shift_natural_dose_ratio"][:, t, m].mean())))
ms = pd.DataFrame(srows); ms.to_csv(OUT / "mask_structure.csv", index=False)
random_ref = dict(share_cells_sharing_window=float(sh_u.mean()), distinct_windows=float(nw_u.mean()), edge_window_fraction=2 / 20, overlap_expected=12 / 120 * 1.0)

# ---------------------------------------------------------------- D. development budget sensitivity (limb stratum, compact CNN, uniform random comparator)
r = pd.read_csv(ROOT / "results" / "response_summary.csv")
r = r[r.stratum == "limb"]
brow = []
for task in TASKS:
    for method in METHODS:
        for budget in (0.05, 0.1, 0.2):
            def val(meth, op):
                return float(r[(r.task == task) & (r.method == meth) & (r.budget == budget) & (r.operator == op)].response.iloc[0])
            a_ind = val(method, "independent_zero") - val("random", "independent_zero")
            a_con = val(method, "energy_matched") - val("random", "energy_matched")
            brow.append(dict(task=task, method=method, budget=budget, advantage_independent_points=100 * a_ind,
                             advantage_consistent_points=100 * a_con, change_points=100 * (a_con - a_ind), relative_change_percent=100 * (a_con - a_ind) / a_ind))
bs_df = pd.DataFrame(brow); bs_df.to_csv(ROOT / "results" / "budget_sensitivity.csv", index=False)

# ---------------------------------------------------------------- E. projected-gradient variants (development cohort, both architectures)
c = pd.read_csv(ROOT / "extension_results" / "comparator_contrasts.csv"); c["rel"] = 100 * c.relative_change
prow = []
for arch in c.architecture.unique():
    for task in TASKS:
        for base in METHODS:
            for comp in ("uniform_random", "amplitude_matched_random"):
                pick = lambda meth: float(c[(c.architecture == arch) & (c.task == task) & (c.method == meth) & (c.comparator == comp)].rel.iloc[0])
                std, proj = pick(base), pick("tangent_" + base)
                prow.append(dict(architecture=arch, task=task, method=base, comparator=comp, standard_relative_change_percent=std,
                                 projected_relative_change_percent=proj, projected_loss_smaller=bool(proj > std)))
pg = pd.DataFrame(prow); pg.to_csv(ROOT / "extension_results" / "projected_gradient_comparison.csv", index=False)

# ---------------------------------------------------------------- report
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("== A/B patient-level and comparator-level ==")
print(cl.round(3).to_string())
print(f"cell-level Spearman, dose-matched comparators only (amp-matched + time-shift, n={len(dm)}): rho={rho_dm.statistic:.3f}")
print(f"cell-level Spearman, 3 comparators incl. uniform (n={(pl.comparator != 'time_shift_natural').sum()}): rho={rho_all.statistic:.3f}")
print("\n== C mask structure ==")
print(ms.round(3).to_string()); print("random reference:", {k: round(v, 3) for k, v in random_ref.items()})
print("\n== D budget sensitivity ==")
print(bs_df.round(3).to_string())
print("\n== E projected-gradient (input_gradient and integrated_gradients only) ==")
sub = pg[pg.method != "saliency"]
print(sub.groupby("comparator").agg(smaller=("projected_loss_smaller", "sum"), n=("projected_loss_smaller", "size"), std_min=("standard_relative_change_percent", "min"), std_max=("standard_relative_change_percent", "max"), proj_min=("projected_relative_change_percent", "min"), proj_max=("projected_relative_change_percent", "max")).round(1).to_string())

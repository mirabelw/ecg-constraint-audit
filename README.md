# ECG Constraint Audit

Code and derived results for the paper *Measurement Constraints and Comparator
Choice in Perturbation-Based Evaluation of ECG Explanations* (under review).

The study trains two small convolutional networks (three seeds each) to detect
myocardial infarction and conduction disturbance on the public PTB-XL 12-lead
ECG dataset. It then measures how perturbation-based explanation scores change
when deletions respect the known limb-lead relationships (Einthoven's law and
Goldberger's equations), and how much those conclusions depend on the random
comparator used to score an explanation's advantage. It is a benchmark-design
study, not a clinical tool.

## Data

The scripts download PTB-XL 1.0.3 automatically from the public PhysioNet S3
mirror on first run (no account needed) and cache it outside the repository:
<https://physionet.org/content/ptb-xl/1.0.3/> (DOI 10.13026/kfzx-aw45, CC BY
4.0). Only the fixed patient samples are fetched: about 132 MB of waveforms for
the 5,500-record development sample, plus 1,000 confirmation records. Raw ECG
waveforms are not included here; the manifests in `results/` and
`confirmation_results/` fix each sample exactly.

## Setup

Python 3.12, CPU only.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Other library versions or hardware may give slightly different numbers.

## Reproduce the paper

Run from the repository root, in this order. Data caches default to
`../data_cache`, `../confirmation_cache` and `../metadata_cache`. The pipelines
regenerate the result folders and overwrite the supplied files, so copy those
first if you want to compare against the supplied run.

| Paper item | Command | Output |
|---|---|---|
| Measurement-subspace and operator checks (Section II-B, II-E) | `python geometry.py` | printed checks (also saved to `results/geometry_checks.json` during training) |
| Development cohort: compact CNN training, original audit, 18-cell ordering, preprocessing corrections, target-matched dose | `python run_pipeline.py --cache ../data_cache` | `results/`, `checkpoints/seed_*.pt` |
| Development extension: residual CNN, seven attribution variants, amplitude-matched comparator, off-subspace energy, mean replacement, development controls (including failed ones) | `python run_extension.py --cache ../data_cache` | `extension_results/`, `checkpoints/residual_seed_*.pt` |
| Confirmation cohort: Fig. 2, Tables I–III, Sections III-B to III-D | `python run_confirmation.py` | `confirmation_results/` |
| Independent check of the confirmation numbers and cohort separation | `python verify_confirmation.py` | `confirmation_results/verification.json` |
| Statistical resolution of method orderings (Section III-B) | `python ranking_resolution.py` | `confirmation_results/ranking_resolution.csv` |
| Supplementary descriptive analyses from saved outputs: patient-level geometry link, mask structure, development budget and projected-gradient summaries (Section III-D, supplement) | `python supplementary_analyses.py` | `confirmation_results/patient_level_geometry_effect.csv`, `confirmation_results/comparator_level_geometry_effect.csv`, `confirmation_results/mask_structure.csv`, `results/budget_sensitivity.csv`, `extension_results/projected_gradient_comparison.csv` |
| Fig. 1 (illustrative masks) and Fig. 2 | `python make_paper_figures.py` | `paper_figures/` |

`run_confirmation.py` uses the six supplied model states, so it can be run
without retraining. The development pipelines train new models and rerun every
step.

## Files

- `geometry.py`: limb-lead acquisition matrix, projector, independent,
  energy-matched and target-matched operators, diagnostics, and unit checks
  (all 64 limb-mask patterns, chest identity, an off-subspace-only predictor).
- `download_data.py`, `download_confirmation.py`: label-independent cohort
  selection and record download; the confirmation download excludes every
  development patient.
- `model.py`: compact CNN and residual CNN.
- `train.py`, `train_extension.py`: training with the validation AUROC gate
  for the compact and residual CNNs.
- `evaluate.py`, `summarize.py`: development audit across budgets, strata and
  operators, with paired patient-bootstrap intervals.
- `check_ig.py`, `baseline_sensitivity.py`: integrated-gradients convergence
  and data-quality sensitivity checks (validation only / disclosed).
- `extension_methods.py`, `evaluate_extension.py`, `summarize_extension.py`,
  `original_relative_analysis.py`: extension variants (SmoothGrad, tangent-projected
  attributions), amplitude/dose-matched comparator, relative-effect
  decomposition and the 18-cell ordering re-check.
- `positive_control.py`, `fitted_positive_control.py`,
  `summarize_control_grid.py`: off-subspace model construction on the
  development cohort, including both failed controls and the post hoc grid
  example.
- `evaluate_confirmation.py`, `summarize_confirmation.py`,
  `verify_confirmation.py`: frozen confirmation with time-shift comparators,
  geometry diagnostics, wrapper confirmation and an independent audit.
- `ranking_resolution.py`: descriptive patient-bootstrap intervals for the pairwise method
  orderings on the confirmation cohort.
- `supplementary_analyses.py`: descriptive analyses computed from saved outputs only
  (patient-level geometry link, guided-mask structure, development budget and
  projected-gradient summaries); needs no waveforms or models.
- `protocol.json`, `extension_protocol.json`,
  `control_construction_protocol.json`, `confirmation_protocol.json`: locally
  frozen analysis plans, recorded by hash in the corresponding results.
- `checkpoints/`: the six trained models used in the paper.
- `make_paper_figures.py`: Fig. 1 and Fig. 2 from `confirmation_results/`.

## Notes on interpretation

- Development analyses (`results/`, `extension_results/`) reuse one test cohort
  and are exploratory. Several choices, including the amplitude-matched and
  time-shift comparators, followed earlier results. The confirmation cohort
  (`confirmation_results/`) uses 1,000 previously unused patients under a
  protocol frozen locally before selection; it was not externally registered.
- Two earlier off-subspace controls failed and are retained. The confirmed
  λ = 10 example was selected post hoc on development data.
- Models are trained and evaluated on centered, limb-projected signals scaled
  by one training-set RMS value, not on an unmodified vendor input pipeline.
- Probability response measures model sensitivity, not ground-truth
  explanation quality. Circularly shifted windows are a structured control,
  not physiological counterfactuals.
- The comparators differ in several ways besides off-subspace energy, so the
  results do not give a causal decomposition by geometry alone.
- All intervals condition on the fitted models (and, for the time shift, the
  sampled offsets). Confirmation-cohort labels come from folds 1–8, which are
  less extensively validated than folds 9–10; labels are used only for
  descriptive AUROC.
- `SHA256SUMS.json` lists file hashes for this repository.

## License

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0, from PhysioNet (PTB-XL 1.0.3).

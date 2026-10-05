"""Build the paper's figures and print table values from confirmation_results/ (Fig. 1 schematic and Fig. 2 of the paper)."""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("paper_figures", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42})
e = pd.read_csv("confirmation_results/effects.csv"); g = pd.read_csv("confirmation_results/geometry.csv")
ARCH = {"original_cnn": "Compact CNN", "residual_cnn": "Residual CNN"}; METH = {"saliency": "Sal.", "input_gradient": "I$\\times$G", "integrated_gradients": "IG"}
COMP = [("uniform_random", "Uniform random", "o", "#8c8c8c"), ("amplitude_matched_random", "Amplitude/dose-matched random", "s", "#c0504d"), ("time_shift_dose_matched", "Time shift, dose-matched", "D", "#1f5f8b")]

# ---------- Fig. 2: relative change in advantage, fresh cohort ----------
fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.95), sharey=True)
for ax, (arch, task) in zip(axes, [(a, t) for a in ARCH for t in ("MI", "CD")]):
    for j, m in enumerate(METH):
        for k, (c, lab, mk, col) in enumerate(COMP):
            r = e[(e.architecture == arch) & (e.task == task) & (e.method == m) & (e.comparator == c)].iloc[0]
            x = j + (k - 1) * 0.24
            if c == "time_shift_dose_matched": lo, hi = r.relative_adjusted_lo, r.relative_adjusted_hi
            else: lo, hi = r.relative_ci95_lo, r.relative_ci95_hi
            ax.errorbar(x, r.relative_change_percent, yerr=[[r.relative_change_percent - lo], [hi - r.relative_change_percent]], fmt=mk, ms=3.2, color=col, elinewidth=0.8, capsize=1.5, label=lab if j == 0 else None)
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels(METH.values(), rotation=0, fontsize=6.5)
    ax.set_title(f"{ARCH[arch]}, {task}", fontsize=7.5); ax.grid(axis="y", lw=0.3, alpha=0.5); ax.set_xlim(-0.5, 2.5)
axes[0].set_ylabel("Relative change in\nadvantage (%)")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=7)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig("paper_figures/fig_comparators.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig. 1: schematic (illustrative masks; exactly 12 of 120 limb lead-window cells each) ----------
rng = np.random.default_rng(3); leads = ["I", "II", "III", "aVR", "aVL", "aVF"]; W = 20
guided = np.zeros((6, W)); guided[[0, 1, 2, 5], 6] = 1; guided[[0, 1, 5], 7] = 1; guided[[1, 2], 14] = 1; guided[[0, 1, 5], 15] = 1
uniform = np.zeros((6, W)); uniform.flat[rng.choice(6 * W, 12, replace=False)] = 1
cand = [l * W + w for l in range(6) for w in (5, 6, 7, 8, 13, 14, 15, 16)]          # high-amplitude windows (illustrative)
amp = np.zeros((6, W)); amp.flat[rng.choice(cand, 12, replace=False)] = 1
shift = np.roll(guided, 5, axis=1)
assert all(m.sum() == 12 for m in (guided, uniform, amp, shift))
fig, axs = plt.subplots(2, 2, figsize=(3.45, 1.9), sharex=True, sharey=True)
for ax, (mask, title) in zip(axs.flat, [(guided, "Guided (explanation)"), (uniform, "Uniform random"), (amp, "Amplitude-matched random"), (shift, "Shared time shift")]):
    ax.imshow(mask, cmap="Greys", vmin=0, vmax=1.4, aspect="auto"); ax.set_title(title, fontsize=7, pad=2)
    ax.set_yticks(range(6)); ax.set_yticklabels(leads, fontsize=5.5); ax.set_xticks([-0.5, 9.5, 19.5]); ax.set_xticklabels(["0", "5", "10 s"], fontsize=5.5)
    ax.set_xticks(np.arange(-.5, W, 1), minor=True); ax.set_yticks(np.arange(-.5, 6, 1), minor=True); ax.grid(which="minor", color="w", lw=0.4); ax.tick_params(which="minor", length=0)
fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6); fig.savefig("paper_figures/fig_masks.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- values for tables ----------
print("normal-energy fraction ranges (min-max over 12 arch x task x method cells):")
for mask in ["guided", "uniform_random", "amplitude_matched_random", "time_shift_dose_matched"]:
    s = g[g["mask"] == mask].normal_energy_fraction; print(f"  {mask:26s} {100*s.min():.1f}-{100*s.max():.1f}%")
print("relative change ranges:"); print(e.groupby("comparator").relative_change_percent.agg(["min", "max"]).round(1).to_string())
print("\nprimary family (time shift, dose matched): relative change [Bonferroni 99.58% interval]")
p = e[e.comparator == "time_shift_dose_matched"]
for _, r in p.iterrows(): print(f"  {r.architecture} {r.task} {r.method:22s} {r.relative_change_percent:+5.1f} [{r.relative_adjusted_lo:+5.1f}, {r.relative_adjusted_hi:+5.1f}]  abs {100*r.advantage_change:+.2f} pp [{100*r.change_adjusted_lo:+.2f},{100*r.change_adjusted_hi:+.2f}]")
print("  resolved positive:", int((p.relative_adjusted_lo > 0).sum()), "| resolved negative:", int((p.relative_adjusted_hi < 0).sum()))
# fresh-cohort guided-response ordering (independent vs energy-matched)
for arch in ARCH:
    for t in ("MI", "CD"):
        q = e[(e.architecture == arch) & (e.task == t) & (e.comparator == "uniform_random")].set_index("method")
        print(f"  ordering {arch} {t}: independent {list(q.guided_raw_response.sort_values(ascending=False).index)} | consistent {list(q.guided_consistent_response.sort_values(ascending=False).index)}")
print("wrapper:", json.dumps({k: v for k, v in json.load(open("confirmation_results/wrapper_summary.json")).items() if k in ("raw_lambda0", "raw_lambda10", "consistent_lambda10")}))

from pathlib import Path
import csv
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "hce_boost_results"
NEURAL_RESULTS = ROOT / "neural_five_seed_results"
REVISION_RESULTS = ROOT / "revision_results"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

COLORS = {
    "LSTM-AE": "#8B9BAE",
    "CNN-BiLSTM": "#6F91C2",
    "VLSTM": "#9382B5",
    "DNN": "#C9817B",
    "RNN": "#63A49B",
    "XGBoost": "#4F78A8",
    "Scratch MLP": "#CFA23C",
    "HCE-Boost": "#08766E",
    "global": "#95A8BA",
    "gain": "#167D75",
}


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


EXPECTED_SEEDS = {11, 22, 33, 44, 55}
METHOD_ORDER = ["LSTM-AE", "CNN-BiLSTM", "VLSTM", "DNN", "RNN",
                "XGBoost", "Scratch MLP", "HCE-Boost"]
DATASET_FILES = {
    "NSL-KDD": ("nsl", "nsl_xgb.json", "nsl_scratch.json", "scratch_cube"),
    "CIC-IDS2017": ("cic", "cic_xgb.json", "cic_scratch.json", "mlp_scratch_cube"),
    "Edge-IIoTset": ("edge", "edge_xgb.json", "edge_scratch.json", "mlp_scratch_cube"),
}


def checked_rows(rows, label):
    seeds = {int(row["seed"]) for row in rows}
    if len(rows) != 5 or seeds != EXPECTED_SEEDS:
        raise RuntimeError(f"{label}: expected five matched seeds, found {sorted(seeds)}")
    return rows


def mean_sd(rows, key="f1"):
    values = np.asarray([float(row[key]) for row in rows])
    return float(values.mean()), float(values.std(ddof=1)), "11;22;33;44;55"


def load_datasets():
    output = {}
    for dataset, (stem, xgb_file, scratch_file, scratch_key) in DATASET_FILES.items():
        neural = json.loads((NEURAL_RESULTS / f"{stem}_neural_baselines_five_seed.json").read_text(encoding="utf-8"))
        vals = {}
        for method in METHOD_ORDER[:5]:
            rows = checked_rows(neural["models"][method]["individual"], f"{dataset}/{method}")
            vals[method] = mean_sd(rows)

        xgb = json.loads((REVISION_RESULTS / xgb_file).read_text(encoding="utf-8"))
        xgb_rows = checked_rows(xgb["runs"]["xgboost_cube"]["individual"], f"{dataset}/XGBoost")
        vals["XGBoost"] = mean_sd(xgb_rows)

        scratch = json.loads((REVISION_RESULTS / scratch_file).read_text(encoding="utf-8"))
        scratch_rows = checked_rows(scratch["runs"][scratch_key]["individual"], f"{dataset}/Scratch MLP")
        vals["Scratch MLP"] = mean_sd(scratch_rows)

        hce = json.loads((RESULTS / f"{stem}_hce_boost_detailed.json").read_text(encoding="utf-8"))
        hce_rows = checked_rows(hce["individual"], f"{dataset}/HCE-Boost")
        vals["HCE-Boost"] = mean_sd([{"f1": row["test"]["f1"]} | {"seed": row["seed"]} for row in hce_rows])

        if set(vals) != set(METHOD_ORDER):
            raise RuntimeError(f"{dataset}: incomplete method set")
        output[dataset] = vals
    return output


datasets = load_datasets()

# Machine-readable source data for the three quantitative panels.
with (OUT / "Fig2_macro_f1_source_data.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["dataset", "method", "macro_f1_percent", "sample_sd_percent", "seeds"])
    for ds, vals in datasets.items():
        for name, (score, sd, status) in vals.items():
            writer.writerow([ds, name, f"{score:.5f}", "" if sd is None else f"{sd:.5f}", status])

fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), gridspec_kw={"wspace": 0.52})
for ax, (ds, vals) in zip(axes, datasets.items()):
    items = sorted(vals.items(), key=lambda x: x[1][0])
    names = [x[0] for x in items]
    scores = [x[1][0] for x in items]
    errors = [0 if x[1][1] is None else x[1][1] for x in items]
    labels = names
    colors = [COLORS[n] for n in names]
    bars = ax.barh(
        range(len(names)), scores, xerr=errors, color=colors, height=0.66,
        edgecolor="white", linewidth=0.45,
        error_kw={"ecolor": "#26343F", "elinewidth": 0.65, "capsize": 1.8, "capthick": 0.65},
    )
    ax.set_yticks(range(len(names)), labels)
    ax.set_title(ds, fontsize=8, fontweight="bold", pad=6)
    xmin = 50 if ds == "NSL-KDD" else (77 if ds == "CIC-IDS2017" else 73)
    xmax = max(score + error for score, error in zip(scores, errors)) + 5.0
    ax.set_xlim(xmin, max(96.2, xmax))
    ax.set_xlabel("Macro F1 (%)")
    ax.grid(axis="x", color="#E8ECEF", linewidth=0.6)
    ax.set_axisbelow(True)
    label_x = max(score + error for score, error in zip(scores, errors)) + 0.7
    for b, value, name in zip(bars, scores, names):
        ax.text(label_x, b.get_y() + b.get_height()/2, f"{value:.2f}",
                va="center", fontsize=6.2,
                fontweight="bold" if name == "HCE-Boost" else "normal")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].text(-0.28, 1.08, "a", transform=axes[0].transAxes, fontsize=9, fontweight="bold")
fig.subplots_adjust(bottom=0.20)
fig.text(0.5, 0.045,
         "All bars show mean test macro F1; error bars show sample SD over seeds 11, 22, 33, 44, and 55.",
         ha="center", fontsize=6.2, color="#485563")
save(fig, "Fig2_macro_f1_comparison")


files = {
    "NSL-KDD": "nsl_hce_boost_detailed.json",
    "CIC-IDS2017": "cic_hce_boost_detailed.json",
    "Edge-IIoTset": "edge_hce_boost_detailed.json",
}
global_scores, final_scores = {}, {}
for ds, fn in files.items():
    obj = json.loads((RESULTS / fn).read_text(encoding="utf-8"))
    global_scores[ds] = np.array([r["global_test"]["f1"] for r in obj["individual"]])
    final_scores[ds] = np.array([r["test"]["f1"] for r in obj["individual"]])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.42})
ax = axes[0]
x = np.arange(3)
w = 0.34
gm = np.array([global_scores[d].mean() for d in files])
gs = np.array([global_scores[d].std(ddof=1) for d in files])
fm = np.array([final_scores[d].mean() for d in files])
fs = np.array([final_scores[d].std(ddof=1) for d in files])
ax.bar(x-w/2, gm, w, yerr=gs, capsize=2.2, color=COLORS["global"], label="Global ensemble")
ax.bar(x+w/2, fm, w, yerr=fs, capsize=2.2, color=COLORS["HCE-Boost"], label="HCE-Boost")
ax.set_xticks(x, ["NSL-KDD", "CIC-IDS2017", "Edge-IIoTset"])
ax.set_ylabel("Macro F1 (%)")
ax.set_ylim(59, 95)
ax.grid(axis="y", color="#E8ECEF", linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(loc="upper left")
for i in range(3):
    ax.text(i, max(gm[i]+gs[i], fm[i]+fs[i])+0.8, f"Δ {fm[i]-gm[i]:+.2f}", ha="center", fontsize=6.5)
ax.set_title("Specialist contribution", fontsize=8, fontweight="bold")
ax.text(-0.15, 1.08, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")

ax = axes[1]
for i, ds in enumerate(files):
    delta = final_scores[ds] - global_scores[ds]
    jitter = np.linspace(-0.05, 0.05, len(delta))
    ax.scatter(np.full(len(delta), i)+jitter, delta, s=23,
               color=COLORS["gain"] if np.any(np.abs(delta) > 1e-9) else COLORS["global"],
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.hlines(delta.mean(), i-0.2, i+0.2, colors="#24323F", linewidth=1.4)
ax.axhline(0, color="#66737E", linewidth=0.7, linestyle="--")
ax.set_xticks(range(3), ["NSL-KDD", "CIC-IDS2017", "Edge-IIoTset"], rotation=18, ha="right")
ax.set_ylabel("Seed-wise specialist gain (pp)")
ax.set_title("Matched-seed effects", fontsize=8, fontweight="bold")
ax.grid(axis="y", color="#E8ECEF", linewidth=0.6)
ax.set_axisbelow(True)
ax.text(-0.18, 1.08, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")
fig.subplots_adjust(bottom=0.18)
save(fig, "Fig3_specialist_ablation")

print(OUT)

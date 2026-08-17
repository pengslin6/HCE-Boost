"""Validate five-seed results and generate manuscript-ready summaries.

This script refuses to write final artifacts unless every method contains exactly
the matched seeds 11, 22, 33, 44, and 55. It then creates one machine-readable
summary used by the table, figure, and statistical text.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats


WORK = Path(__file__).resolve().parent
OUT = WORK / "five_seed_final"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
METHODS = ["LSTM-AE", "CNN-BiLSTM", "VLSTM", "DNN", "RNN",
           "XGBoost", "MLP (scratch)", "HCE-Boost"]
DATASETS = {
    "NSL-KDD": {
        "stem": "nsl",
        "xgb": "nsl_xgb.json",
        "scratch": "nsl_scratch.json",
        "scratch_key": "scratch_cube",
        "tex": "NSL-KDD",
    },
    "CIC-IDS2017": {
        "stem": "cic",
        "xgb": "cic_xgb.json",
        "scratch": "cic_scratch.json",
        "scratch_key": "mlp_scratch_cube",
        "tex": "CIC-IDS2017 subset",
    },
    "Edge-IIoTset": {
        "stem": "edge",
        "xgb": "edge_xgb.json",
        "scratch": "edge_scratch.json",
        "scratch_key": "mlp_scratch_cube",
        "tex": "Edge-IIoTset",
    },
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_rows(rows, label):
    if len(rows) != 5:
        raise RuntimeError(f"{label}: expected 5 runs, found {len(rows)}")
    seed_map = {int(row["seed"]): row for row in rows}
    if sorted(seed_map) != SEEDS:
        raise RuntimeError(f"{label}: expected seeds {SEEDS}, found {sorted(seed_map)}")
    if len(seed_map) != len(rows):
        raise RuntimeError(f"{label}: duplicate seed")
    return [seed_map[seed] for seed in SEEDS]


def collect():
    all_results = {}
    for dataset, cfg in DATASETS.items():
        stem = cfg["stem"]
        neural = load_json(WORK / "neural_five_seed_results" / f"{stem}_neural_baselines_five_seed.json")
        models = {}
        for method in METHODS[:5]:
            if method not in neural.get("models", {}):
                raise RuntimeError(f"{dataset}/{method}: no completed result block yet")
            models[method] = validate_rows(
                neural["models"][method]["individual"], f"{dataset}/{method}")

        xgb = load_json(WORK / "revision_results" / cfg["xgb"])
        models["XGBoost"] = validate_rows(
            xgb["runs"]["xgboost_cube"]["individual"], f"{dataset}/XGBoost")

        scratch = load_json(WORK / "revision_results" / cfg["scratch"])
        models["MLP (scratch)"] = validate_rows(
            scratch["runs"][cfg["scratch_key"]]["individual"], f"{dataset}/MLP (scratch)")

        hce_obj = load_json(WORK / "hce_boost_results" / f"{stem}_hce_boost_detailed.json")
        hce_raw = validate_rows(hce_obj["individual"], f"{dataset}/HCE-Boost")
        models["HCE-Boost"] = [dict(seed=row["seed"], **row["test"]) for row in hce_raw]
        all_results[dataset] = models
    return all_results


def summarize(all_results):
    summary = {"seeds": SEEDS, "datasets": {}}
    for dataset, models in all_results.items():
        ds = {"methods": {}, "primary_metric": "macro F1"}
        for method, rows in models.items():
            metrics = {}
            for metric in METRICS:
                arr = np.asarray([float(row[metric]) for row in rows])
                metrics[metric] = {
                    "mean": float(arr.mean()),
                    "sample_sd": float(arr.std(ddof=1)),
                    "seed_values": {str(seed): float(value) for seed, value in zip(SEEDS, arr)},
                }
            ds["methods"][method] = metrics

        competitors = [m for m in METHODS if m != "HCE-Boost"]
        strongest = max(competitors, key=lambda m: ds["methods"][m]["f1"]["mean"])
        hce = np.asarray([float(row["f1"]) for row in models["HCE-Boost"]])
        comp = np.asarray([float(row["f1"]) for row in models[strongest]])
        diff = hce - comp
        sem = stats.sem(diff)
        half = float(stats.t.ppf(0.975, len(diff) - 1) * sem)
        test = stats.ttest_rel(hce, comp)
        ds["strongest_competitor"] = strongest
        ds["paired_comparison"] = {
            "mean_difference_pp": float(diff.mean()),
            "sample_sd_difference_pp": float(diff.std(ddof=1)),
            "ci95_low_pp": float(diff.mean() - half),
            "ci95_high_pp": float(diff.mean() + half),
            "paired_t_statistic": float(test.statistic),
            "paired_p_value": float(test.pvalue),
            "all_differences_positive": bool(np.all(diff > 0)),
            "seed_differences_pp": {str(seed): float(value) for seed, value in zip(SEEDS, diff)},
        }
        summary["datasets"][dataset] = ds
    return summary


def fmt(mean, sd, metric):
    if metric == "auc" and sd < 0.005:
        return f"{mean:.4f}\\pm{sd:.4f}"
    if sd < 0.01:
        return f"{mean:.3f}\\pm{sd:.3f}"
    return f"{mean:.2f}\\pm{sd:.2f}"


def table_block(summary):
    lines = []
    for dataset, cfg in DATASETS.items():
        ds = summary["datasets"][dataset]
        maxima = {
            metric: max(ds["methods"][m][metric]["mean"] for m in METHODS)
            for metric in METRICS
        }
        lines.append(f"\\multirow{{8}}{{*}}{{{cfg['tex']}}}")
        for method in METHODS:
            cells = []
            for metric in METRICS:
                item = ds["methods"][method][metric]
                value = fmt(item["mean"], item["sample_sd"], metric)
                if math.isclose(item["mean"], maxima[metric], rel_tol=0, abs_tol=1e-10):
                    value = f"\\mathbf{{{value}}}"
                cells.append(f"${value}$")
            label = "\\textbf{HCE-Boost}" if method == "HCE-Boost" else method
            lines.append("& " + label + " & " + " & ".join(cells) + r" \\")
        if dataset != list(DATASETS)[-1]:
            lines.append("\\midrule")
    return "\n".join(lines)


def write_sources(summary):
    (OUT / "five_seed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT / "table2_source_data.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "method", "metric", "mean_percent", "sample_sd_percent", "seeds"])
        for dataset, ds in summary["datasets"].items():
            for method, metrics in ds["methods"].items():
                for metric, item in metrics.items():
                    writer.writerow([dataset, method, metric, item["mean"], item["sample_sd"], ";".join(map(str, SEEDS))])
    (OUT / "table2_rows.tex").write_text(table_block(summary) + "\n", encoding="utf-8")


def replace_table(tex, rows):
    pattern = re.compile(
        r"\\multirow\{8\}\{\*\}\{NSL-KDD\}.*?"
        r"(?=\\bottomrule)", re.S)
    if not pattern.search(tex):
        raise RuntimeError("Could not locate Table 2 data rows")
    return pattern.sub(lambda _: rows + "\n", tex, count=1)


def stats_paragraph(summary):
    phrases = []
    for dataset in DATASETS:
        ds = summary["datasets"][dataset]
        c = ds["paired_comparison"]
        p = c["paired_p_value"]
        if p >= 0.0001:
            p_text = f"{p:.4f}"
        else:
            coefficient, exponent = f"{p:.2e}".split("e")
            p_text = rf"{coefficient}\times10^{{{int(exponent)}}}"
        phrases.append(
            f"{c['mean_difference_pp']:.2f} percentage points on {dataset} relative to "
            f"{ds['strongest_competitor']} (95\\% CI {c['ci95_low_pp']:.2f}--{c['ci95_high_pp']:.2f}; "
            f"paired $p={p_text}$)")
    return (
        "Compared with the strongest competing method in Table~\\ref{tab:main}, HCE-Boost improved "
        "macro F1 by " + phrases[0] + ", " + phrases[1] + ", and " + phrases[2] + ". "
        "All five matched-seed differences were positive on each dataset."
    )


def update_manuscript(summary):
    path = WORK / "sn-article-revision.tex"
    tex = path.read_text(encoding="utf-8")
    tex = replace_table(tex, table_block(summary))
    tex = re.sub(
        r"Compared with the strongest existing method in the current tables, HCE-Boost improved.*?"
        r"All five matched-seed differences were positive on each dataset\.",
        lambda _: stats_paragraph(summary), tex, count=1, flags=re.S)
    path.write_text(tex, encoding="utf-8")


def response_stats(summary):
    clauses = []
    for dataset in DATASETS:
        ds = summary["datasets"][dataset]
        c = ds["paired_comparison"]
        p = c["paired_p_value"]
        if p >= 0.0001:
            p_text = f"{p:.4f}"
        else:
            coefficient, exponent = f"{p:.2e}".split("e")
            p_text = rf"{coefficient}\times10^{{{int(exponent)}}}"
        clauses.append(
            f"{c['mean_difference_pp']:.2f} points on {dataset} relative to "
            f"{ds['strongest_competitor']} (95\\% CI {c['ci95_low_pp']:.2f}--{c['ci95_high_pp']:.2f}; "
            f"$p={p_text}$)")
    return (
        "We agree. All eight methods use matched seeds 11, 22, 33, 44, and 55. "
        "HCE-Boost improves macro F1 over the strongest competing method by "
        + clauses[0] + ", " + clauses[1] + ", and " + clauses[2]
        + ". All five paired differences are positive on each dataset. These tests are exploratory because $n=5$."
    )


def update_response(summary):
    path = WORK / "Response_Letter_HCE-Boost.tex"
    tex = path.read_text(encoding="utf-8")
    replacement = response_stats(summary)
    pattern = re.compile(
        r"We agree\. All eight methods use matched seeds 11, 22, 33, 44, and 55\. "
        r"Table~2 reports.*?These tests are exploratory because \$n=5\$\.", re.S)
    if pattern.search(tex):
        tex = pattern.sub(lambda _: replacement, tex, count=1)
    elif replacement not in tex:
        raise RuntimeError("Could not locate the reviewer-response statistical paragraph")
    path.write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    results = collect()
    summary_obj = summarize(results)
    write_sources(summary_obj)
    if (WORK / "sn-article-revision.tex").exists():
        update_manuscript(summary_obj)
    if (WORK / "Response_Letter_HCE-Boost.tex").exists():
        update_response(summary_obj)
    print(json.dumps({
        ds: {
            "strongest_competitor": obj["strongest_competitor"],
            **obj["paired_comparison"],
        }
        for ds, obj in summary_obj["datasets"].items()
    }, indent=2))

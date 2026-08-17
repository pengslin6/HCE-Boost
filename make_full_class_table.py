"""Generate the complete five-seed HCE-Boost class-wise Table 3 source data."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "five_seed_final"
EXPECTED_SEEDS = [11, 22, 33, 44, 55]

DATASETS = [
    ("NSL-KDD", "nsl_hce_boost_detailed.json", {}),
    (
        "CIC-IDS2017",
        "cic_hce_boost_detailed.json",
        {
            "Web Attack - Brute Force": "Web Attack--Brute Force",
            "Web Attack - XSS": "Web Attack--XSS",
            "Web Attack - Sql Injection": "Web Attack--SQL Injection",
        },
    ),
    (
        "Edge-IIoTset",
        "edge_hce_boost_detailed.json",
        {
            "DDoS_UDP": "DDoS UDP",
            "DDoS_ICMP": "DDoS ICMP",
            "DDoS_HTTP": "DDoS HTTP",
            "SQL_injection": "SQL Injection",
            "DDoS_TCP": "DDoS TCP",
            "Vulnerability_scanner": "Vulnerability Scanner",
            "Port_Scanning": "Port Scanning",
        },
    ),
]


def summary(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    tex_lines: list[str] = []

    for dataset, filename, display_names in DATASETS:
        path = ROOT / "hce_boost_results" / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs = payload["individual"]
        seeds = [int(run["seed"]) for run in runs]
        if seeds != EXPECTED_SEEDS:
            raise ValueError(f"{dataset}: expected seeds {EXPECTED_SEEDS}, found {seeds}")

        class_names = list(runs[0]["test"]["per_class"])
        for class_name in class_names:
            metrics = [run["test"]["per_class"][class_name] for run in runs]
            supports = {int(metric["support"]) for metric in metrics}
            if len(supports) != 1:
                raise ValueError(f"{dataset}/{class_name}: support varies across seeds")

            p_mean, p_sd = summary([float(metric["precision"]) for metric in metrics])
            r_mean, r_sd = summary([float(metric["recall"]) for metric in metrics])
            f_mean, f_sd = summary([float(metric["f1"]) for metric in metrics])
            display_name = display_names.get(class_name, class_name)
            support = supports.pop()
            rows.append(
                {
                    "dataset": dataset,
                    "class": class_name,
                    "display_class": display_name,
                    "support": support,
                    "precision_mean_pct": f"{p_mean:.6f}",
                    "precision_sample_sd_pct": f"{p_sd:.6f}",
                    "recall_mean_pct": f"{r_mean:.6f}",
                    "recall_sample_sd_pct": f"{r_sd:.6f}",
                    "f1_mean_pct": f"{f_mean:.6f}",
                    "f1_sample_sd_pct": f"{f_sd:.6f}",
                }
            )
            tex_lines.append(
                f"{dataset} & {display_name} & {support:,} & "
                f"${p_mean:.2f}\\pm{p_sd:.2f}$ & ${r_mean:.2f}\\pm{r_sd:.2f}$ & "
                f"${f_mean:.2f}\\pm{f_sd:.2f}$ \\\\"
            )
        tex_lines.append("\\midrule")

    if len(rows) != 33:
        raise ValueError(f"expected 33 class rows, found {len(rows)}")
    tex_lines[-1] = "\\bottomrule"

    csv_path = OUT_DIR / "table3_complete_classwise_source.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tex_path = OUT_DIR / "table3_complete_classwise_rows.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"Rows: {len(rows)}")
    print(csv_path)
    print(tex_path)


if __name__ == "__main__":
    main()

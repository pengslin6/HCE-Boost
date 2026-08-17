#!/usr/bin/env python3
"""Five matched-seed runs of all retained neural baselines on one dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "run_edge_legacy_baselines.py"
REVISION_PATH = ROOT / "revision_experiments.py"
OUT_DIR = ROOT / "neural_five_seed_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]
MODEL_ORDER = ["DNN", "RNN", "LSTM-AE", "VLSTM", "CNN-BiLSTM"]
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def aggregate(rows):
    return {
        metric: {
            "mean": statistics.mean(row[metric] for row in rows),
            "sd": statistics.stdev(row[metric] for row in rows),
            "n": len(rows),
        }
        for metric in METRICS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nsl", "cic", "edge"], required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    base = import_file("legacy_baselines", BASE_PATH)
    revision = import_file("revision_experiments_suite", REVISION_PATH)
    data = revision.LOADERS[args.dataset]()
    out_path = OUT_DIR / f"{args.dataset}_neural_baselines_five_seed.json"
    payload = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {
        "dataset": data["name"],
        "split": data["split"],
        "features": int(data["X_train"].shape[1]),
        "classes": len(data["class_names"]),
        "train": len(data["y_train"]),
        "validation": len(data["y_val"]),
        "test": len(data["y_test"]),
        "seeds": SEEDS,
        "protocol": {
            "pretraining": "20-epoch maximum MSE reconstruction; VLSTM adds 0.1 KL",
            "classification": "60-epoch maximum normalized cube-root class-weighted cross entropy",
            "selection": "validation macro F1 with patience 10; no test tuning",
            "batch_size": args.batch_size,
            "optimizer": "Adam, pretraining lr=1e-3; classification lr=5e-4, weight_decay=1e-5",
        },
        "models": {},
    }

    for model_name in MODEL_ORDER:
        block = payload["models"].setdefault(model_name, {"individual": []})
        completed = {row["seed"] for row in block["individual"]}
        for seed in SEEDS:
            if seed in completed:
                print(f"SKIP {args.dataset} {model_name} seed={seed}", flush=True)
                continue
            base.SEED = seed
            base.seed_everything(seed)
            model = base.MODELS[model_name](
                data["X_train"].shape[1], n_cls=len(data["class_names"])
            ).to(base.DEVICE)
            print(f"START {args.dataset} {model_name} seed={seed}", flush=True)
            result = base.train_one(
                model_name, model, data,
                args.pretrain_epochs, args.finetune_epochs, args.patience, args.batch_size,
            )
            result["seed"] = seed
            block["individual"].append(result)
            block["individual"].sort(key=lambda row: row["seed"])
            block["aggregate"] = aggregate(block["individual"]) if len(block["individual"]) > 1 else None
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"SAVED {args.dataset} {model_name} seed={seed}: F1={result['f1']:.5f}", flush=True)

    for block in payload["models"].values():
        block["aggregate"] = aggregate(block["individual"])
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"COMPLETE {out_path}", flush=True)


if __name__ == "__main__":
    main()

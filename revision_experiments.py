#!/usr/bin/env python3
"""Reproducible reference and ablation experiments for HCE-Boost.

The script uses the analysis-ready CSV files supplied with the manuscript.  It
keeps the evaluation splits fixed and changes only the training seed so that
the reported dispersion measures optimisation variability rather than split
variability.  Results are written as JSON/CSV files for direct manuscript use.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("HCE_DATA_DIR", PROJECT / "data"))
OUT_DIR = Path(__file__).resolve().parent / "revision_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]
SEEDS_SMALL = [11, 33, 55]
DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _clean_numeric(frame: pd.DataFrame) -> np.ndarray:
    arr = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _scale_split(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_val, X_test


def load_nsl():
    train = pd.read_csv(DATA_DIR / "KDDTrain.csv")
    test = pd.read_csv(DATA_DIR / "KDDTest.csv")
    names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    mapping = {n.lower(): i for i, n in enumerate(names)}
    cols = [c for c in train.columns if c not in {"tpyes", "label", "isrun"}]
    X = _clean_numeric(train[cols])
    Xt = _clean_numeric(test[cols])
    y = train["tpyes"].str.lower().map(mapping).to_numpy(np.int64)
    yt = test["tpyes"].str.lower().map(mapping).to_numpy(np.int64)
    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )
    Xtr, Xv, Xt = _scale_split(Xtr, Xv, Xt)
    return dict(name="NSL-KDD", X_train=Xtr, y_train=ytr, X_val=Xv,
                y_val=yv, X_test=Xt, y_test=yt, class_names=names,
                split="official test set; 10% stratified validation from official training set")


def _normalise_cic_label(value: str) -> str:
    value = str(value).strip().replace("�", "-").replace("–", "-")
    value = " ".join(value.split())
    return value


def load_cic():
    frame = pd.read_csv(DATA_DIR / "CICIDS2017.csv", low_memory=False)
    frame["Label"] = frame["Label"].map(_normalise_cic_label)
    names = [
        "BENIGN", "PortScan", "DoS Hulk", "DDoS", "DoS GoldenEye",
        "FTP-Patator", "SSH-Patator", "DoS slowloris", "DoS Slowhttptest",
        "Bot", "Web Attack - Brute Force", "Web Attack - XSS",
        "Web Attack - Sql Injection",
    ]
    observed = sorted(frame["Label"].unique())
    if sorted(names) != observed:
        raise ValueError(f"Unexpected CIC labels: {observed}")
    mapping = {n: i for i, n in enumerate(names)}
    X = _clean_numeric(frame.drop(columns=["Label"]))
    y = frame["Label"].map(mapping).to_numpy(np.int64)
    Xdev, Xt, ydev, yt = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    Xtr, Xv, ytr, yv = train_test_split(
        Xdev, ydev, test_size=0.10, random_state=42, stratify=ydev
    )
    Xtr, Xv, Xt = _scale_split(Xtr, Xv, Xt)
    return dict(name="CIC-IDS2017 subset", X_train=Xtr, y_train=ytr,
                X_val=Xv, y_val=yv, X_test=Xt, y_test=yt,
                class_names=names,
                split="fixed stratified 72/8/20 split of the supplied 543,734-record CSV")


def load_edge():
    frame = pd.read_csv(DATA_DIR / "ML-EdgeIIoT-dataset.csv", low_memory=False)
    names = [
        "Normal", "DDoS_UDP", "DDoS_ICMP", "Ransomware", "DDoS_HTTP",
        "SQL_injection", "Uploading", "DDoS_TCP", "Backdoor",
        "Vulnerability_scanner", "Port_Scanning", "XSS", "Password",
        "MITM", "Fingerprinting",
    ]
    mapping = {n: i for i, n in enumerate(names)}
    y = frame["Attack_type"].astype(str).str.strip().map(mapping).to_numpy(np.int64)
    # Remove labels and raw identifiers/payload strings.  Retain columns that
    # are at least 95% numeric, then remove constants using training data only.
    drop = {
        "Attack_label", "Attack_type", "frame.time", "ip.src_host", "ip.dst_host",
        "arp.dst.proto_ipv4", "arp.src.proto_ipv4", "http.file_data",
        "http.request.uri.query", "http.referer", "http.request.full_uri",
        "tcp.options", "tcp.payload", "mqtt.msg",
    }
    raw = frame[[c for c in frame.columns if c not in drop]]
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    keep = numeric.notna().mean(axis=0) >= 0.95
    numeric = numeric.loc[:, keep]
    X = np.nan_to_num(numeric.to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    Xdev, Xt, ydev, yt = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    Xtr, Xv, ytr, yv = train_test_split(
        Xdev, ydev, test_size=0.10, random_state=42, stratify=ydev
    )
    varying = np.nanstd(Xtr, axis=0) > 0
    Xtr, Xv, Xt = Xtr[:, varying], Xv[:, varying], Xt[:, varying]
    Xtr, Xv, Xt = _scale_split(Xtr, Xv, Xt)
    return dict(name="Edge-IIoTset", X_train=Xtr, y_train=ytr,
                X_val=Xv, y_val=yv, X_test=Xt, y_test=yt,
                class_names=names,
                split="fixed stratified 72/8/20 split of the supplied ML CSV")


LOADERS = {"nsl": load_nsl, "cic": load_cic, "edge": load_edge}


class ScratchMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, latent_dim: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, in_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def encode(self, x):
        return self.encoder(x)

    def reconstruct(self, x):
        return self.decoder(self.encoder(x))

    def forward(self, x):
        return self.classifier(self.encoder(x))


def make_loader(X, y, batch_size: int, shuffle: bool, seed: int):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size, shuffle=shuffle, generator=gen, num_workers=0,
    )


def class_weights(y: np.ndarray, n_classes: int, alpha: float) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    base = len(y) / (n_classes * np.maximum(counts, 1.0))
    weights = np.power(base, alpha)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def pretrain(model, X, y, seed, epochs=12, batch_size=4096):
    loader = make_loader(X, y, batch_size, True, seed)
    opt = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=1e-3, weight_decay=1e-5,
    )
    start = time.perf_counter()
    peak = psutil.Process().memory_info().rss
    model.train()
    for _ in range(epochs):
        for xb, _ in loader:
            loss = F.mse_loss(model.reconstruct(xb.to(DEVICE)), xb.to(DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            peak = max(peak, psutil.Process().memory_info().rss)
    return time.perf_counter() - start, peak


@torch.no_grad()
def predict(model, X, batch_size=8192):
    model.eval()
    probs = []
    dummy = np.zeros(len(X), dtype=np.int64)
    for xb, _ in make_loader(X, dummy, batch_size, False, 0):
        probs.append(torch.softmax(model(xb.to(DEVICE)), dim=1).cpu().numpy())
    prob = np.concatenate(probs)
    return prob.argmax(axis=1), prob


def macro_f1(model, X, y):
    pred, _ = predict(model, X)
    return f1_score(y, pred, average="macro", zero_division=0)


def finetune(model, X, y, Xv, yv, seed, alpha=1/3, epochs=30,
             patience=6, batch_size=4096, freeze_encoder=False):
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=5e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y, model.n_classes, alpha))
    loader = make_loader(X, y, batch_size, True, seed)
    best, best_state, stale = -1.0, None, 0
    start = time.perf_counter()
    peak = psutil.Process().memory_info().rss
    epochs_run = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            peak = max(peak, psutil.Process().memory_info().rss)
        score = macro_f1(model, Xv, yv)
        epochs_run = epoch + 1
        if score > best + 1e-5:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return time.perf_counter() - start, peak, epochs_run


def evaluate(model, data):
    y = data["y_test"]
    pred, prob = predict(model, data["X_test"])
    result = {
        "accuracy": 100 * accuracy_score(y, pred),
        "precision": 100 * precision_score(y, pred, average="macro", zero_division=0),
        "recall": 100 * recall_score(y, pred, average="macro", zero_division=0),
        "f1": 100 * f1_score(y, pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(data["class_names"]))).tolist(),
    }
    try:
        result["auc"] = 100 * roc_auc_score(
            y, prob, labels=range(len(data["class_names"])), multi_class="ovr", average="macro"
        )
    except ValueError:
        result["auc"] = None
    p = precision_score(y, pred, labels=range(len(data["class_names"])), average=None, zero_division=0)
    r = recall_score(y, pred, labels=range(len(data["class_names"])), average=None, zero_division=0)
    f = f1_score(y, pred, labels=range(len(data["class_names"])), average=None, zero_division=0)
    result["per_class"] = {
        name: {"precision": 100*p[i], "recall": 100*r[i], "f1": 100*f[i],
               "support": int(np.sum(y == i))}
        for i, name in enumerate(data["class_names"])
    }
    return result


def evaluate_probabilities(y, prob, class_names):
    pred = prob.argmax(axis=1)
    result = {
        "accuracy": 100 * accuracy_score(y, pred),
        "precision": 100 * precision_score(y, pred, average="macro", zero_division=0),
        "recall": 100 * recall_score(y, pred, average="macro", zero_division=0),
        "f1": 100 * f1_score(y, pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(class_names))).tolist(),
    }
    try:
        result["auc"] = 100 * roc_auc_score(
            y, prob, labels=range(len(class_names)), multi_class="ovr", average="macro"
        )
    except ValueError:
        result["auc"] = None
    p = precision_score(y, pred, labels=range(len(class_names)), average=None, zero_division=0)
    r = recall_score(y, pred, labels=range(len(class_names)), average=None, zero_division=0)
    f = f1_score(y, pred, labels=range(len(class_names)), average=None, zero_division=0)
    result["per_class"] = {
        name: {"precision": 100*p[i], "recall": 100*r[i], "f1": 100*f[i],
               "support": int(np.sum(y == i))}
        for i, name in enumerate(class_names)
    }
    return result


def run_xgboost(data, seed):
    from xgboost import XGBClassifier

    seed_everything(seed)
    n_classes = len(data["class_names"])
    weights = class_weights(data["y_train"], n_classes, 1/3).cpu().numpy()
    sample_weight = weights[data["y_train"]]
    model = XGBClassifier(
        n_estimators=160, max_depth=6, learning_rate=0.08,
        min_child_weight=1.0, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, objective="multi:softprob", num_class=n_classes,
        eval_metric="mlogloss", tree_method="hist", n_jobs=12,
        random_state=seed,
    )
    baseline_rss = psutil.Process().memory_info().rss
    start = time.perf_counter()
    model.fit(data["X_train"], data["y_train"], sample_weight=sample_weight, verbose=False)
    seconds = time.perf_counter() - start
    peak = psutil.Process().memory_info().rss
    prob = model.predict_proba(data["X_test"])
    result = evaluate_probabilities(data["y_test"], prob, data["class_names"])
    result.update({
        "seed": seed, "training_seconds": seconds,
        "rss_increment_after_fit_mb": (peak-baseline_rss)/(1024**2),
        "n_estimators": 160, "max_depth": 6, "class_weight_alpha": 1/3,
    })
    return result


def model_counts(model):
    train_params = sum(p.numel() for p in model.parameters())
    deploy_params = sum(p.numel() for p in model.encoder.parameters()) + sum(
        p.numel() for p in model.classifier.parameters()
    )
    d, h, z, c = model.in_dim, 128, model.latent_dim, model.n_classes
    deploy_macs = d*h + h*z + z*h + h*c
    train_macs = deploy_macs + z*h + h*d
    return {
        "training_parameters": train_params,
        "deployment_parameters": deploy_params,
        "training_size_mb_fp32": train_params*4/(1024**2),
        "deployment_size_mb_fp32": deploy_params*4/(1024**2),
        "training_macs_per_sample": train_macs,
        "deployment_macs_per_sample": deploy_macs,
        "deployment_flops_per_sample_2xmac": 2*deploy_macs,
    }


@torch.no_grad()
def latency_protocol(model, in_dim, batch_size, warmups=50, trials=500):
    model.eval()
    x = torch.zeros((batch_size, in_dim), dtype=torch.float32)
    for _ in range(warmups):
        model(x)
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter_ns()
        model(x)
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    arr = np.asarray(samples) / batch_size
    return {
        "batch_size": batch_size, "warmups": warmups, "trials": trials,
        "mean_ms_per_sample": float(arr.mean()),
        "std_ms_per_sample": float(arr.std(ddof=1)),
        "p95_ms_per_sample": float(np.percentile(arr, 95)),
        "throughput_samples_s": float(batch_size / (np.asarray(samples).mean()/1000)),
    }


def run_variant(data, seed, *, pretraining=True, alpha=1/3, latent=32,
                freeze=False, fraction=1.0):
    seed_everything(seed)
    n_classes = len(data["class_names"])
    model = ScratchMLP(data["X_train"].shape[1], n_classes, latent).to(DEVICE)
    baseline_rss = psutil.Process().memory_info().rss
    pre_s, peak1 = (0.0, baseline_rss)
    if pretraining:
        pre_s, peak1 = pretrain(model, data["X_train"], data["y_train"], seed)
    X, y = data["X_train"], data["y_train"]
    if fraction < 1.0:
        idx = np.arange(len(y))
        keep, _ = train_test_split(idx, train_size=fraction, random_state=seed, stratify=y)
        X, y = X[keep], y[keep]
    fine_s, peak2, epochs_run = finetune(
        model, X, y, data["X_val"], data["y_val"], seed,
        alpha=alpha, freeze_encoder=freeze,
    )
    result = evaluate(model, data)
    result.update({
        "seed": seed, "pretraining": pretraining, "alpha": alpha,
        "latent_dim": latent, "freeze_encoder": freeze,
        "label_fraction": fraction, "pretrain_seconds": pre_s,
        "finetune_seconds": fine_s, "finetune_epochs": epochs_run,
        "peak_rss_increment_mb": (max(peak1, peak2)-baseline_rss)/(1024**2),
    })
    result.update(model_counts(model))
    if seed == 33 and pretraining and abs(alpha-1/3) < 1e-9 and latent == 32 and not freeze and fraction == 1.0:
        result["latency_batch1"] = latency_protocol(model, model.in_dim, 1)
        result["latency_batch256"] = latency_protocol(model, model.in_dim, 256, trials=200)
    return result


def aggregate(rows):
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    out = {}
    for metric in metrics:
        vals = [r[metric] for r in rows if r.get(metric) is not None]
        n = len(vals)
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if n > 1 else 0.0
        ci = stats.t.ppf(0.975, n-1) * sd / math.sqrt(n) if n > 1 else 0.0
        out[metric] = {"mean": mean, "sd": sd, "ci95_halfwidth": ci, "n": n}
    out["per_class"] = {}
    for name in rows[0]["per_class"]:
        out["per_class"][name] = {}
        for metric in ["precision", "recall", "f1"]:
            vals = [r["per_class"][name][metric] for r in rows]
            out["per_class"][name][metric] = {
                "mean": statistics.mean(vals),
                "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
        out["per_class"][name]["support"] = rows[0]["per_class"][name]["support"]
    return out


def save_json(name, payload):
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {path}", flush=True)


def run_suite(dataset_key: str, mode: str):
    data = LOADERS[dataset_key]()
    print(data["name"], data["X_train"].shape, data["X_test"].shape, data["split"], flush=True)
    payload = {
        "dataset": data["name"], "split": data["split"],
        "features": int(data["X_train"].shape[1]),
        "train": int(len(data["y_train"])), "validation": int(len(data["y_val"])),
        "test": int(len(data["y_test"])), "class_names": data["class_names"],
        "runs": {},
    }
    if mode in {"main", "all"}:
        key = "reconstruction_mlp_cube"
        rows = []
        for seed in SEEDS:
            print(dataset_key, key, seed, flush=True)
            rows.append(run_variant(data, seed))
        payload["runs"][key] = {"individual": rows, "aggregate": aggregate(rows)}

    if mode in {"xgb", "all"}:
        key = "xgboost_cube"
        rows = []
        for seed in SEEDS:
            print(dataset_key, key, seed, flush=True)
            rows.append(run_xgboost(data, seed))
        payload["runs"][key] = {"individual": rows, "aggregate": aggregate(rows)}

    if mode in {"scratch", "all"}:
        key = "mlp_scratch_cube"
        rows = []
        for seed in SEEDS:
            print(dataset_key, key, seed, flush=True)
            rows.append(run_variant(data, seed, pretraining=False, alpha=1/3))
        payload["runs"][key] = {"individual": rows, "aggregate": aggregate(rows)}

    if mode in {"ablation", "all"}:
        variants = [
            ("scratch_uniform", False, 0.0, False),
            ("scratch_cube", False, 1/3, False),
            ("pretrain_uniform", True, 0.0, False),
            ("pretrain_sqrt", True, 1/2, False),
            ("pretrain_inverse", True, 1.0, False),
            ("pretrain_frozen_cube", True, 1/3, True),
        ]
        for key, pre, alpha, freeze in variants:
            rows = []
            for seed in SEEDS:
                print(dataset_key, key, seed, flush=True)
                rows.append(run_variant(data, seed, pretraining=pre, alpha=alpha, freeze=freeze))
            payload["runs"][key] = {"individual": rows, "aggregate": aggregate(rows)}

    if mode in {"scarcity", "all"}:
        # The 100% pre-trained and scratch conditions are already present in
        # the main and ablation outputs, respectively.
        for frac in [0.10, 0.25, 0.50]:
            for pre in [False, True]:
                key = f"{'pretrain' if pre else 'scratch'}_cube_labels_{int(frac*100)}pct"
                rows = []
                for seed in SEEDS_SMALL:
                    print(dataset_key, key, seed, flush=True)
                    rows.append(run_variant(data, seed, pretraining=pre, fraction=frac))
                payload["runs"][key] = {"individual": rows, "aggregate": aggregate(rows)}

    if mode in {"sensitivity", "all"}:
        # Latent=32 is the main configuration and is read from the main output.
        for latent in [8, 16, 64]:
            key = f"latent_{latent}"
            rows = []
            for seed in SEEDS_SMALL:
                print(dataset_key, key, seed, flush=True)
                rows.append(run_variant(data, seed, latent=latent))
            payload["runs"][key] = {"individual": rows, "aggregate": aggregate(rows)}

    save_json(f"{dataset_key}_{mode}.json", payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=LOADERS, required=True)
    parser.add_argument("--mode", choices=["main", "xgb", "scratch", "ablation", "scarcity", "sensitivity", "all"], required=True)
    args = parser.parse_args()
    for key in args.datasets:
        run_suite(key, args.mode)


if __name__ == "__main__":
    main()

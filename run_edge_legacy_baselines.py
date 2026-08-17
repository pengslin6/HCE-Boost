#!/usr/bin/env python3
"""Re-evaluate the five original neural baselines on the revision Edge split.

The dataset loader is shared with the revision experiments.  These runs are
single-seed (42) comparison runs, selected by validation macro F1 and never by
test labels.  All classifiers use the same normalized cube-root class weights
as the revision tree models.  Results are written after each model so the run
is resumable on CPU-only workstations.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
REVISION_PATH = ROOT / "revision_experiments.py"
OUT_PATH = ROOT / "edge_legacy_baselines_seed42.json"
DEVICE = torch.device("cpu")
SEED = 42
N_THREADS = 12
torch.set_num_threads(N_THREADS)


def load_revision_module():
    spec = importlib.util.spec_from_file_location("revision_experiments", REVISION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class LSTMAE(nn.Module):
    def __init__(self, in_dim, h=128, z=64, n_cls=15):
        super().__init__()
        self.lstm_enc = nn.LSTM(in_dim, h, num_layers=2, batch_first=True, dropout=0.1)
        self.fc_enc = nn.Linear(h, z)
        self.lstm_dec = nn.LSTM(z, h, num_layers=1, batch_first=True)
        self.fc_dec = nn.Linear(h, in_dim)
        self.classifier = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, n_cls))

    def encode(self, x):
        out, _ = self.lstm_enc(x.unsqueeze(1))
        return self.fc_enc(out[:, -1])

    def forward(self, x, mode="finetune"):
        z = self.encode(x)
        if mode == "pretrain":
            out, _ = self.lstm_dec(z.unsqueeze(1))
            return self.fc_dec(out[:, -1]), z
        return self.classifier(z)


class CNNBiLSTM(nn.Module):
    def __init__(self, in_dim, h=128, z=64, n_cls=15):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
        )
        self.lstm = nn.LSTM(64, h // 2, batch_first=True, bidirectional=True)
        self.fc_enc = nn.Linear(h, z)
        self.decoder = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, in_dim))
        self.classifier = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, n_cls))

    def encode(self, x):
        c = self.cnn(x.unsqueeze(1)).permute(0, 2, 1)
        out, _ = self.lstm(c)
        return self.fc_enc(out[:, -1])

    def forward(self, x, mode="finetune"):
        z = self.encode(x)
        if mode == "pretrain":
            return self.decoder(z), z
        return self.classifier(z)


class VLSTM(nn.Module):
    def __init__(self, in_dim, h=128, z=64, n_cls=15):
        super().__init__()
        self.lstm_enc = nn.LSTM(in_dim, h, num_layers=2, batch_first=True, dropout=0.1)
        self.fc_mu = nn.Linear(h, z)
        self.fc_var = nn.Linear(h, z)
        self.lstm_dec = nn.LSTM(z, h, batch_first=True)
        self.fc_out = nn.Linear(h, in_dim)
        self.classifier = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, n_cls))

    def encode(self, x):
        out, _ = self.lstm_enc(x.unsqueeze(1))
        h = out[:, -1]
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu
        return z, mu, logvar

    def forward(self, x, mode="finetune"):
        z, mu, logvar = self.encode(x)
        if mode == "pretrain":
            out, _ = self.lstm_dec(z.unsqueeze(1))
            return self.fc_out(out[:, -1]), mu, logvar, z
        return self.classifier(z)


class DNN(nn.Module):
    def __init__(self, in_dim, h=128, z=64, n_cls=15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.1),
            nn.Linear(h, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.1),
            nn.Linear(h, z), nn.ReLU(),
        )
        self.decoder = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, in_dim))
        self.classifier = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, n_cls))

    def forward(self, x, mode="finetune"):
        z = self.encoder(x)
        if mode == "pretrain":
            return self.decoder(z), z
        return self.classifier(z)


class RNN(nn.Module):
    def __init__(self, in_dim, h=128, z=64, n_cls=15):
        super().__init__()
        self.rnn_enc = nn.GRU(in_dim, h, num_layers=2, batch_first=True, dropout=0.1)
        self.fc_enc = nn.Linear(h, z)
        self.rnn_dec = nn.GRU(z, h, batch_first=True)
        self.fc_dec = nn.Linear(h, in_dim)
        self.classifier = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, n_cls))

    def encode(self, x):
        out, _ = self.rnn_enc(x.unsqueeze(1))
        return self.fc_enc(out[:, -1])

    def forward(self, x, mode="finetune"):
        z = self.encode(x)
        if mode == "pretrain":
            out, _ = self.rnn_dec(z.unsqueeze(1))
            return self.fc_dec(out[:, -1]), z
        return self.classifier(z)


MODELS = {"LSTM-AE": LSTMAE, "CNN-BiLSTM": CNNBiLSTM, "VLSTM": VLSTM, "DNN": DNN, "RNN": RNN}


def loader(X, y, batch_size, shuffle, seed):
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen, num_workers=0)


@torch.inference_mode()
def predict(model, X, batch_size=4096):
    model.eval()
    out = []
    ds = TensorDataset(torch.from_numpy(X))
    for (xb,) in DataLoader(ds, batch_size=batch_size, shuffle=False):
        out.append(torch.softmax(model(xb.to(DEVICE)), dim=1).cpu().numpy())
    return np.concatenate(out)


def macro_f1(model, X, y):
    return f1_score(y, predict(model, X).argmax(1), average="macro", zero_division=0)


def class_weights(y, n_classes, alpha=1/3):
    counts = np.maximum(np.bincount(y, minlength=n_classes).astype(np.float64), 1.0)
    weights = (counts.sum() / (n_classes * counts)) ** alpha
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def train_one(name, model, data, pretrain_epochs=20, finetune_epochs=60, patience=10, batch_size=1024):
    train_loader = loader(data["X_train"], data["y_train"], batch_size, True, SEED)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for epoch in range(pretrain_epochs):
        model.train()
        running = 0.0
        for xb, _ in train_loader:
            xb = xb.to(DEVICE)
            result = model(xb, mode="pretrain")
            if name == "VLSTM":
                recon, mu, logvar, _ = result
                loss = F.mse_loss(recon, xb) + 0.1 * (-0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp()))
            else:
                recon, _ = result
                loss = F.mse_loss(recon, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss) * len(xb)
        print(f"{name} pretrain {epoch+1:02d}/{pretrain_epochs}: loss={running/len(data['y_train']):.5f}", flush=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights(data["y_train"], len(data["class_names"])))
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    best, best_state, stale, best_epoch = -1.0, None, 0, 0
    for epoch in range(finetune_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        score = macro_f1(model, data["X_val"], data["y_val"])
        print(f"{name} finetune {epoch+1:02d}/{finetune_epochs}: val_macro_f1={100*score:.3f}", flush=True)
        if score > best + 1e-5:
            best, best_epoch, stale = score, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    prob = predict(model, data["X_test"])
    pred, y = prob.argmax(1), data["y_test"]
    metrics = {
        "accuracy": 100 * accuracy_score(y, pred),
        "precision": 100 * precision_score(y, pred, average="macro", zero_division=0),
        "recall": 100 * recall_score(y, pred, average="macro", zero_division=0),
        "f1": 100 * f1_score(y, pred, average="macro", zero_division=0),
        "auc": 100 * roc_auc_score(y, prob, labels=range(len(data["class_names"])), multi_class="ovr", average="macro"),
        "best_validation_f1": 100 * best,
        "selected_epoch": best_epoch,
        "training_seconds": time.perf_counter() - t0,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-path", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    seed_everything()
    revision = load_revision_module()
    data = revision.load_edge()
    out_path = args.output_path
    payload = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {
        "dataset": data["name"], "split": data["split"], "seed": SEED,
        "features": int(data["X_train"].shape[1]), "classes": len(data["class_names"]),
        "train": len(data["y_train"]), "validation": len(data["y_val"]), "test": len(data["y_test"]),
        "protocol": {
            "pretraining": "MSE reconstruction; VLSTM adds 0.1 KL",
            "classification": "normalized cube-root class-weighted cross entropy",
            "selection": "validation macro F1 early stopping; no test tuning",
            "batch_size": args.batch_size, "pretrain_epochs_max": args.pretrain_epochs,
            "finetune_epochs_max": args.finetune_epochs, "patience": args.patience,
        },
        "results": {},
    }
    for name in args.models:
        if name in payload["results"]:
            print(f"skip {name}: already present", flush=True)
            continue
        seed_everything()
        model = MODELS[name](data["X_train"].shape[1], n_cls=len(data["class_names"])).to(DEVICE)
        print(f"START {name}", flush=True)
        payload["results"][name] = train_one(
            name, model, data, args.pretrain_epochs, args.finetune_epochs, args.patience, args.batch_size
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"SAVED {name}: {payload['results'][name]}", flush=True)
    print(out_path, flush=True)


if __name__ == "__main__":
    main()

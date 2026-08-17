#!/usr/bin/env python3
"""Validation-first audit of strong tabular baselines on the three supplied datasets.

Hyperparameters and class-weight exponents are selected from validation data.
The fixed test partitions are evaluated only after model fitting. Probability
files are retained so that later fusion rules can be fitted on validation data
without retraining the base learners.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


PROJECT = Path(__file__).resolve().parent
REVISION_SCRIPT = PROJECT / "revision_experiments.py"
OUT_DIR = Path(__file__).resolve().parent / "strong_tabular_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_revision_module():
    spec = importlib.util.spec_from_file_location("revision_experiments", REVISION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_weights(y: np.ndarray, n_classes: int, alpha: float) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    base = len(y) / (n_classes * np.maximum(counts, 1.0))
    weights = np.power(base, alpha)
    weights /= weights.mean()
    return weights[y].astype(np.float32)


def metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(axis=1)
    out = {
        "accuracy": 100.0 * accuracy_score(y, pred),
        "precision": 100.0 * precision_score(y, pred, average="macro", zero_division=0),
        "recall": 100.0 * recall_score(y, pred, average="macro", zero_division=0),
        "f1": 100.0 * f1_score(y, pred, average="macro", zero_division=0),
    }
    try:
        out["auc"] = 100.0 * roc_auc_score(
            y, prob, labels=np.arange(prob.shape[1]), multi_class="ovr", average="macro"
        )
    except ValueError:
        out["auc"] = None
    return out


def fit_lgbm(data: dict, seed: int, alpha: float):
    import lightgbm as lgb

    n_classes = len(data["class_names"])
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=1500,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=12,
        verbosity=-1,
    )
    model.fit(
        data["X_train"],
        data["y_train"],
        sample_weight=sample_weights(data["y_train"], n_classes, alpha),
        eval_set=[(data["X_val"], data["y_val"])],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    return model, int(model.best_iteration_)


def fit_xgb(data: dict, seed: int, alpha: float):
    from xgboost import XGBClassifier

    n_classes = len(data["class_names"])
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=1200,
        learning_rate=0.04,
        max_depth=8,
        min_child_weight=1.0,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=60,
        random_state=seed,
        n_jobs=12,
    )
    model.fit(
        data["X_train"],
        data["y_train"],
        sample_weight=sample_weights(data["y_train"], n_classes, alpha),
        eval_set=[(data["X_val"], data["y_val"])],
        verbose=False,
    )
    return model, int(model.best_iteration + 1)


def fit_catboost(data: dict, seed: int, alpha: float):
    from catboost import CatBoostClassifier

    n_classes = len(data["class_names"])
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=800,
        depth=8,
        learning_rate=0.06,
        l2_leaf_reg=3.0,
        random_strength=0.5,
        border_count=128,
        random_seed=seed,
        thread_count=12,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=60,
        use_best_model=True,
    )
    model.fit(
        data["X_train"],
        data["y_train"],
        sample_weight=sample_weights(data["y_train"], n_classes, alpha),
        eval_set=(data["X_val"], data["y_val"]),
        verbose=False,
    )
    return model, int(model.get_best_iteration() + 1)


FITTERS = {"lgbm": fit_lgbm, "xgb": fit_xgb, "catboost": fit_catboost}


def refit_on_development(model_key: str, model, best_iteration: int, data: dict, alpha: float):
    """Refit the validation-selected learner on train+validation without test feedback."""
    X_dev = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    y_dev = np.concatenate([data["y_train"], data["y_val"]], axis=0)
    weights = sample_weights(y_dev, len(data["class_names"]), alpha)
    if model_key in {"lgbm", "xgb"}:
        refit = clone(model)
        refit.set_params(n_estimators=best_iteration)
        if model_key == "xgb":
            refit.set_params(early_stopping_rounds=None)
            refit.fit(X_dev, y_dev, sample_weight=weights, verbose=False)
        else:
            refit.fit(X_dev, y_dev, sample_weight=weights)
    else:
        from catboost import CatBoostClassifier

        params = model.get_params()
        params.update(
            iterations=best_iteration,
            use_best_model=False,
            verbose=False,
            allow_writing_files=False,
        )
        params.pop("od_type", None)
        params.pop("od_wait", None)
        refit = CatBoostClassifier(**params)
        refit.fit(X_dev, y_dev, sample_weight=weights, verbose=False)
    return refit


def run(dataset_key: str, model_key: str, seed: int, alpha: float):
    revision = load_revision_module()
    data = revision.LOADERS[dataset_key]()
    start = time.perf_counter()
    model, best_iteration = FITTERS[model_key](data, seed, alpha)
    seconds = time.perf_counter() - start
    val_prob = np.asarray(model.predict_proba(data["X_val"]), dtype=np.float64)
    test_prob = np.asarray(model.predict_proba(data["X_test"]), dtype=np.float64)
    refit_start = time.perf_counter()
    refit = refit_on_development(model_key, model, best_iteration, data, alpha)
    refit_seconds = time.perf_counter() - refit_start
    refit_test_prob = np.asarray(refit.predict_proba(data["X_test"]), dtype=np.float64)
    result = {
        "dataset": data["name"],
        "model": model_key,
        "seed": seed,
        "class_weight_alpha": alpha,
        "best_iteration": best_iteration,
        "training_seconds": seconds,
        "refit_seconds": refit_seconds,
        "features": int(data["X_train"].shape[1]),
        "train": int(len(data["y_train"])),
        "validation": int(len(data["y_val"])),
        "test": int(len(data["y_test"])),
        "validation_metrics": metrics(data["y_val"], val_prob),
        "test_metrics": metrics(data["y_test"], test_prob),
        "refit_test_metrics": metrics(data["y_test"], refit_test_prob),
    }
    stem = f"{dataset_key}_{model_key}_seed{seed}_alpha{alpha:g}".replace(".", "p")
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT_DIR / f"{stem}_probabilities.npz",
        val_prob=val_prob,
        test_prob=test_prob,
        refit_test_prob=refit_test_prob,
        y_val=data["y_val"],
        y_test=data["y_test"],
        class_names=np.asarray(data["class_names"]),
    )
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nsl", "cic", "edge"], required=True)
    parser.add_argument("--model", choices=sorted(FITTERS), required=True)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--alpha", type=float, default=1 / 3)
    args = parser.parse_args()
    run(args.dataset, args.model, args.seed, args.alpha)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""HCE-Boost: hierarchy-constrained expert boosting experiments.

This candidate model is intentionally evaluated as a fully supervised method.
It uses the same labelled partition and cube-root class weights for both global
experts.  Hyperparameters and specialist activation depend on validation data;
the fixed test labels are used only for the final metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


PROJECT = Path(__file__).resolve().parent
AUDIT_SCRIPT = PROJECT / "strong_tabular_audit.py"
OUT = PROJECT / "hce_boost_results"
OUT.mkdir(exist_ok=True)
SEEDS = [11, 22, 33, 44, 55]
ALPHA = 1 / 3


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metrics(y, prob, class_names=None):
    pred = prob.argmax(axis=1)
    result = {
        "accuracy": 100 * accuracy_score(y, pred),
        "precision": 100 * precision_score(y, pred, average="macro", zero_division=0),
        "recall": 100 * recall_score(y, pred, average="macro", zero_division=0),
        "f1": 100 * f1_score(y, pred, average="macro", zero_division=0),
    }
    try:
        result["auc"] = 100 * roc_auc_score(y, prob, multi_class="ovr", average="macro")
    except ValueError:
        result["auc"] = None
    if class_names is not None:
        labels = np.arange(len(class_names))
        p = precision_score(y, pred, labels=labels, average=None, zero_division=0)
        r = recall_score(y, pred, labels=labels, average=None, zero_division=0)
        f = f1_score(y, pred, labels=labels, average=None, zero_division=0)
        result["per_class"] = {
            name: {
                "support": int(np.sum(y == i)),
                "precision": 100 * float(p[i]),
                "recall": 100 * float(r[i]),
                "f1": 100 * float(f[i]),
            }
            for i, name in enumerate(class_names)
        }
    return result


def family_groups(class_names):
    """Infer explicit subtype families from label names, without test feedback."""
    groups = {}
    for index, name in enumerate(class_names):
        if " - " in name:
            prefix = name.split(" - ", 1)[0]
        elif "_" in name:
            prefix = name.split("_", 1)[0]
        else:
            continue
        groups.setdefault(prefix, []).append(index)
    return [np.asarray(v, dtype=int) for v in groups.values() if len(v) >= 3]


def activate_groups(y_val, val_prob, groups, minimum_confusion=0.05):
    """Activate a family only when validation shows material within-family confusion."""
    pred = val_prob.argmax(axis=1)
    active = []
    for group in groups:
        mask = np.isin(y_val, group)
        if not np.any(mask):
            continue
        within_error = np.sum((pred[mask] != y_val[mask]) & np.isin(pred[mask], group))
        rate = within_error / np.sum(mask)
        if rate >= minimum_confusion:
            active.append((group, float(rate)))
    return active


def minority_confusion_groups(y_val, val_prob, maximum_support_fraction=0.01,
                              minimum_error_rate=0.01):
    """Find two or more rare classes that are confused with the same anchor class."""
    pred = val_prob.argmax(axis=1)
    n_classes = val_prob.shape[1]
    by_anchor = {}
    rates = {}
    for source in range(n_classes):
        mask = y_val == source
        support = int(mask.sum())
        if support == 0 or support / len(y_val) > maximum_support_fraction:
            continue
        wrong = pred[mask][pred[mask] != source]
        if len(wrong) == 0:
            continue
        counts = np.bincount(wrong, minlength=n_classes)
        anchor = int(counts.argmax())
        rate = counts[anchor] / support
        if rate >= minimum_error_rate:
            by_anchor.setdefault(anchor, []).append(source)
            rates[(anchor, source)] = float(rate)
    active = []
    for anchor, sources in by_anchor.items():
        if len(sources) >= 2:
            group = np.asarray([anchor, *sources], dtype=int)
            signal = float(np.mean([rates[(anchor, source)] for source in sources]))
            active.append((group, signal))
    return active


def specialist_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = (len(y) / (n_classes * np.maximum(counts, 1))) ** ALPHA
    weights /= weights.mean()
    return weights[y].astype(np.float32)


def fit_specialist(data, group, seed):
    from xgboost import XGBClassifier

    mapping = {old: new for new, old in enumerate(group)}
    train_mask = np.isin(data["y_train"], group)
    val_mask = np.isin(data["y_val"], group)
    y_train = np.asarray([mapping[v] for v in data["y_train"][train_mask]], dtype=int)
    y_val = np.asarray([mapping[v] for v in data["y_val"][val_mask]], dtype=int)
    n_classes = len(group)
    common = dict(
        objective="multi:softprob", num_class=n_classes, learning_rate=0.035,
        max_depth=5, min_child_weight=1, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=2, tree_method="hist", eval_metric="mlogloss",
        random_state=seed, n_jobs=12,
    )
    selector = XGBClassifier(n_estimators=1000, early_stopping_rounds=60, **common)
    selector.fit(
        data["X_train"][train_mask], y_train,
        sample_weight=specialist_weights(y_train, n_classes),
        eval_set=[(data["X_val"][val_mask], y_val)], verbose=False,
    )
    iterations = int(selector.best_iteration + 1)

    X_dev = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    y_dev_original = np.concatenate([data["y_train"], data["y_val"]], axis=0)
    dev_mask = np.isin(y_dev_original, group)
    y_dev = np.asarray([mapping[v] for v in y_dev_original[dev_mask]], dtype=int)
    model = XGBClassifier(n_estimators=iterations, **common)
    model.fit(
        X_dev[dev_mask], y_dev,
        sample_weight=specialist_weights(y_dev, n_classes), verbose=False,
    )
    return model, iterations


def redistribute(prob, conditional, group):
    """Preserve the generalist's family mass and only refine its subtype split."""
    result = prob.copy()
    mass = result[:, group].sum(axis=1, keepdims=True)
    result[:, group] = mass * conditional
    return result


def run_seed(dataset_key, data, seed, audit):
    # Validation selects boosting length; both experts are then refitted on the
    # complete development partition with exactly the same class-weight rule.
    xgb_selector, xgb_iterations = audit.fit_xgb(data, seed, ALPHA)
    lgb_selector, lgb_iterations = audit.fit_lgbm(data, seed, ALPHA)
    val_prob = (
        np.asarray(xgb_selector.predict_proba(data["X_val"]), dtype=float)
        + np.asarray(lgb_selector.predict_proba(data["X_val"]), dtype=float)
    ) / 2
    xgb = audit.refit_on_development("xgb", xgb_selector, xgb_iterations, data, ALPHA)
    lgb = audit.refit_on_development("lgbm", lgb_selector, lgb_iterations, data, ALPHA)
    test_prob = (
        np.asarray(xgb.predict_proba(data["X_test"]), dtype=float)
        + np.asarray(lgb.predict_proba(data["X_test"]), dtype=float)
    ) / 2

    global_test = metrics(data["y_test"], test_prob, data["class_names"])
    active = activate_groups(data["y_val"], val_prob, family_groups(data["class_names"]))
    active.extend(minority_confusion_groups(data["y_val"], val_prob))
    specialist_log = []
    for group, confusion_rate in active:
        specialist, iterations = fit_specialist(data, group, seed)
        conditional = np.asarray(specialist.predict_proba(data["X_test"]), dtype=float)
        test_prob = redistribute(test_prob, conditional, group)
        specialist_log.append({
            "classes": [data["class_names"][i] for i in group],
            "validation_within_family_confusion_rate": confusion_rate,
            "iterations": iterations,
        })

    row = {
        "seed": seed,
        "xgb_iterations": xgb_iterations,
        "lgbm_iterations": lgb_iterations,
        "specialists": specialist_log,
        "global_test": global_test,
        "test": metrics(data["y_test"], test_prob, data["class_names"]),
    }
    print(json.dumps({"dataset": dataset_key, **row}), flush=True)
    return row


def aggregate(rows):
    output = {}
    for key in ("accuracy", "precision", "recall", "f1", "auc"):
        values = [row["test"][key] for row in rows if row["test"][key] is not None]
        output[key] = {
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["nsl", "cic", "edge"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    audit = load_module(AUDIT_SCRIPT, "strong_tabular_audit")
    revision = audit.load_revision_module()
    for key in args.datasets:
        data = revision.LOADERS[key]()
        rows = []
        for seed in args.seeds:
            row = run_seed(key, data, seed, audit)
            rows.append(row)
            checkpoint = OUT / f"{key}_hce_boost_seed{seed}.json"
            checkpoint.write_text(json.dumps(row, indent=2), encoding="utf-8")
        payload = {
            "model": "HCE-Boost",
            "dataset": data["name"],
            "split": data["split"],
            "selection_rule": "validation early stopping and validation confusion; no test tuning",
            "class_weight_alpha": ALPHA,
            "individual": rows,
            "aggregate": aggregate(rows),
        }
        path = OUT / f"{key}_hce_boost_detailed.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()

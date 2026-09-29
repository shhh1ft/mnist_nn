"""Анализ расхождений между двумя моделями на одной выборке.

Ищет примеры, где модель A ошиблась, а модель B угадала (и наоборот),
и рисует их сеткой.

Пример:
    python -m src.analyze_errors \
        --a mlp=models/mlp_best.npz,configs/mlp_baseline.json \
        --b cnn=models/cnn_best.npz,configs/cnn_baseline.json \
        --tag mlp_vs_cnn

Создаёт:
    reports/{tag}_a_wrong_b_right.png
    reports/{tag}_b_wrong_a_right.png
    reports/{tag}_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compare import parse_entry
from .data_loader import load_mnist
from .losses import SoftmaxCrossEntropy
from .model import build_model_from_config
from .train import reshape_for_model
from .utils import ensure_dir, softmax
from .visualize import plot_misclassified


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Анализ расхождений между двумя моделями"
    )
    p.add_argument("--a", type=parse_entry, required=True,
                   help="name=model_path,config_path — первая модель")
    p.add_argument("--b", type=parse_entry, required=True,
                   help="name=model_path,config_path — вторая модель")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--reports-dir", type=str, default="reports")
    p.add_argument("--tag", type=str, default="analysis")
    p.add_argument("--max-items", type=int, default=25)
    return p.parse_args()


def predict(model, config: dict, X_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model_type = config["model"].get("type", "mlp")
    X = reshape_for_model(X_flat, model_type)
    logits = model.forward(X, training=False)
    probs = softmax(logits, axis=1)
    return probs, probs.argmax(axis=1)


def load_model(name_and_paths: tuple[str, Path, Path]):
    name, model_path, config_path = name_and_paths
    if not model_path.exists():
        raise SystemExit(f"[{name}] модель не найдена: {model_path}")
    if not config_path.exists():
        raise SystemExit(f"[{name}] конфиг не найден: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    model = build_model_from_config(config, seed=config.get("seed", 42))
    model.load(model_path)
    return name, model, config


def main() -> None:
    args = parse_args()

    name_a, model_a, cfg_a = load_model(args.a)
    name_b, model_b, cfg_b = load_model(args.b)

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(
        args.data_dir, val_size=0.1, seed=42,
    )
    if args.split == "val":
        X, y = X_val, y_val
    else:
        X, y = X_test, y_test
    y_true = y.argmax(axis=1)
    print(f"[data] split={args.split}  N={len(X)}")

    probs_a, pred_a = predict(model_a, cfg_a, X)
    probs_b, pred_b = predict(model_b, cfg_b, X)

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    a_wrong_b_right = np.where(~correct_a & correct_b)[0]
    b_wrong_a_right = np.where(~correct_b & correct_a)[0]
    both_wrong = np.where(~correct_a & ~correct_b)[0]
    both_right = np.where(correct_a & correct_b)[0]

    acc_a = float(correct_a.mean())
    acc_b = float(correct_b.mean())
    print(f"[{name_a}] accuracy = {acc_a:.4f}")
    print(f"[{name_b}] accuracy = {acc_b:.4f}")
    print(f"[stats] {name_a} wrong, {name_b} right: {len(a_wrong_b_right)}")
    print(f"[stats] {name_b} wrong, {name_a} right: {len(b_wrong_a_right)}")
    print(f"[stats] оба ошиблись:                {len(both_wrong)}")
    print(f"[stats] оба угадали:                 {len(both_right)}")

    reports_dir = ensure_dir(args.reports_dir)

    X_flat = X.reshape(len(X), -1)

    # Сетка 1: A ошиблась, B угадала. «Истина» = y_true, «pred» = pred_a
    # чтобы подпись отражала именно ошибку A.
    out_a = reports_dir / f"{args.tag}_{name_a}_wrong_{name_b}_right.png"
    plot_misclassified(
        X_flat,
        y_true, pred_a, probs_a,
        out_a, max_items=args.max_items,
    )
    print(f"[save] {out_a}")

    out_b = reports_dir / f"{args.tag}_{name_b}_wrong_{name_a}_right.png"
    plot_misclassified(
        X_flat,
        y_true, pred_b, probs_b,
        out_b, max_items=args.max_items,
    )
    print(f"[save] {out_b}")

    # Разбивка по классам: какой класс чаще «спасает» вторая модель
    per_class = []
    for c in range(10):
        mask_true_c = y_true == c
        rescued = int((~correct_a & correct_b & mask_true_c).sum())
        lost = int((correct_a & ~correct_b & mask_true_c).sum())
        total_c = int(mask_true_c.sum())
        per_class.append({
            "digit": c,
            "total": total_c,
            "a_wrong_b_right": rescued,
            "b_wrong_a_right": lost,
        })

    analysis = {
        "split": args.split,
        "n_samples": int(len(X)),
        "model_a": {"name": name_a, "accuracy": acc_a},
        "model_b": {"name": name_b, "accuracy": acc_b},
        "counts": {
            "a_wrong_b_right": int(len(a_wrong_b_right)),
            "b_wrong_a_right": int(len(b_wrong_a_right)),
            "both_wrong": int(len(both_wrong)),
            "both_right": int(len(both_right)),
        },
        "per_class": per_class,
    }

    json_path = reports_dir / f"{args.tag}_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"[save] {json_path}")


if __name__ == "__main__":
    main()
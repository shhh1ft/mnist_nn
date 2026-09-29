"""CLI оценки модели на val/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data_loader import load_mnist
from .losses import SoftmaxCrossEntropy
from .metrics import classification_report, format_report
from .model import build_model_from_config
from .train import reshape_for_model
from .utils import ensure_dir, softmax
from .visualize import plot_confusion_matrix, plot_misclassified


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Оценка модели на MNIST")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--reports-dir", type=str, default="reports")
    p.add_argument("--tag", type=str, default="eval")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Модель не найдена: {model_path}")

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    model = build_model_from_config(config, seed=config.get("seed", 42))
    model.load(model_path)
    print(f"[evaluate] загружена модель {model_path}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(
        args.data_dir,
        val_size=config["data"].get("val_size", 0.1),
        seed=config.get("seed", 42),
    )

    model_type = config["model"].get("type", "mlp")
    X = X_val if args.split == "val" else X_test
    y = y_val if args.split == "val" else y_test
    X = reshape_for_model(X, model_type)

    logits = model.forward(X, training=False)
    loss_fn = SoftmaxCrossEntropy()
    loss = loss_fn.forward(logits, y)
    probs = softmax(logits, axis=1)

    y_true = y.argmax(axis=1)
    y_pred = probs.argmax(axis=1)

    report = classification_report(y_true, y_pred, num_classes=10)
    report["loss"] = float(loss)
    report["split"] = args.split
    report["n_samples"] = int(len(X))

    print(f"[evaluate] split={args.split}  N={len(X)}")
    print(f"[evaluate] loss={loss:.4f}  accuracy={report['accuracy']:.4f}")
    print()
    print(format_report(report))

    reports_dir = ensure_dir(args.reports_dir)

    report_json = {k: v for k, v in report.items() if k != "confusion_matrix"}
    report_json["confusion_matrix"] = report["confusion_matrix"].tolist()

    metrics_path = reports_dir / f"{args.tag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)
    print(f"[save] {metrics_path}")

    if not args.no_plot:
        cm_path = reports_dir / f"{args.tag}_confusion_matrix.png"
        plot_confusion_matrix(
            report["confusion_matrix"], cm_path,
            class_names=[str(i) for i in range(10)],
            normalize=True,
        )
        print(f"[save] {cm_path}")

        # X может быть (N, 1, 28, 28) — приводим к (N, 784) для plot
        X_flat = X.reshape(len(X), -1)
        wrong_path = reports_dir / f"{args.tag}_misclassified.png"
        plot_misclassified(
            X_flat, y_true, y_pred, probs,
            wrong_path, max_items=25,
        )
        print(f"[save] {wrong_path}")


if __name__ == "__main__":
    main()
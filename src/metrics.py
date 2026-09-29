"""Метрики качества.

Неделя 3: accuracy.
Неделя 5: confusion matrix, precision/recall/F1, classification_report.

Все функции принимают:
- y_pred: либо логиты/вероятности формы (N, C), либо метки (N,);
- y_true: либо one-hot (N, C), либо метки (N,).
"""

from __future__ import annotations

import numpy as np


def _to_labels(y: np.ndarray) -> np.ndarray:
    """Приводит (N, C) → (N,) через argmax; (N,) оставляет как есть."""
    y = np.asarray(y)
    if y.ndim > 1:
        return y.argmax(axis=1).astype(np.int64)
    return y.astype(np.int64)


def accuracy(logits_or_probs: np.ndarray, targets: np.ndarray) -> float:
    """Доля правильных предсказаний."""
    y_pred = _to_labels(logits_or_probs)
    y_true = _to_labels(targets)
    if len(y_pred) != len(y_true):
        raise ValueError("pred и true разной длины")
    return float((y_pred == y_true).mean())


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
) -> np.ndarray:
    """Матрица ошибок C×C: cm[i, j] = #(true=i, pred=j)."""
    y_true = _to_labels(y_true)
    y_pred = _to_labels(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true и y_pred разной длины")

    if num_classes is None:
        num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0))) + 1

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (precision, recall, f1, confusion_matrix) по классам.

    precision_k = TP_k / (TP_k + FP_k)
    recall_k    = TP_k / (TP_k + FN_k)
    f1_k        = 2·P·R / (P + R)
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return precision, recall, f1, cm


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Полный отчёт: per-class метрики, macro/weighted averages, CM."""
    precision, recall, f1, cm = precision_recall_f1(y_true, y_pred, num_classes)

    C = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(C)]

    support = cm.sum(axis=1).astype(np.int64)
    total = int(support.sum())
    accuracy_val = float(np.diag(cm).sum() / total) if total > 0 else 0.0

    macro = {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }
    weights = support.astype(np.float64) / max(total, 1)
    weighted = {
        "precision": float((precision * weights).sum()),
        "recall": float((recall * weights).sum()),
        "f1": float((f1 * weights).sum()),
    }

    return {
        "accuracy": accuracy_val,
        "per_class": {
            "names": class_names,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        },
        "macro": macro,
        "weighted": weighted,
        "confusion_matrix": cm,
    }


def format_report(report: dict) -> str:
    """Человекочитаемая таблица отчёта."""
    lines: list[str] = []
    pc = report["per_class"]
    header = f"{'class':>6} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, name in enumerate(pc["names"]):
        lines.append(
            f"{name:>6} "
            f"{pc['precision'][i]:>10.4f} "
            f"{pc['recall'][i]:>10.4f} "
            f"{pc['f1'][i]:>10.4f} "
            f"{pc['support'][i]:>10d}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'macro':>6} "
        f"{report['macro']['precision']:>10.4f} "
        f"{report['macro']['recall']:>10.4f} "
        f"{report['macro']['f1']:>10.4f}"
    )
    lines.append(
        f"{'weight':>6} "
        f"{report['weighted']['precision']:>10.4f} "
        f"{report['weighted']['recall']:>10.4f} "
        f"{report['weighted']['f1']:>10.4f}"
    )
    lines.append("")
    lines.append(f"accuracy = {report['accuracy']:.4f}")
    return "\n".join(lines)
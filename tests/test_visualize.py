"""Тесты визуализации (неделя 5).

Проверяют, что функции создают непустые PNG-файлы без ошибок.
Не проверяют пиксели — только факт сохранения и корректный путь.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.metrics import confusion_matrix
from src.visualize import (
    plot_confusion_matrix,
    plot_history,
    plot_misclassified,
)


def test_plot_history_creates_file(tmp_path: Path):
    history = {
        "train_loss": [0.5, 0.3, 0.2],
        "train_acc": [0.85, 0.91, 0.94],
        "val_loss": [0.4, 0.35, 0.28],
        "val_acc": [0.88, 0.90, 0.93],
        "lr": [0.1, 0.1, 0.05],
    }
    out = tmp_path / "history.png"
    plot_history(history, out)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_history_without_lr(tmp_path: Path):
    history = {
        "train_loss": [0.5, 0.3],
        "train_acc": [0.85, 0.91],
        "val_loss": [0.4, 0.35],
        "val_acc": [0.88, 0.90],
    }
    out = tmp_path / "history.png"
    plot_history(history, out)
    assert out.exists()


def test_plot_confusion_matrix_creates_file(tmp_path: Path):
    y_true = np.array([0, 1, 2] * 10)
    y_pred = np.array([0, 1, 1] * 10)
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    out = tmp_path / "cm.png"
    plot_confusion_matrix(cm, out, class_names=["a", "b", "c"])
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_misclassified_creates_file(tmp_path: Path):
    rng = np.random.default_rng(0)
    X = rng.random((20, 784)).astype(np.float32)
    y_true = np.array([0, 1] * 10)
    y_pred = np.array([1, 0] * 10)
    probs = rng.random((20, 3)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)

    out = tmp_path / "wrong.png"
    plot_misclassified(X, y_true, y_pred, probs, out, max_items=8)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_misclassified_no_errors(tmp_path: Path):
    """Если ошибок нет, функция не падает и создаёт placeholder."""
    rng = np.random.default_rng(1)
    X = rng.random((5, 784)).astype(np.float32)
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    probs = rng.random((5, 5)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)

    out = tmp_path / "no_errors.png"
    plot_misclassified(X, y_true, y_pred, probs, out)
    assert out.exists()
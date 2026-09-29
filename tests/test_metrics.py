"""Тесты метрик (неделя 5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    format_report,
    precision_recall_f1,
)


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------

def test_accuracy_perfect():
    y = np.array([0, 1, 2, 3])
    logits = np.eye(4, dtype=np.float32)
    assert accuracy(logits, y) == 1.0


def test_accuracy_half():
    y = np.array([0, 1, 2, 3])
    logits = np.array([
        [1, 0, 0, 0],  # 0 ✓
        [1, 0, 0, 0],  # 0 ✗ (true=1)
        [0, 0, 1, 0],  # 2 ✓
        [0, 0, 1, 0],  # 2 ✗ (true=3)
    ], dtype=np.float32)
    assert accuracy(logits, y) == 0.5


def test_accuracy_accepts_labels():
    y = np.array([0, 1, 2])
    pred_labels = np.array([0, 2, 2])
    assert accuracy(pred_labels, y) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# confusion_matrix
# ---------------------------------------------------------------------------

def test_confusion_matrix_counts():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    # true=0: pred 0 (1), pred 1 (1)
    # true=1: pred 1 (2)
    # true=2: pred 2 (1), pred 0 (1)
    expected = np.array([
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
    ], dtype=np.int64)
    assert np.array_equal(cm, expected)


def test_confusion_matrix_sum_equals_n():
    y_true = np.array([0, 1, 2, 3, 0])
    y_pred = np.array([0, 1, 1, 3, 0])
    cm = confusion_matrix(y_true, y_pred, num_classes=4)
    assert cm.sum() == len(y_true)


def test_confusion_matrix_from_one_hot():
    y_true = np.eye(3, dtype=np.float32)
    y_pred = np.eye(3, dtype=np.float32)
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    assert np.array_equal(cm, np.eye(3, dtype=np.int64) * 1)


# ---------------------------------------------------------------------------
# precision / recall / F1
# ---------------------------------------------------------------------------

def test_precision_recall_f1_perfect():
    y = np.array([0, 1, 2, 3])
    pred = np.array([0, 1, 2, 3])
    p, r, f, cm = precision_recall_f1(y, pred, num_classes=4)
    assert np.allclose(p, 1.0)
    assert np.allclose(r, 1.0)
    assert np.allclose(f, 1.0)


def test_precision_recall_hand_calc():
    """Бинарный пример: TP=2, FP=1, FN=1 → P=2/3, R=2/3, F1=2/3."""
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 1])
    p, r, f, cm = precision_recall_f1(y_true, y_pred, num_classes=2)
    # класс 1: TP=1 (индекс 2), FP=1 (индекс 4), FN=1 (индекс 3)
    # P = 1/(1+1) = 0.5; R = 1/(1+1) = 0.5; F1 = 0.5
    assert p[1] == pytest.approx(0.5)
    assert r[1] == pytest.approx(0.5)
    assert f[1] == pytest.approx(0.5)


def test_f1_zero_when_no_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])  # класс 1 никогда не предсказан
    p, r, f, _ = precision_recall_f1(y_true, y_pred, num_classes=2)
    assert p[1] == pytest.approx(0.0)
    assert r[1] == pytest.approx(0.0)
    assert f[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# classification_report
# ---------------------------------------------------------------------------

def test_classification_report_structure():
    y_true = np.array([0, 1, 2, 3] * 25)
    y_pred = np.array([0, 1, 2, 3] * 25)
    rep = classification_report(y_true, y_pred, num_classes=4)

    assert rep["accuracy"] == 1.0
    assert set(rep.keys()) >= {
        "accuracy", "per_class", "macro", "weighted", "confusion_matrix"
    }
    assert rep["confusion_matrix"].shape == (4, 4)
    assert len(rep["per_class"]["precision"]) == 4
    assert rep["macro"]["f1"] == pytest.approx(1.0)
    assert rep["weighted"]["f1"] == pytest.approx(1.0)


def test_classification_report_weighted_avg():
    """Взвешенное среднее учитывает support.

    Данные:
        y_true = [0]*8 + [1]*2
        y_pred = [0]*8 + [0]*2

    Confusion matrix:
        cm[0,0]=8  cm[0,1]=0
        cm[1,0]=2  cm[1,1]=0

    Класс 0: TP=8, FP=2, FN=0
        precision = 8/10 = 0.8
        recall    = 8/8  = 1.0
        f1        = 2·0.8·1.0/(0.8+1.0) = 8/9 ≈ 0.8889
    Класс 1: TP=0, FP=0, FN=2
        precision = recall = f1 = 0

    macro f1    = (8/9 + 0) / 2 = 4/9 ≈ 0.4444
    weighted f1 = (8/9)·(8/10) + 0·(2/10) = 64/90 ≈ 0.7111
    """
    y_true = np.array([0] * 8 + [1] * 2)
    y_pred = np.array([0] * 8 + [0] * 2)
    rep = classification_report(y_true, y_pred, num_classes=2)

    assert rep["macro"]["f1"] == pytest.approx(4 / 9, abs=1e-6)
    assert rep["weighted"]["f1"] == pytest.approx(64 / 90, abs=1e-6)

    # Дополнительно: точность на этих данных = 8/10
    assert rep["accuracy"] == pytest.approx(0.8, abs=1e-6)

    # per-class проверки для класса 0
    assert rep["per_class"]["precision"][0] == pytest.approx(0.8, abs=1e-6)
    assert rep["per_class"]["recall"][0] == pytest.approx(1.0, abs=1e-6)
    assert rep["per_class"]["f1"][0] == pytest.approx(8 / 9, abs=1e-6)

    # класс 1 никогда не предсказан → нули
    assert rep["per_class"]["precision"][1] == pytest.approx(0.0, abs=1e-6)
    assert rep["per_class"]["recall"][1] == pytest.approx(0.0, abs=1e-6)
    assert rep["per_class"]["f1"][1] == pytest.approx(0.0, abs=1e-6)


def test_format_report_contains_key_lines():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    rep = classification_report(y_true, y_pred, num_classes=3)
    text = format_report(rep)
    assert "precision" in text
    assert "recall" in text
    assert "f1" in text
    assert "macro" in text
    assert "accuracy" in text
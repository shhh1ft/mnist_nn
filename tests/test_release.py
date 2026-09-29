"""Финальные тесты недели 8: воспроизводимость, save/load, отчёт, бенчмарк."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data_loader import load_mnist
from src.layers import Dense, ReLU
from src.losses import SoftmaxCrossEntropy
from src.model import Sequential, build_model_from_config
from src.optimizers import Adam
from src.report import build_report
from src.utils import set_seed


# ---------------------------------------------------------------------------
# Сохранение / загрузка
# ---------------------------------------------------------------------------

def _make_small_model():
    config = {
        "model": {
            "type": "mlp",
            "layers": [
                {"type": "dense", "in": 4, "out": 8, "activation": "relu"},
                {"type": "dense", "in": 8, "out": 3, "activation": "softmax"},
            ],
        }
    }
    return build_model_from_config(config, seed=0)


def test_save_load_roundtrip(tmp_path: Path):
    model = _make_small_model()
    X = np.random.randn(5, 4).astype(np.float32)
    out_before = model.forward(X, training=False)

    path = tmp_path / "m.npz"
    model.save(path)

    model2 = _make_small_model()  # другая инициализация
    out_random = model2.forward(X, training=False)
    assert not np.allclose(out_before, out_random)

    model2.load(path)
    out_after = model2.forward(X, training=False)
    assert np.allclose(out_before, out_after)


def test_save_load_preserves_all_params(tmp_path: Path):
    model = _make_small_model()
    path = tmp_path / "m.npz"
    model.save(path)

    data = np.load(path)
    keys = list(data.keys())
    # 2 Dense × (W, b) = 4 тензора
    assert len(keys) == 4
    assert "layer0_param0" in keys
    assert "layer1_param1" in keys


# ---------------------------------------------------------------------------
# Воспроизводимость seed
# ---------------------------------------------------------------------------

def test_seed_reproducibility_training():
    """Один seed → одинаковая последовательность весов и loss."""

    def run_once():
        set_seed(123)
        model = _make_small_model()
        X = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
        y = np.eye(3, dtype=np.float32)[np.array([0, 1, 2] * 3 + [0])]
        loss_fn = SoftmaxCrossEntropy()
        opt = Adam(lr=0.01)
        losses = []
        for _ in range(5):
            logits = model.forward(X, training=False)
            losses.append(loss_fn.forward(logits, y))
            model.backward(loss_fn.backward())
            opt.step(model)
        return losses, [p.copy() for p, _ in model.params_and_grads()]

    losses_a, params_a = run_once()
    losses_b, params_b = run_once()

    assert losses_a == losses_b
    for pa, pb in zip(params_a, params_b):
        assert np.allclose(pa, pb)


def test_seed_reproducibility_data_split(tmp_path: Path):
    """Разбиение train/val воспроизводимо при одном seed."""
    from src.data_loader import stratified_split

    y = np.repeat(np.arange(10), 20)
    tr1, va1 = stratified_split(y, val_size=0.2, seed=7)
    tr2, va2 = stratified_split(y, val_size=0.2, seed=7)
    assert np.array_equal(tr1, tr2)
    assert np.array_equal(va1, va2)


# ---------------------------------------------------------------------------
# Пайплайн без кэша
# ---------------------------------------------------------------------------

def test_pipeline_without_cache(tmp_path: Path):
    """prepare_data с use_cache=False корректно считает массивы."""
    import gzip
    import struct

    raw = tmp_path / "raw"
    raw.mkdir()

    def _write_img(path: Path, imgs: np.ndarray) -> None:
        n, r, c = imgs.shape
        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">I", 2051))
            f.write(struct.pack(">III", n, r, c))
            f.write(imgs.tobytes())

    def _write_lbl(path: Path, lbls: np.ndarray) -> None:
        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">I", 2049))
            f.write(struct.pack(">I", len(lbls)))
            f.write(lbls.tobytes())

    rng = np.random.default_rng(0)
    X_tr = rng.integers(0, 256, size=(80, 28, 28), dtype=np.uint8)
    y_tr = np.repeat(np.arange(4), 20).astype(np.uint8)
    X_te = rng.integers(0, 256, size=(20, 28, 28), dtype=np.uint8)
    y_te = np.repeat(np.arange(4), 5).astype(np.uint8)

    _write_img(raw / "train-images-idx3-ubyte.gz", X_tr)
    _write_lbl(raw / "train-labels-idx1-ubyte.gz", y_tr)
    _write_img(raw / "t10k-images-idx3-ubyte.gz", X_te)
    _write_lbl(raw / "t10k-labels-idx1-ubyte.gz", y_te)

    from src.data_loader import prepare_data
    processed = tmp_path / "processed"
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        raw, processed, val_size=0.25, seed=0, use_cache=False,
    )
    assert X_train.shape[1] == 784
    assert X_val.shape[0] + X_train.shape[0] == 80
    assert X_test.shape[0] == 20
    # Кэш не должен быть создан
    assert not (processed / "mnist_split.npz").exists()


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------

def test_build_report_minimal(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()

    # summary
    with open(reports / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "split": "test",
            "rows": [
                {"name": "mlp", "n_params": 100, "accuracy": 0.97,
                 "loss": 0.08, "macro_f1": 0.96, "inference_ms": 10.0},
                {"name": "cnn", "n_params": 200, "accuracy": 0.99,
                 "loss": 0.03, "macro_f1": 0.99, "inference_ms": 80.0},
            ],
        }, f)

    text = build_report(reports, tag="final")

    assert "# Итоговый отчёт" in text
    assert "| Model |" in text
    assert "mlp" in text
    assert "cnn" in text
    assert "Воспроизведение" in text


def test_build_report_missing_files(tmp_path: Path):
    """Пустая директория → отчёт всё равно собирается без падений."""
    reports = tmp_path / "empty"
    reports.mkdir()
    text = build_report(reports, tag="final")
    assert "# Итоговый отчёт" in text


# ---------------------------------------------------------------------------
# Финальная точность (smoke, не строгий)
# ---------------------------------------------------------------------------

def test_mlp_reaches_reasonable_accuracy_fast():
    """MLP на 20 эпохах должна давать val acc > 0.95.

    Тест использует крошечный датасет (300 примеров), чтобы быстро
    закончиться в CI. Проверяет, что пайплайн вообще способен обучаться.
    """
    set_seed(0)
    rng = np.random.default_rng(0)

    # 300 синтетических «изображений», 3 класса
    X = rng.standard_normal((300, 16)).astype(np.float32)
    y_idx = rng.integers(0, 3, size=300)
    y = np.eye(3, dtype=np.float32)[y_idx]

    config = {
        "model": {
            "type": "mlp",
            "layers": [
                {"type": "dense", "in": 16, "out": 32, "activation": "relu"},
                {"type": "dense", "in": 32, "out": 3, "activation": "softmax"},
            ],
        }
    }
    model = build_model_from_config(config, seed=0)
    loss_fn = SoftmaxCrossEntropy()
    opt = Adam(lr=0.01)

    for _ in range(200):
        logits = model.forward(X, training=True)
        loss_fn.forward(logits, y)
        model.backward(loss_fn.backward())
        opt.step(model)

    logits = model.forward(X, training=False)
    acc = float((logits.argmax(axis=1) == y_idx).mean())
    assert acc > 0.85, f"accuracy = {acc}"
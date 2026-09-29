"""Тесты вспомогательных функций сравнения моделей (неделя 7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.compare import (
    count_parameters,
    evaluate_model,
    format_comparison_table,
    parse_entry,
)
from src.losses import SoftmaxCrossEntropy
from src.model import Sequential, build_model_from_config
from src.layers import Dense


# ---------------------------------------------------------------------------
# parse_entry
# ---------------------------------------------------------------------------

def test_parse_entry_valid():
    name, model, config = parse_entry("mlp=models/a.npz,configs/a.json")
    assert name == "mlp"
    assert model == Path("models/a.npz")
    assert config == Path("configs/a.json")


def test_parse_entry_with_spaces():
    name, model, config = parse_entry(
        " cnn = models/x.npz , configs/y.json "
    )
    assert name == "cnn"
    assert model == Path("models/x.npz")
    assert config == Path("configs/y.json")


def test_parse_entry_invalid():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        parse_entry("mlp_only_path")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_entry("=models/a.npz,configs/a.json")


# ---------------------------------------------------------------------------
# count_parameters
# ---------------------------------------------------------------------------

def test_count_parameters_dense():
    model = Sequential([Dense(10, 5, seed=0), Dense(5, 3, seed=1)])
    # 10·5 + 5 + 5·3 + 3 = 50 + 5 + 15 + 3 = 73
    assert count_parameters(model) == 73


def test_count_parameters_from_config():
    config = {
        "model": {
            "type": "mlp",
            "layers": [
                {"type": "dense", "in": 4, "out": 3, "activation": "relu"},
                {"type": "dense", "in": 3, "out": 2, "activation": "softmax"},
            ],
        }
    }
    model = build_model_from_config(config, seed=0)
    # 4·3 + 3 + 3·2 + 2 = 12 + 3 + 6 + 2 = 23
    assert count_parameters(model) == 23


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

def test_evaluate_model_perfect():
    config = {
        "model": {
            "type": "mlp",
            "layers": [
                {"type": "dense", "in": 4, "out": 3, "activation": "softmax"},
            ],
        }
    }
    model = build_model_from_config(config, seed=0)
    # Ставим веса так, чтобы argmax совпал с истиной
    model.layers[0].W = np.eye(4, 3, dtype=np.float32)
    model.layers[0].b = np.zeros(3, dtype=np.float32)

    X = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ], dtype=np.float32)
    y = np.eye(3, dtype=np.float32)

    metrics = evaluate_model(model, config, X, y, SoftmaxCrossEntropy())

    assert metrics["accuracy"] == 1.0
    assert metrics["n_params"] == 4 * 3 + 3
    assert metrics["y_true"].tolist() == [0, 1, 2]
    assert metrics["y_pred"].tolist() == [0, 1, 2]
    assert metrics["confusion_matrix"].shape == (10, 10)


def test_evaluate_model_returns_probs_summing_to_one():
    config = {
        "model": {
            "type": "mlp",
            "layers": [
                {"type": "dense", "in": 3, "out": 3, "activation": "softmax"},
            ],
        }
    }
    model = build_model_from_config(config, seed=0)
    X = np.random.randn(5, 3).astype(np.float32)
    y = np.eye(3, dtype=np.float32)[np.array([0, 1, 2, 0, 1])]

    metrics = evaluate_model(model, config, X, y, SoftmaxCrossEntropy())

    row_sums = metrics["probs"].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# format_comparison_table
# ---------------------------------------------------------------------------

def test_format_comparison_table_header_and_rows():
    rows = [
        {"name": "mlp", "n_params": 101770, "accuracy": 0.9778,
         "loss": 0.08, "macro_f1": 0.977, "inference_ms": 45.0},
        {"name": "cnn", "n_params": 122000, "accuracy": 0.9910,
         "loss": 0.03, "macro_f1": 0.991, "inference_ms": 820.0},
    ]
    md = format_comparison_table(rows)

    assert "| Model |" in md
    assert "| Params |" in md
    assert "mlp" in md
    assert "cnn" in md
    assert "0.9778" in md
    assert "0.9910" in md
    assert "101,770" in md
    assert "122,000" in md


def test_format_comparison_table_empty():
    md = format_comparison_table([])
    # Должен остаться хотя бы заголовок
    assert "| Model |" in md
    assert md.count("\n") >= 2


def test_format_comparison_table_single_row():
    rows = [
        {"name": "solo", "n_params": 100, "accuracy": 0.5,
         "loss": 1.0, "macro_f1": 0.4, "inference_ms": 10.0},
    ]
    md = format_comparison_table(rows)
    assert md.count("| solo") == 1
    assert "0.5000" in md
    assert "1.0000" in md
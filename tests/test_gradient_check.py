"""Численная проверка градиентов (недели 2–3).

Все вычисления ведутся в float64, иначе конечные разности теряют точность
на float32. Пороги подобраны под центральную разность с eps=1e-6:
- для линейных функций (Dense) — практически машинная точность;
- для Softmax-CE и полной MLP — ~1e-9.

relative_error считается через L2-норму: это устойчиво к отдельным
элементам с истинным градиентом около нуля.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.layers import Dense, ReLU
from src.losses import SoftmaxCrossEntropy
from src.model import Sequential
from src.optimizers import SGD


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def numerical_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Центральная конечная разность по каждому элементу x.

    x модифицируется in-place; f — callable без аргументов, читающий x.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx].copy()

        x[idx] = orig + eps
        f_plus = f()

        x[idx] = orig - eps
        f_minus = f()

        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Относительная ошибка по L2-норме.

    Устойчиво к элементам с истинным градиентом около нуля:
    не «взрывается», как поэлементный max(|Δ|/|a|).
    """
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(a) + np.linalg.norm(b) + 1e-12
    return float(num / den)


def _cast_layer_to_float64(layer) -> None:
    """Приводит параметры слоя к float64 in-place (для gradient check)."""
    if hasattr(layer, "W"):
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)
        layer.dW = np.zeros_like(layer.W)
        layer.db = np.zeros_like(layer.b)


def _cast_model_to_float64(model: Sequential) -> None:
    for layer in model.layers:
        _cast_layer_to_float64(layer)


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------

def test_gradient_dense_W_and_b():
    rng = np.random.default_rng(0)
    layer = Dense(5, 3, seed=0)
    _cast_layer_to_float64(layer)

    X = rng.standard_normal((4, 5)).astype(np.float64)
    dZ = rng.standard_normal((4, 3)).astype(np.float64)

    layer.forward(X)
    layer.backward(dZ)

    def f_W():
        Z = X @ layer.W + layer.b
        return float(np.sum(dZ * Z))

    err_W = relative_error(numerical_gradient(f_W, layer.W), layer.dW)
    assert err_W < 1e-6, f"dW: relative error = {err_W}"

    def f_b():
        Z = X @ layer.W + layer.b
        return float(np.sum(dZ * Z))

    err_b = relative_error(numerical_gradient(f_b, layer.b), layer.db)
    assert err_b < 1e-6, f"db: relative error = {err_b}"


# ---------------------------------------------------------------------------
# Softmax + Cross-Entropy
# ---------------------------------------------------------------------------

def test_gradient_softmax_cross_entropy():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((5, 4)).astype(np.float64)
    targets = np.eye(4, dtype=np.float64)[rng.integers(0, 4, size=5)]

    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(logits, targets)
    dLogits = loss_fn.backward()

    def f():
        return loss_fn.forward(logits, targets)

    num = numerical_gradient(f, logits)
    err = relative_error(num, dLogits)
    assert err < 1e-6, f"dLogits: relative error = {err}"


# ---------------------------------------------------------------------------
# Полная MLP
# ---------------------------------------------------------------------------

def test_gradient_full_mlp():
    rng = np.random.default_rng(2)
    model = Sequential([
        Dense(6, 5, init="he", seed=10),
        ReLU(),
        Dense(5, 3, init="xavier", seed=20),
    ])
    _cast_model_to_float64(model)

    X = rng.standard_normal((4, 6)).astype(np.float64)
    y = np.eye(3, dtype=np.float64)[rng.integers(0, 3, size=4)]
    loss_fn = SoftmaxCrossEntropy()

    logits = model.forward(X, training=False)
    loss_fn.forward(logits, y)
    model.backward(loss_fn.backward())

    first = model.layers[0]
    second = model.layers[2]

    def f():
        out = model.forward(X, training=False)
        return loss_fn.forward(out, y)

    err1 = relative_error(numerical_gradient(f, first.W), first.dW)
    assert err1 < 1e-6, f"W0: relative error = {err1}"

    err1b = relative_error(numerical_gradient(f, first.b), first.db)
    assert err1b < 1e-6, f"b0: relative error = {err1b}"

    err2 = relative_error(numerical_gradient(f, second.W), second.dW)
    assert err2 < 1e-6, f"W2: relative error = {err2}"

    err2b = relative_error(numerical_gradient(f, second.b), second.db)
    assert err2b < 1e-6, f"b2: relative error = {err2b}"


# ---------------------------------------------------------------------------
# Переобучение маленькой выборки (smoke-тест цикла обучения)
# ---------------------------------------------------------------------------

def test_overfit_small_batch():
    """Маленькая MLP должна переобучиться на 10 примерах."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4)).astype(np.float32)
    y_idx = rng.integers(0, 3, size=10)
    y = np.eye(3, dtype=np.float32)[y_idx]

    model = Sequential([
        Dense(4, 16, init="he", seed=1),
        ReLU(),
        Dense(16, 3, init="xavier", seed=2),
    ])
    loss_fn = SoftmaxCrossEntropy()
    opt = SGD(lr=0.1, momentum=0.9)

    for _ in range(300):
        logits = model.forward(X, training=False)
        loss_fn.forward(logits, y)
        model.backward(loss_fn.backward())
        opt.step(model)

    logits = model.forward(X, training=False)
    acc = float((logits.argmax(axis=1) == y_idx).mean())
    assert acc == 1.0, f"не удалось переобучиться: accuracy = {acc}"
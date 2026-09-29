"""Тесты свёрточных слоёв (неделя 6)."""

from __future__ import annotations

import numpy as np
import pytest

from src.conv import Conv2D, Flatten, MaxPool2D, col2im, im2col


# ---------------------------------------------------------------------------
# im2col / col2im
# ---------------------------------------------------------------------------

def test_im2col_shape_no_pad():
    X = np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4)
    cols = im2col(X, kH=2, kW=2, stride=1, pad=0)
    # H_out = W_out = 3 → cols (1, 4, 9)
    assert cols.shape == (1, 4, 9)


def test_im2col_values_manual():
    # X = [[1, 2], [3, 4]] → единственный патч [1, 2, 3, 4]
    X = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    cols = im2col(X, kH=2, kW=2, stride=1, pad=0)
    assert cols.shape == (1, 4, 1)
    assert np.array_equal(cols[0, :, 0], [1, 2, 3, 4])


def test_im2col_with_padding():
    X = np.array([[[[1]]]], dtype=np.float32)
    cols = im2col(X, kH=3, kW=3, stride=1, pad=1)
    # H_out = (1 + 2 − 3)/1 + 1 = 1
    assert cols.shape == (1, 9, 1)
    # Углы нули, центр = 1
    patch = cols[0, :, 0]
    assert patch[4] == 1.0
    assert patch[0] == 0.0


def test_col2im_roundtrip_identity():
    """col2im(im2col(X)) = X при отсутствии перекрытия патчей.

    При stride == kernel и pad=0 патчи не пересекаются.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, 3, 4, 6)).astype(np.float64)
    cols = im2col(X, kH=2, kW=2, stride=2, pad=0)
    X_back = col2im(cols, X.shape, kH=2, kW=2, stride=2, pad=0)
    assert np.allclose(X_back, X)


def test_col2im_with_stride_lt_kernel():
    """При stride < kernel вклады патчей суммируются."""
    X = np.ones((1, 1, 3, 3), dtype=np.float64)
    cols = im2col(X, kH=2, kW=2, stride=1, pad=0)
    # Считаем сумму по всем патчам через col2im от единичной матрицы
    X_back = col2im(cols, X.shape, kH=2, kW=2, stride=1, pad=0)
    # Каждый пиксель входит в 1, 2 или 4 патча
    expected = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ], dtype=np.float64).reshape(1, 1, 3, 3)
    assert np.allclose(X_back, expected)


# ---------------------------------------------------------------------------
# Conv2D
# ---------------------------------------------------------------------------

def test_conv2d_forward_shape():
    layer = Conv2D(1, 8, kernel_size=3, stride=1, padding=0, seed=0)
    X = np.random.randn(4, 1, 28, 28).astype(np.float32)
    out = layer.forward(X)
    # H_out = W_out = 26
    assert out.shape == (4, 8, 26, 26)


def test_conv2d_forward_manual():
    layer = Conv2D(1, 1, kernel_size=2, stride=1, padding=0, seed=0)
    layer.W = np.array([[[[1, 0], [0, -1]]]], dtype=np.float32)
    layer.b = np.zeros(1, dtype=np.float32)

    X = np.array([[[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]]], dtype=np.float32)
    out = layer.forward(X)
    # Свёртка ядром [[1,0],[0,-1]] на 3×3 даёт 2×2 из −4
    expected = np.array([[[[-4, -4],
                           [-4, -4]]]], dtype=np.float32)
    assert np.allclose(out, expected)


def test_conv2d_forward_padding():
    layer = Conv2D(1, 1, kernel_size=3, stride=1, padding=1, seed=0)
    X = np.random.randn(2, 1, 5, 5).astype(np.float32)
    out = layer.forward(X)
    # padding=1 сохраняет размер
    assert out.shape == (2, 1, 5, 5)


def test_conv2d_backward_shapes():
    layer = Conv2D(2, 3, kernel_size=3, stride=1, padding=0, seed=0)
    X = np.random.randn(2, 2, 6, 6).astype(np.float32)
    out = layer.forward(X)
    dOut = np.random.randn(*out.shape).astype(np.float32)
    dX = layer.backward(dOut)

    assert dX.shape == X.shape
    assert layer.dW.shape == layer.W.shape
    assert layer.db.shape == layer.b.shape


def test_conv2d_invalid_input():
    layer = Conv2D(1, 1, kernel_size=3, seed=0)
    with pytest.raises(ValueError):
        layer.forward(np.random.randn(4, 28, 28).astype(np.float32))


def test_conv2d_params_and_grads():
    layer = Conv2D(2, 3, kernel_size=3, seed=0)
    pairs = layer.params_and_grads()
    assert len(pairs) == 2
    assert pairs[0][0].shape == (3, 2, 3, 3)
    assert pairs[1][0].shape == (3,)


# ---------------------------------------------------------------------------
# MaxPool2D
# ---------------------------------------------------------------------------

def test_maxpool2d_forward_shape():
    layer = MaxPool2D(kernel_size=2, stride=2)
    X = np.random.randn(4, 3, 28, 28).astype(np.float32)
    out = layer.forward(X)
    assert out.shape == (4, 3, 14, 14)


def test_maxpool2d_forward_manual():
    layer = MaxPool2D(kernel_size=2, stride=2)
    X = np.array([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]], dtype=np.float32)
    out = layer.forward(X)
    expected = np.array([[[[6, 8],
                           [14, 16]]]], dtype=np.float32)
    assert np.array_equal(out, expected)


def test_maxpool2d_backward_routes_to_argmax():
    layer = MaxPool2D(kernel_size=2, stride=2)
    X = np.array([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]], dtype=np.float32)
    layer.forward(X)
    dOut = np.ones((1, 1, 2, 2), dtype=np.float32)
    dX = layer.backward(dOut)

    # Единица должна попасть только в позиции максимумов
    # Максимумы были: (1,1)=6, (1,3)=8, (3,1)=14, (3,3)=16
    expected = np.zeros_like(X)
    expected[0, 0, 1, 1] = 1.0
    expected[0, 0, 1, 3] = 1.0
    expected[0, 0, 3, 1] = 1.0
    expected[0, 0, 3, 3] = 1.0
    assert np.array_equal(dX, expected)


def test_maxpool2d_backward_grad_accumulates_over_batch():
    layer = MaxPool2D(kernel_size=2, stride=2)
    X = np.ones((3, 2, 4, 4), dtype=np.float32)
    layer.forward(X)
    dOut = np.full((3, 2, 2, 2), 1.0, dtype=np.float32)
    dX = layer.backward(dOut)
    # Все единицы, argmax=0 в каждом патче (все равны)
    assert dX.shape == X.shape
    # Сумма всех градиентов = сумма dOut
    assert dX.sum() == dOut.sum()


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

def test_flatten_forward():
    layer = Flatten()
    X = np.random.randn(2, 3, 4, 5).astype(np.float32)
    out = layer.forward(X)
    assert out.shape == (2, 60)


def test_flatten_backward_restores_shape():
    layer = Flatten()
    X = np.random.randn(2, 3, 4, 5).astype(np.float32)
    layer.forward(X)
    dA = np.random.randn(2, 60).astype(np.float32)
    dX = layer.backward(dA)
    assert dX.shape == X.shape


# ---------------------------------------------------------------------------
# Gradient check (float64)
# ---------------------------------------------------------------------------

def _cast_layer_to_float64(layer):
    if hasattr(layer, "W"):
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)
        layer.dW = np.zeros_like(layer.W)
        layer.db = np.zeros_like(layer.b)


def numerical_gradient(f, x, eps=1e-6):
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
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a, b):
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(a) + np.linalg.norm(b) + 1e-12
    return float(num / den)


def test_gradient_conv2d():
    rng = np.random.default_rng(0)
    layer = Conv2D(2, 3, kernel_size=3, stride=1, padding=1, seed=0)
    _cast_layer_to_float64(layer)

    X = rng.standard_normal((2, 2, 5, 5)).astype(np.float64)
    layer.forward(X)
    dOut = rng.standard_normal((2, 3, 5, 5)).astype(np.float64)
    layer.backward(dOut)

    def f_W():
        out = layer.forward(X)
        return float(np.sum(dOut * out))

    num_W = numerical_gradient(f_W, layer.W)
    err_W = relative_error(num_W, layer.dW)
    assert err_W < 1e-6, f"dW relative error = {err_W}"

    def f_b():
        out = layer.forward(X)
        return float(np.sum(dOut * out))

    num_b = numerical_gradient(f_b, layer.b)
    err_b = relative_error(num_b, layer.db)
    assert err_b < 1e-6, f"db relative error = {err_b}"


def test_gradient_conv2d_stride2():
    rng = np.random.default_rng(1)
    layer = Conv2D(1, 2, kernel_size=3, stride=2, padding=1, seed=0)
    _cast_layer_to_float64(layer)

    X = rng.standard_normal((2, 1, 6, 6)).astype(np.float64)
    layer.forward(X)
    dOut = rng.standard_normal(layer.forward(X).shape).astype(np.float64)
    layer.backward(dOut)

    def f():
        out = layer.forward(X)
        return float(np.sum(dOut * out))

    num_W = numerical_gradient(f, layer.W)
    err = relative_error(num_W, layer.dW)
    assert err < 1e-6, f"dW (stride=2) relative error = {err}"
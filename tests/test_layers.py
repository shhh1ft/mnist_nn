"""Тесты слоёв (неделя 2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.layers import Dense, Dropout, ReLU


# --------------------------- Dense ---------------------------------

def test_dense_forward_shape():
    layer = Dense(10, 5, seed=0)
    X = np.random.randn(3, 10).astype(np.float32)
    out = layer.forward(X)
    assert out.shape == (3, 5)


def test_dense_backward_shapes():
    layer = Dense(10, 5, seed=0)
    X = np.random.randn(3, 10).astype(np.float32)
    layer.forward(X)
    dZ = np.random.randn(3, 5).astype(np.float32)
    dX = layer.backward(dZ)
    assert dX.shape == (3, 10)
    assert layer.dW.shape == (10, 5)
    assert layer.db.shape == (5,)


def test_dense_he_init_std():
    layer = Dense(784, 128, init="he", seed=0)
    expected = np.sqrt(2.0 / 784)
    assert abs(float(layer.W.std()) - expected) < 0.005


def test_dense_xavier_init_std():
    layer = Dense(100, 50, init="xavier", seed=0)
    expected = np.sqrt(2.0 / (100 + 50))
    assert abs(float(layer.W.std()) - expected) < 0.005


def test_dense_bias_zero():
    layer = Dense(10, 5, seed=0)
    assert np.all(layer.b == 0.0)


def test_dense_invalid_init():
    with pytest.raises(ValueError):
        Dense(10, 5, init="unknown")


def test_dense_forward_manual():
    layer = Dense(3, 2, seed=0)
    layer.W = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    layer.b = np.array([0.5, -0.5], dtype=np.float32)
    X = np.array([[1, 1, 1]], dtype=np.float32)
    out = layer.forward(X)
    expected = np.array([[1 + 3 + 5 + 0.5, 2 + 4 + 6 - 0.5]], dtype=np.float32)
    assert np.allclose(out, expected)


# --------------------------- ReLU ----------------------------------

def test_relu_forward():
    layer = ReLU()
    Z = np.array([[-1, 0, 2], [3, -4, 5]], dtype=np.float32)
    out = layer.forward(Z)
    expected = np.array([[0, 0, 2], [3, 0, 5]], dtype=np.float32)
    assert np.array_equal(out, expected)


def test_relu_backward():
    layer = ReLU()
    Z = np.array([[-1, 0, 2], [3, -4, 5]], dtype=np.float32)
    layer.forward(Z)
    dA = np.ones_like(Z)
    dZ = layer.backward(dA)
    expected = np.array([[0, 0, 1], [1, 0, 1]], dtype=np.float32)
    assert np.array_equal(dZ, expected)


# --------------------------- Dropout -------------------------------

def test_dropout_eval_identity():
    layer = Dropout(0.5, seed=0)
    X = np.random.randn(4, 8).astype(np.float32)
    out = layer.forward(X, training=False)
    assert np.array_equal(out, X)


def test_dropout_train_masks_and_scales():
    layer = Dropout(0.5, seed=0)
    X = np.ones((1000, 10), dtype=np.float32)
    out = layer.forward(X, training=True)
    zeros = float((out == 0).mean())
    assert 0.4 < zeros < 0.6
    nonzero = out[out != 0]
    assert np.allclose(nonzero, 2.0)  # масштаб = 1 / keep


def test_dropout_backward_routes_gradient():
    layer = Dropout(0.5, seed=0)
    X = np.ones((4, 8), dtype=np.float32)
    layer.forward(X, training=True)
    dA = np.ones_like(X)
    dX = layer.backward(dA)
    assert np.allclose(dX, layer.mask)


def test_dropout_invalid_p():
    with pytest.raises(ValueError):
        Dropout(1.0)
    with pytest.raises(ValueError):
        Dropout(-0.1)


def test_dropout_zero_p_identity():
    layer = Dropout(0.0, seed=0)
    X = np.random.randn(3, 4).astype(np.float32)
    out = layer.forward(X, training=True)
    assert np.array_equal(out, X)
"""Тесты MNIST-предобработки (надстройка для GUI)."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocess import _center_by_mass, _resize_bilinear, mnist_preprocess


# ---------------------------------------------------------------------------
# _resize_bilinear
# ---------------------------------------------------------------------------

def test_resize_identity():
    img = np.arange(9, dtype=np.float32).reshape(3, 3)
    out = _resize_bilinear(img, 3, 3)
    assert np.array_equal(out, img)


def test_resize_shape():
    img = np.random.rand(5, 7).astype(np.float32)
    out = _resize_bilinear(img, 20, 10)
    assert out.shape == (20, 10)


def test_resize_constant_preserves_value():
    img = np.full((4, 4), 0.5, dtype=np.float32)
    out = _resize_bilinear(img, 8, 8)
    assert np.allclose(out, 0.5, atol=1e-6)


# ---------------------------------------------------------------------------
# _center_by_mass
# ---------------------------------------------------------------------------

def test_center_by_mass_empty_is_noop():
    img = np.zeros((5, 5), dtype=np.float32)
    out = _center_by_mass(img)
    assert np.array_equal(out, img)


def test_center_by_mass_moves_dot_to_center():
    img = np.zeros((28, 28), dtype=np.float32)
    img[2, 3] = 1.0
    out = _center_by_mass(img)
    # Яркая точка должна оказаться около (14, 14)
    ys, xs = np.where(out > 0.5)
    assert len(ys) == 1
    assert abs(int(ys[0]) - 14) <= 1
    assert abs(int(xs[0]) - 14) <= 1


def test_center_by_mass_preserves_total_mass():
    rng = np.random.default_rng(0)
    img = rng.random((28, 28)).astype(np.float32)
    out = _center_by_mass(img)
    assert abs(float(img.sum()) - float(out.sum())) < 1e-3


# ---------------------------------------------------------------------------
# mnist_preprocess
# ---------------------------------------------------------------------------

def test_mnist_preprocess_empty_returns_zeros():
    img = np.zeros((100, 100), dtype=np.float32)
    out = mnist_preprocess(img, size=28)
    assert out.shape == (28, 28)
    assert out.max() == 0.0


def test_mnist_preprocess_output_shape_and_range():
    rng = np.random.default_rng(1)
    img = (rng.random((300, 300)) < 0.05).astype(np.float32)
    out = mnist_preprocess(img, size=28, inner=20)
    assert out.shape == (28, 28)
    assert out.dtype == np.float32
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_mnist_preprocess_crops_and_centers():
    """Большая жирная точка в углу должна превратиться в компактное пятно
    около центра 28×28."""
    img = np.zeros((280, 280), dtype=np.float32)
    img[30:80, 220:270] = 1.0  # квадрат 50×50 в правом верхнем углу

    out = mnist_preprocess(img, size=28, inner=20)

    ys, xs = np.where(out > 0.05)
    cy = ys.mean()
    cx = xs.mean()
    # центр масс ≈ (13.5, 13.5)
    assert abs(cy - 13.5) <= 1.5, f"cy={cy}"
    assert abs(cx - 13.5) <= 1.5, f"cx={cx}"

    # высота/ширина не больше 20 пикселей
    assert ys.max() - ys.min() + 1 <= 20
    assert xs.max() - xs.min() + 1 <= 20


def test_mnist_preprocess_preserves_aspect_ratio():
    """Прямоугольник 40×20 (2:1) должен остаться примерно 2:1."""
    img = np.zeros((100, 100), dtype=np.float32)
    img[10:30, 10:50] = 1.0  # h=20, w=40

    out = mnist_preprocess(img, size=28, inner=20)
    ys, xs = np.where(out > 0.05)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    assert h <= 11, f"h={h}"  # ожидаем 10 (20 / 2)
    assert w <= 21, f"w={w}"  # ожидаем 20
    assert w > h  # соотношение сохранено


def test_mnist_preprocess_invalid_input():
    with pytest.raises(ValueError):
        mnist_preprocess(np.zeros((1, 2, 3), dtype=np.float32))


def test_mnist_preprocess_idempotent_on_synthetic_mnist():
    """Если на вход подан уже «правильный» MNIST-подобный вход,
    повторная предобработка не должна сильно его менять."""
    rng = np.random.default_rng(42)
    # синтетическая «цифра»: пятно в центре 28×28
    img = np.zeros((28, 28), dtype=np.float32)
    img[8:20, 10:18] = rng.random((12, 8)).astype(np.float32)

    once = mnist_preprocess(img, size=28, inner=20)
    twice = mnist_preprocess(once, size=28, inner=20)

    # второй проход не должен сильно изменить распределение
    diff = float(np.abs(once - twice).mean())
    assert diff < 0.05, f"mean abs diff = {diff}"
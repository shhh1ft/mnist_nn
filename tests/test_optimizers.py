"""Тесты оптимизаторов (недели 3–4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.optimizers import SGD, Adam, _clip_by_global_norm_


class DummyModel:
    def __init__(self, param: np.ndarray, grad: np.ndarray | None = None) -> None:
        self.param = param
        self.grad = np.ones_like(param) if grad is None else grad

    def params_and_grads(self):
        return [(self.param, self.grad)]


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------

def test_sgd_vanilla_step():
    model = DummyModel(np.array([1.0, 2.0], dtype=np.float32))
    opt = SGD(lr=0.1)
    opt.step(model)
    assert np.allclose(model.param, [0.9, 1.9])


def test_sgd_momentum_two_steps():
    model = DummyModel(np.array([1.0], dtype=np.float32))
    opt = SGD(lr=0.1, momentum=0.9)
    opt.step(model)
    assert np.allclose(model.param, [0.9], atol=1e-6)
    opt.step(model)
    assert np.allclose(model.param, [0.71], atol=1e-6)


def test_sgd_weight_decay():
    model = DummyModel(np.array([1.0], dtype=np.float32))
    opt = SGD(lr=0.1, weight_decay=0.1)
    opt.step(model)
    # g = 1 + 0.1·1 = 1.1; W = 1 − 0.1·1.1 = 0.89
    assert np.allclose(model.param, [0.89])


def test_sgd_invalid_lr():
    with pytest.raises(ValueError):
        SGD(lr=0.0)
    with pytest.raises(ValueError):
        SGD(lr=-0.1)


def test_sgd_invalid_momentum():
    with pytest.raises(ValueError):
        SGD(momentum=-0.1)
    with pytest.raises(ValueError):
        SGD(momentum=1.0)


def test_sgd_invalid_weight_decay():
    with pytest.raises(ValueError):
        SGD(weight_decay=-0.1)


def test_sgd_invalid_clip_norm():
    with pytest.raises(ValueError):
        SGD(clip_norm=0.0)
    with pytest.raises(ValueError):
        SGD(clip_norm=-1.0)


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

def test_adam_first_step_matches_hand_calc():
    """Adam с β₁=0.9, β₂=0.999, g=1, lr=0.1, t=1 даёт W → 0.9."""
    model = DummyModel(np.array([1.0], dtype=np.float64))
    opt = Adam(lr=0.1)
    opt.step(model)
    assert np.allclose(model.param, [0.9], atol=1e-6)


def test_adam_two_steps_constant_grad():
    """Второй шаг при том же градиенте: W → 0.8 (примерно)."""
    model = DummyModel(np.array([1.0], dtype=np.float64))
    opt = Adam(lr=0.1)
    opt.step(model)
    opt.step(model)
    assert np.allclose(model.param, [0.8], atol=1e-3)


def test_adam_state_is_per_parameter():
    p1 = np.array([1.0], dtype=np.float32)
    p2 = np.array([2.0], dtype=np.float32)

    class M:
        def params_and_grads(self):
            return [(p1, np.array([1.0], dtype=np.float32)),
                    (p2, np.array([1.0], dtype=np.float32))]

    opt = Adam(lr=0.1)
    opt.step(M())
    assert len(opt.m) == 2
    assert len(opt.v) == 2


def test_adam_weight_decay():
    """weight_decay добавляет wd·param к градиенту."""
    m_a = DummyModel(np.array([1.0], dtype=np.float64))
    m_b = DummyModel(np.array([1.0], dtype=np.float64))
    opt_a = Adam(lr=0.1, weight_decay=0.0)
    opt_b = Adam(lr=0.1, weight_decay=0.5)
    opt_a.step(m_a)
    opt_b.step(m_b)
    # при wd=0.5 градиент больше → шаг больше → param меньше
    assert float(m_b.param[0]) < float(m_a.param[0])


def test_adam_invalid_params():
    with pytest.raises(ValueError):
        Adam(lr=0.0)
    with pytest.raises(ValueError):
        Adam(beta1=1.0)
    with pytest.raises(ValueError):
        Adam(beta2=-0.1)
    with pytest.raises(ValueError):
        Adam(eps=0.0)
    with pytest.raises(ValueError):
        Adam(weight_decay=-0.1)
    with pytest.raises(ValueError):
        Adam(clip_norm=0.0)


def test_adam_converges_on_quadratic():
    """Adam сходится на f(w) = (w − 3)² за разумное число шагов."""
    w = np.array([0.0], dtype=np.float64)

    class M:
        def params_and_grads(self):
            grad = 2.0 * (w - 3.0)
            return [(w, grad)]

    opt = Adam(lr=0.1)
    for _ in range(200):
        opt.step(M())
    assert abs(float(w[0]) - 3.0) < 1e-2


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def test_clip_by_global_norm_scales():
    p = np.array([0.0, 0.0], dtype=np.float32)
    g = np.array([3.0, 4.0], dtype=np.float32)  # ||g|| = 5
    pairs = [(p, g)]
    total = _clip_by_global_norm_(pairs, max_norm=1.0)
    assert abs(total - 5.0) < 1e-5
    assert abs(float(np.linalg.norm(g)) - 1.0) < 1e-5


def test_clip_by_global_norm_no_op_below_threshold():
    p = np.array([0.0], dtype=np.float32)
    g = np.array([0.1], dtype=np.float32)
    pairs = [(p, g)]
    _clip_by_global_norm_(pairs, max_norm=1.0)
    assert np.allclose(g, [0.1])


def test_sgd_with_clipping():
    """Большой градиент должен быть ограничен clip_norm."""
    model = DummyModel(
        np.array([0.0], dtype=np.float32),
        grad=np.array([100.0], dtype=np.float32),
    )
    opt = SGD(lr=1.0, clip_norm=1.0)
    opt.step(model)
    # g после клиппинга ≈ 1 → шаг ≈ 1
    assert abs(float(model.param[0]) + 1.0) < 1e-5


def test_adam_with_clipping():
    model = DummyModel(
        np.array([0.0], dtype=np.float64),
        grad=np.array([1000.0], dtype=np.float64),
    )
    opt = Adam(lr=0.1, clip_norm=1.0)
    opt.step(model)
    # после клиппинга |шаг| ≈ lr (Adam нормирует градиент)
    assert abs(float(model.param[0])) < 0.2
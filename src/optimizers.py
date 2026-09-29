"""Оптимизаторы: SGD, Adam + gradient clipping.

Недели 3–4.

- SGD:     W ← W − lr · (g + wd·W)                    [с momentum: + v]
- Adam:    m ← β₁·m + (1−β₁)·g
           v ← β₂·v + (1−β₂)·g²
           m̂ = m / (1−β₁ᵗ);  v̂ = v / (1−β₂ᵗ)
           W ← W − lr · m̂ / (√v̂ + ε)
- Gradient clipping: масштабирует все градиенты, если их глобальная
  L2-норма превышает max_norm. Применяется внутри step() до обновления.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class _HasParams(Protocol):
    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]: ...


# ---------------------------------------------------------------------------
# Общие утилиты
# ---------------------------------------------------------------------------

def _clip_by_global_norm_(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    max_norm: float,
) -> float:
    """Масштабирует градиенты in-place, если их глобальная норма > max_norm.

    Возвращает исходную глобальную норму (для логирования).
    """
    total_sq = 0.0
    for _, g in pairs:
        total_sq += float(np.sum(g * g))
    total_norm = float(np.sqrt(total_sq))

    if total_norm > max_norm and total_norm > 0.0:
        scale = max_norm / (total_norm + 1e-12)
        for _, g in pairs:
            g *= scale
    return total_norm


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------

class SGD:
    """Стохастический градиентный спуск с momentum и weight_decay."""

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        clip_norm: float | None = None,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr должен быть положительным")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum должен быть в [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay не может быть отрицательным")
        if clip_norm is not None and clip_norm <= 0:
            raise ValueError("clip_norm должен быть положительным или None")

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.velocity: dict[int, np.ndarray] = {}

    def step(self, model: _HasParams) -> None:
        pairs = model.params_and_grads()

        if self.clip_norm is not None:
            _clip_by_global_norm_(pairs, self.clip_norm)

        for param, grad in pairs:
            g = grad
            if self.weight_decay > 0.0:
                g = g + self.weight_decay * param

            if self.momentum > 0.0:
                key = id(param)
                v = self.velocity.get(key)
                if v is None:
                    v = np.zeros_like(param)
                v = self.momentum * v - self.lr * g
                self.velocity[key] = v
                param += v
            else:
                param -= self.lr * g


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class Adam:
    """Adam (Kingma & Ba, 2014) с опциональным gradient clipping.

    L2-регуляризация реализована «по-старому» (L2, добавляется к градиенту).
    AdamW (decoupled weight decay) — потенциальное расширение.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_norm: float | None = None,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr должен быть положительным")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 должен быть в [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 должен быть в [0, 1)")
        if eps <= 0:
            raise ValueError("eps должен быть положительным")
        if weight_decay < 0:
            raise ValueError("weight_decay не может быть отрицательным")
        if clip_norm is not None and clip_norm <= 0:
            raise ValueError("clip_norm должен быть положительным или None")

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm

        self.t = 0
        self.m: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}

    def step(self, model: _HasParams) -> None:
        pairs = model.params_and_grads()

        if self.clip_norm is not None:
            _clip_by_global_norm_(pairs, self.clip_norm)

        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for param, grad in pairs:
            g = grad
            if self.weight_decay > 0.0:
                g = g + self.weight_decay * param

            key = id(param)
            m = self.m.get(key)
            v = self.v.get(key)
            if m is None:
                m = np.zeros_like(param)
                v = np.zeros_like(param)

            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * (g * g)

            self.m[key] = m
            self.v[key] = v

            m_hat = m / bc1
            v_hat = v / bc2

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
"""Функции потерь.

Неделя 2: SoftmaxCrossEntropy — совместная реализация Softmax + Cross-Entropy.
Совмещение даёт:
- численную устойчивость (log-sum-exp);
- простой градиент ∂L/∂Z = (ŷ − y) / N.

Forward считается в float64 для стабильности, даже если вход был float32.
Backward тоже возвращает float64 — присваивание в dW/db слоёв делает cast
к их исходному типу.
"""

from __future__ import annotations

import numpy as np


class SoftmaxCrossEntropy:
    """Softmax + Cross-Entropy для one-hot меток."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps
        self.probs: np.ndarray | None = None
        self.targets: np.ndarray | None = None
        self.N: int | None = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits.shape={logits.shape} != targets.shape={targets.shape}"
            )

        # Все вычисления — в float64
        logits64 = np.asarray(logits, dtype=np.float64)
        targets64 = np.asarray(targets, dtype=np.float64)

        self.targets = targets64
        self.N = logits64.shape[0]

        # численно устойчивый softmax: вычитаем максимум по строке
        shifted = logits64 - logits64.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        self.probs = probs

        # L = -Σ y·log(ŷ) / N
        log_probs = np.log(probs + self.eps)
        loss = -np.sum(targets64 * log_probs) / self.N
        return float(loss)

    def backward(self) -> np.ndarray:
        """∂L/∂logits = (ŷ − y) / N."""
        return (self.probs - self.targets) / self.N
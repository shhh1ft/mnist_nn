"""Слои нейронной сети с ручным forward/backward на NumPy.

Неделя 2: Dense, ReLU, Dropout.

Каждый слой реализует интерфейс Layer:
- forward(X, training=True) -> выход
- backward(dA)              -> dInput
- params_and_grads()        -> список кортежей (param, grad)
"""

from __future__ import annotations

import numpy as np


class Layer:
    """Базовый класс слоя. По умолчанию — без параметров."""

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dA: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return []


class Dense(Layer):
    """Полносвязный слой: Z = X · W + b.

    Инициализация:
    - 'he':     std = sqrt(2 / in_features)              — для ReLU
    - 'xavier': std = sqrt(2 / (in_features + out))     — для tanh/sigmoid
    - 'small':  std = 0.01
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init: str = "he",
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)

        if init == "he":
            std = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
        elif init == "small":
            std = 0.01
        else:
            raise ValueError(f"Неизвестный метод инициализации: {init}")

        self.W = rng.normal(0.0, std, size=(in_features, out_features)).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.X: np.ndarray | None = None
        self.in_features = in_features
        self.out_features = out_features
        self.init = init

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        # dW: (in, out) = Xᵀ · dZ; db: (out,) = Σ dZ по батчу
        self.dW[...] = self.X.T @ dZ
        self.db[...] = dZ.sum(axis=0)
        return dZ @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU(Layer):
    """ReLU(z) = max(0, z)."""

    def __init__(self) -> None:
        self.mask: np.ndarray | None = None

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        self.mask = Z > 0
        return Z * self.mask

    def backward(self, dA: np.ndarray) -> np.ndarray:
        return dA * self.mask


class Dropout(Layer):
    """Inverted dropout.

    Во время обучения: с вероятностью p обнуляет нейрон и масштабирует
    оставшиеся на 1/(1-p), чтобы матожидание выхода сохранилось.
    Во время инференса: тождественное преобразование.
    """

    def __init__(self, p: float, seed: int | None = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError("p должен быть в интервале [0, 1)")
        self.p = p
        self.keep = 1.0 - p
        self.rng = np.random.default_rng(seed)
        self.mask: np.ndarray | None = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.p == 0.0:
            self.mask = None
            return X
        self.mask = (self.rng.random(X.shape) < self.keep).astype(X.dtype) / self.keep
        return X * self.mask

    def backward(self, dA: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return dA
        return dA * self.mask
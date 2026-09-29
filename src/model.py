"""Последовательная модель и сборка из JSON-конфига."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .conv import Conv2D, Flatten, MaxPool2D
from .layers import Dense, Dropout, Layer, ReLU


class Sequential:
    """Контейнер слоёв: forward по порядку, backward — в обратном."""

    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X, training=training)
        return X

    def backward(self, dA: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
        return dA

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in self.layers:
            pairs.extend(layer.params_and_grads())
        return pairs

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            for j, (param, _) in enumerate(layer.params_and_grads()):
                arrays[f"layer{i}_param{j}"] = param
        np.savez(path, **arrays)

    def load(self, path: str | Path) -> None:
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            for j, (param, _) in enumerate(layer.params_and_grads()):
                key = f"layer{i}_param{j}"
                if key in data:
                    param[...] = data[key]


def build_model_from_config(config: dict, seed: int | None = None) -> Sequential:
    """Собирает Sequential из секции 'model' JSON-конфига.

    Поддерживаемые слои:
        dense     — Dense (+ ReLU, если activation='relu')
        dropout   — Dropout
        conv2d    — Conv2D
        maxpool2d — MaxPool2D
        flatten   — Flatten
        relu      — отдельный ReLU-слой

    Активация 'softmax' в dense игнорируется как слой — её роль
    выполняет SoftmaxCrossEntropy.
    """
    spec = config["model"]
    layers_cfg = spec["layers"]
    layers: list[Layer] = []

    for i, lc in enumerate(layers_cfg):
        t = lc["type"]
        layer_seed = None if seed is None else seed + i

        if t == "dense":
            layers.append(
                Dense(
                    lc["in"], lc["out"],
                    init=lc.get("init", "he"),
                    seed=layer_seed,
                )
            )
            activation = lc.get("activation", "linear")
            if activation == "relu":
                layers.append(ReLU())
            elif activation in ("softmax", "linear", None):
                pass
            else:
                raise ValueError(f"Неизвестная активация: {activation}")

        elif t == "dropout":
            layers.append(Dropout(lc["p"], seed=layer_seed))

        elif t == "conv2d":
            layers.append(
                Conv2D(
                    in_channels=lc["in_channels"],
                    out_channels=lc["out_channels"],
                    kernel_size=lc["kernel"],
                    stride=lc.get("stride", 1),
                    padding=lc.get("padding", 0),
                    init=lc.get("init", "he"),
                    seed=layer_seed,
                )
            )
            if lc.get("activation") == "relu":
                layers.append(ReLU())

        elif t == "maxpool2d":
            layers.append(
                MaxPool2D(
                    kernel_size=lc["kernel"],
                    stride=lc.get("stride"),
                )
            )

        elif t == "flatten":
            layers.append(Flatten())

        elif t == "relu":
            layers.append(ReLU())

        else:
            raise ValueError(f"Неизвестный тип слоя: {t}")

    return Sequential(layers)
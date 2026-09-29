"""Вспомогательные утилиты: seed, директории, загрузка MNIST, softmax."""

from __future__ import annotations

import random
import urllib.request
from pathlib import Path

import numpy as np


MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"

MNIST_FILES = {
    "train-images-idx3-ubyte.gz": f"{MNIST_MIRROR}/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": f"{MNIST_MIRROR}/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": f"{MNIST_MIRROR}/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": f"{MNIST_MIRROR}/t10k-labels-idx1-ubyte.gz",
}


def set_seed(seed: int) -> None:
    """Фиксирует seed для модуля random и NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_mnist(target_dir: str | Path) -> Path:
    target = ensure_dir(target_dir)
    for name, url in MNIST_FILES.items():
        dest = target / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"[download] {name}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return target


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Численно устойчивый softmax."""
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)
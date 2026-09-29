"""Загрузка, подготовка и батчирование датасета MNIST.

Модуль реализует:
- чтение бинарных IDX-файлов MNIST (с поддержкой .gz);
- нормализацию пикселей в диапазон [0, 1];
- one-hot кодирование меток;
- стратифицированное разбиение train/validation;
- мини-батчи с перемешиванием;
- кэширование подготовленных массивов в .npz.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Iterator

import numpy as np

from .utils import download_mnist, ensure_dir


# ---------------------------------------------------------------------------
# IDX parsing
# ---------------------------------------------------------------------------

def _open(path: Path, mode: str = "rb"):
    if path.suffix == ".gz":
        return gzip.open(path, mode)
    return open(path, mode)


def read_idx_images(path: str | Path) -> np.ndarray:
    """Читает IDX-файл изображений.

    Возвращает массив формы (N, rows, cols) с dtype uint8.
    """
    path = Path(path)
    with _open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2051:
            raise ValueError(
                f"Неверный magic number {magic} в {path.name}: ожидалось 2051"
            )
        n, rows, cols = struct.unpack(">III", f.read(12))
        buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8)
    expected = n * rows * cols
    if data.size != expected:
        raise ValueError(
            f"{path.name}: размер данных {data.size} не совпадает "
            f"с заголовком ({expected})"
        )
    return data.reshape(n, rows, cols)


def read_idx_labels(path: str | Path) -> np.ndarray:
    """Читает IDX-файл меток. Возвращает массив формы (N,) dtype uint8."""
    path = Path(path)
    with _open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2049:
            raise ValueError(
                f"Неверный magic number {magic} в {path.name}: ожидалось 2049"
            )
        (n,) = struct.unpack(">I", f.read(4))
        buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8)
    if data.size != n:
        raise ValueError(
            f"{path.name}: размер данных {data.size} не совпадает "
            f"с заголовком ({n})"
        )
    return data


# ---------------------------------------------------------------------------
# Преобразования
# ---------------------------------------------------------------------------

def normalize_images(X: np.ndarray) -> np.ndarray:
    """Приводит uint8-пиксели к float32 в диапазоне [0, 1]."""
    return X.astype(np.float32) / 255.0


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """One-hot кодирование меток. Возвращает float32 массив (N, num_classes)."""
    labels = np.asarray(labels, dtype=np.int64)
    return np.eye(num_classes, dtype=np.float32)[labels]


def stratified_split(
    labels: np.ndarray,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Стратифицированное разбиение индексов на train/val по классам.

    Возвращает два массива индексов: (train_idx, val_idx).
    """
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size должен быть в интервале (0, 1)")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_size))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Мини-батчи
# ---------------------------------------------------------------------------

class DataLoader:
    """Генератор мини-батчей с опциональным перемешиванием."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if len(X) != len(y):
            raise ValueError("X и y должны иметь одинаковую длину")
        if batch_size <= 0:
            raise ValueError("batch_size должен быть положительным")
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / self.batch_size))

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        idx = np.arange(len(self.X))
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            batch = idx[start:start + self.batch_size]
            yield self.X[batch], self.y[batch]


# ---------------------------------------------------------------------------
# Подготовка датасета
# ---------------------------------------------------------------------------

def prepare_data(
    raw_dir: str | Path,
    processed_dir: str | Path,
    val_size: float = 0.1,
    seed: int = 42,
    use_cache: bool = True,
) -> tuple[np.ndarray, ...]:
    """Готовит MNIST: train/val (стратифицированно) и test.

    Ожидает, что в raw_dir уже лежат 4 gz-файла MNIST.
    Возвращает кортеж (X_train, y_train, X_val, y_val, X_test, y_test),
    где X — float32 (N, 784), y — one-hot float32 (N, 10).
    """
    raw_dir = Path(raw_dir)
    processed_dir = ensure_dir(processed_dir)
    cache_path = processed_dir / "mnist_split.npz"

    if use_cache and cache_path.exists():
        npz = np.load(cache_path)
        return (
            npz["X_train"], npz["y_train"],
            npz["X_val"], npz["y_val"],
            npz["X_test"], npz["y_test"],
        )

    X_train_full = read_idx_images(raw_dir / "train-images-idx3-ubyte.gz")
    y_train_full = read_idx_labels(raw_dir / "train-labels-idx1-ubyte.gz")
    X_test = read_idx_images(raw_dir / "t10k-images-idx3-ubyte.gz")
    y_test = read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte.gz")

    X_train_full = normalize_images(X_train_full).reshape(len(X_train_full), -1)
    X_test = normalize_images(X_test).reshape(len(X_test), -1)

    train_idx, val_idx = stratified_split(
        y_train_full, val_size=val_size, seed=seed,
    )

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    y_train = one_hot(y_train)
    y_val = one_hot(y_val)
    y_test = one_hot(y_test)

    if use_cache:
        np.savez_compressed(
            cache_path,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
        )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_mnist(
    data_dir: str | Path,
    val_size: float = 0.1,
    seed: int = 42,
    use_cache: bool = True,
    download: bool = True,
) -> tuple[np.ndarray, ...]:
    """Скачивает (при необходимости) и подготавливает MNIST."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    if download:
        ensure_dir(raw_dir)
        download_mnist(raw_dir)

    return prepare_data(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        val_size=val_size,
        seed=seed,
        use_cache=use_cache,
    )
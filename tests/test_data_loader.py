"""Тесты загрузчика MNIST (неделя 1).

Используют синтетические IDX-файлы и фейковые "сырые" данные —
без сетевого доступа, чтобы CI был быстрым.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from src.data_loader import (
    DataLoader,
    normalize_images,
    one_hot,
    prepare_data,
    read_idx_images,
    read_idx_labels,
    stratified_split,
)


# ---------------------------------------------------------------------------
# helpers для записи синтетических IDX
# ---------------------------------------------------------------------------

def _write_idx_images(path: Path, images: np.ndarray) -> None:
    n, rows, cols = images.shape
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">I", 2051))
        f.write(struct.pack(">III", n, rows, cols))
        f.write(images.astype(np.uint8).tobytes())


def _write_idx_labels(path: Path, labels: np.ndarray) -> None:
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">I", 2049))
        f.write(struct.pack(">I", len(labels)))
        f.write(labels.astype(np.uint8).tobytes())


# ---------------------------------------------------------------------------
# IDX reader
# ---------------------------------------------------------------------------

def test_read_idx_images_roundtrip(tmp_path: Path) -> None:
    imgs = np.random.randint(0, 256, size=(5, 28, 28), dtype=np.uint8)
    p = tmp_path / "imgs.gz"
    _write_idx_images(p, imgs)
    loaded = read_idx_images(p)
    assert loaded.shape == (5, 28, 28)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, imgs)


def test_read_idx_images_bad_magic(tmp_path: Path) -> None:
    p = tmp_path / "bad.gz"
    with gzip.open(p, "wb") as f:
        f.write(struct.pack(">I", 1234))
        f.write(struct.pack(">III", 1, 1, 1))
        f.write(b"\x00")
    with pytest.raises(ValueError, match="magic number"):
        read_idx_images(p)


def test_read_idx_images_truncated(tmp_path: Path) -> None:
    p = tmp_path / "trunc.gz"
    with gzip.open(p, "wb") as f:
        f.write(struct.pack(">I", 2051))
        f.write(struct.pack(">III", 4, 28, 28))
        f.write(b"\x00" * 100)  # мало данных
    with pytest.raises(ValueError, match="не совпадает"):
        read_idx_images(p)


def test_read_idx_labels_roundtrip(tmp_path: Path) -> None:
    labels = np.array([0, 1, 2, 3, 4, 9], dtype=np.uint8)
    p = tmp_path / "labels.gz"
    _write_idx_labels(p, labels)
    loaded = read_idx_labels(p)
    assert np.array_equal(loaded, labels)
    assert loaded.dtype == np.uint8


# ---------------------------------------------------------------------------
# Преобразования
# ---------------------------------------------------------------------------

def test_normalize_images() -> None:
    X = np.array([[0, 127, 255]], dtype=np.uint8)
    out = normalize_images(X)
    assert out.dtype == np.float32
    assert out[0, 0] == 0.0
    assert out[0, 2] == 1.0
    assert abs(float(out[0, 1]) - 127 / 255) < 1e-6


def test_one_hot_shape_and_values() -> None:
    y = np.array([0, 3, 9])
    out = one_hot(y, num_classes=10)
    assert out.shape == (3, 10)
    assert out.dtype == np.float32
    assert out[0, 0] == 1.0
    assert out[1, 3] == 1.0
    assert out[2, 9] == 1.0
    assert out.sum() == 3.0


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def test_stratified_split_sizes_and_balance() -> None:
    y = np.array([0] * 100 + [1] * 100 + [2] * 100)
    train_idx, val_idx = stratified_split(y, val_size=0.1, seed=0)
    assert len(train_idx) + len(val_idx) == 300
    for c in (0, 1, 2):
        n_val_c = int(np.sum(y[val_idx] == c))
        assert 8 <= n_val_c <= 12, f"класс {c}: {n_val_c} примеров в val"


def test_stratified_split_no_overlap() -> None:
    y = np.arange(10).repeat(20)
    train_idx, val_idx = stratified_split(y, val_size=0.2, seed=1)
    assert set(train_idx).isdisjoint(set(val_idx))
    assert len(set(train_idx) | set(val_idx)) == len(y)


def test_stratified_split_invalid_val_size() -> None:
    y = np.arange(10)
    with pytest.raises(ValueError):
        stratified_split(y, val_size=0.0)
    with pytest.raises(ValueError):
        stratified_split(y, val_size=1.0)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def test_dataloader_batches_cover_all() -> None:
    X = np.random.rand(100, 10).astype(np.float32)
    y = one_hot(np.random.randint(0, 10, 100))
    loader = DataLoader(X, y, batch_size=32, shuffle=False, seed=0)
    assert len(loader) == 4  # 32 + 32 + 32 + 4
    total = sum(len(Xb) for Xb, _ in loader)
    assert total == 100


def test_dataloader_shuffle_changes_order() -> None:
    X = np.arange(100).reshape(100, 1).astype(np.float32)
    y = np.zeros((100, 1), dtype=np.float32)
    a, _ = next(iter(DataLoader(X, y, batch_size=100, shuffle=False, seed=0)))
    b, _ = next(iter(DataLoader(X, y, batch_size=100, shuffle=True, seed=0)))
    assert not np.array_equal(a, b)


def test_dataloader_shuffle_reproducible() -> None:
    X = np.arange(100).reshape(100, 1).astype(np.float32)
    y = np.zeros((100, 1), dtype=np.float32)
    a, _ = next(iter(DataLoader(X, y, batch_size=100, shuffle=True, seed=7)))
    b, _ = next(iter(DataLoader(X, y, batch_size=100, shuffle=True, seed=7)))
    assert np.array_equal(a, b)


def test_dataloader_length_mismatch() -> None:
    with pytest.raises(ValueError):
        DataLoader(np.zeros((10, 4)), np.zeros((9, 2)))


def test_dataloader_invalid_batch_size() -> None:
    with pytest.raises(ValueError):
        DataLoader(np.zeros((10, 4)), np.zeros((10, 2)), batch_size=0)


# ---------------------------------------------------------------------------
# Полный пайплайн подготовки (без сети)
# ---------------------------------------------------------------------------

def test_prepare_data_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    rng = np.random.default_rng(0)
    # 200 train, 4 класса по 50
    n_train_per_class = 50
    n_classes = 4
    X_tr = rng.integers(0, 256, size=(200, 28, 28), dtype=np.uint8)
    y_tr = np.repeat(np.arange(n_classes), n_train_per_class).astype(np.uint8)
    X_te = rng.integers(0, 256, size=(40, 28, 28), dtype=np.uint8)
    y_te = np.repeat(np.arange(n_classes), 10).astype(np.uint8)

    _write_idx_images(raw / "train-images-idx3-ubyte.gz", X_tr)
    _write_idx_labels(raw / "train-labels-idx1-ubyte.gz", y_tr)
    _write_idx_images(raw / "t10k-images-idx3-ubyte.gz", X_te)
    _write_idx_labels(raw / "t10k-labels-idx1-ubyte.gz", y_te)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        raw_dir=raw,
        processed_dir=tmp_path / "processed",
        val_size=0.1,
        seed=42,
        use_cache=False,
    )

    assert X_train.shape == (180, 784)
    assert X_val.shape == (20, 784)
    assert X_test.shape == (40, 784)
    assert y_train.shape == (180, 10)
    assert y_val.shape == (20, 10)
    assert y_test.shape == (40, 10)
    assert X_train.dtype == np.float32
    assert 0.0 <= float(X_train.min()) and float(X_train.max()) <= 1.0


def test_prepare_data_uses_cache(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(1)
    X_tr = rng.integers(0, 256, size=(40, 28, 28), dtype=np.uint8)
    y_tr = np.repeat(np.arange(4), 10).astype(np.uint8)
    X_te = rng.integers(0, 256, size=(8, 28, 28), dtype=np.uint8)
    y_te = np.repeat(np.arange(4), 2).astype(np.uint8)

    _write_idx_images(raw / "train-images-idx3-ubyte.gz", X_tr)
    _write_idx_labels(raw / "train-labels-idx1-ubyte.gz", y_tr)
    _write_idx_images(raw / "t10k-images-idx3-ubyte.gz", X_te)
    _write_idx_labels(raw / "t10k-labels-idx1-ubyte.gz", y_te)

    processed = tmp_path / "processed"
    first = prepare_data(raw, processed, val_size=0.25, seed=0, use_cache=True)
    cache = processed / "mnist_split.npz"
    assert cache.exists()

    # портим сырые данные, но кэш должен вернуть то же самое
    (raw / "train-images-idx3-ubyte.gz").unlink()
    second = prepare_data(raw, processed, val_size=0.25, seed=0, use_cache=True)

    for a, b in zip(first, second):
        assert np.array_equal(a, b)
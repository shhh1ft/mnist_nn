"""MNIST-style предобработка нарисованной от руки цифры.

Стандартный пайплайн (тот же, что использовался при создании MNIST):
1. обрезать по bounding box ненулевых пикселей;
2. масштабировать так, чтобы большая сторона стала 20 пикселей
   (сохраняя пропорции, билинейная интерполяция);
3. вставить в квадрат 28×28 так, чтобы центр масс совпал с центром.

Без шагов 1–3 модель, обученная на MNIST, на «сырых» рисунках
работает плохо: входы оказываются смещёнными и слишком крупными.
"""

from __future__ import annotations

import numpy as np


def _resize_bilinear(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Билинейное изменение размера. img: (h, w) float32 → (new_h, new_w)."""
    h, w = img.shape
    if (h, w) == (new_h, new_w):
        return img.astype(np.float32, copy=True)

    y = np.linspace(0.0, h - 1.0, new_h)
    x = np.linspace(0.0, w - 1.0, new_w)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    y0 = np.floor(yy).astype(np.int64)
    x0 = np.floor(xx).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)

    wy = (yy - y0).astype(np.float32)
    wx = (xx - x0).astype(np.float32)

    top = img[y0, x0] * (1.0 - wx) + img[y0, x1] * wx
    bot = img[y1, x0] * (1.0 - wx) + img[y1, x1] * wx
    out = top * (1.0 - wy) + bot * wy
    return out.astype(np.float32)


def _center_by_mass(img: np.ndarray) -> np.ndarray:
    """Сдвигает изображение так, чтобы центр масс яркости попал в центр.

    Использует только целочисленный сдвиг (без интерполяции), чтобы
    не размывать штрихи.
    """
    h, w = img.shape
    total = float(img.sum())
    if total <= 0.0:
        return img

    ys, xs = np.mgrid[0:h, 0:w]
    cy = float((ys * img).sum() / total)
    cx = float((xs * img).sum() / total)

    target_y = (h - 1) / 2.0
    target_x = (w - 1) / 2.0

    shift_y = int(round(target_y - cy))
    shift_x = int(round(target_x - cx))

    if shift_y == 0 and shift_x == 0:
        return img

    out = np.zeros_like(img)
    src_y0 = max(0, -shift_y)
    src_y1 = min(h, h - shift_y)
    src_x0 = max(0, -shift_x)
    src_x1 = min(w, w - shift_x)
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return img

    out[src_y0 + shift_y:src_y1 + shift_y,
        src_x0 + shift_x:src_x1 + shift_x] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def mnist_preprocess(
    img: np.ndarray,
    size: int = 28,
    inner: int = 20,
    threshold: float = 0.05,
) -> np.ndarray:
    """Приводит изображение к виду, ожидаемому моделью, обученной на MNIST.

    Параметры:
        img:       (H, W) float32 в [0, 1] — что угодно, любое соотношение;
        size:      итоговый размер квадрата (по умолчанию 28);
        inner:     во сколько пикселей вписать большую сторону (20 — стандарт);
        threshold: порог «значащего» пикселя для bounding box.

    Возвращает (size, size) float32 в [0, 1].
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"ожидался 2D-массив, получено {img.shape}")

    # уже пусто — возвращаем нули нужного размера
    if img.max(initial=0.0) < threshold:
        return np.zeros((size, size), dtype=np.float32)

    # 1) crop по bbox значащих пикселей
    ys, xs = np.where(img > threshold)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = img[y0:y1, x0:x1]

    h, w = crop.shape

    # 2) масштаб так, чтобы большая сторона стала `inner`
    scale = float(inner) / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = _resize_bilinear(crop, new_h, new_w)

    # 3) вставка в квадрат size×size
    canvas = np.zeros((size, size), dtype=np.float32)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    # 4) точное центрирование по центру масс
    canvas = _center_by_mass(canvas)

    # 5) клиппинг на случай интерполяционных «выбросов»
    np.clip(canvas, 0.0, 1.0, out=canvas)
    return canvas
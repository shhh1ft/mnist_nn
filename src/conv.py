"""Свёрточные слои с ручным forward/backward на NumPy.

Реализация через im2col/col2im:
- im2col раскладывает вход (N, C, H, W) в матрицу патчей
  (N, C·kH·kW, H_out·W_out);
- свёртка превращается в матричное умножение (C_out, C·kH·kW) @ cols;
- col2im — обратная операция, собирает градиент в форму входа.

Такой подход делает backward тривиальным и полностью векторизованным.
"""

from __future__ import annotations

import numpy as np

from .layers import Layer


# ---------------------------------------------------------------------------
# im2col / col2im
# ---------------------------------------------------------------------------

def im2col(
    X: np.ndarray,
    kH: int,
    kW: int,
    stride: int = 1,
    pad: int = 0,
) -> np.ndarray:
    """(N, C, H, W) → (N, C·kH·kW, H_out·W_out).

    Использует as_strided для создания «виртуального» тензора патчей
    без явного копирования; reshape триггерит однократную материализацию.
    """
    N, C, H, W = X.shape
    H_out = (H + 2 * pad - kH) // stride + 1
    W_out = (W + 2 * pad - kW) // stride + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError(
            f"Некорректные размеры: H_out={H_out}, W_out={W_out}. "
            f"Проверь kernel/stride/padding."
        )

    if pad > 0:
        X_pad = np.pad(
            X,
            ((0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode="constant",
        )
    else:
        X_pad = X

    s = X_pad.strides
    shape = (N, C, kH, kW, H_out, W_out)
    strides = (s[0], s[1], s[2], s[3], stride * s[2], stride * s[3])
    patches = np.lib.stride_tricks.as_strided(
        X_pad, shape=shape, strides=strides,
    )
    # Логический порядок [n, c, i, j, ho, wo] → [n, c·kH·kW + i·kW + j, ho·W_out + wo]
    cols = patches.reshape(N, C * kH * kW, H_out * W_out)
    return cols


def col2im(
    cols: np.ndarray,
    X_shape: tuple[int, int, int, int],
    kH: int,
    kW: int,
    stride: int = 1,
    pad: int = 0,
) -> np.ndarray:
    """Обратное к im2col: суммирует вклад каждого патча во вход."""
    N, C, H, W = X_shape
    H_out = (H + 2 * pad - kH) // stride + 1
    W_out = (W + 2 * pad - kW) // stride + 1

    cols_reshaped = cols.reshape(N, C, kH, kW, H_out, W_out)
    X_pad = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)

    for i in range(kH):
        i_max = i + stride * H_out
        for j in range(kW):
            j_max = j + stride * W_out
            X_pad[:, :, i:i_max:stride, j:j_max:stride] += (
                cols_reshaped[:, :, i, j, :, :]
            )

    if pad > 0:
        return X_pad[:, :, pad:-pad, pad:-pad]
    return X_pad


# ---------------------------------------------------------------------------
# Conv2D
# ---------------------------------------------------------------------------

class Conv2D(Layer):
    """Свёрточный слой: out = conv(X, W) + b.

    Параметры:
        in_channels, out_channels: число каналов входа/выхода;
        kernel_size: размер ядра (kH = kW = kernel_size);
        stride: шаг свёртки (по умолчанию 1);
        padding: паддинг нулями (по умолчанию 0);
        init: 'he' (для ReLU) или 'xavier'.

    Форма тензоров:
        W:  (C_out, C_in, kH, kW)
        b:  (C_out,)
        X:  (N, C_in, H, W)
        out:(N, C_out, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        init: str = "he",
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH = kernel_size
        self.kW = kernel_size
        self.stride = stride
        self.padding = padding
        self.init = init

        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        if init == "he":
            std = np.sqrt(2.0 / fan_in)
        elif init == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
        else:
            raise ValueError(f"Неизвестный метод инициализации: {init}")

        self.W = rng.normal(
            0.0, std,
            size=(out_channels, in_channels, kernel_size, kernel_size),
        ).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.X_shape: tuple[int, int, int, int] | None = None
        self.cols: np.ndarray | None = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if X.ndim != 4:
            raise ValueError(
                f"Conv2D ожидает 4D-вход (N, C, H, W), получено {X.shape}"
            )
        self.X_shape = X.shape
        N, C, H, W = X.shape

        cols = im2col(X, self.kH, self.kW, self.stride, self.padding)
        self.cols = cols

        W_col = self.W.reshape(self.out_channels, -1)  # (C_out, C·kH·kW)
        # (C_out, K) @ (N, K, P) → (N, C_out, P)
        out_flat = W_col @ cols

        H_out = (H + 2 * self.padding - self.kH) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kW) // self.stride + 1
        out = out_flat.reshape(N, self.out_channels, H_out, W_out)
        out = out + self.b.reshape(1, -1, 1, 1)
        return out

    def backward(self, dOut: np.ndarray) -> np.ndarray:
        N, C_out, H_out, W_out = dOut.shape
        W_col = self.W.reshape(self.out_channels, -1)

        dOut_flat = dOut.reshape(N, C_out, H_out * W_out)

        # dW_col[o, k] = Σ_{n,p} dOut[n,o,p] · cols[n,k,p]
        self.dW[...] = np.einsum(
            "nop,nkp->ok", dOut_flat, self.cols,
        ).reshape(self.W.shape)

        # db[o] = Σ_{n,p} dOut[n,o,p]
        self.db[...] = dOut_flat.sum(axis=(0, 2))

        # dCols[n,k,p] = Σ_o dOut[n,o,p] · W_col[o,k]
        dCols = np.einsum("nop,ok->nkp", dOut_flat, W_col)
        dX = col2im(
            dCols, self.X_shape, self.kH, self.kW,
            self.stride, self.padding,
        )
        return dX

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


# ---------------------------------------------------------------------------
# MaxPool2D
# ---------------------------------------------------------------------------

class MaxPool2D(Layer):
    """Макс-пулинг с сохранением индексов для backward.

    Форма:
        X:   (N, C, H, W)
        out: (N, C, H_out, W_out),  H_out = (H − kH)//stride + 1
    """

    def __init__(self, kernel_size: int = 2, stride: int | None = None) -> None:
        self.kH = kernel_size
        self.kW = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.X_shape: tuple[int, int, int, int] | None = None
        self.argmax: np.ndarray | None = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if X.ndim != 4:
            raise ValueError(
                f"MaxPool2D ожидает 4D-вход, получено {X.shape}"
            )
        self.X_shape = X.shape
        N, C, H, W = X.shape
        kH, kW, s = self.kH, self.kW, self.stride
        H_out = (H - kH) // s + 1
        W_out = (W - kW) // s + 1

        out = np.zeros((N, C, H_out, W_out), dtype=X.dtype)
        argmax = np.zeros((N, C, H_out, W_out, 2), dtype=np.int64)

        for i in range(H_out):
            hs = i * s
            for j in range(W_out):
                ws = j * s
                patch = X[:, :, hs:hs + kH, ws:ws + kW]
                patch_flat = patch.reshape(N, C, -1)
                idx = patch_flat.argmax(axis=2)
                out[:, :, i, j] = np.take_along_axis(
                    patch_flat, idx[..., None], axis=2,
                ).squeeze(-1)
                argmax[:, :, i, j, 0] = idx // kW
                argmax[:, :, i, j, 1] = idx % kW

        self.argmax = argmax
        return out

    def backward(self, dOut: np.ndarray) -> np.ndarray:
        N, C, H, W = self.X_shape
        kH, kW, s = self.kH, self.kW, self.stride
        H_out, W_out = dOut.shape[2], dOut.shape[3]

        dX = np.zeros(self.X_shape, dtype=dOut.dtype)
        n_idx = np.arange(N)[:, None]
        c_idx = np.arange(C)[None, :]

        for i in range(H_out):
            hs = i * s
            for j in range(W_out):
                ws = j * s
                ai = self.argmax[:, :, i, j, 0]
                aj = self.argmax[:, :, i, j, 1]
                np.add.at(
                    dX,
                    (n_idx, c_idx, hs + ai, ws + aj),
                    dOut[:, :, i, j],
                )
        return dX


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

class Flatten(Layer):
    """(N, C, H, W) → (N, C·H·W), обратно — reshape."""

    def __init__(self) -> None:
        self.X_shape: tuple[int, ...] | None = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.X_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        return dA.reshape(self.X_shape)
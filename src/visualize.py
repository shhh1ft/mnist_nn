"""Визуализация: графики обучения, confusion matrix, ошибочные примеры.

Backend matplotlib принудительно Agg — файлы сохраняются без GUI.
Это важно для CI и для запуска на сервере без дисплея.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Графики обучения
# ---------------------------------------------------------------------------

def plot_history(history: dict, save_path: str | Path) -> Path:
    """Loss / accuracy / lr по эпохам. Возвращает путь к PNG."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    has_lr = "lr" in history and len(history["lr"]) > 0
    ncols = 3 if has_lr else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    axes = np.atleast_1d(axes)

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="train", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], label="val", linewidth=1.5)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train", linewidth=1.5)
    axes[1].plot(epochs, history["val_acc"], label="val", linewidth=1.5)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if has_lr:
        axes[2].plot(epochs, history["lr"], color="tab:red", linewidth=1.5)
        axes[2].set_title("Learning rate")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("lr")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str | Path,
    class_names: list[str] | None = None,
    normalize: bool = True,
) -> Path:
    """Тепловая карта матрицы ошибок с аннотациями."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(cm)
    C = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(C)]

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
        row_sum[row_sum == 0] = 1.0
        disp = cm.astype(np.float64) / row_sum
        title = "Confusion matrix (normalized)"
    else:
        disp = cm
        title = "Confusion matrix (counts)"

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(disp, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else disp.max() or 1.0)

    ax.set_xticks(np.arange(C))
    ax.set_yticks(np.arange(C))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = 0.5 if normalize else (disp.max() * 0.5 if disp.max() > 0 else 0.5)
    for i in range(C):
        for j in range(C):
            val = disp[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            color = "white" if val > threshold else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Ошибочные примеры
# ---------------------------------------------------------------------------

def plot_misclassified(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    save_path: str | Path,
    max_items: int = 25,
    image_shape: tuple[int, int] = (28, 28),
    cols: int = 5,
) -> Path:
    """Сетка изображений, где модель ошиблась.

    X:      (N, D) — «сырые» или нормализованные пиксели;
    y_true: (N,) метки;
    y_pred: (N,) предсказанные метки;
    probs:  (N, C) вероятности (для подписи confidence).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    probs = np.asarray(probs)

    wrong = np.where(y_true != y_pred)[0][:max_items]

    if len(wrong) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Ошибочных предсказаний нет",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return save_path

    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    axes = np.atleast_2d(axes)

    for ax, idx in zip(axes.ravel(), wrong):
        img = X[idx].reshape(image_shape)
        ax.imshow(img, cmap="gray")
        conf = float(probs[idx, y_pred[idx]])
        ax.set_title(
            f"true={int(y_true[idx])}, pred={int(y_pred[idx])}\n"
            f"p={conf:.2f}",
            fontsize=8,
        )
        ax.axis("off")

    # Спрятать пустые ячейки
    for k in range(len(wrong), rows * cols):
        axes.ravel()[k].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_comparison(
    histories: dict[str, dict],
    save_path: str | Path,
    title: str = "Сравнение моделей",
) -> Path:
    """Накладывает val_loss и val_acc нескольких моделей на один график.

    histories: {name: history_dict}, где history_dict содержит
               ключи 'val_loss', 'val_acc', опционально 'train_loss', 'train_acc'.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, hist in histories.items():
        if not hist or "val_loss" not in hist:
            continue
        epochs = range(1, len(hist["val_loss"]) + 1)
        axes[0].plot(epochs, hist["val_loss"], label=name, linewidth=1.6)
        axes[1].plot(epochs, hist["val_acc"], label=name, linewidth=1.6)

    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path
"""Графический интерфейс: рисуем цифру мышкой — модель распознаёт.

Запуск:
    python -m src.gui \
        --model models/cnn_best.npz \
        --config configs/cnn_baseline.json

    # или с автоопределением доступных моделей
    python -m src.gui

Зависимости: только tkinter (встроен в CPython на Windows/macOS; на Linux
может потребоваться `sudo apt install python3-tk`), numpy, matplotlib
(используется для чтения .png, но в GUI не нужен — импортируется только
в predict.py). Никаких новых внешних пакетов.
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from .model import build_model_from_config
from .preprocess import mnist_preprocess
from .utils import softmax


# ---------------------------------------------------------------------------
# Параметры отрисовки
# ---------------------------------------------------------------------------

CANVAS_SIZE = 280          # размер холста в пикселях
GRID = 28                  # размер, который ждёт модель
CELL = CANVAS_SIZE // GRID  # 10 пикселей
BRUSH_RADIUS = 12          # радиус «кисти» в пикселях холста
BRUSH_STEP = 2             # шаг интерполяции вдоль линии


# ---------------------------------------------------------------------------
# Работа с буфером рисования
# ---------------------------------------------------------------------------

def _fill_circle(buf: np.ndarray, cx: int, cy: int, r: int) -> None:
    """Заливает круг радиуса r в буфере (in-place)."""
    H, W = buf.shape
    y0 = max(0, cy - r)
    y1 = min(H, cy + r + 1)
    x0 = max(0, cx - r)
    x1 = min(W, cx + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    buf[y0:y1, x0:x1][mask] = 1.0


def _draw_segment(
    buf: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    r: int = BRUSH_RADIUS,
    step: int = BRUSH_STEP,
) -> None:
    """Рисует отрезок между двумя точками как последовательность кругов."""
    dx = x1 - x0
    dy = y1 - y0
    length = float(np.hypot(dx, dy))
    n = max(1, int(length / step))
    for k in range(n + 1):
        t = k / n
        cx = int(round(x0 + dx * t))
        cy = int(round(y0 + dy * t))
        _fill_circle(buf, cx, cy, r)


def downsample_to_grid(buf: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Средний пул: (280, 280) → (28, 28) float32 в [0, 1]."""
    H, W = buf.shape
    if H % grid != 0 or W % grid != 0:
        raise ValueError(f"размер {H}×{W} не делится на {grid}")
    kh, kw = H // grid, W // grid
    return buf.reshape(grid, kh, grid, kw).mean(axis=(1, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class DrawApp:
    def __init__(
        self,
        root: tk.Tk,
        model,
        model_type: str,
        model_name: str,
        use_preprocess: bool = True,
    ) -> None:
        self.root = root
        self.model = model
        self.model_type = model_type
        self.model_name = model_name
        self.use_preprocess = tk.BooleanVar(value=use_preprocess)

        self.buffer = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
        self.last_xy: tuple[int, int] | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root.title(
            f"MNIST — распознавание нарисованной цифры  ·  модель: {self.model_name}"
        )

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # --- левая часть: холст --------------------------------------
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        self.canvas = tk.Canvas(
            left,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            highlightthickness=1,
            highlightbackground="#888",
        )
        self.canvas.pack()

        # сетка 28×28 (опционально) — лёгкие серые линии
        for i in range(1, GRID):
            p = i * CELL
            self.canvas.create_line(p, 0, p, CANVAS_SIZE, fill="#f0f0f0")
            self.canvas.create_line(0, p, CANVAS_SIZE, p, fill="#f0f0f0")

        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # --- правая часть: панель ------------------------------------
        right = ttk.Frame(main, padding=(20, 0, 0, 0))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(
            right, text="Предсказание", font=("Segoe UI", 12, "bold"),
        ).pack(anchor=tk.W)

        self.pred_var = tk.StringVar(value="—")
        ttk.Label(
            right, textvariable=self.pred_var,
            font=("Segoe UI", 56, "bold"), foreground="#1f6feb",
        ).pack(anchor=tk.W, pady=(0, 4))

        self.conf_var = tk.StringVar(value="")
        ttk.Label(
            right, textvariable=self.conf_var,
            font=("Segoe UI", 11),
        ).pack(anchor=tk.W, pady=(0, 12))

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(
            right, text="Распределение вероятностей",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor=tk.W)

        # 10 строк-полос для классов 0..9
        self.bars_frame = ttk.Frame(right)
        self.bars_frame.pack(fill=tk.X, pady=(6, 12))
        self.bars: list[ttk.Progressbar] = []
        self.prob_labels: list[tk.StringVar] = []

        for d in range(10):
            row = ttk.Frame(self.bars_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{d}", width=2,
                      font=("Consolas", 10, "bold")).pack(side=tk.LEFT)
            bar = ttk.Progressbar(
                row, orient=tk.HORIZONTAL, length=200, maximum=1.0,
            )
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 6))
            self.bars.append(bar)
            var = tk.StringVar(value="0.000")
            self.prob_labels.append(var)
            ttk.Label(row, textvariable=var, width=6,
                      font=("Consolas", 9)).pack(side=tk.LEFT)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Checkbutton(
            right,
            text="MNIST-предобработка (crop + center)",
            variable=self.use_preprocess,
            command=self._repredict,
        ).pack(anchor=tk.W, pady=(0, 8))

        btns = ttk.Frame(right)
        btns.pack(fill=tk.X)

        ttk.Button(btns, text="Очистить", command=self._clear).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4),
        )
        ttk.Button(btns, text="Распознать", command=self._repredict).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0),
        )

    # ------------------------------------------------------------------
    # События мыши
    # ------------------------------------------------------------------
    def _on_press(self, event: tk.Event) -> None:
        self.last_xy = (event.x, event.y)
        _fill_circle(self.buffer, event.x, event.y, BRUSH_RADIUS)
        self.canvas.create_oval(
            event.x - BRUSH_RADIUS, event.y - BRUSH_RADIUS,
            event.x + BRUSH_RADIUS, event.y + BRUSH_RADIUS,
            fill="black", outline="black",
        )
        self._clear_prediction()

    def _on_motion(self, event: tk.Event) -> None:
        if self.last_xy is None:
            return
        x0, y0 = self.last_xy
        x1, y1 = event.x, event.y
        _draw_segment(self.buffer, x0, y0, x1, y1)

        # визуальная линия на холсте
        self.canvas.create_line(
            x0, y0, x1, y1,
            fill="black",
            width=2 * BRUSH_RADIUS,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.last_xy = (x1, y1)

    def _on_release(self, event: tk.Event) -> None:
        self.last_xy = None
        self._repredict()

    # ------------------------------------------------------------------
    # Предсказание
    # ------------------------------------------------------------------
    def _clear_prediction(self) -> None:
        self.pred_var.set("—")
        self.conf_var.set("")
        for bar, var in zip(self.bars, self.prob_labels):
            bar["value"] = 0.0
            var.set("0.000")

    def _clear(self) -> None:
        self.buffer.fill(0.0)
        self.canvas.delete("all")
        # перерисовать сетку
        for i in range(1, GRID):
            p = i * CELL
            self.canvas.create_line(p, 0, p, CANVAS_SIZE, fill="#f0f0f0")
            self.canvas.create_line(0, p, CANVAS_SIZE, p, fill="#f0f0f0")
        self._clear_prediction()

    def _repredict(self) -> None:
        if self.buffer.max(initial=0.0) < 0.01:
            self._clear_prediction()
            return

        img28 = downsample_to_grid(self.buffer, GRID)

        if self.use_preprocess.get():
            img28 = mnist_preprocess(img28, size=28, inner=20)

        if self.model_type == "cnn":
            X = img28.reshape(1, 1, 28, 28).astype(np.float32)
        else:
            X = img28.reshape(1, -1).astype(np.float32)

        logits = self.model.forward(X, training=False)
        probs = softmax(logits, axis=1)[0]

        pred = int(probs.argmax())
        conf = float(probs[pred])

        self.pred_var.set(str(pred))
        self.conf_var.set(f"уверенность: {conf * 100:.1f}%")

        for d in range(10):
            p = float(probs[d])
            self.bars[d]["value"] = p
            self.prob_labels[d].set(f"{p:.3f}")


# ---------------------------------------------------------------------------
# Автоопределение доступных моделей
# ---------------------------------------------------------------------------

def _discover_models(
    models_dir: Path = Path("models"),
    configs_dir: Path = Path("configs"),
) -> list[tuple[str, Path, Path]]:
    """Возвращает список (name, model_path, config_path), где name совпадает
    по stem между .npz в models/ и .json в configs/.

    Порядок: сначала cnn_*, потом mlp_* — CNN обычно точнее.
    """
    if not models_dir.exists() or not configs_dir.exists():
        return []

    found: list[tuple[str, Path, Path]] = []
    for npz in sorted(models_dir.glob("*.npz")):
        stem = npz.stem.replace("_best", "")
        # ищем конфиг: configs/{stem}.json
        candidates = [
            configs_dir / f"{stem}.json",
            configs_dir / f"{npz.stem}.json",
        ]
        for cfg in candidates:
            if cfg.exists():
                found.append((stem, npz, cfg))
                break

    def _key(item: tuple[str, Path, Path]) -> tuple[int, str]:
        name = item[0]
        priority = 0 if name.startswith("cnn") else 1
        return (priority, name)

    return sorted(found, key=_key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GUI: нарисовать цифру и распознать её",
    )
    p.add_argument("--model", type=str, default=None,
                   help="Путь к .npz с весами (если не задан — автоопределение)")
    p.add_argument("--config", type=str, default=None,
                   help="Путь к JSON-конфигу модели")
    p.add_argument("--no-preprocess", action="store_true",
                   help="Отключить MNIST-предобработку по умолчанию")
    return p.parse_args()


def _resolve_model(args) -> tuple[str, Path, Path]:
    if args.model and args.config:
        return (
            Path(args.model).stem,
            Path(args.model),
            Path(args.config),
        )

    found = _discover_models()
    if not found:
        raise SystemExit(
            "Не найдено ни одной обученной модели.\n"
            "Обучите модель:  python -m src.train --config configs/mlp_baseline.json\n"
            "Или передайте пути явно:  --model models/cnn_best.npz "
            "--config configs/cnn_baseline.json"
        )
    name, model_path, config_path = found[0]
    print(f"[auto] выбрана модель: {name} ({model_path})")
    print(f"[auto] конфиг: {config_path}")
    if len(found) > 1:
        print("[auto] доступные модели:")
        for n, m, _ in found:
            print(f"        - {n}: {m}")
    return name, model_path, config_path


def main() -> None:
    args = parse_args()
    name, model_path, config_path = _resolve_model(args)

    if not model_path.exists():
        raise SystemExit(f"Модель не найдена: {model_path}")
    if not config_path.exists():
        raise SystemExit(f"Конфиг не найден: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    model = build_model_from_config(config, seed=config.get("seed", 42))
    model.load(model_path)
    model_type = config["model"].get("type", "mlp")

    root = tk.Tk()
    app = DrawApp(
        root,
        model=model,
        model_type=model_type,
        model_name=name,
        use_preprocess=not args.no_preprocess,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
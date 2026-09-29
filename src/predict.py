"""CLI предсказания: одиночное изображение или директория."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .model import build_model_from_config
from .utils import softmax


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Предсказание цифры MNIST")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def load_image(path: Path) -> np.ndarray:
    img = plt.imread(str(path))
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    img = img.astype(np.float32)
    if img.max() > 1.0 + 1e-6:
        img = img / 255.0
    if img.shape != (28, 28):
        raise ValueError(f"{path.name}: ожидалось 28×28, получено {img.shape}")
    return img


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTS:
            raise SystemExit(f"Формат {path.suffix} не поддерживается")
        return [path]
    if path.is_dir():
        files = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )
        if not files:
            raise SystemExit(f"В директории {path} нет изображений")
        return files
    raise SystemExit(f"Путь не найден: {path}")


def prepare_input(img: np.ndarray, model_type: str) -> np.ndarray:
    """MLP: (1, 784); CNN: (1, 1, 28, 28)."""
    if model_type == "cnn":
        return img.reshape(1, 1, 28, 28)
    return img.reshape(1, -1)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    model_path = Path(args.model)
    config_path = Path(args.config)

    for p in (model_path, config_path):
        if not p.exists():
            raise SystemExit(f"Не найдено: {p}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    model = build_model_from_config(config, seed=config.get("seed", 42))
    model.load(model_path)

    model_type = config["model"].get("type", "mlp")
    files = collect_images(image_path)
    multi = len(files) > 1

    for path in files:
        try:
            img = load_image(path)
        except Exception as e:
            print(f"[skip] {path.name}: {e}")
            continue

        X = prepare_input(img, model_type)
        logits = model.forward(X, training=False)
        probs = softmax(logits, axis=1)[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])

        if multi:
            print(f"{path.name:>30}  →  {pred}  (p={conf:.3f})")
        else:
            print(f"[predict] image = {path}")
            print(f"[predict] prediction = {pred}  (p={conf:.3f})")
            print("[predict] probabilities:")
            top_k = min(args.top_k, len(probs))
            order = np.argsort(probs)[::-1][:top_k]
            for digit in order:
                p = float(probs[digit])
                bar = "█" * int(round(p * 40))
                print(f"  {digit}: {p:.4f} {bar}")

    if multi:
        print(f"\n[summary] обработано файлов: {len(files)}")


if __name__ == "__main__":
    main()
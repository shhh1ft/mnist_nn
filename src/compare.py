"""Сравнение нескольких обученных моделей на тестовой выборке MNIST.

Пример:
    python -m src.compare \
        --entry mlp=models/mlp_best.npz,configs/mlp_baseline.json \
        --entry cnn=models/cnn_best.npz,configs/cnn_baseline.json \
        --entry cnn_deep=models/cnn_best.npz,configs/cnn_deep.json \
        --tag comparison

Создаёт:
    reports/{tag}_table.md       — markdown-таблица метрик;
    reports/{tag}_curves.png     — val_loss / val_acc всех моделей;
    reports/{tag}_summary.json   — метрики в машинночитаемом виде.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data_loader import load_mnist
from .losses import SoftmaxCrossEntropy
from .metrics import classification_report
from .model import build_model_from_config
from .train import reshape_for_model
from .utils import ensure_dir, softmax
from .visualize import plot_comparison


# ---------------------------------------------------------------------------
# Парсинг аргумента --entry
# ---------------------------------------------------------------------------

def parse_entry(s: str) -> tuple[str, Path, Path]:
    """'name=path/to/model.npz,path/to/config.json' → (name, model_path, config_path)."""
    if "=" not in s or "," not in s:
        raise argparse.ArgumentTypeError(
            "Формат: name=model_path,config_path"
        )
    name, rest = s.split("=", 1)
    model_path, config_path = rest.split(",", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Имя модели не может быть пустым")
    return name, Path(model_path.strip()), Path(config_path.strip())


# ---------------------------------------------------------------------------
# Оценка одной модели
# ---------------------------------------------------------------------------

def count_parameters(model) -> int:
    return int(sum(p.size for p, _ in model.params_and_grads()))


def evaluate_model(
    model,
    config: dict,
    X_flat: np.ndarray,
    y: np.ndarray,
    loss_fn: SoftmaxCrossEntropy,
) -> dict:
    """Возвращает словарь метрик и предсказаний для одной модели."""
    model_type = config["model"].get("type", "mlp")
    X = reshape_for_model(X_flat, model_type)

    t0 = time.perf_counter()
    logits = model.forward(X, training=False)
    t1 = time.perf_counter()

    loss = loss_fn.forward(logits, y)
    probs = softmax(logits, axis=1)
    y_true = y.argmax(axis=1)
    y_pred = probs.argmax(axis=1)

    report = classification_report(y_true, y_pred, num_classes=10)

    return {
        "accuracy": float(report["accuracy"]),
        "loss": float(loss),
        "macro_f1": float(report["macro"]["f1"]),
        "n_params": count_parameters(model),
        "inference_ms": (t1 - t0) * 1000.0,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
        "confusion_matrix": report["confusion_matrix"],
    }


# ---------------------------------------------------------------------------
# Формирование таблицы
# ---------------------------------------------------------------------------

def format_comparison_table(rows: list[dict]) -> str:
    """rows: [{name, n_params, accuracy, loss, macro_f1, inference_ms}, ...]."""
    header = (
        "| Model | Params | Test Acc | Test Loss | Macro F1 | Inference (ms) |\n"
        "|-------|--------|----------|-----------|----------|----------------|\n"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['name']} "
            f"| {r['n_params']:,} "
            f"| {r['accuracy']:.4f} "
            f"| {r['loss']:.4f} "
            f"| {r['macro_f1']:.4f} "
            f"| {r['inference_ms']:.0f} |\n"
        )
    return "".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Сравнение моделей на MNIST")
    p.add_argument(
        "--entry", action="append", required=True, type=parse_entry,
        help="name=model_path,config_path (можно повторять)",
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--reports-dir", type=str, default="reports")
    p.add_argument("--tag", type=str, default="comparison")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(
        args.data_dir,
        val_size=0.1,
        seed=42,
    )
    if args.split == "val":
        X, y = X_val, y_val
    else:
        X, y = X_test, y_test
    print(f"[data] split={args.split}  N={len(X)}")

    loss_fn = SoftmaxCrossEntropy()

    summary_rows: list[dict] = []
    histories: dict[str, dict] = {}
    predictions: dict[str, dict] = {}

    for name, model_path, config_path in args.entry:
        if not model_path.exists():
            print(f"[skip] {name}: модель не найдена ({model_path})")
            continue
        if not config_path.exists():
            print(f"[skip] {name}: конфиг не найден ({config_path})")
            continue

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        model = build_model_from_config(config, seed=config.get("seed", 42))
        model.load(model_path)

        metrics = evaluate_model(model, config, X, y, loss_fn)
        print(
            f"[{name}] acc={metrics['accuracy']:.4f} "
            f"loss={metrics['loss']:.4f} "
            f"params={metrics['n_params']:,} "
            f"inference={metrics['inference_ms']:.0f} ms"
        )

        summary_rows.append({
            "name": name,
            "n_params": metrics["n_params"],
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "macro_f1": metrics["macro_f1"],
            "inference_ms": metrics["inference_ms"],
        })
        predictions[name] = {
            "y_true": metrics["y_true"],
            "y_pred": metrics["y_pred"],
            "probs": metrics["probs"],
        }

        # --- история обучения (если сохранена) -----------------------
        hist_candidates = [
            Path("reports") / f"{config['model'].get('type', 'mlp')}_history.json",
            Path("reports") / "history.json",
        ]
        for hp in hist_candidates:
            if hp.exists():
                try:
                    with open(hp, encoding="utf-8") as f:
                        data = json.load(f)
                    hist = data.get("history", data)
                    if isinstance(hist, dict) and "val_loss" in hist:
                        histories[name] = hist
                        break
                except Exception:
                    pass

    # --- Отчёт -------------------------------------------------------
    reports_dir = ensure_dir(args.reports_dir)
    table_md = format_comparison_table(summary_rows)

    md_path = reports_dir / f"{args.tag}_table.md"
    md_path.write_text(
        f"# Сравнение моделей ({args.split})\n\n" + table_md,
        encoding="utf-8",
    )
    print(f"[save] {md_path}")

    json_path = reports_dir / f"{args.tag}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "rows": summary_rows}, f, indent=2)
    print(f"[save] {json_path}")

    if histories:
        curves_path = reports_dir / f"{args.tag}_curves.png"
        plot_comparison(histories, curves_path,
                        title=f"MLP vs CNN ({args.split})")
        print(f"[save] {curves_path}")

    # --- Предсказания для последующего анализа ошибок ---------------
    pred_path = reports_dir / f"{args.tag}_predictions.npz"
    np.savez_compressed(
        pred_path,
        **{f"{name}_y_true": p["y_true"] for name, p in predictions.items()},
        **{f"{name}_y_pred": p["y_pred"] for name, p in predictions.items()},
    )
    print(f"[save] {pred_path}")

    print()
    print(table_md)


if __name__ == "__main__":
    main()
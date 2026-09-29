"""Замер inference-времени моделей на CPU.

Пример:
    python -m benchmarks.benchmark \
        --entry mlp=models/mlp_best.npz,configs/mlp_baseline.json \
        --entry cnn=models/cnn_best.npz,configs/cnn_baseline.json \
        --runs 30 --warmup 5

Сохраняет:
    benchmarks/results.json
    benchmarks/benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from src.compare import parse_entry
from src.data_loader import load_mnist
from src.model import build_model_from_config
from src.train import reshape_for_model
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Бенчмарк inference на CPU")
    p.add_argument("--entry", action="append", required=True, type=parse_entry)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="benchmarks")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128])
    p.add_argument("--samples", type=int, default=1000,
                   help="Сколько тестовых примеров брать")
    return p.parse_args()


def time_model(model, X: np.ndarray, runs: int, warmup: int) -> list[float]:
    """Возвращает список времён в миллисекундах."""
    # warmup
    for _ in range(warmup):
        _ = model.forward(X, training=False)

    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.forward(X, training=False)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def main() -> None:
    args = parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(
        args.data_dir, val_size=0.1, seed=42,
    )
    X_test = X_test[:args.samples]
    print(f"[data] тестовых примеров для бенчмарка: {len(X_test)}")

    output_dir = ensure_dir(args.output_dir)

    runs_summary: list[dict] = []

    for name, model_path, config_path in args.entry:
        if not model_path.exists() or not config_path.exists():
            print(f"[skip] {name}: файлы не найдены")
            continue

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        model = build_model_from_config(config, seed=config.get("seed", 42))
        model.load(model_path)

        model_type = config["model"].get("type", "mlp")

        for batch in args.batches:
            n = min(batch, len(X_test))
            Xb = reshape_for_model(X_test[:n], model_type)

            times = time_model(model, Xb, args.runs, args.warmup)
            arr = np.array(times)

            runs_summary.append({
                "model": name,
                "batch": int(n),
                "runs": int(args.runs),
                "mean_ms": float(arr.mean()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "std_ms": float(arr.std()),
                "min_ms": float(arr.min()),
                "max_ms": float(arr.max()),
            })

            print(
                f"[{name}] batch={n:4d} "
                f"mean={arr.mean():7.2f} ms  "
                f"p50={np.percentile(arr, 50):7.2f}  "
                f"p95={np.percentile(arr, 95):7.2f}"
            )

    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"runs": runs_summary}, f, indent=2)
    print(f"[save] {json_path}")

    csv_path = output_dir / "benchmark.csv"
    if runs_summary:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(runs_summary[0].keys()))
            writer.writeheader()
            writer.writerows(runs_summary)
        print(f"[save] {csv_path}")


if __name__ == "__main__":
    main()
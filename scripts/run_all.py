"""CLI: полный пайплайн — обучение, оценка, сравнение, бенчмарк, отчёт.

Пример:
    python -m scripts.run_all --fast         # 5 эпох на каждой модели
    python -m scripts.run_all                # полный прогон по конфигам
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[warn] команда завершилась с кодом {result.returncode}")


def main() -> None:
    p = argparse.ArgumentParser(description="Полный пайплайн mnist_nn")
    p.add_argument("--fast", action="store_true",
                   help="Быстрый прогон (5 эпох, MLP + CNN baseline)")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-tests", action="store_true")
    args = p.parse_args()

    py = sys.executable

    if not args.skip_tests:
        run([py, "-m", "pytest", "tests/", "-q"])

    # --- данные ------------------------------------------------------
    run([py, "-m", "scripts.download_mnist"])
    run([py, "-m", "scripts.prepare_data"])

    # --- обучение ----------------------------------------------------
    if not args.skip_train:
        configs = [
            ("mlp", "configs/mlp_baseline.json"),
            ("cnn", "configs/cnn_baseline.json"),
        ]
        if not args.fast:
            configs += [
                ("cnn_deep", "configs/cnn_deep.json"),
                ("cnn_wide", "configs/cnn_wide.json"),
            ]
        for name, cfg in configs:
            run([py, "-m", "src.train", "--config", cfg])

    # --- оценка ------------------------------------------------------
    if Path("models/mlp_best.npz").exists():
        run([py, "-m", "src.evaluate",
             "--model", "models/mlp_best.npz",
             "--config", "configs/mlp_baseline.json",
             "--tag", "mlp"])
    if Path("models/cnn_best.npz").exists():
        run([py, "-m", "src.evaluate",
             "--model", "models/cnn_best.npz",
             "--config", "configs/cnn_baseline.json",
             "--tag", "cnn"])

    # --- сравнение ---------------------------------------------------
    entries = []
    if Path("models/mlp_best.npz").exists():
        entries += ["--entry", "mlp=models/mlp_best.npz,configs/mlp_baseline.json"]
    if Path("models/cnn_best.npz").exists():
        entries += ["--entry", "cnn=models/cnn_best.npz,configs/cnn_baseline.json"]
    if not args.fast:
        if Path("models/cnn_best.npz").exists():
            entries += [
                "--entry", "cnn_deep=models/cnn_best.npz,configs/cnn_deep.json",
                "--entry", "cnn_wide=models/cnn_best.npz,configs/cnn_wide.json",
            ]
    if entries:
        run([py, "-m", "src.compare", *entries, "--tag", "comparison"])

    # --- анализ ошибок ----------------------------------------------
    if (Path("models/mlp_best.npz").exists()
            and Path("models/cnn_best.npz").exists()):
        run([py, "-m", "src.analyze_errors",
             "--a", "mlp=models/mlp_best.npz,configs/mlp_baseline.json",
             "--b", "cnn=models/cnn_best.npz,configs/cnn_baseline.json",
             "--tag", "mlp_vs_cnn"])

    # --- бенчмарк ---------------------------------------------------
    if entries:
        run([py, "-m", "benchmarks.benchmark", *entries,
             "--runs", "10", "--warmup", "3"])

    # --- финальный отчёт --------------------------------------------
    run([py, "-m", "src.report", "--tag", "final"])

    print("\n[done] Все артефакты в reports/ и benchmarks/")


if __name__ == "__main__":
    main()
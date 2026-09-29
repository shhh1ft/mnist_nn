"""CLI: подготовка MNIST (нормализация, split, кэш в .npz)."""

from __future__ import annotations

import argparse

from src.data_loader import load_mnist


def main() -> None:
    p = argparse.ArgumentParser(description="Подготовить MNIST")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(
        data_dir=args.data_dir,
        val_size=args.val_size,
        seed=args.seed,
        use_cache=not args.no_cache,
        download=not args.no_download,
    )
    print(f"[train] X={X_train.shape}  y={y_train.shape}")
    print(f"[val]   X={X_val.shape}  y={y_val.shape}")
    print(f"[test]  X={X_test.shape}  y={y_test.shape}")


if __name__ == "__main__":
    main()
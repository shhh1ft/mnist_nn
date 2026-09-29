"""CLI: скачивает 4 gz-файла MNIST в data/raw/."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import download_mnist


def main() -> None:
    p = argparse.ArgumentParser(description="Скачать MNIST в data/raw/")
    p.add_argument("--dir", type=str, default="data/raw")
    args = p.parse_args()
    path = download_mnist(Path(args.dir))
    print(f"[done] MNIST в {path}")


if __name__ == "__main__":
    main()
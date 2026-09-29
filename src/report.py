"""Генератор итогового отчёта reports/final_report.md.

Склеивает:
- таблицу сравнения моделей (из reports/comparison_summary.json);
- график кривых обучения (если есть);
- confusion matrix и примеры ошибок (если есть);
- метрики из reports/*_metrics.json.

Пример:
    python -m src.report --tag final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import ensure_dir


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _section_table_md(rows: list[dict]) -> str:
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


def build_report(reports_dir: Path, tag: str) -> str:
    """Собирает markdown-отчёт и возвращает его как строку."""
    parts: list[str] = []
    parts.append(f"# Итоговый отчёт по проекту mnist_nn\n\n")
    parts.append(
        "Реализация, обучение и оценка нейронной сети для распознавания "
        "рукописных цифр MNIST **без использования ML-фреймворков**.\n\n"
    )

    # --- 1. Сравнение моделей ---------------------------------------
    summary = _read_json(reports_dir / f"{tag}_summary.json")
    if summary is None:
        summary = _read_json(reports_dir / "comparison_summary.json")

    parts.append("## 1. Сравнение моделей\n\n")
    if summary and summary.get("rows"):
        parts.append(f"Разбиение: **{summary.get('split', 'test')}**\n\n")
        parts.append(_section_table_md(summary["rows"]))
        parts.append("\n")

        curves = reports_dir / f"{tag}_curves.png"
        if curves.exists():
            parts.append(f"![Кривые обучения]({curves.name})\n\n")
    else:
        parts.append("_Сводка сравнения не найдена. "
                     "Запустите `python -m src.compare`._\n\n")

    # --- 2. Метрики по каждой модели --------------------------------
    parts.append("## 2. Детальные метрики\n\n")
    metric_files = sorted(reports_dir.glob("*_metrics.json"))
    if metric_files:
        for mp in metric_files:
            data = _read_json(mp) or {}
            if not data:
                continue
            name = mp.stem.replace("_metrics", "")
            parts.append(f"### Модель `{name}`\n\n")
            if "accuracy" in data:
                parts.append(f"- accuracy: **{data['accuracy']:.4f}**\n")
            if "loss" in data:
                parts.append(f"- loss: {data['loss']:.4f}\n")
            if "macro" in data and isinstance(data["macro"], dict):
                parts.append(
                    f"- macro F1: {data['macro'].get('f1', 0.0):.4f}\n"
                )
            if "weighted" in data and isinstance(data["weighted"], dict):
                parts.append(
                    f"- weighted F1: {data['weighted'].get('f1', 0.0):.4f}\n"
                )
            parts.append("\n")

            cm_png = reports_dir / f"{name}_confusion_matrix.png"
            if cm_png.exists():
                parts.append(f"![Confusion matrix]({cm_png.name})\n\n")

            wrong_png = reports_dir / f"{name}_misclassified.png"
            if wrong_png.exists():
                parts.append(f"![Ошибочные примеры]({wrong_png.name})\n\n")
    else:
        parts.append("_Метрики не найдены. "
                     "Запустите `python -m src.evaluate`._\n\n")

    # --- 3. Анализ ошибок --------------------------------------------
    analysis = _read_json(reports_dir / f"{tag}_analysis.json")
    if analysis is None:
        analysis = _read_json(reports_dir / "mlp_vs_cnn_analysis.json")

    parts.append("## 3. Анализ расхождений\n\n")
    if analysis:
        c = analysis.get("counts", {})
        a = analysis.get("model_a", {})
        b = analysis.get("model_b", {})
        parts.append(
            f"- **{a.get('name', 'A')}** accuracy = "
            f"{a.get('accuracy', 0.0):.4f}\n"
        )
        parts.append(
            f"- **{b.get('name', 'B')}** accuracy = "
            f"{b.get('accuracy', 0.0):.4f}\n"
        )
        parts.append(f"- A ошиблась, B угадала: {c.get('a_wrong_b_right', 0)}\n")
        parts.append(f"- B ошиблась, A угадала: {c.get('b_wrong_a_right', 0)}\n")
        parts.append(f"- Оба ошиблись: {c.get('both_wrong', 0)}\n\n")

        for img in sorted(reports_dir.glob(f"{tag}_*wrong*_right.png")):
            parts.append(f"![{img.stem}]({img.name})\n\n")
    else:
        parts.append("_Анализ расхождений не найден. "
                     "Запустите `python -m src.analyze_errors`._\n\n")

    # --- 4. Бенчмарк ---------------------------------------------------
    bench = _read_json(reports_dir / "benchmark.json") or \
            _read_json(Path("benchmarks") / "results.json")
    parts.append("## 4. Бенчмарк inference\n\n")
    if bench and "runs" in bench:
        parts.append("| Model | Batch | Runs | Mean (ms) | p50 (ms) | p95 (ms) |\n")
        parts.append("|-------|-------|------|-----------|----------|----------|\n")
        for r in bench["runs"]:
            parts.append(
                f"| {r['model']} | {r['batch']} | {r['runs']} "
                f"| {r['mean_ms']:.2f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} |\n"
            )
        parts.append("\n")
    else:
        parts.append("_Бенчмарк не найден. "
                     "Запустите `python -m benchmarks.benchmark`._\n\n")

    # --- 5. Воспроизведение ------------------------------------------
    parts.append("## 5. Воспроизведение\n\n")
    parts.append(
        "Полный пайплайн воспроизводится командами:\n\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "python -m src.train --config configs/mlp_baseline.json\n"
        "python -m src.train --config configs/cnn_baseline.json\n"
        "python -m src.evaluate --model models/cnn_best.npz "
        "--config configs/cnn_baseline.json --tag cnn\n"
        "python -m src.compare --entry mlp=... --entry cnn=...\n"
        "python -m benchmarks.benchmark --entry cnn=...\n"
        "python -m src.report --tag final\n"
        "```\n"
    )

    return "".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description="Сборка итогового отчёта")
    p.add_argument("--reports-dir", type=str, default="reports")
    p.add_argument("--tag", type=str, default="final")
    p.add_argument("--output", type=str, default=None,
                   help="Путь к файлу (по умолчанию reports/{tag}_report.md)")
    args = p.parse_args()

    reports_dir = ensure_dir(args.reports_dir)
    text = build_report(reports_dir, args.tag)

    output = Path(args.output) if args.output else \
        reports_dir / f"{args.tag}_report.md"
    output.write_text(text, encoding="utf-8")
    print(f"[save] {output}")


if __name__ == "__main__":
    main()
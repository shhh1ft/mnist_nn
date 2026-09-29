"""CLI обучения модели (недели 3–6)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data_loader import load_mnist
from .losses import SoftmaxCrossEntropy
from .model import build_model_from_config
from .optimizers import SGD, Adam
from .schedulers import build_scheduler_from_config
from .trainer import Trainer
from .utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение нейросети на MNIST")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="models")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def build_optimizer(train_cfg: dict):
    name = train_cfg.get("optimizer", "sgd").lower()
    lr = train_cfg.get("learning_rate", 0.1)
    wd = train_cfg.get("l2", 0.0)
    clip = train_cfg.get("gradient_clip")

    if name == "sgd":
        return SGD(
            lr=lr,
            momentum=train_cfg.get("momentum", 0.0),
            weight_decay=wd,
            clip_norm=clip,
        )
    if name == "adam":
        betas = train_cfg.get("betas", [0.9, 0.999])
        return Adam(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=train_cfg.get("eps", 1e-8),
            weight_decay=wd,
            clip_norm=clip,
        )
    raise ValueError(f"Неизвестный оптимизатор: {name}")


def reshape_for_model(X: np.ndarray, model_type: str) -> np.ndarray:
    """Приводит (N, 784) → (N, 1, 28, 28) для CNN; для MLP — как есть."""
    if model_type == "cnn":
        return X.reshape(-1, 1, 28, 28)
    return X


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)

    print(f"[config] {args.config}")
    print(f"[seed]   {seed}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(
        args.data_dir,
        val_size=config["data"].get("val_size", 0.1),
        seed=seed,
        use_cache=not args.no_cache,
    )

    model_type = config["model"].get("type", "mlp")
    X_train = reshape_for_model(X_train, model_type)
    X_val = reshape_for_model(X_val, model_type)
    X_test = reshape_for_model(X_test, model_type)
    print(f"[data] type={model_type}  X_train {X_train.shape}  "
          f"X_val {X_val.shape}  X_test {X_test.shape}")

    model = build_model_from_config(config, seed=seed)
    n_params = sum(p.size for p, _ in model.params_and_grads())
    print(f"[model] слоёв={len(model.layers)}  параметров={n_params}")

    loss_fn = SoftmaxCrossEntropy()

    tr_cfg = config["training"]
    optimizer = build_optimizer(tr_cfg)
    print(
        f"[optim] {type(optimizer).__name__} "
        f"lr={optimizer.lr} weight_decay={optimizer.weight_decay} "
        f"clip_norm={optimizer.clip_norm}"
    )

    scheduler = build_scheduler_from_config(tr_cfg, optimizer.lr)
    print(f"[sched] {type(scheduler).__name__}")

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=tr_cfg.get("epochs", 20),
        batch_size=config["data"].get("batch_size", 64),
        early_stopping_patience=tr_cfg.get("early_stopping_patience", 5),
        scheduler=scheduler,
        seed=seed,
    )

    result = trainer.fit(X_train, y_train, X_val, y_val)

    val_loss, val_acc = trainer._evaluate(X_val, y_val)
    test_loss, test_acc = trainer._evaluate(X_test, y_test)
    print(f"[final] val  loss={val_loss:.4f}  acc={val_acc:.4f}")
    print(f"[final] test loss={test_loss:.4f}  acc={test_acc:.4f}")

    models_dir = ensure_dir(args.output_dir)
    ckpt = models_dir / f"{model_type}_best.npz"
    model.save(ckpt)
    print(f"[save] {ckpt}")

    reports_dir = ensure_dir("reports")
    hist_path = reports_dir / f"{model_type}_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[save] {hist_path}")

    cfg_path = reports_dir / f"{model_type}_last_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[save] {cfg_path}")


if __name__ == "__main__":
    main()
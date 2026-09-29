"""Цикл обучения с early stopping, снапшотами весов и LR-планировщиком."""

from __future__ import annotations

from typing import Any

import numpy as np

from .data_loader import DataLoader
from .model import Sequential
from .schedulers import ConstantLR


class Trainer:
    def __init__(
        self,
        model: Sequential,
        loss_fn,
        optimizer,
        epochs: int = 20,
        batch_size: int = 64,
        early_stopping_patience: int = 5,
        seed: int = 42,
        scheduler=None,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = early_stopping_patience
        self.seed = seed
        self.scheduler = scheduler if scheduler is not None else ConstantLR(optimizer.lr)
        self.verbose = verbose

        self.history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.best_weights: list[np.ndarray] | None = None
        self.patience_counter = 0

    # ------------------------------------------------------------------
    def _train_epoch(self, X: np.ndarray, y: np.ndarray, epoch: int) -> tuple[float, float]:
        loader = DataLoader(
            X, y,
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seed + epoch,
        )
        total_loss = 0.0
        total_correct = 0
        total = 0

        for X_b, y_b in loader:
            logits = self.model.forward(X_b, training=True)
            loss = self.loss_fn.forward(logits, y_b)
            self.model.backward(self.loss_fn.backward())
            self.optimizer.step(self.model)

            total_loss += loss * len(X_b)
            total_correct += int((logits.argmax(axis=1) == y_b.argmax(axis=1)).sum())
            total += len(X_b)

        return total_loss / total, total_correct / total

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        logits = self.model.forward(X, training=False)
        loss = self.loss_fn.forward(logits, y)
        acc = float((logits.argmax(axis=1) == y.argmax(axis=1)).mean())
        return loss, acc

    def _snapshot(self) -> list[np.ndarray]:
        return [param.copy() for param, _ in self.model.params_and_grads()]

    def _restore(self, snapshot: list[np.ndarray]) -> None:
        for (param, _), saved in zip(self.model.params_and_grads(), snapshot):
            param[...] = saved

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict[str, Any]:
        for epoch in range(self.epochs):
            # --- LR scheduler: 0-индексированная эпоха -----------------
            self.optimizer.lr = self.scheduler.get_lr(epoch)
            self.history["lr"].append(float(self.optimizer.lr))

            tr_loss, tr_acc = self._train_epoch(X_train, y_train, epoch)
            va_loss, va_acc = self._evaluate(X_val, y_val)

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)

            if self.verbose:
                print(
                    f"[epoch {epoch + 1:3d}] "
                    f"lr={self.optimizer.lr:.6g} "
                    f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                    f"val loss={va_loss:.4f} acc={va_acc:.4f}"
                )

            if va_loss < self.best_val_loss - 1e-6:
                self.best_val_loss = va_loss
                self.best_epoch = epoch
                self.best_weights = self._snapshot()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    if self.verbose:
                        print(
                            f"[early stop] нет улучшения {self.patience} эпох подряд"
                        )
                    break

        if self.best_weights is not None:
            self._restore(self.best_weights)
            if self.verbose:
                print(
                    f"[best] epoch={self.best_epoch + 1}  "
                    f"val_loss={self.best_val_loss:.4f}"
                )

        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }
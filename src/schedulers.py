"""Планировщики learning rate (неделя 4).

Интерфейс: scheduler.get_lr(epoch) -> float, где epoch — 0-индексированный
номер эпохи. Trainer вызывает метод в начале каждой эпохи и присваивает
значение optimizer.lr.
"""

from __future__ import annotations

import math


class ConstantLR:
    """Постоянный LR."""

    def __init__(self, base_lr: float) -> None:
        if base_lr <= 0:
            raise ValueError("base_lr должен быть положительным")
        self.base_lr = base_lr

    def get_lr(self, epoch: int) -> float:
        return self.base_lr


class StepDecay:
    """LR умножается на gamma каждые step_size эпох.

    lr(epoch) = base_lr · gamma ^ floor(epoch / step_size)
    """

    def __init__(
        self,
        base_lr: float,
        step_size: int,
        gamma: float = 0.5,
    ) -> None:
        if base_lr <= 0:
            raise ValueError("base_lr должен быть положительным")
        if step_size <= 0:
            raise ValueError("step_size должен быть положительным")
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma должен быть в интервале (0, 1)")
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch: int) -> float:
        return self.base_lr * (self.gamma ** (epoch // self.step_size))


class CosineAnnealing:
    """Косинусное затухание LR от base_lr до min_lr за t_max эпох.

    lr(epoch) = min_lr + 0.5·(base_lr − min_lr)·(1 + cos(π·epoch / t_max))
    """

    def __init__(
        self,
        base_lr: float,
        t_max: int,
        min_lr: float = 0.0,
    ) -> None:
        if base_lr <= 0:
            raise ValueError("base_lr должен быть положительным")
        if t_max <= 0:
            raise ValueError("t_max должен быть положительным")
        if min_lr < 0:
            raise ValueError("min_lr не может быть отрицательным")
        if min_lr > base_lr:
            raise ValueError("min_lr не может превышать base_lr")
        self.base_lr = base_lr
        self.t_max = t_max
        self.min_lr = min_lr

    def get_lr(self, epoch: int) -> float:
        # За пределами t_max держим min_lr
        e = min(epoch, self.t_max)
        cos_term = math.cos(math.pi * e / self.t_max)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + cos_term)


def build_scheduler_from_config(
    train_cfg: dict,
    base_lr: float,
) -> ConstantLR | StepDecay | CosineAnnealing:
    """Собирает планировщик из секции 'training' JSON-конфига.

    Если секции 'scheduler' нет — возвращает ConstantLR(base_lr).
    """
    spec = train_cfg.get("scheduler")
    if not spec:
        return ConstantLR(base_lr)

    kind = spec.get("type", "constant").lower()

    if kind == "constant":
        return ConstantLR(base_lr)
    if kind == "step":
        return StepDecay(
            base_lr=base_lr,
            step_size=spec.get("step_size", 10),
            gamma=spec.get("gamma", 0.5),
        )
    if kind in ("cosine", "cosine_annealing"):
        return CosineAnnealing(
            base_lr=base_lr,
            t_max=spec.get("t_max", train_cfg.get("epochs", 30)),
            min_lr=spec.get("min_lr", 0.0),
        )
    raise ValueError(f"Неизвестный тип планировщика: {kind}")
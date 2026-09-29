"""Тесты планировщиков learning rate (неделя 4)."""

from __future__ import annotations

import math

import pytest

from src.schedulers import (
    ConstantLR,
    CosineAnnealing,
    StepDecay,
    build_scheduler_from_config,
)


# ---------------------------------------------------------------------------
# ConstantLR
# ---------------------------------------------------------------------------

def test_constant_lr():
    s = ConstantLR(0.1)
    assert s.get_lr(0) == 0.1
    assert s.get_lr(100) == 0.1


def test_constant_lr_invalid():
    with pytest.raises(ValueError):
        ConstantLR(0.0)


# ---------------------------------------------------------------------------
# StepDecay
# ---------------------------------------------------------------------------

def test_step_decay_values():
    s = StepDecay(base_lr=0.1, step_size=10, gamma=0.5)
    assert s.get_lr(0) == 0.1
    assert s.get_lr(9) == 0.1
    assert s.get_lr(10) == 0.05
    assert s.get_lr(19) == 0.05
    assert s.get_lr(20) == 0.025
    assert s.get_lr(30) == 0.0125


def test_step_decay_invalid():
    with pytest.raises(ValueError):
        StepDecay(0.1, step_size=0)
    with pytest.raises(ValueError):
        StepDecay(0.1, step_size=10, gamma=0.0)
    with pytest.raises(ValueError):
        StepDecay(0.1, step_size=10, gamma=1.0)
    with pytest.raises(ValueError):
        StepDecay(0.0, step_size=10)


# ---------------------------------------------------------------------------
# CosineAnnealing
# ---------------------------------------------------------------------------

def test_cosine_annealing_ends():
    s = CosineAnnealing(base_lr=0.1, t_max=10, min_lr=0.0)
    assert abs(s.get_lr(0) - 0.1) < 1e-9
    assert abs(s.get_lr(10) - 0.0) < 1e-9


def test_cosine_annealing_midpoint():
    s = CosineAnnealing(base_lr=0.1, t_max=10, min_lr=0.0)
    # на половине: 0.5·0.1·(1 + cos(π/2)) = 0.05
    assert abs(s.get_lr(5) - 0.05) < 1e-9


def test_cosine_annealing_min_lr():
    s = CosineAnnealing(base_lr=0.1, t_max=10, min_lr=0.01)
    assert abs(s.get_lr(0) - 0.1) < 1e-9
    assert abs(s.get_lr(10) - 0.01) < 1e-9
    # после t_max остаётся min_lr
    assert abs(s.get_lr(50) - 0.01) < 1e-9


def test_cosine_annealing_monotonic_decreasing():
    s = CosineAnnealing(base_lr=0.1, t_max=20, min_lr=0.0)
    prev = s.get_lr(0)
    for e in range(1, 21):
        cur = s.get_lr(e)
        assert cur <= prev + 1e-12
        prev = cur


def test_cosine_annealing_invalid():
    with pytest.raises(ValueError):
        CosineAnnealing(0.1, t_max=0)
    with pytest.raises(ValueError):
        CosineAnnealing(0.1, t_max=10, min_lr=-0.1)
    with pytest.raises(ValueError):
        CosineAnnealing(0.1, t_max=10, min_lr=0.2)


# ---------------------------------------------------------------------------
# build_scheduler_from_config
# ---------------------------------------------------------------------------

def test_build_scheduler_default_constant():
    s = build_scheduler_from_config({}, base_lr=0.05)
    assert isinstance(s, ConstantLR)


def test_build_scheduler_step():
    s = build_scheduler_from_config(
        {"scheduler": {"type": "step", "step_size": 5, "gamma": 0.5}},
        base_lr=0.1,
    )
    assert isinstance(s, StepDecay)
    assert s.get_lr(5) == 0.05


def test_build_scheduler_cosine():
    s = build_scheduler_from_config(
        {"epochs": 20, "scheduler": {"type": "cosine", "min_lr": 0.001}},
        base_lr=0.1,
    )
    assert isinstance(s, CosineAnnealing)
    assert s.t_max == 20
    assert abs(s.get_lr(20) - 0.001) < 1e-9


def test_build_scheduler_unknown():
    with pytest.raises(ValueError):
        build_scheduler_from_config(
            {"scheduler": {"type": "unknown"}},
            base_lr=0.1,
        )
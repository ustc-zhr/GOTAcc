from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from gotacc.configs.schema import (
    BackendConfig,
    MetaConfig,
    OptimizerConfig,
    RuntimeConfig,
    TaskConfig,
)
from gotacc.runners import task_runner


def _online_task(*, set_best: bool | None = None) -> TaskConfig:
    runtime_kwargs = {
        "save_history": False,
        "plot_convergence": False,
        "verbose": False,
    }
    if set_best is not None:
        runtime_kwargs["set_best"] = set_best
    return TaskConfig(
        meta=MetaConfig(name="paper_runner_probe", machine="FakePV"),
        backend=BackendConfig(
            type="epics",
            bounds=[[-1.0, 1.0]],
            bounds_mode="absolute",
            kwargs={
                "knobs_pvnames": ["FAKE:K1:SET"],
                "knob_readback_pvnames": ["FAKE:K1:RB"],
                "obj_pvnames": ["FAKE:OBJ"],
                "objective_names": ["objective"],
                "obj_weights": [1.0],
                "obj_samples": 1,
                "obj_math": ["mean"],
                "set_interval": 0.0,
                "sample_interval": 0.0,
                "combine_mode": "weighted_sum",
            },
        ),
        optimizer=OptimizerConfig(name="bo", kwargs={}),
        runtime=RuntimeConfig(**runtime_kwargs),
    )


@dataclass
class _FakeBackend:
    fail_init: bool = False
    restore_calls: int = 0
    set_best_calls: int = 0
    close_calls: int = 0

    def init_knob_value(self):
        if self.fail_init:
            raise RuntimeError("initial read failed")
        return np.asarray([0.25], dtype=float)

    def evaluate(self, x):
        return float(np.asarray(x, dtype=float)[0])

    def restore_initial(self):
        self.restore_calls += 1

    def set_best(self, optimizer=None):
        self.set_best_calls += 1

    def close(self):
        self.close_calls += 1


class _FakeOptimizer:
    def optimize(self):
        return {"status": "ok"}


def _patch_runner(monkeypatch, backend: _FakeBackend) -> None:
    monkeypatch.setattr(task_runner, "build_backend", lambda _cfg: backend)
    monkeypatch.setattr(
        task_runner,
        "build_optimizer",
        lambda task_cfg, objective_callable, bounds: _FakeOptimizer(),
    )


def test_online_best_writeback_is_opt_in_for_python_runner(monkeypatch):
    backend = _FakeBackend()
    _patch_runner(monkeypatch, backend)

    cfg = _online_task()
    task_runner.run_task(cfg)

    assert cfg.runtime.set_best is False
    assert backend.set_best_calls == 0
    assert backend.close_calls == 1


def test_online_best_writeback_remains_available_when_explicit(monkeypatch):
    backend = _FakeBackend()
    _patch_runner(monkeypatch, backend)

    task_runner.run_task(_online_task(set_best=True))

    assert backend.set_best_calls == 1
    assert backend.close_calls == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda cfg: setattr(cfg.backend, "bounds", [[1.0, -1.0]]), "low < high"),
        (lambda cfg: setattr(cfg.optimizer, "name", "unknown"), "Unsupported optimizer"),
    ],
)
def test_run_task_validates_direct_python_config_before_backend_creation(
    monkeypatch,
    mutate,
    message,
):
    cfg = _online_task(set_best=False)
    mutate(cfg)
    backend_builds = []
    monkeypatch.setattr(
        task_runner,
        "build_backend",
        lambda task_cfg: backend_builds.append(task_cfg),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        task_runner.run_task(cfg)

    assert backend_builds == []


def test_runner_closes_backend_when_initial_read_fails(monkeypatch):
    backend = _FakeBackend(fail_init=True)
    _patch_runner(monkeypatch, backend)

    with pytest.raises(RuntimeError, match="initial read failed"):
        task_runner.run_task(_online_task(set_best=False))

    assert backend.restore_calls == 0
    assert backend.set_best_calls == 0
    assert backend.close_calls == 1


def test_runner_closes_backend_when_optimizer_construction_fails(monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(task_runner, "build_backend", lambda _cfg: backend)

    def fail_build_optimizer(*_args, **_kwargs):
        raise RuntimeError("optimizer init failed")

    monkeypatch.setattr(task_runner, "build_optimizer", fail_build_optimizer)

    with pytest.raises(RuntimeError, match="optimizer init failed"):
        task_runner.run_task(_online_task(set_best=False))

    assert backend.restore_calls == 0
    assert backend.set_best_calls == 0
    assert backend.close_calls == 1

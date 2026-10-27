from __future__ import annotations

import copy
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from gotacc.gui.services.task_service import TaskService
from gotacc.interfaces.factory import build_backend


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "gui_profile_baseline.json"


def _base_task(tmp_path: Path, *, machine: str) -> dict:
    return {
        "task_name": f"{machine.lower()}_gui_compatibility",
        "description": "Synthetic GUI compatibility profile; FakePV only.",
        "mode": "Online EPICS",
        "objective_type": "Single Objective",
        "algorithm": "BO",
        "max_evaluations": 7,
        "seed": 120,
        "workdir": str(tmp_path),
        "test_function": "",
        "constraints": [],
        "algorithm_params": [],
        "machine": {
            "ca_address": f"FAKE-{machine}",
            "restore_on_abort": True,
            "readback_check": True,
            "readback_tol": 1e-9,
            "set_interval": 0.0,
            "sample_interval": 0.0,
            "write_timeout": 2.0,
            "write_policy": "none",
            "objective_policies": [],
            "constraint_policies": [],
            "write_links": [],
            "mapping": [],
        },
    }


def _half_task(tmp_path: Path) -> dict:
    task = _base_task(tmp_path, machine="HALF")
    task["variables"] = [
        {"Enable": "Yes", "Name": "HC8", "Lower": "-3", "Upper": "3", "Initial": "0.5"},
        {"Enable": "Yes", "Name": "VC8", "Lower": "-3", "Upper": "3", "Initial": "-0.25"},
    ]
    task["objectives"] = [
        {
            "Enable": "Yes",
            "Name": "sigx",
            "Direction": "minimize",
            "Weight": "1",
            "Samples": "2",
            "Math": "mean",
        },
        {
            "Enable": "Yes",
            "Name": "sigy",
            "Direction": "minimize",
            "Weight": "1",
            "Samples": "2",
            "Math": "mean",
        },
    ]
    task["machine"]["mapping"] = [
        {"Role": "knob", "Name": "HC8", "PV Name": "HALF:HC8:SET", "Readback": "HALF:HC8:RB"},
        {"Role": "knob", "Name": "VC8", "PV Name": "HALF:VC8:SET", "Readback": "HALF:VC8:RB"},
        {"Role": "objective", "Name": "sigx", "PV Name": "HALF:SIGX", "Readback": ""},
        {"Role": "objective", "Name": "sigy", "PV Name": "HALF:SIGY", "Readback": ""},
    ]
    return task


def _irfel_task(tmp_path: Path) -> dict:
    task = _base_task(tmp_path, machine="IRFEL")
    task["variables"] = [
        {"Enable": "Yes", "Name": "Q1", "Lower": "8", "Upper": "12", "Initial": "10"},
        {"Enable": "Yes", "Name": "Q2", "Lower": "16", "Upper": "24", "Initial": "20"},
        {"Enable": "Yes", "Name": "HC1", "Lower": "-2", "Upper": "2", "Initial": "-1"},
    ]
    task["objectives"] = [
        {
            "Enable": "Yes",
            "Name": "fel_energy",
            "Direction": "minimize",
            "Weight": "1",
            "Samples": "2",
            "Math": "mean",
        },
        {
            "Enable": "Yes",
            "Name": "transmission",
            "Direction": "maximize",
            "Weight": "1",
            "Samples": "2",
            "Math": "mean",
        },
    ]
    task["machine"].update(
        {
            "write_policy": "delta_link",
            "write_links": [
                {
                    "Source Knob": "Q1",
                    "Target Knob": "Q2",
                    "Scale": "-2",
                    "Readback Tol": "0.01",
                    "Enabled": "True",
                }
            ],
            "objective_policies": [
                {
                    "Enabled": "True",
                    "Policy Name": "fel_energy_guard",
                    "Kwargs JSON": json.dumps(
                        {
                            "target_col": 0,
                            "large_threshold": 1e6,
                            "change_threshold": 1e-6,
                        }
                    ),
                }
            ],
            "mapping": [
                {"Role": "knob", "Name": "Q1", "PV Name": "IRFEL:Q1:SET", "Readback": "IRFEL:Q1:RB"},
                {"Role": "knob", "Name": "Q2", "PV Name": "IRFEL:Q2:SET", "Readback": "IRFEL:Q2:RB"},
                {"Role": "knob", "Name": "HC1", "PV Name": "IRFEL:HC1:SET", "Readback": "IRFEL:HC1:RB"},
                {"Role": "objective", "Name": "fel_energy", "PV Name": "IRFEL:ENERGY", "Readback": ""},
                {"Role": "objective", "Name": "transmission", "PV Name": "IRFEL:TRANS", "Readback": ""},
            ],
        }
    )
    return task


class _FakePV:
    def __init__(self, values: dict[str, float], readbacks: dict[str, str]) -> None:
        self.values = dict(values)
        self.readbacks = dict(readbacks)
        self.reads: list[list[str]] = []
        self.writes: list[dict] = []

    def caget(self, name, timeout=None):
        self.reads.append([str(name)])
        return self.values.get(str(name))

    def caget_many(self, names):
        normalized = [str(name) for name in names]
        self.reads.append(normalized)
        return [self.values.get(name) for name in normalized]

    def caput(self, name, value):
        self._record_write([str(name)], [float(value)], kind="one")

    def caput_many(self, names, values):
        self._record_write(
            [str(name) for name in names],
            [float(value) for value in values],
            kind="many",
        )

    def _record_write(self, names: list[str], values: list[float], *, kind: str) -> None:
        self.writes.append({"kind": kind, "pvs": names, "values": values})
        for name, value in zip(names, values):
            self.values[name] = value
            readback = self.readbacks.get(name)
            if readback:
                self.values[readback] = value


def _normalize_paths(value, root: Path):
    if isinstance(value, dict):
        return {key: _normalize_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_paths(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), "<WORKDIR>")
    return value


def _profile_snapshot(name: str, task: dict, tmp_path: Path, monkeypatch) -> dict:
    from gotacc.interfaces import epics as epics_module

    cfg = TaskService.build_task_config(task)
    ready = TaskService.make_backend_build_ready_config(cfg)
    ready.backend.kwargs["log_path"] = str(tmp_path / f"{name.lower()}.opt")

    if name == "HALF":
        fake = _FakePV(
            {
                "HALF:HC8:RB": 0.5,
                "HALF:VC8:RB": -0.25,
                "HALF:SIGX": 2.0,
                "HALF:SIGY": 3.0,
            },
            {"HALF:HC8:SET": "HALF:HC8:RB", "HALF:VC8:SET": "HALF:VC8:RB"},
        )
        evaluate_x = [0.75, -0.5]
    else:
        fake = _FakePV(
            {
                "IRFEL:Q1:RB": 10.0,
                "IRFEL:Q2:RB": 20.0,
                "IRFEL:HC1:RB": -1.0,
                "IRFEL:ENERGY": 8.0,
                "IRFEL:TRANS": 5.0,
            },
            {
                "IRFEL:Q1:SET": "IRFEL:Q1:RB",
                "IRFEL:Q2:SET": "IRFEL:Q2:RB",
                "IRFEL:HC1:SET": "IRFEL:HC1:RB",
            },
        )
        evaluate_x = [11.0, 0.5]

    monkeypatch.setattr(
        epics_module,
        "_load_epics",
        lambda: (fake.caget, fake.caput, fake.caput_many, fake.caget_many),
    )
    with redirect_stdout(StringIO()):
        backend = build_backend(ready)
        initial = backend.init_knob_value()
        objective = backend.evaluate(np.asarray(evaluate_x, dtype=float))
        policy_state = backend.export_write_policy_state()
        backend.restore_initial()
        backend.close()

    plain_cfg = TaskService._plain_data(cfg.to_dict())
    return _normalize_paths(
        {
            "task_config": plain_cfg,
            "fake_pv": {
                "initial": np.asarray(initial, dtype=float).tolist(),
                "evaluate_x": evaluate_x,
                "objective": np.asarray(objective, dtype=float).reshape(-1).tolist(),
                "policy_state": policy_state,
                "reads": fake.reads,
                "writes": fake.writes,
            },
        },
        tmp_path,
    )


def build_profile_snapshot(tmp_path: Path, monkeypatch) -> dict:
    return {
        "format_version": 1,
        "profiles": {
            "HALF": _profile_snapshot("HALF", _half_task(tmp_path), tmp_path, monkeypatch),
            "IRFEL": _profile_snapshot("IRFEL", _irfel_task(tmp_path), tmp_path, monkeypatch),
        },
    }


def test_half_irfel_gui_profiles_match_frozen_task_and_fakepv_baseline(
    tmp_path,
    monkeypatch,
):
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    actual = build_profile_snapshot(tmp_path, monkeypatch)

    assert actual == expected


def test_gui_compilation_keeps_automatic_best_writeback_disabled(tmp_path):
    for task in (_half_task(tmp_path), _irfel_task(tmp_path)):
        cfg = TaskService.build_task_config(copy.deepcopy(task))
        assert cfg.runtime.set_best is False

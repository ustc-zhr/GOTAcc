"""Deterministic, offline replay of GOTAcc production policy classes.

This experiment calls only ``post_reduce`` on in-memory NumPy arrays.  It does
not instantiate ``EpicsObjective``, load pyepics, connect to an IOC, or write a
PV.  Run it from the repository root on ``feature/paper-reliability-fixes``::

    PYTHONPATH=src python experiments/paper_policy_replay/replay_policy.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from gotacc.interfaces.epics import FelEnergyGuardPolicy
from gotacc.interfaces.policies import POLICY_REGISTRY


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
EXPECTED_BRANCH = "feature/paper-reliability-fixes"
REFERENCE_PRODUCTION_COMMIT = "010518fbece77a4630684065101033ff8588613d"

# Every input is synthetic.  ``stale_constant`` is a proxy for a frozen
# diagnostic; it is not a captured machine waveform.  Dict insertion order is
# the canonical output order.
SEQUENCES: dict[str, dict[str, Any]] = {
    "normal": {
        "samples": [2.0, 3.0, 4.0],
        "class": "nominal",
        "description": "Nominal varying FEL-energy samples.",
    },
    "valid_zero_constant": {
        "samples": [0.0, 0.0, 0.0],
        "class": "ambiguity_control",
        "description": "A flat value that can be physically valid; not labelled an anomaly.",
    },
    "stale_constant": {
        "samples": [7.0, 7.0, 7.0],
        "class": "synthetic_anomaly",
        "description": "Synthetic proxy for a retained/frozen diagnostic value.",
    },
    "high_level": {
        "samples": [2.0e6, 3.0e6, 4.0e6],
        "class": "synthetic_anomaly",
        "description": "Synthetic sustained high-level threshold case.",
    },
    "irfel_1e8_spike": {
        "samples": [2.0, 1.0e8, 2.0],
        "class": "synthetic_anomaly",
        "description": "Synthetic isolated spike at the reported IRFEL anomaly scale.",
    },
    "slow_drift": {
        "samples": [4.0, 4.2, 4.4],
        "class": "nominal",
        "description": "Finite, slowly varying control sequence.",
    },
    "contains_nan": {
        "samples": [float("nan"), 3.0, 4.0],
        "class": "nonfinite_boundary",
        "description": "Boundary case containing NaN.",
    },
    "contains_inf": {
        "samples": [float("inf"), 3.0, 4.0],
        "class": "nonfinite_boundary",
        "description": "Boundary case containing positive infinity.",
    },
}

CONSTRAINT_SEQUENCES: dict[str, dict[str, Any]] = {
    "normal_bpm": {
        "samples": [0.2, 0.3, 0.4],
        "class": "nominal",
        "description": "Finite nonzero BPM samples inside the configured bound.",
    },
    "zero_bpm": {
        "samples": [0.0, 0.0, 0.0],
        "class": "synthetic_anomaly",
        "description": "Synthetic all-zero BPM sequence mapped outside the bound.",
    },
}

EXPECTED_OBJECTIVE_SUMMARY = {
    "none": {
        "evaluations": 8,
        "output_changed": 0,
        "finite_output": 6,
        "nonfinite_output": 2,
        "observable_events": 0,
        "nominal_retained": 2,
        "synthetic_anomaly_changed": 0,
        "nonfinite_passthrough": 2,
    },
    "fel_energy_guard": {
        "evaluations": 8,
        "output_changed": 4,
        "finite_output": 7,
        "nonfinite_output": 1,
        "observable_events": 0,
        "nominal_retained": 2,
        "synthetic_anomaly_changed": 3,
        "nonfinite_passthrough": 1,
    },
    "sample_guard_max_abs": {
        "evaluations": 8,
        "output_changed": 4,
        "finite_output": 7,
        "nonfinite_output": 1,
        "observable_events": 5,
        "nominal_retained": 2,
        "synthetic_anomaly_changed": 3,
        "nonfinite_passthrough": 1,
    },
}


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _json_number(value: float) -> float | str:
    value = float(value)
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "Infinity"
    if np.isneginf(value):
        return "-Infinity"
    return value


def _json_safe(value: Any) -> Any:
    """Convert NumPy and non-finite values to strict JSON representations."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return _json_number(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return _json_number(float(value))
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False)
    if value is None:
        return "not_observable"
    return value


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError(f"Cannot write an empty trace: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(records[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(value) for key, value in record.items()} for record in records
        )


def _sample_statistics(samples: np.ndarray) -> dict[str, float]:
    with np.errstate(invalid="ignore", over="ignore"):
        return {
            "reduced_mean": float(np.mean(samples)),
            "mean_abs": float(np.mean(np.abs(samples))),
            "max_abs": float(np.max(np.abs(samples))),
            "peak_to_peak": float(np.max(samples) - np.min(samples)),
        }


def _same_number(left: float, right: float) -> bool:
    with np.errstate(invalid="ignore", over="ignore"):
        return bool(np.isclose(left, right, equal_nan=True))


def _environment() -> dict[str, Any]:
    source_files = [
        Path("src/gotacc/interfaces/epics.py"),
        Path("src/gotacc/interfaces/policies/sample_guard.py"),
        Path("src/gotacc/interfaces/policies/builtins.py"),
        SCRIPT_DIR.relative_to(REPO_ROOT) / "replay_policy.py",
    ]
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "pytest_version": _distribution_version("pytest"),
        "matplotlib_version": _distribution_version("matplotlib"),
        "gotacc_version": _distribution_version("gotacc"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "git_branch": _run_git("branch", "--show-current"),
        "git_head": _run_git("rev-parse", "HEAD"),
        "reference_production_commit": REFERENCE_PRODUCTION_COMMIT,
        "tracked_status": _run_git("status", "--short", "--untracked-files=no"),
        "production_status": _run_git("status", "--short", "--", "src/gotacc"),
        "source_sha256": {
            str(relative): _sha256(REPO_ROOT / relative) for relative in source_files
        },
        "safety_boundary": {
            "epics_objective_instantiated": False,
            "pyepics_loaded": False,
            "ioc_connected": False,
            "pv_writes": 0,
            "execution_mode": "in-memory post_reduce replay",
        },
    }


def _policies() -> tuple[SimpleNamespace, dict[str, Any], dict[str, dict[str, Any]]]:
    context = SimpleNamespace(
        objective_names=["fel_energy"],
        obj_pvnames=["FAKE:FEL:ENERGY"],
        constraint_names=["orbit_x"],
        constraint_pvnames=["FAKE:BPM:X"],
        constraint_bounds=[(-1.0, 1.0)],
    )
    sample_guard_kwargs = {
        "target": "fel_energy",
        "conditions": [
            {"metric": "max_abs", "operator": "gt", "value": 1.0e6},
            {"metric": "peak_to_peak", "operator": "lt", "value": 1.0e-6},
        ],
        "match": "any",
        "action": {"type": "replace", "value": 0.0},
    }
    configured = {
        "none": None,
        "fel_energy_guard": FelEnergyGuardPolicy(
            target_col=0,
            large_threshold=1.0e6,
            change_threshold=1.0e-6,
        ),
        "sample_guard_max_abs": POLICY_REGISTRY.build(
            "objective", "sample_guard", sample_guard_kwargs
        ),
    }
    metadata = {
        "none": {
            "class": None,
            "configuration": None,
            "event_observability": "not applicable",
        },
        "fel_energy_guard": {
            "class": "gotacc.interfaces.epics.FelEnergyGuardPolicy",
            "configuration": {
                "target_col": 0,
                "large_threshold": 1.0e6,
                "change_threshold": 1.0e-6,
            },
            "event_observability": (
                "event sink attached, but this production policy does not emit events"
            ),
        },
        "sample_guard_max_abs": {
            "class": (
                "gotacc.interfaces.policies.sample_guard.SampleGuardObjectivePolicy"
            ),
            "registry_name": "sample_guard",
            "configuration": sample_guard_kwargs,
            "event_observability": "production policy emits an event when its rule matches",
        },
    }
    for policy in configured.values():
        if policy is not None:
            policy.validate_backend(context)
    return context, configured, metadata


def _objective_replay(
    context: SimpleNamespace,
    configured: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    event_log: list[dict[str, Any]] = []
    for sequence_name, sequence in SEQUENCES.items():
        samples = np.asarray(sequence["samples"], dtype=float)
        raw_samples = samples.reshape(-1, 1)
        statistics = _sample_statistics(samples)
        reduced = statistics["reduced_mean"]
        for policy_name, policy in configured.items():
            events: list[str] = []
            if policy is None:
                processed = reduced
                trigger_observed: bool | None = False
            else:
                policy.set_event_sink(events.append)
                with np.errstate(invalid="ignore", over="ignore"):
                    processed = float(
                        policy.post_reduce(
                            np.asarray([reduced], dtype=float), raw_samples, context
                        )[0]
                    )
                trigger_observed = (
                    bool(events) if policy_name == "sample_guard_max_abs" else None
                )
            record = {
                "sequence": sequence_name,
                "input_class": sequence["class"],
                "description": sequence["description"],
                "samples": sequence["samples"],
                "sample_count": len(sequence["samples"]),
                **statistics,
                "policy": policy_name,
                "processed_objective": processed,
                "output_changed": not _same_number(reduced, processed),
                "output_finite": bool(np.isfinite(processed)),
                "trigger_observed": trigger_observed,
                "event_count": len(events),
                "events": events,
            }
            records.append(record)
            event_log.extend(
                {
                    "stage": "objective",
                    "sequence": sequence_name,
                    "policy": policy_name,
                    "event": event,
                }
                for event in events
            )
    return records, event_log


def _constraint_replay(
    context: SimpleNamespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    guard_kwargs = {
        "target": "orbit_x",
        "conditions": [{"metric": "max_abs", "operator": "le", "value": 1.0e-9}],
        "action": {
            "type": "violate_bound",
            "delta_ratio": 0.1,
            "delta_min": 1.0e-6,
            "scale_floor": 1.0,
        },
    }
    policy = POLICY_REGISTRY.build("constraint", "sample_guard", guard_kwargs)
    policy.validate_backend(context)
    records: list[dict[str, Any]] = []
    event_log: list[dict[str, Any]] = []
    for sequence_name, sequence in CONSTRAINT_SEQUENCES.items():
        samples = np.asarray(sequence["samples"], dtype=float)
        raw_samples = samples.reshape(-1, 1)
        statistics = _sample_statistics(samples)
        reduced = statistics["reduced_mean"]
        for policy_name in ("none", "sample_guard_constraint"):
            events: list[str] = []
            if policy_name == "none":
                processed = reduced
                trigger_observed: bool | None = False
            else:
                policy.set_event_sink(events.append)
                processed = float(
                    policy.post_reduce(
                        np.asarray([reduced], dtype=float), raw_samples, context
                    )[0]
                )
                trigger_observed = bool(events)
            record = {
                "sequence": sequence_name,
                "input_class": sequence["class"],
                "description": sequence["description"],
                "samples": sequence["samples"],
                "sample_count": len(sequence["samples"]),
                **statistics,
                "policy": policy_name,
                "processed_constraint": processed,
                "output_changed": not _same_number(reduced, processed),
                "within_bound": bool(-1.0 <= processed <= 1.0),
                "trigger_observed": trigger_observed,
                "event_count": len(events),
                "events": events,
            }
            records.append(record)
            event_log.extend(
                {
                    "stage": "constraint",
                    "sequence": sequence_name,
                    "policy": policy_name,
                    "event": event,
                }
                for event in events
            )
    metadata = {
        "class": "gotacc.interfaces.policies.sample_guard.SampleGuardConstraintPolicy",
        "registry_name": "sample_guard",
        "configuration": guard_kwargs,
        "constraint_bound": [-1.0, 1.0],
    }
    return records, event_log, metadata


def _objective_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for policy_name in EXPECTED_OBJECTIVE_SUMMARY:
        rows = [record for record in records if record["policy"] == policy_name]
        summary[policy_name] = {
            "evaluations": len(rows),
            "output_changed": sum(record["output_changed"] for record in rows),
            "finite_output": sum(record["output_finite"] for record in rows),
            "nonfinite_output": sum(not record["output_finite"] for record in rows),
            "observable_events": sum(record["event_count"] for record in rows),
            "nominal_retained": sum(
                record["input_class"] == "nominal" and not record["output_changed"]
                for record in rows
            ),
            "synthetic_anomaly_changed": sum(
                record["input_class"] == "synthetic_anomaly"
                and record["output_changed"]
                for record in rows
            ),
            "nonfinite_passthrough": sum(
                record["input_class"] == "nonfinite_boundary"
                and not record["output_finite"]
                for record in rows
            ),
        }
    return summary


def _constraint_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    guarded = [record for record in records if record["policy"] == "sample_guard_constraint"]
    zero = next(record for record in guarded if record["sequence"] == "zero_bpm")
    normal = next(record for record in guarded if record["sequence"] == "normal_bpm")
    return {
        "evaluations": len(guarded),
        "output_changed": sum(record["output_changed"] for record in guarded),
        "observable_events": sum(record["event_count"] for record in guarded),
        "normal_bpm_retained": not normal["output_changed"],
        "zero_bpm_processed": zero["processed_constraint"],
        "zero_bpm_within_bound": zero["within_bound"],
    }


def _verify_expectations(
    objective_summary: dict[str, dict[str, int]],
    constraint_summary: dict[str, Any],
    objective_records: list[dict[str, Any]],
) -> list[str]:
    mismatches: list[str] = []
    if objective_summary != EXPECTED_OBJECTIVE_SUMMARY:
        mismatches.append(
            "objective summary differs from the protocol expectation: "
            f"expected={EXPECTED_OBJECTIVE_SUMMARY!r}, observed={objective_summary!r}"
        )
    expected_constraint = {
        "evaluations": 2,
        "output_changed": 1,
        "observable_events": 1,
        "normal_bpm_retained": True,
        "zero_bpm_processed": 1.2,
        "zero_bpm_within_bound": False,
    }
    if constraint_summary != expected_constraint:
        mismatches.append(
            "constraint summary differs from the protocol expectation: "
            f"expected={expected_constraint!r}, observed={constraint_summary!r}"
        )
    spike_rows = {
        record["policy"]: record
        for record in objective_records
        if record["sequence"] == "irfel_1e8_spike"
    }
    for policy_name in ("fel_energy_guard", "sample_guard_max_abs"):
        if not np.isclose(spike_rows[policy_name]["processed_objective"], 0.0):
            mismatches.append(f"{policy_name} did not map the 1e8 spike sequence to zero")
    return mismatches


def _make_figure(
    path: Path,
    objective_records: list[dict[str, Any]],
    objective_summary: dict[str, dict[str, int]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite_sequences = [
        name
        for name, sequence in SEQUENCES.items()
        if sequence["class"] != "nonfinite_boundary"
    ]
    x_values = np.arange(len(finite_sequences))
    raw_values = [
        next(
            record["reduced_mean"]
            for record in objective_records
            if record["sequence"] == sequence and record["policy"] == "none"
        )
        for sequence in finite_sequences
    ]
    labels = {
        "none": "No policy",
        "fel_energy_guard": "FEL energy guard",
        "sample_guard_max_abs": "Declarative sample guard",
    }
    colors = {
        "none": "#4c78a8",
        "fel_energy_guard": "#f58518",
        "sample_guard_max_abs": "#54a24b",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    axes[0].plot(x_values, raw_values, "k--", linewidth=1.2, label="Arithmetic mean")
    for policy_name in labels:
        processed = [
            next(
                record["processed_objective"]
                for record in objective_records
                if record["sequence"] == sequence and record["policy"] == policy_name
            )
            for sequence in finite_sequences
        ]
        axes[0].scatter(
            x_values,
            processed,
            label=labels[policy_name],
            color=colors[policy_name],
            s=45,
            zorder=3,
        )
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_xticks(x_values, finite_sequences, rotation=35, ha="right")
    axes[0].set_ylabel("Reduced / processed objective (symlog)")
    axes[0].set_title("(a) Synthetic finite-sequence replay")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    metrics = ("output_changed", "finite_output", "observable_events")
    metric_labels = ("Changed outputs", "Finite outputs", "Observable events")
    width = 0.24
    metric_x = np.arange(len(metrics))
    for offset, policy_name in enumerate(labels):
        values = [objective_summary[policy_name][metric] for metric in metrics]
        axes[1].bar(
            metric_x + (offset - 1) * width,
            values,
            width,
            label=labels[policy_name],
            color=colors[policy_name],
        )
    axes[1].set_xticks(metric_x, metric_labels)
    axes[1].set_ylim(0, 8.8)
    axes[1].set_ylabel("Count out of 8 evaluations")
    axes[1].set_title("(b) Observed output and event counts")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("GOTAcc policy replay (synthetic inputs; no IOC/PV access)")
    figure.savefig(
        path,
        dpi=180,
        metadata={
            "Title": "GOTAcc deterministic policy replay",
            "Author": "GOTAcc offline experiment",
        },
    )
    plt.close(figure)


def _write_manifest(out_dir: Path) -> None:
    files = sorted(path for path in out_dir.iterdir() if path.name != "SHA256SUMS")
    lines = [f"{_sha256(path)}  {path.name}" for path in files if path.is_file()]
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def replay(out_dir: Path, *, make_figure: bool = True) -> dict[str, Any]:
    environment = _environment()
    if environment["git_branch"] != EXPECTED_BRANCH:
        raise RuntimeError(
            f"Run this experiment on {EXPECTED_BRANCH!r}; "
            f"observed {environment['git_branch']!r}"
        )
    if environment["production_status"]:
        raise RuntimeError(
            "Production source files under src/gotacc have local modifications; "
            "refusing to mix them into the replay evidence."
        )

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    context, configured, objective_policy_metadata = _policies()
    objective_records, objective_events = _objective_replay(context, configured)
    constraint_records, constraint_events, constraint_policy_metadata = _constraint_replay(
        context
    )
    objective_summary = _objective_summary(objective_records)
    constraint_summary = _constraint_summary(constraint_records)
    mismatches = _verify_expectations(
        objective_summary, constraint_summary, objective_records
    )
    events = [*objective_events, *constraint_events]

    artifact = {
        "schema_version": 1,
        "experiment": "GOTAcc deterministic policy replay",
        "input_provenance": (
            "All sequences are synthetic fixed inputs; none are recorded machine data."
        ),
        "source": {
            "required_branch": EXPECTED_BRANCH,
            "observed_branch": environment["git_branch"],
            "git_head_at_execution": environment["git_head"],
            "reference_production_commit": REFERENCE_PRODUCTION_COMMIT,
        },
        "objective_policies": objective_policy_metadata,
        "constraint_policy": constraint_policy_metadata,
        "objective_summary": objective_summary,
        "constraint_summary": constraint_summary,
        "expectation_check": {
            "matches_protocol": not mismatches,
            "mismatches": mismatches,
        },
        "event_interpretation": {
            "fel_energy_guard": (
                "No events are present because the production class does not call emit_event; "
                "absence of an event is not evidence that its internal predicate was false."
            ),
            "sample_guard": (
                "Events are captured directly from the production policy event sink."
            ),
        },
        "important_limitations": [
            "The replay calls production post_reduce methods directly; it does not test acquisition, optimizer convergence, GUI transitions, or write/readback verification.",
            "No IOC is connected and no PV read or write is performed.",
            "The 1e8 spike and frozen-value inputs are synthetic and only reflect stated machine-anomaly forms; they are not recorded IRFEL waveforms.",
            "A constant value can be valid, so a flat-signal match is not by itself ground-truth anomaly classification.",
            "NaN is passed through by both tested guards in this sequence. Infinity happens to satisfy the amplitude predicate and is replaced; this is not generic nonfinite handling.",
            "A None/PV read failure is an acquisition-stage error and cannot be represented as a numeric post-reduction sample.",
            "Replacing an invalid objective by zero is task-dependent and must not be interpreted as universally safe for minimization objectives.",
        ],
    }

    _write_json(out_dir / "environment.json", environment)
    _write_csv(out_dir / "objective_trace.csv", objective_records)
    _write_json(out_dir / "objective_trace.json", objective_records)
    _write_csv(out_dir / "constraint_trace.csv", constraint_records)
    _write_json(out_dir / "constraint_trace.json", constraint_records)
    _write_json(out_dir / "events.json", events)
    _write_json(out_dir / "summary.json", artifact)
    if make_figure:
        _make_figure(out_dir / "policy_replay_summary.png", objective_records, objective_summary)
    _write_manifest(out_dir)

    print(json.dumps(objective_summary, indent=2, ensure_ascii=False))
    print(f"Expectation check: {'PASS' if not mismatches else 'FAIL'}")
    print(f"Output: {out_dir}")
    if mismatches:
        raise AssertionError("\n".join(mismatches))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
        help="Artifact directory (default: experiments/paper_policy_replay/results)",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip PNG generation (useful for the lightweight regression test).",
    )
    args = parser.parse_args()
    replay(args.out_dir, make_figure=not args.no_figure)


if __name__ == "__main__":
    main()

"""Regression checks for the paper policy replay (offline only)."""

from __future__ import annotations

import json

import pytest

from replay_policy import replay


@pytest.fixture(scope="module")
def replay_artifacts(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("paper_policy_replay")
    artifact = replay(out_dir, make_figure=False)
    return out_dir, artifact


def test_protocol_expectations_match_production_policies(replay_artifacts):
    _, artifact = replay_artifacts
    assert artifact["expectation_check"] == {
        "matches_protocol": True,
        "mismatches": [],
    }
    assert artifact["objective_summary"]["fel_energy_guard"]["output_changed"] == 4
    assert artifact["objective_summary"]["sample_guard_max_abs"]["output_changed"] == 4
    assert artifact["constraint_summary"]["zero_bpm_processed"] == 1.2


def test_json_artifacts_are_strict_and_nonfinite_values_are_labelled(replay_artifacts):
    out_dir, _ = replay_artifacts
    for name in (
        "environment.json",
        "objective_trace.json",
        "constraint_trace.json",
        "events.json",
        "summary.json",
    ):
        with (out_dir / name).open(encoding="utf-8") as stream:
            json.load(stream, parse_constant=lambda value: pytest.fail(value))

    records = json.loads((out_dir / "objective_trace.json").read_text(encoding="utf-8"))
    nan_rows = [record for record in records if record["sequence"] == "contains_nan"]
    assert {record["processed_objective"] for record in nan_rows} == {"NaN"}


def test_events_are_reported_only_when_observed(replay_artifacts):
    out_dir, artifact = replay_artifacts
    events = json.loads((out_dir / "events.json").read_text(encoding="utf-8"))
    assert not [event for event in events if event["policy"] == "fel_energy_guard"]
    assert len(
        [event for event in events if event["policy"] == "sample_guard_max_abs"]
    ) == 5
    assert len(
        [event for event in events if event["policy"] == "sample_guard_constraint"]
    ) == 1
    assert "absence of an event is not evidence" in artifact["event_interpretation"][
        "fel_energy_guard"
    ]

# GOTAcc deterministic policy replay protocol

## Evidence status

The corrected source package was inspected from
`docs/experiments/paper_policy_replay/GOTAcc_round8_policy_replay_package_corrected_20261102.zip`
(SHA-256 `1d0cf3be02bdc5530ad64c95b18378c7b993efb576c467fe13a32c5e2524128e`).
Its three files were used as the starting point. The checked-in experiment adds
strict JSON traces, per-sequence statistics, event observability, environment
and source hashes, a regression test, and a generated figure.

This protocol was executed, not merely syntax-checked. The run used branch
`feature/paper-reliability-fixes` at production reference commit
`010518fbece77a4630684065101033ff8588613d`. The fixed expectations matched the
observed output without changing expected counts.

All replayed values are synthetic. The `1e8` spike reflects the stated order of
magnitude of an IRFEL diagnostic anomaly, but it is not recorded machine data.
The frozen-value sequence is also a constructed proxy.

## Safety and isolation

`replay_policy.py` imports the policy classes and calls their `post_reduce()`
methods on in-memory NumPy arrays. It does not instantiate `EpicsObjective`,
call `_load_epics()`, import pyepics, create a FakePV, connect to an IOC, or
perform a PV read/write. No GUI, EPICS backend, policy registry, or production
source file is modified.

The script refuses to run if:

- the active branch is not `feature/paper-reliability-fixes`; or
- `src/gotacc` contains local modifications.

## Production behavior under test

- `FelEnergyGuardPolicy`: replaces the reduced objective with zero when the
  sample mean absolute magnitude is greater than `1e6`, or when sample
  peak-to-peak variation is less than `1e-6`.
- `SampleGuardObjectivePolicy`: built through `POLICY_REGISTRY` with named
  target `fel_energy`, `max_abs > 1e6 OR peak_to_peak < 1e-6`, and a declarative
  `replace: 0` action.
- `SampleGuardConstraintPolicy`: built through `POLICY_REGISTRY` with named
  target `orbit_x`, `max_abs <= 1e-9`, and a declarative `violate_bound` action
  for the interval `[-1, 1]`.

The dedicated and declarative objective rules are not claimed to be identical
for every possible sequence. They are compared on the fixed sequences below.

## Fixed objective inputs

| Sequence | Samples | Role |
|---|---|---|
| `normal` | `[2, 3, 4]` | Nominal variation |
| `valid_zero_constant` | `[0, 0, 0]` | Ambiguity control; a flat value may be valid |
| `stale_constant` | `[7, 7, 7]` | Synthetic frozen-value proxy |
| `high_level` | `[2e6, 3e6, 4e6]` | Synthetic sustained high-level case |
| `irfel_1e8_spike` | `[2, 1e8, 2]` | Synthetic IRFEL-scale spike |
| `slow_drift` | `[4, 4.2, 4.4]` | Nominal finite control |
| `contains_nan` | `[NaN, 3, 4]` | Nonfinite boundary case |
| `contains_inf` | `[Inf, 3, 4]` | Nonfinite boundary case |

For every sequence the trace records the samples, arithmetic mean, mean
absolute magnitude, maximum absolute magnitude, peak-to-peak variation,
processed value, finite/nonfinite state, numerical change flag, and event sink
output.

## Reproduction commands

Create or activate a Python environment satisfying `pyproject.toml`, including
pytest, then run from the repository root:

```bash
git switch feature/paper-reliability-fixes
git rev-parse HEAD
git status --short -- src/gotacc

PYTHONPATH=src python experiments/paper_policy_replay/replay_policy.py \
  --out-dir experiments/paper_policy_replay/results

PYTHONPATH=src python -m pytest -q \
  experiments/paper_policy_replay/test_replay_policy.py

PYTHONPATH=src python -m pytest -q \
  tests/test_sample_guard_policy.py \
  tests/test_policy_registry.py \
  tests/test_rule_editor.py
```

The recorded run used:

```text
Python       3.11.5
NumPy        2.2.5
pytest       9.0.3
Matplotlib   3.10.0
GOTAcc       1.3.0
Platform     Linux 6.18.40.1-microsoft-standard-WSL2, x86_64
```

## Fixed expectation check

The supplied corrected package defined the following expectations before the
formal run. The experiment preserves them as assertions:

| Policy | Changed outputs | Finite outputs | Observable objective events |
|---|---:|---:|---:|
| No policy | 0/8 | 6/8 | 0 |
| `fel_energy_guard` | 4/8 | 7/8 | 0 |
| Declarative `sample_guard` | 4/8 | 7/8 | 5 |

The constraint expectation is that the normal BPM mean remains `0.3`, while
the all-zero sequence becomes `1.2`, outside `[-1, 1]`, with one emitted
event. `summary.json` records `matches_protocol: true` for the executed run.

An output-change count is not a classification-accuracy count. In particular,
the declarative guard emits an event for the valid-zero control even though the
replacement `0 -> 0` causes no numerical change.

## Event interpretation

The production `SampleGuardObjectivePolicy` and
`SampleGuardConstraintPolicy` call `emit_event()` when their rule matches, so
their captured messages are directly observable events. The production
`FelEnergyGuardPolicy` does not call `emit_event()`. Its event sink remains
empty; this is reported as unavailable observability, not as evidence that an
internal predicate did not match. No FEL trigger logs are inferred from the
processed values.

## Outputs and integrity

The `results/` directory contains CSV and strict JSON traces, environment and
source hashes, event messages, a JSON summary, a PNG figure, and `SHA256SUMS`.
NaN and infinity are represented as the JSON strings `"NaN"` and `"Infinity"`
so the files remain valid strict JSON.

## Limitations

- This is deterministic synthetic-input replay, not real-machine validation.
- Only production `post_reduce()` behavior is exercised; acquisition, PV
  read/write, snapshot/restore, optimizer convergence, and GUI behavior are
  outside scope.
- Flat signals are operationally ambiguous: an unchanging valid signal cannot
  be distinguished from a frozen diagnostic by these values alone.
- NaN propagates through both guards in the selected sequence. Infinity is
  replaced because it satisfies the configured amplitude predicate; this is
  not comprehensive nonfinite handling.
- A `None` read failure belongs to acquisition handling and cannot be replayed
  as a numeric policy sample.
- Replacing an invalid objective with zero is task-dependent and is not a
  universal safe choice for minimization.

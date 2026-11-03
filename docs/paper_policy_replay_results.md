# Section 4 policy replay: executed results

## Result at a glance

The deterministic offline replay passed its fixed expectation check on
`feature/paper-reliability-fixes`. Both the existing FEL Energy Guard and the
declarative Sample Guard processed the synthetic IRFEL-scale `1e8` spike; this
experiment provides no evidence of a spike-detection defect in the FEL guard
and does not support a superiority claim for Sample Guard. Its positive result
is that diagnostic interpretation can be separated from the optimizer and, in
the general case, expressed as named configuration.

The experiment used only in-memory arrays. It did not connect to EPICS, create
an `EpicsObjective`, load pyepics, read or write a PV, or exercise a live
accelerator.

## Code and environment

| Item | Recorded value |
|---|---|
| Branch | `feature/paper-reliability-fixes` |
| Production reference commit | `010518fbece77a4630684065101033ff8588613d` |
| Run time (UTC) | `2026-11-03T03:18:38.222727+00:00` |
| Python | CPython 3.11.5 |
| NumPy | 2.2.5 |
| pytest | 9.0.3 |
| Matplotlib | 3.10.0 |
| GOTAcc package metadata | 1.3.0 |
| Platform | Linux 6.18.40.1-microsoft-standard-WSL2, x86_64 |

The recorded `src/gotacc` status was clean. Source-file and experiment-script
SHA-256 hashes are preserved in
`experiments/paper_policy_replay/results/environment.json`. The supplied
corrected package had SHA-256
`1d0cf3be02bdc5530ad64c95b18378c7b993efb576c467fe13a32c5e2524128e`.

## Real implementation checked

The replay imports and executes the production classes rather than local
reimplementations:

- `gotacc.interfaces.epics.FelEnergyGuardPolicy` checks
  `mean(abs(samples)) > 1e6` or `max(samples) - min(samples) < 1e-6`, then
  replaces the reduced target with zero.
- `SampleGuardObjectivePolicy` is built through `POLICY_REGISTRY` for named
  target `fel_energy`, using `max_abs > 1e6 OR peak_to_peak < 1e-6` and a
  declarative `replace: 0` action.
- `SampleGuardConstraintPolicy` is built through `POLICY_REGISTRY` for named
  target `orbit_x`, using `max_abs <= 1e-9` and a declarative
  `violate_bound` action against `[-1, 1]`.

The script's array shapes and context fields match these production APIs.

## Objective results

Eight independent, three-sample synthetic sequences were evaluated. “Changed”
means only that the post-policy scalar differs from the arithmetic mean. It is
not a count of correctly classified machine states.

| Policy | Changed outputs | Finite outputs | Nominal retained | Synthetic anomaly outputs changed | Nonfinite passthrough | Observable events |
|---|---:|---:|---:|---:|---:|---:|
| No policy | 0/8 | 6/8 | 2/2 | 0/3 | 2/2 | 0 |
| FEL Energy Guard | 4/8 | 7/8 | 2/2 | 3/3 | 1/2 | 0* |
| Declarative Sample Guard | 4/8 | 7/8 | 2/2 | 3/3 | 1/2 | 5 |

\* `FelEnergyGuardPolicy` does not call `emit_event()` in the tested production
implementation. The event sink was attached but remained empty. This is an
observability limitation, not evidence that its internal predicate was false.

### Per-sequence processed objective

| Sequence | Samples | Mean | No policy | FEL Guard | Sample Guard | Directly observed Sample Guard event |
|---|---|---:|---:|---:|---:|---|
| `normal` | `[2, 3, 4]` | 3 | 3 | 3 | 3 | No |
| `valid_zero_constant` | `[0, 0, 0]` | 0 | 0 | 0 | 0 | Yes (`0 -> 0`) |
| `stale_constant` | `[7, 7, 7]` | 7 | 7 | 0 | 0 | Yes |
| `high_level` | `[2e6, 3e6, 4e6]` | 3e6 | 3e6 | 0 | 0 | Yes |
| `irfel_1e8_spike` | `[2, 1e8, 2]` | 33,333,334.666666668 | 33,333,334.666666668 | 0 | 0 | Yes |
| `slow_drift` | `[4, 4.2, 4.4]` | 4.2 | 4.2 | 4.2 | 4.2 | No |
| `contains_nan` | `[NaN, 3, 4]` | NaN | NaN | NaN | NaN | No |
| `contains_inf` | `[Inf, 3, 4]` | Inf | Inf | 0 | 0 | Yes |

For the synthetic `1e8` spike, the FEL rule's mean absolute amplitude is
`33,333,334.666666668`, safely above `1e6` for three samples. The result is
consistent with the implementation and the corrected protocol. No `2e6`
isolated spike is used to argue a guard defect.

The constant-zero control illustrates why event count and numerical-change
count differ. Sample Guard directly emitted a match event, but replacing zero
with zero did not change the output. A constant signal can also be physically
valid; the replay therefore does not assign ground-truth classification labels
to every flat sequence.

## Constraint result

The normal BPM sequence `[0.2, 0.3, 0.4]` retained its mean of `0.3` with no
event. The synthetic all-zero sequence was changed from `0.0` to `1.2`, outside
the configured interval `[-1, 1]`, and the production Sample Guard emitted:

```text
Policy triggered for orbit_x [violate bound]: 0 → 1.2
```

This demonstrates a task-specific validity convention at the policy layer;
it does not establish that all-zero BPM data are universally invalid.

## Reproduction

From the repository root:

```bash
git switch feature/paper-reliability-fixes
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

The formal run printed `Expectation check: PASS`. Raw CSV/JSON, exact event
messages, the environment record, the figure, and hashes are under
`experiments/paper_policy_replay/results/`.

## Known limitations

- Inputs are fixed synthetic sequences, not recorded IRFEL machine data.
- Only policy `post_reduce()` behavior is tested. Acquisition failures,
  optimizer convergence, PV write/readback, snapshot/restore, GUI behavior,
  and live-machine safety are not tested.
- NaN passes through both guards in this replay. Infinity is replaced because
  it happens to satisfy the amplitude predicate, not because a generic
  nonfinite rule exists.
- A `None` PV result is an acquisition-stage failure and is outside a numeric
  post-reduction replay.
- A flat valid signal and a frozen diagnostic can be indistinguishable from
  sample values alone.
- Zero replacement is objective-dependent and may be undesirable for some
  minimization formulations.

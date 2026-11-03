# GOTAcc paper policy replay

This directory contains the reproducible offline evidence for the Section 4
machine-aware policy experiment. It invokes the production
`FelEnergyGuardPolicy`, `SampleGuardObjectivePolicy`, and
`SampleGuardConstraintPolicy` classes on fixed NumPy arrays. It never creates
an EPICS backend, loads pyepics, connects to an IOC, or writes a PV.

Run from the repository root on `feature/paper-reliability-fixes`:

```bash
PYTHONPATH=src python experiments/paper_policy_replay/replay_policy.py \
  --out-dir experiments/paper_policy_replay/results
PYTHONPATH=src python -m pytest -q \
  experiments/paper_policy_replay/test_replay_policy.py
```

The checked-in run used Python 3.11.5, NumPy 2.2.5, pytest 9.0.3, and
Matplotlib 3.10.0 against production commit
`010518fbece77a4630684065101033ff8588613d`. Exact environment data and source
hashes are in `results/environment.json`.

Key outputs:

- `results/objective_trace.csv` and `.json`: every input, statistic, processed
  objective, output-change flag, and directly observed event;
- `results/constraint_trace.csv` and `.json`: the corresponding constraint
  replay;
- `results/events.json`: only events emitted by production policy event sinks;
- `results/summary.json`: aggregate counts, configuration, expectation check,
  and limitations;
- `results/policy_replay_summary.png`: paper-ready overview of finite-sequence
  outputs and aggregate counts;
- `results/SHA256SUMS`: hashes of the generated artifacts.

See `EXPERIMENT_PROTOCOL.md` for the full protocol and
`../../docs/paper_policy_replay_results.md` for the interpreted results.

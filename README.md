# GOTAcc

**GOTAcc** stands for **General Optimization Toolkit for Accelerator Applications**.

GOTAcc is a Python toolkit for accelerator optimization workflows. The current
codebase is centered on a unified `TaskConfig` pipeline that can drive offline
objective functions, EPICS-based online tuning, command-line execution, and a
PyQt5 desktop GUI.

## Current Capabilities

- Single-objective optimizers: BO, TuRBO, RCDS
- Multi-objective optimizers: MOBO, MGGPO, MOPSO, NSGA-II
- Backend abstraction for offline callables and online EPICS evaluation
- Config loading from Python module paths, Python files, and YAML files
- PyQt5 GUI shell for task building, machine mapping, run monitoring, and result inspection

## Installation

Core install:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[yaml]"
pip install -e ".[gui]"
pip install -e ".[epics]"
pip install -e ".[full,dev]"
```

- `yaml`: enable YAML task config loading and YAML PV-library import
- `gui`: install the PyQt5-based GOTAcc Studio GUI
- `epics`: enable the online EPICS backend
- `full`: convenience extra for `yaml`, `gui`, and `epics`
- `dev`: local test and formatting tools

## Running GOTAcc

Installed entry points:

```bash
gotacc-run --config gotacc.configs.py_cfg.para_half
gotacc-gui
```

The `--config` argument accepts:

- Python module paths, for example `gotacc.configs.py_cfg.para_irfel`
- Python files, for example `src/gotacc/configs/py_cfg/para_irfel.py`
- YAML files, for example `src/gotacc/configs/yaml_cfg/irfel_bo.yaml`

Module execution is also supported:

```bash
python -m gotacc.runners.run_cli --config gotacc.configs.py_cfg.para_half
python -m gotacc.gui.main
python -m gotacc.runners.run_debug
```

## Repository Layout

```text
GOTAcc/
├─ CHANGELOG.md
├─ README.md
├─ pyproject.toml
├─ src/
│  └─ gotacc/
│     ├─ algorithms/
│     │  ├─ single_objective/
│     │  └─ multi_objective/
│     ├─ configs/
│     │  ├─ py_cfg/
│     │  ├─ yaml_cfg/
│     │  └─ pv_lists/
│     ├─ gui/
│     ├─ interfaces/
│     ├─ runners/
│     ├─ utils/
│     └─ version.py
├─ examples/
├─ tests/
└─ docs/
```

## Included Examples

- `examples/demo_single_bo_sphere.py`
- `examples/demo_multi_mobo_zdt1.py`
- `examples/demo_epics_mock_single.py`

## Notes

- The package version is sourced from `gotacc.version.__version__`.
- GUI runtime writes theme and matplotlib cache files under `.cache/`.
- Online workflows require a reachable EPICS environment and `pyepics`.

# Changelog

## 1.0.0 - 2026-04-09

- Consolidated runnable entry points around `gotacc.runners.run_cli` and the
  PyQt5 GUI in `gotacc.gui.main`
- Standardized the documented repository layout on the current
  `algorithms / configs / interfaces / runners / gui` structure
- Aligned packaging metadata with the repository version, current console
  scripts, and optional extras for YAML, GUI, and EPICS support
- Refreshed ignore rules for runtime caches and generated output directories

## 0.1.0

- Initial public alpha-stage repository cleanup
- Renamed `algorithms/multi_objectivce/` to `algorithms/multi_objective/`
- Added package `__init__.py` files
- Simplified `pyproject.toml` to match current public repository layout
- Added minimal `examples/` and `tests/`
- Updated README to reflect actual public structure

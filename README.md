# GOTAcc

**GOTAcc** stands for **General Optimization Toolkit for Accelerator Applications**.

It is a Python toolkit for accelerator optimization, intended for both:

- **online tuning**
  - EPICS-based machine optimization
  - injector / linac / FEL tuning
  - practical control-room deployment
- **design optimization**
  - beam dynamics optimization
  - constrained multi-objective studies
  - simulation-based accelerator design workflows

---

## Features

GOTAcc currently focuses on four layers:

1. **Algorithms**
   - single-objective optimization
   - multi-objective optimization
   - Bayesian and evolutionary approaches

2. **Interfaces**
   - EPICS-based online objective evaluation
   - offline test-function benchmarking

3. **Configs**
   - machine- or task-specific parameter sets
   - legacy flat-layout compatibility
   - project-specific optimization presets

4. **CLI / runners**
   - reusable entry points for online and offline workflows

---

## Package Layout

```text
GOTAcc/
├─ pyproject.toml
├─ README.md
├─ src/
│  └─ gotacc/
│     ├─ algorithms/
│     │  ├─ single_objective/
│     │  │  ├─ bo.py
│     │  │  ├─ turbo.py
│     │  │  └─ rcds.py
│     │  └─ multi_objective/
│     │     ├─ mobo.py
│     │     ├─ mggpo.py
│     │     ├─ mopso.py
│     │     └─ nsga2.py
│     ├─ interfaces/
│     │  ├─ epics.py
│     │  └─ test_functions.py
│     ├─ configs/
│     ├─ cli/
│     ├─ utils/
│     └─ compatibility/
├─ examples/
├─ tests/
└─ docs/
# GOTAcc

GOTAcc stands for **General Optimization Toolkit for Accelerator Applications**.

It is a Python toolkit for accelerator optimization, aimed at both:

- **design optimization**
  - beam dynamics design studies
  - constrained multi-objective optimization
  - simulation-based optimization
- **online tuning**
  - EPICS-based machine optimization
  - FEL / linac / injector tuning
  - practical accelerator operation support

---

## Current Status

GOTAcc is currently in an **alpha-stage, flat-layout** development phase.

At the current public stage, the repository contains:

- single-objective optimizers
- multi-objective optimizers
- an EPICS-based online evaluation interface
- basic offline benchmarking support
- project-level configuration and run scripts

This repository is intentionally kept in a relatively simple flat structure for now.
A later release may migrate to a `src/gotacc/` package layout.

---

## Currently Included Algorithms

### Single-objective optimization

Current public single-objective optimizers:

- **BO** — Bayesian Optimization
- **TuRBO** — Trust Region Bayesian Optimization
- **RCDS** — Robust Conjugate Direction Search

Typical use cases:

- maximizing FEL pulse energy
- minimizing beam size
- maximizing transport efficiency
- single-metric online tuning

### Multi-objective optimization

Current public multi-objective optimizers:

- **MOBO** — Multi-Objective Bayesian Optimization
- **MGGPO** — Multi-Generation Gaussian Process Optimizer
- **MOPSO** — Multi-Objective Particle Swarm Optimization
- **NSGA-II** — Non-dominated Sorting Genetic Algorithm II

Typical use cases:

- beam size vs. emittance
- transport efficiency vs. FEL pulse energy
- charge vs. energy spread vs. emittance
- constrained beam dynamics design optimization

---

## Repository Structure

After the first-round cleanup, the recommended repository structure is:

```text
GOTAcc/
├─ algorithms/
│  ├─ __init__.py
│  ├─ single_objective/
│  │  ├─ __init__.py
│  │  ├─ BOOptimizer.py
│  │  ├─ RCDSOptimizer.py
│  │  └─ TuRBOOptimizer.py
│  └─ multi_objective/
│     ├─ __init__.py
│     ├─ MGGPO.py
│     ├─ MOBOOptimizer.py
│     ├─ MOPSOOptimizer.py
│     └─ NSGA2Optimizer.py
├─ interfaces/
│  ├─ __init__.py
│  ├─ EpicsIOC.py
│  └─ test_function.py
├─ examples/
│  ├─ demo_single_bo_local.py
│  └─ demo_epics_ioc.py
├─ tests/
│  └─ test_imports.py
├─ README.md
├─ para_setup.py
├─ pyproject.toml
├─ requirements.txt
└─ runOpt.py
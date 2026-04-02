# GOTAcc

**GOTAcc** stands for **General Optimization Toolkit for Accelerator Applications**.

It is a Python toolkit for accelerator optimization, covering both:

- **design optimization**
  - STCF-BTP
  - other beam dynamics design problems
- **online tuning**
  - FELiChEM
  - HLS2
  - other EPICS-based accelerator systems

At the current stage, GOTAcc focuses on a practical combination of:

- optimization algorithms
- EPICS-based machine interaction
- simulation / test-function based benchmarking
- future extensibility toward a more general accelerator optimization framework

---

## Current Status

GOTAcc is currently under active development.

The repository currently contains:

- single-objective optimizers
- multi-objective optimizers
- EPICS-based online evaluation interface
- test-function based offline evaluation utilities

This project is currently organized in a transitional flat-module style and will gradually evolve into a more structured package layout.

---

## Algorithm Overview

### Single-objective optimization

These algorithms are intended for problems of the form:

$$x \mapsto y,\quad y \in \mathbb{R}$$

They are suitable for cases such as:

- maximizing FEL pulse energy
- minimizing beam size
- maximizing transport efficiency
- single-metric online tuning

Current single-objective algorithms include:

- **BO** — Bayesian Optimization
- **TuRBO** — Trust Region Bayesian Optimization
- **RCDS** — Robust Conjugate Direction Search
- **Rsimplex** — Robust Nelder–Mead Simplex
- **NN-BO** — Neural-network-based Bayesian Optimization *(under development)*

---

### Multi-objective optimization

These algorithms are intended for problems of the form:

$$x \mapsto \mathbf{y},\quad \mathbf{y} \in \mathbb{R}^m$$


They are suitable for cases such as:

- beam size vs. emittance
- transport efficiency vs. FEL pulse energy
- charge vs. energy spread vs. emittance
- constrained beam dynamics design optimization

Current multi-objective algorithms include:

- **MOBO** — Multi-Objective Bayesian Optimization
  - EHVI
  - qEHVI
  - qNEHVI
- **MG-GPO** — Multi-Generation Gaussian Process Optimizer
  - UCB / EHVI based
  - constrained version supported
- **MOPSO** — Multi-Objective Particle Swarm Optimization
- **NSGA-II** — Non-dominated Sorting Genetic Algorithm II

---

## Current Repository Structure

A practical near-term organization is:

```text
GOTAcc/
├─ pyproject.toml
├─ README.md
├─ runOpt.py
├─ para_setup.py
├─ algorithms/
│  ├─ single_objective/
│  │  ├─ BOOptimizer.py
│  │  ├─ TuRBOOptimizer.py
│  │  ├─ RCDSOptimizer.py
│  └─ multi_objectives/
│     ├─ MOBOOptimizer.py
│     ├─ MGGPO.py
│     ├─ NSGA2Optimizer.py
│     └─ MOPSOOptimizer.py
├─ interfaces/
│  ├─ EpicsIOC.py
│  └─ test_function.py
├─ configs/
├─ runners/
├─ save/
├─ docs/
└─ tests/

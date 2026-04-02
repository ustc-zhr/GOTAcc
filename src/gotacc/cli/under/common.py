from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np


def import_from_string(path: str):
    """
    Import an object from a dotted path.

    Example:
        import_from_string("gotacc.interfaces.epics.Obj_EpicsIoc")
    """
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def load_config_module(module_path: str):
    """
    Load a config module by dotted path.

    Example:
        gotacc.configs.half_online
        gotacc.configs.irfel_online
    """
    return importlib.import_module(module_path)


def ensure_parent_dir(path: str | None) -> None:
    if not path:
        return
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def build_absolute_bounds(initial_values, relative_bounds) -> np.ndarray:
    """
    Convert relative bounds to absolute bounds using initial machine values.

    Example:
        initial = [1.0, 2.0]
        rel = [[-0.1, 0.2], [-0.5, 0.5]]
        -> [[0.9, 1.2], [1.5, 2.5]]
    """
    ini = np.asarray(initial_values, dtype=float)
    rel = np.asarray(relative_bounds, dtype=float)

    if rel.shape != (len(ini), 2):
        raise ValueError(
            f"knobs_bounds shape mismatch: expected {(len(ini), 2)}, got {rel.shape}"
        )

    return np.column_stack([ini + rel[:, 0], ini + rel[:, 1]])


def build_single_optimizer(name: str, func, bounds, kwargs: dict[str, Any]):
    """
    Factory for stable public single-objective optimizers.
    """
    name = str(name).lower()

    if name == "bo":
        from gotacc.algorithms.single_objective.bo import BOOptimizer

        return BOOptimizer(func=func, bounds=bounds, **kwargs)

    if name == "turbo":
        from gotacc.algorithms.single_objective.turbo import TuRBOOptimizer

        return TuRBOOptimizer(func=func, bounds=bounds, **kwargs)

    if name == "rcds":
        from gotacc.algorithms.single_objective.rcds import RCDSOptimizer

        return RCDSOptimizer(func=func, bounds=bounds, **kwargs)

    raise ValueError(
        f"Unsupported single-objective optimizer: {name}. "
        "Supported: bo, turbo, rcds."
    )


def build_multi_optimizer(
    name: str,
    func,
    bounds,
    n_objectives: int,
    ref_point,
    kwargs: dict[str, Any],
):
    """
    Factory for stable public multi-objective optimizers.
    """
    name = str(name).lower()

    if name == "mobo":
        from gotacc.algorithms.multi_objective.mobo import MOBOOptimizer

        return MOBOOptimizer(
            func=func,
            bounds=bounds,
            n_objectives=n_objectives,
            ref_point=ref_point,
            **kwargs,
        )

    if name == "mggpo":
        from gotacc.algorithms.multi_objective.mggpo import MultiObjectiveMGGPO

        return MultiObjectiveMGGPO(
            func=func,
            bounds=bounds,
            n_objectives=n_objectives,
            ref_point=ref_point,
            **kwargs,
        )

    if name == "mopso":
        from gotacc.algorithms.multi_objective.mopso import MOPSOOptimizer

        return MOPSOOptimizer(
            func=func,
            bounds=bounds,
            n_objectives=n_objectives,
            ref_point=ref_point,
            **kwargs,
        )

    if name == "nsga2":
        from gotacc.algorithms.multi_objective.nsga2 import NSGA2Optimizer

        return NSGA2Optimizer(
            func=func,
            bounds=bounds,
            n_objectives=n_objectives,
            ref_point=ref_point,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported multi-objective optimizer: {name}. "
        "Supported: mobo, mggpo, mopso, nsga2."
    )


def maybe_save_history(opt, path: str | None = None):
    if not hasattr(opt, "save_history"):
        return
    ensure_parent_dir(path)
    if path is None:
        opt.save_history()
    else:
        try:
            opt.save_history(path=path)
        except TypeError:
            # Backward-compatible fallback
            opt.save_history()


def maybe_plot_convergence(opt, path: str | None = None):
    if not hasattr(opt, "plot_convergence"):
        return
    ensure_parent_dir(path)
    try:
        opt.plot_convergence(path=path)
    except TypeError:
        opt.plot_convergence()


def require_callable(obj, name: str):
    if not callable(obj):
        raise TypeError(f"{name} must be callable.")
    return obj
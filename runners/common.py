import argparse
import importlib
import os
from typing import Any, Dict

import numpy as np


def ensure_parent_dir(path: str | None) -> None:
    if not path:
        return
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def load_config_module(module_path: str):
    """
    Load config module by dotted path, e.g.:
        configs.hls2_single
        configs.irfel_multi
    """
    return importlib.import_module(module_path)


def build_absolute_bounds(initial_values, relative_bounds):
    ini = np.asarray(initial_values, dtype=float)
    rel = np.asarray(relative_bounds, dtype=float)

    if rel.shape != (len(ini), 2):
        raise ValueError(
            f"knobs_bounds shape mismatch: expected {(len(ini), 2)}, got {rel.shape}"
        )

    return np.column_stack([ini + rel[:, 0], ini + rel[:, 1]])


def build_single_optimizer(name: str, func, bounds, kwargs: Dict[str, Any]):
    """
    Stable public single-objective builder for the current public repo.
    For now, keep only the optimizers already used in the public single runner.
    """
    name = str(name).lower()

    if name == "bo":
        from algorithms.single_objective.BOOptimizer import BOOptimizer

        return BOOptimizer(func=func, bounds=bounds, **kwargs)

    if name == "turbo":
        from algorithms.single_objective.TuRBOOptimizer import TuRBOOptimizer

        return TuRBOOptimizer(func=func, bounds=bounds, **kwargs)

    raise ValueError(
        f"Unsupported single-objective optimizer: {name}. "
        "Current stable public runner supports: 'bo', 'turbo'."
    )


def build_multi_optimizer(
    name: str,
    func,
    bounds,
    n_objectives: int,
    ref_point,
    kwargs: Dict[str, Any],
):
    """
    Stable public multi-objective builder aligned with the public algorithm files.
    """
    name = str(name).lower()

    if name == "mobo":
        from algorithms.multi_objective.MOBOOptimizer import MOBOOptimizer

        return MOBOOptimizer(
            func=func,
            bounds=bounds,
            n_objectives=n_objectives,
            ref_point=ref_point,
            **kwargs,
        )

    if name == "mggpo":
        from algorithms.multi_objective.MGGPO import MultiObjectiveMGGPO

        return MultiObjectiveMGGPO(
            func=func,
            bounds=bounds,
            n_objectives=n_objectives,
            ref_point=ref_point,
            **kwargs,
        )

    if name == "mopso":
        from algorithms.multi_objective.MOPSOOptimizer import MOPSOOptimizer

        return MOPSOOptimizer(
            func=func,
            bounds=bounds,
            n_objectives=n_objectives,
            ref_point=ref_point,
            **kwargs,
        )

    if name == "nsga2":
        from algorithms.multi_objective.NSGA2Optimizer import NSGA2Optimizer

        return NSGA2Optimizer(
            func=func,
            bounds=bounds,
            n_objectives=n_objectives,
            ref_point=ref_point,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported multi-objective optimizer: {name}. "
        "Supported: 'mobo', 'mggpo', 'mopso', 'nsga2'."
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
            opt.save_history()


def maybe_plot_convergence(opt, path: str | None = None):
    if not hasattr(opt, "plot_convergence"):
        return
    ensure_parent_dir(path)
    try:
        opt.plot_convergence(path=path)
    except TypeError:
        opt.plot_convergence()


def make_parser(default_config: str):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help=f"Config module path, e.g. {default_config}",
    )
    return parser
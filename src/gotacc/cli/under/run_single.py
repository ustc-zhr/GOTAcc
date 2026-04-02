from __future__ import annotations

import argparse
import time

import numpy as np

from gotacc.cli.under.common import (
    build_absolute_bounds,
    build_single_optimizer,
    import_from_string,
    load_config_module,
    maybe_plot_convergence,
    maybe_save_history,
)


def build_interface(machine_cfg: dict, run_cfg: dict):
    """
    Build the evaluation interface for a single-objective workflow.

    Default:
        gotacc.interfaces.epics.Obj_EpicsIoc

    You may override it via:
        machine_cfg["interface_class"] = "some.module.ClassName"
    """
    interface_class_path = machine_cfg.get(
        "interface_class", "gotacc.interfaces.epics.Obj_EpicsIoc"
    )
    InterfaceClass = import_from_string(interface_class_path)

    return InterfaceClass(
        knobs_pvnames=machine_cfg["knobs_pvnames"],
        obj_pvnames=machine_cfg["obj_pvnames"],
        obj_weights=machine_cfg["obj_weights"],
        obj_samples=machine_cfg["obj_samples"],
        obj_math=machine_cfg["obj_math"],
        interval=machine_cfg["interval"],
        log_path=run_cfg.get("obj_log_path", "save/template_single.opt"),
        readback_check=run_cfg.get("readback_check", False),
        readback_tol=run_cfg.get("readback_tol", None),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run GOTAcc single-objective workflow.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Dotted config module path, e.g. gotacc.configs.half_online",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config_module(args.config)
    machine_cfg = cfg.machine_para()
    opt_cfg = cfg.optimizer_para()
    run_cfg = cfg.run_para()

    t0 = time.time()
    objhub = None

    try:
        objhub = build_interface(machine_cfg, run_cfg)

        initial_values = np.asarray(objhub.init_knob_value(), dtype=float)
        bounds = build_absolute_bounds(initial_values, machine_cfg["knobs_bounds"])

        print("=== machine config ===")
        print(machine_cfg)
        print("=== optimizer config ===")
        print(opt_cfg)
        print("=== run config ===")
        print(run_cfg)
        print("=== absolute bounds ===")
        print(bounds)

        opt = build_single_optimizer(
            name=opt_cfg["name"],
            func=objhub.evaluate_func,
            bounds=bounds,
            kwargs=opt_cfg.get("kwargs", {}),
        )

        opt.optimize()

        if run_cfg.get("save_history", True):
            maybe_save_history(opt, run_cfg.get("history_path"))

        if run_cfg.get("plot", True):
            maybe_plot_convergence(opt, run_cfg.get("plot_path"))

        if run_cfg.get("set_best", True) and hasattr(objhub, "set_best"):
            objhub.set_best()

        print(f"[run_single] total time = {time.time() - t0:.2f} s")

    except Exception:
        if objhub is not None and run_cfg.get("restore_on_error", True):
            if hasattr(objhub, "restore_initial"):
                objhub.restore_initial()
        raise


if __name__ == "__main__":
    main()
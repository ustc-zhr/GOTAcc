import time
import numpy as np

from interfaces.EpicsIOC_multi import MultiObjEpicsIOC
from runners.common import (
    build_absolute_bounds,
    build_multi_optimizer,
    load_config_module,
    make_parser,
    maybe_plot_convergence,
    maybe_save_history,
)


def main():
    parser = make_parser(default_config="configs.irfel_multi")
    args = parser.parse_args()

    cfg = load_config_module(args.config)
    machine_cfg = cfg.machine_para()
    opt_cfg = cfg.optimizer_para()
    run_cfg = cfg.run_para()

    t0 = time.time()
    objhub = None

    try:
        objhub = MultiObjEpicsIOC(
            knobs_pvnames=machine_cfg["knobs_pvnames"],
            obj_pvnames=machine_cfg["obj_pvnames"],
            obj_weights=machine_cfg.get("obj_weights", [1.0] * len(machine_cfg["obj_pvnames"])),
            obj_samples=machine_cfg["obj_samples"],
            obj_math=machine_cfg["obj_math"],
            interval=machine_cfg["interval"],
            log_path=run_cfg.get("obj_log_path", "save/template_multi.opt"),
            linked_pvs=machine_cfg.get("linked_pvs", []),
            zero_guard_index=run_cfg.get("zero_guard_index", None),
            zero_guard_offset=run_cfg.get("zero_guard_offset", 0.0),
            readback_check=run_cfg.get("readback_check", False),
            readback_tol=run_cfg.get("readback_tol", None),
        )

        ini_values = np.asarray(objhub.init_knob_value(), dtype=float)
        bounds = build_absolute_bounds(ini_values, machine_cfg["knobs_bounds"])
        n_objectives = len(machine_cfg["obj_pvnames"])

        opt_kwargs = dict(opt_cfg.get("kwargs", {}))
        ref_point = np.asarray(
            opt_kwargs.pop("ref_point", machine_cfg.get("ref_point")),
            dtype=float,
        )

        print("=== multi-objective machine config ===")
        print(machine_cfg)
        print("=== multi-objective optimizer config ===")
        print(opt_cfg)
        print("=== multi-objective run config ===")
        print(run_cfg)
        print("Optimization bounds:")
        print(bounds)
        print("Reference point:")
        print(ref_point)

        opt = build_multi_optimizer(
            name=opt_cfg["name"],
            func=objhub.evaluate_func,
            bounds=bounds,
            n_objectives=n_objectives,
            ref_point=ref_point,
            kwargs=opt_kwargs,
        )

        opt.optimize()
        print(f"Total time: {time.time() - t0:.2f} s")

        if run_cfg.get("save_history", True):
            maybe_save_history(opt, run_cfg.get("history_path"))

        if run_cfg.get("set_best", True):
            objhub.set_best(
                maximize=opt_cfg.get("maximize", False),
                selector=run_cfg.get("best_selector", "sum"),
            )

        if run_cfg.get("plot", True):
            maybe_plot_convergence(opt, run_cfg.get("plot_path"))

    except Exception as e:
        print(f"[run_multi_epics] error: {e}")
        if objhub is not None and run_cfg.get("restore_on_error", True):
            try:
                objhub.restore_initial()
            except Exception as restore_err:
                print(f"[run_multi_epics] restore failed: {restore_err}")
        raise


if __name__ == "__main__":
    main()
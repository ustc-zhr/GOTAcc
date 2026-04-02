import time
import numpy as np

from interfaces.EpicsIOC import Obj_EpicsIoc
from gotacc.cli.under.common import (
    build_absolute_bounds,
    build_single_optimizer,
    load_config_module,
    make_parser,
    maybe_plot_convergence,
    maybe_save_history,
)


def main():
    parser = make_parser(default_config="configs.hls2_single")
    args = parser.parse_args()

    cfg = load_config_module(args.config)
    machine_cfg = cfg.machine_para()
    opt_cfg = cfg.optimizer_para()
    run_cfg = cfg.run_para()

    t0 = time.time()
    objhub = None

    try:
        objhub = Obj_EpicsIoc(
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

        ini_values = np.asarray(objhub.init_knob_value(), dtype=float)
        bounds = build_absolute_bounds(ini_values, machine_cfg["knobs_bounds"])

        print("=== single-objective machine config ===")
        print(machine_cfg)
        print("=== single-objective optimizer config ===")
        print(opt_cfg)
        print("=== single-objective run config ===")
        print(run_cfg)
        print("Optimization bounds:")
        print(bounds)

        opt = build_single_optimizer(
            name=opt_cfg["name"],
            func=objhub.evaluate_func,
            bounds=bounds,
            kwargs=opt_cfg.get("kwargs", {}),
        )

        opt.optimize()
        print(f"Total time: {time.time() - t0:.2f} s")

        if run_cfg.get("save_history", True):
            maybe_save_history(opt, run_cfg.get("history_path"))

        if run_cfg.get("set_best", True):
            objhub.set_best()

        if run_cfg.get("plot", True):
            maybe_plot_convergence(opt, run_cfg.get("plot_path"))

    except Exception as e:
        print(f"[run_single_epics] error: {e}")
        if objhub is not None and run_cfg.get("restore_on_error", True):
            try:
                objhub.restore_initial()
            except Exception as restore_err:
                print(f"[run_single_epics] restore failed: {restore_err}")
        raise


if __name__ == "__main__":
    main()
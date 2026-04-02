from configs import para_half as legacy


def machine_para():
    cfg = legacy.machine_para()
    return {
        "knobs_pvnames": cfg["knobs_pvnames"],
        "knobs_bounds": cfg["knobs_bounds"],
        "obj_pvnames": cfg["obj_pvnames"],
        "obj_weights": cfg["obj_weights"],
        "obj_samples": cfg["obj_samples"],
        "obj_math": cfg["obj_math"],
        "interval": cfg["interval"],
    }


def optimizer_para():
    cfg = legacy.optimizer_para()
    return {
        "name": cfg["name"],
        "kwargs": cfg.get("kwargs", {}),
    }


def run_para():
    cfg = legacy.run_para()
    return {
        "obj_log_path": cfg.get("obj_log_path", "save/template_single.opt"),
        "history_path": cfg.get("history_path", "save/history_single.csv"),
        "plot_path": cfg.get("plot_path", "save/convergence_single.png"),
        "save_history": cfg.get("save_history", True),
        "plot": cfg.get("plot", True),
        "set_best": cfg.get("set_best", True),
        "restore_on_error": cfg.get("restore_on_error", True),
        "readback_check": cfg.get("readback_check", False),
        "readback_tol": cfg.get("readback_tol", None),
    }
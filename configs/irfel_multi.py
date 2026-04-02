from configs import para_irfel as legacy


def machine_para():
    knobs_pvnames, knobs_bounds = legacy.knob_para()
    obj_pvnames, obj_weights, obj_samples, obj_math, interval = legacy.obj_para()

    return {
        "knobs_pvnames": knobs_pvnames,
        "knobs_bounds": knobs_bounds,
        "obj_pvnames": obj_pvnames,
        "obj_weights": obj_weights,
        "obj_samples": obj_samples,
        "obj_math": obj_math,
        "interval": interval,
        # Keep this as a device-specific setting here, not in the runner.
        "linked_pvs": [
            ("IRFEL:PS:QM15:current:ao", 11),
            ("IRFEL:PS:QM16:current:ao", 10),
        ],
        # Default ref point; tune it later according to your actual objective scale.
        "ref_point": [10.0, 20.0],
    }


def optimizer_para():
    return {
        "name": "mobo",
        "maximize": False,
        "kwargs": {
            "kernel_type": "maternwhite",
            "gp_restarts": 5,
            "acq": "qnehvi",
            "acq_optimizer": "optimize_acqf",
            "acq_opt_kwargs": {
                "num_restarts": 8,
                "raw_samples": 512,
                "qehvi_batch": 1,
            },
            "n_init": 20,
            "n_iter": 80,
            "random_state": 120,
            "verbose": True,
        },
    }


def run_para():
    return {
        "obj_log_path": "save/template_multi.opt",
        "history_path": "save/history_multi.csv",
        "plot_path": "save/convergence_multi.png",
        "save_history": True,
        "plot": True,
        "set_best": True,
        "restore_on_error": True,
        "best_selector": "sum",
        "zero_guard_index": 1,
        "zero_guard_offset": 100.0,
        "readback_check": False,
        "readback_tol": None,
    }
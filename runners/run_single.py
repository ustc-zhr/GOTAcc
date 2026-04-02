import os
import time
import numpy as np


from interfaces.EpicsIOC import Obj_EpicsIoc
from configs import para_half as para_setup


def _ensure_parent_dir(path):
    if path is None:
        return
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def build_optimizer(name, func, bounds, kwargs):
    """
    最小补丁版：
    当前只先支持单目标在线优化器：
    - BO
    - TuRBO

    这样与你现在 Obj_EpicsIoc 返回“加权单目标标量”的接口保持一致。
    """
    name = str(name).lower()

    if name == "bo":
        from algorithms.single_objective.BOOptimizer import BOOptimizer
        return BOOptimizer(func=func, bounds=bounds, **kwargs)

    elif name == "turbo":
        from algorithms.single_objective.TuRBOOptimizer import TuRBOOptimizer
        return TuRBOOptimizer(func=func, bounds=bounds, **kwargs)

    else:
        raise ValueError(
            f"当前最小补丁版 runOpt.py 仅支持 'bo' 或 'turbo'，收到: {name}\n"
            f"若要在线跑 MOBO / NSGA2 / MOPSO / MGGPO，下一步需要先把 "
            f"Obj_EpicsIoc.evaluate_func 从“加权单目标标量”改成“返回目标向量”。"
        )


if __name__ == "__main__":
    t0 = time.time()
    objhub = None

    try:
        # ------------------------------------------------------------------
        # 1) 统一读取配置
        # ------------------------------------------------------------------
        machine_cfg = para_setup.machine_para()
        opt_cfg = para_setup.optimizer_para()
        run_cfg = para_setup.run_para()

        print("=== machine config ===")
        print(machine_cfg)
        print("=== optimizer config ===")
        print(opt_cfg)
        print("=== run config ===")
        print(run_cfg)

        # ------------------------------------------------------------------
        # 2) 建立 EPICS 目标函数接口
        # ------------------------------------------------------------------
        objhub = Obj_EpicsIoc(
            knobs_pvnames=machine_cfg["knobs_pvnames"],
            obj_pvnames=machine_cfg["obj_pvnames"],
            obj_weights=machine_cfg["obj_weights"],
            obj_samples=machine_cfg["obj_samples"],
            obj_math=machine_cfg["obj_math"],
            interval=machine_cfg["interval"],
            log_path=run_cfg.get("obj_log_path", "template.opt"),
            readback_check=run_cfg.get("readback_check", False),
            readback_tol=run_cfg.get("readback_tol", None),
        )

        # ------------------------------------------------------------------
        # 3) 由当前机时刻的实际值 + 相对改变量，生成绝对 bounds
        # ------------------------------------------------------------------
        ini_values = np.asarray(objhub.init_knob_value(), dtype=float)

        knobs_bounds = np.asarray(machine_cfg["knobs_bounds"], dtype=float)
        if knobs_bounds.shape != (len(ini_values), 2):
            raise ValueError(
                f"knobs_bounds 形状错误，期望 {(len(ini_values), 2)}，实际 {knobs_bounds.shape}"
            )

        bounds = np.column_stack([
            ini_values + knobs_bounds[:, 0],
            ini_values + knobs_bounds[:, 1]
        ])
        print("Optimization bounds:")
        print(bounds)

        # ------------------------------------------------------------------
        # 4) 创建优化器
        # ------------------------------------------------------------------
        opt = build_optimizer(
            name=opt_cfg["name"],
            func=objhub.evaluate_func,
            bounds=bounds,
            kwargs=opt_cfg.get("kwargs", {})
        )

        # ------------------------------------------------------------------
        # 5) 运行优化
        # ------------------------------------------------------------------
        opt.optimize()
        print(f"Total time: {time.time() - t0:.2f} s")

        # ------------------------------------------------------------------
        # 6) 后处理
        # ------------------------------------------------------------------
        if run_cfg.get("save_history", True) and hasattr(opt, "save_history"):
            history_path = run_cfg.get("history_path", None)
            _ensure_parent_dir(history_path)
            if history_path is None:
                opt.save_history()
            else:
                opt.save_history(path=history_path)

        if run_cfg.get("set_best", True):
            objhub.set_best()

        if run_cfg.get("plot", True) and hasattr(opt, "plot_convergence"):
            plot_path = run_cfg.get("plot_path", None)
            try:
                # BO / TuRBO 这种支持 path 参数
                opt.plot_convergence(path=plot_path)
            except TypeError:
                # 兼容不带 path 参数的实现
                opt.plot_convergence()

    except Exception as e:
        print(f"错误: {e}")

        # 最小补丁：异常时恢复初始值
        if objhub is not None and para_setup.run_para().get("restore_on_error", True):
            try:
                objhub.restore_initial()
            except Exception as restore_err:
                print(f"恢复初始值失败: {restore_err}")
import time
import numpy as np

from configs import para_irfel as para_setup
from interfaces.EpicsIOC_multi import MultiObjEpicsIOC

from algorithms.multi_objectives.MOBOOptimizer import MOBOOptimizer
from algorithms.multi_objectives.MGGPO import MultiObjectiveMGGPO
from algorithms.multi_objectives.MOPSOOptimizer import MOPSOOptimizer
from algorithms.multi_objectives.NSGA2Optimizer import NSGA2Optimizer


def build_optimizer(name, func, bounds, n_objectives, ref_point):
    name = str(name).lower()

    if name == "mobo":
        return MOBOOptimizer(
            func=func,
            bounds=bounds,
            n_objectives=n_objectives,
            kernel_type="maternwhite",
            gp_restarts=5,
            acq="qnehvi",
            acq_optimizer="optimize_acqf",
            acq_opt_kwargs={"num_restarts": 8, "raw_samples": 512, "qehvi_batch": 1},
            ref_point=ref_point,
            maximize=False,
            n_init=20,
            n_iter=80,
            random_state=120,
            verbose=True,
        )

    elif name == "mggpo":
        return MultiObjectiveMGGPO(
            func=func,
            bounds=bounds,
            n_objectives=n_objectives,
            n_constraints=0,
            kernel_type="rbf",
            gp_restarts=5,
            pop_size=80,
            acq_mode="ehvi",
            ref_point=ref_point,
            n_generations=50,
            maximize=False,
            random_state=120,
            verbose=True,
        )

    elif name == "mopso":
        return MOPSOOptimizer(
            func=func,
            bounds=bounds,
            n_objectives=n_objectives,
            pop_size=100,
            n_generations=100,
            ref_point=ref_point,
            maximize=False,
            random_state=120,
            verbose=True,
        )

    elif name == "nsga2":
        return NSGA2Optimizer(
            func=func,
            bounds=bounds,
            n_objectives=n_objectives,
            pop_size=100,
            n_generations=100,
            ref_point=ref_point,
            maximize=False,
            random_state=120,
            verbose=True,
        )

    else:
        raise ValueError(f"不支持的多目标优化器: {name}")


def main():
    t0 = time.time()
    objhub = None

    try:
        # ------------------------------------------------------------
        # 1. 从配置获取 knobs / objectives
        # ------------------------------------------------------------
        knobs_pvnames, knobs_bounds = para_setup.knob_para()
        obj_pvnames, obj_weights, obj_samples, obj_math, interval = para_setup.obj_para()

        print("变量:", (knobs_pvnames, knobs_bounds))
        print("目标:", (obj_pvnames, obj_weights, obj_samples, obj_math, interval))

        # ------------------------------------------------------------
        # 2. 特殊联动写入（兼容你当前 IRFEL 版本）
        #    当前上传代码里：
        #    QM15 <- x[11]
        #    QM16 <- x[10]
        # ------------------------------------------------------------
        linked_pvs = [
            ("IRFEL:PS:QM15:current:ao", 11),
            ("IRFEL:PS:QM16:current:ao", 10),
        ]

        objhub = MultiObjEpicsIOC(
            knobs_pvnames=knobs_pvnames,
            obj_pvnames=obj_pvnames,
            obj_weights=obj_weights,
            obj_samples=obj_samples,
            obj_math=obj_math,
            interval=interval,
            log_path="save/template_multi.opt",
            linked_pvs=linked_pvs,
            zero_guard_index=1,      # 兼容你当前上传代码里的特殊处理
            zero_guard_offset=100.0,
        )

        # ------------------------------------------------------------
        # 3. 计算绝对边界
        # ------------------------------------------------------------
        ini_values = objhub.init_knob_value()
        bounds = np.array(
            [
                [ini_values[i] + knobs_bounds[i][0], ini_values[i] + knobs_bounds[i][1]]
                for i in range(len(ini_values))
            ],
            dtype=float,
        )
        print("Optimization bounds:")
        print(bounds)

        # ------------------------------------------------------------
        # 4. 多目标设置
        # ------------------------------------------------------------
        n_objectives = len(obj_pvnames)
        ref_point = np.array([10.0, 20.0], dtype=float)  # 先沿用你当前 runOpt.py

        # 可选: "mobo" / "mggpo" / "mopso" / "nsga2"
        optimizer_name = "mobo"

        opt = build_optimizer(
            name=optimizer_name,
            func=objhub.evaluate_func,
            bounds=bounds,
            n_objectives=n_objectives,
            ref_point=ref_point,
        )

        # ------------------------------------------------------------
        # 5. 运行优化
        # ------------------------------------------------------------
        opt.optimize()
        print(f"time: {time.time() - t0:.2f} s")

        # ------------------------------------------------------------
        # 6. 保存历史
        # ------------------------------------------------------------
        if hasattr(opt, "save_history"):
            opt.save_history()

        # ------------------------------------------------------------
        # 7. 将“best”设置回 EPICS
        #    当前先沿用你现有逻辑：对返回 obj_vals 求和，最小者为 best
        # ------------------------------------------------------------
        objhub.set_best(maximize=False, selector="sum")

        # ------------------------------------------------------------
        # 8. 绘图
        # ------------------------------------------------------------
        if hasattr(opt, "plot_convergence"):
            opt.plot_convergence()

    except Exception as e:
        print(f"错误: {e}")
        if objhub is not None:
            try:
                objhub.restore_initial()
            except Exception as restore_err:
                print(f"恢复初始值失败: {restore_err}")


if __name__ == "__main__":
    main()
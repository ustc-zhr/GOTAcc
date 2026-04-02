"""决策变量、目标函数、优化器与运行参数配置（最小补丁版）"""

# ----------------------------------------------------------------------
# 目标函数 setup
# - 用负权重做加权和
# - 优化器内部继续做“最大化这个加权目标”
# ----------------------------------------------------------------------
def obj_para():
    obj_pvnames = ['HALF:IN:FLAG:PRF07:sigx', 'HALF:IN:FLAG:PRF07:sigy']
    obj_weights = [-1.0, -1.0]
    obj_samples = 3
    obj_math = ['mean', 'mean']   # 每个目标对应 'mean' 或 'std'
    interval = 1                  # 写入后/采样间隔（秒）

    return obj_pvnames, obj_weights, obj_samples, obj_math, interval


# ----------------------------------------------------------------------
# knob setup
# 当前保持与你现有 HALF 配置一致
# ----------------------------------------------------------------------
def knob_para():
    cor_x_all = ['XC01', 'XC02', 'XC03', 'XC04', 'XC05', 'XC06', 'XC07', 'XC08', 'XC09', 'XC10', 'XC11']
    cor_y_all = ['YC01', 'YC02', 'YC03', 'YC04', 'YC05', 'YC06', 'YC07', 'YC08', 'YC09', 'YC10', 'YC11']
    quad_all = ['QT01', 'QT02']

    # 选择的 knobs
    indices = []  # 选择 x 向校正子
    cor_x_select = [cor_x_all[i] for i in indices]

    indices = []  # 选择 y 向校正子
    cor_y_select = [cor_y_all[i] for i in indices]

    indices = [0, 1]  # 选择四极
    quad_select = [quad_all[i] for i in indices]

    # x correctors
    cor_x_pvlist = []
    cor_x_bounds = []
    for cor in cor_x_select:
        cor_x_pvlist.append(f"HALF:IN:COR:{cor}:ao")
        cor_x_bounds.append((-5, 5))

    # y correctors
    cor_y_pvlist = []
    cor_y_bounds = []
    for cor in cor_y_select:
        cor_y_pvlist.append(f"HALF:IN:COR:{cor}:ao")
        cor_y_bounds.append((-5, 5))

    # quads
    quad_pvlist = []
    quad_bounds = []
    for quad in quad_select:
        quad_pvlist.append(f"HALF:IN:QUAD:{quad}:K1")
        quad_bounds.append((-2, 2))

    knob_pvlist = cor_x_pvlist + cor_y_pvlist + quad_pvlist
    knob_bounds = cor_x_bounds + cor_y_bounds + quad_bounds

    return knob_pvlist, knob_bounds


# ----------------------------------------------------------------------
# 统一的 machine 配置出口
# ----------------------------------------------------------------------
def machine_para():
    knobs_pvnames, knobs_bounds = knob_para()
    obj_pvnames, obj_weights, obj_samples, obj_math, interval = obj_para()

    return {
        "knobs_pvnames": knobs_pvnames,
        "knobs_bounds": knobs_bounds,
        "obj_pvnames": obj_pvnames,
        "obj_weights": obj_weights,
        "obj_samples": obj_samples,
        "obj_math": obj_math,
        "interval": interval,
    }


# ----------------------------------------------------------------------
# 优化器配置
# 当前最小补丁只建议先支持：
# - "bo"
# - "turbo"
#
# 如需切换算法，只改 name 和 kwargs
# ----------------------------------------------------------------------
def optimizer_para():
    return {
        "name": "bo",   # 可选: "bo", "turbo"
        "kwargs": {
            # ---- BO 默认参数 ----
            "kernel_type": "matern",
            "gp_restarts": 5,
            "acq": "ucb",
            "acq_para": 2.0,
            "acq_para_kwargs": {"beta_strategy": "inv_decay", "beta_lam": 0.01},
            "acq_optimizer": "optimize_acqf",   # 'random', 'sobol', 'optimize_acqf'
            "acq_opt_kwargs": {"num_restarts": 8, "raw_samples": 256, "n_candidates": 8192},
            "n_init": 5,
            "n_iter": 15,
            "random_state": 120,
            "verbose": True,
        }
    }


# ----------------------------------------------------------------------
# 运行过程配置
# ----------------------------------------------------------------------
def run_para():
    return {
        "save_history": True,
        "history_path": None,           # None 表示走各优化器默认命名
        "plot": True,
        "plot_path": None,              # BO/TuRBO 支持 path=None；这里先不强制
        "set_best": True,               # 优化结束后是否把最优点写回 EPICS
        "restore_on_error": True,       # 出错时是否恢复初始值
        "obj_log_path": "template.opt", # Obj_EpicsIoc 的评估记录文件
        "readback_check": False,        # 最小补丁：默认先关，稳定后可开
        "readback_tol": 1e-6,           # 开启 readback_check 后使用
    }


def print_config():
    print("=== machine config ===")
    print(machine_para())
    print("=== optimizer config ===")
    print(optimizer_para())
    print("=== run config ===")
    print(run_para())


if __name__ == "__main__":
    print_config()
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
    cor_x_all = [
        'LA_PS_HIC1:current:ao','LA_PS_HIC2:current:ao','LA_PS_HIC3:current:ao',#0-2

        'LA_PS_HC1:current:ao','LA_PS_HC2:current:ao','LA_PS_HC3:current:ao','LA_PS_HC4:current:ao',#3-6
        'LA_PS_HC5:current:ao','LA_PS_HC6:current:ao','LA_PS_HC7:current:ao','LA_PS_HC8:current:ao',#7-10

        'TL_PS_HC1:current:ao','TL_PS_HC2:current:ao','TL_PS_HC3:current:ao','TL_PS_HC4:current:ao',#11-14
        'TL_PS_HC5:current:ao','TL_PS_HC6:current:ao','TL_PS_HC7:current:ao','TL_PS_HC8:current:ao', 'TL_PS_HC9:current:ao'#15-19
    ]

    cor_y_all = [
        'LA_PS_VIC1:current:ao','LA_PS_VIC2:current:ao','LA_PS_VIC3:current:ao', #0-2

        'LA_PS_VC1:current:ao','LA_PS_VC2:current:ao','LA_PS_VC3:current:ao','LA_PS_VC4:current:ao',#3-6
        'LA_PS_VC5:current:ao','LA_PS_VC6:current:ao','LA_PS_VC7:current:ao','LA_PS_VC8:current:ao',#7-10

        'TL_PS_VC1:current:ao','TL_PS_VC2:current:ao','TL_PS_VC3:current:ao','TL_PS_VC4:current:ao',#11-14
        'TL_PS_VC5:current:ao','TL_PS_VC6:current:ao','TL_PS_VC7:current:ao','TL_PS_VC8:current:ao', 'TL_PS_VC9:current:ao'#15-19
    ]
    quad_all = [
        'LA_PS_Q1:current:ao','LA_PS_Q2:current:ao','LA_PS_Q3:current:ao','LA_PS_Q4:current:ao',#0-3
        'LA_PS_Q5:current:ao','LA_PS_Q6:current:ao','LA_PS_Q7:current:ao','LA_PS_Q8:current:ao',#4-7
        'LA_PS_Q9:current:ao','LA_PS_Q10:current:ao','LA_PS_Q11:current:ao','LA_PS_Q12:current:ao',#8-11
        
        'LA_PS_AQ2:current:ao','LA_PS_AQ3:current:ao',#12-13

        'TL_PS_Q1:current:ao','TL_PS_Q2:current:ao','TL_PS_Q3:current:ao','TL_PS_Q4:current:ao',#14-17
        'TL_PS_Q5:current:ao','TL_PS_Q6:current:ao','TL_PS_Q7:current:ao','TL_PS_Q8:current:ao',#18-21
        'TL_PS_Q9:current:ao','TL_PS_Q10:current:ao','TL_PS_Q11:current:ao','TL_PS_Q12:current:ao'#22-25
    ]

    cavity_all = [
        'LA-PCI-KLY1:Elecontrolphaser','LA-PCI-KLY2:Elecontrolphaser', 'LA-PCI-KLY3:Elecontrolphaser','LA-PCI-KLY4:Elecontrolphaser',#0-3
        'LA-PCI-KLY5:Elecontrolphaser','LA-PCI-KLY6:Elecontrolphaser','LA-PCI-KLY7:Elecontrolphaser','LA-PCI-KLY8:Elecontrolphaser',#4-7
        'LA-PCI-KLY9:Elecontrolphaser' #8
    ]

    bend_all = [
        'TL_PS_SM:current:ao','TL_PS_AM:current:ao', #0-1   switch
        'TL_PS_BM1256:current:ao','TL_PS_BM1:current:ao','TL_PS_BM2:current:ao','TL_PS_BM5:current:ao','TL_PS_BM6:current:ao',#2-6
        'TL_PS_BM34:current:ao'
    ]

    # 选择的knobs
    indices = [7, 8, 9, 10]
    # indices = [0,1,2,3]  # 选择第x、x、x个correctors
    cor_x_pvlist = [cor_x_all[i] for i in indices]
    cor_y_pvlist = [cor_y_all[i] for i in indices]
    cor_x_bounds = [[-3, 3]]*len(indices)
    cor_y_bounds = [[-3, 3]]*len(indices)

    indices = []  # 选择第x、x、x个quads
    quad_pvlist = [quad_all[i] for i in indices]
    quad_bounds = [[-10, 10]]*len(indices) if len(indices)>0 else []

    indices = []  # 选择第x、x、x个quads
    cavity_pvlist = [cavity_all[i] for i in indices]
    cavity_bounds = [[-10, 10]]*len(indices) if len(indices)>0 else []

    indices = []  # 选择第x、x、x个quads
    bend_pvlist = [bend_all[i] for i in indices]
    bend_bounds = [[-10, 10]]*len(indices) if len(indices)>0 else []

    # 合并所有PV列表和边界列表
    knob_pvlist = cor_x_pvlist + cor_y_pvlist + quad_pvlist + cavity_pvlist + bend_pvlist
    knob_bounds = cor_x_bounds + cor_y_bounds + quad_bounds + cavity_bounds + bend_bounds  

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



if __name__ == "__main__":
    print("=== machine config ===")
    print(machine_para())
    print("=== optimizer config ===")
    print(optimizer_para())
    print("=== run config ===")
    print(run_para())
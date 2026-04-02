"""决策变量、目标函数、优化器与运行参数配置（最小补丁版）"""

# ----------------------------------------------------------------------
# 目标函数 setup
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
# ----------------------------------------------------------------------
def knob_para():
    cor_x_all = [
        'IRFEL:PS:HC01:current:ao',
        'IRFEL:PS:HC02:current:ao',
        'IRFEL:PS:HC03:current:ao',
        'IRFEL:PS:HC04:current:ao',
        'IRFEL:PS:HC05:current:ao',
        'IRFEL:PS:HC06:current:ao',
        'IRFEL:PS:HC07:current:ao',
        'IRFEL:PS:HIC01:current:ao',
        'IRFEL:PS:HIC02:current:ao',
        'IRFEL:PS:MS:HC:current:ao'
    ]

    cor_y_all = [
        'IRFEL:PS:VC01:current:ao',
        'IRFEL:PS:VC02:current:ao',
        'IRFEL:PS:VC03:current:ao',
        'IRFEL:PS:VC04:current:ao',
        'IRFEL:PS:VC05:current:ao',
        'IRFEL:PS:VC06:current:ao',
        'IRFEL:PS:VC07:current:ao',
        'IRFEL:PS:VIC01:current:ao',
        'IRFEL:PS:VIC02:current:ao',
        'IRFEL:PS:MS:VC:current:ao'
    ]
    quad_all = [
        'IRFEL:PS:QM01:current:ao',
        'IRFEL:PS:QM02:current:ao',
        'IRFEL:PS:QM03:current:ao',
        'IRFEL:PS:QM04:current:ao',
        'IRFEL:PS:QM05:current:ao',
        'IRFEL:PS:QM06:current:ao',
        'IRFEL:PS:QM07:current:ao',
        'IRFEL:PS:QM08:current:ao',
        'IRFEL:PS:QM09:current:ao',
        'IRFEL:PS:QM10:current:ao',
        'IRFEL:PS:QM11:current:ao',
        'IRFEL:PS:QM12:current:ao',
        'IRFEL:PS:QM13:current:ao',
        'IRFEL:PS:QM14:current:ao',
        'IRFEL:PS:QM15:current:ao',
        'IRFEL:PS:QM16:current:ao',
        'IRFEL:PS:QM17:current:ao',
        'IRFEL:PS:QM18:current:ao',
        'IRFEL:PS:QM19:current:ao',
        'IRFEL:PS:QM19:current:ao'
    ]

    cavity_all = [
        "IRFEL:IN-MW:SHB:SET_PHASE",
        "IRFEL:IN-MW:KLY1:SET_PHASE",
        "IRFEL:IN-MW:KLY2:SET_PHASE"
    ]

    sol_all = [
        "IRFEL:PS:MS01:current:ao",
        "IRFEL:PS:LS01:current:ao",
        "IRFEL:PS:SS01:current:ao",
        "IRFEL:PS:SS02:current:ao"
    ]

    bend_all = [
        "IRFEL:PS:BM04:current:ao", #DM8 9
        "IRFEL:PS:RBM:current:ao", #RB
    ]

    # 选择的knobs
    indices = [0,1]  # 选择第x、x、x个correctors
    cor_x_pvlist = [cor_x_all[i] for i in indices]
    cor_y_pvlist = [cor_y_all[i] for i in indices]
    cor_x_bounds = [[-8, 8]]*len(indices)
    cor_y_bounds = [[-8, 8]]*len(indices)

    indices = []  # 选择第x、x、x个quads
    quad_pvlist = [quad_all[i] for i in indices]
    quad_bounds = [[-8, 8]]*len(indices)

    indices = []  # 选择第x、x、x个cav
    cav_pvlist = [cavity_all[i] for i in indices]
    cav_bounds = [[-5, 5],[-20,20]] if len(indices) != 0 else []

    indices = []  # 选择第x、x、x个sol
    sol_pvlist = [sol_all[i] for i in indices]
    sol_bounds = [[-5, 5]]*len(indices)

    indices = []  # 选择第x、x、x个bend
    bend_pvlist = [bend_all[i] for i in indices]
    bend_bounds = [[-5, 5]]*len(indices)    

    # 合并所有PV列表和边界列表
    knob_pvlist = cor_x_pvlist + cor_y_pvlist + quad_pvlist + cav_pvlist + sol_pvlist + bend_pvlist
    knob_bounds = cor_x_bounds + cor_y_bounds + quad_bounds + cav_bounds + sol_bounds + bend_bounds 

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



if __name__ == "__main__":
    print("=== machine config ===")
    print(machine_para())

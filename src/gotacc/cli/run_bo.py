import time
import numpy as np
from datetime import datetime

from GOTAcc.src.gotacc.algorithms.single_objective.bo import BOOptimizer
from gotacc.interfaces.epics_new import Obj_EpicsIoc
from GOTAcc.src.gotacc.configs.para_irfel import machine_para


# main script to run Bayesian Optimization with EPICS IOC
if __name__ == "__main__":

    t0 = time.time()
    try:  
        # ---获得目标函数与边界---
        # 1. 获取knob和obj和setup的参数
        eval_para = machine_para()
        # knobs_pvnames, knobs_bounds = knob_para()
        # obj_pvnames, obj_weights, obj_samples, obj_math, interval = obj_para()
        print('=== Evaluation Para ===')
        print(eval_para)


        # 2. 建立与目标函数的通道
        objhub = Obj_EpicsIoc(
                        knobs_pvnames=eval_para["knobs_pvnames"],
                        obj_pvnames=eval_para["obj_pvnames"],
                        obj_weights=eval_para["obj_weights"],
                        obj_samples=eval_para["obj_samples"],
                        obj_math=eval_para["obj_math"],
                        interval=eval_para["interval"],)
        
        # 3. 得到绝对边界且转化为符合优化器要求的边界格式(字典)
        ini_values = objhub.init_knob_value()
        vrange = np.array([[ini_values[i] + eval_para["knobs_bounds"][i][0], ini_values[i] + eval_para["knobs_bounds"][i][1]] for i in range(len(ini_values))])
        print("=== Optimization domain ===")
        print(vrange)
        
        #---执行优化---
        # 1. 创建优化对象
        # maximize
        opt = BOOptimizer(
            func=objhub.evaluate_func,
            bounds=vrange,
            kernel_type="rbfwhite",# "rbf", "matern", "rbfwhite", "maternwhite"
            gp_restarts=5,
            acq="ucb",
            acq_para=2.0,
            acq_para_kwargs={"beta_strategy": "inv_decay", "beta_lam": 0.02}, # "exp_decay" "inv_decay" "stage" "fixed"
            acq_optimizer="optimize_acqf", # ['random', 'sobol', 'optimize_acqf']  optimize_acqf为botorch自带的多起点优化器(默认基于L-BFGS-B)
            acq_opt_kwargs={"num_restarts": 8, "raw_samples": 256, "n_candidates": 8192}, # only for 'random' and 'sobol': 'n_candidates'
            n_init=20,
            n_iter=100,
            random_state=120
        )

        # 2. 运行优化
        opt.optimize()
        print(f'time: {time.time() - t0:.2f} s')

        
        # 3. post-process
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        path = f"save/BO_{timestamp}.dat"
        opt.save_history(path) # 保存评估历史
        # objhub.set_init()
        # objhub.set_best()# 设置最优结果到EPICS
        opt.plot_convergence()# 绘制收敛曲线
        

    
    except Exception as e:
        print(f"错误: {e}")

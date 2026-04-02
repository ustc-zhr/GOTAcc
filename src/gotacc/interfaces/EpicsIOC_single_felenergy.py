from epics import caget, PV, caput_many, caget_many
import time
import numpy as np

import para_setup # 用于获取变量的pv名和变化范围


class Obj_EpicsIoc:
    def __init__(self, knobs_pvnames=None,
                 obj_pvnames=None, obj_weights=None, obj_samples=None, obj_math=None, 
                 interval=None):
        # EPICS相关
        self.knobs_pvnames =  knobs_pvnames

        self.obj_pvnames = obj_pvnames
        self.obj_weights = obj_weights
        self.obj_samples = obj_samples
        self.obj_math = obj_math

        self.interval = interval

        self.data = []# 用于记录所有评估目标函数的数据

    def _record_evaluate(self, x, obj_val):
        """记录优化过程中的数据"""
        self.data.append(np.concatenate((x, [obj_val])))

        # 将数据保存到文件
        np.savetxt('template.opt',
                  np.array(self.data),
                  fmt='%.6f')

    def init_knob_value(self):
        
        knobs_pvvalue = caget_many(self.knobs_pvnames)
        if None in knobs_pvvalue:
                raise RuntimeError("knob初始值pv读取失败")
        
        return knobs_pvvalue

    def evaluate_func(self, x):
        """评估目标函数"""
        print('Evaluating with parameters:', x)
        
        # pv输入参数
        x = np.asarray(x).flatten()  # 兼容列表/数组输入
        caput_many(self.knobs_pvnames, x)
        time.sleep(self.interval)
        
        # 多次采样获取目标函数值
        total = np.zeros((self.obj_samples, len(self.obj_pvnames)))
        for i in range(self.obj_samples):
            total[i, :] = caget_many(self.obj_pvnames)
            time.sleep(self.interval*0.2)
        
        # 对多次采样的目标进行数学处理：平均或标准差
        results = []

        # 需要根据irfel辐射实际情况设置这两个阈值
        first_col_large_threshold = 1e6     # “非常大”的阈值，需按实际改
        first_col_change_threshold = 1e-6   # “几乎没有变化”的阈值

        for j, (col, op) in enumerate(zip(total.T, self.obj_math)):
            # 只对第一列做特殊处理
            if j == 0:
                # 条件1：第一列数值非常大
                too_large = np.mean(np.abs(col)) > first_col_large_threshold

                # 条件2：第一列几乎没有变化
                # 可用极差判断，也可用 std 判断
                almost_no_change = (np.max(col) - np.min(col)) < first_col_change_threshold

                if too_large or almost_no_change:
                    results.append(0.0)
                    continue

            # 正常计算
            if op == 'mean':
                results.append(np.mean(col))
            elif op == 'std':
                results.append(np.std(col))
            else:
                raise ValueError(f"Unsupported obj_math operation: {op}")
        # for col, op in zip(total.T, self.obj_math):
        #     if op == 'mean':
        #         results.append(np.mean(col))
        #     elif op == 'std':
        #         results.append(np.std(col))
        
        results = np.asarray(results, dtype=float)
        # 加上权重
        obj_val = float(np.dot(results, self.obj_weights))
        print('results =', results)
        print('Target is:', obj_val)
        # 记录
        self._record_evaluate(x, obj_val)
        
        return obj_val
    
    def set_best(self):
        """将最优结果设置到EPICS"""
        data_array = np.array(self.data)  # 转换为NumPy数组
        best_idx = np.argmax(data_array[:, -1])
        best_x = data_array[best_idx, :-1]
        caput_many(self.knobs_pvnames, best_x)
        print(f"Best parameters set to EPICS: {best_x}")
    
    def set_init(self):
        if self.init_knob is None:
            raise RuntimeError("init_knob is None. Call init_knob_value() first.")
        caput_many(self.knobs_pvnames, self.init_knob)
        print("Set knob values to initial values")


if __name__ == "__main__":

    # 获取knob和obj的参数
    knobs_pvnames, knobs_bounds = para_setup.knob_para()
    obj_pvnames, obj_weights, obj_samples, obj_math, interval = para_setup.obj_para()

    # 建立与目标函数的通道
    objhub = Obj_EpicsIoc(knobs_pvnames=knobs_pvnames,
                          obj_pvnames=obj_pvnames, obj_weights=obj_weights, obj_samples=obj_samples, obj_math=obj_math, 
                          interval=interval)
    
    # 计算绝对边界且转化为符合优化器要求的边界格式(字典)
    ini_values = objhub.init_knob_value()
    # ini_values = [1,2,3,4,5]  # test values
    vrange = np.array([[ini_values[i] + knobs_bounds[i][0], ini_values[i] + knobs_bounds[i][1]] for i in range(len(ini_values))])
    # bounds =  {f"x{i+1}": tuple(row) for i, row in enumerate(vrange)}


    
    
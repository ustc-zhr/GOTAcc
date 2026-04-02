from epics import caget, caput, PV, caput_many, caget_many
import time
import numpy as np


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

    def _record_evaluate(self, x, obj_vals):
        """记录优化过程中的数据"""
        self.data.append(np.concatenate((x, obj_vals)))

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
        caput('IRFEL:PS:QM15:current:ao',x[11])
        caput('IRFEL:PS:QM16:current:ao',x[10])
        time.sleep(self.interval)
        
        # 多次采样获取目标函数值
        total = np.zeros((self.obj_samples, len(self.obj_pvnames)))
        for i in range(self.obj_samples):
            values = caget_many(self.obj_pvnames)
            # 检查是否有None值
            if any(v is None for v in values):
                raise RuntimeError(f"第{i+1}次采样读取PV失败: {self.obj_pvnames}")
            total[i, :] = values
            time.sleep(self.interval)
        
        # 对多次采样的目标进行数学处理：平均或标准差
        results = []
        for col, op in zip(total.T, self.obj_math):
            if op == 'mean':
                results.append(np.mean(col))
            elif op == 'std':
                results.append(np.std(col))
            else:
                raise ValueError(f"不支持的操作: {op}")
        
        # 多目标优化，返回各目标值（可选加权）
        # 如果权重用于缩放，进行元素级乘法
        obj_vals = np.array(results) * np.asarray(self.obj_weights)
        if obj_vals[1] == 0:
            obj_vals = obj_vals + 100
        print('Target is:', obj_vals)
        # 记录
        self._record_evaluate(x, obj_vals)
        
        return obj_vals
    
    def set_best(self, maximize=False):
        """将最优结果设置到EPICS
        
        Parameters
        ----------
        maximize : bool
            是否最大化目标（True）还是最小化（False）。默认False（最小化）。
        """
        if len(self.data) == 0:
            raise RuntimeError("没有评估数据")
        
        data_array = np.array(self.data)  # 转换为NumPy数组
        n_knobs = len(self.knobs_pvnames)
        n_obj = len(self.obj_pvnames)
        
        # 验证数据形状
        if data_array.shape[1] != n_knobs + n_obj:
            raise RuntimeError(f"数据形状不匹配: 期望{n_knobs + n_obj}列，实际{data_array.shape[1]}")
        
        # 提取目标值
        obj_vals = data_array[:, n_knobs:]
        
        weighted_sums = np.sum(obj_vals, axis=1)
        
        # 根据优化方向选择最佳索引
        if maximize:
            best_idx = np.argmax(weighted_sums)
        else:
            best_idx = np.argmin(weighted_sums)
        
        best_x = data_array[best_idx, :n_knobs]
        caput_many(self.knobs_pvnames, best_x)
        caput('IRFEL:PS:QM15:current:ao',best_x[11])
        caput('IRFEL:PS:QM16:current:ao',best_x[10])
        print(f"Best parameters set to EPICS: {best_x}")
        print(f"对应的目标值: {obj_vals[best_idx]}")
        print(f"加权和: {weighted_sums[best_idx]}")

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


    
    
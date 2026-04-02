from epics import caput_many, caget_many
import os
import time
import numpy as np


class Obj_EpicsIoc:
    """
    1. 保留现有接口与主要行为；
    2. 增加 log_path；
    3. 增加 readback_check；
    4. 增加恢复初始值 restore_initial()；
    5. 去掉类本体对 para_setup 的硬依赖。
    """

    def __init__(self,
                 knobs_pvnames=None,
                 obj_pvnames=None,
                 obj_weights=None,
                 obj_samples=None,
                 obj_math=None,
                 interval=None,
                 log_path="template.opt",
                 readback_check=False,
                 readback_tol=None):

        self.knobs_pvnames = list(knobs_pvnames) if knobs_pvnames is not None else []
        self.obj_pvnames = list(obj_pvnames) if obj_pvnames is not None else []
        self.obj_weights = np.asarray(obj_weights if obj_weights is not None else [], dtype=float)
        self.obj_samples = int(obj_samples) if obj_samples is not None else 1
        self.obj_math = list(obj_math) if obj_math is not None else []
        self.interval = float(interval) if interval is not None else 0.0

        self.log_path = log_path
        self.readback_check = bool(readback_check)
        self.readback_tol = readback_tol

        self.data = []                  # 每个元素是 [x..., obj_val]
        self.initial_knob_values = None # 用于异常恢复

        self._validate_config()

    def _validate_config(self):
        if len(self.knobs_pvnames) == 0:
            raise ValueError("knobs_pvnames 不能为空")
        if len(self.obj_pvnames) == 0:
            raise ValueError("obj_pvnames 不能为空")
        if len(self.obj_weights) != len(self.obj_pvnames):
            raise ValueError("obj_weights 长度必须与 obj_pvnames 一致")
        if len(self.obj_math) != len(self.obj_pvnames):
            raise ValueError("obj_math 长度必须与 obj_pvnames 一致")
        if self.obj_samples < 1:
            raise ValueError("obj_samples 必须 >= 1")

    def _ensure_parent_dir(self, path: str):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    def _record_evaluate(self, x, obj_val):
        """记录优化过程中的数据"""
        row = np.concatenate((np.asarray(x, dtype=float), [float(obj_val)]))
        self.data.append(row)

        self._ensure_parent_dir(self.log_path)
        np.savetxt(self.log_path, np.array(self.data), fmt='%.6f')

    def _read_many_checked(self, pvnames, what="PV"):
        values = caget_many(pvnames)
        if values is None:
            raise RuntimeError(f"{what} 读取失败：返回 None")
        if any(v is None for v in values):
            raise RuntimeError(f"{what} 读取失败：存在 None")
        return np.asarray(values, dtype=float)

    def _check_readback(self, target_x):
        rb = self._read_many_checked(self.knobs_pvnames, what="knob readback")
        target_x = np.asarray(target_x, dtype=float)

        if self.readback_tol is None:
            tol = np.zeros_like(target_x)
        else:
            tol = np.asarray(self.readback_tol, dtype=float)
            if tol.ndim == 0:
                tol = np.full_like(target_x, float(tol))
            elif tol.shape != target_x.shape:
                raise ValueError("readback_tol 若为数组，其长度必须与 knobs 数量一致")

        diff = np.abs(rb - target_x)
        ok = np.all(diff <= tol)
        if not ok:
            raise RuntimeError(
                f"readback 检查失败\n"
                f"target = {target_x}\n"
                f"readback = {rb}\n"
                f"abs diff = {diff}\n"
                f"tol = {tol}"
            )

    def _reduce_objectives(self, total):
        """
        total shape = (obj_samples, n_obj)
        obj_math 中每个元素对应一个目标量的统计方式
        """
        results = []
        for col, op in zip(total.T, self.obj_math):
            if op == 'mean':
                results.append(np.mean(col))
            elif op == 'std':
                results.append(np.std(col))
            else:
                raise ValueError(f"不支持的 obj_math: {op}")
        return np.asarray(results, dtype=float)

    def init_knob_value(self):
        knobs_pvvalue = self._read_many_checked(self.knobs_pvnames, what="knob 初始值")
        if self.initial_knob_values is None:
            self.initial_knob_values = knobs_pvvalue.copy()
        return knobs_pvvalue

    def restore_initial(self):
        """恢复初始化时的 knob 值"""
        if self.initial_knob_values is None:
            raise RuntimeError("尚未保存初始值，请先调用 init_knob_value()")
        caput_many(self.knobs_pvnames, self.initial_knob_values.tolist())
        print(f"Restored initial knob values: {self.initial_knob_values}")

    def evaluate_func(self, x):
        """评估目标函数：写 knobs -> 等待 -> 多次采样 -> 加权成单目标"""
        x = np.asarray(x, dtype=float).flatten()

        if len(x) != len(self.knobs_pvnames):
            raise ValueError(
                f"输入参数维度不匹配：len(x)={len(x)}, len(knobs_pvnames)={len(self.knobs_pvnames)}"
            )

        print("Evaluating with parameters:", x)

        # 1) 写入 knobs
        caput_many(self.knobs_pvnames, x.tolist())
        time.sleep(self.interval)

        # 2) 可选：检查 readback
        if self.readback_check:
            self._check_readback(x)

        # 3) 多次采样目标
        total = np.zeros((self.obj_samples, len(self.obj_pvnames)), dtype=float)
        for i in range(self.obj_samples):
            vals = self._read_many_checked(self.obj_pvnames, what="objective PV")
            total[i, :] = vals
            if i < self.obj_samples - 1:
                time.sleep(self.interval)

        # 4) 统计 + 加权
        results = self._reduce_objectives(total)
        obj_val = float(np.dot(results, self.obj_weights))

        print("Raw objective stats:", results)
        print("Weighted target:", obj_val)

        # 5) 记录
        self._record_evaluate(x, obj_val)

        return obj_val

    def get_best(self):
        """返回当前记录中的最优解"""
        if len(self.data) == 0:
            raise RuntimeError("当前没有评估数据，无法获取最优解")
        data_array = np.asarray(self.data, dtype=float)
        best_idx = int(np.argmax(data_array[:, -1]))
        best_x = data_array[best_idx, :-1]
        best_y = float(data_array[best_idx, -1])
        return best_x, best_y

    def set_best(self):
        """将当前记录中的最优结果设置到 EPICS"""
        best_x, best_y = self.get_best()
        caput_many(self.knobs_pvnames, best_x.tolist())
        print(f"Best parameters set to EPICS: {best_x}, best target = {best_y}")


if __name__ == "__main__":
    import para_setup

    machine_cfg = para_setup.machine_para()

    objhub = Obj_EpicsIoc(
        knobs_pvnames=machine_cfg["knobs_pvnames"],
        obj_pvnames=machine_cfg["obj_pvnames"],
        obj_weights=machine_cfg["obj_weights"],
        obj_samples=machine_cfg["obj_samples"],
        obj_math=machine_cfg["obj_math"],
        interval=machine_cfg["interval"],
        log_path="template.opt",
        readback_check=False,
        readback_tol=1e-6,
    )

    ini_values = objhub.init_knob_value()
    knobs_bounds = np.asarray(machine_cfg["knobs_bounds"], dtype=float)
    vrange = np.column_stack([
        ini_values + knobs_bounds[:, 0],
        ini_values + knobs_bounds[:, 1]
    ])
    print("Absolute bounds:")
    print(vrange)
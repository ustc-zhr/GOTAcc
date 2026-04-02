from epics import caput, caput_many, caget_many
import os
import time
import numpy as np


class MultiObjEpicsIOC:
    """
    多目标 EPICS evaluator

    功能：
    1. 写入 knob PV；
    2. 支持额外联动 PV 写入（例如 QM15/QM16 跟随某两个 quad）；
    3. 多次采样目标 PV；
    4. 对每个目标执行 mean/std 等统计；
    5. 返回多目标向量；
    6. 记录所有评估历史；
    7. 支持从历史中选“最佳点”并回写到机器。
    """

    def __init__(
        self,
        knobs_pvnames=None,
        obj_pvnames=None,
        obj_weights=None,
        obj_samples=1,
        obj_math=None,
        interval=0.0,
        log_path="template_multi.opt",
        linked_pvs=None,
        zero_guard_index=None,
        zero_guard_offset=100.0,
    ):
        self.knobs_pvnames = list(knobs_pvnames) if knobs_pvnames is not None else []
        self.obj_pvnames = list(obj_pvnames) if obj_pvnames is not None else []
        self.obj_weights = np.asarray(obj_weights if obj_weights is not None else [], dtype=float)
        self.obj_samples = int(obj_samples)
        self.obj_math = list(obj_math) if obj_math is not None else []
        self.interval = float(interval)
        self.log_path = log_path

        # 额外联动写入：
        # 例如 [("IRFEL:PS:QM15:current:ao", 11), ("IRFEL:PS:QM16:current:ao", 10)]
        # 表示把 x[11], x[10] 额外写到这两个 PV
        self.linked_pvs = list(linked_pvs) if linked_pvs is not None else []

        # 某个目标出现 0 时，可整体加一个 offset，兼容你当前代码里的特殊处理
        self.zero_guard_index = zero_guard_index
        self.zero_guard_offset = float(zero_guard_offset)

        self.data = []
        self.initial_knob_values = None

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

    def _ensure_parent_dir(self, path):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    def _record_evaluate(self, x, obj_vals):
        row = np.concatenate([np.asarray(x, dtype=float), np.asarray(obj_vals, dtype=float)])
        self.data.append(row)

        self._ensure_parent_dir(self.log_path)
        np.savetxt(self.log_path, np.asarray(self.data, dtype=float), fmt="%.6f")

    def _read_many_checked(self, pvnames, what="PV"):
        values = caget_many(pvnames)
        if values is None or any(v is None for v in values):
            raise RuntimeError(f"{what} 读取失败: {pvnames}")
        return np.asarray(values, dtype=float)

    def _write_knobs(self, x):
        x = np.asarray(x, dtype=float).flatten()

        if len(x) != len(self.knobs_pvnames):
            raise ValueError(
                f"输入参数维度不匹配: len(x)={len(x)}, len(knobs_pvnames)={len(self.knobs_pvnames)}"
            )

        caput_many(self.knobs_pvnames, x.tolist())

        # 额外联动 PV
        for pvname, idx in self.linked_pvs:
            caput(pvname, float(x[idx]))

    def init_knob_value(self):
        knobs_pvvalue = self._read_many_checked(self.knobs_pvnames, what="knob 初始值")
        if self.initial_knob_values is None:
            self.initial_knob_values = knobs_pvvalue.copy()
        return knobs_pvvalue

    def restore_initial(self):
        if self.initial_knob_values is None:
            raise RuntimeError("尚未保存初始值，请先调用 init_knob_value()")
        caput_many(self.knobs_pvnames, self.initial_knob_values.tolist())
        for pvname, idx in self.linked_pvs:
            caput(pvname, float(self.initial_knob_values[idx]))
        print(f"Restored initial knobs: {self.initial_knob_values}")

    def _reduce_objectives(self, total):
        results = []
        for col, op in zip(total.T, self.obj_math):
            if op == "mean":
                results.append(np.mean(col))
            elif op == "std":
                results.append(np.std(col))
            else:
                raise ValueError(f"不支持的操作: {op}")
        return np.asarray(results, dtype=float)

    def evaluate_func(self, x):
        """
        返回 shape = (n_objectives,) 的多目标向量
        """
        x = np.asarray(x, dtype=float).flatten()
        print("Evaluating with parameters:", x)

        # 1. 写入 knobs
        self._write_knobs(x)
        time.sleep(self.interval)

        # 2. 多次采样目标
        total = np.zeros((self.obj_samples, len(self.obj_pvnames)), dtype=float)
        for i in range(self.obj_samples):
            values = self._read_many_checked(self.obj_pvnames, what=f"第{i+1}次目标采样")
            total[i, :] = values
            if i < self.obj_samples - 1:
                time.sleep(self.interval)

        # 3. 统计处理
        results = self._reduce_objectives(total)

        # 4. 目标缩放/符号处理
        obj_vals = results * self.obj_weights

        # 兼容你当前上传代码里的特殊规则
        if self.zero_guard_index is not None:
            if obj_vals[self.zero_guard_index] == 0:
                obj_vals = obj_vals + self.zero_guard_offset

        print("Raw reduced objectives:", results)
        print("Returned objective vector:", obj_vals)

        # 5. 记录
        self._record_evaluate(x, obj_vals)

        return obj_vals

    def set_best(self, maximize=False, selector="sum", selector_weights=None):
        """
        从历史中选一个“最佳点”并回写机器。

        Parameters
        ----------
        maximize : bool
            True 表示最大化 score；False 表示最小化 score
        selector : str
            当前支持:
            - "sum": 对返回的 obj_vals 直接求和
            - "weighted_sum": 用 selector_weights 做加权和
        selector_weights : array-like or None
            当 selector="weighted_sum" 时使用
        """
        if len(self.data) == 0:
            raise RuntimeError("没有评估数据")

        data_array = np.asarray(self.data, dtype=float)
        n_knobs = len(self.knobs_pvnames)
        n_obj = len(self.obj_pvnames)

        if data_array.shape[1] != n_knobs + n_obj:
            raise RuntimeError(
                f"数据形状不匹配: 期望 {n_knobs + n_obj} 列，实际 {data_array.shape[1]}"
            )

        obj_vals = data_array[:, n_knobs:]

        if selector == "sum":
            scores = np.sum(obj_vals, axis=1)
        elif selector == "weighted_sum":
            if selector_weights is None:
                raise ValueError("selector='weighted_sum' 时必须提供 selector_weights")
            selector_weights = np.asarray(selector_weights, dtype=float)
            if len(selector_weights) != n_obj:
                raise ValueError("selector_weights 长度必须与目标数一致")
            scores = obj_vals @ selector_weights
        else:
            raise ValueError(f"不支持的 selector: {selector}")

        best_idx = np.argmax(scores) if maximize else np.argmin(scores)
        best_x = data_array[best_idx, :n_knobs]

        self._write_knobs(best_x)

        print(f"Best parameters set to EPICS: {best_x}")
        print(f"对应目标值: {obj_vals[best_idx]}")
        print(f"selector score: {scores[best_idx]}")
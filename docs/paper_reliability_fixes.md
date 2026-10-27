# GOTAcc 论文支撑性可靠性修复记录

基线：`1fd09732f8ee72c2fcb27d2e98d4f3b7889b9b31`
实施分支：`codex/paper-reliability-fixes`
日期：2026-10-27

> 当前 WSL 副本中没有用户所述的 Windows 未提交测试/报告。本轮从指定基线重新建立
> 了对应测试、快照和报告；没有修改或合并现有 `main`。

## 状态摘要

| 项目 | 已实现 | 模拟测试 | 真实装置 |
|---|---:|---:|---:|
| 在线 Best 默认改为显式选择 | 是 | 通过 | 未验证 |
| `run_task()` 两层配置校验 | 是 | 通过 | 不适用 |
| runner 初始化异常关闭 backend | 是 | 通过 | 未验证 |
| GUI 零行为变化快照 | 无 GUI 生产改动 | 通过 | 未验证 |

“模拟测试通过”仅表示 fake backend/FakePV 和离屏 GUI 的可重复结果，不表示在线机器
安全性已经得到验证。

## 逐项生产 diff

### F1：在线 Best 回写改为 opt-in

文件：`src/gotacc/configs/schema.py`

```diff
-    set_best: bool = True
+    set_best: bool = False
```

影响：省略 `runtime.set_best` 的 CLI/Python task 不再在优化完成后自动写 Best。显式
`RuntimeConfig(set_best=True)`、YAML/Python 配置中的 `set_best: true`，以及 CLI
`--set-best` 仍保持原功能。正式 HALF/IRFEL 配置都显式声明该字段；GUI 编译路径也
显式写 `False`。

### F2：`run_task()` 不再绕过 loader 校验

文件：`src/gotacc/runners/task_runner.py`

```diff
+    task_cfg = validate_task_config(task_cfg)
+    task_cfg = validate_task_config_strict(task_cfg)
     validate_optimizer_backend_match(task_cfg)
```

影响：基础结构错误和严格语义错误均在 `build_backend()` 前失败。直接 Python 调用与
CLI loader 具有相同的校验层级。过去依赖无效/非规范配置延迟失败的调用会更早报错。

### F3：初始化异常释放 backend

文件：`src/gotacc/runners/task_runner.py`

新增 `_run_initialization_step()`，覆盖初值读取、边界解析和 optimizer 构造；backend
接口缺失也先关闭再抛错。优化阶段原有 `KeyboardInterrupt`/`Exception` Restore 逻辑
没有改变。初始化尚未进入优化阶段，因此失败时只 close，不调用 Restore。

## 测试与证据 diff

- `tests/test_cli_runner_defaults.py`
  - 无参数、`--set-best` 和 `--no-set-best` 三种 CLI 选择均按预期生效。
- `tests/test_paper_validation_experiments.py`
  - 默认不回写；
  - 显式回写仍调用 backend `set_best()`；
  - 基础/严格配置错误在 backend 构造前失败；
  - 初值读取和 optimizer 构造异常均调用一次 `close()`，且不调用 Restore/Best。
- `tests/test_gui_profile_compatibility.py`
  - 构造 HALF 与 IRFEL 合成 GUI 在线任务；
  - 在 `build_backend()` 前强制 monkeypatch `_load_epics`；
  - 比较任务编译、策略、边界、optimizer 参数、runtime 和 FakePV 行为。
- `tests/fixtures/gui_profile_baseline.json`
  - 冻结生产修改前生成的 HALF/IRFEL TaskConfig 与 FakePV 读写/恢复序列。

## 验证结果

WSL 环境：Python 3.11.5，pytest 9.0.3，Qt offscreen。

| 阶段 | 结果 |
|---|---|
| 指定基线完整测试 | `158 passed in 37.53s` |
| 新测试在修复前 | runner 用例 `5 failed, 1 passed`；GUI/FakePV 快照 `2 passed` |
| 定向修复后 | `11 passed in 0.08s` |
| 完整修复后 | `169 passed in 34.04s` |

静态审查和测试确认所有 EPICS 测试入口使用 monkeypatch/fake；本轮没有连接 IOC，也没有
执行真实 `caput`。

## GUI 兼容性影响

- GUI `EngineWorker` 不调用 `run_task()`。
- GUI `TaskService` 在修改前后均显式生成 `runtime.set_best=False`。
- HALF 快照的 weighted-sum、绝对边界、BO 参数、PV 读写和恢复序列不变。
- IRFEL 快照的 `fel_energy_guard`、delta-link、独立变量边界、BO 参数、readback 和
  恢复序列不变。
- GUI 生产文件 diff 为零。

## 剩余风险

1. 未进行真实 HALF/IRFEL 装置验证，不能声明在线安全已由本轮证明。
2. Windows 上 `test_secondary_dialog_buttons_use_compact_height` 曾在 Qt 离屏事件处理中
   出现 access violation；WSL 中相关选择器两项和完整套件均通过，但 Windows 原问题
   未复测、未关闭。
3. `RuntimeConfig()` 省略 `set_best` 的第三方 Python 调用行为有意改变；需要自动回写
   的调用方必须显式 opt-in。
4. 初始化清理以 fake backend 验证；持有真实外部资源的自定义 backend 仍应在受控环境
   验证其 `close()` 实现。

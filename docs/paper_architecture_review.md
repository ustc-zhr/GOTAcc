# GOTAcc 论文支撑性架构审查

审查基线：`1fd09732f8ee72c2fcb27d2e98d4f3b7889b9b31`
复核环境：WSL，Python 3.11.5，2026-10-27

> 来源说明：用户提到的 Windows 未提交草稿未出现在任何本地分支、stash、WSL
> 用户目录或已挂载 Windows 盘中。本文件是基于指定基线重新完成的审查记录，不把
> 重建内容表述为已读取到的原草稿。

## 审查边界

本轮只审查并允许修改 CLI/Python runner 的三项行为：

1. 未显式配置时是否自动把 Best 回写在线机器；
2. 直接调用 `run_task(TaskConfig)` 是否执行与 loader 相同的配置校验；
3. backend 已创建、但 runner 初始化失败时是否释放 backend。

以下生产实现不在修改范围：`EpicsObjective`、Policy Registry、sample guard、
setpoint write/readback、Restore、Best 选择算法，以及 GUI controller/worker。

## GUI 与 runner 调用链

```text
CLI
  load_task_config() -> run_task()

Python API
  TaskConfig -> run_task()

GUI
  TaskService.build_task_config()
    -> EngineWorker.run()
       -> build_backend()/runner helpers
       -> optimizer.optimize()
```

GUI `EngineWorker` 不调用 `run_task()`。GUI 只复用 `build_optimizer()`、
`resolve_bounds()`、`validate_optimizer_backend_match()` 等 helper，并且
`TaskService.build_task_config()` 显式设置 `runtime.set_best=False`。因此本轮
`run_task()` 的校验和初始化清理修改不进入 GUI 执行路径；runtime 默认值修改由
HALF/IRFEL 冻结快照确认没有改变 GUI 编译结果。

## 基线发现

| 编号 | 基线行为 | 风险 | 修复边界 |
|---|---|---|---|
| R1 | `RuntimeConfig.set_best=True` | 省略 runtime 字段的 CLI/Python 在线任务会自动产生一次额外写入 | 默认改为 `False`；显式 `set_best=True` 和 CLI `--set-best` 仍可选择回写 |
| R2 | loader 校验配置，但直接调用 `run_task()` 可绕过基础和严格校验 | 无效边界或未知优化器可能在 backend 创建后才失败 | `run_task()` 在 `build_backend()` 前执行两层校验 |
| R3 | `init_knob_value()`、边界解析、optimizer 构造均位于已有 `try/finally` 之前 | 初始化异常时 backend 的资源可能未释放 | 初始化步骤异常时调用既有 `close_backend_if_possible()`；不触发 Restore |

## 兼容性判断

- 仓库中的 HALF/IRFEL Python 和 YAML 在线任务均显式声明 `set_best`，其意图不受
  默认值调整影响。
- GUI 始终显式设置 `set_best=False`，并由逐字段快照验证。
- 直接 Python 调用若过去依赖无效配置延迟失败，现在会更早失败；这是有意的严格化。
- 初始化失败只执行资源关闭，不扩大 Restore 的触发范围。
- 没有修改任何 EPICS 写入、readback、策略、恢复或 Best 选择生产实现。

## 审查结论

三项问题都能在不进入 GUI 主执行链、也不改变共享 EPICS 控制实现的前提下修复。
真实装置行为仍需受控机时验证；本审查和自动化测试不能替代装置验收。

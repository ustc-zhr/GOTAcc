# GOTAcc 论文证据矩阵

基线：`1fd09732f8ee72c2fcb27d2e98d4f3b7889b9b31`
最后更新：2026-10-27

状态定义：

- **已实现**：生产代码或验证资产已存在于独立分支；
- **模拟通过**：仅由 fake backend、FakePV、离屏 GUI 或离线测试支持；
- **装置验证**：必须在真实 HALF/IRFEL 且经授权的受控机时完成。

| 论文/可靠性主张 | 实现证据 | 自动化证据 | 当前状态 | 尚缺证据 |
|---|---|---|---|---|
| CLI/Python 在线任务不会因省略配置而自动回写 Best | `RuntimeConfig.set_best=False` | Python runner 默认用例 + CLI 无参数用例 | 已实现；模拟通过 | 真实装置未验证 |
| 显式 Best 回写能力仍可用 | `run_task()` 原 set_best 分支未改；CLI `--set-best` 未改 | Python 显式回写用例 + CLI 双向开关用例 | 已实现；fake backend 通过 | 受控实机写入未验证 |
| direct `run_task()` 在 I/O 前完成结构校验 | `validate_task_config()` 位于 `build_backend()` 前 | invalid bounds 用例且 backend build 次数为 0 | 已实现；模拟通过 | 无装置需求 |
| direct `run_task()` 在 I/O 前完成严格语义校验 | `validate_task_config_strict()` 位于 `build_backend()` 前 | unknown optimizer 用例且 backend build 次数为 0 | 已实现；模拟通过 | 无装置需求 |
| 初始化失败释放 backend 资源 | `_run_initialization_step()` 和接口失败 close | initial read / optimizer init 失败用例 | 已实现；fake backend 通过 | 自定义/真实 backend 资源验证 |
| 优化异常 Restore 语义未改变 | 原 optimize `try/except/finally` 未移动 | 既有 `test_engine_worker.py` 与完整回归 | 代码审查；模拟通过 | 真实装置未验证 |
| GUI 不自动回写 Best | `TaskService.build_task_config()` 仍显式 `False` | HALF/IRFEL 冻结快照 + 专项断言 | 无 GUI 生产改动；模拟通过 | 真实 GUI/装置验收 |
| HALF GUI 任务编译保持一致 | 无 GUI 生产 diff | `gui_profile_baseline.json`：边界、weighted-sum、BO 参数、PV 序列 | 模拟通过 | HALF 实机未验证 |
| IRFEL GUI 策略和写入派生保持一致 | 无 EPICS/Policy 生产 diff | 快照：fel guard、delta-link、readback、restore | 模拟通过 | IRFEL 实机未验证 |
| 本轮没有真实 IOC 写入 | 测试在构造前 monkeypatch `_load_epics`/backend | 静态调用点审查；FakePV trace | 已确认测试隔离 | 不构成装置安全证明 |
| WSL 完整回归通过 | 独立分支工作树 | 基线 `158 passed`；修复后 `169 passed` | 通过 | Windows access violation 未复测 |

## 不应从当前证据推出的结论

- 不能声称真实 HALF/IRFEL 上的写入、readback、Restore 或 Best 选择已验证安全。
- 不能用 WSL Qt 通过结论关闭 Windows 离屏 access violation。
- 不能把合成 GUI profile 当作控制室当前项目文件或现场配置的替代品。
- 不能把单次完整 pytest 结果扩展为长期稳定性或统计性能结论。

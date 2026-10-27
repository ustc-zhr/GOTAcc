# GOTAcc 论文支撑性验证计划

基线提交：`1fd09732f8ee72c2fcb27d2e98d4f3b7889b9b31`

> 来源说明：原 Windows 未提交验证计划在当前 WSL/挂载盘中不可见；本计划按用户给定
> 边界和基线重新建立。

## 安全前提

- 不连接真实 IOC，不设置真实 CA 地址，不执行实机 `caput`。
- 所有 EPICS backend 测试必须在构造前 monkeypatch `_load_epics`，或替换
  `build_backend` 为 fake backend。
- GUI 使用 `QT_QPA_PLATFORM=offscreen`。
- 不修改 `EpicsObjective`、Policy Registry、sample guard、write/readback、Restore
  或 Best 选择生产实现。

## 验证分层

| 层级 | 验证内容 | 通过标准 |
|---|---|---|
| V0 基线 | 指定提交完整 pytest | 全部通过；记录平台差异 |
| V1 runner 单元 | Best 默认 opt-in、显式回写、direct config 两层校验、初始化异常 close | fake backend 断言全部通过 |
| V2 GUI 编译快照 | HALF/IRFEL 合成 GUI task 的 TaskConfig、策略、边界、optimizer 参数、runtime | 与冻结 JSON 完全一致 |
| V3 FakePV 行为 | 初值读取、评估写入、readback、目标采样、delta-link、恢复写入 | PV 名称、顺序和值与冻结 JSON 完全一致 |
| V4 完整回归 | 仓库全部 pytest（含新增测试） | 无失败、无意外跳过 |
| V5 真实装置 | 受控机时下的 read-only 预检和经授权写入 | 本轮不执行，状态必须保持“未验证” |

## 环境与命令

环境：`/home/zhanghaoran/anaconda3/envs/half_linac/bin/python`，Python 3.11.5，
pytest 9.0.3，PyQt5、torch、gpytorch、botorch、pyepics 均可导入。

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src \
  /home/zhanghaoran/anaconda3/envs/half_linac/bin/python -m pytest -ra
```

定向回归：

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src \
  /home/zhanghaoran/anaconda3/envs/half_linac/bin/python -m pytest -q \
  tests/test_cli_runner_defaults.py \
  tests/test_paper_validation_experiments.py \
  tests/test_gui_profile_compatibility.py
```

## 判停条件

出现以下任一情况即停止对应生产修改：

- GUI 编译快照变化；
- FakePV 的 PV 名称、写入顺序、边界、策略或恢复序列变化；
- 测试路径可能落到真实 pyepics I/O；
- 需要修改受保护的共享 EPICS/Policy/Restore/Best 实现；
- Windows Qt access violation 之外出现新的 GUI 回归。

## 结果记录要求

最终文档必须分别标明：代码已实现、模拟测试已通过、真实装置未验证。不得用离线或
FakePV 结果推断“在线安全已验证”。

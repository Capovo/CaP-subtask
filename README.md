# CaP-subtask

将 **τ₀ Proposal 的高层子任务输出**作为 **AgenticWAM Code Expert** 的额外条件，用于机器人操作中的程序、视频和动作联合生成。

```text
当前三相机图像 + 总任务指令 + 任务记忆
                  ↓
          冻结的 τ₀ Proposal
                  ↓ 子任务文本
      AgenticWAM Code / Video / Action Experts
                  ↓
       RoboTwin WAM 或 Hybrid 执行
```

子任务通过固定的 prompt token 输入 Code Expert，训练与推理使用同一实现。规划器记忆在每个 episode 开始时重置。

## 当前状态

- 已实现子任务条件接入、离线缓存、训练数据加载和在线评测入口。
- 已完成 11 项 CPU/HTTP 测试及配置、命令 dry-run。
- 尚未完成新模型的正式训练或完整闭环评测；仓库不提供新的成功率结果。
- 本目录复用 AgenticWAM 实现；模拟器、外部依赖、模型权重及训练数据需另外准备。

## 使用文档

- [部署、数据准备、训练和评测说明](README.txt)
- [RoboTwin Table 1 实验方案](ROBOTWIN_EXPERIMENT_PLAN.txt)
- [实验规格草案](experiments/robotwin_table1/plan.yaml)（不是可直接执行的 Hydra 配置）
- [结果表模板](experiments/robotwin_table1/results_template.csv)
- [新增与修改文件说明](CHANGES.txt)
- [已有验证记录](validation/verification.json)

低层使用 Python 3.10 / Transformers 4.x；τ₀ Proposal 使用单独的 Python 3.12 / Transformers 5.x 环境，通过 HTTP 交互。具体依赖见 [pyproject.toml](pyproject.toml) 和 [Proposal requirements](requirements/proposal/requirements.txt)。

`scripts/activate_local.sh` 等本机辅助脚本及原始记录包含开发机器路径。其他机器需要配置自己的环境、模型和数据路径；`configs/paths/local.yaml`、`configs/robotwin/site.yaml`、本地 Python 包装器、权重、数据和运行缓存不纳入 Git。

## 代码来源与许可

- [AgenticWAM](https://github.com/ZhuoyangLiu2005/agentic-wam)：保留原 [BSD-2-Clause 许可证](LICENSE)。
- [τ₀-VLA](https://github.com/sii-research/tau-0-vla)：Proposal 组件保留其 [Apache-2.0 许可证](requirements/proposal/TAU0_LICENSE)。
- 来源、版本和文件摘要见 [NOTICE.txt](NOTICE.txt) 与 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。

本仓库上传源码、配置、文档和示例；模型权重与训练示范需遵循各自发布方的许可获取。

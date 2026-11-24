CaP-subtask：τ0 高层子任务条件化的 AgenticWAM
日期：2026-11-24

已经实现的模型形式：
  τ0 Proposal(当前 head/left/right RGB, 总任务, 上一次规划记忆)
      -> 当前 subtask、next_memory
  Code Expert(当前视觉前缀, 总任务指令, helper library, τ0 subtask)
      -> 当前 CaP stage
  Video/Action experts 继续通过 MoT 注意力读取代码分支，生成未来视频与动作。

这是冻结规划器 + 条件化低层模型的组合。新增条件是文本 token，不另加未训练的投影层，
不替换总任务指令，不把 τ0 的 think/memory 当作代码标签。τ0 不接收低层损失梯度。
需要用真实 τ0 输出缓存微调低层，才能得到新的训练权重；现有 hybrid.pt 仅作为初始化。
本次未进行正式微调、未下载 τ0 Proposal 9B 权重，也未宣称已取得新模型性能。

本目录继承了运行所需的 AgenticWAM src/configs/scripts/robotwin/deploy/utils，保留原许可证。
原 /home/yilianli/project/agentic-wam 和 tau-0-vla 的源码未修改。
SOURCE_MANIFEST.json 记录复制来源及初始 SHA-256；CHANGE_MANIFEST.json、changes.patch、CHANGES.txt
记录本次功能新增及相对来源的修改。src/tau0_proposal 是单独提取的 τ0 高层 Proposal，
来自 tau-0-vla/src/tau0_vla/high_level/proposal；不包含 τ0 的低层 VLA 或 world model。
τ0 原许可证保留在 requirements/proposal/TAU0_LICENSE。

1. 环境与模型

低层沿用已安装的 agenticwam 环境，通过 PYTHONPATH 优先导入本目录的修改版本：
  source /home/yilianli/project/CaP-subtask/scripts/activate_local.sh
  python -c "import agenticwam; print(agenticwam.__file__)"
应显示 /home/yilianli/project/CaP-subtask/src/agenticwam/__init__.py。
无需重新在共享环境安装同名 agenticwam 包。

τ0 Proposal 依赖 Transformers 5.5.4，低层依赖 4.x，使用单独 Python 3.12 环境。
以下是按需执行的配置命令（本次未执行安装或下载）：
  /home/yilianli/project/tau-0-vla/.env-model/bin/python -m venv /home/yilianli/project/CaP-subtask/.venv-proposal
  .venv-proposal/bin/python -m pip install -r requirements/proposal/requirements.txt
  .venv-proposal/bin/hf download sii-research/tau-0-vla-proposal --local-dir /path/to/tau0-proposal

启动冻结规划器，默认使用 GPU 1：
  TAU0_PROPOSAL_WEIGHTS=/path/to/tau0-proposal bash scripts/serve_tau0_proposal.sh
可设置 PROPOSAL_PYTHON、PROPOSAL_DEVICE、PROPOSAL_PORT，或通过脚本后续参数添加 --adapter。
configs/subtask/planner.yaml 控制低层请求地址、超时和 full_qa/subtask_only 模式。
full_qa 的 next_memory 在每次重规划后传回 τ0；每个新回合清空为 (empty)。

离线验证 τ0 本身（在 Proposal 环境中）：
  PYTHONPATH="$PWD/src" .venv-proposal/bin/python -m tau0_proposal.infer \
    --model /path/to/tau0-proposal --input examples/proposal.jsonl --output /tmp/tau0-example.jsonl
examples/proposal/ 三张图及 tau0_proposal_output.json 是上游提供的真实接口示例，
不是本次新模型的推理/评测结果。

2. 准备训练子任务缓存

本版本默认从 click_alarmclock 开始；可用下列命令扩展到多个任务或全部 50 个任务：
  python scripts/prepare_subtask_data_config.py --tasks click_alarmclock,stack_blocks_two --overwrite
  python scripts/prepare_subtask_data_config.py --tasks all --overwrite

每个任务的数据需是现有 codegen/sim 导出的 LeRobot v2.x 数据，以及原有 stage 切片/文本缓存。
在 configs/paths/local.yaml 配置真实路径：
  robotwin_cap          含 <task>/lerobot/robotwin2.0、_text_cache/<task> 的根目录
  robotwin_cap_slices   含 <task>/raw/slices 的根目录
  robotwin_subtasks     τ0 输出缓存的根目录；默认 data_local/tau0_subtasks，可用 CAP_SUBTASK_CACHE_ROOT 覆盖
  robotwin_dataset_stats 归一化统计；目前复用发布的 dataset_stats.json

对每个任务执行以下三个步骤，例子中的 /data 请换为实际路径：
  python scripts/export_tau0_observations.py \
    --dataset /data/cap/click_alarmclock/lerobot/robotwin2.0 \
    --output data_local/observations/click_alarmclock --stride 24
  python scripts/predict_tau0_subtasks.py \
    --input data_local/observations/click_alarmclock/observations.jsonl \
    --output data_local/observations/click_alarmclock/predictions.jsonl
  python scripts/build_subtask_cache.py \
    --observations data_local/observations/click_alarmclock/observations.jsonl \
    --predictions data_local/observations/click_alarmclock/predictions.jsonl \
    --output data_local/tau0_subtasks/click_alarmclock/subtask_cache.json

export 默认读取全部 episode，也可 --episodes /path/to/episodes_train.json 指定原始 episode 索引。
每个区间只把起始帧的三相机图像送给 τ0，不向它展示未来帧、GT stage 或动作标签。
predict 按 episode 和帧顺序请求，记忆只来自此前的规划输出。
每个输出文件使用新路径；缓存构建器拒绝覆盖、ID 不匹配、无效输出、区间缺失/重叠和元数据变化。

缓存格式：
  schema_version: 1
  provenance: 观测清单/预测清单 SHA-256，以及规划器模型位置
  datasets:
    - root: 数据集绝对路径
      metadata_sha256: LeRobot info/episodes/tasks 元数据摘要
      episodes:
        "5":
          length: 原始 episode 的总帧数
          segments:
            - start: 0
              end: 24
              subtask: Pick up the cup with the right arm.
              proposal_id: 与观测/预测相同的唯一 ID

区间为 [start,end)，每个缓存 episode 必须从 0 连续覆盖到 length。
采用数据集/原始 episode/frame 键，而不是训练索引，避免选择、重排 episode 或读取重试后错位。
更换数据集路径/元数据后应重建缓存。

3. 训练新模型

  source scripts/activate_local.sh
  DRY_RUN=1 bash scripts/train_cap_subtask.sh wandb.enabled=false
  bash scripts/train_cap_subtask.sh wandb.enabled=false

默认两卡、每卡 batch_size=1、梯度累积2；可通过 NPROC 与 Hydra 参数调整。
预检查会检查所有已用路径、子任务缓存的元数据绑定和所选 episode 覆盖率。
当前训练数据和子任务缓存未提供，因此完整 preflight 会明确失败；这是待准备的输入，不是成功训练记录。
只查看启动命令、不检查数据：
  python scripts/launch_train.py --task cap_subtask_robotwin --nproc 2 --dry-run paths=local

配置：configs/model/cap_subtask.yaml，configs/robotwin/task/cap_subtask_robotwin.yaml，
      configs/robotwin/data/cap_subtask.yaml。
训练保留三路联合损失；新增条件只进入 Code Expert 的 prompt，标签仍为原来的代码/EOS。
max_code_tokens 从512扩到768，代码 canvas 保持124；subtask 单独限制为128个 SDAR token。
超长 prompt/代码会报错，避免静默截断。该版本支持 joint 目标，code_memory=false。
原 fast_joint 推理覆盖尚未接入新条件，启用它时会明确报错；请使用标准 joint 路径。
输出：runs/cap_subtask_robotwin/<run_id>/config.yaml、dataset_stats.json、checkpoints/weights/step_*.pt。
新权重保存 tau0_subtask_text_v1 条件元数据；加载时若漏用条件模型配置会拒绝。

4. 推理与评估

单次纯模型推理，不控制机器人：
  python scripts/infer_cap_subtask.py --input /path/to/request.json \
    --ckpt /path/to/new_subtask_checkpoint.pt --output /tmp/cap-subtask-output.json
输入格式见 examples/inference_request.template.json：task、instruction、三路图像路径、14维关节状态。
在线调用 τ0；也可 --proposal-json /path/to/saved_tau0_output.json 复用同一观测的有效输出，
此时输入和 Proposal 输出都必须携带一致的 id。示例中的零关节状态只是结构占位。
输出包含 high_level_subtask、code、action_chunk、action_type=qpos_abs。

RoboTwin 闭环评估，默认 WAM 动作执行模式，τ0 条件照常进入 Code Expert：
  bash scripts/eval_cap_subtask.sh --ckpt /path/to/new_subtask_checkpoint.pt \
    --tasks click_alarmclock --plan smoke --protocols clean --episodes 1 --shards 1
  同一命令添加 --dry-run，仅打印计划；不会启动模型或请求服务。
可设置 EXECUTION_MODE=hybrid 改为 CaP stage 执行 + WAM 回退。
Hybrid 还依赖原有 Molmo/SAM3/GraspNet，需另外安排 GPU/服务资源；默认动作执行模式只需要 τ0 和低层模型。
评测没有默认权重路径，必须显式提供 --ckpt，避免误把原 hybrid.pt 当作新训练权重。
policy_name 使用 cap_subtask_policy，与原 agenticwam_policy 分开。

评估输出沿用 evaluate_results/robotwin*/，并新增 episodeN_subtasks.jsonl：
每次重规划的 subtask、memory_in、next_memory、规划器模型和请求延迟。
代码生成与动作/Hybrid 仲裁日志仍保留。run_meta.json 记录 subtask_conditioning=true。
模拟器、资产、OIDN 和感知环境继续引用原 agentic-wam 的已部署依赖，未重复复制大型资源。

5. 已完成验证与范围

  python -m unittest discover -s tests -v
十一项 CPU/本地 HTTP 测试覆盖：模型工厂条件设置、检查点条件元数据与加载约束；
条件影响真实 CodeBlock logits 与反向梯度；
条件不进入 CE、joint mask 不遮条件；训练/实际 joint 推理 prompt 一致；
缺条件与超长输入拒绝；原始 episode/帧索引与读取重试；缓存错位/失效检查；
三相机真实视频当前帧导出；τ0 HTTP payload、记忆传递/重置与无效输出拒绝。
此外完成 Python 编译、shell语法、环境依赖检查、训练/评测配置 dry-run。
未运行完整9B τ0 + 大模型联合推理、未进行正式训练，也未给出新模型的任务成功率。

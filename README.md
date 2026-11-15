# AI_harness_competition

赛题 9「智训引擎」——工业制造场景下的模型训练与超参数优化：赛题资料与 CPU 参考基线包归档。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `cpu-c-guide-20261110-v1/` | 官方下发的 CPU 参考基线包（赛道 C：涡扇发动机 RUL） |
| `cpu-c-guide-20261110-v1.zip` | 上述目录的原始压缩包，内容重复，仅作存档 |
| `赛题9规则：“智训引擎”—工业制造场景下的模型训练与超参数优化(2).pdf` | 赛题规则原文 |
| `赛题.txt` | 赛道速览：C 赛道为发动机剩余寿命预测 |
| `赛题逐字稿.txt` | 赛题说明会的逐字记录 |
| `harness工程.md` | 赛题要点整理 |
| `harness/` | Harness 最小闭环：漂移检测、受控实验、SwanLab 记录、门控上线与回滚，说明见 [harness/README.md](harness/README.md) |
| `tests/` | Harness 的单元与闭环测试（合成数据） |
| `configs/harness.json` | Harness 全部阈值与预算的默认值 |
| `runs/` | 两个本地漂移场景的运行输出（决策、报告、留出评分、实验台账） |
| `Dockerfile`、`requirements.txt` | 运行镜像与依赖 |

## Harness 最小闭环

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m harness simulate --public cpu-c-guide-20261110-v1/public --out work/sim --scenario sensor-aging --seed 59
.venv/bin/python -m harness run --stream work/sim/stream --workdir work/run
.venv/bin/python -m harness score --workdir work/run --hidden work/sim/hidden
```

每一批依次做漂移检测、回滚核对、归因、受控实验和门控，交出 promote、hold 或 rollback 的决策后再取下一批。工业模型始终由官方 `baselines` 代码训练，基线包文件未改动。各步做法、输出目录、本地验证结果与已知局限见 [harness/README.md](harness/README.md)。

## 基线包 `cpu-c-guide-20261110-v1/`

- 任务：NASA C-MAPSS FD001–FD004，预测发动机剩余使用寿命（RUL，单位：循环）
- 主指标：RMSE，辅助指标 MAE，均为越低越好；平台另报告 NASA 非对称评分（高估寿命惩罚更重）
- 模型：CPU NumPy MLP，24 维输入 → 32 个 tanh 隐藏单元 → 1 个线性输出，共 833 个参数

### 关键文件

| 文件 | 说明 |
| --- | --- |
| `TASK.md` | 任务定义、标签生成方式与评测协议 |
| `TRAINING_GUIDE.md` | 从基线续训的命令与约束 |
| `START.md` | 包内导航 |
| `baselines/` | 可修改的训练／预测代码、A/B/C 配置与测试 |
| `public/train.jsonl` | 公开训练集（113,089 行），梯度更新只使用该集合 |
| `public/validation.jsonl` | 公开验证集（23,049 行），不要并入训练集 |
| `checkpoint.npz` | 预训练权重、Adam 一阶／二阶矩、步数、随机种子与标准化／缩放参数 |
| `training.json` | 原始训练报告：数据摘要、超参、验证前后指标 |
| `history.jsonl` | 逐 epoch 的训练与验证记录 |
| `baseline-info.json` | 基线 ID、文件摘要与参数规模 |

### 基线验证指标

| 指标 | 训练前 | 训练后 |
| --- | --- | --- |
| RMSE（循环） | 82.37 | 52.87 |
| MAE（循环） | 66.53 | 38.90 |

原始训练为 15 epochs，批大小 256，学习率 0.002，固定种子 42，CPU 单线程约 5 秒。

## 快速开始

```sh
python -m pip install -r cpu-c-guide-20261110-v1/baselines/requirements.txt

cd cpu-c-guide-20261110-v1
python -m baselines.train --profile C \
  --train public/train.jsonl --validation public/validation.jsonl \
  --from-baseline checkpoint.npz --epochs 5 \
  --output round-1 --stage-id round-1
```

续训时使用 `--resume round-1/checkpoint.npz`，并输出到新目录。`--from-baseline` 与 `--resume` 都会恢复权重和优化器状态，不会重新初始化。

## 数据与评测边界

- 每行 JSONL 中只有 `inputs` 可作为模型输入，`answer` 是监督目标；`id`、`meta` 仅用于追溯。
- 特征标准化与 RUL 目标缩放只在初始训练集上拟合，续训沿用检查点中的统计量；验证集与隐藏集不得参与拟合。
- 本包不包含隐藏测试数据、标签或发动机名单，隐藏数据由平台独立保管。
- 训练日志中的自报指标不能替代平台评测。

## 配置 A / B / C

`baselines/configs/` 保留了三条赛道的参考配置：

| 配置 | 任务 | 网络 |
| --- | --- | --- |
| A | 六类气体识别 | 输入特征 → 32 个 tanh 隐藏单元 → 6 类 softmax |
| B | 正常／早期退化／严重退化识别 | 输入特征 → 24 个 tanh 隐藏单元 → 3 类 softmax |
| C | RUL 回归 | 24 维输入 → 32 个 tanh 隐藏单元 → 1 个线性输出 |

本包附带的 `public/` 数据、`checkpoint.npz` 与 `training.json` 均对应 **C 赛道**。

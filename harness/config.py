"""全部阈值与预算集中在这里，可用 JSON 文件按字段覆盖。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class DriftConfig:
    psi_bins: int = 10
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    # 单个特征的告警线 = max(psi_alert, noise_floor_mult × 参考窗口内部按设备折半得到的 PSI)
    noise_floor_mult: float = 2.0
    noise_floor_reps: int = 4
    min_alert_features: int = 2      # 至少这么多特征越线才判输入漂移
    severe_share: float = 0.5        # 越线特征占比达到该值记为 severe
    perf_tol: float = 0.05           # 主指标相对锚点劣化超过该比例且显著，判性能漂移
    perf_severe: float = 0.25
    target_psi_alert: float = 0.25
    reference_rows: int = 20000      # 参考窗口保留的样本上限


@dataclass
class ExperimentConfig:
    min_rounds: int = 3              # 前若干轮即使未触发漂移也做一轮小规模实验
    max_trials: int = 3
    max_trials_severe: int = 4
    round_seconds: float = 300.0     # 单批次实验的墙钟预算
    select_fraction: float = 0.2     # 从本批训练部分按设备划出的选型集比例
    max_train_rows: int = 60000
    max_epochs: int = 40
    patience: int = 5                # 选型集主指标连续若干轮不再变好就停
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    lr_step: float = 2.5             # 学习率单因素对照的倍率
    batch_sizes: tuple = (64, 128, 256, 512)
    policy_min_gain: float = 0.005   # 变体在选型集上超过基准这么多才改写策略
    selection_slack: float = 0.02    # 选型集成绩与最优相差在此之内的试验视为并列，并列时取历史批次上更好的
    initial_policy: dict = field(default_factory=lambda: {
        "data_scope": "replay", "replay_ratio": 0.5,
        "learning_rate": 1e-3, "batch_size": 256, "epochs": 20})


@dataclass
class GateConfig:
    min_rel_gain: float = 0.03       # 候选相对在线模型的最小相对增益（低于它的增益与换一批设备带来的波动分不开）
    min_gate_groups: int = 10        # 门控集里的独立设备数少于它就不上线：设备太少时结论换一批设备就可能反转
    confidence: float = 0.95         # 配对自助法的单侧置信水平
    bootstrap: int = 1000
    static_tol: float = 0.0          # 相对官方静态基线允许的劣化
    retention_tol: float = 0.10      # 历史批次上的相对劣化超过它就在决策里标记（提示项，不拦截上线）
    retention_rows: int = 20000
    safety_tol: float = 0.02         # 回归任务：非对称误差允许的劣化
    rollback_margin: float = 0.02    # 在线模型比备选旧版本差这么多且显著就回滚
    static_floor_tol: float = 0.0    # 在线模型在新批次上比静态基线差超过该比例（点估计）就退回静态基线
    rollback_lookback: int = 3       # 回滚备选 = 静态基线 + 最近若干个上线过的版本


@dataclass
class RecordConfig:
    project: str = "zhixun-harness"
    workspace: str | None = None
    mode: str = "offline"            # offline | online | local | disabled
    public: bool = True


@dataclass
class LLMConfig:
    enabled: bool = False
    base_url: str = ""               # OpenAI 兼容接口根地址，例如 https://<pytrio-host>/v1
    api_key_env: str = "PYTRIO_API_KEY"
    model: str = "Qwen3.8-27B"
    timeout: float = 60.0
    max_tokens: int = 1200
    temperature: float = 0.2
    max_calls: int = 64              # 整条流上的调用上限
    extra_body: dict = field(default_factory=dict)


@dataclass
class Config:
    baseline_dir: str = "cpu-c-guide-20261110-v1"
    checkpoint: str | None = None    # 缺省用 baseline_dir/checkpoint.npz
    stream_dir: str = ""
    workdir: str = "work/run"
    seed: int = 42
    group_pattern: str = r"^(.*)_c\d+$"   # 从样本 id 提取设备编号；匹配不上时按连续行分块
    block_rows: int = 64
    drift: DriftConfig = field(default_factory=DriftConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _apply(target, values: dict, where: str) -> None:
    known = {f.name for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"未知配置项 {where}{key}")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"配置项 {where}{key} 应为对象")
            _apply(current, value, f"{where}{key}.")
        else:
            setattr(target, key, tuple(value) if isinstance(current, tuple) else value)


def load_config(path: str | None = None, overrides: dict | None = None) -> Config:
    config = Config()
    if path:
        _apply(config, json.loads(Path(path).read_text(encoding="utf-8")), "")
    if overrides:
        _apply(config, overrides, "")
    return config

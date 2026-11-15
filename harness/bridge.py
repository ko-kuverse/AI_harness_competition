"""接入官方基线包。工业模型始终由官方 baselines.model.Model 训练、保存和加载，本包不改动其代码。"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

SOURCES = ("model.py", "data.py", "train.py", "predict.py")


@dataclass
class Baseline:
    root: Path
    checkpoint: Path
    Model: type
    read_dataset: object
    checkpoint_digest: object
    meta: dict              # 检查点元数据：任务、类别、特征路径、种子等
    source_sha256: dict     # 官方训练代码各文件摘要，写入每次试验的报告
    anchor: dict | None     # 基线包 training.json 里的 validation_after，作为静态基线的性能锚点

    @property
    def task(self) -> str:
        return self.meta["task"]


def load_baseline(root: str | Path, checkpoint: str | Path | None = None) -> Baseline:
    root = Path(root).resolve()
    package = root / "baselines"
    if not (package / "model.py").is_file():
        raise FileNotFoundError(f"{root} 下没有官方 baselines 包")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    model_module = importlib.import_module("baselines.model")
    data_module = importlib.import_module("baselines.data")
    if Path(model_module.__file__).resolve().parent != package:
        raise RuntimeError("已加载的 baselines 包不是指定目录下的那一份")
    checkpoint = Path(checkpoint).resolve() if checkpoint else root / "checkpoint.npz"
    model = model_module.Model.load(checkpoint)
    anchor = None
    report = checkpoint.parent / "training.json"
    if report.is_file():
        loaded = json.loads(report.read_text(encoding="utf-8"))
        if loaded.get("checkpoint_sha256") == model_module.checkpoint_digest(checkpoint):
            anchor = loaded.get("validation_after")
    digests = {name: "sha256:" + hashlib.sha256((package / name).read_bytes()).hexdigest() for name in SOURCES}
    return Baseline(root, checkpoint, model_module.Model, data_module.read_dataset,
                    model_module.checkpoint_digest, dict(model.meta), digests, anchor)

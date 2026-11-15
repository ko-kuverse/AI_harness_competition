"""实验记录：每个试验、每个批次决策各记一个 SwanLab 实验，同时写一份本地 JSONL 台账。

SwanLab 不可用（未安装、初始化失败、写入出错）时只写台账，主循环不受影响。
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .registry import now


def _plain(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _scalars(metrics: dict) -> dict:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            result[key] = int(value)
        elif isinstance(value, (int, float)) and math.isfinite(value):
            result[key] = float(value)
    return result


class Run:
    def __init__(self, recorder: "Recorder", name: str, handle):
        self.recorder, self.name, self.handle = recorder, name, handle

    def log(self, metrics: dict, step: int) -> None:
        data = _scalars(metrics)
        if not data:
            return
        self.recorder.event("metrics", run=self.name, step=step, data=data)
        if self.handle is not None:
            try:
                self.handle.log(data, step=step)
            except Exception as error:  # 记录失败不能中断闭环
                self.recorder.disable(error)
                self.handle = None

    def finish(self) -> None:
        if self.handle is not None:
            try:
                self.handle.finish()
            except Exception as error:
                self.recorder.disable(error)
        self.recorder.event("run_finished", run=self.name)


class Recorder:
    def __init__(self, workdir: Path, cfg):
        self.cfg = cfg
        self.ledger = Path(workdir) / "records" / "events.jsonl"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(workdir) / "swanlog"
        self.mode = cfg.mode
        self.note = None
        self.swanlab = None
        self.runs = 0
        if self.mode != "disabled":
            try:
                import swanlab
                self.swanlab = swanlab
            except Exception as error:
                self.note = f"swanlab 无法导入：{error}"
        if self.swanlab and self.mode == "online" and not os.environ.get("SWANLAB_API_KEY"):
            # 无人值守环境没有交互登录，缺少密钥时改为离线落盘，事后用 swanlab sync 上传。
            self.mode, self.note = "offline", "未设置 SWANLAB_API_KEY，online 改为 offline"
        self.event("recorder", requested_mode=cfg.mode, effective_mode=self.status(), note=self.note)

    def status(self) -> str:
        return self.mode if self.swanlab else "ledger_only"

    def disable(self, error: Exception) -> None:
        self.note = f"swanlab 出错后停用：{type(error).__name__}: {error}"
        self.swanlab = None
        self.event("recorder", effective_mode="ledger_only", note=self.note)

    def event(self, kind: str, **payload) -> None:
        with self.ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": now(), "kind": kind, **_plain(payload)}, ensure_ascii=False) + "\n")

    def start(self, name: str, job_type: str, group: str, config: dict, tags=(), description: str = "") -> Run:
        config = _plain(config)
        self.event("run_started", run=name, job_type=job_type, group=group, tags=list(tags),
                   description=description, config=config)
        handle = None
        if self.swanlab:
            arguments = dict(project=self.cfg.project, workspace=self.cfg.workspace, name=name,
                             description=description[:500], job_type=job_type, group=group, tags=list(tags),
                             config=config, mode=self.mode, log_dir=str(self.log_dir))
            if self.mode == "online":
                arguments.update(public=self.cfg.public, on_online_error="offline")
            try:
                handle = self.swanlab.init(**arguments)
                self.runs += 1
            except Exception as error:
                self.disable(error)
        return Run(self, name, handle)

"""模型注册表：版本、血缘、参考窗口，以及对外发布的当前上线模型。"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .drift import Reference


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


class Registry:
    """v0 永远是官方静态基线。之后每次通过门控的候选得到一个新版本号，旧版本不删除，回滚只改指针。"""

    def __init__(self, root: Path, baseline):
        self.root = Path(root)
        self.baseline = baseline
        self._models: dict = {}
        self.index_path = self.root / "index.json"
        if self.index_path.is_file():
            self.index = read_json(self.index_path)
        else:
            (self.root / "v0").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(baseline.checkpoint, self.root / "v0" / "checkpoint.npz")
            self.index = {"versions": {"v0": {
                "version": "v0", "role": "official_static_baseline", "parent": None, "parent_sha256": None,
                "sha256": baseline.checkpoint_digest(baseline.checkpoint), "batch": None, "trial": None,
                "created_at": now(), "steps": baseline.meta["steps"], "epochs": baseline.meta["epochs"]}}}
            write_json(self.index_path, self.index)

    def versions(self) -> list[str]:
        return sorted(self.index["versions"], key=lambda v: int(v[1:]))

    def info(self, version: str) -> dict:
        return self.index["versions"][version]

    def checkpoint(self, version: str) -> Path:
        return self.root / version / "checkpoint.npz"

    def model(self, version: str):
        if version not in self._models:
            self._models[version] = self.baseline.Model.load(self.checkpoint(version))
        return self._models[version]

    def add(self, source: Path, *, parent: str, batch: str, trial: str, gate: dict) -> str:
        version = f"v{len(self.index['versions'])}"
        copy_atomic(Path(source), self.checkpoint(version))
        model = self.model(version)
        self.index["versions"][version] = {
            "version": version, "role": "promoted_candidate", "parent": parent,
            "parent_sha256": self.info(parent)["sha256"],
            "sha256": self.baseline.checkpoint_digest(self.checkpoint(version)), "batch": batch, "trial": trial,
            "created_at": now(), "steps": model.meta["steps"], "epochs": model.meta["epochs"],
            "gate": {c["name"]: c["passed"] for c in gate["checks"]}}  # retention 为提示项
        write_json(self.index_path, self.index)
        return version

    def reference(self, version: str) -> Reference | None:
        path = self.root / version / "reference.npz"
        return Reference.load(path) if path.is_file() else None

    def set_reference(self, version: str, reference: Reference) -> None:
        reference.save(self.root / version / "reference.npz")

    def publish(self, folder: Path, version: str, decision: dict) -> None:
        """把当前上线模型和本批决策放到固定位置，供平台或下游读取。"""
        copy_atomic(self.checkpoint(version), Path(folder) / "checkpoint.npz")
        write_json(Path(folder) / "DECISION.json", decision)

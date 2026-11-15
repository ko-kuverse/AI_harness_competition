"""本地漂移流模拟器：把公开数据切成按批到达的数据流，另留一份 Harness 看不到的评分数据。

regime        工况与故障模式切换。按子集顺序出批，子集之间的差异是真实存在的分布变化。
sensor-aging  传感器老化后更换。各子集混合后平稳出批，中间几批给若干传感器叠加逐批增大的偏移，
              最后几批偏移归零（换上新传感器）。漂移发生在哪些特征、幅度多大都是已知的。

每批的训练部分取自公开训练集，验证部分取自公开验证集。留出评分数据取自公开训练集中整台划出的设备。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

UNIT = re.compile(r"^(.*)_c\d+$")


def _load(path: Path) -> dict:
    """按设备归组原始行，保持文件内的顺序。"""
    units: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            identity = json.loads(line)["id"]
            match = UNIT.match(identity)
            units.setdefault(match.group(1) if match else identity, []).append(line)
    return units


def _chunks(items: list, parts: int) -> list[list]:
    return [list(chunk) for chunk in np.array_split(np.asarray(items, dtype=object), parts)]


def _write(path: Path, units: dict, names: list, transform=None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", encoding="utf-8") as stream:
        for name in names:
            for line in units[name]:
                stream.write((transform(line) if transform else line) + "\n")
                rows += 1
    return rows


def simulate(public_dir: str, out_dir: str, scenario: str = "regime", *, seed: int = 7,
             hidden_fraction: float = 0.15, order=("fd001", "fd003", "fd002", "fd004"), parts: int = 2,
             drift_levels=(0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 0.0, 0.0), drift_sensors=(1, 2, 3)) -> dict:
    public, out = Path(public_dir), Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} 已存在且非空，请换一个输出目录")
    rng = np.random.default_rng(seed)
    train, validation = _load(public / "train.jsonl"), _load(public / "validation.jsonl")

    def shuffled(units: dict, prefix: str | None = None) -> list:
        names = sorted(n for n in units if prefix is None or n.startswith(prefix + "_"))
        return [names[i] for i in rng.permutation(len(names))]

    plan = []       # (批次名, 训练设备, 验证设备, 留出设备, 偏移量)
    if scenario == "regime":
        for subset in order:
            names = shuffled(train, subset)
            hidden = max(parts, int(round(hidden_fraction * len(names))))
            groups = zip(_chunks(names[hidden:], parts), _chunks(shuffled(validation, subset), parts),
                         _chunks(names[:hidden], parts))
            for letter, (a, b, c) in zip("abcdefgh", groups):
                plan.append((f"{subset}{letter}", a, b, c, 0.0))
    elif scenario == "sensor-aging":
        names = shuffled(train)
        hidden = int(round(hidden_fraction * len(names)))
        batches = len(drift_levels)
        groups = zip(_chunks(names[hidden:], batches), _chunks(shuffled(validation), batches),
                     _chunks(names[:hidden], batches))
        peak = int(np.argmax(drift_levels))
        for k, (a, b, c) in enumerate(groups):
            level = float(drift_levels[k])
            label = f"aging{level:.1f}".replace(".", "p") if level > 0 else ("replaced" if k > peak else "clean")
            plan.append((label, a, b, c, level))
    else:
        raise ValueError("scenario 只能是 regime 或 sensor-aging")

    sigma = None
    if scenario == "sensor-aging":
        values = np.array([json.loads(line)["inputs"]["sensors"] for lines in train.values() for line in lines[::5]])
        sigma = values.std(axis=0)

    truth = {"scenario": scenario, "seed": seed, "hidden_fraction": hidden_fraction, "batches": []}
    for k, (label, a, b, c, level) in enumerate(plan):
        name = f"batch-{k:03d}-{label}"

        def transform(line: str, level=level) -> str:
            row = json.loads(line)
            for j in drift_sensors:
                row["inputs"]["sensors"][j] = round(row["inputs"]["sensors"][j] + level * float(sigma[j]), 4)
            return json.dumps(row, ensure_ascii=False, separators=(",", ":"))

        change = transform if level > 0 else None
        counts = {"train": _write(out / "stream" / name / "train.jsonl", train, a, change),
                  "validation": _write(out / "stream" / name / "validation.jsonl", validation, b, change),
                  "hidden_test": _write(out / "hidden" / name / "test.jsonl", train, c, change)}
        truth["batches"].append({"name": name, "rows": counts, "engines": {"train": len(a), "validation": len(b), "hidden_test": len(c)},
                                 "injected_offset_sigma": level,
                                 "drifted_features": [f"sensors[{j}]" for j in drift_sensors] if level > 0 else []})
    (out / "TRUTH.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return truth

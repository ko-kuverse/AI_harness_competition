"""样本表、设备分组、任务指标与批次流适配器。"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Table:
    ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray      # 设备（或连续行分块）编号，自助法和划分都以它为单位

    def __len__(self) -> int:
        return int(len(self.ids))

    def take(self, index) -> "Table":
        return Table(self.ids[index], self.x[index], self.y[index], self.groups[index])

    def sample(self, rows: int, rng: np.random.Generator) -> "Table":
        if len(self) <= rows:
            return self
        return self.take(np.sort(rng.choice(len(self), size=rows, replace=False)))

    @staticmethod
    def concat(tables: list["Table"]) -> "Table":
        kept = [t for t in tables if len(t)] or tables[:1]
        if len(kept) == 1:
            return kept[0]
        return Table(np.concatenate([t.ids for t in kept]), np.concatenate([t.x for t in kept]),
                     np.concatenate([t.y for t in kept]), np.concatenate([t.groups for t in kept]))


def derive_groups(ids, pattern: str, block_rows: int, prefix: str) -> np.ndarray:
    """id 能匹配 pattern 时取其设备编号；有任何一条匹配不上就整表改用连续行分块。"""
    regex = re.compile(pattern)
    matches = [regex.match(identity) for identity in ids]
    if all(matches):
        return np.asarray([m.group(1) for m in matches], dtype=object)
    return np.asarray([f"{prefix}#blk{i // block_rows}" for i in range(len(ids))], dtype=object)


def split_groups(table: Table, fraction: float, seed: int) -> tuple[Table, Table]:
    """按设备把表分成（其余，划出部分）。划分只依赖设备编号的哈希，重跑结果相同。"""
    groups = sorted(set(table.groups.tolist()), key=lambda g: zlib.crc32(f"{seed}:{g}".encode()))
    if len(groups) < 2:
        cut = max(1, int(round(len(table) * (1 - fraction))))
        mask = np.arange(len(table)) >= cut
    else:
        count = min(len(groups) - 1, max(1, int(round(fraction * len(groups)))))
        chosen = set(groups[:count])
        mask = np.fromiter((g in chosen for g in table.groups), dtype=bool, count=len(table))
    return table.take(~mask), table.take(mask)


class Task:
    """回归用 RMSE、分类用错误率做主指标，两者都是越低越好，门控和自助法共用同一套代码。"""

    def __init__(self, kind: str, classes=()):
        if kind not in ("regression", "classification"):
            raise ValueError("unsupported task")
        self.kind = kind
        self.classes = list(classes)
        self.primary = "rmse" if kind == "regression" else "error_rate"
        self.safety = "nasa_exponent_mean" if kind == "regression" else None
        # 试验内部逐轮挑最好的一轮用的指标。分类的错误率在训练初期常常不动，改看对数损失。
        self.epoch_metric = "rmse" if kind == "regression" else "log_loss"

    def predict(self, model, x: np.ndarray) -> np.ndarray:
        output = model.predict(x)
        return output if self.kind == "regression" else output.argmax(axis=1)

    def losses(self, predicted: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.kind == "regression":
            return (predicted - y) ** 2
        return (predicted != y).astype(np.float64)

    def aggregate(self, loss_sum, count):
        mean = loss_sum / np.maximum(count, 1)
        return np.sqrt(mean) if self.kind == "regression" else mean

    def metrics(self, predicted: np.ndarray, y: np.ndarray) -> dict:
        if not len(y):
            return {"n": 0}
        if self.kind == "regression":
            error = predicted - y
            # NASA 评分的指数项：高估（error>0）除以 10，低估除以 13。取均值比对指数求和稳健得多。
            exponent = np.where(error >= 0, error / 10.0, -error / 13.0)
            return {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)),
                    "over_rate": float(np.mean(error > 0)),
                    "nasa_exponent_mean": float(np.mean(exponent)),
                    "nasa_score": float(np.sum(np.expm1(np.minimum(exponent, 700.0))))}
        labels = sorted(set(y.tolist()) | set(predicted.tolist()))
        f1 = [2 * np.sum((predicted == k) & (y == k)) / (np.sum(predicted == k) + np.sum(y == k)) for k in labels]
        accuracy = float(np.mean(predicted == y))
        return {"n": int(len(y)), "error_rate": 1.0 - accuracy, "accuracy": accuracy, "macro_f1": float(np.mean(f1))}

    def evaluate(self, model, table: Table) -> tuple[dict, np.ndarray]:
        if not len(table):
            return {"n": 0}, np.zeros(0)
        output = model.predict(table.x)
        if self.kind == "regression":
            return self.metrics(output, table.y), self.losses(output, table.y)
        predicted = output.argmax(axis=1)
        likelihood = np.maximum(output[np.arange(len(table)), table.y], 1e-12)
        return ({**self.metrics(predicted, table.y), "log_loss": float(-np.mean(np.log(likelihood)))},
                self.losses(predicted, table.y))


@dataclass
class Batch:
    index: int
    name: str
    train: Table            # 可用于梯度更新
    validation: Table       # 只用于评估与门控
    train_sha256: str
    validation_sha256: str
    repeated_rows: int      # 平台把历史数据随批重发时，被去重掉的行数


class DirectoryStream:
    """把一个目录当作批次流：每个子目录一批，内含 train.jsonl 与 validation.jsonl。

    接入评测平台时只需换掉这个类：next_batch 取一批带标签数据，submit_decision 交回本批决策。
    没有交回决策就取下一批会直接报错，与平台“先判断再发下一批”的约束一致。
    """

    def __init__(self, root: str | Path, baseline, group_pattern: str, block_rows: int):
        self.root = Path(root)
        self.baseline = baseline
        self.group_pattern = group_pattern
        self.block_rows = block_rows
        self.cursor = 0
        self.awaiting: str | None = None
        self.seen: set[str] = set()

    def batch_dirs(self) -> list[Path]:
        return sorted(d for d in self.root.iterdir()
                      if d.is_dir() and (d / "train.jsonl").is_file() and (d / "validation.jsonl").is_file())

    def _table(self, raw: dict, prefix: str) -> tuple[Table, int]:
        ids = np.asarray(raw["ids"], dtype=object)
        fresh = np.fromiter((identity not in self.seen for identity in raw["ids"]), dtype=bool, count=len(ids))
        groups = derive_groups(raw["ids"], self.group_pattern, self.block_rows, prefix)
        return Table(ids, raw["x"], raw["y"], groups).take(fresh), int((~fresh).sum())

    def next_batch(self) -> Batch | None:
        if self.awaiting is not None:
            raise RuntimeError(f"批次 {self.awaiting} 还没有提交决策，不能读取下一批")
        dirs = self.batch_dirs()
        if self.cursor >= len(dirs):
            return None
        folder = dirs[self.cursor]
        meta = self.baseline.meta
        train = self.baseline.read_dataset(folder / "train.jsonl", meta["task"], meta["classes"], meta["paths"])
        validation = self.baseline.read_dataset(folder / "validation.jsonl", meta["task"], meta["classes"], meta["paths"])
        if set(train["ids"]) & set(validation["ids"]):
            raise ValueError(f"{folder.name}: 训练与验证样本 id 重叠")
        train_table, repeated_a = self._table(train, f"{folder.name}/train")
        validation_table, repeated_b = self._table(validation, f"{folder.name}/validation")
        self.seen.update(train["ids"])
        self.seen.update(validation["ids"])
        batch = Batch(self.cursor, folder.name, train_table, validation_table,
                      train["sha256"], validation["sha256"], repeated_a + repeated_b)
        self.cursor += 1
        self.awaiting = batch.name
        return batch

    def submit_decision(self, batch_name: str, decision: dict) -> None:
        if self.awaiting != batch_name:
            raise RuntimeError(f"当前等待决策的批次是 {self.awaiting}，不是 {batch_name}")
        if decision.get("action") not in ("promote", "hold", "rollback") or not decision.get("model", {}).get("sha256"):
            raise ValueError("决策必须给出 action 与所选模型的摘要")
        self.awaiting = None

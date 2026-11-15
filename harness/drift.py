"""漂移检测与归因：输入分布、目标分布、在线模型性能三路信号。"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .data import Table


def feature_name(path: str) -> str:
    """把检查点里的特征路径（JSON）转成 sensors[3] 这样的短名。"""
    try:
        parts = json.loads(path)
    except ValueError:
        return path
    name = ""
    for kind, value in parts:
        name += f"[{value}]" if kind == "index" else (("." if name else "") + str(value))
    return name


PSI_CAP = 10.0     # 分布几乎不重叠时 PSI 可以任意大，统一截到这个值，报告里不出现没有意义的大数


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """分位数分箱的 PSI。两端各加一个参考窗口范围之外的箱，取值越界会直接抬高分数。"""
    quantiles = np.quantile(reference, np.arange(1, bins) / bins)
    eps = 1e-9 * max(1.0, float(np.max(np.abs(reference))))
    edges = np.unique(np.concatenate([quantiles, [reference.min() - eps, reference.max() + eps]]))
    a = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=len(edges) + 1).astype(np.float64)
    b = np.bincount(np.searchsorted(edges, current, side="right"), minlength=len(edges) + 1).astype(np.float64)
    a = (a + 0.5) / (a.sum() + 0.5 * len(a))
    b = (b + 0.5) / (b.sum() + 0.5 * len(b))
    return float(min(PSI_CAP, np.sum((b - a) * np.log(b / a))))


def ks(reference: np.ndarray, current: np.ndarray) -> float:
    a, b = np.sort(reference), np.sort(current)
    points = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, points, side="right") / len(a)
                               - np.searchsorted(b, points, side="right") / len(b))))


def gaussian_psi(mean, std):
    """N(mean, std²) 与 N(0, 1) 之间的对称 KL。PSI 就是分箱后的对称 KL，所以两者共用同一组阈值。"""
    variance = np.clip(std, 0.05, 20.0) ** 2
    return np.minimum(PSI_CAP, 0.5 * (variance + 1 / variance - 2) + 0.5 * mean ** 2 * (1 + 1 / variance))


def class_psi(reference: np.ndarray, current: np.ndarray, classes: int) -> float:
    a = (np.bincount(reference, minlength=classes) + 0.5) / (len(reference) + 0.5 * classes)
    b = (np.bincount(current, minlength=classes) + 0.5) / (len(current) + 0.5 * classes)
    return float(min(PSI_CAP, np.sum((b - a) * np.log(b / a))))


def _half_masks(groups: np.ndarray, reps: int, seed: int):
    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    if len(unique) < 4:
        return
    rng = np.random.default_rng([seed, len(unique)])
    for _ in range(reps):
        chosen = set(unique[rng.permutation(len(unique))[: len(unique) // 2]].tolist())
        yield np.fromiter((g in chosen for g in groups), dtype=bool, count=len(groups))


@dataclass
class Reference:
    """一个模型版本上线时的参考窗口：输入与目标样本、窗口内部的噪声底，以及当时的性能锚点。"""
    batch: str
    x: np.ndarray
    y: np.ndarray
    floor: np.ndarray           # 每个特征按设备折半后的最大 PSI
    target_floor: float
    anchor_score: float         # 上线时该版本在门控集上的主指标
    anchor_high: float          # 上述主指标的自助法上界，判断性能劣化时用它来容纳锚点自身的抽样误差

    @staticmethod
    def build(batch: str, window: Table, anchor: dict, task, cfg, seed: int) -> "Reference":
        window = window.sample(cfg.reference_rows, np.random.default_rng([seed, len(window)]))
        floor, target_floor = np.zeros(window.x.shape[1]), 0.0
        for mask in _half_masks(window.groups, cfg.noise_floor_reps, seed):
            for j in range(window.x.shape[1]):
                floor[j] = max(floor[j], psi(window.x[mask, j], window.x[~mask, j], cfg.psi_bins))
            if task.kind == "regression":
                value = psi(window.y[mask], window.y[~mask], cfg.psi_bins)
            else:
                value = class_psi(window.y[mask], window.y[~mask], len(task.classes))
            target_floor = max(target_floor, value)
        return Reference(batch, window.x, window.y, floor, float(target_floor),
                         float(anchor["score"]), float(anchor["ci_high"]))

    def save(self, path) -> None:
        np.savez(path, batch=np.array(self.batch), x=self.x, y=self.y, floor=self.floor,
                 target_floor=np.array(self.target_floor), anchor_score=np.array(self.anchor_score),
                 anchor_high=np.array(self.anchor_high))

    @staticmethod
    def load(path) -> "Reference":
        with np.load(path, allow_pickle=False) as z:
            return Reference(str(z["batch"]), z["x"], z["y"], z["floor"], float(z["target_floor"]),
                             float(z["anchor_score"]), float(z["anchor_high"]))


def detect_inputs(reference: Reference | None, model, x: np.ndarray, cfg) -> dict:
    """有参考窗口时逐特征算分箱 PSI 与 KS；静态基线没有训练样本可比，就用检查点里的标准化统计量做矩近似。"""
    paths = model.meta["paths"]
    z = (x - model.mean) / model.scale
    rows = []
    if reference is None:
        method = "moments_vs_checkpoint"
        scores = gaussian_psi(z.mean(axis=0), z.std(axis=0))
        for j, path in enumerate(paths):
            rows.append({"feature": feature_name(path), "index": j, "psi": float(scores[j]), "ks": None,
                         "shift_z": float(z[:, j].mean()), "threshold": cfg.psi_alert})
    else:
        method = "binned_vs_reference"
        reference_z = (reference.x - model.mean) / model.scale
        for j, path in enumerate(paths):
            rows.append({"feature": feature_name(path), "index": j,
                         "psi": psi(reference.x[:, j], x[:, j], cfg.psi_bins),
                         "ks": ks(reference.x[:, j], x[:, j]),
                         "shift_z": float(z[:, j].mean() - reference_z[:, j].mean()),
                         "threshold": float(max(cfg.psi_alert, cfg.noise_floor_mult * reference.floor[j]))})
    for row in rows:
        row["alert"] = bool(row["psi"] >= row["threshold"])
    rows.sort(key=lambda r: -r["psi"])
    alerts = sum(r["alert"] for r in rows)
    values = np.array([r["psi"] for r in rows])
    return {"method": method, "reference_batch": reference.batch if reference else None,
            "n_features": len(rows), "n_alert": int(alerts), "n_warn": int(np.sum(values >= cfg.psi_warn)),
            "share_alert": alerts / len(rows), "psi_max": float(values.max()), "psi_median": float(np.median(values)),
            "flag": bool(alerts >= cfg.min_alert_features), "features": rows}


def detect_target(reference: Reference | None, model, y: np.ndarray, task, cfg) -> dict:
    if task.kind == "regression":
        if reference is None:
            mean = (y.mean() - model.meta["target_mean"]) / model.meta["target_scale"]
            value, threshold, method = float(gaussian_psi(mean, y.std() / model.meta["target_scale"])), cfg.target_psi_alert, "moments_vs_checkpoint"
        else:
            value = psi(reference.y, y, cfg.psi_bins)
            threshold, method = max(cfg.target_psi_alert, cfg.noise_floor_mult * reference.target_floor), "binned_vs_reference"
    else:
        if reference is None:
            return {"method": "unavailable", "psi": None, "threshold": None, "flag": False}
        value = class_psi(reference.y, y, len(task.classes))
        threshold, method = max(cfg.target_psi_alert, cfg.noise_floor_mult * reference.target_floor), "class_frequency"
    return {"method": method, "psi": float(value), "threshold": float(threshold), "flag": bool(value >= threshold)}


def detect_performance(anchor: dict | None, score: float, ci_low: float, cfg) -> dict:
    """在线模型在新批次上的主指标对比其上线时的锚点。

    两边都是有限样本上的估计：新批次取自助法下界，锚点取自助法上界，下界仍越过容忍线才算显著劣化。
    """
    if not anchor or anchor["score"] <= 0:
        return {"anchor": None, "anchor_high": None, "score": float(score), "ci_low": float(ci_low),
                "ratio": None, "flag": False, "severe": False}
    return {"anchor": float(anchor["score"]), "anchor_high": float(anchor["high"]), "score": float(score),
            "ci_low": float(ci_low), "ratio": float(score / anchor["score"]),
            "flag": bool(ci_low > anchor["high"] * (1 + cfg.perf_tol)),
            "severe": bool(ci_low > anchor["high"] * (1 + cfg.perf_severe))}


def error_profile(task, predicted: np.ndarray, y: np.ndarray) -> dict:
    """误差落在哪里：回归按目标值三等分看各段 RMSE 与偏差，分类看各类召回。"""
    if task.kind == "regression":
        cuts = np.quantile(y, [1 / 3, 2 / 3])
        segments = []
        for label, mask in (("low", y <= cuts[0]), ("mid", (y > cuts[0]) & (y <= cuts[1])), ("high", y > cuts[1])):
            if mask.any():
                error = predicted[mask] - y[mask]
                segments.append({"segment": label, "n": int(mask.sum()), "rmse": float(np.sqrt(np.mean(error ** 2))),
                                 "bias": float(error.mean())})
        return {"by_target_tercile": segments}
    recall = [{"class": name, "n": int(np.sum(y == k)),
               "recall": float(np.mean(predicted[y == k] == k)) if np.any(y == k) else None}
              for k, name in enumerate(task.classes)]
    return {"by_class": recall}


def diagnose(inputs: dict, target: dict, performance: dict, profile: dict, cfg) -> dict:
    """把三路信号合成一个结论，供触发判断、实验规划和报告使用。"""
    if inputs["flag"] and performance["flag"]:
        kind = "covariate_harmful"
    elif inputs["flag"]:
        kind = "covariate_benign"
    elif performance["flag"]:
        kind = "prior_shift" if target["flag"] else "concept"
    elif target["flag"]:
        kind = "prior_shift_benign"
    else:
        kind = "none"
    if kind == "none":
        severity = "none"
    elif performance["severe"] or inputs["share_alert"] >= cfg.severe_share:
        severity = "severe"
    else:
        severity = "moderate"
    top = [r for r in inputs["features"] if r["alert"]][:5]
    parts = []
    if inputs["flag"]:
        names = "、".join(f"{r['feature']}(PSI {r['psi']:.2f}, 均值偏移 {r['shift_z']:+.2f}σ)" for r in top[:3])
        parts.append(f"输入漂移：{inputs['n_alert']}/{inputs['n_features']} 个特征越线，最显著的是 {names}")
    else:
        parts.append(f"输入分布稳定（越线特征 {inputs['n_alert']} 个，PSI 最大 {inputs['psi_max']:.2f}）")
    if performance["ratio"] is not None:
        change = (performance["ratio"] - 1) * 100
        parts.append(f"在线模型主指标相对锚点 {change:+.1f}%" + ("，劣化显著" if performance["flag"] else ""))
    if target["flag"]:
        parts.append(f"目标分布变化（PSI {target['psi']:.2f}）")
    meaning = {"covariate_harmful": "输入漂移已经损害精度", "covariate_benign": "输入漂移但精度尚未受损",
               "concept": "输入稳定而输入到目标的映射变了", "prior_shift": "目标分布变化并带来精度损失",
               "prior_shift_benign": "目标分布变化但精度尚稳", "none": "未见漂移"}[kind]
    return {"kind": kind, "severity": severity, "flag": kind != "none", "meaning": meaning,
            "summary": "；".join(parts) + f"。结论：{meaning}。",
            "top_features": [{k: r[k] for k in ("feature", "psi", "ks", "shift_z")} for r in top],
            "error_profile": profile}

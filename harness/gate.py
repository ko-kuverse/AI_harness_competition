"""门控：按设备重采样的配对自助法、上线门控、回滚守卫。"""
from __future__ import annotations

import numpy as np

from .data import Table


def _clusters(groups: np.ndarray) -> np.ndarray:
    """设备数太少时自助法没有分辨力，这时把每台设备再按连续 32 行切块作为重采样单位。"""
    names, inverse = np.unique(groups.astype(str), return_inverse=True)
    if len(names) >= 8:
        return inverse
    position = np.zeros(len(groups), dtype=np.int64)
    for g in range(len(names)):
        index = np.flatnonzero(inverse == g)
        position[index] = np.arange(len(index)) // 32
    _, refined = np.unique(inverse * (int(position.max()) + 1) + position, return_inverse=True)
    return refined


def bootstrap(task, losses: list[np.ndarray], groups: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """返回 (n_boot, 模型数) 的主指标矩阵。各模型共用同一组重采样，所以差值是配对的。"""
    inverse = _clusters(groups)
    count = int(inverse.max()) + 1
    sizes = np.bincount(inverse, minlength=count).astype(np.float64)
    sums = [np.bincount(inverse, weights=loss, minlength=count) for loss in losses]
    index = rng.integers(0, count, size=(n_boot, count))
    total = sizes[index].sum(axis=1)
    return np.stack([task.aggregate(s[index].sum(axis=1), total) for s in sums], axis=1)


def score_interval(task, loss: np.ndarray, groups: np.ndarray, cfg, rng) -> dict:
    boot = bootstrap(task, [loss], groups, cfg.bootstrap, rng)[:, 0]
    return {"score": float(task.aggregate(loss.sum(), len(loss))),
            "ci_low": float(np.quantile(boot, 1 - cfg.confidence)),
            "ci_high": float(np.quantile(boot, cfg.confidence))}


def compare(task, loss_a: np.ndarray, loss_b: np.ndarray, groups: np.ndarray, cfg, rng) -> dict:
    """a 相对 b 的主指标差（a − b，负值表示 a 更好）及其单侧置信界。"""
    boot = bootstrap(task, [loss_a, loss_b], groups, cfg.bootstrap, rng)
    delta = boot[:, 0] - boot[:, 1]
    a = float(task.aggregate(loss_a.sum(), len(loss_a)))
    b = float(task.aggregate(loss_b.sum(), len(loss_b)))
    return {"a": a, "b": b, "delta": a - b, "rel": (a - b) / b if b > 0 else 0.0,
            "ci_low": float(np.quantile(delta, 1 - cfg.confidence)),
            "ci_high": float(np.quantile(delta, cfg.confidence)),
            "clusters": int(_clusters(groups).max()) + 1}


def promotion_gate(task, cfg, *, candidate, incumbent, static, gate_set: Table, retention: Table,
                   sanity: dict, rng) -> dict:
    """候选模型要通过全部拦截项才上线。gate_set 在本批次里只在这里用一次。

    retention 是提示项：流上的分布未必回到过去，遗忘旧批次不应拦住对当前分布的修复。
    旧版本都留在注册表里，旧工况回来时由回滚守卫切回去；遗忘幅度照常写进决策和记录。
    """
    primary = task.primary
    checks = []

    def add(name: str, passed: bool, **detail) -> None:
        checks.append({"name": name, "passed": bool(passed), **detail})

    add("sanity", sanity["ok"], **{k: v for k, v in sanity.items() if k != "ok"})
    units = len(set(gate_set.groups.tolist()))
    add("enough_units", units >= cfg.min_gate_groups, units=units, need=cfg.min_gate_groups)
    candidate_metrics, candidate_loss = task.evaluate(candidate, gate_set)
    incumbent_metrics, incumbent_loss = task.evaluate(incumbent, gate_set)
    static_metrics, _ = task.evaluate(static, gate_set)
    versus = compare(task, candidate_loss, incumbent_loss, gate_set.groups, cfg, rng)
    add("gain_vs_incumbent", -versus["rel"] >= cfg.min_rel_gain and versus["ci_high"] < 0,
        candidate=versus["a"], incumbent=versus["b"], rel_gain=-versus["rel"], need_rel_gain=cfg.min_rel_gain,
        delta_ci_high=versus["ci_high"], confidence=cfg.confidence, clusters=versus["clusters"])
    add("not_worse_than_static", candidate_metrics[primary] <= static_metrics[primary] * (1 + cfg.static_tol),
        candidate=candidate_metrics[primary], static=static_metrics[primary], tolerance=cfg.static_tol)
    if len(retention):
        kept, _ = task.evaluate(candidate, retention)
        before, _ = task.evaluate(incumbent, retention)
        loss = kept[primary] / before[primary] - 1 if before[primary] > 0 else 0.0
        add("retention", loss <= cfg.retention_tol, advisory=True, candidate=kept[primary],
            incumbent=before[primary], rel_loss=loss, tolerance=cfg.retention_tol, n=len(retention))
    else:
        add("retention", True, advisory=True, note="尚无历史批次")
    if task.safety:
        add("safety_asymmetric", candidate_metrics[task.safety] <= incumbent_metrics[task.safety] * (1 + cfg.safety_tol),
            metric=task.safety, candidate=candidate_metrics[task.safety], incumbent=incumbent_metrics[task.safety],
            tolerance=cfg.safety_tol)
    return {"passed": all(c["passed"] for c in checks if not c.get("advisory")), "checks": checks,
            "metrics": {"candidate": candidate_metrics, "incumbent": incumbent_metrics, "static": static_metrics}}


def rollback_guard(task, cfg, production: str, models: dict, table: Table, rng, static: str = "v0") -> dict:
    """新批次到达时先核对在线模型。若某个上线过的旧版本在新数据上明显更好，就退回那个版本。

    对静态基线用更严的规则：赛题的硬门槛是不低于静态基线，所以在线模型在新批次上的点估计
    只要比静态基线差（超过 static_floor_tol）就退回，不要求差异显著。
    """
    losses = {name: task.evaluate(model, table)[1] for name, model in models.items()}
    comparisons = []
    for name, loss in losses.items():
        if name == production:
            continue
        result = compare(task, losses[production], loss, table.groups, cfg, rng)
        if name == static:
            rule, eligible = "static_floor", result["rel"] > cfg.static_floor_tol
        else:
            rule, eligible = "significant_margin", result["rel"] >= cfg.rollback_margin and result["ci_low"] > 0
        comparisons.append({"fallback": name, "production": result["a"], "fallback_score": result["b"],
                            "rel_worse": result["rel"], "delta_ci_low": result["ci_low"],
                            "rule": rule, "eligible": bool(eligible)})
    eligible = [c for c in comparisons if c["eligible"]]
    target = min(eligible, key=lambda c: c["fallback_score"])["fallback"] if eligible else None
    return {"triggered": target is not None, "from": production, "to": target, "n": len(table),
            "margin": cfg.rollback_margin, "comparisons": comparisons}

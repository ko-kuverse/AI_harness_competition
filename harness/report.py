"""运行总结（summary.json、report.md）与基于留出数据的事后评分。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import derive_groups
from .gate import compare
from .registry import now, read_json, write_json


def _gain(model, static):
    return None if model is None or not static else (static - model) / static


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


VERDICT_TEXT = {"better": "显著更好", "worse": "显著更差", "not_significant": "不显著", "same_model": "同一模型",
                "too_few_units": "设备太少，不判"}
DRIFT_TEXT = {"none": "无", "covariate_harmful": "输入漂移，精度受损", "covariate_benign": "输入漂移，精度未损",
              "concept": "映射变化", "prior_shift": "目标分布变化，精度受损", "prior_shift_benign": "目标分布变化，精度未损",
              "unknown": "未知"}


def _fmt(value, pattern="{:.2f}"):
    return "–" if value is None else pattern.format(value)


def write_report(workdir: Path, state: dict, task, registry, swanlab_status: str) -> dict:
    rows = state["batches"]
    after = [_gain(r["after"], r["arrival_static"]) for r in rows]
    arrival = [_gain(r["arrival_online"], r["arrival_static"]) for r in rows]
    scored = [(g, r["batch"]) for g, r in zip(after, rows) if g is not None]
    worst = min(scored) if scored else (None, None)
    summary = {
        "generated_at": now(), "task": task.kind, "primary_metric": task.primary, "batches": len(rows),
        "actions": {a: sum(r["action"] == a for r in rows) for a in ("promote", "hold", "rollback")},
        "rollbacks": sum(r["rollback"] for r in rows), "failed_batches": sum(bool(r["error"]) for r in rows),
        "experiment_rounds": state["rounds_done"], "trials_total": state["trials_total"],
        "final_version": state["production"], "versions": registry.versions(),
        "mean_after": _mean([r["after"] for r in rows]), "mean_static": _mean([r["arrival_static"] for r in rows]),
        "mean_arrival_online": _mean([r["arrival_online"] for r in rows]),
        "mean_rel_gain_after_decision": _mean(after), "worst_rel_gain_after_decision": worst[0],
        "worst_batch": worst[1], "mean_rel_gain_at_arrival": _mean(arrival),
        "seconds_total": float(sum(r["seconds"] for r in rows)),
        "train_seconds_total": float(sum(r["train_seconds"] for r in rows)),
        "sample_epochs_total": int(sum(r["sample_epochs"] for r in rows)),
        "llm": state["llm"], "policy": state["policy"], "arms": state["arms"], "swanlab": swanlab_status}
    write_json(workdir / "summary.json", {**summary, "batches_detail": rows})

    metric = task.primary
    lines = [
        "# Harness 运行报告", "",
        f"- 任务类型 {task.kind}，主指标 {metric}（越低越好）",
        f"- 共 {len(rows)} 批：上线 {summary['actions']['promote']} 次，保持 {summary['actions']['hold']} 次，"
        f"回滚 {summary['rollbacks']} 次，处理失败 {summary['failed_batches']} 批",
        f"- 实验 {summary['experiment_rounds']} 轮，试验 {summary['trials_total']} 个，"
        f"训练耗时 {summary['train_seconds_total']:.1f} 秒，全流程 {summary['seconds_total']:.1f} 秒",
        f"- 决策后模型相对静态基线的逐批平均增益 {_fmt(summary['mean_rel_gain_after_decision'], '{:+.1%}')}，"
        f"最差一批 {_fmt(worst[0], '{:+.1%}')}（{worst[1]}）",
        f"- 新批次到达时（更新前）在线模型相对静态基线的逐批平均增益 {_fmt(summary['mean_rel_gain_at_arrival'], '{:+.1%}')}",
        f"- 最终上线版本 {state['production']}，SwanLab 记录方式 {swanlab_status}", "",
        "各列都在该批的验证部分上计算。到达时一列是更新之前的在线模型，属于样本外结果。"
        "决策后一列用的验证集同时也是门控集，数值偏乐观；有留出数据时用 score 子命令另行评分。", "",
        f"| 批次 | 漂移 | 程度 | 试验数 | 决策 | 版本 | 到达时在线 {metric} | 静态基线 {metric} | 决策后 {metric} | 相对静态基线 |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |"]
    for row, gain in zip(rows, after):
        action = row["action"] + ("（先回滚）" if row["rollback"] and row["action"] == "promote" else "")
        version = row["version_after"] if row["version_before"] == row["version_after"] else f"{row['version_before']}→{row['version_after']}"
        lines.append(f"| {row['batch']} | {DRIFT_TEXT.get(row['drift'], row['drift'])} | {row['severity']} | {row['trials']} | {action} | {version} | "
                     f"{_fmt(row['arrival_online'], '{:.3f}')} | {_fmt(row['arrival_static'], '{:.3f}')} | "
                     f"{_fmt(row['after'], '{:.3f}')} | {_fmt(gain, '{:+.1%}')} |")
    lines += ["", "## 策略层", "", f"当前策略：`{state['policy']}`", "",
              "| 试验臂 | 次数 | 假设成立 | 被上线 | 相对对照的平均增益 |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm, stat in sorted(state["arms"].items()):
        lines.append(f"| {arm} | {stat['tried']} | {stat['supported']} | {stat['promoted']} | "
                     f"{stat['gain_sum'] / stat['tried']:+.1%} |")
    (workdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def score(workdir: Path, hidden_dir: Path, baseline, task, registry, cfg) -> dict:
    """用 Harness 从未见过的留出数据，给每批决策选中的模型和静态基线各打一次分。

    留出样本每批只有十来台设备，单批的差值噪声不小，所以同时给出按设备配对自助法的显著性。
    """
    rows = []
    rng = np.random.default_rng(cfg.seed)
    for path in sorted((workdir / "decisions").glob("*.json")):
        decision = read_json(path)
        test = hidden_dir / decision["batch"] / "test.jsonl"
        if not test.is_file():
            continue
        meta = baseline.meta
        raw = baseline.read_dataset(test, meta["task"], meta["classes"], meta["paths"])
        version = decision["model"]["version"]
        if registry.info(version)["sha256"] != decision["model"]["sha256"]:
            raise ValueError(f"{decision['batch']}: 决策里的模型摘要与注册表不一致")
        predicted, reference = task.predict(registry.model(version), raw["x"]), task.predict(registry.model("v0"), raw["x"])
        chosen, static = task.metrics(predicted, raw["y"]), task.metrics(reference, raw["y"])
        verdict = "same_model"
        groups = derive_groups(raw["ids"], cfg.group_pattern, cfg.block_rows, decision["batch"])
        units = len(set(groups.tolist()))
        if version != "v0" and units < cfg.gate.min_gate_groups:
            verdict = "too_few_units"      # 设备太少，显著性判断不可靠
        elif version != "v0":
            versus = compare(task, task.losses(predicted, raw["y"]), task.losses(reference, raw["y"]), groups, cfg.gate, rng)
            verdict = "better" if versus["ci_high"] < 0 else ("worse" if versus["ci_low"] > 0 else "not_significant")
        rows.append({"batch": decision["batch"], "index": decision["batch_index"], "action": decision["action"],
                     "version": version, "n": chosen["n"], "units": units, "chosen": chosen, "static": static,
                     "rel_gain": _gain(chosen[task.primary], static[task.primary]), "versus_static": verdict})
    rows.sort(key=lambda r: r["index"])
    gains = [r["rel_gain"] for r in rows]
    worst = min(rows, key=lambda r: r["rel_gain"]) if rows else None
    total = sum(r["n"] for r in rows)
    result = {"generated_at": now(), "primary_metric": task.primary, "batches": len(rows), "samples": total,
              "mean_chosen": _mean([r["chosen"][task.primary] for r in rows]),
              "mean_static": _mean([r["static"][task.primary] for r in rows]),
              "mean_rel_gain": _mean(gains), "worst_rel_gain": worst["rel_gain"] if worst else None,
              "worst_batch": worst["batch"] if worst else None,
              "batches_not_worse_than_static": sum(g >= 0 for g in gains),
              "significantly_better": sum(r["versus_static"] == "better" for r in rows),
              "significantly_worse": sum(r["versus_static"] == "worse" for r in rows), "rows": rows}
    write_json(workdir / "score.json", result)
    metric = task.primary
    extra = [k for k in ("mae", "nasa_exponent_mean", "macro_f1") if rows and k in rows[0]["chosen"]]
    lines = ["# 留出数据评分", "",
             f"每批用该批决策选中的模型，在同一阶段的留出样本上评分，与静态基线 v0 对比。共 {len(rows)} 批、{total} 个样本。", "",
             f"- 逐批平均 {metric}：所选模型 {_fmt(result['mean_chosen'], '{:.3f}')}，静态基线 {_fmt(result['mean_static'], '{:.3f}')}",
             f"- 逐批平均相对增益 {_fmt(result['mean_rel_gain'], '{:+.1%}')}，最差一批 {_fmt(result['worst_rel_gain'], '{:+.1%}')}（{result['worst_batch']}）",
             f"- 点估计不差于静态基线的批次 {result['batches_not_worse_than_static']}/{len(rows)}；"
             f"按设备配对自助法（单侧 {cfg.gate.confidence:.0%}），显著更好 {result['significantly_better']} 批，"
             f"显著更差 {result['significantly_worse']} 批", "",
             "| 批次 | 决策 | 版本 | 样本数 | 设备数 | 所选模型 " + metric + " | 静态基线 " + metric + " | 相对增益 | 显著性 | " +
             " | ".join(f"所选 {k} | 基线 {k}" for k in extra) + " |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | " + " | ".join("---: | ---:" for _ in extra) + " |"]
    for r in rows:
        lines.append(f"| {r['batch']} | {r['action']} | {r['version']} | {r['n']} | {r['units']} | {r['chosen'][metric]:.3f} | "
                     f"{r['static'][metric]:.3f} | {r['rel_gain']:+.1%} | {VERDICT_TEXT[r['versus_static']]} | " +
                     " | ".join(f"{r['chosen'][k]:.3f} | {r['static'][k]:.3f}" for k in extra) + " |")
    (workdir / "score.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result

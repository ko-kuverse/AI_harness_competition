"""受控实验：读历史、提假设、单因素对照试验，以及策略层的自我修正。

每一轮以在线模型为对照（不更新），以当前策略为基准试验，其余试验各只改一个因素。
试验一律从在线模型的检查点续训，训练、保存都走官方 baselines.model.Model。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import Table
from .gate import compare
from .registry import now, write_json

SCOPES = ("recent", "replay", "cumulative")
SCOPE_TEXT = {"recent": "只用本批数据", "replay": "本批数据加历史回放", "cumulative": "全部已见数据"}
FACTORS = ("data_scope", "replay_ratio", "learning_rate", "batch_size", "epochs")


@dataclass
class TrialSpec:
    trial_id: str
    arm: str
    hypothesis: str
    changed_factor: str | None      # 相对基准试验改动的那个因素；基准自身为 None
    data_scope: str
    replay_ratio: float
    learning_rate: float
    batch_size: int
    epochs: int
    source: str = "rule"            # rule 或 llm

    def factors(self) -> dict:
        return {name: getattr(self, name) for name in FACTORS}


def sanitize(raw, cfg) -> dict | None:
    """把一条试验配置约束到允许的动作空间内，不合法就丢弃。大模型的提议必须先过这里。"""
    try:
        scope = raw["data_scope"]
        rate = float(raw["learning_rate"])
        batch = int(raw.get("batch_size", 256))
        epochs = int(raw.get("epochs", 8))
        ratio = float(raw.get("replay_ratio", 0.5))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if scope not in SCOPES or not all(math.isfinite(v) for v in (rate, ratio)) or rate <= 0:
        return None
    return {"data_scope": scope, "replay_ratio": float(min(2.0, max(0.1, ratio))),
            "learning_rate": float(min(cfg.lr_max, max(cfg.lr_min, rate))),
            "batch_size": int(min(cfg.batch_sizes, key=lambda size: abs(size - batch))),
            "epochs": int(min(cfg.max_epochs, max(1, epochs)))}


def _evidence(digest: list) -> str:
    for entry in reversed(digest):
        if entry.get("trials"):
            best = min(entry["trials"], key=lambda t: t["select_score"])
            return (f"上一轮（{entry['batch']}）选型集最优的是 {best['arm']}，"
                    f"相对对照 {best['rel_gain_vs_control'] * 100:+.1f}%，该批决策为 {entry['action']}")
    return "尚无历史试验，采用初始策略"


def plan_round(diagnosis: dict, policy: dict, digest: list, *, has_history: bool, budget: int,
               rounds_done: int, cfg, proposals: list | None = None) -> list[TrialSpec]:
    """生成本轮试验。proposals 是大模型给出的附加试验（已约束到动作空间），没有则走规则。"""
    base = sanitize(policy, cfg)
    if not has_history:
        base["data_scope"] = "recent"
    specs = [TrialSpec("t1-base", "base", (
        f"{diagnosis['meaning']}。按当前策略（{SCOPE_TEXT[base['data_scope']]}，学习率 {base['learning_rate']:g}，"
        f"至多 {base['epochs']} 轮）从在线模型续训，选型集主指标应低于不更新的对照。依据：{_evidence(digest)}。"),
        None, **base)]

    def variant(arm: str, hypothesis: str, factor: str, source: str = "rule", **change) -> None:
        config = sanitize({**base, **change}, cfg)
        if config and all(config != s.factors() for s in specs) and len(specs) < budget:
            specs.append(TrialSpec(f"t{len(specs) + 1}-{arm}", arm, hypothesis, factor, source=source, **config))

    for index, proposal in enumerate(proposals or []):
        config = sanitize(proposal, cfg)
        if config:
            changed = ",".join(k for k in FACTORS if config[k] != base[k]) or None
            variant(f"llm{index + 1}", str(proposal.get("hypothesis", "大模型提议"))[:400], changed, "llm", **config)

    direction = int(policy.get("lr_direction") or (1 if diagnosis["severity"] == "severe" else -1))
    rate = base["learning_rate"] * cfg.lr_step ** direction
    if not cfg.lr_min <= rate <= cfg.lr_max:
        direction, rate = -direction, base["learning_rate"] * cfg.lr_step ** -direction
    rate_text = (f"漂移程度 {diagnosis['severity']}，现有步长在限定轮数内可能收敛不足；学习率放大到 {rate:g} 应使选型集误差低于基准试验。"
                 if direction > 0 else
                 f"现有步长可能在新旧分布之间来回震荡；学习率缩小到 {rate:g} 应使选型集误差低于基准试验。")
    other = "recent" if base["data_scope"] != "recent" else "replay"
    scope_text = ("新批次的分布已整体切换，历史样本帮助有限；去掉回放、只用本批数据应不差于基准试验。"
                  if other == "recent" else
                  "旧工况仍会出现，混入历史回放应在不损失本批精度的前提下保住旧批次。")
    rule_variants = [("lr-up" if direction > 0 else "lr-down", rate_text, "learning_rate", {"learning_rate": rate})]
    if has_history:
        rule_variants.append((f"scope-{other}", scope_text, "data_scope", {"data_scope": other}))
        if rounds_done % 2:     # 预算只够一个变体时，两个因素轮流检验
            rule_variants.reverse()
        if base["data_scope"] != "cumulative":
            rule_variants.append(("scope-cumulative", "漂移幅度大，用全部已见数据续训能给出更稳的折中，选型集误差应低于基准试验。",
                                  "data_scope", {"data_scope": "cumulative"}))
    else:
        opposite = base["learning_rate"] * cfg.lr_step ** -direction
        rule_variants.append(("lr-up" if direction < 0 else "lr-down",
                              f"与上一个试验相反方向的步长（{opposite:g}），用来确定学习率该往哪边调。",
                              "learning_rate", {"learning_rate": opposite}))
    for arm, hypothesis, factor, change in rule_variants:
        variant(arm, hypothesis, factor, **change)
    return specs


def assemble(spec: TrialSpec, fit: Table, history: Table, cfg, rng) -> Table:
    if spec.data_scope == "recent" or not len(history):
        data = fit
    elif spec.data_scope == "replay":
        data = Table.concat([fit, history.sample(max(1, int(spec.replay_ratio * len(fit))), rng)])
    else:
        data = Table.concat([fit, history])
    return data.sample(cfg.max_train_rows, rng)


def run_trial(spec: TrialSpec, *, baseline, task, parent: str, parent_path: Path, fit: Table, history: Table,
              select: Table, retention: Table, out_dir: Path, recorder, batch, diagnosis: dict, cfg, seed: int) -> dict:
    """执行一个试验：续训、逐轮在选型集上评估、保留最好的一轮、落盘检查点与报告。"""
    primary = task.primary
    rng = np.random.default_rng([seed, batch.index, zlib.crc32(spec.trial_id.encode())])
    data = assemble(spec, fit, history, cfg, rng)
    model = baseline.Model.load(parent_path)
    parent_sha = baseline.checkpoint_digest(parent_path)
    initial_steps, initial_epochs = model.meta["steps"], model.meta["epochs"]
    model.meta["learning_rate"], model.meta["batch_size"] = spec.learning_rate, spec.batch_size
    model.validate()
    matrix_sha = "sha256:" + hashlib.sha256(data.x.tobytes() + data.y.tobytes()).hexdigest()
    control_select, _ = task.evaluate(model, select)
    control_retention, _ = task.evaluate(model, retention)
    name = f"b{batch.index:03d}-{spec.trial_id}"
    run = recorder.start(name, "trial", batch.name, tags=[spec.arm, diagnosis["kind"], spec.source],
                         description=spec.hypothesis, config={
        "batch": batch.name, "batch_index": batch.index, "trial": spec.trial_id, "arm": spec.arm,
        "hypothesis": spec.hypothesis, "changed_factor": spec.changed_factor, "planner": spec.source,
        **spec.factors(), "parent_version": parent, "parent_checkpoint_sha256": parent_sha,
        "seed": model.meta["seed"], "hidden": model.meta["hidden"], "parameter_count": model.meta["parameter_count"],
        "profile": model.meta.get("profile"), "task": task.kind, "primary_metric": primary,
        "train_rows": len(data), "select_rows": len(select), "retention_rows": len(retention),
        "train_matrix_sha256": matrix_sha, "batch_train_sha256": batch.train_sha256,
        "drift_kind": diagnosis["kind"], "drift_severity": diagnosis["severity"],
        "official_source_sha256": baseline.source_sha256})
    run.log({**{f"select/{k}": v for k, v in control_select.items() if k != "n"},
             **{f"retention/{primary}": control_retention.get(primary)}}, step=0)
    best, best_score, best_epoch, rows = None, math.inf, 0, []
    started = time.monotonic()
    for epoch in range(1, spec.epochs + 1):
        loss = model.fit(data.x, data.y, 1)[0]
        on_select, _ = task.evaluate(model, select)
        on_retention, _ = task.evaluate(model, retention)
        rows.append({"trial": name, "epoch": model.meta["epochs"], "step": model.meta["steps"], "loss": loss,
                     "select": on_select, "retention": on_retention})
        run.log({"train/loss": loss, **{f"select/{k}": v for k, v in on_select.items() if k != "n"},
                 f"retention/{primary}": on_retention.get(primary)}, step=epoch)
        if on_select[task.epoch_metric] < best_score - 1e-12:
            best, best_score, best_epoch = copy.deepcopy(model), on_select[task.epoch_metric], epoch
        elif epoch - best_epoch >= cfg.patience:
            break
    seconds = time.monotonic() - started
    epochs_run, model = len(rows), best
    out_dir.mkdir(parents=True)
    checkpoint = out_dir / "checkpoint.npz"
    model.save(checkpoint)
    reloaded = baseline.Model.load(checkpoint)
    probe = select.x[: min(len(select), 2048)]
    difference = float(np.max(np.abs(reloaded.predict(probe) - model.predict(probe))))
    sanity = {"reload_max_abs_diff": difference, "steps_before": initial_steps, "steps_after": model.meta["steps"],
              "parameter_count": model.meta["parameter_count"]}
    sanity["ok"] = bool(difference < 1e-9 and model.meta["steps"] > initial_steps and model.meta["parameter_count"] < 100000)
    after_select, _ = task.evaluate(model, select)
    after_retention, _ = task.evaluate(model, retention)
    gain = (control_select[primary] - after_select[primary]) / control_select[primary] if control_select[primary] > 0 else 0.0
    report = {
        "format": "harness.trial-report.v1", "profile": model.meta.get("profile"), "task": task.kind, "device": "cpu",
        "numpy_version": np.__version__, "python_version": platform.python_version(), "machine": platform.machine(),
        "stage_id": name, "seed": model.meta["seed"], "parameter_count": model.meta["parameter_count"],
        "input_features": len(model.meta["paths"]), "train_rows": len(data), "train_matrix_sha256": matrix_sha,
        "parent_version": parent, "parent_checkpoint_sha256": parent_sha,
        "checkpoint_sha256": baseline.checkpoint_digest(checkpoint),
        "initial_steps": initial_steps, "initial_epochs": initial_epochs,
        "final_steps": model.meta["steps"], "final_epochs": model.meta["epochs"],
        "epochs_run": epochs_run, "best_epoch": best_epoch, "training_seconds": seconds,
        "learning_rate": spec.learning_rate, "batch_size": spec.batch_size,
        "select_before": control_select, "select_after": after_select,
        "retention_before": control_retention, "retention_after": after_retention,
        "spec": asdict(spec), "source_sha256": baseline.source_sha256, "created_at": now()}
    write_json(out_dir / "training.json", report)
    with (out_dir / "history.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    run.log({f"final/select_{primary}": after_select[primary], "final/rel_gain_vs_control": gain,
             "final/best_epoch": best_epoch, "final/epochs_run": epochs_run, "final/seconds": seconds,
             "final/train_rows": len(data)}, step=epochs_run)
    run.finish()
    return {"trial": name, "trial_id": spec.trial_id, "arm": spec.arm, "source": spec.source,
            "hypothesis": spec.hypothesis, "changed_factor": spec.changed_factor, **spec.factors(),
            "checkpoint": str(checkpoint), "checkpoint_sha256": report["checkpoint_sha256"],
            "parent_version": parent, "train_rows": len(data), "epochs_run": epochs_run, "best_epoch": best_epoch,
            "seconds": seconds, "sample_epochs": len(data) * epochs_run,
            "control_select": control_select, "select": after_select, "select_score": after_select[primary],
            "control_retention": control_retention, "retention": after_retention,
            "rel_gain_vs_control": gain, "sanity": sanity, "model": model}


def judge(results: list[dict], *, task, select: Table, control, gate_cfg, cfg, rng) -> None:
    """给每个假设一个结论。基准试验与不更新的对照比，变体与基准试验比，都在选型集上做配对自助法。

    supported 要求差异显著且达到最小幅度，refuted 是显著更差，其余记为 inconclusive。
    """
    control_loss = task.evaluate(control, select)[1]
    losses = [task.evaluate(result["model"], select)[1] for result in results]
    base = results[0]
    for result, loss in zip(results, losses):
        is_base = result is base
        versus = compare(task, loss, control_loss if is_base else losses[0], select.groups, gate_cfg, rng)
        gain = -versus["rel"]
        result["rel_gain_vs_base"] = 0.0 if is_base else gain
        result["evidence"] = {"against": "control" if is_base else base["trial"], "rel_gain": gain,
                              "delta_ci_low": versus["ci_low"], "delta_ci_high": versus["ci_high"],
                              "clusters": versus["clusters"]}
        if versus["ci_high"] < 0 and gain >= (0.0 if is_base else cfg.policy_min_gain):
            result["verdict"] = "supported"
        else:
            result["verdict"] = "refuted" if versus["ci_low"] > 0 else "inconclusive"


def update_policy(policy: dict, results: list[dict], cfg) -> list[str]:
    """策略层自改进：变体的假设获得支持时，把它的取值写进策略，下一轮以它为基准。"""
    notes = []
    base = results[0]
    supported = [r for r in results[1:] if r["verdict"] == "supported"]
    best = min(supported, key=lambda r: r["select_score"]) if supported else base
    if best is not base:
        for factor in FACTORS:
            if policy.get(factor) != best[factor] and base[factor] != best[factor]:
                notes.append(f"{factor}: {base[factor]} → {best[factor]}（{best['arm']} 显著好于基准，{best['rel_gain_vs_base'] * 100:+.1f}%）")
                policy[factor] = best[factor]
    for result in results[1:]:
        if result["changed_factor"] == "learning_rate" and result["source"] == "rule":
            went_up = result["learning_rate"] > base["learning_rate"]
            won = result["verdict"] == "supported"
            policy["lr_direction"] = (1 if went_up else -1) * (1 if won else -1)
    return notes

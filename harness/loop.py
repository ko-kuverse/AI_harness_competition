"""主循环：逐批执行 漂移检测 → 回滚守卫 → 归因 → 受控实验 → 门控 → 决策。先交决策，再取下一批。"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np

from . import llm
from .bridge import load_baseline
from .data import DirectoryStream, Table, Task, split_groups
from .drift import Reference, detect_inputs, detect_performance, detect_target, diagnose, error_profile
from .experiment import SCOPES, judge, plan_round, run_trial, sanitize, update_policy
from .gate import promotion_gate, rollback_guard, score_interval
from .recorder import Recorder
from .registry import Registry, now, read_json, write_json
from .report import write_report

ACTION_CODE = {"promote": 1, "hold": 0, "rollback": -1}
TRIAL_FIELDS = ("trial", "arm", "source", "hypothesis", "changed_factor", "data_scope", "replay_ratio",
                "learning_rate", "batch_size", "epochs", "checkpoint_sha256", "parent_version", "train_rows",
                "epochs_run", "best_epoch", "seconds", "sample_epochs", "control_select", "select", "select_score",
                "control_retention", "retention", "retention_loss", "rel_gain_vs_control", "rel_gain_vs_base", "verdict",
                "evidence", "sanity")


class Harness:
    def __init__(self, cfg):
        self.cfg = cfg
        self.workdir = Path(cfg.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.baseline = load_baseline(cfg.baseline_dir, cfg.checkpoint)
        self.task = Task(self.baseline.task, self.baseline.meta["classes"])
        self.registry = Registry(self.workdir / "registry", self.baseline)
        self.recorder = Recorder(self.workdir, cfg.record)
        self.stream = DirectoryStream(cfg.stream_dir, self.baseline, cfg.group_pattern, cfg.block_rows)
        self.state_path = self.workdir / "state.json"
        if self.state_path.is_file():
            self.state = read_json(self.state_path)
        else:
            self.state = {"next_index": 0, "production": "v0", "lineage": ["v0"],
                          "policy": dict(cfg.experiment.initial_policy), "rounds_done": 0, "trials_total": 0,
                          "digest": [], "batches": [], "arms": {}, "started_at": now(),
                          "llm": {"calls": 0, "failures": 0, "prompt_tokens": 0, "completion_tokens": 0}}
            write_json(self.workdir / "config.json", cfg.to_dict())
        self.history_train: list[Table] = []
        self.history_gate: list[Table] = []

    # ---- 数据划分 -------------------------------------------------------------------------------

    def _partition(self, batch):
        """本批训练部分 → 拟合集 + 选型集；验证部分 → 门控集。平台不单给验证集时从训练部分按设备划出。"""
        if len(batch.validation):
            pool, gate_set, source = batch.train, batch.validation, "validation"
        else:
            pool, gate_set = split_groups(batch.train, 0.25, self.cfg.seed + 1)
            source = "carved_from_train"
        fit, select = split_groups(pool, self.cfg.experiment.select_fraction, self.cfg.seed)
        return pool, fit, select, gate_set, source

    def _absorb(self, pool: Table, gate_set: Table) -> None:
        self.history_train.append(pool)
        self.history_gate.append(gate_set)

    def _baseline_anchor(self) -> float | None:
        anchor = self.baseline.anchor or {}
        if self.task.primary in anchor:
            return float(anchor[self.task.primary])
        if self.task.kind == "classification" and "accuracy" in anchor:
            return 1.0 - float(anchor["accuracy"])
        return None

    def _anchor(self, version: str, batch, pool: Table, gate_set: Table, rng) -> None:
        """记下某个版本上线时的参考窗口（本批全部样本）和它在门控集上的性能锚点。"""
        _, loss = self.task.evaluate(self.registry.model(version), gate_set)
        anchor = score_interval(self.task, loss, gate_set.groups, self.cfg.gate, rng)
        self.registry.set_reference(version, Reference.build(
            batch.name, Table.concat([pool, gate_set]), anchor, self.task, self.cfg.drift, self.cfg.seed))

    # ---- 单批处理 -------------------------------------------------------------------------------

    def _ask_llm(self, context: dict, count: int):
        usage = self.state["llm"]
        if not self.cfg.llm.enabled or count < 1 or usage["calls"] >= self.cfg.llm.max_calls:
            return [], None
        usage["calls"] += 1
        try:
            proposals, info = llm.propose(self.cfg.llm, context, count)
        except Exception as error:
            usage["failures"] += 1
            info = {"error": f"{type(error).__name__}: {error}"[:300], "fallback": "rule"}
            self.recorder.event("llm_failed", **info)
            return [], info
        usage["prompt_tokens"] += info["prompt_tokens"]
        usage["completion_tokens"] += info["completion_tokens"]
        info["proposed"] = len(proposals)
        self.recorder.event("llm_proposals", info=info, proposals=proposals)
        return proposals, info

    def _process(self, batch) -> tuple[dict, dict]:
        cfg, task, registry = self.cfg, self.task, self.registry
        primary = task.primary
        rng = np.random.default_rng([cfg.seed, batch.index])
        if not len(batch.train):
            raise ValueError("本批没有新的训练样本")
        pool, fit, select, gate_set, gate_source = self._partition(batch)
        history = Table.concat(self.history_train) if self.history_train else pool.take(slice(0, 0))
        retention = (Table.concat(self.history_gate).sample(cfg.gate.retention_rows, rng)
                     if self.history_gate else gate_set.take(slice(0, 0)))
        production = self.state["production"]
        online, static = registry.model(production), registry.model("v0")

        # 到达时刻：更新之前，在线模型与静态基线在本批验证部分上的表现
        arrival_online, _ = task.evaluate(online, gate_set)
        arrival_static, _ = task.evaluate(static, gate_set)

        # 1 漂移检测。用本批训练部分，现有的所有版本都没见过这些样本
        reference = registry.reference(production)
        if reference:
            anchor = {"score": reference.anchor_score, "high": reference.anchor_high}
        else:       # 静态基线还没有现场参考窗口时，用基线包训练报告里的验证指标
            value = self._baseline_anchor() if production == "v0" else None
            anchor = {"score": value, "high": value} if value else None
        _, loss_now = task.evaluate(online, pool)
        interval = score_interval(task, loss_now, pool.groups, cfg.gate, rng)
        inputs = detect_inputs(reference, online, pool.x, cfg.drift)
        target = detect_target(reference, online, pool.y, task, cfg.drift)
        performance = detect_performance(anchor, interval["score"], interval["ci_low"], cfg.drift)

        # 2 回滚守卫
        lineage = self.state["lineage"]
        fallbacks = ["v0"] + [v for v in lineage if v not in ("v0", production)][-cfg.gate.rollback_lookback:]
        models = {v: registry.model(v) for v in dict.fromkeys([production] + fallbacks)}
        rollback = (rollback_guard(task, cfg.gate, production, models, pool, rng) if len(models) > 1 else
                    {"triggered": False, "from": production, "to": None, "n": len(pool), "comparisons": []})
        incumbent = rollback["to"] if rollback["triggered"] else production
        incumbent_model = registry.model(incumbent)

        # 3 归因
        profile = error_profile(task, task.predict(online, pool.x), pool.y)
        diagnosis = diagnose(inputs, target, performance, profile, cfg.drift)

        # 4 是否做实验
        triggers = []
        if diagnosis["flag"]:
            triggers.append(f"漂移（{diagnosis['kind']}，{diagnosis['severity']}）")
        if rollback["triggered"]:
            triggers.append(f"在线模型 {production} 回滚到 {incumbent}")
        if not triggers and self.state["rounds_done"] < cfg.experiment.min_rounds:
            triggers.append(f"前 {cfg.experiment.min_rounds} 轮的例行实验")
        if not triggers and self.state["batches"] and self.state["batches"][-1]["action"] == "promote":
            # 上线后新分布成了参考窗口，漂移信号随之消失；刚上线的版本未必已经训到位，所以再跟进一轮
            triggers.append("上一批刚有候选上线，跟进一轮实验")
        enough = len(fit) >= 64 and len(select) >= 16 and len(gate_set) >= 16

        # 5 受控实验  6 门控
        policy = dict(self.state["policy"])
        policy_before = dict(policy)
        results, gate_report, llm_info, notes, why = [], None, None, [], []
        action, final_version = ("rollback" if rollback["triggered"] else "hold"), incumbent
        if triggers and enough:
            routine = not (diagnosis["flag"] or rollback["triggered"])
            budget = 2 if routine else (cfg.experiment.max_trials_severe if diagnosis["severity"] == "severe"
                                        else cfg.experiment.max_trials)
            proposals, llm_info = self._ask_llm({
                "task": task.kind, "primary_metric": f"{primary}，越低越好",
                "diagnosis": {k: diagnosis[k] for k in ("kind", "severity", "summary", "top_features", "error_profile")},
                "online_model": {"version": incumbent, "score_on_new_batch": interval["score"],
                                 "score_when_promoted": anchor["score"] if anchor else None},
                "base_trial": sanitize(policy, cfg.experiment), "has_history": bool(len(history)),
                "history": self.state["digest"][-4:],
                "action_space": {"data_scope": list(SCOPES), "replay_ratio": [0.1, 2.0],
                                 "learning_rate": [cfg.experiment.lr_min, cfg.experiment.lr_max],
                                 "batch_size": list(cfg.experiment.batch_sizes), "epochs": [1, cfg.experiment.max_epochs]},
            }, budget - 1)
            specs = plan_round(diagnosis, policy, self.state["digest"], has_history=bool(len(history)), budget=budget,
                               rounds_done=self.state["rounds_done"], cfg=cfg.experiment, proposals=proposals)
            deadline = time.monotonic() + cfg.experiment.round_seconds
            for spec in specs:
                if results and time.monotonic() > deadline:
                    why.append(f"超出单批实验时间预算，只完成 {len(results)}/{len(specs)} 个试验")
                    break
                results.append(run_trial(
                    spec, baseline=self.baseline, task=task, parent=incumbent, parent_path=registry.checkpoint(incumbent),
                    fit=fit, history=history, select=select, retention=retention,
                    out_dir=self.workdir / "trials" / f"b{batch.index:03d}-{spec.trial_id}", recorder=self.recorder,
                    batch=batch, diagnosis=diagnosis, cfg=cfg.experiment, seed=cfg.seed))
            judge(results, task=task, select=select, control=incumbent_model, gate_cfg=cfg.gate,
                  cfg=cfg.experiment, rng=rng)
            notes = update_policy(policy, results, cfg.experiment)
            for r in results:
                before = r["control_retention"].get(primary, 0.0)
                r["retention_loss"] = r["retention"][primary] / before - 1 if before > 0 else 0.0
            eligible = [r for r in results if r["sanity"]["ok"]]
            best = None
            if eligible:    # 选型集成绩并列（相差在 selection_slack 内）时，取历史批次上遗忘更少的那个
                top = min(r["select_score"] for r in eligible)
                close = [r for r in eligible if r["select_score"] <= top * (1 + cfg.experiment.selection_slack)]
                best = min(close, key=lambda r: (r["retention"].get(primary, 0.0), r["select_score"]))
            if best is None:
                why.append("没有试验通过完整性检查")
            elif best["rel_gain_vs_control"] <= 0:
                why.append("选型集上没有试验优于不更新的对照")
            else:
                gate_report = promotion_gate(task, cfg.gate, candidate=best["model"], incumbent=incumbent_model,
                                             static=static, gate_set=gate_set, retention=retention,
                                             sanity=best["sanity"], rng=rng)
                gate_report["candidate_trial"] = best["trial"]
                if gate_report["passed"]:
                    action = "promote"
                    final_version = registry.add(Path(best["checkpoint"]), parent=incumbent, batch=batch.name,
                                                 trial=best["trial"], gate=gate_report)
                    self._anchor(final_version, batch, pool, gate_set, rng)
                    why.append(f"{best['trial']} 通过门控，登记为 {final_version}")
                    if best["retention_loss"] > cfg.gate.retention_tol:
                        why.append(f"提示：该候选在历史批次上劣化 {best['retention_loss']:+.0%}，旧工况重现时由回滚守卫处理")
                else:
                    failed = "、".join(c["name"] for c in gate_report["checks"]
                                      if not c["passed"] and not c.get("advisory"))
                    why.append(f"{best['trial']} 未通过门控（{failed}）")
        elif triggers:
            why.append("样本不足，本批不做实验")
        else:
            why.append("未见漂移，例行实验轮数已满，本批不做实验")

        after, _ = task.evaluate(registry.model(final_version), gate_set)
        if registry.reference(final_version) is None:
            # 还没有参考窗口的版本（一开始的静态基线）：以它首次被保留上线的这一批作为参考窗口和性能锚点
            self._anchor(final_version, batch, pool, gate_set, rng)
        trials = [{k: r[k] for k in TRIAL_FIELDS} for r in results]
        decision = {
            "format": "harness.decision.v1", "batch": batch.name, "batch_index": batch.index, "action": action,
            "model": {"version": final_version, "sha256": registry.info(final_version)["sha256"],
                      "checkpoint": str(registry.checkpoint(final_version).relative_to(self.workdir))},
            "previous_version": production, "reasons": triggers + why,
            "rows": {"train_new": len(batch.train), "validation_new": len(batch.validation), "fit": len(fit),
                     "select": len(select), "gate": len(gate_set), "gate_source": gate_source,
                     "retention": len(retention), "history": len(history), "repeated": batch.repeated_rows},
            "data": {"train_sha256": batch.train_sha256, "validation_sha256": batch.validation_sha256},
            "arrival": {"online_version": production, "online": arrival_online, "static": arrival_static},
            "after": {"version": final_version, "metrics": after},
            "drift": {"inputs": {**inputs, "features": inputs["features"][:8]}, "target": target,
                      "performance": performance},
            "diagnosis": diagnosis, "rollback": rollback,
            "experiments": {"triggered_by": triggers, "trials": trials, "policy_before": policy_before,
                            "policy_after": policy, "policy_changes": notes, "llm": llm_info},
            "gate": gate_report,
            "cost": {"trials": len(results), "train_seconds": sum(r["seconds"] for r in results),
                     "sample_epochs": sum(r["sample_epochs"] for r in results)},
            "decided_at": now()}
        return decision, {"pool": pool, "gate_set": gate_set, "policy": policy}

    def _failsafe(self, batch, error: Exception) -> tuple[dict, dict]:
        """处理一批数据时出任何异常，都保持在线模型不变并照常交出决策，闭环不中断。"""
        version = self.state["production"]
        text = f"{type(error).__name__}: {error}"
        self.recorder.event("batch_failed", batch=batch.name, error=text, traceback=traceback.format_exc())
        internals = {"policy": self.state["policy"]}
        try:
            pool, _, _, gate_set, _ = self._partition(batch)
            internals.update(pool=pool, gate_set=gate_set)
        except Exception:
            pass
        return {"format": "harness.decision.v1", "batch": batch.name, "batch_index": batch.index, "action": "hold",
                "model": {"version": version, "sha256": self.registry.info(version)["sha256"],
                          "checkpoint": str(self.registry.checkpoint(version).relative_to(self.workdir))},
                "previous_version": version, "reasons": [f"处理异常，保持在线模型：{text}"], "error": text,
                "experiments": {"trials": []}, "cost": {"trials": 0, "train_seconds": 0.0, "sample_epochs": 0},
                "decided_at": now()}, internals

    def _commit(self, batch, decision: dict, internals: dict) -> None:
        state, primary = self.state, self.task.primary
        version = decision["model"]["version"]
        trials = decision["experiments"]["trials"]
        state["production"] = version
        state["lineage"] = [v for v in state["lineage"] if v != version] + [version]
        state["policy"] = internals["policy"]
        state["rounds_done"] += 1 if trials else 0
        state["trials_total"] += len(trials)
        promoted = (decision.get("gate") or {}).get("candidate_trial") if decision["action"] == "promote" else None
        for trial in trials:
            arm = state["arms"].setdefault(trial["arm"], {"tried": 0, "supported": 0, "promoted": 0, "gain_sum": 0.0})
            arm["tried"] += 1
            arm["supported"] += trial["verdict"] == "supported"
            arm["promoted"] += trial["trial"] == promoted
            arm["gain_sum"] += trial["rel_gain_vs_control"]
        diagnosis = decision.get("diagnosis") or {"kind": "unknown", "severity": "unknown", "summary": ""}
        if trials:
            state["digest"] = (state["digest"] + [{
                "batch": batch.name, "drift": diagnosis["kind"], "severity": diagnosis["severity"],
                "action": decision["action"],
                "trials": [{k: t[k] for k in ("arm", "data_scope", "learning_rate", "batch_size", "epochs",
                                              "select_score", "rel_gain_vs_control", "verdict")} for t in trials]}])[-8:]
        arrival, gate = decision.get("arrival") or {}, decision.get("gate") or {}
        drift = decision.get("drift") or {}
        row = {"batch": batch.name, "index": batch.index, "action": decision["action"],
               "version_before": decision["previous_version"], "version_after": version,
               "drift": diagnosis["kind"], "severity": diagnosis["severity"], "trials": len(trials),
               "rollback": bool((decision.get("rollback") or {}).get("triggered")),
               "arrival_online": (arrival.get("online") or {}).get(primary),
               "arrival_static": (arrival.get("static") or {}).get(primary),
               "after": ((decision.get("after") or {}).get("metrics") or {}).get(primary),
               "gate_passed": gate.get("passed"), "seconds": decision["seconds"],
               "train_seconds": decision["cost"]["train_seconds"], "sample_epochs": decision["cost"]["sample_epochs"],
               "error": decision.get("error")}
        state["batches"].append(row)
        state["next_index"] = batch.index + 1

        write_json(self.workdir / "decisions" / f"{batch.name}.json", decision)
        self.registry.publish(self.workdir / "production", version, decision)
        gain_check = next((c for c in gate.get("checks", []) if c["name"] == "gain_vs_incumbent"), {})
        inputs, performance = drift.get("inputs") or {}, drift.get("performance") or {}
        run = self.recorder.start(
            f"b{batch.index:03d}-decision", "decision", batch.name, tags=[decision["action"], diagnosis["kind"]],
            description=diagnosis["summary"], config={
                "batch": batch.name, "batch_index": batch.index, "action": decision["action"],
                "model_version": version, "model_sha256": decision["model"]["sha256"],
                "previous_version": decision["previous_version"], "reasons": decision["reasons"],
                "drift_kind": diagnosis["kind"], "drift_severity": diagnosis["severity"],
                "top_features": diagnosis.get("top_features"), "rollback": decision.get("rollback"),
                "gate_checks": {c["name"]: c["passed"] for c in gate.get("checks", [])},
                "gate_advisory": [c["name"] for c in gate.get("checks", []) if c.get("advisory")],
                "candidate_trial": gate.get("candidate_trial"), "trials": [t["trial"] for t in trials],
                "policy_changes": decision["experiments"].get("policy_changes"), "rows": decision.get("rows")})
        run.log({"drift/psi_max": inputs.get("psi_max"), "drift/psi_median": inputs.get("psi_median"),
                 "drift/n_alert": inputs.get("n_alert"), "drift/target_psi": (drift.get("target") or {}).get("psi"),
                 "drift/perf_ratio": performance.get("ratio"),
                 f"stream/arrival_online_{primary}": row["arrival_online"],
                 f"stream/arrival_static_{primary}": row["arrival_static"], f"stream/after_{primary}": row["after"],
                 "decision/code": ACTION_CODE[decision["action"]], "decision/version": int(version[1:]),
                 "decision/rollback": row["rollback"], "gate/passed": gate.get("passed"),
                 "gate/rel_gain": gain_check.get("rel_gain"), "cost/trials": len(trials),
                 "cost/train_seconds": row["train_seconds"], "cost/sample_epochs": row["sample_epochs"],
                 "cost/batch_seconds": row["seconds"],
                 "cost/llm_tokens": state["llm"]["prompt_tokens"] + state["llm"]["completion_tokens"]}, step=batch.index)
        run.finish()
        self.recorder.event("decision", batch=batch.name, action=decision["action"], version=version,
                            reasons=decision["reasons"])
        if "pool" in internals:
            self._absorb(internals["pool"], internals["gate_set"])
        write_json(self.state_path, state)
        self.stream.submit_decision(batch.name, decision)

    # ---- 整条流 ---------------------------------------------------------------------------------

    def run(self) -> dict:
        while True:
            batch = self.stream.next_batch()
            if batch is None:
                break
            if batch.index < self.state["next_index"]:      # 断点续跑：已决策的批次只用来重建历史
                pool, _, _, gate_set, _ = self._partition(batch)
                self._absorb(pool, gate_set)
                self.stream.awaiting = None
                continue
            started = time.monotonic()
            try:
                decision, internals = self._process(batch)
            except Exception as error:
                decision, internals = self._failsafe(batch, error)
            decision["seconds"] = time.monotonic() - started
            self._commit(batch, decision, internals)
            print(f"[{batch.name}] {decision['action']:8s} -> {decision['model']['version']}  " +
                  "；".join(decision["reasons"]), flush=True)
        return self.finish()

    def finish(self) -> dict:
        summary = write_report(self.workdir, self.state, self.task, self.registry, self.recorder.status())
        primary = self.task.primary
        run = self.recorder.start("stream-summary", "summary", "stream", tags=["summary"],
                                  description="整条数据流的逐批回放：在线模型与静态基线、决策与成本",
                                  config={k: v for k, v in summary.items() if k != "batches"})
        seconds = 0.0
        for row in self.state["batches"]:
            seconds += row["seconds"]
            run.log({f"stream/arrival_online_{primary}": row["arrival_online"],
                     f"stream/arrival_static_{primary}": row["arrival_static"],
                     f"stream/after_{primary}": row["after"], "decision/code": ACTION_CODE[row["action"]],
                     "decision/version": int(row["version_after"][1:]), "cost/cumulative_seconds": seconds,
                     "cost/trials": row["trials"]}, step=row["index"])
        run.finish()
        return summary

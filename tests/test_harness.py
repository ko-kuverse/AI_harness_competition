"""Harness 的单元与闭环测试。全部使用合成数据，不依赖公开数据集，几秒内跑完。

运行：python -B -m unittest discover -s tests -v
"""
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASELINE_DIR = ROOT / "cpu-c-guide-20261110-v1"

from harness import llm  # noqa: E402
from harness.bridge import load_baseline  # noqa: E402
from harness.config import load_config  # noqa: E402
from harness.data import DirectoryStream, Task  # noqa: E402
from harness.drift import ks, psi  # noqa: E402
from harness.experiment import plan_round, sanitize  # noqa: E402
from harness.gate import compare  # noqa: E402
from harness.loop import Harness  # noqa: E402

CLASSES = ["Normal", "Degrading", "SevereFault"]


def make_rows(rng, prefix, engines, cycles, offset=0.0, task="regression"):
    """合成设备数据。offset 只加在模型看到的前两个传感器读数上，真实的输入到目标关系不变。"""
    rows = []
    for unit in range(engines):
        bias = rng.normal(0, 0.2, 5)
        for cycle in range(cycles):
            x = rng.normal(0, 1, 5) + bias
            if task == "regression":
                answer = round(float(max(0.0, 60 + 20 * x[0] - 15 * x[1] + 8 * x[2] + rng.normal(0, 4))), 3)
            else:
                answer = CLASSES[int(np.digitize(x[0] + 0.6 * x[1] + rng.normal(0, 0.3), [-0.5, 0.6]))]
            seen = x.copy()
            seen[:2] += offset
            rows.append({"id": f"{prefix}_u{unit:03d}_c{cycle:05d}",
                         "inputs": {"sensors": [round(float(v), 4) for v in seen]}, "answer": answer})
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def build_world(tmp, offsets, task="regression", seed=0, cumulative=False):
    """生成静态基线检查点和一条批次流。offsets 给出每一批的传感器偏移。"""
    rng = np.random.default_rng(seed)
    tmp = Path(tmp)
    write_jsonl(tmp / "base" / "train.jsonl", make_rows(rng, "base", 40, 30, task=task))
    baseline = load_baseline(BASELINE_DIR)
    classes = CLASSES if task == "classification" else []
    data = baseline.read_dataset(tmp / "base" / "train.jsonl", task, classes)
    model = baseline.Model(task, data["paths"], classes, data["x"], data["y"], hidden=16, seed=42,
                           learning_rate=0.003, batch_size=64)
    model.meta["profile"] = "C" if task == "regression" else "B"
    model.fit(data["x"], data["y"], 40)
    (tmp / "ckpt").mkdir()
    model.save(tmp / "ckpt" / "checkpoint.npz")
    earlier_train, earlier_validation = [], []
    for k, offset in enumerate(offsets):
        train = make_rows(rng, f"b{k}t", 30, 30, offset, task)
        validation = make_rows(rng, f"b{k}v", 12, 30, offset, task)
        folder = tmp / "stream" / f"batch-{k:03d}"
        write_jsonl(folder / "train.jsonl", (earlier_train if cumulative else []) + train)
        write_jsonl(folder / "validation.jsonl", (earlier_validation if cumulative else []) + validation)
        earlier_train += train
        earlier_validation += validation
    return tmp


def config_for(tmp, **overrides):
    base = {"baseline_dir": str(BASELINE_DIR), "checkpoint": str(Path(tmp) / "ckpt" / "checkpoint.npz"),
            "stream_dir": str(Path(tmp) / "stream"), "workdir": str(Path(tmp) / "work"),
            "record": {"mode": "disabled"}, "gate": {"bootstrap": 300},
            "experiment": {"min_rounds": 1, "max_epochs": 15}}
    for key, value in overrides.items():
        base[key] = {**base[key], **value} if isinstance(base.get(key), dict) else value
    return load_config(None, base)


def decisions(tmp):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((Path(tmp) / "work" / "decisions").glob("*.json"))]


class DriftMath(unittest.TestCase):
    def test_psi_and_ks_separate_shift_from_noise(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
        self.assertLess(psi(a, b), 0.02)
        self.assertLess(ks(a, b), 0.05)
        self.assertGreater(psi(a, b + 1.0), 0.5)
        self.assertGreater(ks(a, b + 1.0), 0.3)

    def test_psi_flags_values_leaving_a_constant_reference(self):
        constant = np.full(1000, 518.67)
        self.assertLess(psi(constant, constant.copy()), 1e-9)
        self.assertGreater(psi(constant, constant + 0.5), 1.0)

    def test_paired_bootstrap_needs_a_real_difference(self):
        cfg = load_config().gate
        task = Task("regression")
        rng = np.random.default_rng(2)
        groups = np.repeat(np.arange(40).astype(str).astype(object), 50)
        noise = rng.normal(0, 1, 2000) ** 2
        better = compare(task, noise * 0.5, noise, groups, cfg, np.random.default_rng(3))
        self.assertLess(better["ci_high"], 0)
        same = compare(task, noise, rng.normal(0, 1, 2000) ** 2, groups, cfg, np.random.default_rng(3))
        self.assertGreater(same["ci_high"], 0)
        self.assertLess(same["ci_low"], 0)


class Planning(unittest.TestCase):
    def test_sanitize_clamps_to_action_space_and_rejects_garbage(self):
        cfg = load_config().experiment
        fixed = sanitize({"data_scope": "replay", "learning_rate": 5.0, "batch_size": 100, "epochs": 999}, cfg)
        self.assertEqual((fixed["learning_rate"], fixed["batch_size"], fixed["epochs"]), (cfg.lr_max, 128, cfg.max_epochs))
        for bad in ({"data_scope": "everything", "learning_rate": 0.001}, {"data_scope": "recent"},
                    {"data_scope": "recent", "learning_rate": "fast"}, {"data_scope": "recent", "learning_rate": -1}, "x"):
            self.assertIsNone(sanitize(bad, cfg))

    def test_round_plan_is_base_plus_distinct_single_factor_variants(self):
        cfg = load_config().experiment
        diagnosis = {"kind": "covariate_harmful", "severity": "severe", "meaning": "输入漂移已经损害精度"}
        specs = plan_round(diagnosis, dict(cfg.initial_policy), [], has_history=True, budget=4, rounds_done=0, cfg=cfg)
        self.assertEqual(specs[0].arm, "base")
        self.assertEqual(len(specs), 4)
        self.assertEqual(len({json.dumps(s.factors(), sort_keys=True) for s in specs}), 4)
        for spec in specs[1:]:
            changed = [k for k, v in spec.factors().items() if v != specs[0].factors()[k]]
            self.assertEqual(changed, [spec.changed_factor])

    def test_llm_reply_parsing_survives_thinking_and_braces_in_strings(self):
        text = '<think>{"not": "this"}</think> 好的：{"analysis": "a}b", "trials": [{"hypothesis": "h\\"}"}]} 完'
        self.assertEqual(llm.extract_json(text)["trials"][0]["hypothesis"], 'h"}')


class Stream(unittest.TestCase):
    def test_next_batch_is_blocked_until_decision_is_submitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_world(tmp, [0.0, 0.0])
            cfg = config_for(tmp)
            stream = DirectoryStream(cfg.stream_dir, load_baseline(cfg.baseline_dir, cfg.checkpoint), cfg.group_pattern, cfg.block_rows)
            first = stream.next_batch()
            with self.assertRaises(RuntimeError):
                stream.next_batch()
            with self.assertRaises(ValueError):
                stream.submit_decision(first.name, {"action": "deploy", "model": {"sha256": "x"}})
            stream.submit_decision(first.name, {"action": "hold", "model": {"sha256": "sha256:x"}})
            self.assertEqual(stream.next_batch().index, 1)

    def test_rows_redelivered_with_later_batches_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_world(tmp, [0.0, 0.0], cumulative=True)
            cfg = config_for(tmp)
            stream = DirectoryStream(cfg.stream_dir, load_baseline(cfg.baseline_dir, cfg.checkpoint), cfg.group_pattern, cfg.block_rows)
            first = stream.next_batch()
            stream.awaiting = None
            second = stream.next_batch()
            self.assertEqual((len(first.train), len(second.train), len(second.validation)), (900, 900, 360))
            self.assertEqual(second.repeated_rows, 900 + 360)
            self.assertFalse(set(first.train.ids) & set(second.train.ids))


class ClosedLoop(unittest.TestCase):
    def test_drift_is_detected_attributed_promoted_then_rolled_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_world(tmp, [0.0, 0.0, 2.0, 2.0, 0.0])
            harness = Harness(config_for(tmp))
            summary = harness.run()
            result = decisions(tmp)
            self.assertEqual(len(result), 5)
            self.assertEqual([d["diagnosis"]["kind"] for d in result[:2]], ["none", "none"])
            drifted = result[2]
            self.assertEqual(drifted["action"], "promote")
            self.assertEqual(drifted["diagnosis"]["kind"], "covariate_harmful")
            self.assertEqual({f["feature"] for f in drifted["diagnosis"]["top_features"]}, {"sensors[0]", "sensors[1]"})
            checks = {c["name"]: c for c in drifted["gate"]["checks"]}
            self.assertTrue(all(c["passed"] for c in checks.values() if not c.get("advisory")))
            # 适应偏移后的读数必然丢掉旧分布上的精度：retention 如实标记，但只是提示项，不拦截上线
            self.assertTrue(checks["retention"]["advisory"])
            self.assertFalse(checks["retention"]["passed"])
            self.assertLess(drifted["after"]["metrics"]["rmse"], 0.5 * drifted["arrival"]["static"]["rmse"])
            restored = result[4]
            self.assertEqual((restored["action"], restored["model"]["version"]), ("rollback", "v0"))
            self.assertTrue(restored["rollback"]["triggered"])
            self.assertGreaterEqual(summary["experiment_rounds"], 2)
            # 对外发布的检查点就是决策里那个版本，且能被官方 Model 原样加载
            published = Path(tmp) / "work" / "production" / "checkpoint.npz"
            self.assertEqual(harness.baseline.checkpoint_digest(published), restored["model"]["sha256"])
            self.assertEqual(harness.baseline.Model.load(published).meta["steps"], harness.baseline.meta["steps"])
            # 每个试验都从当时的在线模型续训，步数只增不减
            for decision in result:
                for trial in decision["experiments"]["trials"]:
                    self.assertGreater(trial["sanity"]["steps_after"], trial["sanity"]["steps_before"])
            self.assertTrue((Path(tmp) / "work" / "report.md").is_file())

    def test_no_drift_after_routine_rounds_means_no_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_world(tmp, [0.0, 0.0, 0.0])
            Harness(config_for(tmp)).run()
            result = decisions(tmp)
            self.assertGreater(len(result[0]["experiments"]["trials"]), 0)      # 例行实验
            self.assertEqual(result[0]["action"], "hold")
            for decision in result[1:]:
                self.assertEqual((decision["action"], decision["experiments"]["trials"]), ("hold", []))

    def test_resume_continues_after_last_decided_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_world(tmp, [0.0, 2.0, 2.0])
            later = Path(tmp) / "stream" / "batch-002"
            parked = Path(tmp) / "parked"
            later.rename(parked)
            Harness(config_for(tmp)).run()
            before = {d["batch"]: d["decided_at"] for d in decisions(tmp)}
            self.assertEqual(sorted(before), ["batch-000", "batch-001"])
            parked.rename(later)
            harness = Harness(config_for(tmp))
            harness.run()
            after = decisions(tmp)
            self.assertEqual(len(after), 3)
            self.assertEqual({d["batch"]: d["decided_at"] for d in after[:2]}, before)
            self.assertEqual(after[2]["rows"]["history"], 1800)     # 续跑时前两批被重新装入历史

    def test_failure_inside_a_batch_falls_back_to_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_world(tmp, [0.0, 2.0, 2.0])
            harness = Harness(config_for(tmp))
            original = harness._process

            def flaky(batch):
                if batch.index == 1:
                    raise RuntimeError("injected failure")
                return original(batch)

            harness._process = flaky
            summary = harness.run()
            result = decisions(tmp)
            self.assertEqual((result[1]["action"], result[1]["model"]["version"]), ("hold", "v0"))
            self.assertIn("injected failure", result[1]["error"])
            self.assertEqual(summary["failed_batches"], 1)
            self.assertEqual(result[2]["action"], "promote")        # 下一批照常处理

    def test_classification_stream_runs_the_same_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_world(tmp, [0.0, 1.5, 1.5], task="classification")
            Harness(config_for(tmp)).run()
            result = decisions(tmp)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[1]["action"], "promote")
            self.assertLess(result[1]["after"]["metrics"]["error_rate"], result[1]["arrival"]["static"]["error_rate"])
            self.assertIn("macro_f1", result[1]["after"]["metrics"])

    def test_swanlab_offline_writes_one_run_per_trial_decision_and_summary(self):
        if importlib.util.find_spec("swanlab") is None:
            self.skipTest("swanlab 未安装")
        with tempfile.TemporaryDirectory() as tmp:
            build_world(tmp, [0.0, 2.0])
            harness = Harness(config_for(tmp, record={"mode": "offline"}))
            summary = harness.run()
            self.assertEqual(summary["swanlab"], "offline")
            runs = list((Path(tmp) / "work" / "swanlog").glob("run-*"))
            self.assertEqual(len(runs), summary["trials_total"] + summary["batches"] + 1)
            events = [json.loads(line) for line in (Path(tmp) / "work" / "records" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(e["kind"] == "run_started" for e in events), len(runs))


class FakeChat(BaseHTTPRequestHandler):
    reply = ""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeChat.last_request = body
        payload = json.dumps({"choices": [{"message": {"content": FakeChat.reply}}],
                              "usage": {"prompt_tokens": 321, "completion_tokens": 45}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class LLMPlanner(unittest.TestCase):
    def test_proposals_are_constrained_and_outage_falls_back_to_rules(self):
        FakeChat.reply = "<think>先想一想</think>" + json.dumps({"analysis": "两个传感器整体偏移", "trials": [
            {"hypothesis": "偏移幅度大，需要更大的步长", "data_scope": "replay", "learning_rate": 0.5, "epochs": 12},
            {"hypothesis": "越界的提议", "data_scope": "delete_everything", "learning_rate": 0.001}]}, ensure_ascii=False)
        server = HTTPServer(("127.0.0.1", 0), FakeChat)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/v1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                build_world(tmp, [0.0, 2.0])
                harness = Harness(config_for(tmp, llm={"enabled": True, "base_url": url, "model": "fake"}))
                harness.run()
                trials = decisions(tmp)[1]["experiments"]["trials"]
                from_llm = [t for t in trials if t["source"] == "llm"]
                self.assertEqual(len(from_llm), 1)                                  # 越界的那条被丢弃
                self.assertEqual(from_llm[0]["learning_rate"], harness.cfg.experiment.lr_max)   # 学习率被夹到上限
                self.assertEqual(trials[0]["arm"], "base")
                self.assertEqual(harness.state["llm"]["prompt_tokens"], 321 * harness.state["llm"]["calls"])
                self.assertEqual(FakeChat.last_request["model"], "fake")
        finally:
            server.shutdown()
            server.server_close()
        with tempfile.TemporaryDirectory() as tmp:      # 服务不可达：照常跑完，全部走规则
            build_world(tmp, [0.0, 2.0])
            harness = Harness(config_for(tmp, llm={"enabled": True, "base_url": url, "model": "fake", "timeout": 2}))
            harness.run()
            result = decisions(tmp)
            self.assertEqual(result[1]["action"], "promote")
            self.assertTrue(all(t["source"] == "rule" for d in result for t in d["experiments"]["trials"]))
            self.assertEqual(harness.state["llm"]["failures"], harness.state["llm"]["calls"])


if __name__ == "__main__":
    unittest.main()

"""命令行入口：python -m harness {simulate,run,score}"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m harness", description="工业模型自进化 Harness 最小闭环")
    commands = parser.add_subparsers(dest="command", required=True)

    sim = commands.add_parser("simulate", help="用公开数据生成本地漂移流")
    sim.add_argument("--public", required=True, help="含 train.jsonl 与 validation.jsonl 的目录")
    sim.add_argument("--out", required=True)
    sim.add_argument("--scenario", choices=("regime", "sensor-aging"), default="regime")
    sim.add_argument("--seed", type=int, default=7)

    def common(sub) -> None:
        sub.add_argument("--config", help="JSON 配置文件")
        sub.add_argument("--baseline-dir", help="官方基线包目录")
        sub.add_argument("--checkpoint", help="静态基线检查点，缺省为基线包里的 checkpoint.npz")
        sub.add_argument("--workdir", help="状态、注册表、决策与记录的输出目录")

    run = commands.add_parser("run", help="在一条批次流上运行闭环")
    common(run)
    run.add_argument("--stream", help="批次流目录，每个子目录一批")
    run.add_argument("--swanlab-mode", choices=("offline", "online", "local", "disabled"))
    run.add_argument("--project", help="SwanLab 项目名")
    run.add_argument("--llm", action="store_true", help="启用大模型规划器（PYTRIO_BASE_URL、PYTRIO_MODEL、PYTRIO_API_KEY）")

    sc = commands.add_parser("score", help="用留出数据给各批决策打分")
    common(sc)
    sc.add_argument("--hidden", required=True, help="simulate 生成的 hidden 目录")

    args = parser.parse_args()
    if args.command == "simulate":
        from .simulate import simulate
        truth = simulate(args.public, args.out, args.scenario, seed=args.seed)
        for batch in truth["batches"]:
            print(batch["name"], batch["rows"], "offset_sigma", batch["injected_offset_sigma"])
        return

    overrides: dict = {}
    for key, value in (("baseline_dir", args.baseline_dir), ("checkpoint", args.checkpoint), ("workdir", args.workdir)):
        if value:
            overrides[key] = value
    if args.command == "run":
        stream = args.stream or os.environ.get("HARNESS_STREAM_DIR")
        if stream:
            overrides["stream_dir"] = stream
        record = {k: v for k, v in (("mode", args.swanlab_mode or os.environ.get("HARNESS_SWANLAB_MODE")),
                                    ("project", args.project)) if v}
        if record:
            overrides["record"] = record
        model = {k: v for k, v in (("base_url", os.environ.get("PYTRIO_BASE_URL")),
                                   ("model", os.environ.get("PYTRIO_MODEL"))) if v}
        if args.llm or os.environ.get("HARNESS_LLM") == "1":
            model["enabled"] = True
        if model:
            overrides["llm"] = model
    cfg = load_config(args.config, overrides)

    from .loop import Harness
    if args.command == "run":
        if not cfg.stream_dir:
            parser.error("需要 --stream 或环境变量 HARNESS_STREAM_DIR")
        if cfg.llm.enabled and not cfg.llm.base_url:
            parser.error("启用大模型规划器需要设置 PYTRIO_BASE_URL 或在配置里写 llm.base_url")
        summary = Harness(cfg).run()
        print(json.dumps({k: summary[k] for k in ("batches", "actions", "rollbacks", "experiment_rounds", "trials_total",
                                                  "final_version", "mean_rel_gain_after_decision",
                                                  "mean_rel_gain_at_arrival", "seconds_total", "swanlab")},
                         ensure_ascii=False, indent=2))
        print(f"报告：{Path(cfg.workdir) / 'report.md'}")
    else:
        from .bridge import load_baseline
        from .data import Task
        from .registry import Registry
        from .report import score
        baseline = load_baseline(cfg.baseline_dir, cfg.checkpoint)
        result = score(Path(cfg.workdir), Path(args.hidden), baseline, Task(baseline.task, baseline.meta["classes"]),
                       Registry(Path(cfg.workdir) / "registry", baseline), cfg)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))
        print(f"明细：{Path(cfg.workdir) / 'score.md'}")


if __name__ == "__main__":
    main()

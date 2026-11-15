"""可选的大模型规划器：通过 OpenAI 兼容接口向 PyTRiO 上的驱动模型要本轮的附加试验。

只用标准库发请求。任何失败（超时、返回格式不对、提议越界）都退回规则规划，闭环照常运行。
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

SYSTEM = """你是工业模型持续训练系统的实验规划器。系统每收到一批新数据，会先做漂移检测和归因，再从在线模型续训若干候选，最后由门控决定是否上线。
你的任务：阅读漂移诊断和历史试验结果，提出本轮除基准试验之外要做的对照试验。每个试验要有一条可被选型集结果证实或证伪的假设。
只能在给定动作空间内取值。只输出一个 JSON 对象，格式：
{"analysis": "一两句话的判断", "trials": [{"hypothesis": "...", "data_scope": "recent|replay|cumulative", "replay_ratio": 0.5, "learning_rate": 0.001, "batch_size": 256, "epochs": 8}]}"""


def extract_json(text: str) -> dict:
    """去掉思考段落，取第一个配平的 JSON 对象。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("{")
    while start >= 0:
        depth, in_string, escape = 0, False, False
        for position in range(start, len(text)):
            char = text[position]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:position + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("回复里没有可解析的 JSON 对象")


def propose(cfg, context: dict, count: int) -> tuple[list, dict]:
    """返回（试验提议列表，调用信息）。提议还没有经过动作空间约束，调用方负责 sanitize。"""
    body = {"model": cfg.model, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"请提出至多 {count} 个试验。上下文：\n" +
                          json.dumps(context, ensure_ascii=False)}], **cfg.extra_body}
    request = urllib.request.Request(
        cfg.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ.get(cfg.api_key_env, "")})
    with urllib.request.urlopen(request, timeout=cfg.timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    parsed = extract_json(content)
    trials = parsed.get("trials")
    if not isinstance(trials, list):
        raise ValueError("回复缺少 trials 列表")
    usage = payload.get("usage") or {}
    return [t for t in trials if isinstance(t, dict)][:count], {
        "analysis": str(parsed.get("analysis", ""))[:500], "model": cfg.model,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)), "completion_tokens": int(usage.get("completion_tokens", 0))}

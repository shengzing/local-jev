#!/usr/bin/env python3
"""
Local Jev — 核心决策客户端
复现 Jev 式固定答案打分：不生成文本，只读第一个 token 的 logits 做 softmax。

用法:
    python decide.py                          # 默认工单路由示例
    python decide.py --ticket "退款没到账"      # 自定义工单

需要先启动 SGLang:
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 30000
"""

import argparse
import json
import sys
from typing import Optional

import requests

# ── 配置 ──────────────────────────────────────────────────────────────
BASE_URL = "http://127.0.0.1:30000"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# 默认候选选项（可按场景自定义）
DEFAULT_CHOICES = {
    "A": "billing and payments",
    "B": "technical support",
    "C": "account access",
}
OTHER_LABEL = "OTHER"  # 兜底选项：当已列选项都不对时

# ── Prompt 构建 ────────────────────────────────────────────────────────

def build_prompt(ticket: str, choices: dict[str, str]) -> str:
    """构建决策 prompt，末尾以 'Label:' 结束，让模型下一个 token 应该是标签。"""
    choice_lines = "\n".join(
        f"{label} = {meaning}" for label, meaning in choices.items()
    )
    return f"""Ticket:
{ticket}

Question:
Which category matches the ticket?

Allowed labels:
{choice_lines}

Return only the label.

Label: """


# ── 标签验证 ───────────────────────────────────────────────────────────

def resolve_label_token_ids(choices: dict[str, str], base_url: str, model: str) -> list[int]:
    """
    用 SGLang /tokenize 验证每个标签是否恰好是单 token。
    拒绝非单 token 标签，因为打分需要一个词表位置。
    """
    token_ids = []
    for label in choices:
        resp = requests.post(
            f"{base_url}/tokenize",
            json={
                "model": model,
                "prompt": label,
                "add_special_tokens": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        ids = resp.json()["tokens"]
        if len(ids) != 1:
            raise ValueError(
                f"Label {label!r} is not a single token: {ids}. "
                f"Use a different label that maps to exactly one token."
            )
        print(f"  {label!r} -> token_id {ids[0]}")
        token_ids.append(ids[0])
    return token_ids


# ── 打分请求 ───────────────────────────────────────────────────────────

def score(
    prompt: str,
    label_token_ids: list[int],
    base_url: str = BASE_URL,
    model: str = MODEL,
) -> list[float]:
    """
    调用 SGLang /v1/score，返回每个标签的概率（已 softmax）。
    不生成任何 token，只读取 logits 并归一化。
    """
    resp = requests.post(
        f"{base_url}/v1/score",
        json={
            "model": model,
            "query": prompt,
            "items": [""],          # 对 prompt 末尾的下一个位置打分
            "label_token_ids": label_token_ids,
            "apply_softmax": True,  # 在选定位置做受限 softmax
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["scores"][0]


# ── 决策映射 ───────────────────────────────────────────────────────────

def decide(
    ticket: str,
    choices: Optional[dict[str, str]] = None,
    base_url: str = BASE_URL,
    model: str = MODEL,
) -> dict:
    """
    完整决策流程：构建 prompt → 验证标签 → 打分 → 返回决策 + 概率分布。

    返回:
        {
            "decision": "billing and payments",
            "probabilities": {
                "billing and payments": 0.6777,
                "technical support": 0.3108,
                "account access": 0.0113
            },
            "threshold_passed": True/False   # 是否超过默认阈值 0.70
        }
    """
    choices = choices or DEFAULT_CHOICES
    base_url = base_url.rstrip("/")

    print("[1/3] 构建 prompt...")
    prompt = build_prompt(ticket, choices)
    print(f"  Prompt 末尾: ...{prompt[-80:]}")

    print("[2/3] 验证标签 token...")
    token_ids = resolve_label_token_ids(choices, base_url, model)

    print("[3/3] 打分...")
    scores = score(prompt, token_ids, base_url, model)

    probabilities = {
        choices[label]: float(s)
        for label, s in zip(choices, scores, strict=True)
    }
    decision = max(probabilities, key=probabilities.get)
    top_prob = probabilities[decision]

    THRESHOLD = 0.70
    threshold_passed = top_prob >= THRESHOLD

    result = {
        "decision": decision,
        "probabilities": probabilities,
        "threshold_passed": threshold_passed,
    }
    if not threshold_passed:
        result["warning"] = (
            f"Top probability {top_prob:.4f} < threshold {THRESHOLD}. "
            f"Consider sending for human review."
        )

    return result


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Local Jev — 用 SGLang 做固定答案打分决策"
    )
    parser.add_argument(
        "--ticket", default="I was charged twice for the same subscription.",
        help="输入工单/查询文本",
    )
    parser.add_argument("--base-url", default=BASE_URL, help="SGLang 服务地址")
    parser.add_argument("--model", default=MODEL, help="模型名称")
    args = parser.parse_args()

    print(f"模型: {args.model}")
    print(f"工单: {args.ticket}\n")

    result = decide(args.ticket, base_url=args.base_url, model=args.model)
    print("\n" + json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

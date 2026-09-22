#!/usr/bin/env python3
"""
Local Jev — 工具函数
Prompt 构建、标签验证、结果格式化。
"""

import json
import requests
from typing import Optional


def build_decision_prompt(
    state: str,
    question: str,
    choices: dict[str, str],
    other_label: str = "OTHER",
) -> str:
    """
    构建决策 prompt。
    choices: {"A": "billing", "B": "technical support", ...}
    返回以 'Label: ' 结尾的 prompt，让模型下一个 token 应该是标签字母。
    """
    choice_lines = "\n".join(f"{k} = {v}" for k, v in choices.items())
    if other_label:
        choice_lines += f"\n{other_label} = none of the above"
    return f"""Context:
{state}

Question:
{question}

Allowed labels:
{choice_lines}

Return only the label.

Label: """


def validate_label_tokens(
    choices: dict[str, str],
    base_url: str = "http://127.0.0.1:30000",
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict[str, int]:
    """
    验证每个标签是否恰好是单 token。
    返回 {label: token_id} 映射。
    非单 token 标签会 raise ValueError。
    """
    result = {}
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
                f"Label {label!r} is not single token: {ids}. "
                f"Consider using a different label."
            )
        result[label] = ids[0]
    return result


def format_result(
    decision: str,
    probabilities: dict[str, float],
    threshold: float = 0.70,
) -> str:
    """格式化输出结果"""
    top_prob = probabilities[decision]
    status = "✓ AUTO" if top_prob >= threshold else "⚠ REVIEW"

    lines = [f"\n{'='*50}", f"Decision: {decision}", f"Confidence: {top_prob:.4f}", f"Status: {status}", f"{'='*50}", "\nProbabilities:"]
    for choice, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        lines.append(f"  {choice:30s} {prob:.4f} {bar}")

    if top_prob < threshold:
        lines.append(f"\n⚠ Top probability {top_prob:.4f} < threshold {threshold}")
        lines.append("  Recommend human review.")

    return "\n".join(lines)


def load_dataset(path: str) -> list[dict]:
    """加载 JSONL 数据集"""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases

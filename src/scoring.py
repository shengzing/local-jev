#!/usr/bin/env python3
"""
Local Jev — 受限 softmax 核心数学
打分路径唯一的数学步骤：在候选 logits 间归一化。
"""

import math


def constrained_softmax(logits: list[float]) -> list[float]:
    """
    受限 softmax：只在给定 logits 间归一化，忽略词表其余位置。
    与 SGLang /v1/score 的 apply_softmax=True 行为一致。
    """
    if not logits:
        return []
    max_logit = max(logits)
    exp_vals = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_vals)
    return [v / sum_exp for v in exp_vals]

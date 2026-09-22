#!/usr/bin/env python3
"""
Local Jev — 双路径引擎
封装 scoring（Jev 式打分）和 generation（常规文本生成）两种推理路径，
用于对比延迟和准确率。
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import requests


@dataclass
class DecisionResult:
    """打分路径返回结果"""
    decision: str
    probabilities: dict[str, float]
    latency_ms: float


@dataclass
class GenerationResult:
    """生成路径返回结果"""
    response: str
    parsed_choice: Optional[str]
    latency_ms: float


class LocalJevEngine:
    """双路径推理引擎，连接同一个 SGLang 服务"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000",
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._label_cache: dict[str, list[int]] = {}

    def _build_prompt(self, state: str, question: str, choices: dict[str, str]) -> str:
        choice_lines = "\n".join(f"{k} = {v}" for k, v in choices.items())
        return f"""Context:
{state}

Question:
{question}

Allowed labels:
{choice_lines}

Return only the label.

Label: """

    def _resolve_token_ids(self, choices: dict[str, str]) -> list[int]:
        """验证并缓存标签 token IDs"""
        cache_key = "".join(choices.keys())
        if cache_key in self._label_cache:
            return self._label_cache[cache_key]

        token_ids = []
        for label in choices:
            resp = requests.post(
                f"{self.base_url}/tokenize",
                json={
                    "model": self.model,
                    "prompt": label,
                    "add_special_tokens": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            ids = resp.json()["tokens"]
            if len(ids) != 1:
                raise ValueError(f"Label {label!r} is not single token: {ids}")
            token_ids.append(ids[0])

        self._label_cache[cache_key] = token_ids
        return token_ids

    def decide(
        self,
        state: str,
        question: str,
        choices: dict[str, str],
    ) -> DecisionResult:
        """
        Jev 式打分路径：只读 logits，不生成文本。
        """
        prompt = self._build_prompt(state, question, choices)
        token_ids = self._resolve_token_ids(choices)

        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/v1/score",
            json={
                "model": self.model,
                "query": prompt,
                "items": [""],
                "label_token_ids": token_ids,
                "apply_softmax": True,
            },
            timeout=120,
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000

        scores = resp.json()["scores"][0]
        probabilities = {
            choices[label]: float(s)
            for label, s in zip(choices, scores)
        }
        decision = max(probabilities, key=probabilities.get)

        return DecisionResult(
            decision=decision,
            probabilities=probabilities,
            latency_ms=latency_ms,
        )

    def generate_response(
        self,
        state: str,
        question: str,
        choices: list[tuple[str, str]],
        max_tokens: int = 32,
    ) -> GenerationResult:
        """
        常规生成路径：让模型生成文本，后解析选项。
        """
        choice_lines = "\n".join(f"{k} = {v}" for k, v in choices)
        messages = [
            {
                "role": "system",
                "content": f"You are a classifier. Choose exactly one label.\n\nLabels:\n{choice_lines}\n\nReply with only the label letter.",
            },
            {"role": "user", "content": f"Context: {state}\n\nQuestion: {question}"},
        ]

        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000

        response_text = resp.json()["choices"][0]["message"]["content"]

        # 在前 100 字符内查找匹配的选项
        parsed = None
        search_text = response_text[:100].lower()
        for label, meaning in choices:
            if label.lower() in search_text or meaning.lower() in search_text:
                parsed = meaning
                break

        return GenerationResult(
            response=response_text,
            parsed_choice=parsed,
            latency_ms=latency_ms,
        )

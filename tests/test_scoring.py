#!/usr/bin/env python3
"""
Local Jev — 纯函数单元测试
不依赖模型服务，只测 prompt 构建、受限 softmax、结果格式化。
运行: pytest tests/ -v
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring import constrained_softmax
from src.utils import build_decision_prompt, load_dataset

CHOICES = {
    "A": "billing and payments",
    "B": "technical support",
    "C": "account access",
}


# ── build_decision_prompt ──────────────────────────────────────

class TestBuildPrompt:

    def test_ends_with_label_prefix(self):
        prompt = build_decision_prompt("ticket text", "Which?", CHOICES)
        assert prompt.endswith("Label: ")

    def test_contains_all_choices(self):
        prompt = build_decision_prompt("ticket text", "Which?", CHOICES)
        for label, meaning in CHOICES.items():
            assert f"{label} = {meaning}" in prompt

    def test_contains_state_and_question(self):
        state = "I was charged twice."
        question = "Which category matches?"
        prompt = build_decision_prompt(state, question, CHOICES)
        assert state in prompt
        assert question in prompt

    def test_other_label_appended_by_default(self):
        prompt = build_decision_prompt("t", "q", CHOICES)
        assert "OTHER = none of the above" in prompt

    def test_other_label_can_be_disabled(self):
        prompt = build_decision_prompt("t", "q", CHOICES, other_label=None)
        assert "OTHER" not in prompt

    def test_single_letter_labels_one_per_line(self):
        # Jev 式打分要求每个标签独立一行，格式 "X = 语义"
        prompt = build_decision_prompt("t", "q", CHOICES)
        lines = [ln for ln in prompt.splitlines() if " = " in ln]
        assert len(lines) == 4  # A/B/C + OTHER


# ── constrained_softmax ────────────────────────────────────────

class TestConstrainedSoftmax:

    def test_output_sums_to_one(self):
        probs = constrained_softmax([25.28, 24.50, 21.19])
        assert math.isclose(sum(probs), 1.0, rel_tol=1e-12)

    def test_empty_input(self):
        assert constrained_softmax([]) == []

    def test_single_logit_gives_probability_one(self):
        assert constrained_softmax([3.7]) == [1.0]

    def test_higher_logit_gets_higher_prob(self):
        probs = constrained_softmax([3.0, 1.0])
        assert probs[0] > probs[1]

    def test_numerical_stability_with_large_logits(self):
        # 大 logits 直接 exp 会溢出；实现必须先减 max
        probs = constrained_softmax([10000.0, 9999.0])
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert math.isclose(sum(probs), 1.0, rel_tol=1e-12)

    def test_ignores_rest_of_vocab(self):
        # 受限 softmax 只在给定 logits 间归一化，与其他 token 无关
        a = constrained_softmax([2.0, 1.0])
        b = constrained_softmax([2.0, 1.0, -1e9])
        assert math.isclose(a[0], b[0], rel_tol=1e-9)
        assert math.isclose(a[1], b[1], rel_tol=1e-9)

    def test_order_preserved(self):
        probs = constrained_softmax([5.0, 3.0, 4.0])
        assert probs[0] > probs[2] > probs[1]

    def test_known_value(self):
        # 原文 CUDA logits: billing=25.2776, technical=24.4982, account=21.1888
        probs = constrained_softmax([25.277620, 24.498226, 21.188837])
        assert probs[0] == pytest.approx(0.677762, abs=1e-4)
        assert probs[1] == pytest.approx(0.310879, abs=1e-4)
        assert probs[2] == pytest.approx(0.011359, abs=1e-4)

    def test_uniform_logits_give_uniform_probs(self):
        probs = constrained_softmax([7.0, 7.0, 7.0, 7.0])
        assert probs == pytest.approx([0.25, 0.25, 0.25, 0.25])


# ── load_dataset ───────────────────────────────────────────────

class TestLoadDataset:

    def test_loads_jsonl(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text(
            '{"a": 1}\n{"a": 2}\n',
            encoding="utf-8",
        )
        cases = load_dataset(str(path))
        assert cases == [{"a": 1}, {"a": 2}]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text(
            '{"a": 1}\n\n   \n{"a": 2}\n',
            encoding="utf-8",
        )
        cases = load_dataset(str(path))
        assert len(cases) == 2

    def test_real_datasets_are_valid(self):
        # 仓库自带的数据集必须可加载且字段完整
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("support_routing", "candidate_screening", "expense_review"):
            cases = load_dataset(os.path.join(root, "datasets", f"{name}.jsonl"))
            assert len(cases) > 0
            for case in cases:
                assert set(case) == {"state", "question", "expected"}

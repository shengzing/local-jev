# Local Jev — Reproducing Jev-Style Decision Engine with Open-Source LLMs

> An open-source project reproducing Jev's core inference pattern: instead of generating text, read the first token's logits and apply softmax over known candidate labels to return a probability distribution.

**Author: Jiacheng Bin (jiacb@wiseweb.com.cn)**

English | [中文](./README.md)

## What Is This

[Jev](https://jev.ts.ai) is a commercial decision model built on a simple insight: **many LLM calls don't need text generation at all — they just need to pick from known options.**

This project reproduces that inference path entirely with open-source tooling:

1. Map candidate answers to single-letter labels (A/B/C)
2. Have the model process the prompt, read only the first token position's logits
3. Apply restricted softmax over the candidate token positions
4. Return a probability distribution — zero text generated

The core code is under 100 lines, depending only on SGLang + requests.

## Quick Start

### Option A: SGLang Server (Full /v1/score Endpoint)

```bash
# 1. Install SGLang
python3 -m venv .venv
source .venv/bin/activate
pip install "sglang[all]==0.5.10.post1" requests

# 2. Start model server (auto-downloads model on first run)
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 --port 30000

# 3. Run decision
python src/decide.py
```

Output:

```json
{
  "decision": "billing and payments",
  "probabilities": {
    "billing and payments": 0.6777,
    "technical support": 0.3108,
    "account access": 0.0113
  }
}
```

### Option B: MLX Native (Apple Silicon, No SGLang Required)

```bash
pip install mlx-lm transformers
python verify_mlx.py
```

### Option C: PyTorch + transformers

```bash
pip install transformers torch
python verify.py
```

## Core Principle

### Fixed-Answer Scoring ≠ Structured Output

| Feature | Structured Output (JSON) | Fixed-Answer Scoring (Jev-style) |
|---------|---------------------------|----------------------------------|
| Output form | `{"team": "billing"}` | `billing: 0.91, technical: 0.06, account: 0.03` |
| Inference | Token-by-token generation | Read first token's logits only |
| Information returned | A single choice | Full probability distribution |
| Latency | High (autoregressive loop) | Low (single forward pass) |

### Why Single-Letter Labels

`billing` might be 1 token or multiple tokens depending on the tokenizer; `technical support` almost certainly spans multiple token positions.

Single-letter labels (A/B/C) guarantee each candidate maps to exactly one vocabulary position, while the full semantic meaning is still present in the prompt:

```
A = billing questions and payment problems
B = product errors and technical failures
C = login, password, and account access problems
```

The model reads the full descriptions when processing the prompt. The label is just the token whose logit we check afterward.

### Inference Flow

```
User input ──→ Build prompt (with candidate labels)
                    │
            Verify labels are single tokens (/tokenize)
                    │
            Forward pass, read logits + softmax (/v1/score)
                    │
            Map back to semantic options ──→ Return decision + distribution
```

## Benchmark Results (Apple M4, Qwen2.5-0.5B-Instruct)

| Metric | Scoring Path | Generation Path |
|--------|-------------|----------------|
| Avg latency | 37.7 ms | 112.1 ms |
| Speed factor | — | 3.0x slower |
| Token IDs | A=[32] B=[33] C=[34] | Matches original exactly |
| Original logits | billing=25.28, technical=24.50, account=21.19 | — |
| MLX logits | billing=24.25, technical=24.25, account=21.13 | — |

> Note: MLX fp16 precision causes billing/technical logits to be close, but the core mechanism is fully consistent. Scoring is 3x faster than generation.

## Project Structure

```
local-jev/
├── README.md                    ← Chinese docs
├── README_EN.md                 ← English docs (this file)
├── LICENSE                      ← MIT
├── pyproject.toml               ← Package metadata
├── requirements.txt             ← Dependencies
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── decide.py                ← Core decision client
│   ├── engine.py                ← Dual-path engine (scoring + generation)
│   └── utils.py                 ← Prompt building, label validation
├── benchmark/
│   ├── run_benchmark.py         ← Scoring vs generation comparison
│   └── results/                 ← Benchmark results
├── datasets/
│   ├── support_routing.jsonl    ← Support ticket routing
│   ├── candidate_screening.jsonl← Candidate screening
│   └── expense_review.jsonl     ← Expense review
├── verify.py                    ← PyTorch/MPS verification
├── verify_mlx.py                ← MLX verification (Apple Silicon native)
├── article/
│   └── local-jev-deep-dive.md   ← Technical deep-dive article
└── docs/
    ├── architecture.md           ← Architecture details
    └── calibration.md           ← Calibration & evaluation
```

## When to Use Scoring vs Generation

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Ticket routing, intent detection | Scoring | Finite candidates, just picking |
| Content moderation, risk grading | Scoring | Enumerated classes, distribution has business value |
| Summarization, translation, code gen | Generation | Output content is unpredictable |
| Explanatory judgments | Generation | Needs natural language output |

**Core criterion**: If the application knows all possible answers before inference, use scoring; if the output content is unpredictable, use generation.

## Acknowledgments

- Original concept by [Avi Chawla](https://x.com/_avichawla)
- [SGLang](https://github.com/sgl-project/sglang) for the `/v1/score` endpoint
- This project is independently implemented and not affiliated with Jev or TypeSafe

## License

MIT — Copyright (c) 2026 Jiacheng Bin (贾承斌)

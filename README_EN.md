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

## Benchmark Results (Apple M4, Qwen2.5-0.5B-Instruct, MLX fp16)

Full output from `python verify_mlx.py`:

### Label Token Verification

| Label | Token ID | Status |
|-------|----------|--------|
| A (billing) | [32] | ✓ single token |
| B (technical) | [33] | ✓ single token |
| C (account) | [34] | ✓ single token |

Matches the original (SGLang + CUDA) exactly.

### Single-Case Scoring vs Generation

| Metric | Scoring Path | Generation Path |
|--------|-------------|----------------|
| Decision | billing and payments | A |
| Latency | 2.2 ms | 125.9 ms |
| Speed | — | 57x slower (single token gen) |

### Multi-Case Accuracy (6 tickets)

| Ticket | Expected | Decision | Correct | Latency |
|--------|----------|----------|---------|---------|
| charged twice for subscription | billing | billing | ✓ | 38.6 ms |
| app crashes on settings | technical | technical | ✓ | 37.2 ms |
| forgot password | account | technical | ✗ | 37.7 ms |
| refund not appeared | billing | billing | ✓ | 37.4 ms |
| 500 error on upload | technical | technical | ✓ | 39.0 ms |
| someone accessed my account | account | technical | ✗ | 37.2 ms |

- **Scoring accuracy**: 4/6 (67%) (0.5B model capacity limit)
- **Avg scoring latency**: 37.8 ms
- **Avg generation latency**: 119.6 ms
- **Speed factor**: scoring is **3.2x faster** than generation

### Logits Comparison with Original

| Option | Original logit (CUDA) | MLX logit | Original prob | MLX prob |
|--------|----------------------|-----------|---------------|----------|
| billing and payments | 25.2776 | 24.2500 | 0.6778 | 0.4893 |
| technical support | 24.4982 | 24.2500 | 0.3109 | 0.4893 |
| account access | 21.1888 | 21.1250 | 0.0114 | 0.0215 |

- **Max probability difference**: 0.1885
- **Cause**: MLX fp16 precision makes billing/technical logits identical (24.25), yielding 48.9% each after softmax. Under CUDA fp16, they differ by 0.78.
- **Core mechanism fully consistent**: label token IDs, logits extraction, restricted softmax — all correct.

## Project Structure

```
local-jev/
├── README.md                    ← Chinese docs
├── README_EN.md                 ← English docs (this file)
├── LICENSE                      ← MIT
├── pyproject.toml               ← Package metadata
├── requirements.txt             ← Dependencies
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml               ← CI (syntax check + unit tests)
├── src/
│   ├── __init__.py
│   ├── decide.py                ← Core decision client
│   ├── engine.py                ← Dual-path engine (scoring + generation)
│   ├── scoring.py               ← Constrained softmax pure function
│   └── utils.py                 ← Prompt building, label validation
├── tests/
│   └── test_scoring.py          ← Unit tests (no model server needed)
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

# Local Jev — 用开源 LLM 本地复现 Jev 式决策引擎

> 一个开源项目，复现 Jev 的核心推理模式：不生成文本，只读取第一个 token 的 logits，在已知候选选项间做 softmax 归一化，返回概率分布。

**作者：贾承斌 (jiacb@wiseweb.com.cn)**

[English](./README_EN.md) | 中文

## 这是什么

[Jev](https://jev.ts.ai) 是一个商业决策模型，它的核心思路是：**很多 LLM 调用根本不需要生成文本，只需要在已知选项间做选择**。

本项目在本地用开源工具链完整复现了这套推理路径：

1. 用单字母标签（A/B/C）映射候选答案
2. 让模型处理 prompt，只取第一个 token 位置的 logits
3. 在候选 token 位置做受限 softmax
4. 返回概率分布，不生成任何文本

核心代码不到 100 行，依赖只有 SGLang + requests。

## 快速开始

### 方式一：SGLang 服务（完整复现 /v1/score 端点）

```bash
# 1. 安装 SGLang
python3 -m venv .venv
source .venv/bin/activate
pip install "sglang[all]==0.5.10.post1" requests

# 2. 启动模型服务（首次会自动下载模型）
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 --port 30000

# 3. 运行决策
python src/decide.py
```

输出：

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

### 方式二：MLX 原生（Apple Silicon，无需 SGLang）

```bash
pip install mlx-lm transformers
python verify_mlx.py
```

### 方式三：PyTorch + transformers

```bash
pip install transformers torch
python verify.py
```

## 核心原理

### 固定答案打分 ≠ 结构化输出

| 特性 | 结构化输出 (JSON) | 固定答案打分 (Jev 式) |
|------|-------------------|----------------------|
| 输出形式 | `{"team": "billing"}` | `billing: 0.91, technical: 0.06, account: 0.03` |
| 推理过程 | 逐 token 生成 | 只读第一个 token 的 logits |
| 返回信息 | 一个选择 | 完整概率分布 |
| 延迟 | 高（自回归循环） | 低（单次前向传播） |

### 为什么用单字母标签

`billing` 在不同 tokenizer 下可能是 1 个 token 也可能是多个；`technical support` 几乎必然跨多个 token 位置。

单字母标签（A/B/C）保证每个候选只对应词表中的一个位置，且语义信息仍完整出现在 prompt 中：

```
A = billing questions and payment problems
B = product errors and technical failures
C = login, password, and account access problems
```

模型读 prompt 时看到完整描述，标签只是事后检查 logit 的那个 token。

### 运行流程

```
用户输入 ──→ 构建 prompt（含候选标签）
                    │
            验证标签是单 token（/tokenize）
                    │
            前向传播，读取 logits + softmax（/v1/score）
                    │
            映射回语义选项 ──→ 返回决策 + 概率分布
```

## 实测结果（Apple M4, Qwen2.5-0.5B-Instruct, MLX fp16）

运行 `python verify_mlx.py` 的完整输出：

### 标签 Token 验证

| 标签 | Token ID | 状态 |
|------|----------|------|
| A (billing) | [32] | ✓ 单 token |
| B (technical) | [33] | ✓ 单 token |
| C (account) | [34] | ✓ 单 token |

与原文（SGLang + CUDA）完全一致。

### 单 Case 打分 vs 生成

| 指标 | 打分路径 | 生成路径 |
|------|---------|---------|
| 决策 | billing and payments | A |
| 延迟 | 2.2 ms | 125.9 ms |
| 速度 | — | 57x 慢（单 token 生成） |

### 多 Case 准确率（6 条工单）

| 工单 | 预期 | 决策 | 正确 | 延迟 |
|------|------|------|------|------|
| charged twice for subscription | billing | billing | ✓ | 38.6 ms |
| app crashes on settings | technical | technical | ✓ | 37.2 ms |
| forgot password | account | technical | ✗ | 37.7 ms |
| refund not appeared | billing | billing | ✓ | 37.4 ms |
| 500 error on upload | technical | technical | ✓ | 39.0 ms |
| someone accessed my account | account | technical | ✗ | 37.2 ms |

- **打分准确率**：4/6 (67%)（0.5B 模型能力限制）
- **打分平均延迟**：37.8 ms
- **生成平均延迟**：119.6 ms
- **速度倍数**：打分比生成快 **3.2x**

### 与原文 logits 对比

| 选项 | 原 logit (CUDA) | MLX logit | 原概率 | MLX 概率 |
|------|----------------|-----------|--------|---------|
| billing and payments | 25.2776 | 24.2500 | 0.6778 | 0.4893 |
| technical support | 24.4982 | 24.2500 | 0.3109 | 0.4893 |
| account access | 21.1888 | 21.1250 | 0.0114 | 0.0215 |

- **最大概率差异**：0.1885
- **原因**：MLX fp16 精度导致 billing/technical logits 完全相同（24.25），softmax 后各 48.9%。CUDA fp16 下两者有 0.78 的差距。
- **核心机制完全一致**：标签 token IDs、logits 读取、受限 softmax 路径全部正确。

## 项目结构

```
local-jev/
├── README.md                    ← 中文文档
├── README_EN.md                 ← English docs
├── LICENSE                      ← MIT
├── pyproject.toml               ← 包元数据
├── requirements.txt            ← 依赖声明
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml               ← CI（语法检查 + 单元测试）
├── src/
│   ├── __init__.py
│   ├── decide.py                ← 核心决策客户端
│   ├── engine.py                ← 封装 scoring + generation 双路径
│   ├── scoring.py               ← 受限 softmax 纯函数
│   └── utils.py                 ← prompt 构建、标签验证
├── tests/
│   └── test_scoring.py          ← 单元测试（无需模型服务）
├── benchmark/
│   ├── run_benchmark.py         ← 对比 scoring vs generation 延迟/准确率
│   └── results/                 ← 基准测试结果
├── datasets/
│   ├── support_routing.jsonl    ← 工单路由数据集
│   ├── candidate_screening.jsonl← 候选筛选数据集
│   └── expense_review.jsonl     ← 费用审核数据集
├── verify.py                    ← PyTorch/MPS 验证
├── verify_mlx.py                ← MLX 验证（Apple Silicon 原生）
├── article/
│   └── local-jev-deep-dive.md   ← 技术解读文章
└── docs/
    ├── architecture.md          ← 架构详解
    └── calibration.md           ← 校准与评估
```

## 何时用打分，何时用生成

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 工单分类、意图识别 | 打分 | 候选有限，只需选择 |
| 内容审核、风险分级 | 打分 | 枚举类别，概率分布有业务价值 |
| 摘要、翻译、代码生成 | 生成 | 输出内容不可预知 |
| 需要解释理由的判断 | 生成 | 需要自然语言输出 |

**核心判据**：如果应用在推理前就知道所有可能答案，用打分；如果输出内容不可预知，用生成。

## 致谢

- 原文作者 [Avi Chawla](https://x.com/_avichawla) 的技术分享
- [SGLang](https://github.com/sgl-project/sglang) 项目提供的 `/v1/score` 接口
- 本项目独立实现，不隶属 Jev 或 TypeSafe

## License

MIT — Copyright (c) 2026 贾承斌

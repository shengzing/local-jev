# 用开源LLM在本地复现Jev：100%本地决策引擎全解析

> 本文从第三方技术视角，深度解读 Avi Chawla 分享的"本地复现 Jev"方案。保留原文所有配图位置，代码经整理可直接运行。

## 背景：很多 LLM 调用，其实不需要"生成文本"

最近，AI 开发者 Avi Chawla 在社交媒体分享了一套方案——如何把一个开源 LLM，不经过任何重新训练，改造成一个本地、高速的"决策引擎"，其效果对标的是目前颇受关注的商业产品 Jev。

先看一个典型场景：一张工单写着"我的订阅被重复扣费了"，系统需要把它分到三个团队之一——账单、技术支持、还是账号权限。

常规做法是让 LLM"写"一个答案——可能是一句话、一个标签、或者一个 JSON 对象，应用程序再去解析这段文本，提取出选中的团队。

[图像]

但如果所有可能的答案本来就是已知的、有限的，让模型"写作"其实是多余的开销。

更高效的思路，是把这类请求当作一次"决策"而非"生成"。应用程序把工单内容和三个候选答案一起交给模型，模型一次性返回每个候选项的概率分数：

```
billing           0.91
technical support 0.06
account access    0.03
```

billing 以最高概率胜出，而且应用程序还能清楚地看到这个选择"赢"得有多明显。这正是 Jev 展现的核心行为——本文要做的，就是在完全本地、开源的环境下把这套推理模式复现出来。

虽然 Jev 本身是闭源产品，但这种推理模式已经在多个开源语言模型上可用。具体来说，可以借助 SGLang 框架提供的 `/v1/score` 接口，在 Qwen、DeepSeek 等模型上跑通。

[图像]

需要提前说明：这里复现的是 Jev 的推理路径，不是它的完整系统——Jev 背后还包含训练和校准工作，这些是单纯打分接口给不了的。

## 一、"固定答案打分"和"结构化输出"，是两码事

很多人容易把这种打分机制和"结构化输出（structured output）"搞混，因为两者最终都会限制应用程序拿到的结果类型，但它们在推理服务器内部做的事情完全不同。

同样是那张工单，结构化输出可能会让模型输出：

```json
{"team": "billing"}
```

指定的 schema 确保了输出不会是非法对象，但这不代表模型是"选择"出这个团队的。其底层过程是：模型依然要一个 token 一个 token 地生成左花括号、字段名、值、右花括号，生成完成后，应用程序再去读取 `team` 字段的值。

[视频位置：0:00 / 0:08]

而打分模式（也就是 Jev 的做法）则不同：应用程序把三个团队作为完整的候选答案集合提供给模型，服务器读取每个候选项对应的一个分数，直接返回概率分布，全程不生成任何 JSON 对象。

[图像]

这样一来，应用可以拿到完整的三个数值，而不只是一个"billing"：

```
结果1                    结果2
billing           0.91  billing           0.46
technical support 0.06  technical support 0.44
account access    0.03  account access    0.10
```

两种情况下，模型最终选择的都是 billing，但含义完全不同——第一种是压倒性优势，第二种几乎势均力敌。应用程序完全可以据此设定规则：第一种直接自动路由，第二种则转人工复核。

而且，模型只负责给出分数，下游怎么用这些分数，规则完全由应用代码决定——比如可以要求最高分必须超过 0.80，且领先第二名至少 0.20。这类阈值写在代码里，可以随时测试、调整。

[图像]

需要澄清一点：0.91 的含义是"billing 在这三个候选项中占据了 91% 的概率质量"，不代表模型有 91% 的概率是对的。要衡量模型的实际准确率，还需要有标注数据做校准，这是后文会提到的校准问题。

简单总结：结构化输出生成的是一个合法的对象；固定答案打分返回的，是应用程序已知候选集合上的一个概率分布。

## 二、先搞懂 LLM 怎么"生成第一个 token"

要理解如何把一个普通的因果语言模型（causal LLM）改造成 Jev 式的打分模型，得先弄清楚常规生成过程中第一步发生了什么。

[图像]

流程大致是：

1. Tokenizer 把输入的 prompt 转换成 token ID 序列
2. 模型处理这个序列，为下一个位置生成一个向量
3. 这个向量里的每一个数值，对应词表中的一个 token——词表有多大，向量就有多长
4. 这些原始数值叫作 logits，数值越大代表模型越倾向于选择这个 token 作为下一个输出，但它们此时还不是概率

在正常生成过程中，服务器会根据解码规则（比如温度参数等）处理这个向量，选出一个 token，拼接到 prompt 后面，然后模型再为下一个位置生成新的向量……如此循环，直到遇到停止符或达到输出长度上限。

[图像]

但如果我们只是想做一次"有限选项的决策"，其实只需要关心第一个向量就够了。

回到工单分类的例子，假设应用支持三个答案：billing、technical support、account access。在 prompt 中给每个答案分配一个简短的字母标签：

```
A = billing
B = technical support
C = account access
```

prompt 最后要求模型只返回一个标签：

```
Route the support ticket into exactly one category.
Ticket:
I was charged twice for the same subscription.
Allowed labels:
A = billing
B = technical support
C = account access
Return only the label.
Label:
```

prompt 以"Label:"结尾，这意味着模型下一步本该生成的，正是 A、B 或 C 当中的一个。模型处理完这段 prompt 后，会照常在这个位置输出一个完整的词表大小的向量——这个向量里既包含 A 的 logit，也包含 B、C 的 logit，以及词表中其它所有 token 的 logit。

这时，打分逻辑只需要做四步操作：

[图像]

1. 找到 A、B、C 分别对应的 token ID
2. 从词表向量中，读出这三个位置上的 logit
3. 忽略其余所有的 logit
4. 只对这三个数值做 softmax 归一化

假设选中的三个 logit 分别是 8.2、5.5、4.8，限定范围内的 softmax 会得到大约 0.91、0.06、0.03，再把这三个位置映射回 billing、technical support、account access 即可。

这里的归一化只在声明过的候选项内部进行——我们问的不是"A 在整个词表范围内的概率是多少"，而是"在应用已经排除了所有其他答案的前提下，模型在 A、B、C 之间是如何分配偏好的"。

[图像]

这正是 SGLang 框架已经内置实现的能力——`/v1/score` 接口。它会跑一遍 prompt，读取指定 token 位置的分数并返回，省去了我们自己修改 Qwen 模型实现、手动提取最终张量的麻烦。

**为什么用 A、B、C 而不是直接对"billing"打分？** 因为一个看起来完整的词，不一定就是一个 token。比如"billing"在某个 tokenizer 下可能是一个 token，在另一个 tokenizer 下可能被拆成好几个；"technical support"这种短语几乎肯定会跨越多个 token 位置。

单字母标签巧妙地绕开了这个问题：每个候选项在同一个输出位置上，都只对应词表里的一个条目，语义信息依然体现在 prompt 里：

```
A = billing questions and payment problems
B = product errors and technical failures
C = login, password, and account access problems
```

模型在处理 prompt 时会读取到这些完整的描述，标签本身只是之后要检查 logit 的那个 token。

不过仍然需要验证每个标签确实是单一 token——因为 tokenizer 经常把"前导空格"也编码进 token 本身，"A"和" A"可能对应不同的 token ID。

[图像]

此外，chat template 有时也会在答案位置前插入空白符或控制 token。正确的做法是：用模型自身的 chat template 渲染出完整 prompt，确定答案位置期望的确切续写内容，把这段续写发给 `/tokenize` 接口检查——如果返回的不是恰好一个 token，就拒绝使用这个标签。

这套标签映射逻辑完全封装在打分客户端内部：应用程序只需要传入"billing""technical_support"这样的语义选项，它永远不会接触到 A、B、C。这就是 API 与模型标签解耦的含义。

最后，答案列表还需要一个"逃生通道"：当已列选项都不合适时。比如一个安全事件到达了只支持 billing/technical/account access 的路由器，受限 softmax 仍然会把所有概率质量分配给这三个错误选项。解法是加入 `OTHER` 或 `ESCALATE` 标签作为兜底。

## 三、用 SGLang 搭建本地打分端点

整个复现只需要一个推理服务。SGLang 把 Qwen 加载到 GPU 内存，暴露原生 HTTP 端点。我们的 Python 脚本直接发请求即可。

完整流程：

1. 用 SGLang 启动 Qwen
2. 用字母标签编写决策 prompt
3. 让 SGLang 对标签做 tokenize
4. 发一个请求到 `/v1/score`
5. 把返回的概率映射回选项

[图像]

### 第 1 步：用 SGLang 启动 Qwen

创建 Python 环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "sglang[all]==0.5.10.post1" "requests==2.34.2"
```

然后启动模型服务：

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 30000
```

首次启动会从 Hugging Face 下载模型，之后复用本地缓存。加载完成后，Qwen 常驻内存，SGLang 监听 30000 端口。

### 第 2 步：定义选项并构建 prompt

创建 `decide.py`：

```python
import json
import requests

BASE_URL = "http://127.0.0.1:30000"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

choices = {
    "A": "billing and payments",
    "B": "technical support",
    "C": "account access",
}

ticket = "I was charged twice for the same subscription."

choice_lines = "\n".join(
    f"{label} = {meaning}" for label, meaning in choices.items()
)

prompt = f"""Ticket:
{ticket}

Question:
Which category matches the ticket?

Allowed labels:
{choice_lines}

Return only the label.

Label: """
```

字典同时记录了标签的两个表示：A 是打分的 token，"billing" 是返回给应用的语义含义。它们的顺序在 tokenize、打分和结果映射中必须保持一致。

prompt 末尾停在"Label:"——我们要的是模型在这个位置本应生成的下一个 token 的分数，而不是让 SGLang 真的生成它。

### 第 3 步：解析标签的 token ID

```python
label_token_ids = []
for label in choices:
    response = requests.post(
        f"{BASE_URL}/tokenize",
        json={
            "model": MODEL,
            "prompt": label,
            "add_special_tokens": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    token_ids = response.json()["tokens"]
    if len(token_ids) != 1:
        raise ValueError(
            f"{label!r} is not a single token: {token_ids}"
        )
    print(f"{label!r} -> {token_ids}")
    label_token_ids.append(token_ids[0])
```

SGLang 的 `/tokenize` 端点返回 Qwen tokenizer 产生的整数 ID。我们拒绝任何被拆成多个 token 的标签，因为打分需要每个答案对应一个词表位置。

对于 `Qwen/Qwen2.5-0.5B-Instruct`，输出：

```
'A' -> [32]
'B' -> [33]
'C' -> [34]
```

每个列表只有一个整数，确认每个标签占一个 token。后续打分请求将读取词表向量中第 32、33、34 位置。

### 第 4 步：请求三个概率

```python
response = requests.post(
    f"{BASE_URL}/v1/score",
    json={
        "model": MODEL,
        "query": prompt,
        "items": [""],
        "label_token_ids": label_token_ids,
        "apply_softmax": True,
    },
    timeout=120,
)
response.raise_for_status()
score_response = response.json()
print(json.dumps(score_response, indent=2))
scores = score_response["scores"][0]
```

- `query`：完整 prompt
- `items: [""]`：对 prompt 末尾的下一个位置打分
- `label_token_ids`：告诉 SGLang 读取词表输出的哪三个位置
- `apply_softmax`：把选定位置的 logits 归一化为概率

返回结果（实测）：

```json
{
  "scores": [
    [0.67776233, 0.310878605, 0.011359035]
  ]
}
```

三个值依次对应 A、B、C。softmax 之前的原始 logits 分别是 25.28、24.50、21.19。不同硬件和精度设置可能有细微差异。

### 第 5 步：映射回决策

```python
probabilities = {
    choices[label]: float(score)
    for label, score in zip(choices, scores, strict=True)
}
decision = max(probabilities, key=probabilities.get)

print(json.dumps({
    "decision": decision,
    "probabilities": probabilities,
}, indent=2))
```

运行脚本：

```json
{
  "decision": "billing and payments",
  "probabilities": {
    "billing and payments": 0.6777623295783997,
    "technical support": 0.31087860465049744,
    "account access": 0.011359035037457943
  }
}
```

billing 赢了，但只有 0.678 的概率。如果策略要求 0.70，这张工单会被送去人工复核而不是自动路由。这个阈值应该来自标注数据的评估。

## 四、打分 vs 生成的延迟对比

单次请求展示了机制。为了测试多决策场景下的表现，作者还搭建了一个小型应用，对比打分路径和生成路径。

[图像]

测试覆盖多个开源模型：Qwen 3 4B、Qwen 2.5 0.5B 和 1.5B、SmolLM2 1.7B、TinyLlama 1.1B、DeepSeek-R1-Distill-Qwen 1.5B。

[图像]

**Jev 式路径**调用 decision 方法：

```python
result = engine.decide(request)
answer = result["answers"]["decision"]
choice = answer["choice"]
probabilities = answer["probabilities"]
```

`decide()` 内部，SGLang 通过 `/v1/score` 收到 prompt，请求包含 A/B/C 的 token ID。SGLang 跑一遍 prompt，读取这三个 next-token 分数，归一化后停止。响应不含任何生成 token。

**常规生成路径**调用 generation 方法：

```python
result = engine.generate_response(
    case["state"],
    case["question"],
    list(case["criteria"].items()),
    max_tokens=32,
)
```

这个方法把同样的 state、question 和选项发给 `/v1/chat/completions`。Qwen 生成答案和简短解释，最多 32 个 token。应用在前 100 字符内搜索匹配的选项名，匹配则判对。

[视频位置：0:02 / 0:30]

速度差异的原因，前面已经讨论过了——打分只做一次前向传播，生成需要自回归循环。

作者还用 100 个固定本地数据集的 case 做了模拟。

[图像]

当前数据集覆盖工单路由、候选筛选和费用审核。它们的预期标签让界面能同时展示速度和正确率。

两个 worker 通过 Barrier 同时起跑：

```python
starting_line = threading.Barrier(2)

def worker(lane, runner):
    starting_line.wait()
    for case in cases:
        updates.put(runner(case))

workers = [
    threading.Thread(target=worker, args=("jev", run_jev)),
    threading.Thread(target=worker, args=("llm", run_llm)),
]
for worker_thread in workers:
    worker_thread.start()
```

Barrier 同时释放两个 worker。每条路径按顺序处理各自的 case。Jev 路径在前一个请求完成后立即发送下一个打分请求，生成路径同理。

[图像]

两条路径同时保持活跃，但不是一次性发送全部 200 个请求。SGLang 接收来自两条路径的并发工作，通过连续批处理调度。两种请求类型共享同一块 GPU、内存带宽和调度器。

[视频位置：0:02 / 0:28]

注：视频在 8 秒后加速，所以 LLM 请求看起来完成得很快。

两条路径同时起步，使用同一个 SGLang 服务上的 `Qwen/Qwen3-4B-Instruct-2507`。

## 五、如何选择正确的路径

打分不是生成的替代品，Jev 式方法只适用于应用在推理前就定义了输出空间的场景。

理想的兼容工作负载应具备：
- 一个有限的有意义标签集合
- 每个标签映射到一个不同的下游操作
- 调用者只需要标签和概率分布，不需要新文本

结构化输出不同——它定义的是响应的语法，解码器仍然逐 token 生成字段名和值。

当值不能事先枚举时，用结构化生成；打分只消除了已知候选值时的解码步骤。

[视频位置：0:02 / 0:10]

一个简单的判断方法：

- **输出内容不可预知 → 生成**
- **选项集合已知，只需选择 → 打分**——这时第一个 next-token 向量已经包含了排序信息，返回排序就避免了调用者不需要的自回归循环。

再次强调，这个项目复现了 Jev 式的推理机制，但没有复现 Jev 的权重、RLCD 过程或评估栈。作者表示后续会覆盖这些内容。

## 项目开源

本文对应的开源项目已发布在 GitHub：`local-jev`。项目包含完整的决策客户端、双路径引擎、基准测试脚本和三个示例数据集，MIT 协议，可一键复现文中所有实验。

**项目结构：**

| 模块 | 文件 | 职责 |
|------|------|------|
| 决策客户端 | `src/decide.py` | 构建 prompt、验证标签、调用 /v1/score |
| 双路径引擎 | `src/engine.py` | 封装 scoring + generation 对比 |
| 工具函数 | `src/utils.py` | prompt 构建、结果格式化、数据集加载 |
| 基准测试 | `benchmark/run_benchmark.py` | 双线程并行延迟/准确率对比 |
| 数据集 | `datasets/*.jsonl` | 工单路由、候选筛选、费用审核 |

**一行运行：**

```bash
python src/decide.py --ticket "I was charged twice for the same subscription."
```

---

*本文基于 Avi Chawla 的技术分享整理，原文链接见其社交媒体。项目独立实现，不隶属 Jev 或 TypeSafe。*

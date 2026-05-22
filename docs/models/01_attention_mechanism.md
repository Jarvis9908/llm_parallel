# 注意力机制详解

## 概述

注意力机制（Attention Mechanism）是现代深度学习中最核心的组件之一，也是 Transformer 架构的基石。其核心思想是：在处理序列数据时，模型不应平等对待所有输入，而应根据当前任务的需要，动态地"关注"输入中更重要的部分。

从发展历程来看，注意力机制最早应用于机器翻译中的序列到序列模型（Bahdanau Attention, 2014），随后在 2017 年的《Attention is All You Need》中被提炼为自注意力（Self-Attention）机制，并由此催生了整个 Transformer 家族。

本文将系统梳理注意力机制的直觉理解、数学原理、算法变体、代码实现及实践考量。

## 直觉理解

### 读书划重点的类比

想象你在读一篇长文章准备考试。你不会逐字逐句地同等对待所有内容——你会用荧光笔划出关键句子，对重要段落反复阅读，而略过不重要的过渡句。注意力机制做的正是这件事：**在海量信息中聚焦关键部分**。

### 查询-键-值的类比

注意力机制借鉴了数据库检索的思想：

| 概念 | 数据库类比 | 注意力机制含义 |
|------|-----------|---------------|
| Query (Q) | 搜索关键词 | 当前位置"想找什么" |
| Key (K) | 数据库索引 | 每个位置"能提供什么" |
| Value (V) | 数据库记录 | 每个位置的实际内容 |

Q 与 K 的相似度决定了"应该多关注哪个位置"，相似度高的位置对应的 V 会被赋予更大的权重。

## 数学原理

### 缩放点积注意力（Scaled Dot-Product Attention）

给定查询矩阵 $Q$、键矩阵 $K$、值矩阵 $V$，注意力计算公式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**逐步推导：**

1. **计算注意力得分**：$S = QK^T$，其中 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，结果 $S \in \mathbb{R}^{n \times m}$。每个元素 $S_{ij} = q_i \cdot k_j$ 衡量第 $i$ 个查询与第 $j$ 个键的相似度。

2. **缩放**：$S' = S / \sqrt{d_k}$。当 $d_k$ 较大时，点积的方差也会增大（假设 Q、K 各元素独立，方差为 $d_k$），导致 softmax 进入梯度极小的饱和区。除以 $\sqrt{d_k}$ 将方差归一化为 1，保证梯度稳定。

3. **Softmax 归一化**：$A = \text{softmax}(S')$，对每一行独立做 softmax，使注意力权重和为 1。

4. **加权求和**：$O = AV$，用注意力权重对值矩阵加权求和，得到最终输出。

### 多头注意力（Multi-Head Attention, MHA）

单头注意力只有一组 Q/K/V 投影，表达能力有限。多头注意力让模型同时关注不同子空间的信息：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个头：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

投影矩阵 $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^K \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$，$W^O \in \mathbb{R}^{hd_v \times d_{model}}$。

通常 $d_k = d_v = d_{model} / h$，这样多头计算的总参数量与单头相当。

**多头的意义**：不同头可以关注不同类型的模式——一个头关注语法关系，另一个关注语义关系，还有一个关注位置邻近关系。

### 分组查询注意力（Grouped-Query Attention, GQA）

GQA 是 MHA 和 MQA 的折中方案。设查询头数为 $n_{heads}$，KV 头数为 $n_{kv\_heads}$：

- **MHA**：$n_{kv\_heads} = n_{heads}$，每个查询头有独立的 KV
- **GQA**：$n_{kv\_heads} < n_{heads}$，每 $g = n_{heads} / n_{kv\_heads}$ 个查询头共享一组 KV
- **MQA**：$n_{kv\_heads} = 1$，所有查询头共享一组 KV

GQA 的计算：

$$\text{head}_i = \text{Attention}(QW_i^Q, K_{\lfloor i/g \rfloor}W_{\lfloor i/g \rfloor}^K, V_{\lfloor i/g \rfloor}W_{\lfloor i/g \rfloor}^V)$$

**参数效率分析**：KV 头数减少直接降低了 KV Cache 的显存占用。以 LLaMA-2 70B 为例，从 MHA（64头）切换到 GQA（8个 KV 头），KV Cache 显存减少约 8 倍。

### 多查询注意力（Multi-Query Attention, MQA）

MQA 是 GQA 的极端情况，所有查询头共享同一组 K 和 V：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW^K, VW^V), \quad i = 1, ..., h$$

MQA 的优势在于推理时 KV Cache 极小，劣势是可能损失模型质量。

### KV Cache

**动机**：自回归生成时，第 $t$ 步的计算需要第 $1$ 到 $t-1$ 步的所有 K 和 V。如果每步都重新计算，计算量是 $O(t^2)$。

**实现**：将每步计算得到的 K 和 V 缓存起来，新步只需计算当前 token 的 Q/K/V，然后将新的 K/V 追加到缓存中：

```
# 伪代码
new_k = W_k(x_t)          # [1, d_k]
new_v = W_v(x_t)          # [1, d_v]
cached_k = cat(cached_k, new_k, dim=0)  # [t, d_k]
cached_v = cat(cached_v, new_v, dim=0)  # [t, d_v]
output = attention(q_t, cached_k, cached_v)
```

**显存分析**：KV Cache 的显存占用为 $2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times seq\_len$。对于长序列，KV Cache 可能成为显存瓶颈。

### Flash Attention

**问题**：标准注意力的显存开销为 $O(N^2)$（需存储完整的 $N \times N$ 注意力矩阵），且对 GPU HBM 的读写次数多。

**核心思想**：分块计算（Tiling），将 Q、K、V 分成小块，在 SRAM（片上高速缓存）中完成注意力计算，避免将完整的注意力矩阵写入 HBM。

**算法要点**：

1. 将 Q 分成 $B_r$ 行的块，K、V 分成 $B_c$ 行的块
2. 对每个 Q 块和 K/V 块，在 SRAM 中计算局部注意力
3. 使用在线 softmax 技巧（数值稳定的增量计算），逐步累积输出
4. 只需 $O(N)$ 的额外显存存储 softmax 归一化因子

**效果**：显存从 $O(N^2)$ 降至 $O(N)$，计算速度提升 2-4 倍（减少 HBM 读写）。

## 算法流程

### 标准 Multi-Head Attention 流程

```
输入: X ∈ R^{n×d_model}
1. Q = X @ W_Q, K = X @ W_K, V = X @ W_V   # 线性投影
2. 将 Q, K, V 重塑为 [n_heads, n, d_head]
3. S = Q @ K^T / sqrt(d_head)                # 计算注意力得分
4. A = softmax(S, dim=-1)                    # 归一化
5. O = A @ V                                 # 加权求和
6. 将 O 重塑为 [n, d_model]
7. Output = O @ W_O                          # 输出投影
```

### GQA 流程

```
输入: X ∈ R^{n×d_model}
1. Q = X @ W_Q  → [n_heads, n, d_head]
2. K = X @ W_K  → [n_kv_heads, n, d_head]    # 更少的 KV 头
3. V = X @ W_V  → [n_kv_heads, n, d_head]
4. 将 K, V 扩展到 n_heads 个头（重复 g 次）
5. 执行标准 MHA 计算
```

## 代码实现

本项目注意力机制的实现位于 `models/common/attention.py`，主要包含以下组件：

```python
# 核心类结构示意
class ScaledDotProductAttention:
    """缩放点积注意力"""

class MultiHeadAttention:
    """多头注意力，支持 MHA / GQA / MQA"""

class FlashAttention:
    """Flash Attention 封装"""
```

关键实现细节：

- GQA 通过 `n_kv_heads` 参数控制，当 `n_kv_heads < n_heads` 时自动启用 KV 共享
- KV Cache 通过 `kv_cache` 参数传入，支持增量更新
- Flash Attention 在支持的 GPU 上自动启用

详细代码参见：[`models/common/attention.py`](../../models/common/attention.py)

### 注意力变体的演进时间线

| 年份 | 方法 | 关键改进 | 代表模型 |
|------|------|---------|---------|
| 2017 | MHA | 多头并行注意力 | 原始 Transformer |
| 2019 | MQA | 所有头共享 KV | PaLM |
| 2023 | GQA | 分组共享 KV | LLaMA 2/3 |
| 2024 | MLA | KV 低秩压缩 | DeepSeek V2/V3 |
| 2022 | Flash Attention | 分块计算减少显存 | 广泛采用 |
| 2023 | Sliding Window | 限制注意力窗口 | Mistral |

### 自注意力 vs 交叉注意力

| 类型 | Q 来源 | K/V 来源 | 典型用途 |
|------|--------|---------|---------|
| 自注意力（Self-Attention） | 输入自身 | 输入自身 | 编码器、解码器 |
| 交叉注意力（Cross-Attention） | 解码器 | 编码器输出 | Encoder-Decoder |
| 掩码自注意力（Masked Self-Attention） | 已生成 token | 已生成 token | Decoder-Only |

### 注意力计算复杂度分析

| 操作 | 计算量 | 显存 |
|------|--------|------|
| Q/K/V 投影 | $O(n \cdot d^2)$ | $O(n \cdot d)$ |
| 注意力得分 $QK^T$ | $O(n^2 \cdot d)$ | $O(n^2)$ |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| 加权求和 $AV$ | $O(n^2 \cdot d)$ | $O(n \cdot d)$ |
| 输出投影 | $O(n \cdot d^2)$ | $O(n \cdot d)$ |
| **总计** | $O(n^2 \cdot d + n \cdot d^2)$ | $O(n^2 + n \cdot d)$ |

当 $n \gg d$ 时，注意力得分计算 $O(n^2 \cdot d)$ 是瓶颈；当 $d \gg n$ 时，线性投影 $O(n \cdot d^2)$ 是瓶颈。

## 实践考量

### MHA vs GQA vs MQA 的选择

| 方案 | KV Cache 大小 | 模型质量 | 推理速度 | 适用场景 |
|------|-------------|---------|---------|---------|
| MHA | 最大 | 最好 | 最慢 | 追求质量的中小模型 |
| GQA | 中等 | 接近 MHA | 较快 | 大模型的主流选择 |
| MQA | 最小 | 略有损失 | 最快 | 对推理速度要求极高 |

**实践建议**：
- 7B 以下模型：MHA 即可，KV Cache 不是瓶颈
- 7B-70B 模型：GQA（8 个 KV 头是常见选择）
- 70B+ 模型：GQA 或 MQA，配合 KV Cache 量化

### KV Cache 显存优化

1. **GQA/MQA**：减少 KV 头数，直接降低缓存大小
2. **KV Cache 量化**：将 FP16 的 KV Cache 量化为 INT8 或 INT4
3. **PagedAttention**：将 KV Cache 分页管理，避免显存碎片
4. **Sliding Window Attention**：只缓存最近 W 个 token 的 KV

### 注意力掩码

- **因果掩码（Causal Mask）**：解码器中，防止看到未来 token
- **填充掩码（Padding Mask）**：批处理中，忽略 padding 位置
- **滑动窗口掩码**：限制注意力范围，降低计算量

**因果掩码的实现**：

```python
# 下三角掩码矩阵
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
# 应用掩码
scores = scores.masked_fill(mask, float('-inf'))
```

因果掩码确保位置 $i$ 只能关注位置 $0, 1, ..., i$，不能看到未来的信息。这是 Decoder-Only 模型自回归生成的基础。

### 注意力温度调节

除了标准的缩放因子 $\sqrt{d_k}$，还可以引入温度参数 $\tau$ 来调节注意力的"锐度"：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\tau}\right)V$$

- $\tau < \sqrt{d_k}$：注意力更集中（更"确定"），倾向于关注少数位置
- $\tau > \sqrt{d_k}$：注意力更分散（更"均匀"），对所有位置更平等

温度调节在知识蒸馏和采样策略中有重要应用。

## 与其他技术的关系

- **Transformer**：注意力机制是 Transformer 的核心组件，详见 [Transformer 架构详解](./02_transformer_architecture.md)
- **位置编码**：注意力本身是位置无关的，需要位置编码注入位置信息，详见 [位置编码详解](./05_positional_encoding.md)
- **LLaMA 3**：使用 GQA 优化推理效率，详见 [LLaMA 3 架构详解](./03_llama3_architecture.md)
- **DeepSeek V3**：使用 MLA 进一步压缩 KV Cache，详见 [DeepSeek V3 详解](./04_deepseek_v3_architecture.md)
- **归一化层**：注意力输出后通常接归一化层，详见 [归一化层详解](./06_normalization.md)

## 参考资料

1. Vaswani, A., et al. "Attention is All You Need." NeurIPS 2017.
2. Shazeer, N. "Fast Transformer Decoding: One Write-Head is All You Need." arXiv 2019. (MQA)
3. Ainslie, J., et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023.
4. Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.
5. Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv 2023.
6. Korthikanti, V., et al. "Reducing Activation Recomputation in Large Transformer Models." MLSys 2023.

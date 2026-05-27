# Attention 机制详解

> 上一篇：[模型架构总览](overview.md) ｜ 下一篇：[Transformer](transformer.md)

## 概述

Attention（注意力）机制是现代 LLM 的核心组件，让模型能够动态地关注输入序列中的不同位置。本篇详解从基础的 Scaled Dot-Product Attention 到 Multi-Head Attention、Grouped Query Attention 的完整演进。

**前置知识：** 张量运算、线性变换（nn.Linear）、Softmax
**代码位置：** [`models/common/attention.py`](../../models/common/attention.py)

## 核心原理

### Scaled Dot-Product Attention

给定 Query (Q)、Key (K)、Value (V) 三个矩阵，Attention 的计算公式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$：Query 矩阵，代表"我在找什么"
- $K \in \mathbb{R}^{m \times d_k}$：Key 矩阵，代表"我有什么特征"
- $V \in \mathbb{R}^{m \times d_v}$：Value 矩阵，代表"我的实际内容"
- $d_k$：Key 的维度
- $\sqrt{d_k}$：缩放因子，防止点积值过大导致 softmax 梯度消失

**直觉类比：** 想象你在图书馆找书。Q 是你的搜索词，K 是每本书的标签，V 是书的内容。Attention 就是计算你的搜索词和每本书标签的匹配度（$QK^T$），然后按匹配度加权取出书的内容。

**为什么要缩放？** 当 $d_k$ 很大时，$QK^T$ 的方差会线性增长，导致 softmax 输出趋向 one-hot（梯度接近 0）。除以 $\sqrt{d_k}$ 将方差拉回到 1。

代码对应（`attention.py:49-70`）：

```python
def _scaled_dot_product_attention(self, q, k, v, mask=None):
    # q: (B, n_heads, S_q, d_k), k: (B, n_heads, S_k, d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # QK^T / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)  # softmax
    attn_weights = self.dropout(attn_weights)
    output = torch.matmul(attn_weights, v)  # * V
    return output
```

### Multi-Head Attention (MHA)

将 Q/K/V 各自投影到 $h$ 个子空间，每个头独立做 Attention，最后拼接：

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

其中：
- $h$：头数（`n_heads`）
- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model} \times d_k}$：第 $i$ 个头的投影矩阵，$d_k = d_{model} / h$
- $W^O \in \mathbb{R}^{d_{model} \times d_{model}}$：输出投影矩阵

**为什么需要多头？** 单个 Attention 头只能学习一种关注模式。多头让模型同时关注不同的语义子空间——比如一个头关注语法关系，另一个头关注语义相似性。

代码对应（`attention.py:26-47`）：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        self.head_dim = dim // n_heads
        self.w_q = nn.Linear(dim, dim, bias=False)  # 包含所有头的 W_i^Q
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)   # W^O
```

张量形状变化流程：

```
输入 x: (B, S, dim)
  → W_q 投影: (B, S, dim)
  → split_heads: (B, n_heads, S, head_dim)
  → Attention: (B, n_heads, S, head_dim)
  → merge_heads: (B, S, dim)
  → W_o 投影: (B, S, dim)
```

### Grouped Query Attention (GQA)

GQA 是 MHA 和 MQA 之间的折中方案。将 Q 头分成 $g$ 组，每组共享一对 K/V 头：

$$n_{kv\_heads} = n_{heads} / g$$

当 $g = 1$ 时退化为 MHA，当 $g = n_{heads}$ 时退化为 MQA。

**为什么 LLaMA 3 用 GQA？** MHA 的 KV Cache 大小与头数成正比。GQA 将 KV 头数减少到 $1/g$，在几乎不损失模型质量的前提下，KV Cache 减少 $g$ 倍，推理速度大幅提升。

实现方式：通过 `repeat_interleave` 将 K/V 头扩展到与 Q 头匹配：

```python
# GroupedQueryAttention (attention.py:107-144)
# k: (B, n_kv_heads, S, head_dim) → repeat → (B, n_heads, S, head_dim)
k = k.repeat_interleave(self.n_rep, dim=1)
v = v.repeat_interleave(self.n_rep, dim=1)
```

其中 `n_rep = n_heads // n_kv_heads`，即每组的大小。

### MHA vs GQA vs MQA 对比

| 变体 | Q 头数 | KV 头数 | KV Cache 大小 | 代表模型 |
|------|--------|---------|--------------|---------|
| MHA | h | h | O(h) | GPT-2, BERT |
| GQA | h | g (g<h) | O(g) | LLaMA 2/3, Gemma |
| MQA | h | 1 | O(1) | PaLM, Falcon |

## 架构图解

### Attention 计算流程

```
Input x
  ├→ Linear(W_q) → Q ─┐
  ├→ Linear(W_k) → K ─┤
  └→ Linear(W_v) → V ─┤
                        ├→ Split Heads → Q_h, K_h, V_h
                        │
                        ├→ Q_h × K_h^T / √d_k → Scores (B, h, S, S)
                        │
                        ├→ Mask (optional) → Scores_masked
                        │
                        ├→ Softmax → Attention Weights
                        │
                        ├→ × V_h → Head Outputs (B, h, S, d_k)
                        │
                        └→ Merge Heads → Concat → Linear(W_o) → Output
```

### GQA 头分组示意

```
MHA (4 heads):     Q₀K₀V₀  Q₁K₁V₁  Q₂K₂V₂  Q₃K₃V₃  (每头独立 KV)
GQA (4Q, 2KV):    Q₀K₀V₀  Q₁K₀V₀  Q₂K₁V₁  Q₃K₁V₁  (2头共享1组KV)
MQA (4Q, 1KV):    Q₀K₀V₀  Q₁K₀V₀  Q₂K₀V₀  Q₃K₀V₀  (所有头共享1组KV)
```

## 代码实现分析

### 关键文件清单

| 文件 | 行号范围 | 职责 |
|------|---------|------|
| `attention.py:26-99` | `MultiHeadAttention` | 标准 MHA 实现 |
| `attention.py:101-261` | `GroupedQueryAttention` | GQA/MQA 实现 |

### MultiHeadAttention 关键参数

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int,       # 模型维度 d_model
                       n_heads: int,    # 注意力头数 h
                       dropout: float = 0.1):
```

- `dim`：输入/输出维度，必须能被 `n_heads` 整除
- `head_dim = dim // n_heads`：每个头的维度 $d_k$
- `scale = head_dim ** 0.5`：缩放因子 $\sqrt{d_k}$

### GroupedQueryAttention 关键参数

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int,         # 模型维度
                       n_heads: int,      # Q 头数
                       n_kv_heads: int,   # KV 头数 (≤ n_heads)
                       dropout: float = 0.1):
```

- `n_kv_heads`：K/V 头数，必须能被 `n_heads` 整除
- `n_rep = n_heads // n_kv_heads`：每个 KV 头被重复的次数
- 当 `n_kv_heads == 1` 时，退化为 MQA

## 与其他方案的对比

### Attention 的演进路线

```
Bahdanau Attention (2014)  — 加性 Attention，用 MLP 计算对齐分数
    ↓
Luong Attention (2015)     — 乘性 Attention，用点积计算相似度
    ↓
Scaled Dot-Product (2017)  — 加入 √d_k 缩放，成为 Transformer 标准
    ↓
Multi-Head (2017)          — 多头并行关注不同子空间
    ↓
GQA (2023)                 — 分组共享 KV，减少 KV Cache
    ↓
MLA (2024)                 — 低秩压缩 KV，进一步减少 Cache → 见 DeepSeek V3 文档
```

## 动手实践

→ [notebook 01: Attention 基础](../../notebooks/01_attention_basics.ipynb)

推荐练习：
1. 修改 `n_heads` 观察输出维度变化
2. 对比 MHA 和 GQA 的 KV Cache 大小
3. 可视化 Attention 权重矩阵，理解 softmax 的"关注度"含义

## 延伸阅读

- Vaswani et al., "Attention Is All You Need" (2017) — Transformer 原始论文
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023) — GQA 论文
- "The Illustrated Transformer" by Jay Alammar — 经典可视化讲解

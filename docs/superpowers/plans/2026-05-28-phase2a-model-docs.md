# Phase 2a: Model Architecture Detailed Docs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create 4 detailed model architecture documentation files that explain core principles, math formulas, code mappings, and architecture evolution for beginners.

**Architecture:** Each doc follows a consistent template (概述 → 核心原理 → 架构图解 → 代码实现分析 → 对比 → 动手实践 → 延伸阅读). Chinese-English mixed language. Code references use `file.py:line_range` format. Formulas use LaTeX with symbol annotations.

**Tech Stack:** Markdown, LaTeX math, ASCII diagrams

---

## File Structure

```
docs/models/
├── attention.md          # Task 1 — Attention mechanism deep-dive (shared by all models)
├── transformer.md        # Task 2 — Original Transformer architecture
├── llama3.md             # Task 3 — LLaMA 3 decoder-only architecture
└── deepseek-v3.md        # Task 4 — DeepSeek V3 MLA + MoE architecture
```

---

## Task 1: Create `docs/models/attention.md`

**Files:**
- Create: `docs/models/attention.md`

- [ ] **Step 1: Create the file with full content**

This is the foundational document shared by all three models. It covers Scaled Dot-Product Attention, MHA, GQA, and MQA with full math derivations and code mapping.

```markdown
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
```

- [ ] **Step 2: Verify and commit**

Run: `ls -la docs/models/attention.md && wc -l docs/models/attention.md`

```bash
git add docs/models/attention.md
git commit -m "docs: add Attention mechanism detailed documentation"
```

---

## Task 2: Create `docs/models/transformer.md`

**Files:**
- Create: `docs/models/transformer.md`

- [ ] **Step 1: Create the file with full content**

```markdown
# Transformer 原理详解

> 上一篇：[Attention 机制详解](attention.md) ｜ 下一篇：[LLaMA 3](llama3.md)

## 概述

Transformer 是 2017 年由 Vaswani 等人提出的序列到序列模型，完全基于 Attention 机制，抛弃了 RNN/CNN 结构。它是所有现代 LLM（包括 LLaMA、GPT、DeepSeek）的鼻祖。

**前置知识：** [Attention 机制详解](attention.md)、位置编码、残差连接、LayerNorm
**代码位置：** [`models/transformer/`](../../models/transformer/)

## 核心原理

### 整体架构

Transformer 由 Encoder 和 Decoder 两部分组成：

- **Encoder**：处理输入序列，每个位置都能看到所有输入位置（双向）
- **Decoder**：自回归生成输出序列，每个位置只能看到之前的位置（单向）+ 通过 Cross-Attention 关注 Encoder 输出

### Encoder Layer

每个 Encoder Layer 包含两个子层：

1. **Multi-Head Self-Attention**：输入序列自己对自己做 Attention
2. **Feed-Forward Network (FFN)**：两层线性变换 + GELU 激活

每个子层都使用 **Post-Norm**（残差连接后做 LayerNorm）：

$$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

代码对应（`encoder.py:12-48`）：

```python
class EncoderLayer(nn.Module):
    def forward(self, x, mask=None):
        # 子层 1: Self-Attention (Post-Norm)
        attn_out, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))  # 残差 + LayerNorm
        # 子层 2: FFN (Post-Norm)
        ffn_out = self.feedforward(x)
        x = self.norm2(x + self.dropout(ffn_out))   # 残差 + LayerNorm
        return x
```

### Decoder Layer

每个 Decoder Layer 包含三个子层：

1. **Masked Multi-Head Self-Attention**：带因果掩码的自注意力，防止看到未来 token
2. **Multi-Head Cross-Attention**：Q 来自 Decoder，K/V 来自 Encoder 输出
3. **Feed-Forward Network (FFN)**：与 Encoder 相同

因果掩码的作用：在自回归生成时，token $i$ 只能关注 token $1, 2, ..., i$，不能看到 $i+1$ 及之后的位置。

代码对应（`decoder.py:13-62`）：

```python
class DecoderLayer(nn.Module):
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 子层 1: Masked Self-Attention
        attn_out, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # 子层 2: Cross-Attention (Q 来自 Decoder, K/V 来自 Encoder)
        cross_out, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_out))
        # 子层 3: FFN
        ffn_out = self.feedforward(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x
```

### Position-wise FFN

FFN 对每个位置独立地做两次线性变换：

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

其中 $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$，$d_{ff} = 4 \times d_{model}$。

代码对应（`feedforward.py:12-33`）：

```python
class FFN(nn.Module):
    def __init__(self, dim, ff_hidden_dim, dropout=0.1):
        self.linear1 = nn.Linear(dim, ff_hidden_dim)   # W_1
        self.linear2 = nn.Linear(ff_hidden_dim, dim)    # W_2
    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))  # GELU(xW1)W2
```

### Sinusoidal Position Encoding

由于 Transformer 没有循环结构，需要额外注入位置信息。原始论文使用固定的正弦/余弦函数：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中 $pos$ 是位置索引，$i$ 是维度索引。

**直觉：** 每个维度使用不同频率的正弦波，就像二进制编码一样，让模型能区分不同位置。低维度变化快（区分近距离），高维度变化慢（编码远距离关系）。

代码对应（`positional_encoding.py:12-40`）：

```python
def sinusoidal_pe(seq_len, dim):
    pe = torch.zeros(1, seq_len, dim)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
    pe[0, :, 0::2] = torch.sin(position * div_term)  # 偶数维: sin
    pe[0, :, 1::2] = torch.cos(position * div_term)  # 奇数维: cos
    return pe
```

## 架构图解

### Encoder-Decoder 数据流

```
Source tokens → Embedding + PE → [EncoderLayer × N] → Encoder output
                                                      ↓
Target tokens → Embedding + PE → [DecoderLayer × N] → Linear → Softmax → 输出概率

DecoderLayer 内部:
  x → Masked Self-Attention → residual + LN →
    → Cross-Attention (Q=x, K/V=Encoder output) → residual + LN →
    → FFN → residual + LN → output
```

### 张量形状变化

```
输入 src: (batch=2, src_len=16)  token IDs
  → TokenEmbedding: (2, 16, 128)  + sinusoidal_pe: (1, 16, 128)
  → EncoderLayer × 2:
    Self-Attention: (2, 16, 128)
    FFN: (2, 16, 128)
  → Encoder output: (2, 16, 128)

输入 tgt: (batch=2, tgt_len=20)  token IDs
  → TokenEmbedding: (2, 20, 128)  + sinusoidal_pe: (1, 20, 128)
  → DecoderLayer × 2:
    Masked Self-Attention: (2, 20, 128)
    Cross-Attention: Q=(2,20,128), K/V=(2,16,128) → (2, 20, 128)
    FFN: (2, 20, 128)
  → Linear(dim, vocab_size): (2, 20, 1000)  logits
```

## 代码实现分析

### 关键文件清单

| 文件 | 职责 | 关键类 |
|------|------|--------|
| `config.py` | 超参数 | `TransformerConfig` |
| `encoder.py` | Encoder 实现 | `EncoderLayer`, `Encoder` |
| `decoder.py` | Decoder 实现 | `DecoderLayer`, `Decoder` |
| `model.py` | 完整模型 | `Transformer` |

### TransformerConfig 参数说明

```python
@dataclass
class TransformerConfig:
    vocab_size: int = 1000    # 词表大小
    dim: int = 128            # 模型维度 d_model
    n_heads: int = 4          # 注意力头数
    n_layers: int = 2         # Encoder/Decoder 层数
    ff_hidden_dim: int = 512  # FFN 隐藏层维度 (4 × dim)
    max_seq_len: int = 512    # 最大序列长度
    dropout: float = 0.1      # Dropout 比例
    eps: float = 1e-6         # LayerNorm epsilon
```

注意：本项目使用小规模参数（dim=128, n_layers=2），方便在消费级 GPU/CPU 上运行。

## 与其他模型的对比

Transformer 是基线模型，后续 LLM 的改进点：

| 改进项 | Transformer | 后续改进 | 改进原因 |
|--------|-------------|---------|---------|
| 架构 | Encoder-Decoder | Decoder-only | 自回归生成不需要 Encoder |
| 归一化 | LayerNorm (Post-Norm) | RMSNorm (Pre-Norm) | 训练更稳定 |
| 位置编码 | Sinusoidal (固定) | RoPE (可学习旋转) | 长序列外推 |
| 激活 | GELU | SwiGLU | 门控提升表达力 |
| Attention | MHA | GQA | 减少 KV Cache |

## 动手实践

→ [notebook 02: Transformer walkthrough](../../notebooks/02_transformer_walkthrough.ipynb)

推荐练习：
1. 打印每一步的张量形状，理解数据流
2. 修改 `n_layers` 观察模型大小和输出变化
3. 尝试将 Post-Norm 改为 Pre-Norm，对比训练稳定性

## 延伸阅读

- Vaswani et al., "Attention Is All You Need" (2017)
- "The Annotated Transformer" by Harvard NLP — 逐行代码解读
- "The Illustrated Transformer" by Jay Alammar
```

- [ ] **Step 2: Verify and commit**

```bash
git add docs/models/transformer.md
git commit -m "docs: add Transformer detailed documentation"
```

---

## Task 3: Create `docs/models/llama3.md`

**Files:**
- Create: `docs/models/llama3.md`

- [ ] **Step 1: Create the file with full content**

```markdown
# LLaMA 3 架构详解

> 上一篇：[Transformer](transformer.md) ｜ 下一篇：[DeepSeek V3](deepseek-v3.md)

## 概述

LLaMA 3 是 Meta 于 2024 年发布的 Decoder-only 大语言模型，代表了当前 LLM 架构的主流范式。相比原始 Transformer，它采用了 6 项关键改进，在训练效率和推理速度上都有显著提升。

**前置知识：** [Attention 机制详解](attention.md)（尤其是 GQA 部分）、[Transformer](transformer.md)
**代码位置：** [`models/llama3/`](../../models/llama3/)

## 核心原理

### 与 Transformer 的六大区别

| 改进项 | Transformer | LLaMA 3 | 改进原因 |
|--------|-------------|---------|---------|
| 架构 | Encoder-Decoder | Decoder-only | 自回归生成不需要 Encoder，去掉后模型更简洁 |
| 归一化 | Post-Norm LayerNorm | Pre-Norm RMSNorm | Pre-Norm 训练更稳定；RMSNorm 比 LayerNorm 快 10-15% |
| 位置编码 | Sinusoidal | RoPE | 支持长序列外推，可扩展到训练长度之外 |
| FFN | GELU-FFN | SwiGLU-FFN | 门控机制提升表达能力 |
| Attention | MHA | GQA | 减少 KV Cache，推理速度更快 |
| 残差连接 | 之后归一化 | 之前归一化 | 避免深层网络的梯度问题 |

### RMSNorm

RMSNorm（Root Mean Square Normalization）是 LayerNorm 的简化版，去掉了均值中心化：

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$
$$\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}$$

其中 $\gamma$ 是可学习的缩放参数，$\epsilon$ 是防止除零的小常数。

**为什么比 LayerNorm 快？** LayerNorm 需要计算均值和方差（两遍扫描），RMSNorm 只需要计算 RMS（一遍扫描）。

代码对应（`normalization.py:33-56`）：

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight  # self.weight = γ
```

### SwiGLU FFN

SwiGLU 使用三个权重矩阵（比标准 FFN 多一个），通过门控机制控制信息流：

$$\text{SwiGLU}(x) = (xW_1 \odot \text{SiLU}(xW_3))W_2$$

其中 $\odot$ 是逐元素乘法，$\text{SiLU}(x) = x \cdot \sigma(x)$。

**直觉：** $W_3(x)$ 作为"门"，决定哪些信息通过。SiLU 激活比 GELU 更平滑，在零点附近有更好的梯度特性。

代码对应（`feedforward.py:36-58`）：

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, dim, ff_hidden_dim, dropout=0.1):
        self.w1 = nn.Linear(dim, ff_hidden_dim, bias=False)  # 门控投影
        self.w2 = nn.Linear(ff_hidden_dim, dim, bias=False)  # 输出投影
        self.w3 = nn.Linear(dim, ff_hidden_dim, bias=False)  # 值投影
    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))  # 门控乘法
```

### Rotary Positional Embedding (RoPE)

RoPE 通过对 Q/K 向量施加旋转变换来编码位置信息：

$$q'_m = R_m q_m, \quad k'_n = R_n k_n$$

其中 $R_m$ 是旋转矩阵。关键性质：$q'_m$ 和 $k'_n$ 的点积只依赖于相对位置 $(m-n)$，不依赖于绝对位置。

$$q_m^T k_n \to q_m^T R_{m-n} k_n$$

**为什么支持外推？** 因为旋转角度可以任意扩展，不像 Sinusoidal PE 在训练长度之外没有定义。

代码对应（`positional_encoding.py:43-146`）：

```python
class RotaryPositionalEncoding(nn.Module):
    def _apply_rope(self, x):
        # x: (B, n_heads, S, head_dim)
        # 将 head_dim 分成两半，做 2D 旋转
        x1, x2 = x.chunk(2, dim=-1)
        cos = self.cos_cache[:x.size(2)]  # (S, head_dim//2)
        sin = self.sin_cache[:x.size(2)]
        return torch.cat([x1 * cos - x2 * sin,
                          x1 * sin + x2 * cos], dim=-1)
```

### KV Cache

自回归生成时，每生成一个新 token 都需要计算 Attention。如果不缓存，每个 token 都要重新计算所有之前 token 的 K/V，复杂度为 $O(n^2)$。

KV Cache 缓存已计算的 K 和 V，新 token 只需计算自己的 Q，然后与缓存的 K/V 做 Attention：

```
Step 1: 计算 token_1 的 K₁, V₁ → 存入 cache
Step 2: 计算 token_2 的 K₂, V₂ → 追加到 cache, Q₂ 与 [K₁,K₂] 做 Attention
Step n: 计算 token_n 的 Kₙ, Vₙ → 追加到 cache, Qₙ 与 [K₁,...,Kₙ] 做 Attention
```

**KV Cache 大小：** `2 × n_layers × n_kv_heads × seq_len × head_dim × dtype_bytes`

代码对应（`llama3/model.py:200-250`）：

```python
class LLaMA3Model(nn.Module):
    def create_kv_cache(self, max_seq_len):
        # 为每一层创建 KV 缓存
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(max_seq_len)

class KVCache:
    def __init__(self, max_seq_len):
        self.k = None  # 动态追加
        self.v = None
    def update(self, new_k, new_v):
        if self.k is None:
            self.k, self.v = new_k, new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)
        return self.k, self.v
```

### Pre-Norm 架构

LLaMA 3 使用 Pre-Norm：先做 RMSNorm，再做子层计算：

$$x_{out} = x + \text{SubLayer}(\text{RMSNorm}(x))$$

对比 Transformer 的 Post-Norm：$x_{out} = \text{LayerNorm}(x + \text{SubLayer}(x))$

**为什么 Pre-Norm 更好？** Post-Norm 在深层网络中，残差路径上的梯度需要经过 LayerNorm，可能不稳定。Pre-Norm 的残差路径是干净的恒等映射，梯度可以直接回传。

## 架构图解

### LLaMA 3 Transformer Block

```
Input x
  ├→ RMSNorm ─→ GQAttention + RoPE ─→ + (残差)
  │                                      ↓
  └──────────────────────────────────────→ + → x'
                                              │
  ┌──────────────────────────────────────→ + ←─┘
  │                                         │
  └→ RMSNorm ─→ SwiGLU FFN ──────────────→ +
                                            ↓
                                         Output
```

### 自回归生成流程

```
input_ids: [tok₁, tok₂, ..., tokₙ]
  → Embedding: (1, n, dim)
  → TransformerBlock × N (with KV Cache):
    tok₁: 计算 K₁, V₁ → cache
    tok₂: 计算 K₂, V₂ → cache, Q₂ × [K₁,K₂] → attn
    ...
  → RMSNorm → Linear → logits (1, n, vocab_size)
  → argmax/sampling → next_token
  → append → repeat
```

## 代码实现分析

### 关键文件清单

| 文件 | 职责 | 关键类 |
|------|------|--------|
| `config.py` | 超参数 | `LLaMA3Config` |
| `model.py:14-90` | 单层 Block | `TransformerBlock` |
| `model.py:93-195` | 基础模型 | `LLaMA3Model` |
| `model.py:198-373` | 带 LM Head 的模型 | `LLaMA3ForCausalLM` |

### LLaMA3Config 参数说明

```python
@dataclass
class LLaMA3Config:
    vocab_size: int = 1000
    dim: int = 128            # 模型维度
    n_heads: int = 4          # Q 头数
    n_kv_heads: int = 2       # KV 头数 (GQA: n_kv_heads < n_heads)
    n_layers: int = 4         # Transformer Block 层数
    ff_hidden_dim: int = 352  # SwiGLU 隐藏层维度
    max_seq_len: int = 512    # 最大序列长度
    dropout: float = 0.0      # LLaMA 通常不用 dropout
    eps: float = 1e-5         # RMSNorm epsilon
    rope_theta: float = 10000.0  # RoPE 基础频率
```

`head_dim` 属性自动计算为 `dim // n_heads`。

### 自回归生成方法

```python
class LLaMA3ForCausalLM(nn.Module):
    def generate(self, input_ids, max_new_tokens=20, temperature=1.0):
        # 创建 KV Cache
        self.backbone.create_kv_cache(max_seq_len=input_ids.size(1) + max_new_tokens)
        for _ in range(max_new_tokens):
            logits = self(input_ids)            # 前向传播 (带 KV Cache)
            next_logits = logits[:, -1, :] / temperature   # 取最后一个位置
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)       # 采样
            input_ids = torch.cat([input_ids, next_token], dim=1)  # 追加
        return input_ids
```

## 动手实践

→ [notebook 03: LLaMA 3 walkthrough](../../notebooks/03_llama3_walkthrough.ipynb)

推荐练习：
1. 对比有无 KV Cache 的生成速度差异
2. 修改 `n_kv_heads` 从 4 到 1（退化为 MQA），观察输出差异
3. 修改 `rope_theta` 从 10000 到 1000000，测试长序列外推能力

## 延伸阅读

- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023)
- Rozière et al., "Code Llama: Open Foundation Models for Code" (2023)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) — RoPE 原始论文
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019) — RMSNorm 论文
```

- [ ] **Step 2: Verify and commit**

```bash
git add docs/models/llama3.md
git commit -m "docs: add LLaMA 3 detailed documentation"
```

---

## Task 4: Create `docs/models/deepseek-v3.md`

**Files:**
- Create: `docs/models/deepseek-v3.md`

- [ ] **Step 1: Create the file with full content**

```markdown
# DeepSeek V3 架构详解

> 上一篇：[LLaMA 3](llama3.md) ｜ 返回：[模型架构总览](overview.md)

## 概述

DeepSeek V3 是 DeepSeek 于 2024 年发布的 MoE（Mixture of Experts）大语言模型，总参数 671B，每次推理仅激活 37B 参数。相比 LLaMA 3，它引入了两项重大创新：MLA（Multi-head Latent Attention）低秩压缩注意力和 MoE 混合专家系统。

**前置知识：** [LLaMA 3](llama3.md)（理解 GQA、RoPE、RMSNorm）、[Attention 机制详解](attention.md)
**代码位置：** [`models/deepseek_v3/`](../../models/deepseek_v3/)

## 核心原理

### MLA (Multi-head Latent Attention)

MLA 的核心思想：将 KV 压缩到一个低秩的 latent 空间，大幅减少 KV Cache。

#### 传统 MHA 的 KV Cache 问题

标准 MHA 需要缓存每个 token、每个头的完整 K 和 V 向量：

$$\text{KV Cache} = 2 \times n_{layers} \times n_{heads} \times seq\_len \times d_{head}$$

对于 128K 上下文长度，这个值非常大。

#### MLA 的低秩压缩

MLA 引入压缩-解压两步：

**压缩（Down-Projection）：** 将输入 $x$ 压缩到低维 latent 向量 $c_{KV}$

$$c_{KV} = W_{DKV} \cdot x, \quad c_{KV} \in \mathbb{R}^{d_c}$$

其中 $d_c \ll n_{heads} \times d_{head}$，$W_{DKV} \in \mathbb{R}^{d_{model} \times d_c}$。

**解压（Up-Projection）：** 从 $c_{KV}$ 恢复出 K 和 V

$$k = W_{UK} \cdot c_{KV}, \quad v = W_{UV} \cdot c_{KV}$$

**KV Cache 只需存储 $c_{KV}$：** 大小从 $2 \times n_{heads} \times d_{head}$ 减少到 $d_c$，压缩比为 $\frac{d_c}{2 \times n_{heads} \times d_{head}}$（通常 5-13 倍）。

#### 解耦 RoPE

MLA 将位置信息与内容信息分离：
- **内容部分**：通过低秩压缩（$c_{KV}$）处理
- **位置部分**：通过单独的 $q_{pe}$, $k_{pe}$ 编码 RoPE

$$q = [q_c; q_{pe}], \quad k = [k_c; k_{pe}]$$

这样做是因为 RoPE 的旋转操作会破坏低秩结构。将 RoPE 应用在独立的位置向量上，既保留了位置信息，又不影响压缩效率。

代码对应（`mla.py:38-130`）：

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        # 低秩压缩投影
        self.W_DKV = nn.Linear(dim, kv_lora_rank, bias=False)       # 压缩
        self.W_UK = nn.Linear(kv_lora_rank, n_heads * v_head_dim, bias=False)  # 解压 K
        self.W_UV = nn.Linear(kv_lora_rank, n_heads * v_head_dim, bias=False)  # 解压 V
        # 解耦 RoPE
        self.W_QR = nn.Linear(dim, n_heads * qk_rope_head_dim, bias=False)  # Q 位置
        self.W_KR = nn.Linear(dim, n_heads * qk_rope_head_dim, bias=False)  # K 位置
```

### MoE (Mixture of Experts)

MoE 将 FFN 层替换为多个 Expert（专家网络），由 Router 动态决定每个 token 发给哪些 Expert。

#### Router（路由器）

Router 是一个线性层 + softmax，输出每个 token 对每个 expert 的路由分数：

$$g = \text{softmax}(x \cdot W_g), \quad g \in \mathbb{R}^{n_{tokens} \times n_{experts}}$$

然后选择 Top-K 个 expert：

$$\text{indices}, \text{scores} = \text{TopK}(g, K=n_{activated})$$

代码对应（`moe.py:12-56`）：

```python
class Router(nn.Module):
    def __init__(self, config):
        self.gate = nn.Linear(dim, n_routed_experts, bias=False)
    def forward(self, x):
        scores = torch.softmax(self.gate(x), dim=-1)     # (n_tokens, n_experts)
        topk_scores, topk_indices = torch.topk(scores, self.n_activated_experts)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # 归一化
        return topk_indices, topk_scores
```

#### Shared Expert vs Routed Expert

- **SharedExpert**：所有 token 都经过的 Expert，捕获通用知识
- **RoutedExpert**：由 Router 动态选择的 Expert，捕获专业领域知识

最终输出 = Shared Expert 输出 + Routed Expert 加权和：

$$y = \text{SharedExpert}(x) + \sum_{i \in \text{selected}} g_i \cdot \text{RoutedExpert}_i(x)$$

代码对应（`moe.py:80-130`）：

```python
class SharedExpert(nn.Module):
    def forward(self, x):
        return self.swiglu_ffn(x)  # 所有 token 都过

class MoELayer(nn.Module):
    def forward(self, x):
        shared_out = self.shared_expert(x)         # 通用知识
        indices, scores = self.router(x)           # 路由选择
        # scatter-add 实现加权路由
        routed_out = self._dispatch_and_compute(x, indices, scores)
        return shared_out + routed_out
```

## 架构图解

### DeepSeek V3 Transformer Block

```
Input x
  ├→ RMSNorm ─→ MLA (低秩压缩 Attention + 解耦 RoPE) ─→ +
  │                                                        ↓
  └────────────────────────────────────────────────────────→ + → x'
                                                                │
  ┌────────────────────────────────────────────────────────→ + ←─┘
  │                                                           │
  └→ RMSNorm ─→ MoE (SharedExpert + Router → RoutedExperts) → +
                                                              ↓
                                                           Output
```

### MLA 压缩-解压流程

```
Input x: (B, S, dim)
  ├→ W_DKV 压缩: (B, S, d_c)        ← 只存这个到 KV Cache!
  │    ├→ W_UK 解压 K: (B, S, n_heads * v_head_dim)
  │    └→ W_UV 解压 V: (B, S, n_heads * v_head_dim)
  ├→ W_Q 投影 Q 内容: (B, S, n_heads * q_head_dim)
  ├→ W_QR 投影 Q 位置: (B, S, n_heads * rope_dim)  → RoPE
  └→ W_KR 投影 K 位置: (B, S, n_heads * rope_dim)  → RoPE

拼接: Q = [Q_content; Q_pe], K = [K_content; K_pe]
Attention(Q, K, V) → Output
```

### MoE 路由流程

```
Input tokens: tok₁, tok₂, ..., tokₙ
  → Router gate: 每个 token 计算 expert 分数
  → Top-K 选择: tok₁ → [exp₂, exp₅], tok₂ → [exp₁, exp₇], ...
  → Dispatch: 将 token 发送到对应 expert
  → Expert 计算: 各 expert 独立处理
  → Gather: 收集结果并加权求和
  → + SharedExpert 输出 → 最终输出
```

## 代码实现分析

### 关键文件清单

| 文件 | 职责 | 关键类 |
|------|------|--------|
| `config.py` | 超参数 | `DeepSeekV3Config` |
| `mla.py:38-274` | MLA 注意力 | `MultiHeadLatentAttention` |
| `moe.py:12-56` | 路由器 | `Router` |
| `moe.py:58-78` | 共享专家 | `SharedExpert` |
| `moe.py:80-130` | 路由专家 | `RoutedExpert` |
| `moe.py:132-211` | MoE 层 | `MoELayer` |
| `model.py:14-65` | 单层 Block | `DeepSeekV3Block` |
| `model.py:68-145` | 基础模型 | `DeepSeekV3Model` |
| `model.py:148-234` | 带 LM Head | `DeepSeekV3ForCausalLM` |

### DeepSeekV3Config 关键参数

```python
@dataclass
class DeepSeekV3Config:
    dim: int = 128                    # 模型维度
    n_heads: int = 4                  # 注意力头数
    kv_lora_rank: int = 32           # KV 低秩压缩维度 d_c
    qk_rope_head_dim: int = 16       # RoPE 位置维度
    v_head_dim: int = 32             # V 头维度
    n_routed_experts: int = 8        # 路由专家总数
    n_shared_experts: int = 1        # 共享专家数
    n_activated_experts: int = 2     # 每次激活的路由专家数
    moe_intermediate_dim: int = 128  # 每个 expert 的 FFN 隐藏层维度
```

**压缩比计算：** 传统 KV Cache = `2 * n_heads * v_head_dim = 2 * 4 * 32 = 256`，MLA Cache = `kv_lora_rank = 32`，压缩比 = 256/32 = **8 倍**。

## 动手实践

→ [notebook 04: DeepSeek V3 walkthrough](../../notebooks/04_deepseek_v3_walkthrough.ipynb)

推荐练习：
1. 对比 MLA 和 MHA 的 KV Cache 大小
2. 修改 `n_activated_experts` 观察路由分布变化
3. 可视化 Router 的路由分数矩阵，理解 token-to-expert 分配

## 延伸阅读

- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024)
- DeepSeek-AI, "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" (2024)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2022) — MoE 基础
```

- [ ] **Step 2: Verify and commit**

```bash
git add docs/models/deepseek-v3.md
git commit -m "docs: add DeepSeek V3 detailed documentation"
```

---

## Self-Review

**1. Spec coverage:** All 4 model docs from Phase 2 spec are covered (attention.md, transformer.md, llama3.md, deepseek-v3.md).

**2. Placeholder scan:** No TBD/TODO/fill-in-later found. All code references use actual function/class names from the codebase.

**3. Type consistency:** All class names (MultiHeadAttention, GroupedQueryAttention, SwiGLUFFN, RMSNorm, etc.) and function signatures match the actual source code. Config field names match dataclass definitions.

**4. Cross-reference consistency:** All `上一篇/下一篇` navigation links form a chain: overview → attention → transformer → llama3 → deepseek-v3 → overview. All `notebook` links point to existing files.

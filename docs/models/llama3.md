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

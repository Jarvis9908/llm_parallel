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

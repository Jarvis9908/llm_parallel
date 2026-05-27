# 通用组件 (Common Components)

模型架构的共享基础组件，被 Transformer、LLaMA 3、DeepSeek V3 共同使用。

## 文件说明

| 文件 | 功能 | 关键内容 |
|------|------|---------|
| `activation.py` | 激活函数 | `gelu()`, `silu()`, `relu()` |
| `attention.py` | 注意力机制 | `MultiHeadAttention`, `GroupedQueryAttention` |
| `embeddings.py` | 词嵌入 | `TokenEmbedding` |
| `feedforward.py` | 前馈网络 | `FFN`, `SwiGLUFFN` |
| `normalization.py` | 归一化层 | `LayerNorm`, `RMSNorm` |
| `positional_encoding.py` | 位置编码 | `sinusoidal_pe()`, `RotaryPositionalEncoding` |

## 演进关系

```
Transformer 使用:  gelu, MultiHeadAttention, FFN, LayerNorm, sinusoidal_pe, TokenEmbedding
LLaMA 3 使用:     silu, GroupedQueryAttention, SwiGLUFFN, RMSNorm, RotaryPositionalEncoding, TokenEmbedding
DeepSeek V3 使用:  silu, SwiGLUFFN, RMSNorm, RotaryPositionalEncoding, TokenEmbedding
                   (+ 自有 MLA 和 MoE 实现)
```

## 详细文档

→ [模型架构总览](../../docs/models/overview.md)

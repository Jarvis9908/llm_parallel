# 模型架构总览

本项目实现了三个代表性的 LLM 架构，展示了从经典到现代的演进过程：

```
Transformer (2017)  →  LLaMA 3 (2024)  →  DeepSeek V3 (2024)
  Encoder-Decoder       Decoder-only        Decoder-only + MoE
  MHA + FFN             GQA + SwiGLU        MLA + MoE + SwiGLU
  Sinusoidal PE         RoPE                RoPE (decoupled)
  LayerNorm             RMSNorm             RMSNorm
```

## 三者核心对比

| 特性 | Transformer | LLaMA 3 | DeepSeek V3 |
|------|-------------|---------|-------------|
| 架构类型 | Encoder-Decoder | Decoder-only | Decoder-only |
| 注意力机制 | MHA | GQA | MLA（低秩压缩） |
| 位置编码 | Sinusoidal | RoPE | RoPE（解耦） |
| 归一化 | LayerNorm (Post-Norm) | RMSNorm (Pre-Norm) | RMSNorm (Pre-Norm) |
| FFN | GELU-FFN | SwiGLU-FFN | SwiGLU-FFN + MoE |
| KV Cache | 无 | 有 | 有（低秩压缩版） |
| 主要用途 | 序列到序列任务 | 通用文本生成 | 高效大规模推理 |

## 演进脉络

### 从 Transformer 到 LLaMA 3

LLaMA 3 做了以下关键改进：
- **Decoder-only**：去掉 Encoder，只保留 Decoder，更适合自回归生成任务
- **RMSNorm 替换 LayerNorm**：计算更简单（不需要计算均值），效果相当
- **RoPE 替换 Sinusoidal PE**：更好的长序列外推能力
- **SwiGLU 替换 GELU**：门控机制提升 FFN 的表达能力
- **GQA 替换 MHA**：减少 KV Cache，提升推理效率
- **Pre-Norm 替换 Post-Norm**：训练更稳定

### 从 LLaMA 3 到 DeepSeek V3

DeepSeek V3 在 LLaMA 基础上引入了两个重大创新：
- **MLA（Multi-head Latent Attention）**：将 KV 压缩到低秩空间，KV Cache 减少到原来的 5-13%
- **MoE（Mixture of Experts）**：671B 总参数量，但每次只激活 37B，实现大模型能力与推理效率的平衡

## 学习建议

建议按 Transformer → LLaMA 3 → DeepSeek V3 的顺序学习，每个模型都在前一个的基础上做了改进，理解改进的原因比记住实现细节更重要。

- [Transformer 模块](../models/transformer/README.md) → [notebook 02](../../notebooks/02_transformer_walkthrough.ipynb)
- [LLaMA 3 模块](../models/llama3/README.md) → [notebook 03](../../notebooks/03_llama3_walkthrough.ipynb)
- [DeepSeek V3 模块](../models/deepseek_v3/README.md) → [notebook 04](../../notebooks/04_deepseek_v3_walkthrough.ipynb)

# Transformer 模块

原始 Transformer 架构（Vaswani et al., 2017 "Attention Is All You Need"）。Encoder-Decoder 结构，适用于序列到序列任务如机器翻译。

## 文件说明

| 文件 | 功能 | 关键内容 |
|------|------|---------|
| `config.py` | 超参数配置 | `TransformerConfig` dataclass |
| `encoder.py` | Encoder 实现 | `EncoderLayer`, `Encoder` |
| `decoder.py` | Decoder 实现 | `DecoderLayer`, `Decoder` |
| `model.py` | 完整模型 | `Transformer` (Encoder-Decoder) |

## 架构要点

- **Post-Norm**: LayerNorm 在残差连接之后（原始论文的做法）
- **MHA**: 标准 Multi-Head Attention，每个头独立的 Q/K/V
- **Sinusoidal PE**: 固定的正弦位置编码
- **Cross-Attention**: Decoder 通过 Cross-Attention 关注 Encoder 输出

## 快速开始

```python
from models.transformer.config import TransformerConfig
from models.transformer.model import Transformer

config = TransformerConfig(vocab_size=1000, dim=128, n_heads=4, n_layers=2)
model = Transformer(config)

import torch
src = torch.randint(0, 1000, (2, 16))  # (batch=2, src_seq=16)
tgt = torch.randint(0, 1000, (2, 20))  # (batch=2, tgt_seq=20)
output = model(src, tgt)  # (2, 20, 1000) logits
```

## 详细文档

→ [模型架构总览](../../docs/models/overview.md)
→ [notebook 02: Transformer walkthrough](../../notebooks/02_transformer_walkthrough.ipynb)

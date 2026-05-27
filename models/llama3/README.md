# LLaMA 3 模块

LLaMA 3 Decoder-only 架构，包含现代 LLM 的核心改进：GQA、RoPE、SwiGLU、RMSNorm、KV Cache。

## 文件说明

| 文件 | 功能 | 关键内容 |
|------|------|---------|
| `config.py` | 超参数配置 | `LLaMA3Config` (含 `n_kv_heads`, `rope_theta`) |
| `model.py` | 完整模型 | `TransformerBlock`, `LLaMA3Model`, `LLaMA3ForCausalLM` |

## 架构要点

- **Pre-Norm**: RMSNorm 在残差连接之前（训练更稳定）
- **GQA**: `n_kv_heads < n_heads`，多个 Q 头共享一组 K/V
- **RoPE**: 旋转位置编码，支持长序列外推
- **SwiGLU**: 三参数门控 FFN，`W1(x) * SiLU(W3(x))` 再过 `W2`
- **KV Cache**: 自回归生成时缓存已计算的 K/V，避免重复计算

## 快速开始

```python
from models.llama3.config import LLaMA3Config
from models.llama3.model import LLaMA3ForCausalLM

config = LLaMA3Config(vocab_size=1000, dim=128, n_heads=4, n_kv_heads=2, n_layers=4)
model = LLaMA3ForCausalLM(config)

import torch
input_ids = torch.randint(0, 1000, (1, 8))
output = model(input_ids)  # (1, 8, 1000) logits

# 自回归生成
generated = model.generate(input_ids, max_new_tokens=10)  # (1, 18)
```

## 详细文档

→ [模型架构总览](../../docs/models/overview.md)
→ [notebook 03: LLaMA 3 walkthrough](../../notebooks/03_llama3_walkthrough.ipynb)

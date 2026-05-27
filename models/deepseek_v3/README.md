# DeepSeek V3 模块

DeepSeek V3 Decoder-only 架构，两大核心创新：MLA（Multi-head Latent Attention）低秩压缩注意力和 MoE（Mixture of Experts）混合专家。

## 文件说明

| 文件 | 功能 | 关键内容 |
|------|------|---------|
| `config.py` | 超参数配置 | `DeepSeekV3Config` (含 MLA/MoE 参数) |
| `mla.py` | MLA 注意力 | `MultiHeadLatentAttention` — KV 低秩压缩 + 解耦 RoPE |
| `moe.py` | MoE 层 | `Router`, `SharedExpert`, `RoutedExpert`, `MoELayer` |
| `model.py` | 完整模型 | `DeepSeekV3Block`, `DeepSeekV3Model`, `DeepSeekV3ForCausalLM` |

## 架构要点

### MLA (Multi-head Latent Attention)
- 将 KV 投影到低秩 latent 空间：`c_kv = W_DKV * x`（压缩）→ `k, v = W_UK * c_kv, W_UV * c_kv`（解压）
- KV Cache 只需存储 `c_kv`，体积减少 5-13 倍
- RoPE 解耦：位置信息通过单独的 `q_pe, k_pe` 编码，不经过低秩压缩

### MoE (Mixture of Experts)
- **Router**: Top-K softmax 门控，决定每个 token 发给哪些 routed expert
- **SharedExpert**: 所有 token 都经过的 SwiGLU 专家（捕获通用知识）
- **RoutedExpert**: 由 Router 动态选择的 SwiGLU 专家（捕获专业领域知识）
- 总参数量大（671B），但每次前向只激活 37B 参数

## 快速开始

```python
from models.deepseek_v3.config import DeepSeekV3Config
from models.deepseek_v3.model import DeepSeekV3ForCausalLM

config = DeepSeekV3Config(
    vocab_size=1000, dim=128, n_heads=4, n_layers=4,
    n_routed_experts=8, n_shared_experts=1, n_activated_experts=2
)
model = DeepSeekV3ForCausalLM(config)

import torch
input_ids = torch.randint(0, 1000, (1, 8))
output = model(input_ids)  # (1, 8, 1000) logits
```

## 详细文档

→ [模型架构总览](../../docs/models/overview.md)
→ [notebook 04: DeepSeek V3 walkthrough](../../notebooks/04_deepseek_v3_walkthrough.ipynb)

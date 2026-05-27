# 上下文并行模块

处理超长序列的并行策略：将序列沿长度维度切分，通过 Ring Attention 等方式实现长序列 Attention。

## 文件说明

| 文件 | 功能 | 关键内容 |
|------|------|---------|
| `ring_attention.py` | Ring Attention | `ring_attention_step`, `rotate_kv` — 沿 Ring 旋转 KV 块 |
| `sequence_partition.py` | 序列切分 | `partition_sequence`, `create_cp_causal_mask` — 序列分块 + 因果掩码调整 |
| `cp_integration.py` | CP 与其他策略集成 | `analyze_cp_tp_memory`, `recommend_parallel_config` — CP+TP 混合分析 |

## 核心概念

### Ring Attention

将长序列沿长度维度切分，每个 GPU 持有一段 Q。KV 块沿 Ring 拓扑旋转：

```
Step 0: GPU0(Q0,K0,V0) GPU1(Q1,K1,V1) GPU2(Q2,K2,V2) GPU3(Q3,K3,V3)
Step 1: GPU0(Q0,K1,V1) GPU1(Q1,K2,V2) GPU2(Q2,K3,V3) GPU3(Q3,K0,V0)  (KV 旋转一步)
Step 2: GPU0(Q0,K2,V2) ...  (继续旋转)
Step 3: GPU0(Q0,K3,V3) ...  (最后一轮)
```

每步计算本地 Q 与当前 KV 的 Attention，最终累积得到完整 Attention 结果。

### 因果掩码调整

标准因果掩码假设完整序列。切分后需要为每个子序列生成正确的局部因果掩码，考虑全局位置偏移。

## 快速开始

```python
from parallel.context_parallel.sequence_partition import partition_sequence, create_cp_causal_mask
import torch

# 将序列切分到 4 个 GPU
full_seq = torch.randn(1, 4096, 128)  # 长度 4096
local_seq = partition_sequence(full_seq, rank=1, world_size=4)  # (1, 1024, 128)

# 生成局部因果掩码
mask = create_cp_causal_mask(local_seq_len=1024, rank=1, world_size=4)
```

## 详细文档

→ [并行策略总览](../../docs/parallel/overview.md)
→ [notebook 09: 上下文并行](../../notebooks/09_expert_and_context_parallel.ipynb)

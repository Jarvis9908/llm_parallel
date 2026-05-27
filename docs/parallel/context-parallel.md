# 上下文并行详解

> 上一篇：[专家并行](expert-parallel.md) ｜ 下一篇：[推理并行](inference.md)

## 概述

上下文并行（Context Parallelism, CP）处理超长序列（128K+），将序列沿长度维度切分到多个 GPU，通过 Ring Attention 实现分布式 Attention 计算。

**前置知识：** [Attention 机制详解](../models/attention.md)、[通信原语](communication.md)
**代码位置：** [`parallel/context_parallel/`](../../parallel/context_parallel/)

## 核心原理

### 序列切分

将长度为 L 的序列均匀切分到 P 个 GPU：

```python
def partition_sequence(x, rank, world_size):
    chunk_len = x.shape[1] // world_size
    return x[:, rank * chunk_len:(rank + 1) * chunk_len, :]
```

**问题：** 每个 GPU 只有局部 Q，但 Attention 需要全局的 K 和 V。

### Ring Attention

解决方案：KV 沿 Ring 拓扑旋转，每步计算局部 Q 与当前 KV 的部分 Attention，累积得到完整结果。

```
Step 0: GPU0(Q0,K0,V0) → Attn(Q0,K0,V0)
Step 1: GPU0(Q0,K1,V1) → Attn(Q0,K1,V1) + 累积
Step 2: GPU0(Q0,K2,V2) → Attn(Q0,K2,V2) + 累积
Step 3: GPU0(Q0,K3,V3) → Attn(Q0,K3,V3) + 累积 = 完整 Attention
```

关键：使用 Online Softmax，在每步增量更新，不需要存储所有步骤的中间结果。

代码对应（`ring_attention.py:12-56`）：

```python
def ring_attention_step(q_local, k_current, v_current, prev_max, prev_sum, prev_out):
    """单步 Ring Attention（在线 softmax）"""
    scores = torch.matmul(q_local, k_current.transpose(-2, -1)) / scale
    # Online softmax: 增量更新
    curr_max = scores.max(dim=-1, keepdim=True).values
    new_max = torch.max(prev_max, curr_max)
    # 重新缩放之前的累积值
    corr = torch.exp(prev_max - new_max)
    curr_exp = torch.exp(scores - new_max)
    new_sum = corr * prev_sum + curr_exp.sum(dim=-1, keepdim=True)
    new_out = corr * prev_out + torch.matmul(curr_exp, v_current)
    return new_out / new_sum, new_max, new_sum
```

### 因果掩码调整

标准因果掩码假设完整序列。切分后需要考虑全局位置：

```python
def create_cp_causal_mask(local_seq_len, rank, world_size):
    """为 CP 切分后的子序列生成因果掩码"""
    global_start = rank * local_seq_len
    # 全局位置范围: [global_start, global_start + local_seq_len)
    row_pos = torch.arange(global_start, global_start + local_seq_len)
    col_pos = torch.arange(0, (rank + 1) * local_seq_len)  # 只能看到当前及之前的 KV
    mask = row_pos.unsqueeze(1) >= col_pos.unsqueeze(0)
    return mask
```

## 架构图解

### Ring Attention 数据流

```
时间 →
GPU0: Q0 ←K0,V0→  Q0 ←K1,V1→  Q0 ←K2,V2→  Q0 ←K3,V3→
         rotate KV →   rotate KV →   rotate KV →

每个箭头: 计算 Attn(Q_local, K_recv, V_recv) + Online Softmax 累积
```

## 动手实践

→ [notebook 09: 上下文并行](../../notebooks/09_expert_and_context_parallel.ipynb)

## 延伸阅读

- Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023)
- Jacobs et al., "Systematic Generalization with Edge Transformers" (2023)

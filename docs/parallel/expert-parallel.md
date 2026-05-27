# 专家并行详解

> 上一篇：[流水线并行](pipeline-parallel.md) ｜ 下一篇：[上下文并行](context-parallel.md)

## 概述

专家并行（Expert Parallelism, EP）将 MoE 模型中的不同 Expert 分配到不同 GPU，通过 All-to-All 通信完成 token 的分发和收集。它与数据并行互补：在 expert 维度做 EP，在非 expert 维度做 DP。

**前置知识：** [MoE 架构](../models/deepseek-v3.md)、[通信原语 — All-to-All](communication.md)
**代码位置：** [`parallel/expert_parallel/`](../../parallel/expert_parallel/)

## 核心原理

### Expert 分配

将 E 个 expert 均匀分配到 P 个 GPU：

```python
def partition_experts(rank, num_experts, world_size):
    experts_per_gpu = num_experts // world_size
    start = rank * experts_per_gpu
    return list(range(start, start + experts_per_gpu))
```

例如：8 expert, 4 GPU → GPU0: [0,1], GPU1: [2,3], GPU2: [4,5], GPU3: [6,7]

### Token Dispatch 流程

```
1. Router 计算: 每个 token 选择 Top-K expert
2. All-to-All Dispatch: 将 token 按 expert 所在 GPU 分组发送
3. Expert 计算: 各 GPU 上的 expert 处理收到的 token
4. All-to-All Gather: 将处理结果发送回原 GPU
```

代码对应（`token_dispatch.py:12-45`）：

```python
def dispatch_tokens_to_experts(tokens, expert_indices, num_experts, world_size):
    """按 expert 索引分组 token"""
    groups = [[] for _ in range(num_experts)]
    for i, idx in enumerate(expert_indices):
        groups[idx].append(tokens[i])
    return groups

def all_to_all_dispatch_example(local_tokens, rank, world_size):
    """All-to-All: 每个 GPU 发送/接收 token"""
    send_counts = [len(local_tokens[i]) for i in range(world_size)]
    recv_counts = [0] * world_size
    dist.all_to_all_single(
        torch.tensor(recv_counts), torch.tensor(send_counts))
    # 按 send_counts/recv_counts 交换实际数据
```

### 与数据并行的组合

典型配置：EP 在 expert 维度，DP 在数据维度：

```
MoE 模型 (8 expert, 4 GPU):
  EP: GPU0: expert 0,1  GPU1: expert 2,3  GPU2: expert 4,5  GPU3: expert 6,7
  DP: 每个 GPU 处理不同的数据 batch（同步非 expert 参数的梯度）
```

## 动手实践

→ [notebook 09: 专家并行](../../notebooks/09_expert_and_context_parallel.ipynb)

## 延伸阅读

- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" (2020)
- Rajbhandari et al., "Mixture-of-Experts Meets Instruction Tuning" (2024)

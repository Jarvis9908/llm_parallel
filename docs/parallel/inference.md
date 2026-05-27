# 推理并行详解

> 上一篇：[上下文并行](context-parallel.md) ｜ 返回：[并行策略总览](overview.md)

## 概述

推理阶段的并行优化与训练不同：没有反向传播，但面临 KV Cache 显存瓶颈和自回归生成的低效问题。本篇详解三种推理优化策略：KV Cache 分片、Prefill/Decode 分离、投机解码。

**前置知识：** [KV Cache](../models/llama3.md)、[通信原语](communication.md)
**代码位置：** [`parallel/inference/`](../../parallel/inference/)

## 核心原理

### KV Cache 分片

按 head 维度将 KV Cache 分到多个 GPU：

```
KV Cache 总大小 = 2 × n_layers × n_heads × seq_len × head_dim × dtype
每 GPU 大小 = 总大小 / num_gpus
```

代码对应（`kv_cache_shard.py:12-50`）：

```python
def shard_kv_cache_by_heads(kv_cache, rank, world_size):
    """按 head 维度切分 KV Cache"""
    n_heads = kv_cache.shape[1]
    heads_per_gpu = n_heads // world_size
    start = rank * heads_per_gpu
    return kv_cache[:, start:start + heads_per_gpu, :, :]

def kv_cache_memory_analysis(num_layers, num_heads, seq_len, head_dim, num_gpus):
    total_bytes = 2 * num_layers * num_heads * seq_len * head_dim * 2  # fp16
    per_gpu_bytes = total_bytes / num_gpus
    return {
        'total_mb': total_bytes / 1e6,
        'per_gpu_mb': per_gpu_bytes / 1e6,
        'savings_ratio': 1 - 1 / num_gpus
    }
```

### Prefill vs Decode

推理分两个阶段，瓶颈不同：

| 阶段 | 操作 | 瓶颈 | 适合策略 |
|------|------|------|---------|
| Prefill | 处理完整输入 prompt | Compute-bound | Tensor Parallel（大矩阵乘） |
| Decode | 逐 token 生成 | Memory-bound | KV Cache 分片（大量 KV 访问） |

代码对应（`prefill_decode.py:12-60`）：

```python
def analyze_prefill_characteristics(seq_len, dim, num_gpus):
    """Prefill: 计算密集型"""
    flops = 2 * seq_len * seq_len * dim  # Attention FLOPS
    return {'flops': flops, 'type': 'compute_bound',
            'recommendation': 'Tensor Parallel'}

def analyze_decode_characteristics(seq_len, dim, num_gpus):
    """Decode: 访存密集型"""
    kv_cache_size = 2 * seq_len * dim  # 简化
    return {'kv_cache_bytes': kv_cache_size, 'type': 'memory_bound',
            'recommendation': 'KV Cache 分片 + 多 batch'}
```

### Speculative Decoding

用小模型（Draft）快速生成候选 token，大模型（Target）一次性验证：

```
标准自回归: Target 生成 5 个 token → 5 次前向传播
投机解码:   Draft 生成 5 个候选 → Target 1 次前向验证全部
如果全部接受: 同样质量，但只需 1 次大模型前向
```

加速比取决于接受率：

$$\text{speedup} = \frac{1}{1 - r^k}$$

其中 $r$ 为接受率，$k$ 为 draft token 数。

代码对应（`speculative_decoding.py:12-60`）：

```python
def draft_generate(draft_model, input_ids, num_draft_tokens):
    """小模型快速生成候选 token"""
    candidates = []
    for _ in range(num_draft_tokens):
        logits = draft_model(input_ids)
        next_token = sample(logits[:, -1, :])
        candidates.append(next_token)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return torch.cat(candidates, dim=1)

def target_verify(target_model, input_ids, candidates):
    """大模型一次性验证所有候选"""
    full_input = torch.cat([input_ids, candidates], dim=1)
    logits = target_model(full_input)  # 一次前向!
    # 比较每个位置的 argmax 与候选 token
    accepted = []
    for i in range(candidates.shape[1]):
        pos = input_ids.shape[1] + i
        if logits[:, pos-1, :].argmax(-1) == candidates[:, i]:
            accepted.append(candidates[:, i])
        else:
            break  # 拒绝后续所有
    return torch.stack(accepted, dim=1) if accepted else None

def speedup_analysis(accept_rate, draft_tokens):
    return 1 / (1 - accept_rate ** draft_tokens)
```

## 动手实践

→ [notebook 10: 推理并行](../../notebooks/10_inference_parallel.ipynb)

推荐练习：
1. 计算不同序列长度下的 KV Cache 显存需求
2. 分析 accept_rate 对投机解码加速比的影响
3. 对比 Prefill 和 Decode 阶段的计算特征

## 延伸阅读

- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) — vLLM

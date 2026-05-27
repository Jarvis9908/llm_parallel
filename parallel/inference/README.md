# 推理并行模块

推理阶段的并行优化策略，解决 KV Cache 显存瓶颈和自回归生成效率问题。

## 文件说明

| 文件 | 功能 | 关键内容 |
|------|------|---------|
| `kv_cache_shard.py` | KV Cache 分片 | `shard_kv_cache_by_heads`, `gather_kv_cache`, `kv_cache_memory_analysis` |
| `prefill_decode.py` | Prefill/Decode 分析 | `analyze_prefill_characteristics`, `analyze_decode_characteristics`, `recommend_strategy` |
| `speculative_decoding.py` | 投机解码 | `draft_generate`, `target_verify`, `speedup_analysis` |

## 核心概念

### KV Cache 显存问题

自回归生成时，每个 token 都需要缓存之前所有 token 的 K 和 V。对于长序列：

```
KV Cache 内存 = 2 × num_layers × num_heads × seq_len × head_dim × dtype_size
```

KV Cache 分片：按 head 维度将 KV Cache 分到多个 GPU，减少单卡显存。

### Prefill vs Decode

| 阶段 | 特点 | 瓶颈 | 适合策略 |
|------|------|------|---------|
| Prefill | 处理完整输入，计算密集 | Compute | Tensor Parallel |
| Decode | 逐 token 生成，访存密集 | Memory | KV Cache 分片 |

### Speculative Decoding

用小模型（Draft）快速生成候选 token，大模型（Target）一次性验证：
```
Draft: 生成 K 个候选 token（快但不准）
Target: 一次前向验证所有候选（慢但准）
如果全部接受，一次 Target 前向生成了 K 个 token
```

加速比取决于接受率，通常可达 2-3x。

## 快速开始

```python
from parallel.inference.kv_cache_shard import kv_cache_memory_analysis
from parallel.inference.speculative_decoding import speedup_analysis

# 分析 KV Cache 内存需求
mem = kv_cache_memory_analysis(num_layers=32, num_heads=32, seq_len=4096,
                                head_dim=128, num_gpus=4)
print(f"每 GPU KV Cache: {mem['per_gpu_mb']:.0f} MB")

# 分析投机解码加速比
speedup = speedup_analysis(accept_rate=0.8, draft_tokens=5)
print(f"理论加速比: {speedup:.2f}x")
```

## 详细文档

→ [并行策略总览](../../docs/parallel/overview.md)
→ [notebook 10: 推理并行](../../notebooks/10_inference_parallel.ipynb)

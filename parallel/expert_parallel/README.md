# 专家并行模块

将 MoE 模型中的不同 Expert 分配到不同 GPU，通过 All-to-All 通信完成 token 分发和收集。

## 文件说明

| 文件 | 功能 | 关键内容 |
|------|------|---------|
| `expert_partition.py` | 专家分配 | `partition_experts`, `get_expert_owner` — 计算每个 rank 拥有的专家 |
| `token_dispatch.py` | Token 分发 | `dispatch_tokens_to_experts`, `all_to_all_dispatch_example` — 按路由结果分发 token |

## 核心概念

### Expert Parallel 流程

```
1. Router 计算每个 token 的 expert 选择
2. All-to-All Dispatch: 将 token 发送到对应 expert 所在的 GPU
3. 各 GPU 上的 expert 处理收到的 token
4. All-to-All Gather: 将处理结果收集回原 GPU
```

### 与数据并行的区别

- 数据并行：每个 GPU 有**完整模型**（所有 expert），同步梯度
- 专家并行：每个 GPU 只有**部分 expert**，通过 All-to-All 交换 token

通常组合使用：在 expert 维度做专家并行，在非 expert 维度做数据并行。

## 快速开始

```python
from parallel.expert_parallel.expert_partition import partition_experts, get_expert_owner

# 8 个 expert 分配到 4 个 GPU
for rank in range(4):
    experts = partition_experts(rank=rank, num_experts=8, world_size=4)
    print(f"Rank {rank}: experts {experts}")

# 查看某个 expert 在哪个 GPU 上
owner = get_expert_owner(expert_idx=3, num_experts=8, world_size=4)
print(f"Expert 3 is on rank {owner}")
```

## 详细文档

→ [并行策略总览](../../docs/parallel/overview.md)
→ [notebook 09: 专家并行](../../notebooks/09_expert_and_context_parallel.ipynb)

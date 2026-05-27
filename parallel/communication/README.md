# 通信原语模块

分布式训练的基础——集合通信原语。所有并行策略都建立在这些原语之上。

## 文件说明

| 文件 | 功能 | 关键内容 |
|------|------|---------|
| `primitives.py` | 通信原语手写实现 | `naive_all_reduce`, `ring_all_reduce`, `naive_all_gather`, `naive_broadcast`, `naive_reduce_scatter` |
| `setup.py` | 分布式环境管理 | `init_process_group`, `get_rank`, `get_world_size`, `cleanup` |
| `topologies.py` | 通信拓扑分析 | `analyze_ring_topology`, `analyze_tree_topology`, `analyze_mesh_topology`, `visualize_topology` |

## 核心概念

### 集合通信原语

| 原语 | 含义 | 复杂度 |
|------|------|--------|
| Broadcast | 一对多广播 | O(N) |
| AllReduce | 全局归约 + 广播 | Naive: O(N*P²), Ring: O(2N) |
| AllGather | 收集所有数据拼接 | O(N*(P-1)) |
| ReduceScatter | 归约后分散 | O(N*(P-1)) |

### Ring AllReduce

最经典的梯度同步算法，分两个阶段：
1. **Scatter-Reduce**: 数据沿 Ring 传递，每步做局部 reduce
2. **AllGather**: 归约结果沿 Ring 广播

总通信量 = `2 * (P-1)/P * N ≈ 2N`，与 GPU 数量基本无关。

## 快速开始

```python
from parallel.communication.primitives import naive_all_reduce, ring_all_reduce
from parallel.communication.topologies import analyze_ring_topology, visualize_topology

# 分析 Ring AllReduce 通信成本
cost = analyze_ring_topology(data_size_mb=100, num_gpus=8, bandwidth_gbps=25)
print(f"Ring AllReduce 时间: {cost['total_time_ms']:.2f} ms")

# 可视化拓扑
visualize_topology('ring', 4)
```

## 详细文档

→ [并行策略总览](../../docs/parallel/overview.md)
→ [notebook 05: 通信原语](../../notebooks/05_communication_primitives.ipynb)

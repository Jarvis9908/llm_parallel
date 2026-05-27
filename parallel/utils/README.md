# 并行工具模块

分布式并行的辅助工具：通信模拟、张量分片、可视化。

## 文件说明

| 文件 | 功能 | 关键内容 |
|------|------|---------|
| `comm_simulator.py` | 通信量模拟器 | `simulate_all_reduce`, `simulate_all_to_all`, `compare_parallel_strategies` |
| `shard_utils.py` | 张量分片工具 | `split_tensor`, `gather_tensor`, `visualize_sharding` |
| `visualizer.py` | 可视化工具 | `plot_topology`, `plot_bubble_time` |

## 快速开始

```python
from parallel.utils.comm_simulator import simulate_all_reduce, compare_parallel_strategies
from parallel.utils.visualizer import plot_topology, plot_bubble_time

# 模拟 Ring AllReduce 通信时间
time_ms = simulate_all_reduce(data_size_mb=256, num_gpus=8, bandwidth_gbps=25)
print(f"Ring AllReduce: {time_ms:.2f} ms")

# 对比不同并行策略的通信开销
compare_parallel_strategies(model_params_gb=7, seq_len=2048, batch_size=32, num_gpus=8)
```

## 详细文档

→ [并行策略总览](../../docs/parallel/overview.md)

# 流水线并行模块

将模型按层切分到不同 GPU，通过流水线方式执行多个 micro-batch，减少单卡显存占用。

## 文件说明

| 文件 | 功能 | 关键内容 |
|------|------|---------|
| `layer_partition.py` | 层分配 | `partition_layers`, `get_layer_range` — 将 Transformer 层分到各 rank |
| `gpiped.py` | GPipe 调度 | `gpiped_forward`, `compute_gpipe_bubble_time` — 全前向后全反向 |
| `f1b1.py` | 1F1B 调度 | `f1b1_schedule`, `compute_1f1b_bubble_time` — 交替前向反向 |

## 核心概念

### Bubble 问题

```
GPipe 调度 (4 个 micro-batch, 4 个 stage):
  GPU0: [F0][F1][F2][F3]          [B0][B1][B2][B3]
  GPU1:     [F0][F1][F2][F3]      [B0][B1][B2][B3]
  GPU2:         [F0][F1][F2][F3]  [B0][B1][B2][B3]
  GPU3:             [F0][F1][F2][F3][B0][B1][B2][B3]
                ↑ 空闲等待 = Bubble

1F1B 调度 (交替执行减少 Bubble):
  GPU0: [F0][F1][F2][F3][B0][F4][B1][F5][B2]...[B3]
  GPU1:     [F0][F1][F2][B0][F3][B1][F4][B2]...
  稳态阶段每个 GPU 同时有一个前向和一个反向在执行
```

Bubble 比例：
- GPipe: `(P-1) / (P-1+M)`，M 为 micro-batch 数
- 1F1B: `(P-1) / (P-1+M)`，但峰值显存更低

## 快速开始

```python
from parallel.pipeline_parallel.layer_partition import get_layer_range
from parallel.pipeline_parallel.gpiped import compute_gpipe_bubble_time

# 查看 rank=1 负责哪些层（12 层模型，4 个 GPU）
start, end = get_layer_range(rank=1, total_layers=12, world_size=4)
print(f"Rank 1: layers [{start}, {end})")  # layers [3, 6)

# 计算 GPipe bubble 比例
bubble_ratio = compute_gpipe_bubble_time(num_stages=4, num_micro_batches=8)
print(f"GPipe bubble 比例: {bubble_ratio:.2%}")
```

## 详细文档

→ [并行策略总览](../../docs/parallel/overview.md)
→ [notebook 08: 流水线并行](../../notebooks/08_pipeline_parallel.ipynb)

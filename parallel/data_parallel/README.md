# 数据并行模块

最基础的并行策略：数据切分到多卡，每卡有完整模型副本，通过同步梯度保持一致。

## 文件说明

| 文件 | 功能 | 关键内容 |
|------|------|---------|
| `dp.py` | 原始数据并行 | `sync_gradients_naive` — 逐参数 AllReduce 梯度 |
| `ddp.py` | 分布式数据并行概念 | `broadcast_model`, `gradient_bucket_sync` — 梯度桶 + 计算通信重叠 |
| `gradient_accumulation.py` | 梯度累积 | `GradientAccumulator`, `compute_effective_batch_size` |

## 核心概念

### DP → DDP 的演进

```
DP (DataParallel):
  每个 step: Forward → AllReduce(grad) → Update
  问题: 梯度同步阻塞，GPU 空等

DDP (DistributedDataParallel):
  梯度桶: 将多个参数的梯度打包成桶，减少通信次数
  重叠: 反向传播时，后层梯度计算完立即同步，与前层计算重叠
```

### 梯度累积

将大 batch 拆成多个 micro-batch，累积多步梯度后再同步更新。效果等价于大 batch 训练，但显存占用按 micro-batch 计算。

有效 batch_size = micro_batch_size × accumulation_steps × world_size

## 快速开始

```python
from parallel.data_parallel.dp import sync_gradients_naive
from parallel.data_parallel.gradient_accumulation import compute_effective_batch_size

# 计算有效 batch size
eff_bs = compute_effective_batch_size(micro_batch=4, accumulation_steps=8, world_size=4)
print(f"有效 batch size: {eff_bs}")  # 128
```

## 详细文档

→ [并行策略总览](../../docs/parallel/overview.md)
→ [notebook 06: 数据并行](../../notebooks/06_data_parallel.ipynb)

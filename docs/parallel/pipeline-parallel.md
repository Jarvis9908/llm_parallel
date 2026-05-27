# 流水线并行详解

> 上一篇：[张量并行](tensor-parallel.md) ｜ 下一篇：[专家并行](expert-parallel.md)

## 概述

流水线并行（Pipeline Parallelism, PP）将模型按层切分到不同 GPU，通过流水线方式执行多个 micro-batch，让多个 GPU 同时工作，减少单卡显存占用。

**前置知识：** [通信原语](communication.md)、micro-batch 概念
**代码位置：** [`parallel/pipeline_parallel/`](../../parallel/pipeline_parallel/)

## 核心原理

### 层分配

将 L 层模型均匀分配到 P 个 GPU：

```python
def get_layer_range(rank, total_layers, world_size):
    layers_per_rank = total_layers // world_size
    start = rank * layers_per_rank
    end = start + layers_per_rank
    return start, end
```

例如：12 层模型，4 个 GPU → GPU0: [0,3), GPU1: [3,6), GPU2: [6,9), GPU3: [9,12)

### GPipe 调度

最简单的策略：所有 micro-batch 先全部前向传播，再全部反向传播。

```
4 micro-batch, 4 stage:

GPU0: [F0][F1][F2][F3]          [B0][B1][B2][B3]
GPU1:     [F0][F1][F2][F3]      [B0][B1][B2][B3]
GPU2:         [F0][F1][F2][F3]  [B0][B1][B2][B3]
GPU3:             [F0][F1][F2][F3][B0][B1][B2][B3]
```

**Bubble 问题：** GPU0-F3 完成后到 GPU3-F0 开始之间，GPU0-2 都在空等。这段时间称为 Bubble。

Bubble 比例 = $\frac{P-1}{P-1+M}$，其中 M 为 micro-batch 数。M 越大，Bubble 比例越小。

代码对应（`gpiped.py:12-35`）：

```python
def gpiped_forward(micro_batches, rank, world_size):
    results = []
    for mb in micro_batches:
        if rank == 0:
            out = layer(mb)
        else:
            out = recv_from_prev_rank()  # 接收上一个 stage 的输出
            out = layer(out)
        if rank < world_size - 1:
            send_to_next_rank(out)       # 发送给下一个 stage
        results.append(out)
    return results

def compute_gpipe_bubble_time(num_stages, num_micro_batches):
    return (num_stages - 1) / (num_stages - 1 + num_micro_batches)
```

### 1F1B 调度

交替执行前向和反向，在稳态阶段每个 GPU 同时有一个前向和一个反向在执行：

```
4 micro-batch, 4 stage (warmup=3):

GPU0: [F0][F1][F2][F3][B0][F4][B1]...[B3]
GPU1:     [F0][F1][F2][B0][F3][B1]...
GPU2:         [F0][F1][B0][F2][B1]...
GPU3:             [F0][B0][F1][B1]...
```

三个阶段：
1. **Warmup**：前向传播 ramp up，填充流水线
2. **Steady**：交替前向/反向，GPU 利用率最高
3. **Cooldown**：反向传播 drain，清空流水线

**优势：** Bubble 比例相同，但峰值激活值内存更低（反向传播更早开始，释放激活值）。

代码对应（`f1b1.py:12-62`）：

```python
def f1b1_schedule(rank, world_size, num_micro_batches):
    warmup = world_size - 1
    # Warmup: 纯前向
    for i in range(warmup):
        forward(micro_batch[i])
    # Steady: 交替
    for i in range(warmup, num_micro_batches):
        backward(micro_batch[i - warmup])  # 反向最早的那个
        forward(micro_batch[i])             # 前向新的
    # Cooldown: 纯反向
    for i in range(num_micro_batches - warmup, num_micro_batches):
        backward(micro_batch[i])
```

## 架构图解

### Bubble 可视化

```
GPipe (M=4, P=4):  Bubble = 3/7 ≈ 43%
  ▓▓▓░░░░░░░░▓▓▓▓
    ▓▓▓░░░░░░░░▓▓▓▓
      ▓▓▓░░░░░░░░▓▓▓▓
        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓

1F1B (M=4, P=4):  Bubble = 3/7 ≈ 43% (但峰值显存更低)
  ▓▓▓░▓░▓░▓░▓░▓░▓
    ▓▓▓░▓░▓░▓░▓░▓
      ▓▓░▓░▓░▓░▓░▓
        ▓░▓░▓░▓░▓░▓

▓ = 计算, ░ = 空闲 (Bubble)
```

## 动手实践

→ [notebook 08: 流水线并行](../../notebooks/08_pipeline_parallel.ipynb)

推荐练习：
1. 计算不同 micro-batch 数下的 Bubble 比例
2. 对比 GPipe 和 1F1B 的峰值激活值内存
3. 分析层数分配不均匀对 Bubble 的影响

## 延伸阅读

- Huang et al., "Efficient Training of Giant Neural Networks using Pipeline Parallelism" (2019) — GPipe
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021) — 1F1B

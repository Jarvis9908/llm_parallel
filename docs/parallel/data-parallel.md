# 数据并行详解

> 上一篇：[通信原语](communication.md) ｜ 下一篇：[张量并行](tensor-parallel.md)

## 概述

数据并行是最基础、最常用的分布式并行策略。核心思想：将训练数据切分到多个 GPU，每个 GPU 持有完整的模型副本，前向和反向计算独立进行，通过同步梯度保持模型一致性。

**前置知识：** [通信原语](communication.md)（AllReduce, Broadcast）
**代码位置：** [`parallel/data_parallel/`](../../parallel/data_parallel/)

## 核心原理

### Naive Data Parallel (DP)

最基本的实现：每个 GPU 持有完整模型副本，每个 step 后 AllReduce 同步梯度。

```
每个 step:
  1. 每个 GPU: Forward(batch_i) → loss_i → Backward → grad_i
  2. AllReduce(grad) → grad_avg = Σgrad_i / P
  3. 每个 GPU: param = param - lr * grad_avg
```

**问题：** 反向传播完成后，所有 GPU 必须等待梯度同步完成才能继续，GPU 空等。

代码对应（`dp.py:12-40`）：

```python
def sync_gradients_naive(model, rank, world_size):
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size  # 取平均
```

### Distributed Data Parallel (DDP)

DDP 通过两个关键优化解决 DP 的效率问题：

#### 梯度桶（Gradient Bucketing）

将多个小参数的梯度打包成一个桶，减少通信次数：

```
无桶: 1000 个参数 → 1000 次 AllReduce
有桶: 1000 个参数 → 10 次 AllReduce（每桶 100 个参数）
```

桶按参数大小从大到小排序，确保大的梯度先被同步。

#### 计算-通信重叠

反向传播从最后一层开始。当最后一层的梯度计算完成后，立即启动该层梯度的 AllReduce，同时继续计算前一层的梯度：

```
时间线:
  Layer N backward → grad_N ready → AllReduce(grad_N) ──→
  Layer N-1 backward → grad_{N-1} ready → AllReduce(grad_{N-1}) ──→
  ...
  Layer 1 backward → grad_1 ready → AllReduce(grad_1) ──→ 完成
```

通信和计算重叠，总时间 ≈ max(计算时间, 通信时间)。

代码对应（`ddp.py:30-80`）：

```python
def gradient_bucket_sync(model, rank, world_size, bucket_size_mb=25):
    # 按参数大小排序，组成桶
    buckets = _form_buckets(model.parameters(), bucket_size_mb)
    handles = []
    for bucket in buckets:
        # 反向传播到这个桶时，立即启动 AllReduce
        flat_grad = torch.cat([p.grad.flatten() for p in bucket])
        handle = dist.all_reduce(flat_grad, async_op=True)  # 异步!
        handles.append(handle)
    # 等待所有通信完成
    for h in handles:
        h.wait()
```

### 梯度累积（Gradient Accumulation）

当 GPU 显存不够放完整 batch 时，将大 batch 拆成多个 micro-batch，累积多步梯度后再更新：

```
有效 batch_size = micro_batch_size × accumulation_steps × world_size

Step 1: Forward(micro_batch_1) → grad_1 (累积, 不更新)
Step 2: Forward(micro_batch_2) → grad_2 (累积到 grad_1)
...
Step N: Forward(micro_batch_N) → grad_N (累积) → AllReduce → Update
```

代码对应（`gradient_accumulation.py:12-50`）：

```python
class GradientAccumulator:
    def __init__(self, model, accumulation_steps):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.step = 0
    def should_sync(self):
        self.step += 1
        return self.step % self.accumulation_steps == 0

def compute_effective_batch_size(micro_batch, accumulation_steps, world_size):
    return micro_batch * accumulation_steps * world_size
```

## 架构图解

### DP vs DDP 通信模式

```
DP (Naive):
  GPU0: [F][B]──sync──[F][B]──sync──
  GPU1: [F][B]──sync──[F][B]──sync──
  GPU2: [F][B]──sync──[F][B]──sync──
  GPU3: [F][B]──sync──[F][B]──sync──
        ↑ 计算    ↑ 同步阻塞

DDP (重叠):
  GPU0: [F][B→sync→][F][B→sync→]
  GPU1: [F][B→sync→][F][B→sync→]
  GPU2: [F][B→sync→][F][B→sync→]
  GPU3: [F][B→sync→][F][B→sync→]
        ↑ 同步与计算重叠，无空等
```

## 动手实践

→ [notebook 06: 数据并行](../../notebooks/06_data_parallel.ipynb)

推荐练习：
1. 计算不同 world_size 下的有效 batch size
2. 对比有无梯度累积的显存占用
3. 分析通信量随模型参数量的变化

## 延伸阅读

- Li et al., "PyTorch Distributed: Experiences on Accelerating Data Parallel Training" (2020) — DDP 论文
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020)

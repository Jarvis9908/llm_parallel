# 通信原语详解

> 上一篇：[并行策略总览](overview.md) ｜ 下一篇：[数据并行](data-parallel.md)

## 概述

集合通信（Collective Communication）是分布式并行的基础。所有并行策略——数据并行、张量并行、流水线并行——最终都需要通过通信原语在 GPU 之间交换数据。本篇详解 5 种核心通信原语的实现原理和通信拓扑分析。

**前置知识：** 分布式计算基本概念（rank, world_size）
**代码位置：** [`parallel/communication/`](../../parallel/communication/)

## 核心原理

### Broadcast（广播）

将一个 rank 的数据发送到所有其他 rank。

$$\text{Broadcast}(x_{src}, src) \rightarrow x_{src} \text{ at all ranks}$$

- 通信量：$O(N)$，N 为数据大小
- 用途：模型初始化时将 rank 0 的参数同步到所有 rank

代码对应（`primitives.py:12-28`）：

```python
def naive_broadcast(tensor, src_rank, rank, world_size):
    if rank == src_rank:
        for i in range(world_size):
            if i != rank:
                dist.send(tensor, dst=i)  # 发送给所有其他 rank
    else:
        dist.recv(tensor, src=src_rank)   # 从源 rank 接收
```

### AllReduce（全局归约）

将所有 rank 的数据做归约操作（如求和），结果广播到所有 rank。

$$\text{AllReduce}(x_0, x_1, ..., x_{P-1}) \rightarrow \sum_{i=0}^{P-1} x_i \text{ at all ranks}$$

**Naive 实现：** 每个 rank 依次向其他 rank 发送数据，复杂度 $O(N \times P^2)$。

**Ring AllReduce：** 分两个阶段：
1. **Scatter-Reduce**（P-1 步）：每个 rank 发送 $N/P$ 数据给下一个 rank，做局部 reduce
2. **AllGather**（P-1 步）：归约后的数据沿 Ring 广播

总通信量 = $2 \times \frac{P-1}{P} \times N \approx 2N$，与 GPU 数量基本无关。

代码对应（`primitives.py:60-110`）：

```python
def ring_all_reduce(tensor, rank, world_size):
    chunk_size = tensor.numel() // world_size
    # 阶段 1: Scatter-Reduce
    for step in range(world_size - 1):
        send_chunk = (rank - step) % world_size
        recv_chunk = (rank - step - 1) % world_size
        send(tensor[send_chunk * chunk_size:(send_chunk + 1) * chunk_size],
             dst=(rank + 1) % world_size)
        recv(tensor[recv_chunk * chunk_size:(recv_chunk + 1) * chunk_size],
             src=(rank - 1) % world_size)
        # 局部 reduce
    # 阶段 2: AllGather（类似过程，但只发送不 reduce）
```

### AllGather（全局收集）

将所有 rank 的数据收集拼接到每个 rank。

$$\text{AllGather}(x_i) \rightarrow [x_0, x_1, ..., x_{P-1}] \text{ at all ranks}$$

- 通信量：$O(N \times (P-1))$
- 用途：张量并行中收集各 rank 的部分输出

### ReduceScatter（归约分散）

先做全局归约，然后将结果分散到各 rank。

$$\text{ReduceScatter}(x_0, ..., x_{P-1}) \rightarrow \left(\sum x_i\right)_j \text{ at rank } j$$

- 通信量：$O(N \times (P-1))$
- 用途：张量并行中梯度的归约处理

### All-to-All（全交换）

每个 rank 将数据分成 P 份，第 i 份发送给 rank i。

$$\text{All-to-All}: \text{rank } i \text{ sends chunk } j \text{ to rank } j$$

- 通信量：$O(N)$（每个 rank 发送和接收各 N/P × (P-1)）
- 用途：专家并行中的 token 分发

## 通信拓扑分析

### Ring 拓扑

```
GPU0 → GPU1 → GPU2 → GPU3 → GPU0
```

- AllReduce 时间：$2 \times \frac{P-1}{P} \times \frac{N}{B}$，B 为带宽
- 带宽最优（总通信量接近 2N），但延迟与 P 成正比
- 适合：大数据量、对带宽敏感的场景（如数据并行梯度同步）

### Tree 拓扑

```
        GPU0
       /    \
    GPU1    GPU2
    /  \
  GPU3  GPU4
```

- Broadcast 时间：$O(\log P \times (N/B + L))$，L 为延迟
- 延迟最优（$\log P$ 步），但带宽有瓶颈（根节点）
- 适合：小数据量、对延迟敏感的场景（如参数广播）

### Mesh 拓扑（NVLink/NVSwitch）

```
GPU0 — GPU1
 |       |
GPU2 — GPU3
```

- 2D Mesh：AllReduce 分行/列两步，通信量 $2\sqrt{P} \times N$
- NVSwitch 提供全连接，延迟极低
- 适合：张量并行（需要频繁的小数据量通信）

代码对应（`topologies.py`）：

```python
def analyze_ring_topology(data_size_mb, num_gpus, bandwidth_gbps):
    # Ring AllReduce: 2*(P-1)/P * N/B
    comm_bytes = 2 * (num_gpus - 1) / num_gpus * data_size_mb * 1e6
    bandwidth_bytes = bandwidth_gbps * 1e9 / 8
    return {'total_time_ms': comm_bytes / bandwidth_bytes * 1000}
```

## 动手实践

→ [notebook 05: 通信原语](../../notebooks/05_communication_primitives.ipynb)

推荐练习：
1. 对比 naive_all_reduce 和 ring_all_reduce 的通信量
2. 修改 GPU 数量观察 Ring AllReduce 时间变化
3. 可视化不同拓扑的通信路径

## 延伸阅读

- Thakur et al., "Optimization of Collective Communication Operations in MPICH" (2005)
- NVIDIA NCCL 文档 — 工业级集合通信库

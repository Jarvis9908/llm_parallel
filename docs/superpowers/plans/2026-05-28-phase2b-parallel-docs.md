# Phase 2b: Parallel Strategy Detailed Docs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create 7 detailed parallel strategy documentation files covering communication primitives through inference optimization, with math formulas, code mappings, and architecture diagrams.

**Architecture:** Each doc follows the same template as Phase 2a (概述 → 核心原理 → 架构图解 → 代码实现分析 → 对比 → 动手实践 → 延伸阅读). Chinese-English mixed language.

**Tech Stack:** Markdown, LaTeX math, ASCII diagrams

---

## File Structure

```
docs/parallel/
├── communication.md        # Task 1 — AllReduce, AllGather, topology analysis
├── data-parallel.md        # Task 2 — DP, DDP, gradient accumulation
├── tensor-parallel.md      # Task 3 — Column/Row parallel, Sequence Parallel, Megatron
├── pipeline-parallel.md    # Task 4 — GPipe, 1F1B, bubble analysis
├── expert-parallel.md      # Task 5 — Expert partition, token dispatch, All-to-All
├── context-parallel.md     # Task 6 — Ring Attention, sequence partition
└── inference.md            # Task 7 — KV Cache sharding, prefill/decode, speculative
```

---

## Task 1: Create `docs/parallel/communication.md`

**Files:**
- Create: `docs/parallel/communication.md`

- [ ] **Step 1: Create the file with full content**

```markdown
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
```

- [ ] **Step 2: Verify and commit**

```bash
git add docs/parallel/communication.md
git commit -m "docs: add communication primitives detailed documentation"
```

---

## Task 2: Create `docs/parallel/data-parallel.md`

**Files:**
- Create: `docs/parallel/data-parallel.md`

- [ ] **Step 1: Create the file with full content**

```markdown
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
```

- [ ] **Step 2: Verify and commit**

```bash
git add docs/parallel/data-parallel.md
git commit -m "docs: add data parallel detailed documentation"
```

---

## Task 3: Create `docs/parallel/tensor-parallel.md`

**Files:**
- Create: `docs/parallel/tensor-parallel.md`

- [ ] **Step 1: Create the file with full content**

```markdown
# 张量并行详解

> 上一篇：[数据并行](data-parallel.md) ｜ 下一篇：[流水线并行](pipeline-parallel.md)

## 概述

张量并行（Tensor Parallelism, TP）将单层的权重矩阵切分到多个 GPU，每个 GPU 只计算模型的一部分。当单层计算量超过单 GPU 容量时，TP 是必需的策略。

**前置知识：** [通信原语](communication.md)（AllReduce, AllGather）、矩阵乘法
**代码位置：** [`parallel/tensor_parallel/`](../../parallel/tensor_parallel/)

## 核心原理

### Column Parallel Linear

将权重矩阵 W 按列（输出维度）切分：

$$W = [W_0 | W_1 | ... | W_{P-1}], \quad W_i \in \mathbb{R}^{d_{in} \times d_{out}/P}$$

每个 GPU 计算 $Y_i = XW_i$，然后通过 AllGather 拼接完整输出：

$$Y = [Y_0, Y_1, ..., Y_{P-1}] = X[W_0 | W_1 | ... | W_{P-1}] = XW$$

代码对应（`column_parallel.py:12-40`）：

```python
def split_weight_column(weight, rank, world_size):
    chunk_size = weight.shape[1] // world_size  # 按输出维度切
    return weight[:, rank * chunk_size:(rank + 1) * chunk_size]

def column_parallel_linear(x, weight_local, rank, world_size):
    # 每个 GPU 计算部分输出
    y_local = torch.matmul(x, weight_local)
    # AllGather 拼接完整输出
    y_all = [torch.zeros_like(y_local) for _ in range(world_size)]
    dist.all_gather(y_all, y_local)
    return torch.cat(y_all, dim=-1)  # 拼接列
```

### Row Parallel Linear

将权重矩阵 W 按行（输入维度）切分：

$$W = [W_0; W_1; ...; W_{P-1}], \quad W_i \in \mathbb{R}^{d_{in}/P \times d_{out}}$$

输入 X 对应切分，每个 GPU 计算 $Y_i = X_iW_i$，然后通过 AllReduce 求和：

$$Y = \sum_{i=0}^{P-1} X_i W_i = XW$$

代码对应（`row_parallel.py:12-40`）：

```python
def split_weight_row(weight, rank, world_size):
    chunk_size = weight.shape[0] // world_size  # 按输入维度切
    return weight[rank * chunk_size:(rank + 1) * chunk_size, :]

def row_parallel_linear(x_local, weight_local, rank, world_size):
    # 每个 GPU 独立计算
    y_local = torch.matmul(x_local, weight_local)
    # AllReduce 求和（不需要拼接，结果已经完整）
    dist.all_reduce(y_local, op=dist.ReduceOp.SUM)
    return y_local
```

### Column + Row 组合

在 Transformer 中，Attention 和 FFN 各有一个 Column Parallel 和一个 Row Parallel：

```
Attention:
  QKV projection (Column Parallel) → Split into Q, K, V
  Output projection (Row Parallel) → AllReduce sum

FFN:
  First linear (Column Parallel) → expand
  Second linear (Row Parallel) → AllReduce sum → residual
```

### Sequence Parallel (SP)

标准 TP 中，LayerNorm 和 Dropout 的激活值在每个 GPU 上都是完整副本。SP 将这些操作的激活值沿 sequence 维度切分：

```
Input → Scatter(seq) → LayerNorm (local seq) → ColumnParallel → AllGather(full seq)
     → RowParallel → AllReduce → Scatter(seq) → LayerNorm (local seq) → ...
```

SP 的激活值内存减少 P 倍（P 为 TP world_size）。

代码对应（`sequence_parallel.py:12-60`）：

```python
def scatter_along_seq(x, rank, world_size):
    chunk_size = x.shape[1] // world_size
    return x[:, rank * chunk_size:(rank + 1) * chunk_size, :]

def gather_along_seq(x_local, rank, world_size):
    all_x = [torch.zeros_like(x_local) for _ in range(world_size)]
    dist.all_gather(all_x, x_local)
    return torch.cat(all_x, dim=1)  # 沿 seq 维度拼接
```

### Megatron 风格 TP+SP

将 Column/Row Parallel 和 SP 组合成完整的 Transformer Block：

```python
def megatron_transformer_block_fwd(x, rank, world_size):
    # Attention
    x_scatter = scatter_along_seq(x, rank, world_size)   # SP
    x_ln = layer_norm(x_scatter)
    qkv = column_parallel_linear(x_ln, w_qkv, ...)       # TP
    attn = attention(qkv)
    attn_full = gather_along_seq(attn, ...)               # SP 恢复
    attn_out = row_parallel_linear(attn_full, w_o, ...)   # TP + AllReduce
    x = x + attn_out                                      # 残差
    # FFN（同样模式）
    x_scatter = scatter_along_seq(x, rank, world_size)
    ...
```

## 架构图解

### Column Parallel vs Row Parallel

```
Column Parallel (Y = XW, 按输出维度切 W):
  W: [128 × 512] → W₀[128×256], W₁[128×256]
  GPU0: Y₀ = X·W₀ → [B,S,256]
  GPU1: Y₁ = X·W₁ → [B,S,256]
  AllGather: Y = [Y₀,Y₁] → [B,S,512] ✓

Row Parallel (Y = XW, 按输入维度切 W):
  W: [512 × 128] → W₀[256×128], W₁[256×128]
  X: [B,S,512] → X₀[B,S,256], X₁[B,S,256]
  GPU0: Y₀ = X₀·W₀ → [B,S,128]
  GPU1: Y₁ = X₁·W₁ → [B,S,128]
  AllReduce: Y = Y₀+Y₁ → [B,S,128] ✓
```

## 动手实践

→ [notebook 07: 张量并行](../../notebooks/07_tensor_parallel.ipynb)

推荐练习：
1. 实现 Column Parallel Linear 并验证输出与原始 Linear 一致
2. 对比 TP 和 DP 的通信量差异
3. 分析 SP 对激活值内存的节省效果

## 延伸阅读

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" (2023) — Sequence Parallel
```

- [ ] **Step 2: Verify and commit**

```bash
git add docs/parallel/tensor-parallel.md
git commit -m "docs: add tensor parallel detailed documentation"
```

---

## Task 4: Create `docs/parallel/pipeline-parallel.md`

**Files:**
- Create: `docs/parallel/pipeline-parallel.md`

- [ ] **Step 1: Create the file with full content**

```markdown
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
```

- [ ] **Step 2: Verify and commit**

```bash
git add docs/parallel/pipeline-parallel.md
git commit -m "docs: add pipeline parallel detailed documentation"
```

---

## Task 5: Create `docs/parallel/expert-parallel.md`

**Files:**
- Create: `docs/parallel/expert-parallel.md`

- [ ] **Step 1: Create the file with full content**

```markdown
# 专家并行详解

> 上一篇：[流水线并行](pipeline-parallel.md) ｜ 下一篇：[上下文并行](context-parallel.md)

## 概述

专家并行（Expert Parallelism, EP）将 MoE 模型中的不同 Expert 分配到不同 GPU，通过 All-to-All 通信完成 token 的分发和收集。它与数据并行互补：在 expert 维度做 EP，在非 expert 维度做 DP。

**前置知识：** [MoE 架构](../models/deepseek-v3.md)、[通信原语 — All-to-All](communication.md)
**代码位置：** [`parallel/expert_parallel/`](../../parallel/expert_parallel/)

## 核心原理

### Expert 分配

将 E 个 expert 均匀分配到 P 个 GPU：

```python
def partition_experts(rank, num_experts, world_size):
    experts_per_gpu = num_experts // world_size
    start = rank * experts_per_gpu
    return list(range(start, start + experts_per_gpu))
```

例如：8 expert, 4 GPU → GPU0: [0,1], GPU1: [2,3], GPU2: [4,5], GPU3: [6,7]

### Token Dispatch 流程

```
1. Router 计算: 每个 token 选择 Top-K expert
2. All-to-All Dispatch: 将 token 按 expert 所在 GPU 分组发送
3. Expert 计算: 各 GPU 上的 expert 处理收到的 token
4. All-to-All Gather: 将处理结果发送回原 GPU
```

代码对应（`token_dispatch.py:12-45`）：

```python
def dispatch_tokens_to_experts(tokens, expert_indices, num_experts, world_size):
    """按 expert 索引分组 token"""
    groups = [[] for _ in range(num_experts)]
    for i, idx in enumerate(expert_indices):
        groups[idx].append(tokens[i])
    return groups

def all_to_all_dispatch_example(local_tokens, rank, world_size):
    """All-to-All: 每个 GPU 发送/接收 token"""
    send_counts = [len(local_tokens[i]) for i in range(world_size)]
    recv_counts = [0] * world_size
    dist.all_to_all_single(
        torch.tensor(recv_counts), torch.tensor(send_counts))
    # 按 send_counts/recv_counts 交换实际数据
```

### 与数据并行的组合

典型配置：EP 在 expert 维度，DP 在数据维度：

```
MoE 模型 (8 expert, 4 GPU):
  EP: GPU0: expert 0,1  GPU1: expert 2,3  GPU2: expert 4,5  GPU3: expert 6,7
  DP: 每个 GPU 处理不同的数据 batch（同步非 expert 参数的梯度）
```

## 动手实践

→ [notebook 09: 专家并行](../../notebooks/09_expert_and_context_parallel.ipynb)

## 延伸阅读

- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" (2020)
- Rajbhandari et et al., "Mixture-of-Experts Meets Instruction Tuning" (2024)
```

- [ ] **Step 2: Verify and commit**

```bash
git add docs/parallel/expert-parallel.md
git commit -m "docs: add expert parallel detailed documentation"
```

---

## Task 6: Create `docs/parallel/context-parallel.md`

**Files:**
- Create: `docs/parallel/context-parallel.md`

- [ ] **Step 1: Create the file with full content**

```markdown
# 上下文并行详解

> 上一篇：[专家并行](expert-parallel.md) ｜ 下一篇：[推理并行](inference.md)

## 概述

上下文并行（Context Parallelism, CP）处理超长序列（128K+），将序列沿长度维度切分到多个 GPU，通过 Ring Attention 实现分布式 Attention 计算。

**前置知识：** [Attention 机制详解](../models/attention.md)、[通信原语](communication.md)
**代码位置：** [`parallel/context_parallel/`](../../parallel/context_parallel/)

## 核心原理

### 序列切分

将长度为 L 的序列均匀切分到 P 个 GPU：

```python
def partition_sequence(x, rank, world_size):
    chunk_len = x.shape[1] // world_size
    return x[:, rank * chunk_len:(rank + 1) * chunk_len, :]
```

**问题：** 每个 GPU 只有局部 Q，但 Attention 需要全局的 K 和 V。

### Ring Attention

解决方案：KV 沿 Ring 拓扑旋转，每步计算局部 Q 与当前 KV 的部分 Attention，累积得到完整结果。

```
Step 0: GPU0(Q0,K0,V0) → Attn(Q0,K0,V0)
Step 1: GPU0(Q0,K1,V1) → Attn(Q0,K1,V1) + 累积
Step 2: GPU0(Q0,K2,V2) → Attn(Q0,K2,V2) + 累积
Step 3: GPU0(Q0,K3,V3) → Attn(Q0,K3,V3) + 累积 = 完整 Attention
```

关键：使用 Online Softmax，在每步增量更新，不需要存储所有步骤的中间结果。

代码对应（`ring_attention.py:12-56`）：

```python
def ring_attention_step(q_local, k_current, v_current, prev_max, prev_sum, prev_out):
    """单步 Ring Attention（在线 softmax）"""
    scores = torch.matmul(q_local, k_current.transpose(-2, -1)) / scale
    # Online softmax: 增量更新
    curr_max = scores.max(dim=-1, keepdim=True).values
    new_max = torch.max(prev_max, curr_max)
    # 重新缩放之前的累积值
    corr = torch.exp(prev_max - new_max)
    curr_exp = torch.exp(scores - new_max)
    new_sum = corr * prev_sum + curr_exp.sum(dim=-1, keepdim=True)
    new_out = corr * prev_out + torch.matmul(curr_exp, v_current)
    return new_out / new_sum, new_max, new_sum
```

### 因果掩码调整

标准因果掩码假设完整序列。切分后需要考虑全局位置：

```python
def create_cp_causal_mask(local_seq_len, rank, world_size):
    """为 CP 切分后的子序列生成因果掩码"""
    global_start = rank * local_seq_len
    # 全局位置范围: [global_start, global_start + local_seq_len)
    row_pos = torch.arange(global_start, global_start + local_seq_len)
    col_pos = torch.arange(0, (rank + 1) * local_seq_len)  # 只能看到当前及之前的 KV
    mask = row_pos.unsqueeze(1) >= col_pos.unsqueeze(0)
    return mask
```

## 架构图解

### Ring Attention 数据流

```
时间 →
GPU0: Q0 ←K0,V0→  Q0 ←K1,V1→  Q0 ←K2,V2→  Q0 ←K3,V3→
         rotate KV →   rotate KV →   rotate KV →

每个箭头: 计算 Attn(Q_local, K_recv, V_recv) + Online Softmax 累积
```

## 动手实践

→ [notebook 09: 上下文并行](../../notebooks/09_expert_and_context_parallel.ipynb)

## 延伸阅读

- Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023)
- Jacobs et al., "Systematic Generalization with Edge Transformers" (2023)
```

- [ ] **Step 2: Verify and commit**

```bash
git add docs/parallel/context-parallel.md
git commit -m "docs: add context parallel detailed documentation"
```

---

## Task 7: Create `docs/parallel/inference.md`

**Files:**
- Create: `docs/parallel/inference.md`

- [ ] **Step 1: Create the file with full content**

```markdown
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
```

- [ ] **Step 2: Verify and commit**

```bash
git add docs/parallel/inference.md
git commit -m "docs: add inference parallel detailed documentation"
```

---

## Self-Review

**1. Spec coverage:** All 7 parallel docs from Phase 2 spec are covered.

**2. Placeholder scan:** No TBD/TODO found. All code references use actual function names from the codebase.

**3. Type consistency:** Function names (sync_gradients_naive, gradient_bucket_sync, column_parallel_linear, etc.) match source code. Config fields match dataclass definitions.

**4. Cross-reference chain:** overview → communication → data-parallel → tensor-parallel → pipeline-parallel → expert-parallel → context-parallel → inference → overview

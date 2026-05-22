# 数据并行详解

## 概述

数据并行（Data Parallelism）是最直观、最广泛使用的分布式训练策略。其核心思想是将同一模型复制到多个设备上，每个设备处理不同的数据子集，通过梯度同步保证模型一致性。从朴素的 Parameter Server 到 Ring AllReduce，再到 FSDP/ZeRO 的显存优化，数据并行的演进反映了分布式训练对通信效率和显存利用的持续追求。

## 直觉理解

**数据并行 = 同一模型复制多份，各看不同数据**

想象一个教室里有多个学生（GPU），每个人有相同的课本（模型），但做不同的习题（数据子集）。做完后大家核对答案（梯度同步），确保每个人的理解一致（模型参数同步）。

- **朴素 DP**：一个老师（Parameter Server）收集所有人的答案，批改后发回——老师是瓶颈
- **DDP**：同学们围成一圈（Ring），依次传递答案，每个人自己汇总——没有单点瓶颈
- **FSDP/ZeRO**：课本太厚，每个人只带一部分章节，需要时互相借阅——节省空间

## 数学原理

### 数据并行的梯度聚合

设模型参数为 $\theta$，第 $i$ 个 GPU 上的数据子集为 $\mathcal{D}_i$，局部梯度为：

$$g_i = \nabla_{\theta} \mathcal{L}(\theta; \mathcal{D}_i)$$

全局梯度为各局部梯度的平均：

$$g = \frac{1}{N} \sum_{i=0}^{N-1} g_i$$

参数更新：

$$\theta \leftarrow \theta - \eta \cdot g = \theta - \frac{\eta}{N} \sum_{i=0}^{N-1} g_i$$

**关键性质**：这等价于在完整数据集 $\mathcal{D} = \bigcup_i \mathcal{D}_i$ 上的一个完整梯度下降步。

### Parameter Server 的瓶颈分析

Parameter Server 架构中，一个节点负责收集和分发梯度：

- **通信量**：每个 worker 发送 $|\theta|$ 字节给 server，server 发送 $|\theta|$ 字节给每个 worker
- **总通信量**：$2(N-1)|\theta|$
- **瓶颈**：server 的带宽成为瓶颈，通信时间 $T = 2(N-1)\beta|\theta|$

### Ring AllReduce 梯度同步

参见 [集合通信原语详解](./01_communication_primitives.md)，Ring AllReduce 的通信量为 $\frac{2(N-1)}{N}|\theta| \approx 2|\theta|$，与 $N$ 无关。

**对比**：
| 方案 | 通信量/GPU | 带宽瓶颈 |
|------|-----------|---------|
| Parameter Server | $2|\theta|$ | server 端 $2(N-1)|\theta|$ |
| Ring AllReduce | $\approx 2|\theta|$ | 无单点瓶颈 |

### ZeRO 显存优化分析

训练时的显存消耗由三部分组成：

1. **模型状态**（Model States）：参数 $\Psi$、梯度 $G$、优化器状态 $O$
2. **激活值**（Activations）：前向传播的中间结果
3. **临时缓冲区**（Temporary Buffers）

对于 Adam 优化器，模型状态的显存占用：
- 参数：$2\Psi$ 字节（FP16）
- 梯度：$2\Psi$ 字节（FP16）
- 优化器状态：$12\Psi$ 字节（FP32 的参数副本 $4\Psi$ + momentum $4\Psi$ + variance $4\Psi$）
- **总计**：$16\Psi$ 字节

#### ZeRO-1：优化器状态分片

将优化器状态均匀分到 $N$ 个 GPU，每个 GPU 只存 $1/N$：
- 每卡显存：$2\Psi + 2\Psi + 12\Psi/N = (4 + 12/N)\Psi$

#### ZeRO-2：优化器状态 + 梯度分片

进一步将梯度也分片：
- 每卡显存：$2\Psi + 2\Psi/N + 12\Psi/N = (2 + 14/N)\Psi$

#### ZeRO-3：全部模型状态分片

参数、梯度、优化器状态全部分片：
- 每卡显存：$2\Psi/N + 2\Psi/N + 12\Psi/N = 16\Psi/N$

| 阶段 | 每卡显存 | 通信量（vs DDP） |
|------|---------|-----------------|
| DDP | $16\Psi$ | $1\times$ |
| ZeRO-1 | $(4 + 12/N)\Psi$ | $1\times$ |
| ZeRO-2 | $(2 + 14/N)\Psi$ | $1\times$ |
| ZeRO-3 | $16\Psi/N$ | $1.5\times$ |

**ZeRO-3 通信量增加的原因**：前向传播需要 AllGather 参数，反向传播需要 AllGather 参数 + ReduceScatter 梯度。

### 梯度累积的数学等价性

当 GPU 显存不足以容纳大 batch 时，使用梯度累积：

设目标 batch size 为 $B$，使用 $N$ 个 GPU，每个 GPU 的 micro-batch size 为 $b$，累积 $K$ 步：

$$K = \frac{B}{N \cdot b}$$

累积后的梯度：

$$g = \frac{1}{K} \sum_{k=0}^{K-1} g_k = \frac{1}{K} \sum_{k=0}^{K-1} \nabla_{\theta} \mathcal{L}(\theta; \mathcal{D}_k)$$

**数学等价性**：$K$ 步梯度累积 + 1 次更新 $\equiv$ 1 步使用完整 batch 的更新

**注意**：BatchNorm 等依赖 batch 统计量的层在 micro-batch 上计算的统计量与完整 batch 不同，可能需要使用 GroupNorm 或同步 BN。

## 算法流程

### DDP 训练流程

```
1. 初始化：每个 GPU 复制完整模型参数
2. 每个训练步：
   a. 前向传播：各 GPU 独立计算（不同数据）
   b. 反向传播：各 GPU 独立计算梯度
   c. 梯度同步：Ring AllReduce 求平均
   d. 参数更新：各 GPU 独立更新（结果一致）
```

### FSDP 训练流程

```
1. 初始化：将模型状态分片到各 GPU
2. 每层前向传播：
   a. AllGather：收集该层的完整参数
   b. 前向计算
   c. 丢弃非本分片的参数
3. 每层反向传播：
   a. AllGather：收集该层的完整参数
   b. 反向计算梯度
   c. ReduceScatter：梯度归约并分片
   d. 丢弃非本分片的参数和梯度
4. 优化器更新：只更新本分片的优化器状态
```

### 梯度累积流程

```
for accumulation_step in range(K):
    # 前向传播（micro-batch）
    loss = model(micro_batch)
    # 反向传播（梯度累积）
    loss = loss / K  # 缩放以保持等价性
    loss.backward()
    # 梯度在 .grad 中累积

# K 步后执行一次同步和更新
all_reduce(gradients)  # DDP 模式下
optimizer.step()
optimizer.zero_grad()
```

## 代码实现

本项目中的数据并行实现位于 `parallel/data_parallel/` 目录：

| 文件 | 内容 |
|------|------|
| `dp.py` | 朴素数据并行（Parameter Server 风格） |
| `ddp.py` | Ring AllReduce 梯度同步的 DDP |
| `gradient_accumulation.py` | 梯度累积实现 |

```python
# 示例：使用 DDP
from parallel.data_parallel.ddp import DDPModel

# model = DDPModel(base_model, process_group=dp_group)
# 梯度同步在 backward 时自动触发
```

详细代码请参考：[`parallel/data_parallel/`](../../parallel/data_parallel/)

## 实践考量

### 通信与计算重叠

DDP 的关键优化是在反向传播过程中，边计算梯度边同步：

```python
# PyTorch DDP 的梯度分桶策略
# 将参数分成桶（bucket），一个桶的梯度计算完后立即启动 AllReduce
# 同时继续计算下一个桶的梯度
```

**桶大小选择**：
- 太小：AllReduce 启动开销大
- 太大：重叠效果差
- 默认：25MB（PyTorch DDP 默认值）

### ZeRO 阶段选择指南

| 场景 | 推荐阶段 | 原因 |
|------|---------|------|
| 模型能放入单卡 | DDP | 通信量最少 |
| 优化器状态放不下 | ZeRO-1 | 无额外通信开销 |
| 梯度也放不下 | ZeRO-2 | 无额外通信开销 |
| 模型参数就放不下 | ZeRO-3 | 必须分片参数 |

### 梯度累积的实践建议

1. **累积步数不宜过大**：过大的 $K$ 会导致训练速度变慢（每 $K$ 步才更新一次）
2. **与 DDP 结合**：先在每张卡上累积 $K$ 步，再跨卡 AllReduce
3. **学习率调整**：增大 batch size 时通常需要调整学习率（线性缩放规则：$\eta' = \eta \cdot \frac{B'}{B}$）
4. **注意 BN**：小 micro-batch 上的 BN 统计量不稳定，考虑使用 SyncBatchNorm

### DDP 的梯度分桶机制

PyTorch DDP 将模型参数分成多个桶（bucket），反向传播时按桶进行 AllReduce：

```python
# 梯度分桶示意
# 参数按逆序分桶（反向传播先计算后层的梯度）
buckets = [
    [param_A, param_B],   # 桶 0：最后几层的参数
    [param_C, param_D],   # 桶 1
    [param_E, param_F],   # 桶 2：前几层的参数
]

# 反向传播时：
# 1. 桶 0 的梯度计算完成 → 启动 AllReduce
# 2. 同时继续计算桶 1 的梯度
# 3. 桶 1 计算完成 → 启动 AllReduce
# 4. 依此类推...
```

**桶大小的影响**：
- 太小（如 1MB）：AllReduce 启动次数多，延迟开销大
- 太大（如 1GB）：等待时间长，通信与计算重叠效果差
- PyTorch 默认：25MB，在大多数场景下表现良好

### DDP 与 FSDP 的选择

| 特性 | DDP | FSDP |
|------|-----|------|
| 模型复制 | 每卡完整副本 | 参数分片 |
| 显存占用 | 与卡数无关 | 随卡数减少 |
| 通信模式 | AllReduce | AllGather + ReduceScatter |
| 通信量 | $2V$ | $3V$（ZeRO-3） |
| 适用模型大小 | < 单卡显存 | 任意大小 |
| 使用复杂度 | 低 | 中等 |

**建议**：模型能放入单卡时用 DDP，否则用 FSDP。

### 常见问题

1. **梯度不同步**：确保所有 GPU 执行相同数量的迭代
2. **模型不一致**：初始化后必须广播参数，确保所有 GPU 起点一致
3. **显存碎片**：FSDP 的频繁 AllGather/释放可能导致显存碎片，可设置 `use_cpu_offload`
4. **DDP 死锁**：在 DDP 模型中使用自定义同步操作时，可能导致死锁，应使用 `no_sync` 上下文管理器
5. **FSDP 的 forward 钩子**：FSDP 通过前向钩子实现 AllGather，自定义钩子可能冲突

## 与其他技术的关系

| 技术 | 与数据并行的关系 |
|------|----------------|
| 张量并行 | 可组合为 DP+TP，DP 在 TP 组间做梯度同步 |
| 流水线并行 | 可组合为 DP+PP，DP 在 PP 组间做梯度同步 |
| 专家并行 | MoE 模型中，非专家参数用 DP，专家参数用 EP |
| 混合精度训练 | DP 同步 FP16 梯度，ZeRO 的优化器状态用 FP32 |
| 梯度检查点 | 减少激活值显存，与 ZeRO 互补 |

## 参考资料

1. **ZeRO 论文**: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", SC 2020
2. **PyTorch DDP 文档**: [Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
3. **FSDP 文档**: [Fully Sharded Data Parallel](https://pytorch.org/docs/stable/fsdp.html)
4. **Parameter Server 论文**: Li et al., "Scaling Distributed Machine Learning with the Parameter Server", OSDI 2014
5. **Ring AllReduce**: Baidu Research, "Bringing HPC Techniques to Deep Learning"

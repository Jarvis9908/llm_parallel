# 流水线并行详解

## 概述

流水线并行（Pipeline Parallelism, PP）将模型按层切分到不同设备上，每个设备负责一段连续的层。数据像流水线一样依次通过各设备，使不同设备能同时处理不同的微批次（micro-batch），从而提高硬件利用率。从 GPipe 的同步流水线到 PipeDream 的异步方案，流水线并行的核心挑战在于减少"气泡"（bubble）——设备空闲等待的时间。

## 直觉理解

**流水线并行 = 像工厂流水线一样，各层分工协作**

想象一条汽车装配线：
- 工位 A 负责安装底盘
- 工位 B 负责安装引擎
- 工位 C 负责安装外壳

朴素方案：A 装完一辆车的底盘 → B 装引擎 → C 装外壳。同一时间只有一个工位在工作，效率极低。

流水线方案：A 装完第 1 辆车的底盘后立即开始第 2 辆，同时 B 在装第 1 辆的引擎。多个工位同时工作，吞吐量大幅提升。

**气泡**：流水线启动和排空时，部分工位空闲，这就是 bubble。

## 数学原理

### 朴素流水线的 Bubble 分析

设 $N$ 为流水线阶段数（GPU 数），$T$ 为每个 micro-batch 在一个阶段的前向+反向时间。

**朴素方案**（一个 batch，无 micro-batch）：
- 前向：阶段 1 → 2 → ... → N，时间 $NT$
- 反向：阶段 N → N-1 → ... → 1，时间 $NT$
- 总时间：$2NT$
- 理想时间（无 bubble）：$2T$
- Bubble 比例：$\frac{2NT - 2T}{2NT} = \frac{N-1}{N}$

### GPipe 的 Micro-Batch 填充

GPipe 将一个 batch 分成 $M$ 个 micro-batch，依次填入流水线。

**前向阶段**：
- 时刻 $t$：阶段 $i$ 处理 micro-batch $t - i + 1$（如果 $t \geq i$ 且 $t - i + 1 \leq M$）

**反向阶段**：所有 micro-batch 前向完成后，依次反向。

**时间分析**：
- 前向总时间：$(N + M - 1) \cdot T_f$
- 反向总时间：$(N + M - 1) \cdot T_b$
- 总时间：$(N + M - 1)(T_f + T_b)$
- 理想时间（无 bubble）：$M(T_f + T_b)$
- Bubble 比例：$\frac{N-1}{N+M-1}$

当 $M \gg N$ 时，bubble 比例趋近于 0。

**GPipe 的显存问题**：所有 micro-batch 的激活值都需要保存到反向传播，显存占用为 $O(M \cdot L)$，其中 $L$ 为单 micro-batch 的激活值大小。

### 1F1B 调度的 Bubble 减少

1F1B（One Forward One Backward）策略：当一个 micro-batch 的前向传播完成后，如果该 micro-batch 的反向传播可以开始，就立即执行，而不是等所有前向完成。

**调度规则**：
1. **预热阶段**：前 $N-1$ 个 micro-batch 只做前向
2. **稳定阶段**：交替执行 1 个前向 + 1 个反向
3. **冷却阶段**：最后 $N-1$ 个 micro-batch 只做反向

**时间分析**：
- 设 $T_f = T_b = T$（简化）
- 预热：$(N-1)T$
- 稳定：$(M - N + 1) \times 2T$
- 冷却：$(N-1)T$
- 总时间：$(2M + 2N - 4)T$（近似）
- Bubble 比例：$\frac{N-1}{M}$

**显存优势**：1F1B 最多同时保存 $N$ 个 micro-batch 的激活值（vs GPipe 的 $M$ 个）。

### Bubble Time 数学分析

更精确的分析，设 $T_f$ 为前向时间，$T_b$ 为反向时间：

**GPipe**：
$$T_{\text{GPipe}} = (M + N - 1)(T_f + T_b)$$
$$\text{Bubble} = (N-1)(T_f + T_b)$$

**1F1B**：
$$T_{\text{1F1B}} = (M + N - 1)T_f + (M + N - 1)T_b$$

当 $T_f \approx T_b$ 时：
$$\text{Bubble}_{\text{1F1B}} \approx (N-1)(T_f + T_b)$$

虽然 bubble 绝对时间相似，但 1F1B 的峰值显存远低于 GPipe。

### PipeDream 的异步更新

PipeDream 允许使用过时参数（stale weights）进行前向计算，避免等待反向传播完成。

**Weight Stashing**：为每个 micro-batch 保存一份模型参数快照，反向传播时使用对应快照的梯度更新。

**问题**：
- 显存开销：需要保存多份参数
- 一致性：使用过时参数可能导致训练不稳定
- 实际中较少使用，1F1B 是更主流的方案

### 层划分策略

将 $L$ 层模型划分到 $N$ 个阶段，目标是使每个阶段的计算量相近。

**等分策略**：每个阶段 $L/N$ 层。但不同层的计算量不同（如注意力层 vs FFN 层），可能导致负载不均。

**负载均衡策略**：
1. **按计算量划分**：统计每层 FLOPs，使各阶段总 FLOPs 相近
2. **按显存划分**：考虑激活值大小，使各阶段显存占用相近
3. **按参数量划分**：使各阶段参数量相近

**实际建议**：
- 注意力层和 FFN 层应尽量均匀分配到各阶段
- 避免将所有计算密集的层放在同一阶段

## 算法流程

### GPipe 调度

```
阶段:     0        1        2        3
时刻
  1     F(m=0)     .        .        .
  2     F(m=1)   F(m=0)     .        .
  3     F(m=2)   F(m=1)   F(m=0)     .
  4     F(m=3)   F(m=2)   F(m=1)   F(m=0)
  5       .      F(m=3)   F(m=2)   F(m=1)
  6       .        .      F(m=3)   F(m=2)
  7       .        .        .      F(m=3)
  --- 所有前向完成，开始反向 ---
  8       .        .        .      B(m=3)
  9       .        .      B(m=3)   B(m=2)
 10       .      B(m=3)   B(m=2)   B(m=1)
 11     B(m=3)   B(m=2)   B(m=1)   B(m=0)
 12     B(m=2)   B(m=1)   B(m=0)     .
 13     B(m=1)   B(m=0)     .        .
 14     B(m=0)     .        .        .

F = 前向, B = 反向, m = micro-batch 编号
```

### 1F1B 调度

```
阶段:     0        1        2        3
时刻
  1     F(m=0)     .        .        .
  2     F(m=1)   F(m=0)     .        .
  3     F(m=2)   F(m=1)   F(m=0)     .
  4     F(m=3)   F(m=2)   F(m=1)   F(m=0)   ← 预热完成
  5     B(m=0)   F(m=3)   F(m=2)   F(m=1)   ← 1F1B 开始
  6     F(m=4)   B(m=0)   F(m=3)   F(m=2)
  7     B(m=1)   F(m=4)   B(m=0)   F(m=3)
  8     F(m=5)   B(m=1)   F(m=4)   B(m=0)
  9     B(m=2)   F(m=5)   B(m=1)   F(m=4)
 10     F(m=6)   B(m=2)   F(m=5)   B(m=1)
 11     B(m=3)   F(m=6)   B(m=2)   F(m=5)
  --- 冷却阶段 ---
 12       .      B(m=3)   B(m=6)   B(m=5)
 13       .        .      B(m=3)   B(m=6)
 14       .        .        .      B(m=3)

注意：阶段 0 的 B(m=0) 在时刻 5 就开始了（vs GPipe 的时刻 11）
```

## 代码实现

本项目中的流水线并行实现位于 `parallel/pipeline_parallel/` 目录：

| 文件 | 内容 |
|------|------|
| `gpiped.py` | GPipe 风格的同步流水线实现 |
| `f1b1.py` | 1F1B 调度策略实现 |
| `layer_partition.py` | 层划分和负载均衡工具 |

```python
# 示例：使用 GPipe 风格流水线
from parallel.pipeline_parallel.gpiped import GPipeModel

# model = GPipeModel(layers, num_microbatches=8)
# 自动将层分配到各阶段并执行 GPipe 调度
```

详细代码请参考：[`parallel/pipeline_parallel/`](../../parallel/pipeline_parallel/)

## 实践考量

### Micro-Batch 数量选择

$$M \geq N \times k$$

其中 $k$ 为经验系数，通常 $k \geq 2$。$M$ 越大 bubble 越小，但：
- GPipe：显存随 $M$ 线性增长
- 1F1B：显存与 $M$ 无关（最多保存 $N$ 份激活值）

### 通信开销

流水线并行的通信是点对点的（相邻阶段之间）：
- 每次传递的激活值大小：$bsh$（batch × seq × hidden）
- 每个 micro-batch 传递 2 次（前向 1 次 + 反向 1 次）
- 总通信量：$2M \times bsh$

与 TP 的 AllReduce 相比，PP 的通信量通常更小。

### 与梯度累积的关系

流水线并行天然包含梯度累积：$M$ 个 micro-batch 的梯度累积后才更新参数。

等效 batch size = $M \times b_{\text{micro}} \times N_{\text{DP}}$

### 3D 并行中的 PP 配置

在 TP+PP+DP 的 3D 并行中，PP 的配置需要综合考虑：

**典型配置**（以 512 GPU 训练 175B 模型为例）：
- TP = 8（单节点内）
- PP = 4（跨节点）
- DP = 16（剩余 GPU）

**PP 度选择原则**：
1. PP 度不宜过大：bubble 比例与 $N$ 成正比
2. 每个阶段的层数应足够多：确保计算时间远大于通信时间
3. 优先增大 DP 度：DP 不引入 bubble

### 激活值重计算与 PP

流水线并行中，激活值显存是关键瓶颈。激活值重计算（Activation Checkpointing）可以显著减少显存：

- **无重计算**：保存所有激活值，显存 $O(M \cdot L)$
- **全部重计算**：只保存输入，反向时重新前向，显存 $O(L)$，但计算量增加 33%
- **选择性重计算**：只重计算注意力（显存大户），FFN 的激活值保存，显存减少约 60%

在 1F1B 模式下，选择性重计算是最常用的方案。

### 常见问题

1. **负载不均衡**：某阶段计算时间远长于其他阶段，导致整体效率下降
2. **显存峰值**：GPipe 模式下中间阶段的显存峰值最高
3. **层划分困难**：非均匀模型（如 MoE）的层划分需要特殊处理
4. **数值精度**：不同阶段之间的激活值传递可能引入精度损失，建议使用 BF16
5. **通信与计算重叠**：PP 的点对点通信可以与相邻阶段的计算重叠

## 与其他技术的关系

| 技术 | 与流水线并行的关系 |
|------|------------------|
| 数据并行 | PP 组内做流水线，PP 组间做数据并行 |
| 张量并行 | 可组合为 3D 并行：TP 在层内，PP 在层间，DP 在外层 |
| 梯度累积 | PP 的 micro-batch 天然实现梯度累积 |
| 序列并行 | PP 的每个阶段内可以使用序列并行 |
| 虚拟流水线 | 将每个阶段的层数减半，阶段数翻倍，减少 bubble |

### 虚拟流水线并行（Virtual Pipeline Parallelism）

Megatron-LM v2 提出：将模型划分为 $2N$ 个阶段（而非 $N$ 个），每个 GPU 交替执行 2 个不相邻的阶段。

**效果**：bubble 比例从 $\frac{N-1}{M}$ 降低到约 $\frac{N-1}{2M}$。

## 参考资料

1. **GPipe 论文**: Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism", NeurIPS 2019
2. **PipeDream 论文**: Narayanan et al., "PipeDream: Fast and Efficient Pipeline Parallel DNN Training", SOSP 2019
3. **Megatron-LM v2**: Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", SC 2021
4. **1F1B 调度**: PipeDream-Flush (Megatron-LM 采用的方案)
5. **Virtual Pipeline**: Megatron-LM v2 中的 Interleaved Pipeline

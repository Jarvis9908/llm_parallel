# 并行策略选择指南

## 概述

选择正确的并行策略是高效训练大模型的关键。一个不合适的并行策略可能导致：显存不足无法训练、通信开销拖慢训练速度、或者 GPU 利用率低下浪费算力。本指南将帮助你根据模型规模、硬件条件和训练目标，选择最优的并行策略组合。

---

## 直觉理解

选择并行策略就像选择出行方式：

| 出行距离 | 出行方式 | 对应并行策略 | 原因 |
|---------|---------|------------|------|
| 短途（1km） | 走路 | 数据并行 (DP) | 简单直接，不需要额外开销 |
| 中途（10km） | 开车 | 张量并行 (TP) | 需要基础设施（高速互联），但效率高 |
| 长途（100km） | 坐飞机 | 流水线并行 (PP) + TP + DP | 必须组合多种方式，单种方式不够 |
| 超长途（跨国） | 多模式联运 | TP + PP + DP + EP/CP | 复杂场景需要组合策略 |

关键洞察：**并行策略不是越多越好**。每种并行都引入通信开销，选择的关键是在通信开销和并行收益之间找到平衡。

---

## 数学原理

### 通信开销公式

#### Ring All-Reduce 通信量

对于 $N$ 个 GPU，每个 GPU 持有大小为 $B$ 的数据：

$$\text{通信量} = \frac{2(N-1)}{N} \times B \approx 2B \quad (\text{当 } N \text{ 较大时})$$

其中系数 2 来源于 reduce-scatter 和 all-gather 两个阶段。

#### All-Gather / Reduce-Scatter 通信量

$$\text{通信量} = \frac{N-1}{N} \times B \approx B$$

### 显存占用公式

单 GPU 训练一个参数量为 $\Phi$ 的模型，显存占用约为：

| 组件 | FP32 显存 | FP16 + 混合精度 |
|------|----------|----------------|
| 模型参数 | $4\Phi$ 字节 | $2\Phi$ 字节 |
| 梯度 | $4\Phi$ 字节 | $2\Phi$ 字节 |
| 优化器状态 (Adam) | $8\Phi$ 字节 | $4\Phi$ 字节（FP32 主拷贝 + 动量 + 方差） |
| 激活值 | 与序列长度和批量成正比 | 可用梯度检查点降低 |
| **总计** | **$\approx 16\Phi$** | **$\approx 8\Phi + \text{激活值}$** |

例如：7B 模型在 FP16 混合精度下，仅参数+梯度+优化器就需要约 56 GB。

### 通信/计算重叠条件

并行策略有效的关键条件是**通信时间 ≤ 计算时间**，即通信可以被计算掩盖：

$$T_{\text{comm}} \leq T_{\text{compute}}$$

对于张量并行，这意味着：

$$\frac{2B}{BW_{\text{interconnect}}} \leq \frac{2B \cdot F}{BW_{\text{compute}}}$$

其中 $F$ 是算术强度（FLOPs/byte），$BW_{\text{interconnect}}$ 是互联带宽，$BW_{\text{compute}}$ 是计算带宽。**NVLink 的高带宽使得 TP 在节点内可以高效重叠，而跨节点则难以实现。**

---

## 六大并行策略适用场景对比

| 策略 | 适用场景 | 通信量 | 显存节省 | 互联要求 | 实现复杂度 |
|------|---------|-------|---------|---------|-----------|
| **数据并行 (DP)** | 模型可放入单卡 | $2B$ (AllReduce) | 无（每卡完整模型） | 低（可跨节点） | ★☆☆☆☆ |
| **张量并行 (TP)** | 单层无法放入单卡 | $2B/t$ (每层 AllReduce) | 参数按 $1/t$ 缩减 | **极高**（需 NVLink） | ★★★☆☆ |
| **流水线并行 (PP)** | 模型层数多 | $B/p$ (点对点) | 参数按 $1/p$ 缩减 | 中（可跨节点） | ★★★★☆ |
| **专家并行 (EP)** | MoE 模型 | $O(B \times E)$ (All-to-All) | 专家参数按 $1/e$ 缩减 | 中 | ★★★★☆ |
| **上下文并行 (CP)** | 超长序列 | $2B/c$ (Ring Attention) | 激活值按 $1/c$ 缩减 | 中 | ★★★☆☆ |
| **推理优化** | 推理阶段 | 视策略而定 | KV-Cache 分片等 | 低-中 | ★★☆☆☆ |

> $t$ = TP 度，$p$ = PP 度，$e$ = EP 度，$c$ = CP 度，$B$ = 通信数据量

---

## 并行策略选择决策树

```
开始
 │
 ├─ 模型能否放入单卡显存？
 │   ├─ 是 → 数据并行 (DP)
 │   │       │
 │   │       └─ GPU 数量 > 1？
 │   │           ├─ 是 → DDP 或 FSDP
 │   │           └─ 否 → 单卡训练
 │   │
 │   └─ 否 → 单层能否放入单卡？
 │       ├─ 是 → 流水线并行 (PP)
 │       │       │
 │       │       └─ GPU 数量 > PP 度？
 │           │       ├─ 是 → PP + DP
 │           │       └─ 否 → 纯 PP
 │           │
 │       └─ 否 → 张量并行 (TP) + 流水线并行 (PP)
 │               │
 │               ├─ 是否为 MoE 模型？
 │               │   ├─ 是 → 加入专家并行 (EP)
 │               │   └─ 否 → 继续
 │               │
 │               ├─ 序列长度 > 32K？
 │               │   ├─ 是 → 加入上下文并行 (CP)
 │               │   └─ 否 → 继续
 │               │
 │               └─ GPU 数量 > TP × PP？
 │                   ├─ 是 → 加入数据并行 (DP)
 │                   └─ 否 → TP + PP
```

### 关键决策因素

1. **模型大小**：决定是否需要模型并行（TP/PP）
2. **GPU 数量与拓扑**：决定 TP 的可行性和 DP 的规模
3. **显存限制**：决定并行度和是否需要 ZeRO/FSDP
4. **序列长度**：决定是否需要上下文并行
5. **模型类型**：MoE 模型需要专家并行

---

## 组合策略推荐

### DP + TP：单机多卡

```
┌─────────────────────────────────┐
│           节点 (8×A100)          │
│  ┌───────┐ ┌───────┐ ┌───────┐ │
│  │GPU 0  │ │GPU 1  │ │GPU 2  │ │  ← TP 组 1 (TP=2)
│  │TP=0   │ │TP=1   │ │TP=0   │ │
│  ├───────┤ ├───────┤ ├───────┤ │
│  │GPU 3  │ │GPU 4  │ │GPU 5  │ │  ← TP 组 2 (TP=2)
│  │TP=1   │ │TP=0   │ │TP=1   │ │
│  ├───────┤ ├───────┤         │ │
│  │GPU 6  │ │GPU 7  │         │ │  ← TP 组 3/4
│  └───────┘ └───────┘         │ │
└─────────────────────────────────┘
  DP=4, TP=2
```

- **适用**：单节点内，模型单层较大
- **TP 度选择**：通常 2-8，受限于节点内 GPU 数量和 NVLink 带宽
- **通信**：TP 组内 NVLink 高速通信，DP 组间 AllReduce

### DP + PP：多机少卡

```
┌──────────┐     ┌──────────┐
│  节点 1   │     │  节点 2   │
│ GPU0: S0 │────→│ GPU0: S2 │    ← PP 流水线 1
│ GPU1: S1 │────→│ GPU1: S3 │    ← PP 流水线 2
└──────────┘     └──────────┘
  DP=2, PP=2
```

- **适用**：跨节点训练，节点间带宽有限
- **优势**：PP 的点对点通信量小，适合低带宽互联
- **注意**：需要平衡各阶段的计算量（气泡率）

### TP + PP + DP：3D 并行

```
         PP 阶段 0    PP 阶段 1    PP 阶段 2    PP 阶段 3
        ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
DP=0    │ TP组(0) │→│ TP组(1) │→│ TP组(2) │→│ TP组(3) │
        └─────────┘ └─────────┘ └─────────┘ └─────────┘
        ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
DP=1    │ TP组(4) │→│ TP组(5) │→│ TP组(6) │→│ TP组(7) │
        └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

- **适用**：大规模训练（70B+ 模型，多节点）
- **原则**：TP 限制在节点内，PP 跨节点，DP 跨流水线副本
- **经典配置**：TP=4/8, PP=2-8, DP=剩余

### TP + PP + DP + EP：MoE 模型

- **适用**：MoE 模型（如 DeepSeek-V3、Mixtral）
- **EP 作用**：将不同专家分布到不同 GPU，减少每卡显存
- **通信**：EP 引入 All-to-All 通信，需要较高互联带宽
- **配置建议**：EP 度 = 专家数 / 每卡专家数

### TP + PP + DP + CP：长序列

- **适用**：训练超长上下文模型（128K+ 序列）
- **CP 作用**：沿序列维度切分，降低每卡激活值显存
- **通信**：Ring Attention 的环形通信，带宽需求适中
- **配置建议**：CP 度 = 序列长度 / 每卡序列长度

---

## 实际案例分析

### 案例 1：7B 模型

| 项目 | 配置 |
|------|------|
| 模型 | 7B 参数 |
| 显存需求 | ~56 GB（参数+梯度+优化器） |
| 硬件 | 1× A100 80GB |
| 并行策略 | **DP (DDP 或 FSDP)** |
| 说明 | 模型可放入单卡，使用 FSDP 可进一步降低显存峰值 |

```bash
# 单卡 FSDP 训练
torchrun --nproc_per_node=1 train.py --strategy fsdp
```

### 案例 2：13B 模型

| 项目 | 配置 |
|------|------|
| 模型 | 13B 参数 |
| 显存需求 | ~104 GB |
| 硬件 | 2× A100 80GB |
| 并行策略 | **TP=2 + DP** |
| 说明 | 单层可放入单卡，TP=2 切分模型，NVLink 保证通信效率 |

```bash
# 2卡 TP 训练
torchrun --nproc_per_node=2 train.py --strategy tp --tp_degree=2
```

### 案例 3：70B 模型

| 项目 | 配置 |
|------|------|
| 模型 | 70B 参数 |
| 显存需求 | ~560 GB |
| 硬件 | 8× A100 80GB (单节点) |
| 并行策略 | **TP=4 + PP=2 + DP=1** 或 **TP=8 + DP=1** |
| 说明 | 需要 3D 并行或大 TP 度。TP=8 需要全节点 NVLink 互联 |

```bash
# 8卡 3D 并行训练
torchrun --nproc_per_node=8 train.py \
    --strategy 3d --tp_degree=4 --pp_degree=2
```

### 案例 4：405B 模型

| 项目 | 配置 |
|------|------|
| 模型 | 405B 参数 |
| 显存需求 | ~3.2 TB |
| 硬件 | 64× A100 80GB (8 节点) |
| 并行策略 | **TP=8 + PP=4 + DP=2** (+ EP 如果是 MoE) |
| 说明 | 典型的大规模训练配置。TP 限制在节点内，PP 跨节点 |

```bash
# 64卡 多节点训练
torchrun --nnodes=8 --nproc_per_node=8 train.py \
    --strategy 3d --tp_degree=8 --pp_degree=4 --dp_degree=2
```

### 配置速查表

| 模型大小 | 最少 GPU (A100 80G) | 推荐策略 | TP 度 | PP 度 |
|---------|--------------------|---------|-------|-------|
| 7B | 1 | DP / FSDP | 1 | 1 |
| 13B | 2 | TP + DP | 2 | 1 |
| 30B | 4 | TP + DP | 4 | 1 |
| 70B | 8 | TP + PP + DP | 4-8 | 1-2 |
| 175B | 16-32 | TP + PP + DP | 8 | 2-4 |
| 405B+ | 64+ | TP + PP + DP + EP | 8 | 4-8 |

---

## 与其他技术的关系

| 并行策略 | 相关优化技术 | 关系说明 |
|---------|------------|---------|
| DP | ZeRO/FSDP、梯度累积 | ZeRO 将 DP 与显存优化结合 |
| TP | 序列并行 (SP) | SP 是 TP 在序列维度上的扩展 |
| PP | 激活重计算、微批次 | 微批次减少流水线气泡 |
| EP | 负载均衡、辅助损失 | 负载均衡确保 EP 效率 |
| CP | Flash Attention | Flash Attention 降低单卡注意力显存 |
| 所有 | 混合精度 (FP16/BF16) | 混合精度减半显存，所有策略都应启用 |

---

## 参考资料

### 核心论文

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — 3D 并行（TP+PP+DP）的经典论文
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — FSDP 的理论基础
- [DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters](https://arxiv.org/abs/2004.13366) — DeepSpeed 框架

### 并行策略专项

- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) — 大规模 3D 并行实践
- [PipeDream: Efficient Pipeline Parallel DNN Training](https://arxiv.org/abs/1806.03377) — 1F1B 流水线调度
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841) — MoE 并行训练

### 工业实践

- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) — 计算最优训练配置
- [LLaMA 3 技术报告](https://ai.meta.com/blog/meta-llama-3/) — 大规模分布式训练实践
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) — MoE + FP8 + 多种并行组合

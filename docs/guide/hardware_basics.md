# 硬件基础知识

## 概述

理解硬件是优化分布式训练的基础。并行策略的选择、通信开销的评估、显存瓶颈的分析——这些都离不开对 GPU 架构、互联带宽和存储层次的深入理解。本指南将帮助你建立硬件层面的直觉，让你在面对性能问题时能快速定位瓶颈所在。

---

## 直觉理解

GPU 像一座工厂：

| 工厂组件 | GPU 对应 | 说明 |
|---------|---------|------|
| 车间 (SM) | 流多处理器 | 执行计算的核心单元 |
| 仓库 (显存) | HBM | 存储数据和模型参数 |
| 高速公路 (NVLink) | NVLink 互联 | 车间之间快速运输 |
| 普通道路 (PCIe) | PCIe 总线 | 连接仓库和外部 |
| 调度中心 | Warp 调度器 | 分配任务给车间 |

关键洞察：**工厂的产能不只取决于车间数量，还取决于仓库能否及时供料、道路是否通畅。** 同理，GPU 的训练效率不只取决于算力，还取决于显存带宽和互联带宽。

---

## GPU 架构基础

### SM（流多处理器）

SM (Streaming Multiprocessor) 是 GPU 的计算核心单元。每个 SM 包含：

```
┌─────────────────────────────────────────┐
│              一个 SM                      │
│  ┌──────────┐ ┌──────────┐              │
│  │ CUDA 核心 │ │ CUDA 核心 │  ... × 128  │  ← 整数和浮点运算
│  │  (INT/FP) │ │  (INT/FP) │             │
│  └──────────┘ └──────────┘              │
│  ┌──────────────────────────┐           │
│  │     Tensor Core          │  ... × 4  │  ← 矩阵乘法加速
│  │  (FP16/BF16/FP8/INT8)    │           │
│  └──────────────────────────┘           │
│  ┌──────────────────────────┐           │
│  │  L1 Cache / Shared Mem   │  192 KB   │  ← 低延迟存储
│  └──────────────────────────┘           │
│  ┌──────────────────────────┐           │
│  │   Warp 调度器             │  4 个     │  ← 32 线程为一组调度
│  └──────────────────────────┘           │
│  ┌──────────────────────────┐           │
│  │   Register File          │  256 KB   │  ← 最快存储
│  └──────────────────────────┘           │
└─────────────────────────────────────────┘
```

| 组件 | 数量 (A100) | 功能 |
|------|------------|------|
| CUDA 核心 | 64/SM × 108 SM = 6912 | 通用计算 |
| Tensor Core | 4/SM × 108 SM = 432 | 矩阵乘法加速 |
| L1 Cache/Shared Memory | 192 KB/SM | 低延迟存储 |
| Register File | 256 KB/SM | 最快存储 |

### CUDA 核心与 Tensor Core

**CUDA 核心**执行标量运算（逐元素操作），**Tensor Core** 执行矩阵乘法加速：

```python
# CUDA 核心执行的操作（逐元素）
c[i] = a[i] + b[i]  # 逐元素加法
c[i] = a[i] * b[i]  # 逐元素乘法

# Tensor Core 执行的操作（矩阵乘法）
C = A @ B  # 矩阵乘法，一次计算 16×8×8 (FP16) 或 16×16×16 (INT8)
```

| 操作类型 | 执行单元 | 吞吐量 (A100) |
|---------|---------|--------------|
| 逐元素 FP16 加法 | CUDA 核心 | ~312 TFLOPS |
| 矩阵乘法 FP16 | Tensor Core | ~312 TFLOPS |
| 矩阵乘法 BF16 | Tensor Core | ~312 TFLOPS |
| 矩阵乘法 FP8 | Tensor Core | ~624 TFLOPS |
| 矩阵乘法 INT8 | Tensor Core | ~624 TOPS |

> **关键洞察**：Transformer 训练中 90%+ 的计算是矩阵乘法（GEMM），因此 Tensor Core 的利用率决定了训练速度。

### 显存带宽

显存带宽决定了数据从显存传输到计算单元的速度：

$$\text{带宽} = \text{显存频率} \times \text{总线宽度}$$

| GPU | 显存带宽 | 对训练的影响 |
|-----|---------|------------|
| A100 80GB | 2,039 GB/s | 大模型推理可能带宽瓶颈 |
| H100 80GB | 3,352 GB/s | 显著改善推理吞吐 |
| H200 141GB | 4,800 GB/s | 超大模型推理友好 |

> **算术强度**：计算量 / 通信量。矩阵乘法的算术强度高（计算密集），逐元素操作算术强度低（带宽密集）。当算术强度低于 GPU 的"平衡点"时，性能受限于带宽而非算力。

---

## GPU 显存层次结构

```
┌─────────────────────────────────────────────────────────┐
│                    显存层次结构                           │
│                                                          │
│  容量:   80 GB        40 MB       192 KB      256 KB    │
│          ↑              ↑           ↑           ↑       │
│  带宽: 2,039 GB/s   ~6 TB/s    ~19 TB/s    ~38 TB/s    │
│          ↑              ↑           ↑           ↑       │
│  延迟:  ~200 ns      ~50 ns     ~20 ns      ~1 ns      │
│                                                          │
│  ┌──────┐   ┌───────┐   ┌──────┐   ┌──────┐           │
│  │ HBM  │ → │L2 Cache│ → │L1/SM │ → │ Reg  │           │
│  │(全局) │   │(全局)  │   │(每SM) │   │(每SM)│           │
│  └──────┘   └───────┘   └──────┘   └──────┘           │
│                                                          │
│  容量递减 ←──────────────────────────────→ 容量递增      │
│  带宽递减 ←──────────────────────────────→ 带宽递增      │
│  延迟递减 ←──────────────────────────────→ 延迟递增      │
└─────────────────────────────────────────────────────────┘
```

### 各层次详解

| 层次 | 容量 (A100) | 带宽 | 延迟 | 用途 |
|------|------------|------|------|------|
| **HBM** (高带宽显存) | 80 GB | 2,039 GB/s | ~200 ns | 模型参数、梯度、优化器状态 |
| **L2 Cache** | 40 MB | ~6 TB/s | ~50 ns | 热点数据缓存（如当前层的权重） |
| **L1 Cache / Shared Memory** | 192 KB/SM | ~19 TB/s | ~20 ns | 块内数据复用（如 Tile 计算） |
| **Register** | 256 KB/SM | ~38 TB/s | ~1 ns | 临时变量、循环索引 |

### 对训练的启示

1. **模型参数存储在 HBM**：模型越大，HBM 需求越大，这是 OOM 的直接原因
2. **计算时参数从 HBM 加载到 L2**：如果参数能留在 L2，计算效率更高
3. **Flash Attention 的原理**：将注意力计算分块 (Tiling)，使 QK^T 的中间结果留在 L1/Shared Memory，避免写回 HBM
4. **混合精度的原理**：FP16 参数占 HBM 减半，且 Tensor Core 对 FP16 吞吐更高

---

## NVLink vs PCIe 带宽对比

GPU 之间的通信带宽是分布式训练性能的关键因素。

### 带宽对比

| 互联方式 | 单向带宽 | 双向带宽 | 延迟 | 适用场景 |
|---------|---------|---------|------|---------|
| PCIe 4.0 x16 | 32 GB/s | 64 GB/s | ~500 ns | 跨节点、低成本 |
| PCIe 5.0 x16 | 64 GB/s | 128 GB/s | ~400 ns | 新一代服务器 |
| NVLink 3.0 (4 links) | 100 GB/s | 200 GB/s | ~100 ns | 节点内 A100 |
| NVLink 4.0 (8 links) | 200 GB/s | 400 GB/s | ~80 ns | 节点内 H100 |
| NVLink 5.0 (8 links) | 400 GB/s | 800 GB/s | ~50 ns | 节点内 B200 |

### 对训练策略的影响

```
┌─────────────────────────────────────────────────────┐
│                  互联带宽 vs 并行策略                  │
│                                                      │
│  带宽需求:  低 ←────────────────────────→ 高         │
│            DP    PP    CP    EP    TP                │
│                                                      │
│  NVLink:   不需要   可选   推荐   推荐   必须        │
│  PCIe:     足够     足够   勉强   勉强   不够        │
│                                                      │
│  关键结论:                                           │
│  - TP 必须使用 NVLink（每层都需要 AllReduce）         │
│  - PP 可以使用 PCIe（点对点通信量小）                 │
│  - DP 对带宽要求最低（仅梯度同步）                    │
└─────────────────────────────────────────────────────┘
```

### 为什么 TP 需要 NVLink？

张量并行中，每个 Transformer 层都需要两次 AllReduce（一次在前向，一次在反向）：

```
前向传播:
  输入 → 列并行线性层 → AllReduce → 行并行线性层 → AllReduce → 输出
         (TP 通信 1)                    (TP 通信 2)

反向传播:
  梯度 → ReduceScatter → 行并行反向 → AllReduce → 列并行反向 → 梯度
         (TP 通信 3)                    (TP 通信 4)
```

每个 Transformer 层需要 **4 次集合通信**。如果使用 PCIe (64 GB/s)，通信开销将远超计算时间，GPU 利用率极低。

### 为什么 PP 可以用 PCIe？

流水线并行中，相邻阶段之间只传递激活值和梯度（点对点通信）：

```
Stage 0 → Stage 1 → Stage 2 → Stage 3
  激活值 →    激活值 →    激活值 →
  ← 梯度     ← 梯度     ← 梯度
```

通信量 = batch_size × seq_len × hidden_dim，远小于 TP 的 AllReduce 通信量。

---

## 多卡拓扑对训练的影响

### NVLink 拓扑

同节点内的 GPU 通过 NVLink 形成全互联或部分互联拓扑：

```
4-GPU NVLink 拓扑 (A100 典型配置):

    GPU 0 ──────── GPU 1
     │  ╲            ╱  │
     │    ╲        ╱    │
     │      ╲    ╱      │
     │        ╳        │
     │      ╱    ╲      │
     │    ╱        ╲    │
     │  ╱            ╲  │
    GPU 2 ──────── GPU 3

每条线 = NVLink (50 GB/s 双向)
GPU 0 ↔ GPU 1: 4 NVLinks = 200 GB/s 双向
GPU 0 ↔ GPU 2: 4 NVLinks = 200 GB/s 双向
GPU 0 ↔ GPU 3: 2 NVLinks = 100 GB/s 双向 (可能不是全互联)
```

### NUMA 效应

在多 CPU 插槽的服务器中，GPU 与 CPU 的物理距离影响数据传输效率：

```
┌─────────────────────────────────────────────┐
│                  服务器                       │
│                                              │
│  ┌─────────────┐     ┌─────────────┐        │
│  │   CPU 0      │     │   CPU 1      │       │
│  │  (NUMA 0)    │     │  (NUMA 1)    │       │
│  │  GPU0  GPU1  │     │  GPU2  GPU3  │       │
│  └──────┬───────┘     └──────┬───────┘       │
│         │    QPI/UPI 互联    │               │
│         └────────────────────┘               │
│                                              │
│  CPU 0 → GPU 0/1: 快速 (本地 NUMA)           │
│  CPU 0 → GPU 2/3: 慢速 (跨 NUMA)             │
└─────────────────────────────────────────────┘
```

**最佳实践**：数据加载进程应绑定到与 GPU 同一 NUMA 节点的 CPU 上。

```bash
# 查看 NUMA 拓扑
numactl --hardware

# 绑定进程到 NUMA 节点
numactl --cpunodebind=0 --membind=0 python train.py
```

### 拓扑感知

```bash
# 查看 GPU 拓扑
nvidia-smi topo -m

# 输出示例：
#         GPU0  GPU1  GPU2  GPU3  CPU Affinity
# GPU0     X    NV2   NV2   SYS   0-23
# GPU1    NV2    X    NV2   SYS   0-23
# GPU2    NV2   NV2    X    SYS   24-47
# GPU3    SYS   SYS   SYS    X    24-47
#
# NV1/NV2 = NVLink 连接 (数字表示 NVLink 数量)
# SYS = 跨 NUMA 节点 (通过 PCIe + QPI)
# NODE = 同 NUMA 节点但无 NVLink
# PHB = 同 PCIe 主桥
```

**拓扑感知的并行配置**：

- TP 组应放在 NVLink 直连的 GPU 上（如 GPU0+GPU1）
- PP 阶段可以跨 NUMA 节点（如 Stage0 在 GPU0-1，Stage1 在 GPU2-3）
- DP 组可以跨节点（通信量最小）

---

## 常见 GPU 型号参数对比

| 参数 | A100 40GB | A100 80GB | H100 80GB | H200 141GB | B200 192GB |
|------|-----------|-----------|-----------|------------|------------|
| **显存** | 40 GB HBM2 | 80 GB HBM2e | 80 GB HBM3 | 141 GB HBM3e | 192 GB HBM3e |
| **显存带宽** | 1,555 GB/s | 2,039 GB/s | 3,352 GB/s | 4,800 GB/s | 8,000 GB/s |
| **FP16 算力** | 312 TFLOPS | 312 TFLOPS | 990 TFLOPS | 990 TFLOPS | 2,250 TFLOPS |
| **FP8 算力** | — | — | 1,979 TFLOPS | 1,979 TFLOPS | 4,500 TFLOPS |
| **INT8 算力** | 624 TOPS | 624 TOPS | 1,979 TOPS | 1,979 TOPS | 4,500 TOPS |
| **NVLink 版本** | 3.0 | 3.0 | 4.0 | 4.0 | 5.0 |
| **NVLink 带宽** | 200 GB/s | 200 GB/s | 400 GB/s | 400 GB/s | 800 GB/s |
| **PCIe 版本** | 4.0 | 4.0 | 5.0 | 5.0 | 5.0 |
| **SM 数量** | 108 | 108 | 132 | 132 | 160 |
| **TDP** | 250W | 300W | 700W | 700W | 1000W |
| **架构** | Ampere | Ampere | Hopper | Hopper | Blackwell |

### 选择建议

| 场景 | 推荐 GPU | 原因 |
|------|---------|------|
| 学习/实验 | A100 40GB | 性价比高，足够跑 7B 模型 |
| 中等规模训练 | A100 80GB | 80GB 显存可容纳 13B 模型 |
| 大规模训练 | H100 80GB | 3× 算力 + 2× 带宽 + FP8 |
| 超大模型推理 | H200 141GB | 大显存 + 高带宽，推理友好 |
| 前沿训练 | B200 192GB | 最高算力 + 最大显存 |

### 显存与模型大小的关系

| 模型参数量 | FP16 参数显存 | 训练总显存 (≈8×) | 推荐 GPU |
|-----------|-------------|-----------------|---------|
| 7B | 14 GB | ~56 GB | A100 80GB × 1 |
| 13B | 26 GB | ~104 GB | A100 80GB × 2 |
| 30B | 60 GB | ~240 GB | A100 80GB × 4 |
| 70B | 140 GB | ~560 GB | H100 80GB × 8 |
| 175B | 350 GB | ~1.4 TB | H100 80GB × 16-32 |
| 405B | 810 GB | ~3.2 TB | H100 80GB × 64 |

> **注意**：训练显存 ≈ 8× 参数量（FP16 参数 + 梯度 + Adam 优化器状态），实际还取决于 batch size 和序列长度。使用 ZeRO/FSDP 可以分摊显存到多卡。

---

## 与其他技术的关系

| 硬件知识 | 相关并行策略 | 关系说明 |
|---------|------------|---------|
| NVLink 带宽 | 张量并行 | TP 需要 NVLink 保证通信效率 |
| HBM 容量 | 所有策略 | 显存容量决定模型并行需求 |
| Tensor Core | 混合精度 | FP8/BF16 利用 Tensor Core 加速 |
| NUMA 拓扑 | 数据并行 | 数据加载应感知 NUMA 亲和性 |
| PCIe 带宽 | 流水线并行 | PP 通信量小，PCIe 即可满足 |
| 显存带宽 | 推理优化 | 推理通常是带宽瓶颈而非算力瓶颈 |

---

## 参考资料

### 官方文档

- [NVIDIA A100 架构白皮书](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf)
- [NVIDIA H100 架构白皮书](https://resources.nvidia.com/en-us-tensor-core)
- [NVIDIA B200 架构白皮书](https://www.nvidia.com/en-us/data-center/b200/)
- [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### 技术文章

- [GPU 性能分析入门](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda/)
- [NVLink 技术概述](https://www.nvidia.com/en-us/data-center/nvlink/)
- [Understanding GPU Architecture for Deep Learning](https://lilianweng.github.io/posts/2024-07-07-hardware/)

### 工具

- [nvidia-smi 文档](https://nvidia.custhelp.com/app/answers/detail/a_id/3751)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [CUDA Toolkit 文档](https://docs.nvidia.com/cuda/)

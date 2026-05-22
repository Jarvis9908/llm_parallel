# 张量并行详解

## 概述

张量并行（Tensor Parallelism, TP）是一种将单个算子（如矩阵乘法）切分到多个设备上并行计算的技术。与数据并行复制整个模型不同，张量并行让每个设备只持有模型参数的一部分，通过设备间通信协作完成计算。Megatron-LM 提出的列切分与行切分组合策略是张量并行的经典方案，后续的序列并行进一步优化了显存和通信。

## 直觉理解

**张量并行 = 把一个矩阵切成多块，各算各的**

想象一个大型矩阵乘法 $Y = XA$，其中 $A$ 太大一张卡放不下。解决方案：
- **列切分**：把 $A$ 按列切成 $[A_1, A_2]$，两张卡分别算 $XA_1$ 和 $XA_2$，结果拼接
- **行切分**：把 $A$ 按行切成 $\begin{bmatrix}A_1 \\ A_2\end{bmatrix}$，把 $X$ 按列切成 $[X_1, X_2]$，分别算 $X_1A_1$ 和 $X_2A_2$，结果相加

Megatron-LM 的巧妙之处在于：列切分和行切分交替使用，中间不需要通信！

## 数学原理

### 1D 列切分（Column Parallel）

设权重矩阵 $A \in \mathbb{R}^{d \times k}$，按列切分为 $A = [A_1, A_2, \ldots, A_N]$，其中 $A_i \in \mathbb{R}^{d \times k/N}$。

$$Y = XA = X[A_1, A_2, \ldots, A_N] = [XA_1, XA_2, \ldots, XA_N]$$

- 每个 GPU $i$ 计算 $Y_i = XA_i$，得到 $Y_i \in \mathbb{R}^{b \times k/N}$
- **输入**：每个 GPU 需要完整的 $X$（需要广播或已持有）
- **输出**：每个 GPU 持有 $Y$ 的一部分列
- **通信**：前向无通信（如果 $X$ 已持有），反向需要 AllReduce 同步 $X$ 的梯度

### 1D 行切分（Row Parallel）

设权重矩阵 $B \in \mathbb{R}^{k \times m}$，按行切分为 $B = \begin{bmatrix} B_1 \\ B_2 \\ \vdots \\ B_N \end{bmatrix}$，其中 $B_i \in \mathbb{R}^{k/N \times m}$。

$$Z = YB = [Y_1, Y_2, \ldots, Y_N] \begin{bmatrix} B_1 \\ B_2 \\ \vdots \\ B_N \end{bmatrix} = \sum_{i=1}^{N} Y_i B_i$$

- 每个 GPU $i$ 计算 $Z_i = Y_i B_i$，得到 $Z_i \in \mathbb{R}^{b \times m}$
- **输入**：每个 GPU 需要对应的 $Y_i$（恰好是列切分的输出！）
- **输出**：需要 AllReduce 求和得到完整 $Z$
- **通信**：前向需要 AllReduce，反向无通信

### Megatron-LM 的 TP 组合策略

Transformer 的 MLP 层由两个线性层组成：

$$\text{MLP}(X) = \text{GeLU}(XW_1)W_2$$

**关键洞察**：$W_1$ 用列切分，$W_2$ 用行切分，中间无需通信！

```
X ──→ [列切分 W_1] ──→ GeLU ──→ [行切分 W_2] ──→ AllReduce ──→ 输出
         GPU 0: XW_1[0]     GPU 0: GeLU(XW_1[0])W_2[0]
         GPU 1: XW_1[1]     GPU 1: GeLU(XW_1[1])W_2[1]
```

详细推导：
1. $Y_i = \text{GeLU}(XW_{1,i})$，GPU $i$ 独立计算
2. $Z_i = Y_i W_{2,i}$，GPU $i$ 独立计算
3. $Z = \sum_i Z_i$，AllReduce 求和

**通信量**：每个 Transformer 层只需 2 次 AllReduce（前向 1 次 + 反向 2 次 = 3 次，但反向的梯度同步可合并）。

### 自注意力的张量并行

自注意力层也可以类似切分：

$$\text{Attention}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \cdot W_o$$

- $Q = XW_Q$，$K = XW_K$，$V = XW_V$：对 $W_Q, W_K, W_V$ 做列切分
- 注意力计算：每个 GPU 独立计算 $1/N$ 的注意力头
- $W_o$：做行切分
- 输出 AllReduce

**注意**：列切分 $W_Q, W_K, W_V$ 等价于将注意力头分配到不同 GPU。

### 2D/2.5D/3D 切分策略

1D 切分只沿一个维度切分矩阵，更高维切分可进一步减少显存：

| 策略 | 切分方式 | 显存/GPU | 通信量/GPU |
|------|---------|---------|-----------|
| 1D | 按列或按行 | $O(d^2/N)$ | $O(d^2/N)$ |
| 2D | 行列同时切 | $O(d^2/N^2)$ | $O(d^2/N)$ |
| 2.5D | 2D + 冗余 | $O(d^2/(N/q))$ | $O(d^2/N)$ |
| 3D | 三个维度切 | $O(d^2/N^{4/3})$ | $O(d^2/N^{2/3})$ |

2D 切分将矩阵 $A \in \mathbb{R}^{d \times d}$ 切成 $N = P \times Q$ 块 $A_{ij}$，GPU $(i,j)$ 持有 $A_{ij}$。计算时需要沿行和列两个方向通信。

### 序列并行（Sequence Parallelism）

**问题**：Megatron-LM 的 TP 中，LayerNorm 和 Dropout 等操作在每个 GPU 上重复计算（因为输入 $X$ 在每个 GPU 上都有副本），浪费显存。

**解决方案**：将 $X$ 沿序列维度也切分，每个 GPU 只持有 $1/N$ 的序列：

- LayerNorm/Dropout：各 GPU 独立计算自己的序列段
- 列切分线性层前：AllGather 收集完整序列
- 行切分线性层后：ReduceScatter 分发回序列段

**通信量对比**：
- 原始 TP：2 次 AllReduce（每层），通信量 $2 \times 2V$
- 序列并行 TP：1 次 AllGather + 1 次 ReduceScatter（每层），通信量 $2 \times V$

通信量相同，但 LayerNorm/Dropout 的激活值显存减少为 $1/N$。

## 算法流程

### Megatron-LM MLP 层前向传播

```
输入: X (每个 GPU 持有完整副本)

1. 列切分线性层:
   GPU_i: Y_i = GeLU(X @ W1_i)    # W1_i 是 W1 的第 i 列块

2. 行切分线性层:
   GPU_i: Z_i = Y_i @ W2_i        # W2_i 是 W2 的第 i 行块

3. AllReduce:
   Z = AllReduce(sum, Z_0, Z_1, ..., Z_{N-1})

输出: Z (每个 GPU 持有完整结果)
```

### 序列并行前向传播

```
输入: X_i (每个 GPU 持有序列的第 i 段)

1. LayerNorm:
   GPU_i: X_i = LayerNorm(X_i)    # 独立计算

2. AllGather:
   X = AllGather(X_0, X_1, ..., X_{N-1})  # 收集完整序列

3. 列切分线性层 + GeLU:
   GPU_i: Y_i = GeLU(X @ W1_i)

4. 行切分线性层:
   GPU_i: Z_i = Y_i @ W2_i

5. ReduceScatter:
   Z_i = ReduceScatter(sum, Z_0, ..., Z_{N-1})  # 归约并分发

输出: Z_i (每个 GPU 持有结果的第 i 段)
```

## 代码实现

本项目中的张量并行实现位于 `parallel/tensor_parallel/` 目录：

| 文件 | 内容 |
|------|------|
| `column_parallel.py` | 列切分线性层的实现 |
| `row_parallel.py` | 行切分线性层的实现 |
| `megatron_style.py` | Megatron-LM 风格的 TP 组合 |
| `embedding_parallel.py` | 嵌入层的并行切分 |
| `sequence_parallel.py` | 序列并行实现 |

```python
# 示例：使用 Megatron-LM 风格的 TP
from parallel.tensor_parallel.megatron_style import TensorParallelLinear

# 列切分 + 行切分组合
# col_linear = TensorParallelLinear(dim, hidden_dim, split_dim='column')
# row_linear = TensorParallelLinear(hidden_dim, dim, split_dim='row')
```

详细代码请参考：[`parallel/tensor_parallel/`](../../parallel/tensor_parallel/)

## 实践考量

### TP 度（Tensor Parallel Degree）选择

| GPU 数 | 推荐 TP 度 | 原因 |
|--------|-----------|------|
| 8 (单节点) | 8 或 4 | NVLink 高带宽，TP 通信开销小 |
| 16 (2 节点) | 8 | TP 限制在节点内，避免跨节点通信 |
| 64+ | 4-8 | TP 不跨节点，其余用 DP/PP |

**原则**：TP 度应等于或小于单节点 GPU 数，因为 TP 通信频繁且对延迟敏感。

### 通信量分析

设隐藏维度为 $h$，序列长度为 $s$，batch size 为 $b$，TP 度为 $N$：

**标准 TP（每层）**：
- 前向：1 次 AllReduce，数据量 $2bsh/N$（FP16）
- 反向：2 次 AllReduce，数据量 $2 \times 2bsh/N$
- 总计：$6bsh/N$

**序列并行 TP（每层）**：
- 前向：1 次 AllGather + 1 次 ReduceScatter，数据量 $2 \times bsh/N$
- 反向：1 次 AllGather + 2 次 ReduceScatter，数据量 $3 \times bsh/N$
- 总计：$5bsh/N$

### 显存节省

TP 度为 $N$ 时：
- 模型参数：$1/N$
- 优化器状态：$1/N$（如果也做了分片）
- 激活值：约 $1/N$（序列并行时更优）

### TP 的反向传播通信分析

张量并行的反向传播需要仔细分析通信模式：

**列切分线性层 $Y = XA_i$ 的反向**：
- $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot A_i^T$：每个 GPU 独立计算局部梯度
- 但完整的 $\frac{\partial L}{\partial X}$ 需要所有 GPU 的结果求和 → AllReduce
- $\frac{\partial L}{\partial A_i}$：每个 GPU 独立计算，无需通信

**行切分线性层 $Z = Y_i B_i$ 的反向**：
- $\frac{\partial L}{\partial Y_i} = \frac{\partial L}{\partial Z} \cdot B_i^T$：每个 GPU 独立计算
- $\frac{\partial L}{\partial B_i}$：每个 GPU 独立计算，无需通信

**Megatron-LM 组合的反向**：
- 列切分反向 → AllReduce（$\frac{\partial L}{\partial X}$）
- 行切分反向 → 无通信（$\frac{\partial L}{\partial Y_i}$ 已经是分好的）

因此，每个 Transformer 层的反向传播只需 2 次 AllReduce。

### TP 与混合精度训练

张量并行与混合精度训练结合时需要注意：
- 通信使用 FP16/BF16 格式，减少通信量
- 梯度累积使用 FP32，避免精度损失
- AllReduce 的输入和输出都是低精度格式

### 嵌入层的张量并行

词嵌入矩阵 $E \in \mathbb{R}^{V \times d}$（$V$ 为词表大小）也需要切分：

**列切分**：将 $E$ 按词表维度切分，每个 GPU 持有 $E_i \in \mathbb{R}^{V/N \times d}$。
- 前向：输入 token ID 通过取模映射到对应 GPU，查表后 AllReduce
- 反向：梯度通过 ReduceScatter 分发

**行切分**：将 $E$ 按嵌入维度切分，每个 GPU 持有 $E_i \in \mathbb{R}^{V \times d/N}$。
- 前向：每个 GPU 独立查表，结果拼接
- 反向：梯度按列切分，无需通信

### 常见问题

1. **负载不均衡**：注意力头数必须能被 TP 度整除
2. **通信瓶颈**：跨节点 TP 性能急剧下降，应避免
3. **Dropout 一致性**：TP 中的 Dropout 需要确保不同 GPU 的随机种子设置正确
4. **初始化一致性**：TP 中各 GPU 的参数初始化必须协调，确保切分后的参数组合与原始参数一致
5. **序列并行与 Flash Attention**：序列并行与 Flash Attention 可以组合使用，Flash Attention 减少显存访问，序列并行减少显存占用

## 与其他技术的关系

| 技术 | 与张量并行的关系 |
|------|----------------|
| 数据并行 | TP 组内做张量并行，TP 组间做数据并行 |
| 流水线并行 | TP 在层内切分，PP 在层间切分，可组合为 3D 并行 |
| 序列并行 | TP 的扩展，减少非矩阵运算的显存冗余 |
| 专家并行 | 非专家部分用 TP，专家部分用 EP |
| 上下文并行 | CP 沿序列切分，与 TP 的序列切分互补 |

## 参考资料

1. **Megatron-LM 论文**: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism", arXiv 2019
2. **Megatron-LM v2**: Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", SC 2021
3. **序列并行**: Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models", arXiv 2022
4. **2D/2.5D/3D 并行**: Wang et al., "2D Parallelism" series
5. **Megatron-Core**: [NVIDIA Megatron Core](https://github.com/NVIDIA/Megatron-LM)

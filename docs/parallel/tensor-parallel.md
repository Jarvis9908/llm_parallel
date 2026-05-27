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

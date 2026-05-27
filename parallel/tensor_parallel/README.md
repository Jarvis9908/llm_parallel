# 张量并行模块

将单层的权重矩阵切分到多个 GPU 上，每个 GPU 只计算一部分，通过集合通信拼接结果。

## 文件说明

| 文件 | 功能 | 关键内容 |
|------|------|---------|
| `column_parallel.py` | 列并行 Linear | `column_parallel_linear`, `split_weight_column` — 按输出维度切分权重 |
| `row_parallel.py` | 行并行 Linear | `row_parallel_linear`, `split_weight_row` — 按输入维度切分权重 |
| `embedding_parallel.py` | 并行 Embedding | `embedding_parallel_forward` — 词表切分到多卡 |
| `sequence_parallel.py` | 序列并行 | `scatter_along_seq`, `gather_along_seq`, `sp_transition_fwd` |
| `megatron_style.py` | Megatron 风格组合 | `megatron_transformer_block_fwd` — TP+SP 完整 Transformer Block |

## 核心概念

### Column Parallel vs Row Parallel

```
Column Parallel (Y = XW, 按列切 W):
  W = [W1 | W2 | ... | Wp]
  每个 GPU 计算 Yi = XWi
  最后 AllGather 拼接: Y = [Y1, Y2, ..., Yp]

Row Parallel (Y = XW, 按行切 W):
  W = [W1; W2; ...; Wp]  (纵向切)
  输入 X 对应切分: X = [X1, X2, ..., Xp]
  每个 GPU 计算 Yi = XiWi
  最后 AllReduce 求和: Y = ΣYi
```

### Sequence Parallel

在 TP 区域内，将 LayerNorm/Dropout 的激活值沿 sequence 维度切分，减少激活值显存占用。计算 Attention 时再 AllGather 拼回完整序列。

### Megatron 风格

将 Column Parallel + Row Parallel + Sequence Parallel 组合成完整的 Transformer Block：
```
Input → Scatter(seq) → LayerNorm → ColumnParallel(Attention) → AllGather → RowParallel → AllReduce → Residual
     → Scatter(seq) → LayerNorm → ColumnParallel(FFN) → AllGather → RowParallel → AllReduce → Residual
```

## 快速开始

```python
from parallel.tensor_parallel.column_parallel import column_parallel_linear, split_weight_column
from parallel.tensor_parallel.row_parallel import row_parallel_linear, split_weight_row
import torch

# 列并行示例（模拟 2 个 GPU）
full_weight = torch.randn(256, 128)
w_local = split_weight_column(full_weight, rank=0, world_size=2)  # (256, 64)
```

## 详细文档

→ [并行策略总览](../../docs/parallel/overview.md)
→ [notebook 07: 张量并行](../../notebooks/07_tensor_parallel.ipynb)

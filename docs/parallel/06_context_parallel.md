# 上下文并行详解

## 概述

上下文并行（Context Parallelism, CP），也称为序列并行（Sequence Parallelism in the context of long sequences），是专门为超长序列训练设计的并行策略。当序列长度增长到数十万甚至百万 token 时，单张 GPU 的显存无法容纳完整的注意力计算，需要将序列分段分配到不同设备上。Ring Attention 是上下文并行的核心算法，通过环形通信实现分段注意力的在线计算。

## 直觉理解

**上下文并行 = 超长序列切分段，各段独立算注意力**

想象一群人共同审阅一份超长文档：
- 每个人只看文档的一段
- 但写摘要时需要参考全文（注意力需要全局信息）
- 解决方案：大家围成一圈，依次传递各自的内容，每个人逐步累积对全文的理解
- 每传一轮，每个人就多了解一段内容，最终每个人都掌握了全文信息

**与张量并行中的序列并行的区别**：
- 张量并行的序列并行：沿序列切分以减少 LayerNorm/Dropout 的冗余，注意力仍在每张卡上完整计算
- 上下文并行：沿序列切分以减少注意力的显存，注意力通过环形通信协作计算

## 数学原理

### 长序列训练的显存瓶颈

自注意力的计算和显存需求：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**显存瓶颈**：注意力矩阵 $QK^T \in \mathbb{R}^{s \times s}$，显存占用 $O(s^2)$。

| 序列长度 | 注意力矩阵大小 (FP16) | 单层显存 |
|---------|---------------------|---------|
| 4K | 32 MB | 可接受 |
| 32K | 2 GB | 较大 |
| 128K | 32 GB | 超出单卡 |
| 1M | 2 TB | 不可行 |

**关键洞察**：序列长度增长时，注意力矩阵的显存增长是二次的，这是主要瓶颈。

### Ring Attention 的 Online Softmax 推导

Ring Attention 的核心挑战：在不完整注意力矩阵上计算 softmax。

#### 标准 Softmax 的数值稳定性

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$$

其中 $m = \max_j x_j$ 是最大值，用于数值稳定。

#### Online Softmax

设注意力分数分两块 $S^{(1)}$ 和 $S^{(2)}$，需要在线合并：

**块 1 的局部结果**：
$$m^{(1)} = \max(S^{(1)}), \quad l^{(1)} = \sum e^{S^{(1)} - m^{(1)}}, \quad o^{(1)} = \frac{\sum e^{S^{(1)} - m^{(1)}} \cdot V^{(1)}}{l^{(1)}}$$

**合并块 2 的结果**：
$$m = \max(m^{(1)}, m^{(2)}), \quad l = l^{(1)} e^{m^{(1)} - m} + l^{(2)} e^{m^{(2)} - m}$$

$$o = \frac{l^{(1)} e^{m^{(1)} - m} \cdot o^{(1)} + l^{(2)} e^{m^{(2)} - m} \cdot o^{(2)}}{l}$$

**推广到 $N$ 块**：Ring Attention 中，每个 GPU 逐步接收其他 GPU 的 $K, V$ 块，使用 online softmax 公式更新累积结果。

#### Ring Attention 完整推导

设序列被分为 $N$ 段，GPU $i$ 持有 $Q_i, K_i, V_i$。

**第 $k$ 轮**（$k = 0, 1, \ldots, N-1$）：

1. GPU $i$ 持有当前的 $K_{\text{recv}}, V_{\text{recv}}$（来自上一轮的传递）
2. 计算局部注意力分数：
   $$S_i^{(k)} = \frac{Q_i \cdot K_{\text{recv}}^T}{\sqrt{d_k}}$$
3. 更新 online softmax 统计量：
   $$m_i^{\text{new}} = \max(m_i, \max(S_i^{(k)}))$$
   $$l_i^{\text{new}} = l_i \cdot e^{m_i - m_i^{\text{new}}} + \sum e^{S_i^{(k)} - m_i^{\text{new}}}$$
4. 更新输出：
   $$O_i^{\text{new}} = \frac{O_i \cdot l_i \cdot e^{m_i - m_i^{\text{new}}} + \sum e^{S_i^{(k)} - m_i^{\text{new}}} \cdot V_{\text{recv}}}{l_i^{\text{new}}}$$
5. 将 $K_{\text{recv}}, V_{\text{recv}}$ 传递给下一个 GPU

经过 $N$ 轮后，每个 GPU 的 $O_i$ 就是完整的注意力输出。

### 序列并行的 All-Gather/Reduce-Scatter 策略

对于非注意力层（如 MLP、LayerNorm），不需要全局信息，可以直接按序列段独立计算：

```
输入: X_i (每个 GPU 持有序列的第 i 段)

1. LayerNorm: 各 GPU 独立计算
   Y_i = LayerNorm(X_i)

2. 注意力层: Ring Attention（如上所述）

3. MLP 层: 各 GPU 独立计算
   Z_i = MLP(Y_i)

输出: Z_i
```

但如果 MLP 使用了张量并行，则需要通信：

```
1. AllGather: 收集完整序列用于 TP 列切分
   X = AllGather(X_0, X_1, ..., X_{N-1})

2. TP 列切分 + 行切分 + AllReduce

3. ReduceScatter: 将结果分发回各 GPU 的序列段
   Z_i = ReduceScatter(Z)
```

### CP + TP/EP 组合方案

#### CP + TP

- TP 在注意力头维度切分，CP 在序列维度切分
- 两者正交，可以组合
- 通信模式：CP 用环形通信传递 KV，TP 用 AllReduce 同步

**组合方式**：
1. 将 GPU 分为 CP 组和 TP 组
2. CP 组内：Ring Attention 传递 KV
3. TP 组内：AllReduce 同步注意力输出

#### CP + EP

- CP 在序列维度切分，EP 在专家维度切分
- 两者正交，可以组合
- 注意：EP 的 All-to-All 通信需要完整的 token 集合

**组合方式**：
1. CP 组内各 GPU 持有部分序列
2. EP 的 All-to-All 在 CP 组内进行（token 路由到专家）
3. 每个 GPU 上的专家只处理本序列段的 token

## 算法流程

### Ring Attention 前向传播

```
输入: Q_i, K_i, V_i (GPU i 持有序列第 i 段的 QKV)

初始化:
  O_i = 0          # 累积输出
  m_i = -inf       # 累积最大值
  l_i = 0          # 累积分母

for k in range(N):  # N 个 GPU
  # 当前持有的 KV（初始为自己的，之后为接收的）
  if k == 0:
    K_cur, V_cur = K_i, V_i
  else:
    K_cur, V_cur = 接收来自上一个 GPU 的 KV

  # 计算局部注意力
  S = Q_i @ K_cur.T / sqrt(d_k)

  # Online softmax 更新
  m_new = max(m_i, max(S, dim=-1))
  l_new = l_i * exp(m_i - m_new) + sum(exp(S - m_new), dim=-1)
  O_i = (O_i * l_i * exp(m_i - m_new) + exp(S - m_new) @ V_cur) / l_new

  m_i = m_new
  l_i = l_new

  # 传递 KV 给下一个 GPU
  发送 K_cur, V_cur 给 GPU_{(i+1) % N}

输出: O_i (GPU i 持有序列第 i 段的注意力输出)
```

### Ring Attention 反向传播

反向传播需要重新传递 KV（或保存中间结果），计算梯度：

```
for k in range(N):
  # 接收 KV
  K_cur, V_cur = 接收 KV

  # 重新计算注意力分数
  S = Q_i @ K_cur.T / sqrt(d_k)
  P = softmax(S)  # 使用保存的 m_i, l_i

  # 计算梯度
  dV_cur += P.T @ dO_i
  dS = (dO_i @ V_cur.T) * P * (1 - P)  # softmax 的梯度
  dQ_i += dS @ K_cur / sqrt(d_k)
  dK_cur += dS.T @ Q_i / sqrt(d_k)

  # 传递 KV 和梯度
  发送 dK_cur, dV_cur 给对应 GPU
```

## 代码实现

本项目中的上下文并行实现位于 `parallel/context_parallel/` 目录：

| 文件 | 内容 |
|------|------|
| `ring_attention.py` | Ring Attention 的在线 softmax 实现 |
| `sequence_partition.py` | 序列分段和通信策略 |
| `cp_integration.py` | CP 与 TP/EP 的组合集成 |

```python
# 示例：使用 Ring Attention
from parallel.context_parallel.ring_attention import RingAttention

# attn = RingAttention(
#     d_model=4096,
#     n_heads=32,
#     cp_group=cp_process_group,
# )
# output = attn(query, key, value)
```

详细代码请参考：[`parallel/context_parallel/`](../../parallel/context_parallel/)

## 实践考量

### CP 度选择

| 序列长度 | 隐藏维度 | 推荐 CP 度 | 原因 |
|---------|---------|-----------|------|
| 32K | 4096 | 1-2 | 单卡可容纳 |
| 128K | 4096 | 4-8 | 注意力矩阵需要分片 |
| 512K | 4096 | 8-16 | 必须大幅分片 |
| 1M+ | 4096 | 16-32 | 极长序列 |

**原则**：CP 度应使得每段序列的注意力矩阵能放入单卡显存。

### 通信量分析

Ring Attention 的通信量：
- 每轮传递 KV：$2 \times s/N \times d$（K 和 V）
- 总轮数：$N$
- 但通信与计算可以重叠（计算当前 KV 的注意力时，同时传递下一轮 KV）
- **有效通信量**：如果计算时间 > 通信时间，通信可完全隐藏

### 实际长序列训练的配置建议

1. **Flash Attention + CP**：先用 Flash Attention 减少显存，不够再加 CP
2. **梯度检查点**：长序列训练的激活值很大，必须使用梯度检查点
3. **混合精度**：使用 BF16 而非 FP16，避免长序列训练中的数值溢出
4. **序列长度调度**：训练时逐步增加序列长度（如 4K → 32K → 128K）

### 常见问题

1. **因果注意力掩码**：Ring Attention 需要正确处理因果掩码，避免未来信息泄露
2. **变长序列**：不同样本的序列长度不同时，需要 padding 或动态批处理
3. **通信延迟**：跨节点 Ring Attention 延迟高，应尽量在节点内完成

## 与其他技术的关系

| 技术 | 与上下文并行的关系 |
|------|------------------|
| 张量并行 | CP 切序列，TP 切注意力头，可正交组合 |
| 数据并行 | CP 组间可做 DP |
| 专家并行 | CP 和 EP 正交，可组合 |
| Flash Attention | FA 减少显存访问，CP 减少显存占用，互补 |
| 稀疏注意力 | 局部注意力可减少 CP 的通信需求 |

## 参考资料

1. **Ring Attention 论文**: Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context", NeurIPS 2023
2. **Striped Attention**: Wu et al., "Striped Attention: Faster Ring Attention for Causal Transformers", 2024
3. **Long Context Training**: Meta, "Effective Long Context Scaling of Foundation Models"
4. **Flash Attention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
5. **Online Softmax**: Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018

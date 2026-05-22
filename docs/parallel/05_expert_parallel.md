# 专家并行详解

## 概述

专家并行（Expert Parallelism, EP）是混合专家（Mixture of Experts, MoE）模型特有的并行策略。MoE 模型通过稀疏激活机制，让每个 token 只激活部分专家，从而在不增加计算量的情况下扩展模型参数量。专家并行将不同的专家分配到不同设备上，token 根据路由决策被发送到对应的专家设备上处理，完成后再收集回来。

## 直觉理解

**专家并行 = 不同专家住不同卡，token 按需投递**

想象一家大型医院：
- 有多个专科医生（专家），每个医生在不同诊室（GPU）
- 病人（token）挂号时，分诊台（路由器）根据症状决定去看哪个医生
- 病人走到对应诊室（All-to-All 通信）
- 看完病后，病人回到大厅汇总（All-to-All 通信）

**核心挑战**：
- 负载均衡：某些专家可能特别热门，排长队；某些冷门，空闲
- 通信开销：token 需要在 GPU 之间来回传递

## 数学原理

### MoE 路由机制

MoE 层的计算过程：

$$\text{MoE}(x) = \sum_{i \in \text{TopK}(g(x))} g_i(x) \cdot E_i(x)$$

其中：
- $x \in \mathbb{R}^d$ 是输入 token 的表示
- $g(x) = \text{softmax}(W_g x)$ 是门控函数，$W_g \in \mathbb{R}^{E \times d}$
- $\text{TopK}(g(x))$ 选择概率最大的 $K$ 个专家
- $g_i(x)$ 是第 $i$ 个专家的权重（归一化后）
- $E_i(x)$ 是第 $i$ 个专家的输出

#### Top-K 路由

最常用的路由策略，每个 token 选择 $K$ 个专家（通常 $K=1$ 或 $K=2$）：

$$\text{TopK}(g(x), K) = \text{top-}K \text{ indices of } g(x)$$

**K=1（Switch 路由）**：每个 token 只去一个专家，计算量最少
**K=2（经典路由）**：每个 token 去两个专家，加权求和，效果更好

#### Expert Choice 路由

传统 Top-K 是"token 选专家"，Expert Choice 是"专家选 token"：

1. 计算所有 token 对所有专家的亲和度矩阵 $S = X W_g$
2. 每个专家选择亲和度最高的 $C$ 个 token（$C = \text{总 token 数} \times K / E$）
3. 天然实现负载均衡——每个专家处理相同数量的 token

**优势**：无需额外的负载均衡损失
**劣势**：某些 token 可能被多个专家选中，某些可能不被选中

### 负载均衡

#### 辅助损失（Auxiliary Loss）

为了鼓励负载均衡，添加辅助损失：

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot p_i$$

其中：
- $f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\{i \in \text{TopK}(g(x_t))\}$ 是专家 $i$ 被选中的频率
- $p_i = \frac{1}{T} \sum_{t=1}^{T} g_i(x_t)$ 是专家 $i$ 的平均门控概率
- $\alpha$ 是辅助损失系数（通常 $0.01$）

**直觉**：$f_i \cdot p_i$ 在 $f_i = p_i = 1/E$ 时最小（完全均匀），偏离均匀时增大。

#### 容量因子（Capacity Factor）

每个专家设置最大处理 token 数：

$$\text{capacity} = \text{CapacityFactor} \times \frac{T \times K}{E}$$

- CapacityFactor > 1：允许一定程度的过载
- CapacityFactor < 1：强制截断，可能丢弃 token
- 超出容量的 token 通常通过残差连接直接传递

### All-to-All 通信

专家并行的核心通信操作是 All-to-All：

**前向传播**：
1. 每个 GPU 持有一批 token，根据路由决定每个 token 应该去哪个专家
2. All-to-All dispatch：将 token 发送到对应专家所在的 GPU
3. 各 GPU 上的专家处理收到的 token
4. All-to-All combine：将处理后的 token 发回原 GPU

**All-to-All 通信量**：
- 每个 GPU 发送/接收约 $\frac{T \times K}{N}$ 个 token
- 通信量：$O(T \times K \times d / N)$

### 专家并行的通信瓶颈分析

设 $T$ 为 token 数，$d$ 为隐藏维度，$N$ 为 GPU 数，$E$ 为专家数，$K$ 为 Top-K 值。

**单步 MoE 层的通信**：
- All-to-All dispatch：发送 $\frac{T \cdot K}{N} \cdot d$ 数据
- All-to-All combine：发送 $\frac{T \cdot K}{N} \cdot d$ 数据
- 总通信量：$\frac{2TKd}{N}$

**与 TP 通信对比**：
- TP 每层 AllReduce：$\frac{2bsh}{N}$（$b$ 为 batch，$s$ 为 seq len）
- EP 每层 All-to-All：$\frac{2TKd}{N}$

当 $T = bs$ 时，EP 通信量约为 TP 的 $K$ 倍。但 EP 只在 MoE 层通信，TP 在每层都通信。

### EP + TP 组合

在大型 MoE 模型中，通常同时使用 EP 和 TP：

- **EP**：将专家分配到不同 GPU
- **TP**：每个专家内部做张量并行

**组合策略**：
1. 将 GPU 分为 EP 组和 TP 组
2. EP 组内：不同 GPU 持有不同专家
3. TP 组内：同一专家的参数切分到多个 GPU

**通信模式**：
- EP 通信（All-to-All）：在 EP 组内
- TP 通信（AllReduce）：在 TP 组内
- 两种通信串行，不能重叠

## 算法流程

### MoE 前向传播（专家并行）

```
输入: X ∈ R^{T×d} (T 个 token)

1. 路由计算:
   gates = softmax(X @ W_g)           # R^{T×E}
   topk_indices, topk_gates = TopK(gates, K)

2. All-to-All Dispatch:
   将 token 按路由结果发送到对应专家所在 GPU
   X_dispatched = AllToAll(X, topk_indices)

3. 专家计算:
   for each expert on this GPU:
       Y_expert = expert(X_expert)     # 独立计算

4. All-to-All Combine:
   将专家输出发回原 GPU
   Y_combined = AllToAll(Y_expert, reverse_mapping)

5. 加权求和:
   output = sum(topk_gates * Y_combined)
```

### 负载均衡训练流程

```
1. 前向传播中记录:
   - 每个专家被选中的次数 f_i
   - 每个专家的平均门控概率 p_i

2. 计算辅助损失:
   L_aux = alpha * E * sum(f_i * p_i)

3. 总损失:
   L_total = L_task + L_aux

4. 反向传播时:
   - 辅助损失的梯度也会更新门控参数 W_g
   - 鼓励门控函数更均匀地分配 token
```

## 代码实现

本项目中的专家并行实现位于 `parallel/expert_parallel/` 目录：

| 文件 | 内容 |
|------|------|
| `expert_partition.py` | 专家到 GPU 的分配和 All-to-All 通信 |
| `token_dispatch.py` | Token 路由和分发逻辑 |

```python
# 示例：使用专家并行
from parallel.expert_parallel.expert_partition import ExpertParallelLayer

# ep_layer = ExpertParallelLayer(
#     num_experts=8,
#     top_k=2,
#     d_model=4096,
#     d_ff=16384,
#     ep_group=expert_group,
# )
```

详细代码请参考：[`parallel/expert_parallel/`](../../parallel/expert_parallel/)

## 实践考量

### 专家数量与 GPU 数量的关系

| 配置 | 专家数 | GPU 数 | 每 GPU 专家数 |
|------|--------|--------|-------------|
| 小型 | 8 | 8 | 1 |
| 中型 | 64 | 8 | 8 |
| 大型 | 256 | 64 | 4 |
| 超大 | 256 | 256 | 1 |

**原则**：每 GPU 至少 1 个专家，通常 1-8 个专家/GPU 效率最高。

### 容量因子调优

| CapacityFactor | 效果 | 适用场景 |
|----------------|------|---------|
| 1.0 | 严格均衡，可能丢 token | 训练初期 |
| 1.25 | 适度冗余 | 通用推荐 |
| 1.5 | 较大冗余 | 负载不均衡严重时 |
| 2.0 | 大量冗余 | 极端不均衡 |

### 通信优化

1. **通信合并**：将多个小 All-to-All 合并为一个大 All-to-All
2. **通信与计算重叠**：在专家计算时预取下一批 token
3. **专家复制**：热门专家复制到多个 GPU，减少跨 GPU 通信

### 常见问题

1. **路由崩溃**：所有 token 都路由到少数专家，其他专家不被使用
   - 解决：增大辅助损失系数、使用 Expert Choice 路由
2. **token 丢弃**：超出专家容量时丢弃 token
   - 解决：增大容量因子、使用 token 丢弃的残差连接
3. **All-to-All 延迟**：跨节点 All-to-All 延迟高
   - 解决：EP 限制在节点内，跨节点用 DP

## 与其他技术的关系

| 技术 | 与专家并行的关系 |
|------|----------------|
| 数据并行 | 非专家参数用 DP 同步梯度 |
| 张量并行 | 专家内部可用 TP 切分，组合为 EP+TP |
| 流水线并行 | MoE 层作为一个整体放在某个 PP 阶段 |
| 上下文并行 | EP 和 CP 可正交组合 |
| 稀疏注意力 | MoE 的稀疏性与稀疏注意力互补 |

## 参考资料

1. **GShard 论文**: Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", ICLR 2021
2. **Switch Transformer 论文**: Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR 2022
3. **Expert Choice 路由**: Zhou et al., "Mixture-of-Experts with Expert Choice Routing", NeurIPS 2022
4. **ST-MoE**: Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", 2022
5. **DeepSeek-V2/V3 MoE**: DeepSeek 团队的 MoE 架构创新

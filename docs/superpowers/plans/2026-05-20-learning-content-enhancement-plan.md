# 学习内容增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 llm_parallel 仓库系统性地添加详细文字说明和讲解内容，使代码层、Notebook 层、文档层三层载体形成完整的学习体验。

**Architecture:** 渐进式分层增强 — 先补充代码层 docstring 和深化实现，再增强 Notebook 添加可视化和练习题，最后创建独立文档形成知识体系。每层完成后即可交付价值。

**Tech Stack:** Python 3.10+, PyTorch 2.0+, matplotlib (notebook 可视化), Jupyter Notebook, Markdown

---

## Phase 1: 代码层增强

### Task 1: 补充 tensor_parallel 模块 docstring

**Files:**
- Modify: `parallel/tensor_parallel/column_parallel.py`
- Modify: `parallel/tensor_parallel/row_parallel.py`
- Modify: `parallel/tensor_parallel/megatron_style.py`
- Modify: `parallel/tensor_parallel/sequence_parallel.py`
- Modify: `parallel/tensor_parallel/embedding_parallel.py`

- [ ] **Step 1: 补充 column_parallel.py docstring**

为文件级 docstring 和每个函数添加"直觉→公式→代码"三段式 docstring。替换现有文件级 docstring：

```python
"""列并行线性层 (Column Parallel Linear)

直觉理解：
    想象一个大矩阵乘法 Y = XW，其中 W 的列太多一张卡放不下。
    列并行的做法是把 W 按列切成 W₁, W₂, ..., Wₙ，每张卡持有一份，
    各卡用相同的输入 X 分别计算 Yᵢ = XWᵢ，最后 all-gather 拼接得到完整 Y。

数学原理：
    设 W ∈ ℝ^(d×h)，world_size = P，则每卡持有 Wᵢ ∈ ℝ^(d×h/P)。
    输入 X ∈ ℝ^(B×S×d) 在所有 rank 上相同。
    本地输出：Yᵢ = XWᵢ ∈ ℝ^(B×S×h/P)
    全局输出：Y = [Y₁, Y₂, ..., Yₚ] ∈ ℝ^(B×S×h)  (all-gather 沿最后一维拼接)

    通信量：all-gather 传输 B×S×h 个元素（每卡发送 B×S×h/P，接收 (P-1)×B×S×h/P）

代码流程：
    1. 每卡用本地权重 W_local 做矩阵乘法得到 local_out
    2. all-gather 收集所有 rank 的 local_out
    3. 沿最后一维拼接得到完整输出

与行并行的关系：
    列并行输出端需要 all-gather，而行并行输出端需要 all-reduce。
    Megatron-LM 将列并行和行并行配对使用，使列并行的 all-gather
    被行并行的 all-reduce 替代，减少一次通信。
"""
```

为 `column_parallel_linear` 函数补充 docstring：

```python
def column_parallel_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    列并行前向传播。

    直觉：每张卡用相同的输入 X 乘以自己持有的部分列权重，得到部分输出，
    然后 all-gather 拼接所有卡的部分输出，得到完整结果。

    数学：
        Y_local = X @ W_local    # (B, S, h/P)
        Y_full  = AllGather(Y_local, dim=-1)  # (B, S, h)

    Args:
        x: 输入张量，形状 (B, S, dim)，在所有 rank 上相同
        weight: 本地权重，形状 (dim, hidden_dim // world_size)，已经切分好的

    Returns:
        完整输出张量，形状 (B, S, hidden_dim)，all-gather 后的拼接结果

    Shape 推导：
        x: (B, S, dim) @ weight: (dim, h/P) → local_out: (B, S, h/P)
        all-gather → (B, S, h)
    """
```

为 `split_weight_column` 函数补充 docstring：

```python
def split_weight_column(weight: torch.Tensor) -> torch.Tensor:
    """
    将完整权重按列切分到当前 rank。

    直觉：把一个大矩阵竖着切成 P 条，每张卡拿一条。

    数学：
        W_local = W[:, rank*chunk : (rank+1)*chunk]
        其中 chunk = h // P

    Args:
        weight: 完整权重矩阵，形状 (dim, hidden_dim)

    Returns:
        当前 rank 对应的权重切片，形状 (dim, hidden_dim // world_size)
    """
```

- [ ] **Step 2: 补充 row_parallel.py docstring**

替换文件级 docstring：

```python
"""行并行线性层 (Row Parallel Linear)

直觉理解：
    列并行把矩阵按列切，行并行则按行切。想象 Y = XW，把 W 按行切成
    W₁, W₂, ..., Wₙ，输入 X 也要对应切成 X₁, X₂, ..., Xₙ。
    各卡计算 Yᵢ = XᵢWᵢ，最后 all-reduce 求和得到完整 Y。

数学原理：
    设 W ∈ ℝ^(h×d)，world_size = P，则每卡持有 Wᵢ ∈ ℝ^(h/P×d)。
    输入 X ∈ ℝ^(B×S×h) 被按列切分为 Xᵢ ∈ ℝ^(B×S×h/P)。
    本地输出：Yᵢ = XᵢWᵢ ∈ ℝ^(B×S×d)
    全局输出：Y = Σᵢ Yᵢ ∈ ℝ^(B×S×d)  (all-reduce 求和)

    通信量：all-reduce 传输 B×S×d 个元素

    数学等价性：Y = XW = [X₁,...,Xₚ][W₁;...;Wₚ] = Σᵢ XᵢWᵢ

代码流程：
    1. 每卡用本地输入 X_local 和本地权重 W_local 做矩阵乘法
    2. all-reduce 对所有 rank 的结果求和
    3. 得到完整输出（无需拼接，all-reduce 后每卡持有完整结果）

与列并行的配对：
    Megatron-LM 的关键设计：列并行后接行并行。
    列并行的输出 (B, S, h/P) 恰好是行并行需要的输入格式，
    省去了列并行输出端的 all-gather，改为行并行输出端的 all-reduce。
    这样一对列并行+行并行只需一次 all-reduce 通信。
"""
```

为 `row_parallel_linear` 函数补充 docstring：

```python
def row_parallel_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    行并行前向传播。

    直觉：每张卡用自己持有的部分输入和部分权重做乘法，得到部分结果，
    然后 all-reduce 求和所有卡的部分结果，得到完整输出。

    数学：
        Y_local = X_local @ W_local    # (B, S, d)
        Y_full  = AllReduce(Y_local, SUM)  # (B, S, d)

    Args:
        x: 本地输入张量，形状 (B, S, hidden_dim // world_size)，已按列切分
        weight: 本地权重，形状 (hidden_dim // world_size, dim)，已切分好的

    Returns:
        完整输出张量，形状 (B, S, dim)，all-reduce 求和后的结果

    Shape 推导：
        x: (B, S, h/P) @ weight: (h/P, d) → local_out: (B, S, d)
        all-reduce(SUM) → (B, S, d)
    """
```

为 `split_weight_row` 函数补充 docstring：

```python
def split_weight_row(weight: torch.Tensor) -> torch.Tensor:
    """
    将完整权重按行切分到当前 rank。

    直觉：把一个大矩阵横着切成 P 条，每张卡拿一条。

    数学：
        W_local = W[rank*chunk : (rank+1)*chunk, :]
        其中 chunk = h // P

    Args:
        weight: 完整权重矩阵，形状 (hidden_dim, dim)

    Returns:
        当前 rank 对应的权重切片，形状 (hidden_dim // world_size, dim)
    """
```

- [ ] **Step 3: 补充 megatron_style.py docstring**

替换文件级 docstring：

```python
"""Megatron-LM 风格 TP+SP Transformer 块

直觉理解：
    Megatron-LM 的核心洞察是：Transformer 块中的 Attention 和 FFN
    天然可以拆成"列并行→行并行"的对子。QKV 投影用列并行，
    Output 投影用行并行，它们之间不需要 all-gather/all-reduce，
    直接传递局部结果即可。一对列并行+行并行只需一次 all-reduce。

    序列并行 (SP) 进一步优化：在不需要通信的层（LayerNorm、Dropout）
    沿序列维度切分激活值，减少激活显存。

数学原理：
    标准 Transformer 块的数据流：
        x → LayerNorm → Attention → 残差 → LayerNorm → FFN → 残差

    TP+SP 下的数据流（P = world_size）：
        x_local (B, S/P, D)                          ← SP 区域
        → All-Gather → x_full (B, S, D)              ← 离开 SP
        → Column Parallel QKV: (B, S, 3h/P)          ← 列并行
        → Attention (本地头)
        → Row Parallel Output: (B, S, D)              ← 行并行 + All-Reduce
        → 残差连接
        → Reduce-Scatter → x_local (B, S/P, D)       ← 进入 SP
        → LayerNorm (本地序列)                         ← SP 区域
        → All-Gather → x_full (B, S, D)              ← 离开 SP
        → Column Parallel Gate+Up: (B, S, 2h/P)      ← 列并行
        → SwiGLU 激活
        → Row Parallel Down: (B, S, D)                ← 行并行 + All-Reduce
        → 残差连接
        → Reduce-Scatter → x_local (B, S/P, D)       ← 进入 SP

    通信次数：每个 Transformer 块 4 次通信
        - 2 次 All-Gather（进入 Attention 和 FFN 前）
        - 2 次 Reduce-Scatter（Attention 和 FFN 后）

代码流程：
    1. SP 区域：LayerNorm 在本地序列块上计算
    2. All-Gather 恢复完整序列
    3. Column Parallel：QKV/Gate+Up 投影
    4. Row Parallel：Output/Down 投影 + All-Reduce
    5. Reduce-Scatter 回到 SP 区域
"""
```

为 `megatron_transformer_block_fwd` 函数补充 docstring：

```python
def megatron_transformer_block_fwd(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    use_sp: bool = True,
) -> torch.Tensor:
    """
    模拟 Megatron 风格 Transformer 块的 TP+SP 前向。

    直觉：把 Transformer 块的每对投影层（QKV→O, Gate+Up→Down）
    拆成列并行+行并行的组合，中间不需要全局通信。在非投影层
    （LayerNorm、Dropout）沿序列维度切分以节省激活显存。

    数学：
        每个 Transformer 块的通信量 = 4 × B × S × D × sizeof(dtype)
        - 2 次 All-Gather: B×S×D 元素
        - 2 次 Reduce-Scatter: B×S×D 元素
        相比纯 TP（2 次 All-Reduce = 4 次 All-Gather/Reduce-Scatter），
        通信量相同但激活显存减少 (P-1)/P。

    Args:
        x: 输入张量，形状 (B, S, D)
        w_qkv: QKV 投影权重（列并行切分后），形状 (D, 3*h/P)
        w_o: Output 投影权重（行并行切分后），形状 (h/P, D)
        w_gate_up: Gate+Up 投影权重（列并行切分后），形状 (D, 2*h/P)
        w_down: Down 投影权重（行并行切分后），形状 (h/P, D)
        use_sp: 是否启用序列并行

    Returns:
        输出张量，形状 (B, S, D)
    """
```

- [ ] **Step 4: 补充 sequence_parallel.py docstring**

替换文件级 docstring：

```python
"""序列并行 (Sequence Parallelism, SP)

直觉理解：
    在张量并行 (TP) 中，Attention 和 FFN 的计算需要完整的序列数据，
    但 LayerNorm 和 Dropout 只在单个 token 上操作，不需要跨 token 信息。
    序列并行利用这一点：在 LayerNorm/Dropout 阶段沿序列维度切分激活值，
    每张卡只存 1/P 的序列，减少激活显存占用。

数学原理：
    设序列长度为 S，world_size = P，则每卡持有 S/P 个 token 的激活值。

    LayerNorm 在 SP 下：
        μ_local = mean(x_local)  # 仅在本地 S/P 个 token 上计算
        σ²_local = var(x_local)  # 同上
        x_norm_local = (x_local - μ_local) / √(σ²_local + ε)

    注意：严格来说，LayerNorm 需要全局统计量，SP 下使用 RMSNorm
    （不需要均值）或接受近似误差。Megatron-LM 的 SP 实际使用
    Reduce-Scatter 和 All-Gather 在 TP 和 SP 之间切换。

    激活显存节省：
        无 SP：每卡存储 (B, S, D) 的完整激活
        有 SP：每卡存储 (B, S/P, D) 的局部激活
        节省比例：(P-1)/P

代码流程：
    1. scatter_along_seq: 将 (B, S, D) 切分为 (B, S/P, D)
    2. 在本地块上执行 LayerNorm/Dropout
    3. gather_along_seq: 恢复为 (B, S, D) 供 Attention/FFN 使用

与 TP 的关系：
    SP 不是独立的并行策略，而是 TP 的优化。SP 区域（LayerNorm/Dropout）
    和 TP 区域（Attention/FFN）交替出现，通过 All-Gather 和 Reduce-Scatter
    在两者之间切换分片维度。
"""
```

为 `scatter_along_seq`、`gather_along_seq`、`sp_transition_fwd` 函数补充 docstring：

```python
def scatter_along_seq(x: torch.Tensor) -> torch.Tensor:
    """
    将张量沿序列维度切分，每个 rank 保留本地 chunk。

    直觉：把一条长序列切成 P 段，每张卡拿一段。

    数学：
        x_local = x[:, rank*chunk : (rank+1)*chunk, :]
        其中 chunk = S // P

    Args:
        x: 完整输入张量，形状 (B, S, D)

    Returns:
        本地序列块，形状 (B, S // world_size, D)
    """
```

```python
def gather_along_seq(x: torch.Tensor, total_seq_len: int) -> torch.Tensor:
    """
    从所有 rank 收集序列块，拼回完整序列。

    直觉：每张卡把自己的一段序列交出来，拼成完整序列。

    数学：
        x_full = Concat([x₀, x₁, ..., xₚ], dim=1)[:, :S, :]

    Args:
        x: 本地序列块，形状 (B, S/P, D)
        total_seq_len: 完整序列长度 S

    Returns:
        完整序列张量，形状 (B, total_seq_len, D)
    """
```

```python
def sp_transition_fwd(x: torch.Tensor, from_sp: bool = True) -> torch.Tensor:
    """
    SP 与非 SP 区域之间的转换。

    直觉：进出 LayerNorm 时需要切换序列的分片状态。

    数学：
        from_sp=True  (离开 SP): All-Gather (B, S/P, D) → (B, S, D)
        from_sp=False (进入 SP): Reduce-Scatter (B, S, D) → (B, S/P, D)
        当前简化实现用 scatter 代替 Reduce-Scatter。

    Args:
        x: 输入张量
        from_sp: True 表示离开 SP 区域（需要 all-gather），
                 False 表示进入 SP 区域（需要 reduce-scatter/切分）

    Returns:
        转换后的张量
    """
```

- [ ] **Step 5: 补充 embedding_parallel.py docstring**

替换文件级 docstring：

```python
"""Embedding 并行 (Embedding Parallelism)

直觉理解：
    大语言模型的词表通常很大（如 LLaMA 3 的 128K），Embedding 矩阵
    (vocab_size × dim) 占用大量显存。Embedding 并行将词表沿 vocab 维度
    切分到不同 rank，每张卡只存一部分词向量，查表后 all-reduce 收集结果。

数学原理：
    设词表大小 V，隐藏维度 D，world_size = P。
    每卡持有 Embedding 矩阵 Eᵢ ∈ ℝ^(V/P × D)。

    前向过程：
    1. 每个 rank 对输入 token_ids 查本地 Embedding：
       - 如果 token_id ∈ [i*V/P, (i+1)*V/P)，rank i 返回对应向量
       - 其他 rank 返回零向量
    2. All-Reduce 求和：E(token) = Σᵢ Eᵢ(token) = E_{owner(token)}(token)

    显存节省：每卡存储 V/P × D 个参数，节省 (P-1)/P。

    通信量：All-Reduce 传输 B × S × D 个元素。

代码流程：
    1. 每个 rank 对 token_ids 查本地 Embedding（非本地 token 返回零）
    2. All-Reduce 求和得到完整 Embedding 输出

    当前简化实现：假设 token_ids 已在本地词表范围内，直接查表后 all-reduce。

与 TP 的关系：
    Embedding 并行可以看作 TP 在 Embedding 层的特例——
    沿 vocab 维度切分等价于对 Embedding 矩阵做列并行。
    在 Megatron-LM 中，Embedding 层与第一层 Column Parallel Linear
    共享切分方式，避免额外的通信。
"""
```

为 `embedding_parallel_forward` 函数补充 docstring：

```python
def embedding_parallel_forward(
    embed_weight: torch.Tensor, token_ids: torch.LongTensor
) -> torch.Tensor:
    """
    Embedding 并行前向传播。

    直觉：每张卡只存一部分词向量，查表后 all-reduce 求和得到完整结果。

    数学：
        local_out = Embedding(token_ids, E_local)  # (B, S, D)
        full_out  = AllReduce(local_out, SUM)       # (B, S, D)

    Args:
        embed_weight: 本地 Embedding 权重，形状 (vocab_size_local, dim)
        token_ids: 输入 token id，形状 (B, S)

    Returns:
        完整 Embedding 输出，形状 (B, S, D)

    注意：
        当前简化实现假设 token_ids 已在本地词表范围内。
        完整实现需要：(1) 判断 token 是否属于本地范围，
        (2) 非本地 token 返回零向量，(3) all-reduce 求和。
    """
```

- [ ] **Step 6: 验证修改后代码可正常运行**

Run: `python -c "from parallel.tensor_parallel import column_parallel, row_parallel, megatron_style, sequence_parallel, embedding_parallel; print('All imports OK')"`

Expected: `All imports OK`

---

### Task 2: 补充 expert_parallel 模块 docstring

**Files:**
- Modify: `parallel/expert_parallel/expert_partition.py`
- Modify: `parallel/expert_parallel/token_dispatch.py`

- [ ] **Step 1: 补充 expert_partition.py docstring**

替换文件级 docstring：

```python
"""专家分区 (Expert Partition)

直觉理解：
    MoE（混合专家）模型中有多个 Expert 网络，专家并行将不同的 Expert
    分配到不同的 GPU 上。就像一个公司有多个部门，每个部门专门处理
    特定类型的任务，员工（token）根据需求被派到对应部门。

数学原理：
    设 N 个 Expert，P 张 GPU，则每卡持有 ⌈N/P⌉ 个 Expert。

    均匀分配（N % P == 0）：
        Rank i 持有 Expert [i*N/P, (i+1)*N/P)
        每卡 N/P 个 Expert

    非均匀分配（N % P ≠ 0）：
        前 N%P 个 rank 各多持 1 个 Expert
        Rank i 持有 ⌊N/P⌋ + (1 if i < N%P else 0) 个 Expert

    负载均衡目标：各卡 Expert 数量差异不超过 1。

代码流程：
    1. partition_experts(): 计算当前 rank 持有的 Expert 索引列表
    2. get_expert_owner(): 查询指定 Expert 属于哪个 rank

与 All-to-All 通信的关系：
    Expert 分区决定了 token 的路由目标。当 token 需要的 Expert
    在其他 rank 上时，需要通过 All-to-All 通信将 token 发送到
    对应 rank，计算完成后再发回来。
"""
```

为 `partition_experts` 和 `get_expert_owner` 函数补充 docstring：

```python
def partition_experts(n_experts: int, rank: int, world_size: int) -> list[int]:
    """
    计算当前 rank 持有的 Expert 索引列表。

    直觉：把 N 个 Expert 尽可能均匀地分给 P 张卡，多的给前面的卡。

    数学：
        均匀部分：每卡 ⌊N/P⌋ 个
        余数部分：前 N%P 个 rank 各多 1 个
        Rank i 的范围：[i*⌊N/P⌋ + min(i, N%P), 下一个 rank 的 start)

    Args:
        n_experts: Expert 总数
        rank: 当前 rank 编号
        world_size: GPU 总数

    Returns:
        当前 rank 持有的 Expert 索引列表
    """
```

```python
def get_expert_owner(expert_idx: int, world_size: int) -> int:
    """
    返回持有指定 Expert 的 rank。

    直觉：给定一个 Expert 编号，查它在哪张卡上。

    数学：
        owner = expert_idx % world_size（简单取模分配）

    Args:
        expert_idx: Expert 索引
        world_size: GPU 总数

    Returns:
        持有该 Expert 的 rank 编号
    """
```

- [ ] **Step 2: 补充 token_dispatch.py docstring**

替换文件级 docstring：

```python
"""Token 分发 (Token Dispatch)

直觉理解：
    MoE 模型中，Router 决定每个 token 应该由哪个 Expert 处理。
    当 Expert 分布在不同 GPU 上时，token 需要被"投递"到对应的卡，
    处理完再"取回"。这就是 Token 分发——像快递分拣中心一样，
    根据目的地（Expert 所在 rank）将包裹（token）分类投递。

数学原理：
    设 N 个 Expert，P 张 GPU，batch 中 B×S 个 token，Top-K 路由。

    路由矩阵 R ∈ {0,1}^(B×S×N)：
        R[b,s,e] = 1 表示 token (b,s) 被路由到 Expert e

    分发过程（All-to-All）：
        1. 每个 rank 根据 R 将本地 token 按 Expert 归属分组
        2. All-to-All 通信：将 token 发送到 Expert 所在的 rank
        3. 每个 rank 接收属于自己的 token，执行 Expert 计算
        4. All-to-All 通信：将计算结果发回原 rank
        5. 原 rank 根据 R 将结果合并

    通信量：2 × B × S × D × sizeof(dtype)（两次 All-to-All）

    负载均衡挑战：
        如果大量 token 被路由到少数 Expert（路由坍塌），
        会导致某些 rank 过载、其他 rank 空闲。
        解决方案：辅助损失（auxiliary loss）、容量因子（capacity factor）、
        Expert Choice 路由等。

代码流程：
    1. dispatch_tokens_to_experts(): 根据 router 结果将 token 分组
    2. all_to_all_dispatch_example(): 演示 All-to-All 通信在 EP 中的角色
"""
```

为 `dispatch_tokens_to_experts` 和 `all_to_all_dispatch_example` 函数补充 docstring：

```python
def dispatch_tokens_to_experts(
    x: torch.Tensor, router_indices: torch.Tensor, n_experts: int
) -> dict[int, torch.Tensor]:
    """
    根据 router indices 将 token 分发到对应的 Expert。

    直觉：拿着一张路由表，把每个 token 送到它该去的 Expert 那里。

    数学：
        对每个 Expert e，收集所有 router_indices 中包含 e 的 token：
        tokens_e = {x[b,s] | ∃k: router_indices[b,s,k] = e}

    Args:
        x: 输入 token 张量，形状 (B, S, D)
        router_indices: Top-K 路由结果，形状 (B, S, K)，每个 token 的 K 个 Expert 索引
        n_experts: Expert 总数

    Returns:
        字典 {expert_idx: token_tensor}，每个 Expert 对应的 token 集合
    """
```

```python
def all_to_all_dispatch_example(
    x: torch.Tensor, rank: int, world_size: int
) -> torch.Tensor:
    """
    演示 All-to-All 通信在 EP 中的角色。

    直觉：每张卡既是发送方也是接收方，需要把不属于本卡的 token
    发出去，同时接收属于本卡的 token。

    数学：
        All-to-All 是最通用的集合通信原语：
        每个 rank i 向 rank j 发送数据 send[i][j]
        每个 rank j 从 rank i 接收数据 recv[j][i]
        通信量 = P × 单次发送量

    Args:
        x: 本地 token 张量
        rank: 当前 rank
        world_size: GPU 总数

    Returns:
        分发后的 token 张量（当前简化实现直接返回输入）
    """
```

- [ ] **Step 3: 验证修改后代码可正常运行**

Run: `python -c "from parallel.expert_parallel import expert_partition, token_dispatch; print('All imports OK')"`

Expected: `All imports OK`

---

### Task 3: 补充 pipeline_parallel 模块 docstring

**Files:**
- Modify: `parallel/pipeline_parallel/gpiped.py`
- Modify: `parallel/pipeline_parallel/f1b1.py`
- Modify: `parallel/pipeline_parallel/layer_partition.py`

- [ ] **Step 1: 补充 gpiped.py docstring**

替换文件级 docstring：

```python
"""GPipe 流水线调度

直觉理解：
    想象一条汽车装配线：车身依次经过焊接→喷漆→组装→质检 4 个工位。
    GPipe 的做法是：先把所有 mini-car 都推过焊接，再全部推过喷漆……
    这样简单，但中间每个工位都要等前一个工位全部完成才能开始，
    造成大量空闲时间（bubble）。

数学原理：
    设 P 个 stage（工位），M 个 micro-batch。

    GPipe 调度时间线：
        Stage 0: [F0 F1 F2 F3]                    [B0 B1 B2 B3]
        Stage 1:          [F0 F1 F2 F3]           [B0 B1 B2 B3]
        Stage 2:                   [F0 F1 F2 F3]  [B0 B1 B2 B3]

        F = forward, B = backward
        中间空白区域 = bubble

    Bubble time 公式：
        bubble_ratio = (P - 1) / (P - 1 + M)

        当 M >> P 时，bubble 趋近于 0。
        当 M = P 时，bubble ≈ 50%。

    激活显存问题：
        GPipe 需要保存所有 M 个 micro-batch 的中间激活值，
        直到 backward 阶段才能释放。激活显存峰值 = M × 单个 micro-batch 激活。

代码流程：
    1. 将 mini-batch 切分为 M 个 micro-batch
    2. 所有 micro-batch 依次 forward（保存所有中间值）
    3. 所有 micro-batch 依次 backward（释放中间值）
    4. 累加梯度，更新参数
"""
```

为 `gpiped_forward` 和 `compute_gpipe_bubble_time` 函数补充 docstring：

```python
def gpiped_forward(
    micro_batches: list[torch.Tensor],
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    """
    GPipe 风格前向传播：所有 micro-batch 依次 forward。

    直觉：把所有小批次一个接一个推过当前 stage，全部完成后才开始 backward。

    数学：
        outputs = [forward(mb₀), forward(mb₁), ..., forward(mb_{M-1})]
        需要保存所有 M 个输出供 backward 使用。

    Args:
        micro_batches: 切分后的 micro-batch 列表
        forward_fn: 前向传播函数

    Returns:
        所有 micro-batch 的输出列表

    注意：
        激活显存峰值 = M × 单个 micro-batch 激活大小。
        这是 GPipe 的主要缺点，1F1B 调度可以缓解。
    """
```

```python
def compute_gpipe_bubble_time(n_micro_batches: int, n_stages: int) -> float:
    """
    计算 GPipe 的 bubble time 比例。

    直觉：流水线启动和排空期间，部分 stage 空闲，这就是 bubble。

    数学：
        bubble_ratio = (P - 1) / (P - 1 + M)

        推导：
        - 总时间步 = (P-1) + M  （启动 P-1 步 + 处理 M 个 micro-batch）
        - Bubble 时间步 = P - 1  （启动/排空期间）
        - Bubble 比例 = (P-1) / (P-1+M)

    Args:
        n_micro_batches: micro-batch 数量 M
        n_stages: 流水线 stage 数量 P

    Returns:
        Bubble time 比例，范围 [0, 1)
    """
```

- [ ] **Step 2: 补充 f1b1.py docstring**

替换文件级 docstring：

```python
"""1F1B 流水线调度 (One Forward One Backward)

直觉理解：
    GPipe 的问题是：所有 forward 做完才做 backward，激活显存峰值高。
    1F1B 的改进是：一旦某个 micro-batch 的 forward 完成，立刻做它的 backward，
    释放中间激活值。就像装配线上，一辆车焊完立刻开始拆（反向传播），
    不用等所有车都焊完。

数学原理：
    设 P 个 stage，M 个 micro-batch。

    1F1B 调度时间线（P=4, M=8）：
        Stage 0: [F0 F1 F2] [F3 B0 F4 B1 F5 B2 F6 B3 F7 B4] [B5 B6 B7]
        Stage 1:       [F0 F1 F2] [F3 B0 F4 B1 F5 B2 F6 B3] [B4 B5 B6 B7]
        Stage 2:             [F0 F1 F2] [F3 B0 F4 B1 F5 B2] [B3 B4 B5 B6 B7]
        Stage 3:                   [F0 F1 F2] [F3 B0 F4 B1] [B2 B3 B4 B5 B6 B7]

        三个阶段：
        1. Warmup: 连续 P-1 个 forward（填充流水线）
        2. Steady: 交替 1F+1B（保持流水线满载，同时释放激活）
        3. Cooldown: 处理剩余 backward

    激活显存优势：
        GPipe 峰值：M 个 micro-batch 的激活
        1F1B 峰值：P 个 micro-batch 的激活（仅需保存 warmup 阶段的）
        当 M >> P 时，1F1B 显存节省显著。

    Bubble time：
        与 GPipe 相同：bubble_ratio = (P-1) / (P-1+M)
        1F1B 减少的是激活显存，不是 bubble。

代码流程：
    1. Warmup 阶段：连续做 P-1 个 forward
    2. Steady 阶段：交替做 1 forward + 1 backward
    3. Cooldown 阶段：处理剩余 backward
"""
```

为 `f1b1_schedule` 和 `compute_1f1b_bubble_time` 函数补充 docstring：

```python
def f1b1_schedule(
    micro_batches: list[torch.Tensor],
    forward_fn,
    backward_fn,
    n_warmup: int = 3,
) -> list[torch.Tensor]:
    """
    1F1B (One Forward One Backward) 调度。

    直觉：先填满流水线（warmup），然后一边进一边出（steady），
    最后排空（cooldown）。

    数学：
        Warmup: P-1 个 forward（保存激活）
        Steady: M-P+1 轮 {1 forward + 1 backward}（边进边出）
        Cooldown: P-1 个 backward（排空）

        激活峰值 = P（仅需保存 warmup 阶段的 P-1 个 + 当前 1 个）

    Args:
        micro_batches: 切分后的 micro-batch 列表
        forward_fn: 前向传播函数
        backward_fn: 反向传播函数
        n_warmup: warmup 阶段的 forward 次数，通常 = P - 1

    Returns:
        所有 micro-batch 的 loss 列表
    """
```

```python
def compute_1f1b_bubble_time(n_micro_batches: int, n_stages: int) -> float:
    """
    计算 1F1B 调度的 bubble time 比例。

    直觉：1F1B 的 bubble 与 GPipe 相同，都是流水线启动/排空造成的。

    数学：
        n_warmup = P - 1
        total_steps = 2 × (n_warmup + M) - 1
        idle_steps = n_warmup × 2
        bubble_ratio = idle_steps / total_steps

        注意：1F1B 的 bubble 与 GPipe 相同，优势在于激活显存而非 bubble。

    Args:
        n_micro_batches: micro-batch 数量 M
        n_stages: 流水线 stage 数量 P

    Returns:
        Bubble time 比例
    """
```

- [ ] **Step 3: 补充 layer_partition.py docstring**

替换文件级 docstring：

```python
"""流水线层切分 (Layer Partition)

直觉理解：
    流水线并行的第一步是把模型按层切成若干 stage。就像把一本书分成
    几个人同时看，每个人负责一部分章节。关键是尽量让每个人的工作量
    相当，避免有人忙死有人闲死。

数学原理：
    设 L 层 Transformer，P 个 stage（GPU），则每卡持有 ⌈L/P⌉ 层。

    均匀分配（L % P == 0）：
        Stage i 负责层 [i*L/P, (i+1)*L/P)

    非均匀分配（L % P ≠ 0）：
        前 L%P 个 stage 各多 1 层
        Stage i 负责 ⌊L/P⌋ + (1 if i < L%P else 0) 层

    负载均衡考量：
        理想情况：每个 stage 的计算时间相同。
        实际挑战：不同层的计算量可能不同（如 MoE 层 vs Dense 层）。
        高级策略：按实际计算时间而非层数来划分。

代码流程：
    1. get_layer_range(): 计算当前 rank 负责的层范围
    2. partition_layers(): 将层列表分配到当前 rank（简化版）
"""
```

为 `partition_layers` 和 `get_layer_range` 函数补充 docstring：

```python
def partition_layers(layers: list[torch.nn.Module]) -> list[torch.nn.Module]:
    """
    简化的层分配：所有层都在本地（单机演示用）。

    注意：实际分布式训练中，每个 rank 只持有自己负责的层子集。
    此函数用于单机演示，返回完整的层列表。
    """
```

```python
def get_layer_range(n_layers: int, rank: int, world_size: int) -> tuple[int, int]:
    """
    计算当前 rank 负责的层范围 [start, end)。

    直觉：把 L 层尽量均匀地分给 P 个 rank，多出来的给前面的 rank。

    数学：
        base = ⌊L/P⌋
        remainder = L % P
        start = rank × base + min(rank, remainder)
        end = start + base + (1 if rank < remainder else 0)

    Args:
        n_layers: 总层数 L
        rank: 当前 rank 编号
        world_size: GPU 总数 P

    Returns:
        (start, end) 层索引范围，左闭右开
    """
```

- [ ] **Step 4: 验证修改后代码可正常运行**

Run: `python -c "from parallel.pipeline_parallel import gpiped, f1b1, layer_partition; print('All imports OK')"`

Expected: `All imports OK`

---

### Task 4: 补充 context_parallel 模块 docstring

**Files:**
- Modify: `parallel/context_parallel/ring_attention.py`
- Modify: `parallel/context_parallel/sequence_partition.py`
- Modify: `parallel/context_parallel/cp_integration.py`

- [ ] **Step 1: 补充 ring_attention.py docstring**

替换文件级 docstring：

```python
"""环形注意力 (Ring Attention)

直觉理解：
    处理超长序列时，单张 GPU 放不下完整的 KV Cache。Ring Attention
    把序列切成段，每张卡持有一段，KV 像接力棒一样在环形拓扑中传递。
    每张卡用本地的 Q 和传过来的 KV 计算部分注意力，最终累积得到
    完整结果。就像多人接力跑，每个人跑一段，棒子传给下一个人。

数学原理：
    设序列长度 S，world_size = P，每卡持有 S/P 个 token 的 Q。

    标准 Attention：
        O = softmax(QK^T / √d) V

    Ring Attention 分块计算：
        对每个 step t = 0, 1, ..., P-1：
            1. Rank i 持有 Q_i 和 KV_{(i+t) mod P}
            2. 计算局部注意力分数：S_i^t = Q_i @ K_{(i+t)%P}^T / √d
            3. 更新 running max：m_i = max(m_i, max(S_i^t, dim=-1))
            4. 更新 running sum：l_i = l_i × exp(m_i_old - m_i) + Σ exp(S_i^t - m_i)
            5. 更新输出：O_i = O_i × (l_i_old / l_i) × exp(m_i_old - m_i) + (exp(S_i^t - m_i) / l_i) @ V_{(i+t)%P}
            6. 将 KV 传给下一个 rank（环形传递）

    Online Softmax 关键：
        标准 softmax 需要一次性看到所有 K 才能计算分母。
        Ring Attention 使用 online softmax（增量计算）：
        - 维护 running max m 和 running sum l
        - 每收到新的 KV 块，用修正因子更新已有结果
        - 最终结果与标准 softmax 数学等价

    通信量：每步传输 2 × (B × n_heads × S/P × d) 个元素（K 和 V）
            总通信量 = P × 2 × B × n_heads × S/P × d = 2 × B × n_heads × S × d
            与标准 Attention 的 O(S²) 计算量相比，通信量是 O(S) 级别。

代码流程：
    1. ring_attention_step(): 计算单步部分注意力
    2. rotate_kv(): 在环形拓扑中传递 KV block
    3. 重复 P 步，累积得到完整注意力输出

    当前简化实现：直接对完整 Q/K/V 做 softmax，未实现 online softmax。
    完整实现需要维护 running max 和 running sum，逐步修正结果。
"""
```

为 `ring_attention_step` 和 `rotate_kv` 函数补充 docstring：

```python
def ring_attention_step(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, step: int
) -> torch.Tensor:
    """
    环形注意力的单步计算。

    直觉：用本地的 Q 和当前持有的 KV 计算这一步的部分注意力。

    数学（简化版，未实现 online softmax）：
        scores = Q @ K^T / √d
        attn = softmax(scores)
        output = attn @ V

    完整版（online softmax）：
        1. 计算局部分数：S = Q @ K^T / √d
        2. 更新 running max：m_new = max(m_old, max(S, dim=-1))
        3. 计算修正因子：α = exp(m_old - m_new)
        4. 更新输出：O = O × α + softmax(S - m_new) @ V
        5. 更新 running sum：l = l × α + sum(softmax(S - m_new))

    Args:
        q: 查询张量，形状 (B, n_heads, S_local, d_head)
        k: 键张量（当前持有的 KV block），形状 (B, n_heads, S_local, d_head)
        v: 值张量（当前持有的 KV block），形状 (B, n_heads, S_local, d_head)
        step: 当前步数（0 到 P-1）

    Returns:
        部分注意力输出，形状 (B, n_heads, S_local, d_head)
    """
```

```python
def rotate_kv(
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]], direction: int = 1
):
    """
    在环形拓扑中传递 KV block。

    直觉：KV 像接力棒一样，每步传给下一个 rank。

    数学：
        direction=1:  KV 从 rank i 传到 rank (i+1) % P
        direction=-1: KV 从 rank i 传到 rank (i-1) % P

    Args:
        kv_cache: KV block 列表，每个元素为 (K, V) 元组
        direction: 传递方向，1=向前，-1=向后

    Returns:
        传递后的 KV block 列表（当前简化实现为 no-op）
    """
```

- [ ] **Step 2: 补充 sequence_partition.py docstring**

替换文件级 docstring：

```python
"""上下文并行中的序列维度切分和 causal mask 调整

直觉理解：
    上下文并行把超长序列切成多段，每张卡处理一段。但自回归模型
    有因果性约束：每个 token 只能看到它之前的 token。切分后，
    后面的段需要"知道"前面段的存在，所以 causal mask 需要调整。

数学原理：
    设序列长度 S，world_size = P，每卡持有 S/P 个 token。

    序列切分：
        Rank i 持有 token [i*S/P, (i+1)*S/P)

    Causal mask 调整：
        Rank 0: 标准 causal mask（只看本地之前的 token）
        Rank i (i>0): 可以看到前面所有段的 token + 本地 causal 部分

        mask[j, k] = True (masked/不可见) 如果：
            k ≥ local_start + j + 1  (本地 causal 约束)
            且 k < local_start       (前面的段全部可见)

    Shape：
        本地 mask 形状：(S/P, local_start + S/P)
        = (S/P, (i+1)*S/P)  对于 Rank i

代码流程：
    1. partition_sequence(): 将输入沿 seq_len 均分到各 rank
    2. create_cp_causal_mask(): 生成调整后的 causal mask
"""
```

为 `partition_sequence` 和 `create_cp_causal_mask` 函数补充 docstring：

```python
def partition_sequence(x: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """
    将输入沿 seq_len 均分到各 rank。

    直觉：把一条长序列切成 P 段，每张卡拿一段。

    数学：
        chunk_size = S // P
        x_local = x[:, rank*chunk_size : (rank+1)*chunk_size, :]
        最后一个 rank 取到序列末尾（处理不能整除的情况）

    Args:
        x: 输入张量，形状 (B, S, D)
        rank: 当前 rank 编号
        world_size: GPU 总数

    Returns:
        本地序列块，形状 (B, S/P, D) 或 (B, S/P+remainder, D)（最后一个 rank）
    """
```

```python
def create_cp_causal_mask(
    seq_len: int, rank: int, world_size: int
) -> torch.Tensor:
    """
    生成上下文并行下的 causal mask。

    直觉：后面的 rank 可以"偷看"前面 rank 的所有 token，
    但不能看自己位置之后的 token。

    数学：
        chunk_size = S // P
        local_start = rank * chunk_size
        mask 形状：(chunk_size, local_start + chunk_size)
        mask[i, k] = False（可见）如果 k < local_start + i + 1
        mask[i, k] = True（masked）如果 k ≥ local_start + i + 1

    Args:
        seq_len: 完整序列长度 S
        rank: 当前 rank 编号
        world_size: GPU 总数

    Returns:
        Causal mask 张量，True 表示 masked（不可见），False 表示可见
    """
```

- [ ] **Step 3: 补充 cp_integration.py docstring**

替换文件级 docstring：

```python
"""CP 与其他并行策略的混合方案

直觉理解：
    单靠一种并行策略往往不够。比如 TP 切分模型权重，CP 切分序列长度，
    两者组合可以同时减少权重显存和激活显存。就像搬家时既把家具拆开（TP），
    又分批运送（CP），两种策略互补。

数学原理：
    CP+TP 混合显存分析：
        总激活显存 ∝ B × S × D
        TP 后：每卡激活 ∝ B × S × D / P_tp
        CP 后：每卡激活 ∝ B × (S/P_cp) × D / P_tp
        CP+TP 后：每卡激活 ∝ B × S × D / (P_tp × P_cp)

    并行策略选择启发式规则：
        模型 < 1GB: DP only（模型够小，无需切分）
        模型 1-10GB: TP + DP（单机多卡）
        模型 > 10GB: TP + PP + DP（多机多卡）
        长序列 (>8K): 额外加 CP（序列太长，激活显存不够）

    注意：这是简化启发式，实际选择还需考虑：
        - GPU 间互联带宽（NVLink vs PCIe）
        - 模型架构（Dense vs MoE）
        - 训练 vs 推理的不同需求

代码流程：
    1. analyze_cp_tp_memory(): 分析 CP+TP 混合下的显存占用
    2. recommend_parallel_config(): 根据模型大小和 GPU 数推荐并行配置
"""
```

为 `analyze_cp_tp_memory` 和 `recommend_parallel_config` 函数补充 docstring：

```python
def analyze_cp_tp_memory(
    seq_len: int, dim: int, n_heads: int, tp_size: int, cp_size: int
) -> dict:
    """
    分析 CP+TP 混合下的显存占用。

    直觉：TP 减少权重显存，CP 减少激活显存，两者乘积是总节省。

    数学：
        总激活 = S × D（简化，忽略 batch 和 heads）
        每卡激活 = S × D / (P_tp × P_cp)
        节省比例 = 1 - 1/(P_tp × P_cp)

    Args:
        seq_len: 序列长度 S
        dim: 隐藏维度 D
        n_heads: 注意力头数
        tp_size: 张量并行度 P_tp
        cp_size: 上下文并行度 P_cp

    Returns:
        包含 total_activation_memory、per_device_activation、reduction_ratio 的字典
    """
```

```python
def recommend_parallel_config(
    model_size_gb: float, seq_len: int, n_gpus: int
) -> str:
    """
    根据模型大小和 GPU 数量推荐并行配置。

    直觉：小模型不需要切分，大模型需要多种并行策略组合。

    启发式规则：
        模型 < 1GB: DP only
        模型 1-10GB: TP + DP
        模型 > 10GB: TP + PP + DP
        长序列 (>8K): 额外加 CP

    Args:
        model_size_gb: 模型参数占用显存（GB）
        seq_len: 输入序列长度
        n_gpus: 可用 GPU 数量

    Returns:
        推荐的并行策略描述字符串
    """
```

- [ ] **Step 4: 验证修改后代码可正常运行**

Run: `python -c "from parallel.context_parallel import ring_attention, sequence_partition, cp_integration; print('All imports OK')"`

Expected: `All imports OK`

---

### Task 5: 补充 inference 模块 docstring

**Files:**
- Modify: `parallel/inference/speculative_decoding.py`
- Modify: `parallel/inference/prefill_decode.py`
- Modify: `parallel/inference/kv_cache_shard.py`

- [ ] **Step 1: 补充 speculative_decoding.py docstring**

`speculative_decoding.py` 已有较好的 docstring，主要补充文件级 docstring。

替换文件级 docstring：

```python
"""推测解码 (Speculative Decoding)

直觉理解：
    大模型生成文本是一个 token 一个 token 地生成（自回归），每步都要
    跑一次完整前向传播，很慢。推测解码的思路：让一个小模型（草稿模型）
    先快速猜几个 token，然后让大模型一次性验证这些猜测。猜对的直接
    采用，猜错的从大模型的分布重新采样。就像考试时先快速写答案，
    再仔细检查——大部分答案是对的，只需修改少数错误。

数学原理：
    设草稿模型生成 K 个候选 token，大模型接受其中 n 个（n ≤ K）。

    验证过程：
        1. 将 prompt + K 个候选 token 拼接，大模型一次前向传播
        2. 大模型输出每个位置的 logits：p_target(x | x_{<t})
        3. 逐位置验证：
           - 接受概率：min(1, p_target(x_t) / p_draft(x_t))
           - 如果接受：继续验证下一个位置
           - 如果拒绝：从修正分布采样：
             p_corrected(x) ∝ max(0, p_target(x) - p_draft(x))
        4. 所有被接受的 token + 修正采样 token = 最终输出

    加速比分析：
        不用推测解码：K 个 token 需要 K 次大模型前向
        用推测解码：1 次草稿模型（K 步）+ 1 次大模型前向 = 总共约 2 步
        加速比 ≈ K × T_target / (K × T_draft + T_target)
        其中 T_draft << T_target（草稿模型快得多）

    无损保证：
        推测解码的输出分布与大模型自回归的输出分布完全相同
        （数学证明：接受-拒绝采样保持了目标分布）。

代码流程：
    1. draft_generate(): 草稿模型快速生成 K 个候选 token
    2. target_verify(): 大模型并行验证候选 token
    3. speedup_analysis(): 计算加速比
"""
```

- [ ] **Step 2: 补充 prefill_decode.py docstring**

`prefill_decode.py` 已有较好的 docstring，主要补充文件级 docstring。

替换文件级 docstring：

```python
"""Prefill vs Decode 阶段的并行策略切换分析

直觉理解：
    LLM 推理分两个阶段：Prefill（预填充）一次性处理整个 prompt，
    像一口气读完一本书；Decode（解码）逐个生成新 token，像逐字
    写续集。两个阶段的计算特征完全不同，最优并行策略也不同。

数学原理：
    Prefill 阶段：
        - 输入：S 个 token 的完整 prompt
        - 计算：O(S × d²) 的矩阵乘法（计算密集）
        - 特征：大矩阵乘法，GPU 利用率高
        - 最优策略：TP（张量并行），将大矩阵运算分布到多卡
        - 输出：KV Cache（供 Decode 阶段使用）

    Decode 阶段：
        - 输入：1 个新 token + 完整 KV Cache
        - 计算：O(S × d) 的注意力计算（访存密集）
        - 特征：小矩阵乘法 + 大量 KV Cache 读取，GPU 利用率低
        - 最优策略：DP（数据并行），将多个请求 batch 处理
        - 或 EP（专家并行），对 MoE 模型进行路由

    策略切换：
        长序列场景：Prefill 用 TP → 生成 KV Cache → Decode 切换为 DP
        短序列场景：全程 DP 即可，无需切换

    FLOPs 估算：
        单 token FLOPs ≈ 2 × d²（简化，忽略 attention）
        Prefill 总 FLOPs = S × 2d²
        Decode 单步 FLOPs = 2d²（与序列长度无关的部分）

代码流程：
    1. analyze_prefill_characteristics(): 分析 Prefill 阶段计算特征
    2. analyze_decode_characteristics(): 分析 Decode 阶段计算特征
    3. recommend_strategy(): 根据序列长度推荐并行策略
"""
```

- [ ] **Step 3: 补充 kv_cache_shard.py docstring**

`kv_cache_shard.py` 已有较好的 docstring，主要补充文件级 docstring。

替换文件级 docstring：

```python
"""KV Cache 分片 (KV Cache Sharding)

直觉理解：
    推理时 KV Cache 随序列增长而增大，单卡可能放不下。KV Cache 分片
    按 head 维度将 K 和 V 切分到多卡，每卡只存部分 head 的缓存。
    就像把一个大文件分成几个部分，分别存在不同硬盘上。

数学原理：
    KV Cache 形状：(B, n_heads, S, d_head)
    每个元素 float16 占 2 字节，K 和 V 各一份。

    总显存 = 2 × B × n_heads × S × d_head × 2 bytes
           = 4 × B × n_heads × S × d_head bytes

    分片后每卡显存 = 总显存 / P（P = GPU 数量）

    示例（LLaMA 3 70B）：
        B=32, n_heads=64, S=4096, d_head=128
        总 KV Cache = 4 × 32 × 64 × 4096 × 128 = 4.29 GB
        4 卡分片后每卡 = 1.07 GB

    分片策略：
        按 head 分片（当前实现）：每卡持有 n_heads/P 个 head 的 KV
        按 sequence 分片：每卡持有 S/P 个 token 的 KV（需要 All-Gather）
        按 batch 分片：每卡持有 B/P 个样本的 KV（最简单但灵活性差）

代码流程：
    1. shard_kv_cache_by_heads(): 按 head 维度切分 KV Cache
    2. gather_kv_cache(): 收集所有 rank 的 KV Cache（需要时）
    3. kv_cache_memory_analysis(): 显存占用分析
"""
```

- [ ] **Step 4: 验证修改后代码可正常运行**

Run: `python -c "from parallel.inference import speculative_decoding, prefill_decode, kv_cache_shard; print('All imports OK')"`

Expected: `All imports OK`

---

### Task 6: 深化 Ring Attention 实现 — online softmax

**Files:**
- Modify: `parallel/context_parallel/ring_attention.py`

- [ ] **Step 1: 实现 online softmax 版本的 ring_attention_step**

在 `ring_attention.py` 中添加新函数 `ring_attention_online_softmax`：

```python
def ring_attention_online_softmax(
    q: torch.Tensor,
    k_blocks: list[torch.Tensor],
    v_blocks: list[torch.Tensor],
) -> torch.Tensor:
    """
    使用 online softmax 的环形注意力完整实现。

    直觉：逐块接收 KV，像滚雪球一样逐步累积注意力结果。
    每收到一块新的 KV，就更新 running max 和 running sum，
    并用修正因子调整已有结果。

    数学（online softmax 推导）：
        初始化：O = 0, m = -inf, l = 0

        对每个 KV block t：
            1. 计算局部分数：S_t = Q @ K_t^T / √d
            2. 更新 running max：m_new = max(m_old, max(S_t, dim=-1, keepdim=True))
            3. 计算修正因子：
               α = exp(m_old - m_new)  # 修正已有 O 和 l
               β = exp(S_t - m_new)     # 新块的 softmax 分子
            4. 更新输出：O = O * α * (l_old / l_new) + (β / l_new) @ V_t
               简化：O = O * α + β @ V_t  (l 的修正可以合并)
            5. 更新 running sum：l = l * α + sum(β, dim=-1, keepdim=True)
            6. 更新 running max：m = m_new

        最终：O = O / l  (归一化)

    Args:
        q: 查询张量，形状 (B, n_heads, S_local, d_head)
        k_blocks: KV block 列表中的 K 块，共 P 个，每个形状 (B, n_heads, S_local, d_head)
        v_blocks: KV block 列表中的 V 块，共 P 个，每个形状 (B, n_heads, S_local, d_head)

    Returns:
        完整注意力输出，形状 (B, n_heads, S_local, d_head)
    """
    B, n_heads, S_local, d_head = q.shape
    scale = d_head ** 0.5

    # 初始化 running 统计量
    O = torch.zeros_like(q)                              # (B, n_heads, S_local, d_head)
    m = torch.full((B, n_heads, S_local, 1), float('-inf'), device=q.device, dtype=q.dtype)  # running max
    l = torch.zeros((B, n_heads, S_local, 1), device=q.device, dtype=q.dtype)  # running sum

    for k_block, v_block in zip(k_blocks, v_blocks):
        # 1. 计算局部注意力分数
        scores = (q @ k_block.transpose(-2, -1)) / scale  # (B, n_heads, S_local, S_local)

        # 2. 更新 running max
        m_block = scores.max(dim=-1, keepdim=True).values  # (B, n_heads, S_local, 1)
        m_new = torch.maximum(m, m_block)

        # 3. 计算修正因子
        alpha = torch.exp(m - m_new)    # 修正已有结果
        beta = torch.exp(scores - m_new)  # 新块的 softmax 分子

        # 4. 更新输出
        O = O * alpha + beta @ v_block

        # 5. 更新 running sum
        l = l * alpha + beta.sum(dim=-1, keepdim=True)

        # 6. 更新 running max
        m = m_new

    # 最终归一化
    O = O / l
    return O
```

- [ ] **Step 2: 实现 rotate_kv 的实际轮转逻辑**

替换 `rotate_kv` 函数：

```python
def rotate_kv(
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]], direction: int = 1
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    在环形拓扑中传递 KV block。

    直觉：KV 像接力棒一样，每步传给下一个 rank。
    在单机模拟中，通过列表旋转来模拟跨 rank 传递。

    数学：
        direction=1:  KV 从 rank i 移动到 rank (i+1) % P
        direction=-1: KV 从 rank i 移动到 rank (i-1) % P

    实现：
        列表向右旋转 direction 位（模拟 KV 沿环传递）
        每步调用后，rank i 看到的 KV 原本属于 rank (i-direction) % P

    Args:
        kv_cache: KV block 列表，每个元素为 (K, V) 元组
        direction: 传递方向，1=向前（右旋转），-1=向后（左旋转）

    Returns:
        旋转后的 KV block 列表
    """
    if not kv_cache:
        return kv_cache
    n = len(kv_cache)
    shift = direction % n
    return kv_cache[-shift:] + kv_cache[:-shift]
```

- [ ] **Step 3: 更新 __main__ 演示代码**

替换 `__main__` 块，添加 online softmax 演示：

```python
if __name__ == "__main__":
    print("=== ring_attention demo ===")

    # 模拟 2 个头，4 个 rank，每卡 seq_len=2，d_head=4
    B = 1
    n_heads = 2
    total_seq = 8
    n_ranks = 4
    seq_local = total_seq // n_ranks
    d_head = 4

    # 生成完整的 Q, K, V
    torch.manual_seed(42)
    q_full = torch.randn(B, n_heads, total_seq, d_head)
    k_full = torch.randn(B, n_heads, total_seq, d_head)
    v_full = torch.randn(B, n_heads, total_seq, d_head)

    # 切分为各 rank 的 KV blocks
    k_blocks = [k_full[:, :, i*seq_local:(i+1)*seq_local, :] for i in range(n_ranks)]
    v_blocks = [v_full[:, :, i*seq_local:(i+1)*seq_local, :] for i in range(n_ranks)]

    # Rank 0 的本地 Q
    q_local = q_full[:, :, :seq_local, :]

    # 使用 online softmax 的完整实现
    print(f"Online softmax Ring Attention (rank 0, {n_ranks} ranks):")
    output_online = ring_attention_online_softmax(q_local, k_blocks, v_blocks)
    print(f"  Output shape: {output_online.shape}")

    # 与标准注意力对比验证
    scale = d_head ** 0.5
    scores = (q_local @ k_full.transpose(-2, -1)) / scale
    attn = torch.softmax(scores, dim=-1)
    output_standard = attn @ v_full
    print(f"  Standard attention output shape: {output_standard.shape}")

    max_diff = (output_online - output_standard).abs().max().item()
    print(f"  Max difference (online vs standard): {max_diff:.6f}")
    assert max_diff < 1e-4, f"Online softmax result differs from standard: {max_diff}"
    print(f"  Verification: PASSED (online softmax matches standard attention)")

    # KV 旋转演示
    print(f"\nKV rotation demo:")
    kv_cache = [(k_blocks[i], v_blocks[i]) for i in range(n_ranks)]
    print(f"  Initial owner: rank 0 has KV block 0")
    for step in range(n_ranks):
        kv_cache = rotate_kv(kv_cache, direction=1)
        print(f"  After step {step+1}: rank 0 now has KV block from original rank {(1+step) % n_ranks}")
```

- [ ] **Step 4: 验证 online softmax 实现正确性**

Run: `python parallel/context_parallel/ring_attention.py`

Expected: 输出显示 `Verification: PASSED` 且 max difference < 1e-4

---

### Task 7: 深化 1F1B 调度实现 — 时间线模拟

**Files:**
- Modify: `parallel/pipeline_parallel/f1b1.py`

- [ ] **Step 1: 实现完整 1F1B 调度模拟**

在 `f1b1.py` 中添加新函数 `simulate_1f1b_timeline`：

```python
def simulate_1f1b_timeline(
    n_micro_batches: int, n_stages: int, forward_time: float = 1.0, backward_time: float = 2.0
) -> dict:
    """
    模拟 1F1B 调度的完整时间线。

    直觉：把 1F1B 调度的三个阶段（warmup、steady、cooldown）的
    每个操作的时间戳精确计算出来，用于可视化和分析。

    数学：
        Warmup 阶段（stage s 做 n_warmup - s 个 forward）：
            n_warmup = n_stages - 1
            Stage s 的 warmup forward 数 = n_warmup - s

        Steady 阶段（1F1B 交替）：
            每个 stage 做 M - n_warmup + s 轮 1F+1B

        Cooldown 阶段（处理剩余 backward）：
            Stage s 的 cooldown backward 数 = s + 1

    Args:
        n_micro_batches: micro-batch 数量 M
        n_stages: 流水线 stage 数量 P
        forward_time: 单次 forward 耗时
        backward_time: 单次 backward 耗时

    Returns:
        包含 timeline（每个 stage 的操作列表）、total_time、bubble_ratio 的字典
    """
    n_warmup = n_stages - 1
    timeline = {s: [] for s in range(n_stages)}

    for stage in range(n_stages):
        current_time = stage * forward_time  # 每个 stage 的启动延迟
        warmup_count = n_warmup - stage
        steady_count = n_micro_batches - warmup_count

        # Warmup: 连续 forward
        for i in range(warmup_count):
            mb_idx = i
            timeline[stage].append({
                'type': 'forward',
                'micro_batch': mb_idx,
                'start': current_time,
                'end': current_time + forward_time,
            })
            current_time += forward_time

        # Steady: 1F1B 交替
        for i in range(steady_count):
            mb_fwd = warmup_count + i
            mb_bwd = i
            timeline[stage].append({
                'type': 'forward',
                'micro_batch': mb_fwd,
                'start': current_time,
                'end': current_time + forward_time,
            })
            current_time += forward_time
            timeline[stage].append({
                'type': 'backward',
                'micro_batch': mb_bwd,
                'start': current_time,
                'end': current_time + backward_time,
            })
            current_time += backward_time

        # Cooldown: 剩余 backward
        for i in range(stage + 1):
            mb_bwd = steady_count + i
            if mb_bwd < n_micro_batches:
                timeline[stage].append({
                    'type': 'backward',
                    'micro_batch': mb_bwd,
                    'start': current_time,
                    'end': current_time + backward_time,
                })
                current_time += backward_time

    # 计算总时间和 bubble
    total_time = max(
        max(op['end'] for op in ops) for ops in timeline.values()
    )
    ideal_time = n_micro_batches * (forward_time + backward_time)
    bubble_ratio = 1 - ideal_time / (total_time * n_stages)

    return {
        'timeline': timeline,
        'total_time': total_time,
        'ideal_time': ideal_time,
        'bubble_ratio': bubble_ratio,
    }
```

- [ ] **Step 2: 更新 __main__ 演示代码**

在 `f1b1.py` 的 `__main__` 块中添加时间线模拟演示：

```python
    # 时间线模拟
    print("\n1F1B Timeline Simulation (4 stages, 8 micro-batches):")
    result = simulate_1f1b_timeline(n_micro_batches=8, n_stages=4, forward_time=1.0, backward_time=2.0)
    for stage in range(4):
        ops = result['timeline'][stage]
        print(f"  Stage {stage}: {len(ops)} ops, total time = {max(op['end'] for op in ops):.1f}")
    print(f"  Total time: {result['total_time']:.1f}")
    print(f"  Bubble ratio: {result['bubble_ratio']:.4f}")
```

- [ ] **Step 3: 验证时间线模拟**

Run: `python parallel/pipeline_parallel/f1b1.py`

Expected: 输出时间线信息，无报错

---

### Task 8: 深化 Speculative Decoding — 验证逻辑

**Files:**
- Modify: `parallel/inference/speculative_decoding.py`

- [ ] **Step 1: 实现完整的 target_verify 验证逻辑**

替换 `target_verify` 函数：

```python
def target_verify(
    target_model,
    prompt: torch.LongTensor,
    candidates: torch.LongTensor,
    draft_model=None,
    temperature: float = 1.0,
) -> tuple[torch.LongTensor, int]:
    """
    用大模型并行验证候选 token 序列（完整实现）。

    直觉：大模型一次性看完所有候选 token，逐个检查"我会不会也生成这个 token"。
    猜对的保留，猜错的从大模型的分布重新采样。

    数学（接受-拒绝采样）：
        对每个候选位置 t：
            1. 计算接受概率：p_accept = min(1, p_target(x_t) / p_draft(x_t))
            2. 以概率 p_accept 决定是否接受
            3. 如果接受：继续验证下一个位置
            4. 如果拒绝：
               a. 从修正分布采样：p_corrected(x) ∝ max(0, p_target(x) - p_draft(x))
               b. 返回已接受的 token + 修正采样的 token

        无损保证：最终输出分布与目标模型自回归分布完全相同。

    Args:
        target_model: 大参数量的目标模型
        prompt: 输入 prompt token 序列，形状 (batch, seq_len)
        candidates: 草稿模型生成的候选 token，形状 (batch, n_candidates)
        draft_model: 草稿模型（用于计算接受概率），如果为 None 则接受所有候选
        temperature: 采样温度

    Returns:
        (accepted_tokens, n_accepted):
            accepted_tokens: 被接受的 token 序列 + 修正采样 token，形状 (batch, variable)
            n_accepted: 被接受的候选 token 数量（不含修正采样 token）
    """
    target_model.eval()
    batch_size = prompt.shape[0]
    n_candidates = candidates.shape[1]

    # 拼接 prompt 和候选序列，一次前向传播
    full_input = torch.cat([prompt, candidates], dim=1)
    with torch.no_grad():
        logits = target_model(full_input)

    # 如果没有草稿模型，简化为接受所有候选
    if draft_model is None:
        return candidates, n_candidates

    # 计算草稿模型的概率
    draft_input = torch.cat([prompt, candidates[:, :-1]], dim=1)
    with torch.no_grad():
        draft_logits = draft_model(draft_input)

    # 逐位置验证
    accepted_list = []
    n_accepted = 0

    for t in range(n_candidates):
        # 目标模型在位置 prompt_len + t - 1 的预测（预测第 t 个候选）
        target_probs = torch.softmax(logits[:, prompt.shape[1] + t - 1] / temperature, dim=-1)
        draft_probs = torch.softmax(draft_logits[:, prompt.shape[1] + t - 1] / temperature, dim=-1)

        # 候选 token
        candidate_token = candidates[:, t]  # (batch,)

        # 接受概率
        p_target = target_probs.gather(1, candidate_token.unsqueeze(1)).squeeze(1)
        p_draft = draft_probs.gather(1, candidate_token.unsqueeze(1)).squeeze(1)
        p_accept = torch.min(torch.ones_like(p_target), p_target / (p_draft + 1e-10))

        # 采样决定是否接受
        accept = torch.rand_like(p_accept) < p_accept

        if accept.all():
            accepted_list.append(candidate_token)
            n_accepted += 1
        else:
            # 至少一个 batch 元素拒绝，记录已接受的 token 并从修正分布采样
            accepted_list.append(candidate_token * accept.long())

            # 修正分布采样：p_corrected ∝ max(0, p_target - p_draft)
            corrected_probs = torch.clamp(target_probs - draft_probs, min=0)
            corrected_probs = corrected_probs / (corrected_probs.sum(dim=-1, keepdim=True) + 1e-10)
            corrected_token = torch.multinomial(corrected_probs, 1).squeeze(1)
            # 在拒绝的位置使用修正采样的 token
            for b in range(batch_size):
                if not accept[b]:
                    accepted_list[-1][b] = corrected_token[b]
            n_accepted += accept.sum().item()
            break

    # 如果所有候选都被接受，额外从目标模型采样一个 token
    if len(accepted_list) == n_candidates:
        target_probs_last = torch.softmax(logits[:, -1] / temperature, dim=-1)
        extra_token = torch.multinomial(target_probs_last, 1).squeeze(1)
        accepted_list.append(extra_token)

    accepted_tokens = torch.stack(accepted_list, dim=1)
    return accepted_tokens, n_accepted
```

- [ ] **Step 2: 验证修改后代码可正常运行**

Run: `python -c "from parallel.inference.speculative_decoding import target_verify, draft_generate, speedup_analysis; print('Import OK')"`

Expected: `Import OK`

---

### Task 9: 运行完整测试套件验证代码层修改

- [ ] **Step 1: 运行现有测试**

Run: `python -m pytest tests/ -v`

Expected: 所有测试通过

- [ ] **Step 2: 验证所有模块可正常导入**

Run: `python -c "from parallel.tensor_parallel import column_parallel, row_parallel, megatron_style, sequence_parallel, embedding_parallel; from parallel.expert_parallel import expert_partition, token_dispatch; from parallel.pipeline_parallel import gpiped, f1b1, layer_partition; from parallel.context_parallel import ring_attention, sequence_partition, cp_integration; from parallel.inference import speculative_decoding, prefill_decode, kv_cache_shard; print('All parallel modules import OK')"`

Expected: `All parallel modules import OK`

---

## Phase 2: Notebook 层增强

### Task 10: 增强 01_attention_basics.ipynb

**Files:**
- Modify: `notebooks/01_attention_basics.ipynb`

- [ ] **Step 1: 为每个主题模块添加"直觉理解"和"数学原理" Markdown 单元格**

在每个代码段前插入 Markdown 单元格，包含：
- 直觉理解（1-2 段话 + 类比）
- 数学原理（公式推导，使用 LaTeX 语法）

- [ ] **Step 2: 添加 matplotlib 可视化**

添加代码单元格，使用 matplotlib 绘制：
- 注意力权重热力图（彩色，替代 ASCII 图）
- KV Cache 内存对比柱状图

- [ ] **Step 3: 添加练习题**

在 notebook 末尾添加 Markdown 单元格，包含 2-3 道练习题：
- 思考题："为什么缩放因子是 √d 而不是 d？"
- 编程题："修改 GQA 的组数，观察参数量变化"
- 分析题："计算 MHA/GQA/MQA 在不同头数下的 KV Cache 大小"

---

### Task 11-18: 增强其余 9 个 Notebook

**Files:**
- Modify: `notebooks/02_transformer_walkthrough.ipynb`
- Modify: `notebooks/03_llama3_walkthrough.ipynb`
- Modify: `notebooks/04_deepseek_v3_walkthrough.ipynb`
- Modify: `notebooks/05_communication_primitives.ipynb`
- Modify: `notebooks/06_data_parallel.ipynb`
- Modify: `notebooks/07_tensor_parallel.ipynb`
- Modify: `notebooks/08_pipeline_parallel.ipynb`
- Modify: `notebooks/09_expert_and_context_parallel.ipynb`
- Modify: `notebooks/10_inference_parallel.ipynb`

每个 notebook 执行相同的增强步骤：

- [ ] **为每个主题模块添加"直觉理解"和"数学原理" Markdown 单元格**
- [ ] **添加 matplotlib 可视化**（具体图表见设计文档 2.2 节）
- [ ] **添加练习题**（2-3 道，含思考题/编程题/分析题）

---

### Task 19: 创建 11_end_to_end_training.ipynb

**Files:**
- Create: `notebooks/11_end_to_end_training.ipynb`

- [ ] **Step 1: 创建端到端训练 notebook**

内容结构：
1. 概述：端到端训练流程简介
2. 数据准备：使用简单 tokenizer + 随机数据演示
3. 模型构建：使用仓库中的 LLaMA3 组件
4. 训练循环：loss 计算、优化器、学习率调度
5. 生成推理：使用训练后的模型生成文本
6. 练习题

---

### Task 20: 创建 12_parallel_strategy_guide.ipynb

**Files:**
- Create: `notebooks/12_parallel_strategy_guide.ipynb`

- [ ] **Step 1: 创建并行策略选择指南 notebook**

内容结构：
1. 概述：为什么需要选择并行策略
2. 六大并行策略的适用场景对比
3. 并行策略选择决策树（可视化）
4. 不同规模下的推荐组合
5. 成本-收益分析
6. 实际案例分析
7. 练习题

---

### Task 21: 创建 13_debugging_distributed.ipynb

**Files:**
- Create: `notebooks/13_debugging_distributed.ipynb`

- [ ] **Step 1: 创建分布式训练调试 notebook**

内容结构：
1. 概述：分布式训练调试的挑战
2. 常见错误类型及排查流程
3. NCCL 错误排查
4. 数值问题排查
5. 显存问题排查
6. 调试工具使用
7. 练习题

---

## Phase 3: 文档层增强

### Task 22: 创建综合指南文档

**Files:**
- Create: `docs/guide/getting_started.md`
- Create: `docs/guide/parallel_strategy_guide.md`
- Create: `docs/guide/debugging_guide.md`
- Create: `docs/guide/hardware_basics.md`

- [ ] **Step 1: 创建 getting_started.md**

按文档模板撰写：概述 → 直觉理解 → 学习路径 → 环境配置 → 使用指南 → 参考资料

- [ ] **Step 2: 创建 parallel_strategy_guide.md**

按文档模板撰写：概述 → 直觉理解 → 决策树 → 组合策略 → 案例分析 → 参考资料

- [ ] **Step 3: 创建 debugging_guide.md**

按文档模板撰写：概述 → 常见错误分类 → NCCL 排查 → 数值问题 → 显存问题 → 工具清单 → 参考资料

- [ ] **Step 4: 创建 hardware_basics.md**

按文档模板撰写：概述 → GPU 架构 → 显存层次 → NVLink vs PCIe → 多卡拓扑 → GPU 型号对比 → 参考资料

---

### Task 23: 创建模型架构讲解文档

**Files:**
- Create: `docs/models/01_attention_mechanism.md`
- Create: `docs/models/02_transformer_architecture.md`
- Create: `docs/models/03_llama3_architecture.md`
- Create: `docs/models/04_deepseek_v3_architecture.md`
- Create: `docs/models/05_positional_encoding.md`
- Create: `docs/models/06_normalization.md`
- Create: `docs/models/07_activation_functions.md`

每个文档按统一模板撰写：概述 → 直觉理解 → 数学原理 → 算法流程 → 代码实现 → 实践考量 → 与其他技术的关系 → 参考资料。

- [ ] **Step 1: 创建 01_attention_mechanism.md**
- [ ] **Step 2: 创建 02_transformer_architecture.md**
- [ ] **Step 3: 创建 03_llama3_architecture.md**
- [ ] **Step 4: 创建 04_deepseek_v3_architecture.md**
- [ ] **Step 5: 创建 05_positional_encoding.md**
- [ ] **Step 6: 创建 06_normalization.md**
- [ ] **Step 7: 创建 07_activation_functions.md**

---

### Task 24: 创建分布式并行讲解文档

**Files:**
- Create: `docs/parallel/01_communication_primitives.md`
- Create: `docs/parallel/02_data_parallel.md`
- Create: `docs/parallel/03_tensor_parallel.md`
- Create: `docs/parallel/04_pipeline_parallel.md`
- Create: `docs/parallel/05_expert_parallel.md`
- Create: `docs/parallel/06_context_parallel.md`
- Create: `docs/parallel/07_inference_optimization.md`

每个文档按统一模板撰写：概述 → 直觉理解 → 数学原理 → 算法流程 → 代码实现 → 实践考量 → 与其他技术的关系 → 参考资料。

- [ ] **Step 1: 创建 01_communication_primitives.md**
- [ ] **Step 2: 创建 02_data_parallel.md**
- [ ] **Step 3: 创建 03_tensor_parallel.md**
- [ ] **Step 4: 创建 04_pipeline_parallel.md**
- [ ] **Step 5: 创建 05_expert_parallel.md**
- [ ] **Step 6: 创建 06_context_parallel.md**
- [ ] **Step 7: 创建 07_inference_optimization.md**

---

### Task 25: 更新 README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 更新 README 添加文档链接**

在 README 的"项目结构"部分添加 `docs/` 目录的详细说明，并在"学习路线"部分添加对应文档的链接。

---

## 自我审查清单

- [x] **Spec 覆盖率**：设计文档中的每一项要求都有对应的 Task
  - 代码层 docstring 补充：Task 1-5
  - 代码层实现深化：Task 6-8
  - Notebook 层增强：Task 10-18
  - Notebook 层新增：Task 19-21
  - 文档层综合指南：Task 22
  - 文档层模型讲解：Task 23
  - 文档层并行讲解：Task 24
  - README 更新：Task 25
- [x] **Placeholder 扫描**：无 TBD/TODO，所有步骤包含具体代码或内容描述
- [x] **类型一致性**：函数签名和类型在所有 Task 中保持一致

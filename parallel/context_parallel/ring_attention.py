"""环形注意力 (Ring Attention)。序列沿长度切分，KV block 在环形拓扑中轮转传递。

直觉
----
超长序列被切成多段，每段放在一张 GPU 上。KV 像接力棒一样在 GPU 之间
环形传递——每传一次，每张 GPU 就能多"看到"一段序列的 Key/Value，
最终绕行一圈后，每张 GPU 都计算出了完整的注意力结果。

数学
----
1. Ring Attention 分块计算：
   将序列 S 切成 P 段，每段长度 S/P。第 i 张 GPU 持有 Q_i，
   在第 t 步持有 KV_{(i+t) % P}。
   每步计算 partial attention：attn_i^{(t)} = softmax(Q_i @ K_{(i+t)%P}^T) @ V_{(i+t)%P}

2. Online Softmax 增量计算（完整版）：
   由于 softmax 需要全局归一化，不能简单将各步结果相加。
   需要维护 running max m 和 running sum l：
     m^{(t)} = max(m^{(t-1)}, max(scores^{(t)}))
     l^{(t)} = e^{m^{(t-1)}-m^{(t)}} × l^{(t-1)} + sum(softmax(scores^{(t)}, m^{(t)}))
     output = (l^{(t-1)} × e^{m^{(t-1)}-m^{(t)}} × output^{(t-1)} + softmax(scores^{(t)}, m^{(t)}) @ V) / l^{(t)}

3. 通信量：每步传递 KV block，大小 = 2 × B × n_heads × (S/P) × d_head，
   共 P-1 步，总通信量 = O(S) 级别（与序列长度线性相关，而非二次）。

注意：当前简化实现未实现 online softmax，直接对局部 scores 做 softmax，
仅用于演示 Ring Attention 的通信模式。

代码流程
--------
1. ``ring_attention_step`` —— 单步：用本地 Q 和当前 KV 计算 partial attention
2. ``rotate_kv`` —— 将 KV block 在环形拓扑中传递一步
"""
import torch


def ring_attention_step(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, step: int
) -> torch.Tensor:
    """
    环形注意力的单步：用本地 Q 和当前持有的 KV 计算 partial attention。

    直觉：每张 GPU 用自己的 Q 和当前手上的 KV 算一次注意力，
    然后 KV 传给下一张 GPU，下一步再用新的 KV 算一次。

    数学：
        简化版（当前实现）：
            scores = Q @ K^T / √d
            attn = softmax(scores)  # 局部 softmax，非全局归一化
            output = attn @ V

        完整版（online softmax）：
            需维护 running max m 和 running sum l，每步修正之前的输出：
            m^{(t)} = max(m^{(t-1)}, max(scores^{(t)}))
            l^{(t)} = e^{m^{(t-1)}-m^{(t)}} × l^{(t-1)} + Σ(softmax(scores^{(t)}, m^{(t)}))
            output^{(t)} = (l^{(t-1)} × e^{m^{(t-1)}-m^{(t)}} × output^{(t-1)}
                           + softmax(scores^{(t)}, m^{(t)}) @ V) / l^{(t)}

    Args:
        q: Query 张量，形状 (B, n_heads, S_local, d_head)
        k: Key 张量，形状 (B, n_heads, S_local, d_head)
        v: Value 张量，形状 (B, n_heads, S_local, d_head)
        step: 当前环形传递步数

    Returns:
        torch.Tensor: 当前步的 attention 输出，形状 (B, n_heads, S_local, d_head)
    """
    scale = q.shape[-1] ** 0.5
    scores = (q @ k.transpose(-2, -1)) / scale
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


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
            4. 更新输出：O = O * α + β @ V_t
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
    O = torch.zeros_like(q)
    m = torch.full((B, n_heads, S_local, 1), float('-inf'), device=q.device, dtype=q.dtype)
    l = torch.zeros((B, n_heads, S_local, 1), device=q.device, dtype=q.dtype)

    for k_block, v_block in zip(k_blocks, v_blocks):
        # 1. 计算局部注意力分数
        scores = (q @ k_block.transpose(-2, -1)) / scale

        # 2. 更新 running max
        m_block = scores.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m, m_block)

        # 3. 计算修正因子
        alpha = torch.exp(m - m_new)
        beta = torch.exp(scores - m_new)

        # 4. 更新输出
        O = O * alpha + beta @ v_block

        # 5. 更新 running sum
        l = l * alpha + beta.sum(dim=-1, keepdim=True)

        # 6. 更新 running max
        m = m_new

    # 最终归一化
    O = O / l
    return O


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

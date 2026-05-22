"""Embedding 并行 —— 大词表 Embedding 矩阵按 vocab 维度切分，每卡存部分词向量，all-reduce 求和得到完整 Embedding。

直觉理解
--------
想象一本超大字典被拆成 P 本小字典分给 P 个人，每个人只查自己那本里的词。
如果查到的词在别人的字典里，就返回零向量；最后所有人把结果加起来（all-reduce），
就能得到正确的词向量。

数学原理
--------
设全局 Embedding 矩阵 E ∈ ℝ^(V×D)，world_size = P，当前 rank 为 i：

1. 权重切分：Eᵢ = E[i·(V/P):(i+1)·(V/P), :]  ∈ ℝ^(V/P × D)
2. 本地查表：对于 token id t，若 t ∈ [i·V/P, (i+1)·V/P)，则 embᵢ(t) = Eᵢ[t - i·V/P]，
   否则 embᵢ(t) = 0  ∈ ℝ^D
3. 通信聚合：emb(t) = AllReduce(Σᵢ embᵢ(t)) = E[t]  ∈ ℝ^D

显存节省：每卡只存 V/P 行词向量，节省 (P-1)/P 的 Embedding 显存。
通信量：每次前向传播 all-reduce 传输 B×S×D 个元素。

与 TP 的关系：Embedding 并行是张量并行在 Embedding 层的具体应用，
权重沿 vocab 维度切分，与列并行/行并行构成完整的 TP 方案。

代码流程
--------
1. embedding_parallel_forward: 本地查表 + all-reduce 求和
"""
import torch
import torch.distributed as dist
from parallel.communication.setup import get_rank, get_world_size


def embedding_parallel_forward(
    embed_weight: torch.Tensor, token_ids: torch.LongTensor
) -> torch.Tensor:
    """Embedding 并行前向：本地查表 + all-reduce 求和得到完整 Embedding。

    直觉
    ----
    每个人只查自己那本小字典，查不到就返回零向量，最后所有人把结果加起来。

    数学
    ----
    对于 token id t：
      embᵢ(t) = Eᵢ[t - i·V/P]  若 t ∈ [i·V/P, (i+1)·V/P)
               = 0               否则
      emb(t) = AllReduce(Σᵢ embᵢ(t)) = E[t]    ∈ ℝ^D

    输出形状：(B, S, D)

    Args:
        embed_weight: 本地 Embedding 权重，形状 (V/P, D)，只包含当前 rank 负责的词向量。
        token_ids: Token ID 张量，形状 (B, S)，使用全局词表索引。

    Returns:
        Embedding 输出，形状 (B, S, D)，all-reduce 后所有 rank 上相同。

    Note:
        当前为简化实现：假设 token_ids 已在本地词表范围内，直接查表后 all-reduce。
        完整实现需要：(1) 将全局 token_ids 映射到本地范围；(2) 超出本地范围的 token
        返回零向量；(3) all-reduce 求和。当前实现仅适用于单卡或 token_ids 已预处理
        到本地范围的场景。
    """
    # 简化：直接对本地 embedding 做 all-reduce（适用于 token ids 已在本地范围的场景）
    local_out = torch.nn.functional.embedding(token_ids, embed_weight)
    dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
    return local_out


if __name__ == "__main__":
    from parallel.communication.setup import init_process_group, cleanup

    print("=" * 60)
    print("Embedding 并行演示")
    print("=" * 60)

    try:
        if not dist.is_initialized():
            init_process_group(backend="gloo")
            print("已初始化 gloo 后端（单进程演示模式）")
    except Exception:
        print("无法初始化分布式环境，跳过通信演示")

    rank = get_rank()
    ws = get_world_size()

    print(f"当前 rank: {rank}, world_size: {ws}")

    B, S, dim = 2, 4, 16
    vocab_size_local = 128
    embed_w = torch.randn(vocab_size_local, dim)
    token_ids = torch.randint(0, vocab_size_local, (B, S))

    if dist.is_initialized():
        out = embedding_parallel_forward(embed_w, token_ids)
        expected = torch.nn.functional.embedding(token_ids, embed_w)
        assert out.shape == (B, S, dim), f"Wrong output shape: {out.shape}"
        if ws == 1:
            assert torch.allclose(out, expected, atol=1e-5), "Output mismatch"
        print(f"输出形状: {list(out.shape)} — OK")
    else:
        print("(跳过通信：分布式环境未初始化)")

    print("\nEmbedding 并行将词表沿 vocab 维度切分到各 rank。")
    print("每个 rank 只存储部分词向量，查表后 all-reduce 得到完整结果。")
    print("这可以显著减少大词表模型（如多语言模型）的显存占用。")

    if dist.is_initialized():
        cleanup()

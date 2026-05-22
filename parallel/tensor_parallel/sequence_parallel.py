"""序列并行 (Sequence Parallel, SP) —— 在 TP 区域外沿序列维度切分激活值，减少 LayerNorm/Dropout 的显存占用。

直觉理解
--------
LayerNorm 和 Dropout 只需要对单个 token 操作，不需要跨 token 的信息。
既然如此，为什么每张卡都要存完整的序列？把序列沿长度方向切成 P 段，
每卡只存 S/P 个 token 的激活值，显存立省 (P-1)/P。

数学原理
--------
设输入 X ∈ ℝ^(B×S×D)，world_size = P：

1. 每卡持有 X_local = X[:, rank·(S/P):(rank+1)·(S/P), :]  ∈ ℝ^(B × S/P × D)
2. 激活显存节省：(P-1)/P（每卡只存 1/P 的序列长度）
3. SP ↔ 非 SP 区域转换：
   - 离开 SP 区域（进入 TP 区域）：All-Gather 恢复完整序列  → 通信量 B×S×D
   - 进入 SP 区域（离开 TP 区域）：Reduce-Scatter 切分序列  → 通信量 B×S×D

与 TP 的关系：SP 不是独立的并行策略，而是 TP 的优化。在标准 TP 中，
LayerNorm/Dropout 的激活值在所有 rank 上冗余存储；SP 通过沿序列维度切分
消除这种冗余，代价是每次进出 SP 区域需要一次通信（All-Gather / Reduce-Scatter）。

代码流程
--------
1. scatter_along_seq: 将完整序列沿 seq 维度切分到各 rank
2. gather_along_seq: 从各 rank 收集序列块，拼回完整序列
3. sp_transition_fwd: SP 与非 SP 区域之间的转换入口
"""
import torch
import torch.distributed as dist
from parallel.communication.setup import get_rank, get_world_size


def scatter_along_seq(x: torch.Tensor) -> torch.Tensor:
    """将张量沿序列维度切分，每个 rank 保留本地 chunk。

    直觉
    ----
    把一条长队伍按人数均分成 P 段，每段分给一个组长管理。

    数学
    ----
    X_local = X[:, rank·(S/P):(rank+1)·(S/P), :]    ∈ ℝ^(B × S/P × D)
    其中 S/P = x.shape[1] // world_size

    Args:
        x: 完整输入张量，形状 (B, S, D)。

    Returns:
        本地序列块，形状 (B, S/P, D)。
    """
    rank = get_rank()
    world_size = get_world_size()
    seq_len = x.shape[1]
    chunk_size = seq_len // world_size
    return x[:, rank * chunk_size : (rank + 1) * chunk_size].contiguous()


def gather_along_seq(x: torch.Tensor, total_seq_len: int) -> torch.Tensor:
    """从所有 rank 收集序列块，all-gather 拼回完整序列。

    直觉
    ----
    各组长把自己管理的队伍段汇报上来，拼接成完整的长队伍。

    数学
    ----
    X_full = AllGather([X₀, X₁, ..., Xₚ₋₁])    ∈ ℝ^(B × S × D)
    通信量：B × S × D 个元素

    Args:
        x: 本地序列块，形状 (B, S/P, D)。
        total_seq_len: 完整序列长度 S。

    Returns:
        完整序列张量，形状 (B, S, D)。
    """
    world_size = get_world_size()
    full = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(full, x)
    return torch.cat(full, dim=1)[:, :total_seq_len].contiguous()


def sp_transition_fwd(x: torch.Tensor, from_sp: bool = True) -> torch.Tensor:
    """SP 与非 SP 区域之间的转换。

    直觉
    ----
    from_sp=True 像是把分散的拼图拼回全景图（All-Gather），
    from_sp=False 像是把全景图剪成 P 份分给各人（Scatter）。

    数学
    ----
    - from_sp=True（离开 SP 区域）：
      X_full = AllGather(X_local)    ∈ ℝ^(B × S × D)
      通信量：B × S × D
    - from_sp=False（进入 SP 区域）：
      X_local = X[:, rank·(S/P):(rank+1)·(S/P), :]    ∈ ℝ^(B × S/P × D)
      无通信（本地切分）

    注意：在完整 Megatron 实现中，from_sp=False 应使用 Reduce-Scatter
    （通信量 B×S×D），而非简单的本地 Scatter。当前简化实现使用 Scatter。

    Args:
        x: 输入张量。from_sp=True 时形状 (B, S/P, D)；from_sp=False 时形状 (B, S, D)。
        from_sp: True 表示从 SP 区域转出（All-Gather），False 表示转入 SP 区域（Scatter）。

    Returns:
        from_sp=True: 完整序列张量，形状 (B, S, D)。
        from_sp=False: 本地序列块，形状 (B, S/P, D)。
    """
    if from_sp:
        return gather_along_seq(x, get_world_size() * x.shape[1])
    else:
        return scatter_along_seq(x)


if __name__ == "__main__":
    from parallel.communication.setup import init_process_group, cleanup

    print("=" * 60)
    print("序列并行 (Sequence Parallel) 演示")
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

    B, S, D = 2, 8, 16
    x = torch.randn(B, S, D)
    seq_chunk = S // ws

    # Test scatter
    x_scattered = scatter_along_seq(x)
    assert x_scattered.shape == (B, seq_chunk, D), f"Wrong scattered shape: {x_scattered.shape}"
    print(f"完整序列形状: {list(x.shape)} -> 本地序列块: {list(x_scattered.shape)}")

    if dist.is_initialized():
        # Test scatter + gather round-trip
        x_restored = gather_along_seq(x_scattered, S)
        assert x_restored.shape == (B, S, D), f"Wrong restored shape: {x_restored.shape}"
        if ws == 1:
            assert torch.allclose(x_restored, x, atol=1e-5), "Round-trip mismatch"
        print(f"all-gather 恢复后形状: {list(x_restored.shape)} — OK")

        # Test sp_transition_fwd
        x_from_sp = sp_transition_fwd(x_scattered, from_sp=True)
        assert x_from_sp.shape == (B, S, D), f"Wrong from_sp shape: {x_from_sp.shape}"

        x_to_sp = sp_transition_fwd(x, from_sp=False)
        assert x_to_sp.shape == (B, seq_chunk, D), f"Wrong to_sp shape: {x_to_sp.shape}"
        print(f"sp_transition_fwd: from_sp={list(x_from_sp.shape)}, to_sp={list(x_to_sp.shape)} — OK")
    else:
        print("(跳过通信操作：分布式环境未初始化)")

    print("\n序列并行在 TP 区域内沿序列维度切分 LayerNorm/Dropout 的激活值。")
    print("通过 scatter/gather 在 TP 和 SP 分片维度之间切换，减少激活显存占用。")

    if dist.is_initialized():
        cleanup()

"""序列并行 (Sequence Parallel, SP)。在 TP 区域内沿序列维度切分 LayerNorm/Dropout 的激活值，减少激活显存。

与 Megatron-LM SP 一致：在 column parallel 区域外（LayerNorm/Dropout）沿 sequence 维度切分，
跨 rank 做 all-gather 或 reduce-scatter 来切换分片维度。
"""
import torch
import torch.distributed as dist
from parallel.communication.setup import get_rank, get_world_size


def scatter_along_seq(x: torch.Tensor) -> torch.Tensor:
    """将 tensor 沿 sequence 维度切分，每个 rank 保留本地 chunk"""
    rank = get_rank()
    world_size = get_world_size()
    seq_len = x.shape[1]
    chunk_size = seq_len // world_size
    return x[:, rank * chunk_size : (rank + 1) * chunk_size].contiguous()


def gather_along_seq(x: torch.Tensor, total_seq_len: int) -> torch.Tensor:
    """从所有 rank 收集 sequence chunk，拼回完整序列"""
    world_size = get_world_size()
    full = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(full, x)
    return torch.cat(full, dim=1)[:, :total_seq_len].contiguous()


def sp_transition_fwd(x: torch.Tensor, from_sp: bool = True) -> torch.Tensor:
    """
    SP 与非 SP 区域之间的转换。
    from_sp=True: 离开 SP 区域，需要 all-gather 恢复完整序列
    from_sp=False: 进入 SP 区域，需要 reduce-scatter 或简单切分
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

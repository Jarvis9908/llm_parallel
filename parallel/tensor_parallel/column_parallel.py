"""列并行：权重矩阵按列切分。每个 rank 持有 W 的一部分列，输入相同，输出拼接。"""
import torch
import torch.distributed as dist
from parallel.communication.setup import get_rank, get_world_size


def column_parallel_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    列并行前向传播。
    x: (B, S, dim), weight: (dim, hidden_dim // world_size) — 已经切分好的本地权重
    返回: (B, S, hidden_dim) — all-gather 后的完整输出
    """
    local_out = x @ weight  # (B, S, hidden_local)
    # All-gather 拼接所有 rank 的输出
    full_out = [torch.zeros_like(local_out) for _ in range(get_world_size())]
    dist.all_gather(full_out, local_out)
    return torch.cat(full_out, dim=-1)


def split_weight_column(weight: torch.Tensor) -> torch.Tensor:
    """将权重按列切分到当前 rank"""
    rank = get_rank()
    world_size = get_world_size()
    chunk_size = weight.shape[1] // world_size
    return weight[:, rank * chunk_size : (rank + 1) * chunk_size].clone()


if __name__ == "__main__":
    from parallel.communication.setup import init_process_group, cleanup

    print("=" * 60)
    print("列并行 (Column Parallel) 演示")
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

    B, S, dim, hidden_dim = 2, 4, 8, 16
    full_w = torch.randn(dim, hidden_dim)
    local_w = split_weight_column(full_w)
    assert local_w.shape == (dim, hidden_dim // ws), f"Wrong local shape: {local_w.shape}"
    print(f"本地权重形状: {list(local_w.shape)} (全局: {list(full_w.shape)})")

    x = torch.randn(B, S, dim)

    if dist.is_initialized():
        out = column_parallel_linear(x, local_w)
        expected = x @ full_w
        assert out.shape == (B, S, hidden_dim), f"Wrong output shape: {out.shape}"
        if ws == 1:
            assert torch.allclose(out, expected, atol=1e-5), "Output mismatch"
        print(f"输出形状: {list(out.shape)} — OK")
    else:
        print("(跳过通信：分布式环境未初始化)")

    print("\n列并行将权重沿列方向切分，输入在所有 rank 上相同。")
    print("各 rank 计算局部输出后，通过 all-gather 拼接得到完整结果。")

    if dist.is_initialized():
        cleanup()

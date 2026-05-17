"""行并行：权重矩阵按行切分。每个 rank 持有 W 的一部分行，输入对应切分，输出需 all-reduce。"""
import torch
import torch.distributed as dist
from parallel.communication.setup import get_rank, get_world_size


def row_parallel_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    行并行前向传播。
    x: (B, S, hidden_dim // world_size) — 已按列切分的输入
    weight: (hidden_dim // world_size, dim) — 已切分的本地权重
    返回: (B, S, dim) — all-reduce 求和后的完整输出
    """
    local_out = x @ weight  # (B, S, dim)
    dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
    return local_out


def split_weight_row(weight: torch.Tensor) -> torch.Tensor:
    rank = get_rank()
    world_size = get_world_size()
    chunk_size = weight.shape[0] // world_size
    return weight[rank * chunk_size : (rank + 1) * chunk_size].clone()


if __name__ == "__main__":
    from parallel.communication.setup import init_process_group, cleanup

    print("=" * 60)
    print("行并行 (Row Parallel) 演示")
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

    B, S, hidden_dim, dim = 2, 4, 16, 8
    full_w = torch.randn(hidden_dim, dim)
    local_w = split_weight_row(full_w)
    hidden_local = hidden_dim // ws
    assert local_w.shape == (hidden_local, dim), f"Wrong local shape: {local_w.shape}"
    print(f"本地权重形状: {list(local_w.shape)} (全局: {list(full_w.shape)})")

    x = torch.randn(B, S, hidden_local)
    print(f"本地输入形状: {list(x.shape)}")

    if dist.is_initialized():
        out = row_parallel_linear(x, local_w)
        assert out.shape == (B, S, dim), f"Wrong output shape: {out.shape}"
        print(f"输出形状: {list(out.shape)} — OK")
    else:
        print("(跳过通信：分布式环境未初始化)")

    print("\n行并行将权重沿行方向切分，输入也对应切分。")
    print("各 rank 计算局部结果后，通过 all-reduce 求和得到完整输出。")
    print("行并行通常紧跟在列并行之后，免去列并行输出端的 all-gather。")

    if dist.is_initialized():
        cleanup()

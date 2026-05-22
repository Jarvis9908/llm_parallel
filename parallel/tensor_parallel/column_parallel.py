"""列并行 (Column Parallel Linear) —— 权重矩阵按列切分，各卡同输入、异权重，输出 all-gather 拼接。

直觉理解
--------
想象一块大蛋糕竖着切成 P 条，每条分给一个人。列并行也是这样：把权重矩阵 W 沿列方向
切成 P 份，每张卡拿到一份窄矩阵，用相同的输入 X 分别做矩阵乘法，最后把各卡的输出
像拼图一样 all-gather 拼回完整结果。

数学原理
--------
设全局权重 W ∈ ℝ^(d×h)，world_size = P，当前 rank 为 i：

1. 权重切分：Wᵢ = W[:, i·(h/P) : (i+1)·(h/P)]  ∈ ℝ^(d × h/P)
2. 本地计算：Yᵢ = X @ Wᵢ  ∈ ℝ^(B×S × h/P)
3. 通信聚合：Y = AllGather([Y₀, Y₁, ..., Yₚ₋₁])  ∈ ℝ^(B×S × h)

通信量分析：all-gather 传输 B×S×h 个元素（每卡发送 B×S×h/P，接收 (P-1)·B×S×h/P）。

与行并行的关系：列并行 + 行并行天然配对使用。列并行输出端的 all-gather 可以被
后续行并行输入端的 all-reduce 替代，从而省去一次通信。详见 row_parallel.py。

代码流程
--------
1. split_weight_column: 将完整权重按列切分，每卡保留属于自己的列块
2. column_parallel_linear: 用本地列块做矩阵乘法，all-gather 拼接所有 rank 的输出
"""
import torch
import torch.distributed as dist
from parallel.communication.setup import get_rank, get_world_size


def column_parallel_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """列并行线性层前向传播：各卡用相同输入和本地列块权重计算，all-gather 拼接完整输出。

    直觉
    ----
    每个人拿同一份考卷（输入 X），但只算自己那几道题（本地列块 Wᵢ），
    最后把所有人的答案汇总（all-gather），得到完整答卷。

    数学
    ----
    Y_local = X @ W_local          ∈ ℝ^(B×S × h/P)
    Y_full = AllGather(Y_local)    ∈ ℝ^(B×S × h)

    Args:
        x: 输入张量，形状 (B, S, d)，所有 rank 上相同。
        weight: 本地列块权重，形状 (d, h/P)，已经按列切分好。

    Returns:
        完整输出张量，形状 (B, S, h)，all-gather 后所有 rank 上相同。

    Note:
        当列并行与行并行配对使用时，此函数的 all-gather 可省略，
        直接将 (B, S, h/P) 的本地输出传给行并行即可。
    """
    local_out = x @ weight  # (B, S, hidden_local)
    # All-gather 拼接所有 rank 的输出
    full_out = [torch.zeros_like(local_out) for _ in range(get_world_size())]
    dist.all_gather(full_out, local_out)
    return torch.cat(full_out, dim=-1)


def split_weight_column(weight: torch.Tensor) -> torch.Tensor:
    """将权重矩阵按列切分，返回当前 rank 对应的列块。

    直觉
    ----
    把矩阵竖着切：每卡取自己那几列，就像把一本书按页码分给不同人。

    数学
    ----
    W_local = W[:, rank·chunk : (rank+1)·chunk]    ∈ ℝ^(d × h/P)
    其中 chunk = h // world_size

    Args:
        weight: 完整权重矩阵，形状 (d, h)。

    Returns:
        当前 rank 的列块权重，形状 (d, h/P)。
    """
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

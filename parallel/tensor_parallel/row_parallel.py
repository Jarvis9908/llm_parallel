"""行并行 (Row Parallel Linear) —— 权重矩阵按行切分，输入对应切分，各卡计算后 all-reduce 求和。

直觉理解
--------
列并行是竖着切蛋糕，行并行则是横着切：把权重矩阵 W 沿行方向切成 P 份，每张卡拿到
一段矮矩阵，同时输入 X 也要沿最后一维对应切分。各卡分别做矩阵乘法后，因为数学上
Y = XW = Σᵢ XᵢWᵢ，所以只需一次 all-reduce 求和即可得到完整输出。

数学原理
--------
设全局权重 W ∈ ℝ^(h×d)，world_size = P，当前 rank 为 i：

1. 权重切分：Wᵢ = W[i·(h/P) : (i+1)·(h/P), :]  ∈ ℝ^(h/P × d)
2. 输入切分：Xᵢ = X[:, :, i·(h/P) : (i+1)·(h/P)]  ∈ ℝ^(B×S × h/P)
3. 本地计算：Yᵢ = Xᵢ @ Wᵢ  ∈ ℝ^(B×S × d)
4. 通信聚合：Y = AllReduce(Σᵢ Yᵢ)  ∈ ℝ^(B×S × d)

数学等价性证明：
    Y = X @ W
      = [X₁, X₂, ..., Xₚ] @ [W₁; W₂; ...; Wₚ]    （分块矩阵乘法）
      = X₁W₁ + X₂W₂ + ... + XₚWₚ
      = Σᵢ XᵢWᵢ

通信量分析：all-reduce 传输 B×S×d 个元素。

与列并行的配对：列并行输出 (B×S×h/P) 恰好是行并行所需的输入形状，
因此一对「列并行 → 行并行」只需一次 all-reduce，省去了列并行单独使用时的 all-gather。
"""
import torch
import torch.distributed as dist
from parallel.communication.setup import get_rank, get_world_size


def row_parallel_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """行并行线性层前向传播：各卡用切分后的输入和本地行块权重计算，all-reduce 求和得到完整输出。

    直觉
    ----
    每个人拿到考卷的不同部分（切分后的输入 Xᵢ），配合自己手上的答案模板（行块 Wᵢ），
    各自算出部分分数，最后把所有人的分数加起来（all-reduce）就是总分。

    数学
    ----
    Y_local = X_local @ W_local    ∈ ℝ^(B×S × d)
    Y = AllReduce(Σᵢ Yᵢ)          ∈ ℝ^(B×S × d)

    等价性：Y = X @ W = [X₁,...,Xₚ] @ [W₁;...;Wₚ] = Σᵢ XᵢWᵢ

    Args:
        x: 本地输入张量，形状 (B, S, h/P)，已沿最后一维切分。
        weight: 本地行块权重，形状 (h/P, d)，已按行切分。

    Returns:
        完整输出张量，形状 (B, S, d)，all-reduce 后所有 rank 上相同。

    Note:
        行并行通常紧跟列并行使用，列并行的本地输出 (B, S, h/P) 直接作为行并行的输入，
        无需额外通信，最终只需一次 all-reduce。
    """
    local_out = x @ weight  # (B, S, dim)
    dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
    return local_out


def split_weight_row(weight: torch.Tensor) -> torch.Tensor:
    """将权重矩阵按行切分，返回当前 rank 对应的行块。

    直觉
    ----
    把矩阵横着切：每卡取自己那几行，就像把一栋楼按楼层分给不同人管理。

    数学
    ----
    W_local = W[rank·chunk : (rank+1)·chunk, :]    ∈ ℝ^(h/P × d)
    其中 chunk = h // world_size

    Args:
        weight: 完整权重矩阵，形状 (h, d)。

    Returns:
        当前 rank 的行块权重，形状 (h/P, d)。
    """
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

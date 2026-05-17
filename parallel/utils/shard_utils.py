"""权重切分与重组工具：支持沿任意维度拆分张量，并提供可视化展示。"""
import torch


def split_tensor(tensor: torch.Tensor, dim: int, rank: int, world_size: int) -> torch.Tensor:
    """
    沿指定维度将张量均匀切分为 world_size 份，返回当前 rank 对应分片。

    常用于张量并行（TP）中将权重矩阵按行或按列拆分到多个 GPU 上，
    每个 rank 获得大小相等的权重分片。

    Args:
        tensor: 待切分的原始张量
        dim: 切分所沿的维度
        rank: 当前设备的序号，取值范围 [0, world_size)
        world_size: 并行 GPU 总数

    Returns:
        torch.Tensor: 当前 rank 对应的本地分片（已调用 clone 保证内存独立）
    """
    chunk_size = tensor.shape[dim] // world_size
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(rank * chunk_size, (rank + 1) * chunk_size)
    return tensor[tuple(slices)].clone()


def gather_tensor(local: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    """
    沿指定维度将所有 rank 的本地分片拼接恢复为完整张量。

    在多机多卡场景中，需要调用 all-gather 通信原语来收集分片。
    当前为单机简化实现，直接返回本地张量。

    Args:
        local: 当前 rank 的本地分片
        dim: 拼接所沿的维度
        world_size: 参与并行的 GPU 总数

    Returns:
        torch.Tensor: 拼接后的完整张量
    """
    return local


def visualize_sharding(weight: torch.Tensor, tp_size: int, pp_size: int):
    """
    用 ASCII 字符画可视化权重的切分情况。

    展示总参数量、张量并行（TP）切分后每片大小、
    以及管道并行（PP）进一步切分后的每微片大小。

    Args:
        weight: 待切分的权重张量
        tp_size: 张量并行的 GPU 数量
        pp_size: 管道并行的阶段数量
    """
    total_params = weight.numel()
    per_tp = total_params // tp_size
    per_pp = per_tp // pp_size
    print(f"总参数量: {total_params:,}")
    print(f"TP 每片: {per_tp:,}")
    print(f"TP+PP 每微片: {per_pp:,}")
    print(f"TP 切分: {' | '.join(['#' * 10] * tp_size)}")
    print(f"PP 切分: {' -> '.join(['-' * 5] * pp_size)}")


if __name__ == "__main__":
    # 演示：将 4x8 矩阵沿 dim=0 按 2 路 TP 切分
    weight = torch.arange(32).reshape(4, 8).float()
    print(f"原始张量形状: {weight.shape}")

    for rank in range(2):
        shard = split_tensor(weight, dim=0, rank=rank, world_size=2)
        print(f"[Rank {rank}] 分片形状: {shard.shape}, 内容:\n{shard}")

    print("\n=== 权重切分可视化 ===")
    visualize_sharding(weight, tp_size=2, pp_size=2)

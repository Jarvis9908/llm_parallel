"""流水线并行的层切分。将模型的 TransformerBlock 按层分配到不同 rank。

直觉
----
把一本厚书按章节分给几个人同时看——每个人负责一段章节，
前一个人看完自己那部分后把"笔记"传给下一个人继续看。
在流水线并行中，模型按层切成多个 stage，每个 GPU 负责一个 stage。

数学
----
1. 均匀分配：L 层分给 P 个 rank，每 rank 分到 ⌊L/P⌋ 层，
   余数 L%P 个 rank 各多分 1 层。

2. 非均匀分配：当不同层的计算量不同（如 Attention 层 vs FFN 层），
   可按计算量比例分配，目标是让每个 stage 的前向+反向耗时尽量接近。

3. 负载均衡考量：若各 stage 耗时差异大，最慢的 stage 会成为瓶颈，
   导致其他 stage 空闲等待。因此层分配应使各 stage 计算量均衡。

代码流程
--------
1. ``partition_layers`` —— 简化版层分配（单机演示）
2. ``get_layer_range`` —— 计算指定 rank 负责的层范围
"""
import torch


def partition_layers(layers: list[torch.nn.Module]) -> list[torch.nn.Module]:
    """简化的层分配：所有层都在本地（单机演示用）。

    实际分布式场景中，每个 rank 只保留自己负责的那部分层，
    其余层替换为通信占位符。本函数为简化演示，直接返回全部层。
    """
    return layers


def get_layer_range(n_layers: int, rank: int, world_size: int) -> tuple[int, int]:
    """计算当前 rank 负责的层范围 [start, end)。

    直觉：把 L 层尽量均匀地分给 P 个 rank——每人先拿 ⌊L/P⌋ 层，
    剩余的 L%P 层从前到后每个 rank 多分 1 层。

    数学：
        base = ⌊L / P⌋, remainder = L % P
        rank i 的起始层 = i × base + min(i, remainder)
        rank i 的层数 = base + (1 if i < remainder else 0)
        即前 remainder 个 rank 各多分 1 层

    Args:
        n_layers: 模型总层数 L
        rank: 当前 rank 编号，取值 [0, world_size)
        world_size: GPU 总数 P

    Returns:
        tuple[int, int]: (start, end) 层范围，左闭右开
    """
    layers_per_rank = n_layers // world_size
    remainder = n_layers % world_size
    start = rank * layers_per_rank + min(rank, remainder)
    end = start + layers_per_rank + (1 if rank < remainder else 0)
    return start, end


if __name__ == "__main__":
    print("=== layer_partition demo ===")
    # 模拟 12 层 Transformer 分配到 4 个 rank
    n_layers = 12
    world_size = 4
    for rank in range(world_size):
        start, end = get_layer_range(n_layers, rank, world_size)
        print(f"Rank {rank}: layers [{start}:{end}) ({end - start} layers)")

    # 模拟 13 层非均匀分配
    n_layers = 13
    print(f"\nUneven partition ({n_layers} layers, {world_size} ranks):")
    for rank in range(world_size):
        start, end = get_layer_range(n_layers, rank, world_size)
        print(f"Rank {rank}: layers [{start}:{end}) ({end - start} layers)")

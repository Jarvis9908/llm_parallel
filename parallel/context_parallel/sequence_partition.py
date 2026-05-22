"""上下文并行中的序列维度切分和 causal mask 调整。

直觉
----
超长序列被切成多段，每段放在一张 GPU 上独立计算注意力。
但因果模型（causal model）要求后面的 token 不能看到前面的 token——
切分后，后面的 rank 天然看不到前面 rank 的 token，这反而符合因果约束。
但前面 rank 的 token 需要对后面 rank 可见（作为上下文），
因此 causal mask 需要调整：后面的 rank 可以"偷看"前面 rank 的 token。

数学
----
1. 序列切分：将长度为 S 的序列切给 P 个 rank，
   chunk_size = S // P，最后一个 rank 取到末尾（处理余数）。

2. Causal mask 调整规则：
   对于 rank r 上的局部位置 i（全局位置 = r × chunk_size + i），
   它可以看到：
     - 所有之前 rank 的 token（全局位置 < r × chunk_size）
     - 本 rank 内因果允许的 token（全局位置 ≤ r × chunk_size + i）
   即 mask[i, k] = False（可见）当 k < local_start + i + 1，
   否则 mask[i, k] = True（遮蔽）。

代码流程
--------
1. ``partition_sequence`` —— 将输入沿 seq_len 均分到各 rank
2. ``create_cp_causal_mask`` —— 生成 CP 下的 causal mask
"""
import torch


def partition_sequence(x: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """将输入沿 seq_len 均分到各 rank。

    直觉：把一条长序列切成 P 段，每段交给一张 GPU 处理。

    数学：
        chunk_size = S // P
        rank r 的范围 = [r × chunk_size, (r+1) × chunk_size)
        最后一个 rank 的范围延伸到 S（处理余数部分）

    Args:
        x: 输入张量，形状 (B, S, D)
        rank: 当前 rank 编号
        world_size: GPU 总数 P

    Returns:
        torch.Tensor: 当前 rank 分到的序列片段，形状 (B, S_local, D)
    """
    seq_len = x.shape[1]
    chunk_size = seq_len // world_size
    start = rank * chunk_size
    end = start + chunk_size if rank < world_size - 1 else seq_len
    return x[:, start:end].contiguous()


def create_cp_causal_mask(
    seq_len: int, rank: int, world_size: int
) -> torch.Tensor:
    """
    CP 下的 causal mask。

    直觉：后面的 rank 可以"偷看"前面 rank 的 token——因为因果约束只禁止
    看到未来的 token，而前面 rank 的 token 属于"过去"，理应可见。

    数学：
        chunk_size = S // P, local_start = rank × chunk_size
        mask[i, k] 规则：
        - k < local_start + i + 1 → False（可见，允许注意力）
        - k ≥ local_start + i + 1 → True（遮蔽，禁止注意力）
        其中 i 是局部位置索引，k 是全局位置索引

        rank 0: 标准 causal mask（local_start=0）
        rank > 0: 扩展 mask，能看到之前所有分片的 token

    Args:
        seq_len: 完整序列长度 S
        rank: 当前 rank 编号
        world_size: GPU 总数 P

    Returns:
        torch.Tensor: causal mask，形状 (chunk_size, local_start + chunk_size)，
        True 表示遮蔽，False 表示可见
    """
    chunk_size = seq_len // world_size
    local_start = rank * chunk_size
    mask = torch.ones(chunk_size, local_start + chunk_size, dtype=torch.bool)
    for i in range(chunk_size):
        mask[i, :local_start + i + 1] = False  # 允许看到之前分片 + 本地 causal
    return mask


if __name__ == "__main__":
    print("=== sequence_partition demo ===")

    # 序列切分
    B, seq_len, D = 2, 16, 8
    x = torch.arange(B * seq_len * D, dtype=torch.float32).reshape(B, seq_len, D)
    world_size = 4

    print(f"Input shape: {x.shape}, splitting across {world_size} ranks:")
    for rank in range(world_size):
        chunk = partition_sequence(x, rank, world_size)
        print(f"  Rank {rank}: chunk shape {chunk.shape}")

    # Causal mask 生成
    print(f"\nCP causal mask (seq_len=8, world_size=2):")
    for rank in range(2):
        mask = create_cp_causal_mask(seq_len=8, rank=rank, world_size=2)
        print(f"  Rank {rank} mask shape: {mask.shape}")
        print(f"    False count (visible): {mask.numel() - mask.sum().item()}")
        print(f"    True count (masked): {mask.sum().item()}")

    # 边界情况：无法均匀切分
    x_uneven = torch.randn(2, 10, 8)
    print(f"\nUneven partition (seq_len=10, world_size=3):")
    for rank in range(3):
        chunk = partition_sequence(x_uneven, rank, 3)
        print(f"  Rank {rank}: chunk shape {chunk.shape}")

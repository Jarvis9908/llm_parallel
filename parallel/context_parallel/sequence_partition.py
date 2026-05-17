"""上下文并行中的序列维度切分和 causal mask 调整。"""
import torch


def partition_sequence(x: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """将输入沿 seq_len 均分到各 rank"""
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
    在切分序列后，每个 rank 只能看到自己分片内 + 之前分片的 token。
    rank 0: 标准 causal mask
    rank > 0: 扩展 mask，能看到之前所有分片
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

"""专家并行：将 MoE 的 Expert 分布到不同 GPU 上。"""
import torch


def partition_experts(n_experts: int, rank: int, world_size: int) -> list[int]:
    """计算当前 rank 持有的 expert 索引列表"""
    experts_per_rank = n_experts // world_size
    start = rank * experts_per_rank
    end = start + experts_per_rank
    if rank == world_size - 1:
        end = n_experts  # 最后一个 rank 收尾
    return list(range(start, end))


def get_expert_owner(expert_idx: int, world_size: int) -> int:
    """返回持有指定 expert 的 rank"""
    return expert_idx % world_size


if __name__ == "__main__":
    print("=== expert_partition demo ===")

    # 8 个 Expert 分到 4 个 GPU
    n_experts = 8
    world_size = 4
    print(f"Partitioning {n_experts} experts across {world_size} ranks:")
    for rank in range(world_size):
        experts = partition_experts(n_experts, rank, world_size)
        print(f"  Rank {rank}: experts {experts}")

    # 查询每个 expert 属于哪个 rank
    print("\nExpert → Rank mapping:")
    for expert_idx in range(n_experts):
        owner = get_expert_owner(expert_idx, world_size)
        print(f"  Expert {expert_idx} -> Rank {owner}")

    # 非均匀情况 (13 experts, 4 ranks)
    n_experts_uneven = 13
    print(f"\nUneven partition ({n_experts_uneven} experts, {world_size} ranks):")
    for rank in range(world_size):
        experts = partition_experts(n_experts_uneven, rank, world_size)
        print(f"  Rank {rank}: experts {experts}")

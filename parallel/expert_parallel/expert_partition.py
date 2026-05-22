"""专家并行：将 MoE 的 Expert 分布到不同 GPU 上。

直觉
----
MoE（Mixture of Experts）模型中的 Expert 就像公司里的不同部门——
每个 Expert 擅长处理特定类型的 token。Expert Parallelism 把这些"部门"
分配到不同的 GPU 上，每张卡负责若干 Expert 的计算，从而突破单卡显存瓶颈。

数学
----
1. 均匀分配：N 个 Expert 分给 P 张卡，每卡分到 ⌊N/P⌋ 个 Expert，
   余数 N%P 个卡各多分 1 个。
   负载均衡目标：max |E_i| - min |E_i| ≤ 1，其中 E_i 是第 i 张卡的 Expert 集合。

2. 非均匀分配：当 Expert 计算量不同时，可按计算量比例分配，
   目标是最小化 max_i(T_i)，其中 T_i 是第 i 张卡的总计算时间。

3. 与 All-to-All 通信的关系：token 被路由到不同 Expert 所在的卡后，
   需要通过 All-to-All 通信将 token 发送到目标卡，计算完成后再发回。
   通信量 ∝ 2 × 被路由的 token 数 × 隐层维度。

代码流程
--------
1. ``partition_experts`` —— 计算每张卡持有哪些 Expert
2. ``get_expert_owner`` —— 给定 Expert 编号，查询它所在的 rank
"""
import torch


def partition_experts(n_experts: int, rank: int, world_size: int) -> list[int]:
    """计算当前 rank 持有的 expert 索引列表。

    直觉：把 N 个 Expert 尽量均匀地分给 P 张卡，就像把 N 本书
    平均分给 P 个人——每人先拿 ⌊N/P⌋ 本，剩下的余数本从前到后
    每人多发一本（本实现中最后一个 rank 收尾所有余数）。

    数学：
        base = ⌊N / P⌋
        第 i 张卡的 Expert 范围 = [i × base, (i+1) × base)
        最后一张卡额外包含余数部分，范围延伸到 N

    Args:
        n_experts: Expert 总数 N
        rank: 当前 GPU 的编号，取值 [0, world_size)
        world_size: GPU 总数 P

    Returns:
        list[int]: 当前 rank 持有的 Expert 索引列表
    """
    experts_per_rank = n_experts // world_size
    start = rank * experts_per_rank
    end = start + experts_per_rank
    if rank == world_size - 1:
        end = n_experts  # 最后一个 rank 收尾
    return list(range(start, end))


def get_expert_owner(expert_idx: int, world_size: int) -> int:
    """返回持有指定 expert 的 rank。

    直觉：给定一个 Expert 编号，查它被分配到了哪张卡——
    类似查表：Expert 0 在卡 0，Expert 1 在卡 1，……循环分配。

    数学：
        owner = expert_idx % world_size
    即按取模方式循环分配，保证 Expert 均匀散布到各卡。

    Args:
        expert_idx: Expert 编号，取值 [0, N)
        world_size: GPU 总数 P

    Returns:
        int: 持有该 Expert 的 rank 编号
    """
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

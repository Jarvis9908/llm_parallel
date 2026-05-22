"""Token 分发：根据 router 结果将 token 发送到对应 expert 所在的 rank。

直觉
----
Router 就像快递分拣中心——每个 token 是一个包裹，Router 决定它该去
哪个 Expert（目的地），然后分发系统把包裹送到对应的 GPU 上处理。
处理完后再把结果送回原来的 GPU。

数学
----
1. 路由矩阵：设 R ∈ {0,1}^(B×S×N) 为路由矩阵，R[b,s,e]=1 表示
   token (b,s) 被路由到 Expert e。Top-K 路由下每行恰好 K 个 1。

2. All-to-All 通信过程：
   - Phase 1 (dispatch): 每个 rank 将本地 token 按 Expert 归属发送到目标 rank
   - Phase 2 (compute): 各 rank 对收到的 token 执行本地 Expert 计算
   - Phase 3 (gather): 将计算结果发回原 rank
   通信量 ≈ 2 × B × S × D（D 为隐层维度），即每个 token 来回各传一次。

3. 负载均衡挑战：若 Router 倾向于选择少数 Expert，会导致某些 GPU
   过载而其他 GPU 空闲。常见缓解方法包括 auxiliary loss、capacity factor、
   expert choice 等路由策略。

代码流程
--------
1. ``dispatch_tokens_to_experts`` —— 单机版：按 Expert 分组 token
2. ``all_to_all_dispatch_example`` —— 多机版：演示 All-to-All 通信角色
"""
import torch


def dispatch_tokens_to_experts(
    x: torch.Tensor, router_indices: torch.Tensor, n_experts: int
) -> dict[int, torch.Tensor]:
    """
    根据 router indices 将 token 分发到对应的 expert。

    直觉：拿着路由表（router_indices），把每个 token 送到它被路由到的
    Expert 那里——就像拿着分拣单把包裹放到对应的分拣筐中。

    数学：
        tokens_e = { x[b,s] | ∃k: router_indices[b,s,k] = e }
    即对于每个 Expert e，收集所有被路由到该 Expert 的 token。

    单机版本：直接按 expert 分组（无跨机通信）。

    Args:
        x: 输入 token 张量，形状 (B, S, D)
        router_indices: Router 输出的 top-K Expert 索引，形状 (B, S, K)
        n_experts: Expert 总数

    Returns:
        dict[int, torch.Tensor]: {expert_idx: token_tensor}，每个 Expert
        收到的 token 拼成一个 (n_tokens, D) 的张量
    """
    B, S, D = x.shape
    top_k = router_indices.shape[-1]
    expert_tokens = {}
    for expert_idx in range(n_experts):
        mask = (router_indices == expert_idx)
        token_mask = mask.any(dim=-1)  # (B, S)
        if token_mask.any():
            expert_tokens[expert_idx] = x[token_mask]
    return expert_tokens


def all_to_all_dispatch_example(
    x: torch.Tensor, rank: int, world_size: int
) -> torch.Tensor:
    """
    演示 all-to-all 通信在 EP 中的角色。

    直觉：每张卡既是发送方也是接收方——把自己不负责的 token 发出去，
    同时接收别的卡发来、自己负责的 token。就像所有人同时互寄快递。

    数学：
        All-to-All 通信量 = P × 单次发送量
        其中 P 为 GPU 数，单次发送量 ≈ (B×S/P) × D（假设均匀路由）

    每个 rank 持有部分 token，需要根据 expert 归属跨 rank 交换。
    实际中在每个 rank 上本地计算 router，然后 all-to-all 分发。

    Args:
        x: 本地 rank 持有的 token 张量
        rank: 当前 rank 编号
        world_size: GPU 总数

    Returns:
        torch.Tensor: 分发后的 token 张量（简化实现直接返回输入）
    """
    # 简化：本地处理，不做实际通信
    return x


if __name__ == "__main__":
    print("=== token_dispatch demo ===")

    # 模拟 batch: (B=2, S=4, D=8) 的 token，4 个 expert，top_k=2
    B, S, D = 2, 4, 8
    x = torch.randn(B, S, D)
    # router 输出 top-2 expert indices
    router_indices = torch.tensor([
        [[0, 2], [1, 3], [0, 1], [2, 3]],
        [[1, 0], [3, 2], [2, 1], [0, 3]],
    ])

    n_experts = 4
    dispatched = dispatch_tokens_to_experts(x, router_indices, n_experts)

    print(f"Input shape: {x.shape}")
    print(f"Number of experts: {n_experts}")
    for expert_idx, tokens in dispatched.items():
        print(f"  Expert {expert_idx}: tokens shape {tokens.shape}")

    # all-to-all demo
    print(f"\nAll-to-all dispatch demo (rank=0, world_size=4):")
    result = all_to_all_dispatch_example(x, rank=0, world_size=4)
    print(f"  Output shape: {result.shape}")

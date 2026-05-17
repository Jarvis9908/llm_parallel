"""Token 分发：根据 router 结果将 token 发送到对应 expert 所在的 rank。"""
import torch


def dispatch_tokens_to_experts(
    x: torch.Tensor, router_indices: torch.Tensor, n_experts: int
) -> dict[int, torch.Tensor]:
    """
    根据 router indices 将 token 分发到对应的 expert。
    单机版本：直接按 expert 分组（无跨机通信）。

    返回: {expert_idx: tensor_of_tokens}
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
    每个 rank 持有部分 token，需要根据 expert 归属跨 rank 交换。
    实际中在每个 rank 上本地计算 router，然后 all-to-all 分发。
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

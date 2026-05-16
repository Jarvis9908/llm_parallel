"""
DeepSeek V3 Mixture of Experts (MoE) 层实现。

包含 Router（Top-K 门控路由）、SharedExpert（共享专家）、
RoutedExpert（路由专家）和 MoELayer（组合 MoE 层）。

每个 token 首先通过共享专家，然后根据 Router 的 Top-K 选择
激活若干个路由专家，最终输出为共享专家输出与加权路由专家输出之和。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.activation import silu
from models.deepseek_v3.config import DeepSeekV3Config


class Router(nn.Module):
    """
    Top-K 门控路由器。

    对每个 token 计算所有专家的 logits，使用 top-k 选择得分最高的 k 个专家，
    并对选中的 k 个专家得分做 softmax 归一化。

    Attributes:
        weight: 线性投影层 (dim -> n_experts)，无偏置。
        top_k: 每个 token 激活的专家数量。
    """

    def __init__(self, dim: int, n_experts: int, top_k: int):
        """
        Args:
            dim: 输入隐藏维度。
            n_experts: 路由专家总数。
            top_k: 每个 token 激活的专家数量。
        """
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Linear(dim, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量，形状 (B, S, dim)。

        Returns:
            scores: Top-K 专家得分（经过 softmax），形状 (B, S, top_k)。
            indices: Top-K 专家索引，形状 (B, S, top_k)。
        """
        logits = self.weight(x)                       # (B, S, n_experts)
        raw_scores, indices = torch.topk(logits, self.top_k, dim=-1)  # (B, S, top_k)
        scores = F.softmax(raw_scores, dim=-1)         # (B, S, top_k)
        return scores, indices


class SharedExpert(nn.Module):
    """
    共享专家（SwiGLU 结构）。

    所有 token 都会通过共享专家，计算方式与 SwiGLUFFN 相同：
    output = (SiLU(x @ w_gate) * (x @ w_up)) @ w_down

    所有线性层均无偏置。
    """

    def __init__(self, dim: int, intermediate_dim: int):
        """
        Args:
            dim: 输入/输出维度。
            intermediate_dim: 中间隐藏维度。
        """
        super().__init__()
        self.w_gate = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_up = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状 (B, S, dim)。

        Returns:
            输出张量，形状 (B, S, dim)。
        """
        gate = silu(self.w_gate(x))   # (B, S, intermediate_dim)
        up = self.w_up(x)             # (B, S, intermediate_dim)
        h = gate * up                 # 逐元素门控
        return self.w_down(h)         # (B, S, dim)


class RoutedExpert(nn.Module):
    """
    路由专家（SwiGLU 结构）。

    结构与 SharedExpert 相同，但仅处理被 Router 选中路由到该专家的 token。
    在 MoELayer 中由外部按 token 索引调度计算。
    """

    def __init__(self, dim: int, intermediate_dim: int):
        """
        Args:
            dim: 输入/输出维度。
            intermediate_dim: 中间隐藏维度。
        """
        super().__init__()
        self.w_gate = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_up = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 被路由到该专家的 token，形状 (N, dim)。

        Returns:
            输出张量，形状 (N, dim)。
        """
        gate = silu(self.w_gate(x))   # (N, intermediate_dim)
        up = self.w_up(x)             # (N, intermediate_dim)
        h = gate * up                 # 逐元素门控
        return self.w_down(h)         # (N, dim)


class MoELayer(nn.Module):
    """
    Mixture of Experts 层。

    组合共享专家和多个路由专家。每个 token 先通过共享专家，
    再根据 Router 的 Top-K 选择激活若干个路由专家。
    最终输出 = 共享专家输出 + 加权路由专家输出之和。

    路由专家的计算采用 scatter-add：对每个专家，收集所有被路由到它的 token，
    计算加权输出，再按原位置散射加回。
    """

    def __init__(self, config: DeepSeekV3Config):
        """
        Args:
            config: DeepSeekV3Config 配置对象。
        """
        super().__init__()

        dim = config.dim
        n_routed_experts = config.n_routed_experts
        n_activated_experts = config.n_activated_experts
        moe_intermediate_dim = config.moe_intermediate_dim

        # Router: Top-K 门控
        self.router = Router(
            dim=dim,
            n_experts=n_routed_experts,
            top_k=n_activated_experts,
        )

        # 共享专家
        self.shared_expert = SharedExpert(
            dim=dim,
            intermediate_dim=moe_intermediate_dim,
        )

        # 路由专家列表
        self.routed_experts = nn.ModuleList([
            RoutedExpert(dim=dim, intermediate_dim=moe_intermediate_dim)
            for _ in range(n_routed_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状 (B, S, dim)。

        Returns:
            输出张量，形状 (B, S, dim)。
        """
        B, S, D = x.shape

        # 1. 共享专家：所有 token 通过
        shared_out = self.shared_expert(x)            # (B, S, dim)

        # 2. Router：获取 Top-K 专家得分和索引
        router_scores, router_indices = self.router(x)  # (B, S, top_k), (B, S, top_k)

        # 3. 路由专家：为每个 token 累加加权专家输出
        flat_x = x.view(-1, D)                        # (B*S, dim)
        flat_scores = router_scores.view(-1, router_scores.shape[-1])  # (B*S, top_k)
        flat_indices = router_indices.view(-1, router_indices.shape[-1])  # (B*S, top_k)

        routed_out = torch.zeros_like(flat_x)          # (B*S, dim)

        for expert_idx in range(len(self.routed_experts)):
            # 找到所有 token 中该专家在 top_k 中的位置（可能有 0-2 个）
            expert_mask = (flat_indices == expert_idx)  # (B*S, top_k), bool
            token_indices, k_indices = expert_mask.nonzero(as_tuple=True)

            if token_indices.numel() == 0:
                continue  # 没有 token 被路由到该专家

            # 收集被路由到该专家的 token
            expert_input = flat_x[token_indices]          # (num_tokens, dim)
            expert_output = self.routed_experts[expert_idx](expert_input)  # (num_tokens, dim)

            # 加权：乘以 Router 分配的得分
            weight = flat_scores[token_indices, k_indices].unsqueeze(-1)  # (num_tokens, 1)
            weighted_output = expert_output * weight       # (num_tokens, dim)

            # Scatter-add: 按原位加回
            routed_out.index_add_(0, token_indices, weighted_output)

        routed_out = routed_out.view(B, S, D)          # (B, S, dim)

        # 4. 最终输出
        return shared_out + routed_out

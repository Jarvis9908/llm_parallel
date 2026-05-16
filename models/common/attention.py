"""
多头注意力机制手写实现：Multi-Head Attention (MHA)、Grouped Query Attention (GQA) 和 Multi-Query Attention (MQA)。

MHA: Q/K/V 各有 n_heads 个独立头，通过缩放点积注意力聚合信息
GQA: K/V 头数 n_kv_heads <= n_heads，通过 repeat_interleave 将 KV 头扩展到与 Q 头匹配
MQA: GQA 的特殊情况 (n_kv_heads=1)，一个 KV 头共享给所有 Q 头

GQA/MQA 是 LLaMA2、DeepSeek V3 等模型使用的注意力机制，
在推理速度和模型质量之间取得了很好的平衡（减少 KV 缓存大小）。
"""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    标准多头注意力机制。
    对输入的 Q/K/V 各做 n_heads 个线性投影，
    每个头独立执行缩放点积注意力，最后拼接所有头的输出。
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) 必须能被 n_heads ({n_heads}) 整除"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** 0.5

        # Q/K/V 投影矩阵 (dim -> dim)
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        # 输出投影矩阵
        self.w_o = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入张量按头拆分。
        (B, S, D) -> (B, n_heads, S, head_dim)
        """
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多头输出合并回原始维度。
        (B, n_heads, S, head_dim) -> (B, S, D)
        """
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, -1)

    def _scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        缩放点积注意力。
        Attention(Q, K, V) = softmax(Q @ K^T / scale) @ V

        Args:
            q: query  (B, n_heads, S, head_dim)
            k: key    (B, n_heads, S, head_dim)
            v: value  (B, n_heads, S, head_dim)
            mask: 布尔掩码，True 的位置将被设为 -inf

        Returns:
            (B, n_heads, S, head_dim)
        """
        scores = (q @ k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ v

    @staticmethod
    def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建因果注意力掩码，形状为 (1, 1, seq_len, seq_len)。
        mask[0, 0, i][j] = True 当 j > i（未来位置），即上三角掩码。
        已预留 batch 和 heads 维度以便直接广播到 scores 张量。
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(self, x: torch.Tensor, use_causal_mask: bool = False) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量 (B, S, dim)
            use_causal_mask: 是否使用因果掩码（Decoder 自回归场景）

        Returns:
            输出张量 (B, S, dim)
        """
        q = self.w_q(x)   # (B, S, dim)
        k = self.w_k(x)
        v = self.w_v(x)

        q = self._split_heads(q)   # (B, n_heads, S, head_dim)
        k = self._split_heads(k)
        v = self._split_heads(v)

        mask = None
        if use_causal_mask:
            mask = self._create_causal_mask(x.shape[1], x.device)

        out = self._scaled_dot_product_attention(q, k, v, mask)  # (B, n_heads, S, head_dim)
        out = self._merge_heads(out)                              # (B, S, dim)
        return self.w_o(out)


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 (GQA)。
    K/V 投影使用 n_kv_heads 个头（少于 Q 的 n_heads 个头），
    通过 repeat_interleave 将 KV 头复制到与 Q 头数一致。

    特殊场景：
    - n_kv_heads == n_heads: 等价于标准 MHA
    - n_kv_heads == 1:      等价于 MQA (Multi-Query Attention)
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, \
            f"dim ({dim}) 必须能被 n_heads ({n_heads}) 整除"
        assert n_heads % n_kv_heads == 0, \
            f"n_heads ({n_heads}) 必须能被 n_kv_heads ({n_kv_heads}) 整除"

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** 0.5
        self.n_rep = n_heads // n_kv_heads   # 每个 KV 头被多少个 Q 头复用

        # Q 投影: 全尺寸
        self.w_q = nn.Linear(dim, dim, bias=False)
        # K/V 投影: 缩减尺寸 (n_kv_heads * head_dim)
        self.w_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        # 输出投影
        self.w_o = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads_q(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, dim) -> (B, n_heads, S, head_dim)"""
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def _split_heads_kv(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, n_kv_heads * head_dim) -> (B, n_kv_heads, S, head_dim)"""
        B, S, _ = x.shape
        return x.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 KV 头从 n_kv_heads 扩展到 n_heads。
        每个 KV 头重复 n_rep 次，通过 repeat_interleave 实现。
        (B, n_kv_heads, S, head_dim) -> (B, n_heads, S, head_dim)
        """
        return x.repeat_interleave(self.n_rep, dim=1)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_heads, S, head_dim) -> (B, S, dim)"""
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, -1)

    def _scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        缩放点积注意力。
        Attention(Q, K, V) = softmax(Q @ K^T / scale) @ V
        """
        scores = (q @ k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ v

    @staticmethod
    def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """创建因果注意力掩码 (1, 1, seq_len, seq_len)，上三角布尔掩码"""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor, use_causal_mask: bool = False) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量 (B, S, dim)
            use_causal_mask: 是否使用因果掩码

        Returns:
            输出张量 (B, S, dim)
        """
        q = self.w_q(x)   # (B, S, dim)
        k = self.w_k(x)   # (B, S, n_kv_heads * head_dim)
        v = self.w_v(x)   # (B, S, n_kv_heads * head_dim)

        q = self._split_heads_q(q)     # (B, n_heads,    S, head_dim)
        k = self._split_heads_kv(k)    # (B, n_kv_heads, S, head_dim)
        v = self._split_heads_kv(v)    # (B, n_kv_heads, S, head_dim)

        k = self._repeat_kv(k)         # (B, n_heads,    S, head_dim)
        v = self._repeat_kv(v)         # (B, n_heads,    S, head_dim)

        mask = None
        if use_causal_mask:
            mask = self._create_causal_mask(x.shape[1], x.device)

        out = self._scaled_dot_product_attention(q, k, v, mask)
        out = self._merge_heads(out)
        return self.w_o(out)


if __name__ == "__main__":
    x = torch.randn(2, 16, 64)

    # MHA
    mha = MultiHeadAttention(dim=64, n_heads=8)
    out_mha = mha(x)
    assert out_mha.shape == x.shape
    print(f"MHA: {x.shape} -> {out_mha.shape}")

    # GQA
    gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=2)
    out_gqa = gqa(x)
    assert out_gqa.shape == x.shape
    print(f"GQA: {x.shape} -> {out_gqa.shape}")

    # MQA
    mqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=1)
    out_mqa = mqa(x)
    assert out_mqa.shape == x.shape
    print(f"MQA: {x.shape} -> {out_mqa.shape}")

    # Backward check
    x_grad = torch.randn(2, 16, 64, requires_grad=True)
    out = mha(x_grad)
    out.sum().backward()
    assert x_grad.grad is not None

    print("All attention checks passed.")

"""
多头潜在注意力（Multi-head Latent Attention, MLA）实现。

MLA 是 DeepSeek V3 的核心注意力机制，通过将 KV 压缩到低秩潜在空间来大幅降低
KV 缓存内存占用。同时采用解耦 RoPE 设计，仅对 Q 和 K 的一小部分维度施加旋转
位置编码，从而避免位置信息在 KV 压缩过程中丢失。

核心概念：
- 标准 MHA 的 KV 缓存：每层每 token 存储 n_heads × head_dim（K）+ n_heads × head_dim（V）
- MLA 的 KV 缓存：每层每 token 仅需存储 kv_lora_rank（压缩潜在向量）+
  qk_rope_head_dim（解耦 RoPE 键），推理时通过轻量上投影恢复完整 K/V
"""
import torch
import torch.nn as nn
import math

from models.deepseek_v3.config import DeepSeekV3Config


class MultiHeadLatentAttention(nn.Module):
    """多头潜在注意力（Multi-head Latent Attention）。

    将 K 和 V 压缩到低秩潜在空间，减少 KV 缓存大小。
    采用解耦 RoPE：Q 的 RoPE 部分为每头独立，K 的 RoPE 部分在所有头之间共享，
    直接将输入投影得到（不经潜在压缩），从而保留精确的位置信息。

    前向传播流程：
    1. Q = split_heads(w_q(x))
    2. kv_a = w_kv_a(x) → 分离为 kv_latent 和 k_rope_raw
    3. K = split_heads(w_k_b(kv_latent)), V = split_heads(w_v_b(kv_latent))
    4. 对 Q 的最后 qk_rope_head_dim 维和 K 的 RoPE 部分施加 RoPE
    5. 标准缩放点积注意力（可选因果掩码）
    6. 合并头部，输出投影
    """

    def __init__(self, config: DeepSeekV3Config):
        """初始化 MLA 模块。

        Args:
            config: DeepSeekV3Config 配置对象，包含 dim、n_heads、
                    kv_lora_rank、qk_rope_head_dim、max_seq_len、rope_theta。
        """
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.max_seq_len = config.max_seq_len
        self.scale = self.head_dim ** 0.5

        # Q 投影：输入 → Q（n_heads × head_dim）
        self.w_q = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)

        # KV 压缩投影：输入 → KV 潜在向量 + K RoPE 原始部分
        # 输出维度：kv_lora_rank（压缩潜在）+ qk_rope_head_dim（解耦 RoPE 键）
        self.w_kv_a = nn.Linear(
            self.dim,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )

        # K 上投影：潜在向量 → 完整 K（n_heads × head_dim）
        self.w_k_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
        )

        # V 上投影：潜在向量 → 完整 V（n_heads × head_dim）
        self.w_v_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
        )

        # 输出投影
        self.w_o = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        # 预计算 RoPE 的 cos/sin 表格（针对 qk_rope_head_dim 频率）
        self._build_rope_cache(config.rope_theta)

    def _build_rope_cache(self, rope_theta: float):
        """预计算 RoPE 的 cos 和 sin 缓存表。

        频率公式：theta_i = rope_theta^(-2i / qk_rope_head_dim)
        对每个位置 m，计算 cos(m * theta_i) 和 sin(m * theta_i)。
        缓存通过 register_buffer 注册，随模型移动/保存但不参与梯度。

        Args:
            rope_theta: RoPE 频率基数，默认 10000.0。
        """
        rope_dim = self.qk_rope_head_dim
        # 频率：theta^(-2i/rope_dim) for i = 0, 1, ..., rope_dim//2 - 1
        freqs = 1.0 / (
            rope_theta ** (
                torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim
            )
        )  # (rope_dim//2,)

        # 为每个位置计算 m * theta_i
        positions = torch.arange(self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        freqs_complex = positions * freqs.unsqueeze(0)  # (max_seq_len, rope_dim//2)

        cos_cached = freqs_complex.cos().float()  # (max_seq_len, rope_dim//2)
        sin_cached = freqs_complex.sin().float()  # (max_seq_len, rope_dim//2)

        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """将输入沿最后一维分成两半，交换并取负前半部分。

        rotate_half([x1, x2]) = [-x2, x1]

        Args:
            x: shape (..., dim)，最后一维必须是偶数。

        Returns:
            shape (..., dim)
        """
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """对输入张量施加旋转位置编码。

        使用 torch.cat([cos, cos], dim=-1) 扩展 cos/sin（而非 repeat_interleave），
        以匹配 _rotate_half 中 (i, i+dim/2) 的配对方式。

        Args:
            x: 输入张量，shape (..., seq_len, rope_dim)。
            start_pos: 起始位置偏移（用于 KV Cache 增量解码场景）。

        Returns:
            RoPE 变换后的张量，shape 与输入相同。
        """
        seq_len = x.shape[-2]
        rope_dim = x.shape[-1]

        # 取出当前序列位置对应的 cos/sin
        cos = self.cos_cached[start_pos:start_pos + seq_len]  # (seq_len, rope_dim//2)
        sin = self.sin_cached[start_pos:start_pos + seq_len]

        # 使用 torch.cat 扩展（NOT repeat_interleave）
        cos_full = torch.cat([cos, cos], dim=-1)  # (seq_len, rope_dim)
        sin_full = torch.cat([sin, sin], dim=-1)

        # 根据输入维度调整广播形状
        if x.dim() == 4:
            # (B, n_heads, S, rope_dim) → 需要 (1, 1, S, rope_dim)
            cos_full = cos_full.unsqueeze(0).unsqueeze(0)
            sin_full = sin_full.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            # (B, S, rope_dim) → 需要 (1, S, rope_dim)
            cos_full = cos_full.unsqueeze(0)
            sin_full = sin_full.unsqueeze(0)
        else:
            raise ValueError(f"不支持的输入维度 {x.dim()}，需要 3 或 4 维。")

        cos_full = cos_full.to(x.dtype)
        sin_full = sin_full.to(x.dtype)

        return x * cos_full + self._rotate_half(x) * sin_full

    def _scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """缩放点积注意力。

        Attention(Q, K, V) = softmax(Q @ K^T / scale) @ V

        Args:
            q: 查询向量 (B, n_heads, S, head_dim)
            k: 键向量   (B, n_heads, S, head_dim)
            v: 值向量   (B, n_heads, S, head_dim)
            mask: 布尔掩码，True 的位置将被设为 -inf。

        Returns:
            (B, n_heads, S, head_dim)
        """
        scores = (q @ k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        return attn_weights @ v

    @staticmethod
    def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """创建因果注意力掩码。

        上三角掩码：mask[i][j] = True 当 j > i（未来位置不可见）。
        预留 batch 和 heads 维度，形状为 (1, 1, seq_len, seq_len)。

        Args:
            seq_len: 序列长度。
            device: 张量设备。

        Returns:
            布尔掩码 (1, 1, seq_len, seq_len)
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(
        self, x: torch.Tensor, use_causal_mask: bool = False,
    ) -> torch.Tensor:
        """MLA 前向传播。

        Args:
            x: 输入张量 (B, S, dim)
            use_causal_mask: 是否使用因果掩码（Decoder 自回归场景）。

        Returns:
            输出张量 (B, S, dim)
        """
        B, S, _ = x.shape

        # 1. Q 投影并拆分头部
        q = self.w_q(x)  # (B, S, n_heads * head_dim)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # (B, n_heads, S, head_dim)

        # 2. KV 压缩：w_kv_a 输出包含潜在向量和解耦 RoPE 键
        kv_a = self.w_kv_a(x)  # (B, S, kv_lora_rank + qk_rope_head_dim)
        kv_latent = kv_a[:, :, :self.kv_lora_rank]         # (B, S, kv_lora_rank)
        k_rope_raw = kv_a[:, :, self.kv_lora_rank:]         # (B, S, qk_rope_head_dim)

        # 3. 对 k_rope_raw 施加 RoPE，然后扩展到所有头
        k_rope = self._apply_rope(k_rope_raw)  # (B, S, qk_rope_head_dim)
        k_rope = k_rope.unsqueeze(1).expand(
            B, self.n_heads, S, self.qk_rope_head_dim,
        )  # (B, n_heads, S, qk_rope_head_dim)

        # 4. 上投影 K 和 V
        k = self.w_k_b(kv_latent)  # (B, S, n_heads * head_dim)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # (B, n_heads, S, head_dim)

        v = self.w_v_b(kv_latent)  # (B, S, n_heads * head_dim)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # (B, n_heads, S, head_dim)

        # 5. 对 Q 的最后 qk_rope_head_dim 维施加 RoPE
        q_nope = q[:, :, :, :-self.qk_rope_head_dim]
        q_rope = q[:, :, :, -self.qk_rope_head_dim:]
        q_rope = self._apply_rope(q_rope)
        q = torch.cat([q_nope, q_rope], dim=-1)
        # (B, n_heads, S, head_dim)

        # 6. 将 K 的最后 qk_rope_head_dim 维替换为共享 RoPE 键
        k_nope = k[:, :, :, :-self.qk_rope_head_dim]
        k = torch.cat([k_nope, k_rope], dim=-1)
        # (B, n_heads, S, head_dim)

        # 7. 缩放点积注意力
        mask = None
        if use_causal_mask:
            mask = self._create_causal_mask(S, x.device)

        out = self._scaled_dot_product_attention(q, k, v, mask)
        # (B, n_heads, S, head_dim)

        # 8. 合并头部并输出投影
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        # (B, S, n_heads * head_dim)
        return self.w_o(out)

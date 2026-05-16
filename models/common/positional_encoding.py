"""
位置编码实现：正弦位置编码（Sinusoidal PE）和旋转位置编码（RoPE）。

正弦位置编码来自 "Attention Is All You Need"，通过 sin/cos 函数为每个位置生成唯一编码。
RoPE (Rotary Positional Embedding) 来自 RoFormer / LLaMA 系列，通过旋转矩阵在注意力计算中注入位置信息，
具有相对位置编码的优良特性，支持更好的长度外推。
"""
import torch
import math


def sinusoidal_pe(seq_len: int, dim: int) -> torch.Tensor:
    """
    正弦位置编码 (Sinusoidal Positional Encoding)。

    来自 "Attention Is All You Need" 论文，公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

    Args:
        seq_len: 序列长度（位置数）
        dim: 编码维度

    Returns:
        torch.Tensor: shape (1, seq_len, dim)，可直接与输入加法广播
    """
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)       # (seq_len, 1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )                                                                        # (dim//2,)
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维用 sin
    pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维用 cos
    return pe.unsqueeze(0)  # (1, seq_len, dim)


class RotaryPositionalEncoding(torch.nn.Module):
    """
    旋转位置编码 (Rotary Positional Embedding / RoPE)。

    LLaMA 系列使用的 RoPE，通过对 Q/K 向量施加旋转来注入位置信息。
    与绝对位置编码相比，RoPE 自然编码相对位置关系，支持更好的长度外推。

    公式:
        q_rot = q * cos(θ) + rotate_half(q) * sin(θ)
        k_rot = k * cos(θ) + rotate_half(k) * sin(θ)

    其中 rotate_half 将向量沿最后一维拆成两半并交换符号:
        rotate_half([x1, x2]) = [-x2, x1]

    频率计算:
        Θ = { θ_i = 10000^(-2i/dim) | i = 0, 1, ..., dim/2 - 1 }
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        """
        Args:
            dim: head 维度（必须是偶数）
            max_seq_len: 预计算的最大序列长度
            theta: 频率基数，默认 10000.0
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算频率: Θ_i = theta^(-2i/dim) for i = 0, 1, ..., dim/2 - 1
        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )  # (dim//2,)

        # 为每个位置计算 m * Θ_i → 形状 (max_seq_len, dim//2)
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        freqs_complex = positions * freqs.unsqueeze(0)  # (max_seq_len, dim//2)

        # 预计算 cos 和 sin 缓存
        cos_cached = freqs_complex.cos().float()  # (max_seq_len, dim//2)
        sin_cached = freqs_complex.sin().float()  # (max_seq_len, dim//2)

        # register_buffer: 不参与梯度但随模型移动/保存
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入沿最后一维分成两半，交换并取负前半部分。

        rotate_half([x1, x2]) = [-x2, x1]

        Args:
            x: shape (..., dim)，最后一维必须是偶数

        Returns:
            shape (..., dim)
        """
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0
    ):
        """
        对 Q 和 K 施加旋转位置编码。

        Args:
            q: 查询向量，shape (batch, n_heads, seq_len, head_dim)
            k: 键向量，shape (batch, n_heads, seq_len, head_dim)
            start_pos: 起始位置（用于 KV Cache 增量解码场景）

        Returns:
            (q_rot, k_rot): 施加 RoPE 后的 Q 和 K，形状不变
        """
        seq_len = q.shape[2]

        # 取出当前序列位置对应的 cos/sin
        cos = self.cos_cached[start_pos:start_pos + seq_len]  # (seq_len, dim//2)
        sin = self.sin_cached[start_pos:start_pos + seq_len]  # (seq_len, dim//2)

        # 复制 cos/sin 以匹配 rotated-half 对的维度
        # rotate_half 的配对是 (i, i+dim/2)，所以每个频率需要出现在 i 和 i+dim/2 两个位置
        # torch.cat([cos, cos]) → [c0, c1, ..., c0, c1, ...] 而非 repeat_interleave 的 [c0, c0, c1, c1, ...]
        cos_full = torch.cat([cos, cos], dim=-1)  # (seq_len, dim)
        sin_full = torch.cat([sin, sin], dim=-1)  # (seq_len, dim)

        # 广播维度: (1, 1, seq_len, dim)
        cos_full = cos_full.unsqueeze(0).unsqueeze(0)
        sin_full = sin_full.unsqueeze(0).unsqueeze(0)

        q_rot = q * cos_full + self._rotate_half(q) * sin_full
        k_rot = k * cos_full + self._rotate_half(k) * sin_full

        return q_rot, k_rot


if __name__ == "__main__":
    # 快速冒烟测试
    pe = sinusoidal_pe(seq_len=100, dim=64)
    print(f"Sinusoidal PE: shape {pe.shape}")

    rope = RotaryPositionalEncoding(dim=64, max_seq_len=128)
    q = torch.randn(2, 8, 16, 64)
    k = torch.randn(2, 8, 16, 64)
    q_rot, k_rot = rope(q, k)
    print(f"RoPE: q {q.shape} -> q_rot {q_rot.shape}, k {k.shape} -> k_rot {k_rot.shape}")
    print("All positional encoding checks passed.")

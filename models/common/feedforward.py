"""
前馈网络手写实现：标准 FFN 和 SwiGLU FFN。

标准 FFN:  x -> Linear1 -> GELU -> Linear2 -> output
SwiGLU FFN: x -> (SiLU(x·W_gate) * (x·W_up)) · W_down
            其中 * 表示逐元素乘法，SiLU 作为门控激活函数

SwiGLU 是 LLaMA 和 DeepSeek 系列使用的 FFN 变体，相比标准 FFN 效果好且更稳定。
"""
import torch
from models.common.activation import gelu, silu


class FFN(torch.nn.Module):
    """标准两层前馈网络，Transformer Encoder/Decoder 中使用。FFN(x) = GELU(x @ W1 + b1) @ W2 + b2"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim)   # 升维投影
        self.w2 = torch.nn.Linear(hidden_dim, dim)    # 降维投影
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w1(x)          # (B, S, hidden_dim)
        h = gelu(h)             # GELU 激活
        h = self.dropout(h)
        return self.w2(h)       # (B, S, dim)


class SwiGLUFFN(torch.nn.Module):
    """
    SwiGLU 前馈网络。LLaMA 系列和 DeepSeek V3 使用的 FFN 结构。
    SwiGLU(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
    相比标准 FFN 多了一个 gate 投影，参数量增加 50%（3W² vs 2W²），但效果显著更好。
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = torch.nn.Linear(dim, hidden_dim, bias=False)  # 门控投影
        self.w_up = torch.nn.Linear(dim, hidden_dim, bias=False)    # 值投影
        self.w_down = torch.nn.Linear(hidden_dim, dim, bias=False)  # 输出投影
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w_gate(x))    # SiLU 门控
        up = self.w_up(x)              # 值
        h = gate * up                  # 逐元素门控
        h = self.dropout(h)
        return self.w_down(h)


if __name__ == "__main__":
    x = torch.randn(2, 8, 64)
    ffn = FFN(64, 256)
    print(f"FFN:        {x.shape} -> {ffn(x).shape}")

    swiglu = SwiGLUFFN(64, 256)
    print(f"SwiGLU FFN: {x.shape} -> {swiglu(x).shape}")

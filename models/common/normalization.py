"""
归一化层手写实现：LayerNorm 和 RMSNorm。

LayerNorm: y = (x - μ) / √(σ² + ε) * γ + β
RMSNorm:   y = x / RMS(x) * γ, 其中 RMS(x) = √(mean(x²) + ε)

RMSNorm 是 LLaMA 系列使用的归一化方式，相比 LayerNorm 去掉了中心化步骤（不需要计算均值），
计算效率更高。
"""
import torch


class LayerNorm(torch.nn.Module):
    """
    标准 Layer Normalization。
    对输入最后一维做归一化：减均值、除标准差，再做可学习的 affine 变换。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))   # γ: 缩放参数
        self.bias = torch.nn.Parameter(torch.zeros(dim))    # β: 平移参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # 有偏估计
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization。
    LLaMA 使用的归一化方式，不需要计算均值，比 LayerNorm 快约 10-15%。

    公式: y = x / √(mean(x²) + ε) * γ
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))   # γ: 可学习缩放参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


if __name__ == "__main__":
    x = torch.randn(2, 4, 8)
    ln = LayerNorm(8)
    print(f"LayerNorm: input shape {x.shape} -> output shape {ln(x).shape}")

    rms = RMSNorm(8)
    print(f"RMSNorm:  input shape {x.shape} -> output shape {rms(x).shape}")

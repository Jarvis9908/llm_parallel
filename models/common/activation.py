"""
激活函数手写实现。
参考：GELU (Gaussian Error Linear Unit), SiLU/Swish, ReLU
"""
import torch
import math


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    GELU 激活函数（tanh 近似版本）
    GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    与原始 Gaussian CDF 形式的误差 < 0.1%，但计算更快。
    """
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
    return 0.5 * x * (1.0 + torch.tanh(inner))


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU (Sigmoid Linear Unit)，也叫 Swish
    SiLU(x) = x * σ(x)
    在 LLaMA 系列中用作 FFN 的激活函数。
    """
    return x * torch.sigmoid(x)


def relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU(x) = max(0, x)"""
    return torch.maximum(x, torch.zeros_like(x))


if __name__ == "__main__":
    x = torch.randn(2, 4)
    assert torch.allclose(gelu(x), torch.nn.functional.gelu(x, approximate='tanh'), atol=1e-5)
    assert torch.allclose(silu(x), torch.nn.functional.silu(x), atol=1e-5)
    print("All activation checks passed.")

"""
Transformer 超参数配置模块。

使用 Python dataclass 定义所有 Transformer 相关的超参数，
包括模型维度、注意力头数、层数、FFN 维度等。
"""
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Transformer 超参数配置

    Attributes:
        vocab_size: 词汇表大小，默认 30000
        dim: 模型维度 (d_model)，默认 512
        n_heads: 多头注意力头数，默认 8
        n_layers: Encoder 和 Decoder 各 N 层，默认 6
        ff_hidden_dim: FFN 隐藏层维度，默认 2048
        max_seq_len: 最大序列长度，默认 512
        dropout: Dropout 比例，默认 0.1
        eps: LayerNorm epsilon，默认 1e-5
    """
    vocab_size: int = 30000
    dim: int = 512               # d_model
    n_heads: int = 8
    n_layers: int = 6            # Encoder 和 Decoder 各 N 层
    ff_hidden_dim: int = 2048    # FFN 隐藏层维度
    max_seq_len: int = 512
    dropout: float = 0.1
    eps: float = 1e-5            # LayerNorm epsilon

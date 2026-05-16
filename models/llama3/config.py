"""
LLaMA3 模型配置模块。

使用 Python dataclass 定义 LLaMA3 的所有超参数，
包括模型维度、注意力头数、GQA 配置、层数、SwiGLU FFN 维度等。
"""
from dataclasses import dataclass


@dataclass
class LLaMA3Config:
    """LLaMA3 超参数配置。

    LLaMA3 使用 RoPE 位置编码 + RMSNorm 归一化 + GQA 注意力 + SwiGLU FFN。

    Attributes:
        vocab_size: 词汇表大小，默认 32000
        dim: 模型维度 (hidden_size)，默认 512
        n_heads: Query 注意力头数，默认 8
        n_kv_heads: KV 注意力头数（GQA），默认 4
        n_layers: Transformer Block 层数，默认 8
        ff_hidden_dim: SwiGLU FFN 中间层维度，默认 1376
        max_seq_len: 最大序列长度，默认 2048
        dropout: Dropout 比例（LLaMA3 通常不使用），默认 0.0
        eps: RMSNorm epsilon，默认 1e-6
        rope_theta: RoPE 频率基数，默认 10000.0
    """
    vocab_size: int = 32000
    dim: int = 512               # hidden_size
    n_heads: int = 8             # query heads
    n_kv_heads: int = 4          # KV heads (GQA)
    n_layers: int = 8
    ff_hidden_dim: int = 1376    # SwiGLU intermediate dim
    max_seq_len: int = 2048
    dropout: float = 0.0         # LLaMA3 typically no dropout
    eps: float = 1e-6
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度。"""
        return self.dim // self.n_heads

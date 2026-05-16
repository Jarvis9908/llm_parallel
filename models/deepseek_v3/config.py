"""DeepSeek V3 模型配置。"""
from dataclasses import dataclass


@dataclass
class DeepSeekV3Config:
    """DeepSeek V3 架构配置。

    Attributes:
        vocab_size: 词表大小
        dim: 模型隐藏维度
        n_heads: 注意力头数
        n_layers: Transformer 层数
        max_seq_len: 最大序列长度
        eps: LayerNorm/RMSNorm 的 epsilon
        kv_lora_rank: KV 潜在压缩维度（MLA 核心参数）
        qk_rope_head_dim: 解耦 RoPE 部分的维度
        n_routed_experts: 路由专家数量
        n_shared_experts: 共享专家数量
        n_activated_experts: 每个 token 激活的专家数（top_k）
        moe_intermediate_dim: MoE 前馈网络中间维度
        rope_theta: RoPE 频率基数
    """

    vocab_size: int = 32000
    dim: int = 512
    n_heads: int = 8
    n_layers: int = 8
    max_seq_len: int = 2048
    eps: float = 1e-6

    # MLA (Multi-head Latent Attention)
    kv_lora_rank: int = 256      # KV 潜在压缩维度
    qk_rope_head_dim: int = 32   # RoPE 部分维度（解耦 RoPE）

    # MoE (Mixture of Experts)
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2  # top_k
    moe_intermediate_dim: int = 512

    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度。"""
        return self.dim // self.n_heads

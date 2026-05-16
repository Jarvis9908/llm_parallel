"""
DeepSeek V3 完整模型实现。

架构: TokenEmbedding → [DeepSeekV3Block × N_layers] → RMSNorm → lm_head

DeepSeekV3Block 使用 Pre-Norm 残差结构，结合 MLA 注意力和 MoE 前馈:
    x = x + MLA(RMSNorm(x))
    x = x + MoE(RMSNorm(x))

其中 MLA 负责吸收输入上下文，MoE 负责以条件计算的方式建模知识。
"""
import torch
import torch.nn as nn

from models.deepseek_v3.config import DeepSeekV3Config
from models.deepseek_v3.mla import MultiHeadLatentAttention
from models.deepseek_v3.moe import MoELayer
from models.common.normalization import RMSNorm
from models.common.embeddings import TokenEmbedding


class DeepSeekV3Block(nn.Module):
    """DeepSeek V3 的单个 Transformer Block（Pre-Norm 风格）。

    包含一个 MLA 自注意力子层和一个 MoE 前馈子层，
    每个子层前都有 RMSNorm 预归一化，子层后加残差连接。

    结构:
        x = x + MLA(RMSNorm(x))   ← 多头潜在注意力
        x = x + MoE(RMSNorm(x))   ← 专家混合前馈
    """

    def __init__(self, config: DeepSeekV3Config):
        """初始化 DeepSeek V3 Block。

        Args:
            config: DeepSeekV3Config 配置对象。
        """
        super().__init__()

        # Pre-Norm: 两个 RMSNorm
        self.norm1 = RMSNorm(config.dim, eps=config.eps)  # 注意力前归一化
        self.norm2 = RMSNorm(config.dim, eps=config.eps)  # MoE 前归一化

        # MLA 自注意力
        self.attn = MultiHeadLatentAttention(config)

        # MoE 前馈网络
        self.moe = MoELayer(config)

    def forward(self, x: torch.Tensor, use_causal_mask: bool = False) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量 (B, S, dim)
            use_causal_mask: 是否使用因果掩码（自回归场景）

        Returns:
            输出张量 (B, S, dim)
        """
        # MLA 自注意力 + 残差连接
        x = x + self.attn(self.norm1(x), use_causal_mask=use_causal_mask)

        # MoE 前馈 + 残差连接
        x = x + self.moe(self.norm2(x))

        return x


class DeepSeekV3Model(nn.Module):
    """DeepSeek V3 基础模型（不含 LM Head）。

    将 token id 序列映射为隐状态序列，包含 Token Embedding、
    N 层 DeepSeekV3Block 和最终 RMSNorm。
    """

    def __init__(self, config: DeepSeekV3Config):
        """初始化 DeepSeek V3 模型。

        Args:
            config: DeepSeekV3Config 配置对象。
        """
        super().__init__()
        self.config = config

        self.embed = TokenEmbedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            DeepSeekV3Block(config) for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, eps=config.eps)

    def forward(
        self,
        tokens: torch.LongTensor,
        use_causal_mask: bool = True,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            tokens: token id 序列 (B, S)
            use_causal_mask: 是否使用因果掩码

        Returns:
            隐状态序列 (B, S, dim)
        """
        x = self.embed(tokens)  # (B, S, dim)

        for layer in self.layers:
            x = layer(x, use_causal_mask=use_causal_mask)

        x = self.norm(x)
        return x


class DeepSeekV3ForCausalLM(nn.Module):
    """DeepSeek V3 因果语言模型（含 LM Head）。

    在 DeepSeekV3Model 之上添加线性投影层（LM Head），
    将隐状态映射到词汇表空间以输出 logits，
    并提供自回归文本生成功能（简化版，无 KV Cache）。
    """

    def __init__(self, config: DeepSeekV3Config):
        """初始化因果语言模型。

        Args:
            config: DeepSeekV3Config 配置对象。
        """
        super().__init__()
        self.config = config
        self.model = DeepSeekV3Model(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.LongTensor,
        use_causal_mask: bool = True,
    ) -> torch.Tensor:
        """前向传播，输出 logits。

        Args:
            tokens: token id 序列 (B, S)
            use_causal_mask: 是否使用因果掩码

        Returns:
            logits: (B, S, vocab_size)
        """
        h = self.model(tokens, use_causal_mask=use_causal_mask)
        return self.lm_head(h)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float) -> torch.LongTensor:
        """从 logits 中采样下一个 token。

        Args:
            logits: (B, 1, vocab_size)
            temperature: 温度参数，0 表示贪心解码

        Returns:
            next_token: (B, 1)
        """
        if temperature == 0:
            return logits.argmax(dim=-1)  # (B, 1)

        logits = logits / temperature
        probs = torch.softmax(logits.squeeze(1), dim=-1)  # (B, V)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
        return next_token

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        """自回归文本生成（简化版，无 KV Cache）。

        每次迭代将完整的历史序列重新输入模型，
        取最后一个位置的 logits 采样下一个 token。

        Args:
            prompt: 提示 token id (B, prompt_len)
            max_new_tokens: 最多生成的新 token 数
            temperature: 采样温度，0 表示贪心解码

        Returns:
            生成序列 (B, prompt_len + max_new_tokens)
        """
        generated = prompt.clone()

        for _ in range(max_new_tokens):
            # 完整历史前向传播
            logits = self.forward(generated, use_causal_mask=True)  # (B, S, vocab_size)
            next_logits = logits[:, -1:, :]  # (B, 1, vocab_size)
            next_token = self._sample(next_logits, temperature)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated


if __name__ == "__main__":
    # 快速冒烟测试
    config = DeepSeekV3Config(
        dim=128, n_heads=4, n_layers=2, kv_lora_rank=64, qk_rope_head_dim=16,
        n_routed_experts=4, n_shared_experts=1, n_activated_experts=2,
        moe_intermediate_dim=256, max_seq_len=64,
    )

    # DeepSeekV3Block
    block = DeepSeekV3Block(config)
    x = torch.randn(2, 8, 128)
    out = block(x)
    print(f"DeepSeekV3Block: {x.shape} -> {out.shape}")

    # DeepSeekV3Model
    model = DeepSeekV3Model(config)
    tokens = torch.randint(0, config.vocab_size, (2, 16))
    h = model(tokens)
    print(f"DeepSeekV3Model: {tokens.shape} -> {h.shape}")

    # DeepSeekV3ForCausalLM
    lm = DeepSeekV3ForCausalLM(config)
    logits = lm(tokens)
    print(f"DeepSeekV3ForCausalLM: {tokens.shape} -> {logits.shape}")

    # Generate
    lm.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    generated = lm.generate(prompt, max_new_tokens=6, temperature=1.0)
    print(f"Generate: prompt {prompt.shape} -> generated {generated.shape}")

    print("All DeepSeek V3 model checks passed.")
    print(f"Total params: {sum(p.numel() for p in lm.parameters()):,}")

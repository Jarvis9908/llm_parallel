"""DeepSeek V3 MLA 模块测试。"""
import torch
import sys
sys.path.insert(0, '.')
from models.deepseek_v3.config import DeepSeekV3Config
from models.deepseek_v3.mla import MultiHeadLatentAttention


class TestMLA:
    """Multi-head Latent Attention 测试套件。"""

    def test_shape(self):
        """输出形状应与输入形状一致。"""
        config = DeepSeekV3Config(
            dim=128, n_heads=4, kv_lora_rank=64, qk_rope_head_dim=16,
        )
        mla = MultiHeadLatentAttention(config)
        x = torch.randn(2, 16, 128)
        out = mla(x)
        assert out.shape == x.shape

    def test_fewer_params_than_mha(self):
        """MLA 参数量应少于标准 MHA（得益于 KV 低秩压缩）。"""
        config = DeepSeekV3Config(
            dim=128, n_heads=4, kv_lora_rank=64, qk_rope_head_dim=16,
        )
        mla = MultiHeadLatentAttention(config)
        mla_params = sum(p.numel() for p in mla.parameters())

        from models.common.attention import MultiHeadAttention
        mha = MultiHeadAttention(dim=128, n_heads=4)
        mha_params = sum(p.numel() for p in mha.parameters())

        assert mla_params < mha_params

    def test_causal_mask(self):
        """因果掩码应阻止位置 i 关注位置 j > i。"""
        config = DeepSeekV3Config(
            dim=128, n_heads=4, kv_lora_rank=64, qk_rope_head_dim=16,
        )
        mla = MultiHeadLatentAttention(config)
        mla.eval()
        x = torch.randn(1, 4, 128)
        out = mla(x, use_causal_mask=True)
        # 修改位置 3 的输入不应影响位置 1 的输出（因果掩码）
        x2 = x.clone()
        x2[0, 3] = 999.0
        out2 = mla(x2, use_causal_mask=True)
        assert torch.allclose(out[0, 1], out2[0, 1], atol=1e-4)

    def test_backward(self):
        """梯度应通过 MLA 正确反向传播。"""
        config = DeepSeekV3Config(
            dim=128, n_heads=4, kv_lora_rank=64, qk_rope_head_dim=16,
        )
        mla = MultiHeadLatentAttention(config)
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = mla(x)
        out.sum().backward()
        assert x.grad is not None


from models.deepseek_v3.moe import Router, SharedExpert, RoutedExpert, MoELayer


class TestRouter:
    """Router 测试套件。"""

    def test_shape(self):
        """Router 输出 scores 和 indices 的形状应为 (B, S, top_k)。"""
        router = Router(dim=128, n_experts=8, top_k=2)
        x = torch.randn(2, 16, 128)
        scores, indices = router(x)
        assert scores.shape == (2, 16, 2)
        assert indices.shape == (2, 16, 2)

    def test_top_k_indices(self):
        """Router 输出的专家索引应在有效范围内。"""
        router = Router(dim=128, n_experts=8, top_k=2)
        x = torch.randn(2, 16, 128)
        _, indices = router(x)
        assert indices.min() >= 0
        assert indices.max() < 8


class TestMoELayer:
    """MoE Layer 测试套件。"""

    def test_shape(self):
        """MoE 层输出形状应与输入形状一致。"""
        config = DeepSeekV3Config(
            dim=128, n_routed_experts=4, n_shared_experts=1,
            n_activated_experts=2, moe_intermediate_dim=256
        )
        moe = MoELayer(config)
        x = torch.randn(2, 8, 128)
        out = moe(x)
        assert out.shape == x.shape

    def test_backward(self):
        """梯度应通过 MoE 层正确反向传播。"""
        config = DeepSeekV3Config(
            dim=128, n_routed_experts=4, n_shared_experts=1,
            n_activated_experts=2, moe_intermediate_dim=256
        )
        moe = MoELayer(config)
        x = torch.randn(2, 8, 128, requires_grad=True)
        out = moe(x)
        out.sum().backward()
        assert x.grad is not None


from models.deepseek_v3.model import DeepSeekV3Block, DeepSeekV3Model, DeepSeekV3ForCausalLM


class TestDeepSeekV3Block:
    """DeepSeek V3 Transformer Block 测试套件。"""

    def test_shape(self):
        """输出形状应与输入形状一致。"""
        config = DeepSeekV3Config(
            dim=128, n_heads=4, kv_lora_rank=64, qk_rope_head_dim=16,
            n_routed_experts=4, n_shared_experts=1, n_activated_experts=2,
            moe_intermediate_dim=256
        )
        block = DeepSeekV3Block(config)
        x = torch.randn(2, 8, 128)
        out = block(x)
        assert out.shape == x.shape


class TestDeepSeekV3Model:
    """DeepSeek V3 基础模型测试套件。"""

    def test_shape(self):
        """输出形状应为 (B, S, dim)。"""
        config = DeepSeekV3Config(
            dim=128, n_heads=4, n_layers=2, kv_lora_rank=64, qk_rope_head_dim=16,
            n_routed_experts=4, n_shared_experts=1, n_activated_experts=2,
            moe_intermediate_dim=256
        )
        model = DeepSeekV3Model(config)
        tokens = torch.randint(0, config.vocab_size, (2, 16))
        out = model(tokens)
        assert out.shape == (2, 16, config.dim)


class TestDeepSeekV3ForCausalLM:
    """DeepSeek V3 因果语言模型测试套件。"""

    def test_shape(self):
        """logits 输出形状应为 (B, S, vocab_size)。"""
        config = DeepSeekV3Config(
            dim=128, n_heads=4, n_layers=2, kv_lora_rank=64, qk_rope_head_dim=16,
            n_routed_experts=4, n_shared_experts=1, n_activated_experts=2,
            moe_intermediate_dim=256, vocab_size=1000
        )
        model = DeepSeekV3ForCausalLM(config)
        tokens = torch.randint(0, 1000, (2, 8))
        logits = model(tokens)
        assert logits.shape == (2, 8, 1000)

    def test_generate(self):
        """自回归生成应产生正确长度的序列。"""
        config = DeepSeekV3Config(
            dim=128, n_heads=4, n_layers=2, kv_lora_rank=64, qk_rope_head_dim=16,
            n_routed_experts=4, n_shared_experts=1, n_activated_experts=2,
            moe_intermediate_dim=256, vocab_size=100, max_seq_len=64
        )
        model = DeepSeekV3ForCausalLM(config)
        model.eval()
        prompt = torch.randint(0, 100, (1, 4))
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=6, temperature=1.0)
        assert generated.shape[1] == 10

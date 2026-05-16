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

import torch
import sys
sys.path.insert(0, '.')
from models.common.attention import MultiHeadAttention, GroupedQueryAttention


class TestMultiHeadAttention:
    def test_shape(self):
        mha = MultiHeadAttention(dim=64, n_heads=8)
        x = torch.randn(2, 16, 64)
        out = mha(x)
        assert out.shape == x.shape

    def test_causal_mask(self):
        mha = MultiHeadAttention(dim=64, n_heads=8)
        mha.eval()
        x = torch.randn(1, 4, 64)
        out_causal = mha(x, use_causal_mask=True)
        x2 = x.clone()
        x2[0, 3] = 999.0
        out2 = mha(x2, use_causal_mask=True)
        assert torch.allclose(out_causal[0, 1], out2[0, 1], atol=1e-4)

    def test_backward(self):
        mha = MultiHeadAttention(dim=64, n_heads=8)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = mha(x)
        out.sum().backward()
        assert x.grad is not None


class TestGroupedQueryAttention:
    def test_shape_mha_mode(self):
        gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=8)
        x = torch.randn(2, 16, 64)
        out = gqa(x)
        assert out.shape == x.shape

    def test_shape_gqa_mode(self):
        gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=2)
        x = torch.randn(2, 16, 64)
        out = gqa(x)
        assert out.shape == x.shape

    def test_shape_mqa_mode(self):
        gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=1)
        x = torch.randn(2, 16, 64)
        out = gqa(x)
        assert out.shape == x.shape

    def test_fewer_kv_params(self):
        mha = MultiHeadAttention(dim=64, n_heads=8)
        gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=2)
        mha_params = sum(p.numel() for p in mha.parameters())
        gqa_params = sum(p.numel() for p in gqa.parameters())
        assert gqa_params < mha_params

    def test_causal_mask(self):
        gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=2)
        gqa.eval()
        x = torch.randn(1, 4, 64)
        out = gqa(x, use_causal_mask=True)
        x2 = x.clone()
        x2[0, 3] = 999.0
        out2 = gqa(x2, use_causal_mask=True)
        assert torch.allclose(out[0, 1], out2[0, 1], atol=1e-4)

    def test_backward(self):
        gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=2)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = gqa(x)
        out.sum().backward()
        assert x.grad is not None

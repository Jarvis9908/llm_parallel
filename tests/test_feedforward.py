import torch
import sys
sys.path.insert(0, '.')
from models.common.feedforward import FFN, SwiGLUFFN


class TestFFN:
    def test_shape(self):
        ffn = FFN(dim=64, hidden_dim=256)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_backward(self):
        ffn = FFN(dim=64, hidden_dim=256)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        for name, p in ffn.named_parameters():
            assert p.grad is not None, f"{name} should have grad"


class TestSwiGLUFFN:
    def test_shape(self):
        ffn = SwiGLUFFN(dim=64, hidden_dim=256)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_three_matrices(self):
        """SwiGLU 需要 3 个投影矩阵（w1, w2, w3），比标准 FFN 多一个"""
        ffn = SwiGLUFFN(dim=64, hidden_dim=256)
        param_count = sum(p.numel() for p in ffn.parameters())
        # 3 * (64 * 256) + 2 * bias_of_256: gate(256) + up(256) + down(256) = 3 * 256 ≈ 768 biases, but biases=False
        # With bias=False: 3 * (64*256) + 3*256 = 49920
        expected_approx = 3 * 64 * 256 + 3 * 256
        assert abs(param_count - expected_approx) < 1000

    def test_backward(self):
        ffn = SwiGLUFFN(dim=64, hidden_dim=256)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None

import torch
import sys
sys.path.insert(0, '.')
from models.common.activation import gelu, silu, relu

class TestActivations:
    def test_gelu_shape(self):
        x = torch.randn(2, 4, 8)
        out = gelu(x)
        assert out.shape == x.shape

    def test_gelu_approx(self):
        """GELU: x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))"""
        x = torch.tensor([0.0, 1.0, -1.0])
        out = gelu(x)
        expected = torch.nn.functional.gelu(x, approximate='tanh')
        assert torch.allclose(out, expected, atol=1e-5)

    def test_silu_shape(self):
        x = torch.randn(2, 4, 8)
        out = silu(x)
        assert out.shape == x.shape

    def test_silu_values(self):
        x = torch.tensor([0.0, 1.0, -1.0])
        out = silu(x)
        expected = torch.nn.functional.silu(x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_relu(self):
        x = torch.tensor([-1.0, 0.0, 2.0])
        out = relu(x)
        assert torch.equal(out, torch.tensor([0.0, 0.0, 2.0]))

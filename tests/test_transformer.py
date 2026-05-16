"""
Transformer 模块测试。

覆盖：
  - Task 2.1: EncoderLayer + Encoder（config, shape, backward, 多层堆叠）
  - Task 2.2: DecoderLayer + Decoder
  - Task 2.3: 完整 Transformer (Encoder + Decoder)
"""
import torch
import sys
sys.path.insert(0, '.')
from models.transformer.config import TransformerConfig
from models.transformer.encoder import EncoderLayer, Encoder
from models.transformer.decoder import DecoderLayer, Decoder
from models.transformer.model import Transformer


class TestEncoderLayer:
    """EncoderLayer 单元测试"""

    def test_shape(self):
        """验证输出形状与输入一致"""
        config = TransformerConfig(dim=64, n_heads=8, ff_hidden_dim=256)
        layer = EncoderLayer(config)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_backward(self):
        """验证梯度可以反向传播通过 EncoderLayer"""
        config = TransformerConfig(dim=64, n_heads=8, ff_hidden_dim=256)
        layer = EncoderLayer(config)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None, "Gradient should flow back to input"
        assert not torch.isnan(x.grad).any(), "Gradient should not contain NaN"


class TestEncoder:
    """Encoder 完整测试"""

    def test_shape(self):
        """验证输出形状与输入一致"""
        config = TransformerConfig(dim=64, n_heads=8, n_layers=4, ff_hidden_dim=256)
        encoder = Encoder(config)
        x = torch.randn(2, 32, 64)
        out = encoder(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_multiple_layers(self):
        """验证正确堆叠了指定数量的 EncoderLayer"""
        config = TransformerConfig(dim=64, n_heads=8, n_layers=6, ff_hidden_dim=256)
        encoder = Encoder(config)
        assert len(encoder.layers) == 6, f"Expected 6 layers, got {len(encoder.layers)}"

    def test_backward(self):
        """验证梯度可以反向传播通过整个 Encoder"""
        config = TransformerConfig(dim=64, n_heads=8, n_layers=2, ff_hidden_dim=256)
        encoder = Encoder(config)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = encoder(x)
        out.sum().backward()
        assert x.grad is not None, "Gradient should flow back to input"
        assert not torch.isnan(x.grad).any(), "Gradient should not contain NaN"

    def test_with_token_ids(self):
        """验证 Encoder 可以接收 token ids (LongTensor) 输入"""
        config = TransformerConfig(vocab_size=1000, dim=64, n_heads=8, n_layers=2, ff_hidden_dim=256)
        encoder = Encoder(config)
        token_ids = torch.randint(0, 1000, (2, 16))
        out = encoder(token_ids)
        assert out.shape == (2, 16, config.dim), \
            f"Expected (2, 16, {config.dim}), got {out.shape}"


class TestDecoderLayer:
    """DecoderLayer 单元测试"""

    def test_shape(self):
        """验证输出形状与输入一致"""
        config = TransformerConfig(dim=64, n_heads=8, ff_hidden_dim=256)
        layer = DecoderLayer(config)
        x = torch.randn(2, 16, 64)
        encoder_out = torch.randn(2, 20, 64)
        out = layer(x, encoder_out)
        assert out.shape == x.shape

    def test_causal_self_attn(self):
        """Decoder 的 self-attention 必须使用 causal mask"""
        config = TransformerConfig(dim=64, n_heads=8, ff_hidden_dim=256)
        layer = DecoderLayer(config)
        layer.eval()  # 关闭 dropout 避免随机性
        x = torch.randn(1, 4, 64)
        encoder_out = torch.randn(1, 8, 64)
        out = layer(x, encoder_out)
        x2 = x.clone()
        x2[0, 3] = 999.0
        out2 = layer(x2, encoder_out)
        assert torch.allclose(out[0, 1], out2[0, 1], atol=1e-4)

    def test_backward(self):
        """验证梯度可以反向传播通过 DecoderLayer"""
        config = TransformerConfig(dim=64, n_heads=8, ff_hidden_dim=256)
        layer = DecoderLayer(config)
        x = torch.randn(2, 16, 64, requires_grad=True)
        encoder_out = torch.randn(2, 20, 64)
        out = layer(x, encoder_out)
        out.sum().backward()
        assert x.grad is not None


class TestDecoder:
    """Decoder 完整测试"""

    def test_shape(self):
        """验证输出形状正确"""
        config = TransformerConfig(dim=64, n_heads=8, n_layers=4, ff_hidden_dim=256)
        decoder = Decoder(config)
        tgt = torch.randint(0, config.vocab_size, (2, 16))
        encoder_out = torch.randn(2, 32, 64)
        out = decoder(tgt, encoder_out)
        assert out.shape == (2, 16, 64)

    def test_backward(self):
        """验证梯度可以反向传播通过整个 Decoder"""
        config = TransformerConfig(dim=64, n_heads=8, n_layers=2, ff_hidden_dim=256)
        decoder = Decoder(config)
        tgt = torch.randint(0, config.vocab_size, (2, 16))
        encoder_out = torch.randn(2, 32, 64)
        out = decoder(tgt, encoder_out)
        out.sum().backward()


class TestTransformer:
    def test_shape(self):
        config = TransformerConfig(dim=64, n_heads=8, n_layers=2, ff_hidden_dim=256)
        model = Transformer(config)
        src = torch.randint(0, config.vocab_size, (2, 20))
        tgt = torch.randint(0, config.vocab_size, (2, 16))
        out = model(src, tgt)
        assert out.shape == (2, 16, config.vocab_size)

    def test_backward(self):
        config = TransformerConfig(dim=64, n_heads=8, n_layers=2, ff_hidden_dim=256)
        model = Transformer(config)
        src = torch.randint(0, config.vocab_size, (2, 20))
        tgt = torch.randint(0, config.vocab_size, (2, 16))
        out = model(src, tgt)
        loss = out.sum()
        loss.backward()
        assert model.encoder.layers[0].self_attn.w_q.weight.grad is not None

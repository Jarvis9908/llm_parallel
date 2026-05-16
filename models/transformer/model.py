"""
完整 Transformer 模型（Encoder-Decoder）。
用于序列到序列任务（如机器翻译）。
"""
import torch
from models.transformer.config import TransformerConfig
from models.transformer.encoder import Encoder
from models.transformer.decoder import Decoder


class Transformer(torch.nn.Module):
    """Encoder-Decoder Transformer。Encoder 处理源序列 → Decoder 自回归生成目标序列。"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.lm_head = torch.nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, src_ids: torch.LongTensor, tgt_ids: torch.LongTensor) -> torch.Tensor:
        """
        src_ids: (batch, src_seq_len)   tgt_ids: (batch, tgt_seq_len)
        returns: (batch, tgt_seq_len, vocab_size) logits
        """
        encoder_output = self.encoder(src_ids)           # (B, S_src, D)
        decoder_output = self.decoder(tgt_ids, encoder_output)  # (B, S_tgt, D)
        return self.lm_head(decoder_output)              # (B, S_tgt, V)


if __name__ == "__main__":
    config = TransformerConfig(vocab_size=1000, dim=256, n_heads=8, n_layers=3)
    model = Transformer(config)
    src = torch.randint(0, 1000, (2, 32))
    tgt = torch.randint(0, 1000, (2, 16))
    out = model(src, tgt)
    print(f"Transformer: src {src.shape} + tgt {tgt.shape} -> logits {out.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

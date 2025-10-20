import torch
from impl.config import GPTModelCfg
from impl.layer_normalization import LayerNorm
from compat.openai_compat_multiheadattn import OpenAICompatibleMultiHeadAttention
from compat.openai_compat_ffn import OpenAICompatibleFeedForward

class OpenAICompatibleTransformerBlock(torch.nn.Module):
    def __init__(self, cfg: GPTModelCfg):
        super().__init__()
        self.attn = OpenAICompatibleMultiHeadAttention(
            d_in=cfg.emb_dim, d_out=cfg.emb_dim, context_length=cfg.context_length,
            num_heads=cfg.n_heads, dropout=cfg.drop_rate, qkv_bias=cfg.qkv_bias
        )
        self.mlp = OpenAICompatibleFeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = torch.nn.Dropout(cfg.drop_rate)

    def forward(self, x) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
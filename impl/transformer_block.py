import torch
from impl.config import GPTModelCfg, get_currently_chosen_cfg
from impl.multi_head_attention import MultiHeadAttention
from impl.feed_forward_network import FeedForward
from impl.layer_normalization import LayerNorm

torch.set_default_device("cuda")

class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg: GPTModelCfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim, d_out=cfg.emb_dim, context_length=cfg.context_length,
            num_heads=cfg.n_heads, dropout=cfg.drop_rate, qkv_bias=cfg.qkv_bias
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = torch.nn.Dropout(cfg.drop_rate)

    def forward(self, x) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

def main():
    x = torch.rand(2, 4, 768, device="cuda")
    block = TransformerBlock(get_currently_chosen_cfg())
    output = block(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    print("\n------------------\nInput: ")
    print(x)
    print("\n------------------")

    print("\n------------------\nOutput: ")
    print(output)
    print("\n------------------")

if __name__ == "__main__":
    main()
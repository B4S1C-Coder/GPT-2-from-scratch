import torch
from impl.gelu_activation import GELU
from impl.config import GPTModelCfg

class FeedForward(torch.nn.Module):
    def __init__(self, cfg: GPTModelCfg):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim), # Expansion
            GELU(),                                        # Activation
            torch.nn.Linear(4 * cfg.emb_dim, cfg.emb_dim), # Contraction
        )
    
    def forward(self, x):
        return self.layers(x)
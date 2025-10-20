import torch
from impl.gelu_activation import GELU
from impl.config import GPTModelCfg

class OpenAICompatibleFeedForward(torch.nn.Module):
    def __init__(self, cfg: GPTModelCfg):
        super().__init__()
        self.c_fc = torch.nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim)
        self.gelu = GELU()
        self.c_proj = torch.nn.Linear(4 * cfg.emb_dim, cfg.emb_dim)
    
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
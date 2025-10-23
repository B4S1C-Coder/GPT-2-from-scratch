# This file contains the config and the model class for easy loading in Google Colab.
# You can also use this file as a self-contained access just to the implementation
# itself without any utility functions.

import torch
from collections import namedtuple

GPTModelCfg = namedtuple(
    'ScratchModelConfig',
    [
        'vocab_size',
        'context_length',
        'emb_dim',
        'n_heads',
        'n_layers',
        'drop_rate',
        'qkv_bias',
    ]
)

# OpenAI Compatible implementation of the GPTModel class
class OpenAICompatibleGPTModel(torch.nn.Module):
    def __init__(self, cfg: GPTModelCfg) -> None:
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = torch.nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = torch.nn.Dropout(cfg.drop_rate)

        self.trf_blocks = torch.nn.Sequential(
            *[OpenAICompatibleTransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = torch.nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
    
    def forward(self, in_idx) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# Custom implementation of GPT-2
class GPTModel(torch.nn.Module):
    def __init__(self, cfg: GPTModelCfg) -> None:
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = torch.nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = torch.nn.Dropout(cfg.drop_rate)

        self.trf_blocks = torch.nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = torch.nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
    
    def forward(self, in_idx) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

GPT_2_CFG_124M = GPTModelCfg(
    vocab_size     = 50257,
    context_length = 1024,
    emb_dim        = 768,
    n_heads        = 12,
    n_layers       = 12,
    drop_rate      = 0.1,
    qkv_bias       = False
)

OPENAI_GPT_2_CFG_124M = GPTModelCfg(
    vocab_size     = 50257,
    context_length = 1024,
    emb_dim        = 768,
    n_heads        = 12,
    n_layers       = 12,
    drop_rate      = 0.1,
    qkv_bias       = True
)

OPENAI_GPT_2_CFG_355M = GPTModelCfg(
    vocab_size     = 50257,
    context_length = 1024,
    emb_dim        = 1024,
    n_heads        = 16,
    n_layers       = 24,
    drop_rate      = 0.0,
    qkv_bias       = True
)

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class OpenAICompatibleFeedForward(torch.nn.Module):
    def __init__(self, cfg: GPTModelCfg):
        super().__init__()
        self.c_fc = torch.nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim)
        self.gelu = GELU()
        self.c_proj = torch.nn.Linear(4 * cfg.emb_dim, cfg.emb_dim)
    
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class OpenAICompatibleMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # Reduce projection dim to match desired output dim
        self.head_dim = d_out // num_heads

        # One big linear for Q, K, V combined
        self.c_attn = torch.nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.c_proj = torch.nn.Linear(d_out, d_out)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, n, _ = x.shape

        # Compute Q, K, V in one go
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_out, dim=2)

        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(2, 3)) / (self.head_dim ** 0.5)
        mask = self.mask[:n, :n].bool()
        att.masked_fill_(mask, -torch.inf)

        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_out)
        out = self.c_proj(out)
        return out
    
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
    
class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # if unbiased is true we divide by (n - 1) [i.e. denominator is n - 1] instead of n
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var)
        return self.scale * norm_x + self.shift

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # Reduce projection dim to match desired output dim
        self.head_dim = d_out // num_heads

        self.w_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        # Linear layer to combine head outputs
        self.out_proj = torch.nn.Linear(d_out, d_out)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.w_k(x) # shape: (b, num_tokens, d_out)
        queries = self.w_q(x)
        values = self.w_v(x)

        # Implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # This is done to group by heads instead of the initial grouping by token. This is for
        # computational purposes.
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vectors = (attention_weights @ values).transpose(1, 2)
        context_vectors = context_vectors.contiguous().view(b, num_tokens, self.d_out)
        context_vectors = self.out_proj(context_vectors)

        return context_vectors

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
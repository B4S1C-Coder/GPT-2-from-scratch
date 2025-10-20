import torch

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
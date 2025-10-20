import torch

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

def main():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89, 0.55, 0.87, 0.66],  # Row 1
        [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],  # Row 2
        [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]]  # Row 3
    )

    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape) 

    batch_size, context_length, d_in = batch.shape
    d_out = 6
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

if __name__ == "__main__":
    main()
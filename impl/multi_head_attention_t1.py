import torch
from impl.causal_attention import CausalAttention

print("Defualt torch device: ", torch.get_default_device())
torch.set_default_device("cuda")
print("Default device set to: ", torch.get_default_device())

class MultiHeadAttentionWrapper(torch.nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])
    
    def forward(self, x):
        return torch.cat([ head(x) for head in self.heads ], dim=-1)

def main():
    inputs = torch.tensor([
        [0.43, 0.15, 0.89], # Your    (X^1)
        [0.55, 0.87, 0.66], # journey (X^2)
        [0.57, 0.85, 0.64], # starts  (X^3)
        [0.22, 0.58, 0.33], # with    (X^4)
        [0.77, 0.25, 0.10], # one     (X^5)
        [0.05, 0.80, 0.55]  # step    (X^6)
    ])

    d_in = inputs.shape[1]
    d_out = 2
    batch = torch.stack((inputs, inputs), dim = 0)
    context_length = batch.shape[1]

    mhaw = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, 2)
    context_vectors = mhaw(batch)

    print(context_vectors)
    print(context_vectors.shape)

if __name__ == "__main__":
    main()
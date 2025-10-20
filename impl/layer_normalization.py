import torch

def illustration():
    # 2 batches, containing 5 points (x1, x2, x3, x4, x5) each
    batch_example = torch.randn(2, 5)

    layer = torch.nn.Sequential(
        torch.nn.Linear(5, 6),
        torch.nn.ReLU()
    )

    # Outputs of the neural network (for eg.)
    out = layer(batch_example)
    print(out)

    # If keep dim is false we'll get a 2d vector [v1 v2] instead of a 2x1 matrix [[v1] [v2]]
    mean = out.mean(dim=-1, keepdim=True)
    var  = out.var(dim=-1, keepdim=True)

    print("Mean:", mean)
    print("Var :", var)

    out_norm = (out - mean) / torch.sqrt(var)
    print("Normalized layer outputs:", out_norm)

    on_mean = out_norm.mean(dim=-1, keepdim=True)
    on_var  = out_norm.var(dim=-1, keepdim=True)

    print("Normalized Output Mean:", on_mean)
    print("Normalized Output Var :", on_var)

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

def main():
    batch_example = torch.randn(2, 5)
    ln = LayerNorm(5)
    out_ln = ln(batch_example)

    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
    print("mean:", mean)
    print("var:", var)

    print(out_ln)

if __name__ == "__main__":
    main()
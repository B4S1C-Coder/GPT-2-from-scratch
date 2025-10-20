import torch
import numpy as np

def steps():
    inputs = torch.tensor([
        [0.43, 0.15, 0.89], # Your    (X^1)
        [0.55, 0.87, 0.66], # journey (X^2)
        [0.57, 0.85, 0.64], # starts  (X^3)
        [0.22, 0.58, 0.33], # with    (X^4)
        [0.77, 0.25, 0.10], # one     (X^5)
        [0.05, 0.80, 0.55]  # step    (X^6)
    ])

    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2

    W_q = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_k = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_v = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    # q_2 = x_2 @ W_q
    # k_2 = x_2 @ W_k
    # v_2 = x_2 @ W_v

    queries = inputs @ W_q
    values = inputs @ W_v
    keys = inputs @ W_k

    print(f"Calculated q, k, v --> {queries.shape}, {keys.shape}, {values.shape}")

    attention_scores = queries @ keys.T

    d_k = keys.shape[-1]
    attention_weights = torch.softmax(attention_scores / d_k**0.5, dim=-1)
    print(attention_weights)

    context_vectors = attention_weights @ values
    print(context_vectors)

def compute_variance(dim, num_trails=1000):
    dot_products = []
    scaled_dot_products = []

    for _ in range(num_trails):
        q = np.random.randn(dim)
        k = np.random.randn(dim)

        dot_product = np.dot(q, k)
        dot_products.append(dot_product)

        scaled_dot_product = dot_product / np.sqrt(dim)
        scaled_dot_products.append(scaled_dot_product)

    variance_before_scaling = np.var(dot_products)
    variance_after_scaling  = np.var(scaled_dot_products)

    return variance_before_scaling, variance_after_scaling

class SelfAttention_v1(torch.nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_q = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.w_k = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.w_v = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.w_k
        queries = x @ self.w_q
        values = x @ self.w_v

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attention_weights @ values
        return context_vec
    
# Instead of using torch.Parameter & rand, we could use torch.Linear which is better suited
# for matrix multiplication since the bias units are disabled.

class SelfAttention_v2(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.w_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.w_k(x)
        queries = self.w_q(x)
        values = self.w_v(x)

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attention_weights @ values
        return context_vec

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

    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))

    print("----------------")

    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))

if __name__ == "__main__":
    main()
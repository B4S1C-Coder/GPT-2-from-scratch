import torch

class CausalAttention_ov1(torch.nn.Module):
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

        context_length = attention_scores.shape[0]
        # Lower traingular matrix filled with ones
        mask_simple = torch.tril(torch.ones(context_length, context_length))
        unnormed_masked_attn_w = attention_weights * mask_simple

        row_sums = unnormed_masked_attn_w.sum(dim=1, keepdim=True)
        normed_maked_attn_w = unnormed_masked_attn_w / row_sums

        return normed_maked_attn_w @ values

class CausalAttention_ov2(torch.nn.Module):
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
        context_length = attention_scores.shape[0]

        umask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        umasked = attention_scores.masked_fill(umask.bool(), -torch.inf)

        attention_weights = torch.softmax(
            umasked / keys.shape[-1]**0.5, dim=-1
        )

        return attention_weights @ values
    
class CausalAttention_ov3(torch.nn.Module):
    def __init__(self, d_in, d_out, dropout_rate=0.5, qkv_bias=False):
        super().__init__()
        self.w_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        keys = self.w_k(x)
        queries = self.w_q(x)
        values = self.w_v(x)
        
        attention_scores = queries @ keys.T
        context_length = attention_scores.shape[0]

        umask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        umasked = attention_scores.masked_fill(umask.bool(), -torch.inf)

        attention_weights = torch.softmax(
            umasked / keys.shape[-1]**0.5, dim=-1
        )

        attention_weights = self.dropout(attention_weights)
        return attention_weights @ values

class CausalAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.w_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)

        # Not strictly necessary for all use cases but offers some advantages here
        # When we use CausualAttention class in our LLM, buffers are automatically moved
        # to the appropriate device (CPU or GPU) along with our model.

        # So we don't need to manually ensure these tensors are on the same device as the model
        # parameters, avoiding device mismatch errors.
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.w_k(x)
        queries = self.w_q(x)
        values = self.w_v(x)

        attention_scores = queries @ keys.transpose(1, 2)
        # Intellisense errors in self.mask.bool() is not an actual error, the code will run just fine
        # the .bool() is essentially a typecast (in PyTorch) and not an actual method, which the static
        # checker assumes, this results in intellisense errors, otherwise it's perfectly valid code
        # (& runs too).
        
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim = -1
        )

        attention_weights = self.dropout(attention_weights)
        context_vectors = attention_weights @ values

        return context_vectors
    
def causal_usage():
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
    
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vectors = ca(batch)
    print(context_vectors)
    print(context_vectors.shape) # (2, 6, 2) --> Two matrices stacked on top of each other

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

    cam_ov1 = CausalAttention_ov3(d_in, d_out, 0.1)
    masked_attention_weights = cam_ov1(inputs)

    print(masked_attention_weights)

if __name__ == "__main__":
    causal_usage()
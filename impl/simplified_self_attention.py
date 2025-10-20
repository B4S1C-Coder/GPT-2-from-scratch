import torch

# Embedding Vectors - Input Vectors (Token Embedding + Positional Embedding)
inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your    (X^1)
    [0.55, 0.87, 0.66], # journey (X^2)
    [0.57, 0.85, 0.64], # starts  (X^3)
    [0.22, 0.58, 0.33], # with    (X^4)
    [0.77, 0.25, 0.10], # one     (X^5)
    [0.05, 0.80, 0.55]  # step    (X^6)
])

query = inputs[1]
attn_scores = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores[i] = torch.dot(x_i, query)

print("Attention Scores:", attn_scores)

# We normalize these scores for interpretability and also so that they
# sum up to 1. Attention weights & scores are intuitively the same. The
# only difference being that the attention weights sum up to 1.

# attn_weights_tmp = attn_scores / attn_scores.sum()
# print("Attention Weights:", attn_weights_tmp)
# print("Sum:", attn_weights_tmp.sum())

def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x) / torch.exp(x).sum(dim=0)

# attn_weights_naive = softmax_naive(attn_scores)

# Better to use PyTorch's implementation of softmax for better numerical stability
attn_weights = torch.softmax(attn_scores, dim=0)

print("Attention Weights:", attn_weights)
print("Sum:", attn_weights.sum())

context_vector = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vector += attn_weights[i] * x_i

print(context_vector)

# ### INEFFEICIENT WAY ###
# attention_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attention_scores[i, j] = torch.dot(x_i, x_j)

attention_scores = inputs @ inputs.T
print("Attention Scores:\n", attention_scores)

# dim specifies the dimension of the input tensor along which the function will be computed
# by setting dim = -1, we are telling pytorch to apply normalization along the last dimension
# of the attention_scores tensor essentially the columns
attention_weights = torch.softmax(attention_scores,dim=-1)
print("Attention Weights:\n", attention_weights)


all_context_vectors = attention_weights @ inputs
print("Context Vectors:\n", all_context_vectors)
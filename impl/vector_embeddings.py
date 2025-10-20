import torch
from impl.input_target_pairs import create_dataloader_v1

vocab_size = 50257
output_dims = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dims)

with open('datasets/the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

max_length = 4

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

token_embeddings = token_embedding_layer(inputs)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dims)
# argange(max_length) --> 0 1 2 3 4 ... (max_length - 1)
# basically the positions in the postion - (256 element vector) lookup table
# aka the positional embedding layer
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
# With max_length = 4, we got 4 positional vectors corresponding corresponding to
# each of the 4 positions.
# print(pos_embeddings.shape)

# We can now add these postional vectors directly with the token embeddings
# the input_embeddings now captures both the semantic relationships (via token embeddings)
# and positional relationships (via positional embeddings)
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape) # 8 4 256
print(input_embeddings)
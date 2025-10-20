import torch
torch.set_default_device("cuda")

from impl.gpt_model import GPTModel
from impl.config import get_currently_chosen_cfg

inputs = torch.tensor([
    [16833, 3626, 6100],
    [40, 1107, 588]
])

outputs = torch.tensor([
    [3626, 6100, 345],
    [1107, 588, 11311]
])

model = GPTModel(get_currently_chosen_cfg())

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
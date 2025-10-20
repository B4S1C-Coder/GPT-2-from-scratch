import torch
from compat.openai_compat_gpt import OpenAICompatibleGPTModel
from impl.config import get_currently_chosen_cfg

checkpoint = torch.load('bin/gpt2_124m_openai_checkpoint.pth')
model = OpenAICompatibleGPTModel(get_currently_chosen_cfg())

missing, unexpected = model.load_state_dict(checkpoint, strict=False)

print("Missing:", missing)
print("Unexpected:", unexpected)
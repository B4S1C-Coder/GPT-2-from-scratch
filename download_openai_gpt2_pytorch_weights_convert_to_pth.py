import torch
from transformers import GPT2Model

model = GPT2Model.from_pretrained("openai-community/gpt2-medium")
torch.save(model.state_dict(), "bin/gpt2_355m_openai_checkpoint.pth")

print("Model saved")
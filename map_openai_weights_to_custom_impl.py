import torch
from compat.openai_compat_gpt import OpenAICompatibleGPTModel
from impl.config import get_currently_chosen_cfg

def map_openai_to_custom_keys(openai_state_dict):
    mapped = {}

    for k, v in openai_state_dict.items():
        new_k = k
        transpose = False

        # Embeddings
        if k == "wte.weight":
            new_k = "tok_emb.weight"
        elif k == "wpe.weight":
            new_k = "pos_emb.weight"

        # Final LayerNorm
        elif k.startswith("ln_f."):
            if k.endswith("weight"):
                new_k = "final_norm.scale"
            elif k.endswith("bias"):
                new_k = "final_norm.shift"

        # Transformer blocks
        elif k.startswith("h."):
            parts = k.split(".")
            block_idx = int(parts[1])
            submodule = parts[2]
            param = parts[-1]

            base = f"trf_blocks.{block_idx}."

            if submodule == "ln_1":
                base += "norm1."
                if param == "weight":
                    new_k = base + "scale"
                elif param == "bias":
                    new_k = base + "shift"

            elif submodule == "ln_2":
                base += "norm2."
                if param == "weight":
                    new_k = base + "scale"
                elif param == "bias":
                    new_k = base + "shift"

            elif submodule == "attn":
                if parts[3] == "c_attn":
                    new_k = f"trf_blocks.{block_idx}.attn.c_attn.{param}"
                    if param == "weight":
                        transpose = True
                    elif param == "bias":
                        transpose = False
                elif parts[3] == "c_proj":
                    new_k = f"trf_blocks.{block_idx}.attn.c_proj.{param}"
                    if param == "weight":
                        transpose = True

            elif submodule == "mlp":
                new_k = f"trf_blocks.{block_idx}.mlp.{parts[3]}.{param}"
                if param == "weight":
                    transpose = True

        # Transpose linear weights
        if transpose:
            v = v.T

        mapped[new_k] = v

    return mapped

# Load official GPT-2 checkpoint
checkpoint = torch.load('bin/gpt2_124m_openai_checkpoint.pth', map_location="cpu")

# Map checkpoint keys to your model
mapped_ckpt = map_openai_to_custom_keys(checkpoint)

# Initialize your model
model = OpenAICompatibleGPTModel(get_currently_chosen_cfg())

# Load mapped weights
missing, unexpected = model.load_state_dict(mapped_ckpt, strict=False)

print("Missing:", missing)
print("Unexpected:", unexpected)

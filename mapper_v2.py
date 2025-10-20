import torch
from compat.openai_compat_gpt import OpenAICompatibleGPTModel
from impl.config import get_currently_chosen_cfg

def map_openai_to_custom_keys(checkpoint):
    """
    Maps an OpenAI GPT-2 state_dict to OpenAICompatibleGPTModel state_dict.
    Handles weight transposes and bias copying.
    """
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_k = k
        transpose = False

        # Token & position embeddings
        if k.startswith("wte"):
            new_k = "tok_emb.weight"
        elif k.startswith("wpe"):
            new_k = "pos_emb.weight"

        # Transformer blocks
        elif k.startswith("h."):
            parts = k.split(".")  # Example: h.0.attn.c_attn.weight
            block_idx = parts[1]
            submodule = parts[2]
            param = parts[-1]

            # LayerNorm
            if submodule.startswith("ln_"):
                ln_num = "1" if submodule == "ln_1" else "2"
                if param == "weight":
                    new_k = f"trf_blocks.{block_idx}.norm{ln_num}.scale"
                elif param == "bias":
                    new_k = f"trf_blocks.{block_idx}.norm{ln_num}.shift"

            # Attention
            elif submodule == "attn":
                if parts[3] == "c_attn":
                    new_k = f"trf_blocks.{block_idx}.attn.c_attn.{param}"
                    if param == "weight":
                        transpose = True
                    elif param == "bias":
                        new_k = f"trf_blocks.{block_idx}.attn.c_attn.bias"
                        transpose = False
                elif parts[3] == "c_proj":
                    new_k = f"trf_blocks.{block_idx}.attn.c_proj.{param}"
                    if param == "weight":
                        transpose = True
                    elif param == "bias":
                        transpose = False

            # MLP
            elif submodule == "mlp":
                if parts[3] == "c_fc":
                    new_k = f"trf_blocks.{block_idx}.mlp.c_fc.{param}"
                    if param == "weight":
                        transpose = True
                    elif param == "bias":
                        transpose = False
                elif parts[3] == "c_proj":
                    new_k = f"trf_blocks.{block_idx}.mlp.c_proj.{param}"
                    if param == "weight":
                        transpose = True
                    elif param == "bias":
                        transpose = False

        # Final LayerNorm
        elif k == "ln_f.weight":
            new_k = "final_norm.scale"
        elif k == "ln_f.bias":
            new_k = "final_norm.shift"

        # Transpose weights if needed
        if transpose:
            v = v.T

        new_state_dict[new_k] = v

    return new_state_dict

# --------------------------
# Example usage:
# --------------------------
checkpoint = torch.load("bin/gpt2_124m_openai_checkpoint.pth")
mapped_state_dict = map_openai_to_custom_keys(checkpoint)

model = OpenAICompatibleGPTModel(get_currently_chosen_cfg()._replace(qkv_bias=True))
missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)

print("Missing:", missing)
print("------")
print("Unexpected:", unexpected)

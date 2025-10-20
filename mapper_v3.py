import torch
from compat.openai_compat_gpt import OpenAICompatibleGPTModel
from impl.config import get_currently_chosen_cfg, OPENAI_GPT_2_CFG_355M

def map_openai_to_custom_keys(openai_state_dict, model):
    """
    Map OpenAI GPT-2 checkpoint keys to your OpenAICompatibleGPTModel keys.
    """
    mapped_state_dict = {}

    for k, v in openai_state_dict.items():
        new_k = k

        # Embeddings
        if k == "wte.weight":
            new_k = "tok_emb.weight"
        elif k == "wpe.weight":
            new_k = "pos_emb.weight"
        
        # Final layer norm
        elif k == "ln_f.weight":
            new_k = "final_norm.scale"
        elif k == "ln_f.bias":
            new_k = "final_norm.shift"

        # Final linear layer (lm_head might not exist in all checkpoints)
        elif k == "lm_head.weight":
            new_k = "out_head.weight"

        # Transformer blocks
        elif k.startswith("h."):
            # Replace h. with trf_blocks.
            new_k = k.replace("h.", "trf_blocks.")
            
            # Layer norms
            new_k = new_k.replace("ln_1.weight", "norm1.scale")
            new_k = new_k.replace("ln_1.bias", "norm1.shift")
            new_k = new_k.replace("ln_2.weight", "norm2.scale")
            new_k = new_k.replace("ln_2.bias", "norm2.shift")

            # Transpose weights for Linear layers (GPT-2 uses Conv1D which stores transposed)
            if "attn.c_attn.weight" in new_k or "attn.c_proj.weight" in new_k:
                v = v.t()
            if "mlp.c_fc.weight" in new_k or "mlp.c_proj.weight" in new_k:
                v = v.t()

        else:
            # Skip any unrecognized keys
            print(f"Skipping unrecognized key: {k}")
            continue

        mapped_state_dict[new_k] = v

    # Handle weight tying: if lm_head doesn't exist, tie it to token embeddings
    if "out_head.weight" not in mapped_state_dict and "tok_emb.weight" in mapped_state_dict:
        print("Weight tying: using tok_emb.weight for out_head.weight")
        mapped_state_dict["out_head.weight"] = mapped_state_dict["tok_emb.weight"]

    return mapped_state_dict

# Example usage:
checkpoint = torch.load("bin/gpt2_355m_openai_checkpoint.pth")
model = OpenAICompatibleGPTModel(OPENAI_GPT_2_CFG_355M)

mapped_state_dict = map_openai_to_custom_keys(checkpoint, model)

missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)

print("Missing:", missing)
print("-------------")
print("Unexpected:", unexpected)

if len(unexpected) == 0:
    torch.save(model.state_dict(), "bin/gpt2_355m_compat_openai.pth")
    print("Converted weights saved.")

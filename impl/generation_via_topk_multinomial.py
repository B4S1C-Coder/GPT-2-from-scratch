import torch
from impl.utils import (
    load_reduced_context_gpt2,
    generate,
    text_to_token_ids,
    token_ids_to_text,
    get_gpt2_tokenizer
)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise RuntimeError("CUDA device not found.")

def main():
    tokenizer = get_gpt2_tokenizer()
    model, _ = load_reduced_context_gpt2(device)

    if not model:
        raise RuntimeError("GPTModel could not be loaded.")

    token_ids = generate(
        model=model, idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15, context_size=256, top_k=25, temperature=1.4
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

if __name__ == "__main__":
    main()
import torch
import tiktoken
from impl.config import GPTModelCfg, get_currently_chosen_cfg
from impl.transformer_block import TransformerBlock
from impl.layer_normalization import LayerNorm

torch.set_default_device("cuda")

class GPTModel(torch.nn.Module):
    def __init__(self, cfg: GPTModelCfg) -> None:
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = torch.nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = torch.nn.Dropout(cfg.drop_rate)

        self.trf_blocks = torch.nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = torch.nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
    
    def forward(self, in_idx) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)

        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

def illustrate():
    batch = torch.tensor([
        [6109, 3626, 6100, 345],
        [6109, 1110, 6622, 257]
    ])

    model = GPTModel(get_currently_chosen_cfg())
    out = model(batch)

    print("Input batch shape:", batch.shape)
    print("Output shape     :", out.shape)

    print("--------------")
    print(batch)
    print("--------------")
    print(out)

    print("--------------")
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_params)

def main():
    model = GPTModel(get_currently_chosen_cfg())
    model.eval()

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #A
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=30,
        context_size=get_currently_chosen_cfg().context_length
    )

    print("Output:", out)
    print("Length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)

if __name__ == "__main__":
    main()
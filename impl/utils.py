import torch
import tiktoken
from impl.config import GPT_2_CFG_124M, OPENAI_GPT_2_CFG_124M, OPENAI_GPT_2_CFG_355M
from impl.gpt_model import GPTModel
from compat.openai_compat_gpt import OpenAICompatibleGPTModel

def perform_non_cpu_backend_check(
    backend: str="cuda", raise_on_non_availability: bool=True
) -> str:
    """ checks if specified backend exists or not """
    if backend == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif backend == "mps" and torch.backends.mps.is_available():
        return "mps"
    else:
        if raise_on_non_availability:
            raise RuntimeError("No CUDA/MPS backend was found.")
        return "cpu"

def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def get_gpt2_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding('gpt2')

def load_reduced_context_gpt2(device: str, load_optimizer: bool=False
                              ) -> tuple[GPTModel | None, torch.optim.AdamW | None]:
    model, optimizer = None, None
    model_cfg = GPT_2_CFG_124M._replace(context_length=256)
    checkpoint = torch.load("bin/gpt2_124m_reduced_ctx__checkpoint.pth", map_location=device)

    model = GPTModel(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if load_optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return model, optimizer

def load_openai_124m_gpt2(device: str, eval_mode: bool = False) -> OpenAICompatibleGPTModel:
    model = OpenAICompatibleGPTModel(OPENAI_GPT_2_CFG_124M).to(device)
    model.load_state_dict(torch.load("bin/gpt2_124m_compat_openai.pth", map_location=device))

    if eval_mode:
        model.eval()

    return model

def load_openai_355m_gpt2(device: str, eval_mode: bool = False) -> OpenAICompatibleGPTModel:
    model = OpenAICompatibleGPTModel(OPENAI_GPT_2_CFG_355M).to(device)
    model.load_state_dict(torch.load("bin/gpt2_355m_compat_openai.pth", map_location=device))

    if eval_mode:
        model.eval()

    return model

def generate(
    model: GPTModel | OpenAICompatibleGPTModel, idx: torch.Tensor, max_new_tokens: int, context_size: int,
    temperature: float=0.0, top_k: int | None=None, eos_id: torch.Tensor | None=None
) -> torch.Tensor:
    """ Generates next token via top-k samling + multinomial otherwise argmax. """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
        
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Stop generating early if end-of-sequence token is encountered & specified
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


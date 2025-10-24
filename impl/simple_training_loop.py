import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise RuntimeError("CUDA Device not found.")

from impl.input_target_pairs import create_dataloader_v1
from impl.evaluating_on_dataset import calc_loss_loader, calc_loss_batch
from impl.config import get_currently_chosen_cfg, GPTModelCfg
from impl.gpt_model import (
    text_to_token_ids,
    generate_text_simple,
    token_ids_to_text,
    GPTModel
)
import tiktoken
import time
import matplotlib.pyplot as plt
from datetime import datetime

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " ")) # For compact printing
    model.train()

def train_model_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs,
    eval_freq, eval_iter, start_context, tokenizer
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    
    return train_losses, val_losses, track_tokens_seen

def create_data_loaders(
    model_cfg: GPTModelCfg, tokenizer: tiktoken.Encoding, file_path: str="datasets_/the-verdict.txt"
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    with open(file_path, 'r', encoding='utf-8') as f:
        text_data = f.read()

    total_chars = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    print("Total characters:", total_chars)
    print("Total tokens    :", total_tokens)

    train_ratio = 0.90 # 90% train, 10% test
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=model_cfg.context_length,
        stride=model_cfg.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=model_cfg.context_length,
        stride=model_cfg.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    print("Performing sanity checks ...")

    if total_tokens * train_ratio < model_cfg.context_length:
        raise RuntimeWarning(
            "Not enough tokens for the training loader. Lower model_cfg.context_length"
            " OR increase the training ratio."
        )

    if total_tokens * (1 - train_ratio) < model_cfg.context_length:
        raise RuntimeWarning(
            "Not enough tokens for the validation loader. Lower model_cfg.context_length"
            " OR decrease the training ratio."
        )

    print("Checks complete ...")

    return train_loader, val_loader

def main():
    model_cfg = get_currently_chosen_cfg()._replace(
        context_length=256
    )

    tokenizer = tiktoken.get_encoding('gpt2')
    
    start_time = time.time()
    model = GPTModel(model_cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    
    num_epocs = 10

    train_loader, val_loader = create_data_loaders(model_cfg, tokenizer)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epocs,
        eval_freq=5, eval_iter=5, start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_mins = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_mins:.2f} minutes.")

    save_path = "bin/gpt2_124m_reduced_ctx__checkpoint.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses
    }, save_path)

    print(f"Model saved to {save_path}")

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    # plt.plot(tokens_seen, label="Tokens Seen")

    plt.legend(loc='upper right')
    output_path = f"simple_trainig_result_{datetime.now().isoformat().replace(':', '-')}.png"
    plt.savefig(output_path)

    print(f"Result graph saved to: {output_path}")

if __name__ == "__main__":
    main()

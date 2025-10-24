import torch
import tiktoken
import matplotlib.pyplot as plt
from datetime import datetime
from reasoning.dataset import get_dataloaders
from reasoning.model import OpenAICompatibleGPTModel, OPENAI_GPT_2_CFG_355M

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = target_batch.view(-1)

    loss = torch.nn.functional.cross_entropy(
        logits_flat, targets_flat, ignore_index=-100
    )

    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    num_batches_processed = 0

    model.eval()

    with torch.no_grad():
        for i, (input_batch, attention_mask, target_batch) in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            num_batches_processed += 1
    
    model.train()
    return total_loss / num_batches_processed if num_batches_processed > 0 else 0.

def train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs, eval_freq, eval_iter, save_path="gpt_reasoning.pth"
):
    
    model.to(device)
    model.train()

    global_step = 0
    best_val_loss = float('inf')

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        for input_batch, attention_mask, target_batch in train_loader:
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(
                    train_loader, model, device, num_batches=eval_iter
                )

                val_loss = calc_loss_loader(
                    val_loader, model, device, num_batches=eval_iter
                )

                print(
                    f"Epoch {epoch + 1}, Step {global_step}: "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Save only best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved best model with val loss: {val_loss:.4f}")
            
            global_step += 1
    
    print("Training complete.")
    return train_losses, val_losses

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA Device not found.")
    
    tokenizer = tiktoken.get_encoding('gpt2')
    
    model = OpenAICompatibleGPTModel(OPENAI_GPT_2_CFG_355M).to(device)
    model.load_state_dict(torch.load("bin/gpt2_355m_compat_openai.pth", map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

    train_loader, _, val_loader = get_dataloaders(device=device, tokenizer=tokenizer)

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=3,
        eval_freq=100,
        eval_iter=10,
        save_path='bin/gpt2_355m_reasonin_v1.pth'
    )

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend(loc='upper right')

    plt.savefig(f"graphs/reasoning_finetuning_result_{datetime.now().isoformat().replace(':', '-')}.png")

if __name__ == "__main__":
    main()
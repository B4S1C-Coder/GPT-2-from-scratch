import torch
from impl.config import OPENAI_GPT_2_CFG_124M
from impl.utils import (
    perform_non_cpu_backend_check,
    load_openai_124m_gpt2
)
import time
from classification.spam_dataset import get_data_loaders
import matplotlib.pyplot as plt
from datetime import datetime

def construct_classification_model(device: str, num_classes: int=2):
    model = load_openai_124m_gpt2(device=device, eval_mode=False)

    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False
    
    # This new out_head has requires_grad = True by default
    model.out_head = torch.nn.Linear(
        in_features=OPENAI_GPT_2_CFG_124M.emb_dim, out_features=num_classes
    )

    # Unfreeze the last transformer block for fine-tuning
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    
    # Unfreeze the final norm for fine-tuning
    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    return model

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :] # last row of logits matrix
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches, to match the total number  of batches in the
        # dataloader, if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss


def train_classifier_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter
):
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
    
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )
        
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen

def classify_review(text, model, tokenizer, device, max_length, pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
        predicted_label = int(torch.argmax(logits, dim=-1).item())

    return "spam" if predicted_label == 1 else "not spam"

def main():
    device = perform_non_cpu_backend_check()
    start_time = time.time()

    train_loader, val_loader, test_loader = get_data_loaders(device)

    model = construct_classification_model(device, 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5
    )

    end_time = time.time()
    execution_time_mins = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_mins:.2f} minutes.")

    torch.save(model.state_dict(), "bin/gpt2_spam_classif_weights.pth")
    print("Weights saved")

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.plot(train_accs, label="Train accuracy")
    plt.plot(val_accs, label="Validation accuracy")

    plt.legend(loc='upper right')
    output_path = f"graphs/simple_classifier_trainig_result_{datetime.now().isoformat().replace(':', '-')}.png"
    plt.savefig(output_path)

    print(f"Result graph saved to: {output_path}")

if __name__ == "__main__":
    main()

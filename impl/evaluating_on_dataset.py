import torch
import tiktoken
from impl.input_target_pairs import create_dataloader_v1
from impl.config import get_currently_chosen_cfg
from impl.gpt_model import GPTModel

# torch.set_default_device('cpu')

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise RuntimeError("GPU not available.")

# Temporarily override, GPT - 2 configs to deal with small datasets_.
MODEL_CFG = get_currently_chosen_cfg()._replace(
    context_length=256
)

print(f"Using default device: {device}")

with open('datasets_/the-verdict.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

tokenizer = tiktoken.get_encoding('gpt2')
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
    max_length=MODEL_CFG.context_length,
    stride=MODEL_CFG.context_length,
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=MODEL_CFG.context_length,
    stride=MODEL_CFG.context_length,
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print("Performing sanity checks ...")

if total_tokens * train_ratio < MODEL_CFG.context_length:
    raise RuntimeWarning(
        "Not enough tokens for the training loader. Lower MODEL_CFG.context_length"
        " OR increase the training ratio."
    )

if total_tokens * (1 - train_ratio) < MODEL_CFG.context_length:
    raise RuntimeWarning(
        "Not enough tokens for the validation loader. Lower MODEL_CFG.context_length"
        " OR decrease the training ratio."
    )

print("Checks complete ...")

print("Train loader shapes:")
for x, y in train_loader:
    print(x.shape, y.shape)


print("Validation loader shapes:")
for x, y in val_loader:
    print(x.shape, y.shape)

model = GPTModel(MODEL_CFG)
model.eval()

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
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

model.to(device)

print("Evaluating performance on dataset ...")

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss  :", train_loss)
print("Validation loss:", val_loss)
import json
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tiktoken

def format_input(entry: dict) -> str:
    # This is the Alpaca template
    instruction_text = (
        "Below is an instruction that describes a task."
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

def split_dataset(
    data_file_path: str="datasets_/instruction-data.json"
) -> tuple[dict, dict, dict]:

    with open(data_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    return train_data, test_data, val_data

def custom_collate_fn(
    batch,
    device: str,
    pad_token_id: int=50256,
    ignore_index: int=-100,
    allowed_max_length: int | None=None
) -> tuple[torch.Tensor, torch.Tensor]:
    
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = (targets == pad_token_id)
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer: "tiktoken.Encoding"):
        self.data = data

        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            self.encoded_texts.append(tokenizer.encode(full_text))
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)

def get_data_loaders(
    device: str,
    tokenizer: "tiktoken.Encoding",
    num_workers: int=0,
    batch_size: int=8
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    train_data, test_data, val_data = split_dataset()
    generator = torch.Generator(device=device)

    customized_collate_fn = partial(
        custom_collate_fn, device=device, allowed_max_length=1024
    )

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        generator=generator
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        generator=generator
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        generator=generator
    )

    return train_loader, val_loader, test_loader
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import namedtuple
from typing import TYPE_CHECKING
from functools import partial

if TYPE_CHECKING:
    import tiktoken

PromptData = namedtuple(
    'PromptData',
    [
        'user_content',
        'think_content',
        'answer_content'
    ]
)

def prompt_data(data: dict) -> PromptData:
    return PromptData(
        user_content=data["messages"][0]["content"],
        think_content=data["messages"][1]["info"]["think_content"],
        answer_content=data["messages"][1]["info"]["answer_content"]
    )

def create_prompt(data: PromptData) -> str:
    return (
        f"Question: {data.user_content}\n"
        "Let's think step by step.\n"
        f"{data.think_content}\n"
        f"Answer: {data.answer_content}"
    )

def split_dataset(
    data_file_path: str | Path, train_ratio: float, test_ratio: float
) -> tuple[list[PromptData], list[PromptData], list[PromptData]]:

    with open(data_file_path, 'r', encoding='utf-8') as f:
        json_strs = f.read()

    reasoning_docs = [
        prompt_data(json.loads(json_str)) for json_str in json_strs.splitlines()
    ]

    trainP = int(len(reasoning_docs) * train_ratio)
    testP  = int(len(reasoning_docs) * test_ratio)

    return (
        reasoning_docs[:trainP],                # Train
        reasoning_docs[trainP: trainP + testP], # Test
        reasoning_docs[trainP + testP:]         # Validate
    )

def collate_fn(
    batch: list[list[int]],
    device: str,
    pad_token_id: int = 50256,
    max_length: int = 1024,
    ignore_index: int = -100
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    batch = [seq[:max_length] for seq in batch]
    max_len = max(len(seq) for seq in batch)

    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), pad_token_id)
    attention_mask = torch.zeros((batch_size, max_len))
    labels = torch.full((batch_size, max_len), ignore_index)

    for i, seq in enumerate(batch):
        seq_len = len(seq)

        # Input IDs: Actual token sequence
        input_ids[i, :seq_len] = torch.tensor(seq)
        # Attention mask: 1 for real, 0 for padding tokens
        attention_mask[i, :seq_len] = 1
        # Shift labels by +1 for next token prediction
        if seq_len > 1:
            labels[i, :seq_len - 1] = torch.tensor(seq[1:])
    
    return input_ids.to(device), attention_mask.to(device), labels.to(device)

class ReasoningDataset(Dataset):
    def __init__(self, data: list[PromptData], encoding: "tiktoken.Encoding"):
        self.data = data
        self.encoded_texts = []

        for pdata in data:
            prompt = create_prompt(pdata)
            self.encoded_texts.append(encoding.encode(prompt))
    
    def __getitem__(self, index: int) -> list[int]:
        return self.encoded_texts[index]
    
    def __len__(self) -> int:
        return len(self.data)

def get_dataloaders(
    device: str,
    tokenizer: "tiktoken.Encoding",
    n_workers: int=0,
    batch_size: int=8,
    train_ratio: float=0.8,
    test_ratio: float=0.1
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    DATASET_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / '..' / 'datasets_'
    data_file_path = DATASET_DIR / 'am_0.9M_sample_1k.jsonl'

    train_data, test_data, val_data = split_dataset(
        data_file_path=data_file_path, train_ratio=train_ratio, test_ratio=test_ratio
    )

    dataloader_collate_fn = partial(
        collate_fn, pad_token_id=50256, max_length=1024, device=device
    )
    generator = torch.Generator(device='cpu')

    train_dataset = ReasoningDataset(train_data, tokenizer)
    test_dataset = ReasoningDataset(test_data, tokenizer)
    val_dataset = ReasoningDataset(val_data, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        generator=generator,
        collate_fn=dataloader_collate_fn,
        num_workers=n_workers,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        generator=generator,
        collate_fn=dataloader_collate_fn,
        num_workers=n_workers,
        shuffle=False,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        generator=generator,
        collate_fn=dataloader_collate_fn,
        num_workers=n_workers,
        shuffle=False,
        drop_last=False
    )

    return train_loader, test_loader, val_loader

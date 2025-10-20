import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

tokenizer = tiktoken.get_encoding("gpt2")

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids  = []
        self.target_ids = []
        
        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# drop_last batch to prevent loss spikes if last batch is
# smaller than the batch_size
def create_dataloader_v1(
    txt, batch_size=4, max_length=256, stride=128, shuffle=True,
    drop_last=True, num_workers=16, device='cuda'
) -> DataLoader:

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    # Shuffle is used to improve generalization by changing the order of the I/p-T pairs
    # Drop last is used to drop the last I/p-T pair if they are smaller than the context_size
    # i.e. the max_length in this case 256
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        generator=generator
    )

    return dataloader

def main():
    with open('datasets/the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()

    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    # [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]
    # INPUT --> OUTPUT
    #              40  --> 367
    #           40 367 --> 2885
    #      40 367 2885 --> 1464
    # 40 367 2885 1464 --> 1807

def illustration():
    with open('datasets/the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()

    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

    context_size = 4

    for i in range(1, context_size + 1):
        context = enc_text[:i]
        desired = enc_text[i]

        print(context, "--->", desired)

if __name__ == "__main__":
    main()

import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def create_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    return balanced_df

def random_split(df: pd.DataFrame, train_frac: float, val_frac: float
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    # Shuffle
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Split indices
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)

    # Split
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # truncate if longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]
        
        # pad to longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0

        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)

            if encoded_length > max_length:
                max_length = encoded_length
        
        return max_length

def construct_split_files():
    data_file_path = 'datasets/SMSSpamCollection.tsv'
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)

    train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)

    train_df.to_csv("datasets/spam_train.csv", index=False)
    val_df.to_csv("datasets/spam_val.csv", index=False)
    test_df.to_csv("datasets/spam_test.csv", index=False)

    print("split .csv files saved")

def get_data_loaders(device: str):
    tokenizer = tiktoken.get_encoding('gpt2')
    train_dataset = SpamDataset("datasets/spam_train.csv", tokenizer)
    val_dataset = SpamDataset("datasets/spam_val.csv", tokenizer, train_dataset.max_length)
    test_dataset = SpamDataset("datasets/spam_test.csv", tokenizer, train_dataset.max_length)

    num_workers, batch_size = 0, 8

    generator = torch.Generator(device=device)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        generator=generator
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        generator=generator
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        generator=generator
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    get_data_loaders(device="cuda")
